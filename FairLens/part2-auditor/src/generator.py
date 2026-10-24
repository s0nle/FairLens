"""
generator.py: Module 1. Create synthetic test applicants.

Two test sets go into ONE file:
  population  realistic applicants sampled from real HMDA rows (with small noise),
              used for group fairness (approval rates by race, sex, age, neighborhood)
  matched     base applicants plus copies that change exactly ONE factor
              (neighborhood, race, ethnicity, sex, or age), used for swap tests

Two files are written:
  applications_for_bank.csv  case_id + model input columns ONLY (goes to Part 1)
  audit_key.csv              everything, including demographics (stays here)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import (TRACT_COLUMNS, AuditorError, protected_column_list,
                       resolve_path, to_numeric)

logger = logging.getLogger("auditor.generator")

KNOWN_RACES = ["White", "Black or African American", "Asian",
               "American Indian or Alaska Native",
               "Native Hawaiian or Other Pacific Islander", "2 or more minority races"]
YOUNG_AGES = {"<25", "25-34", "35-44", "45-54", "55-64"}
OLD_AGES = {"65-74", ">74"}
NOISE_ROUNDING = {"income": 0, "loan_amount": -3, "property_value": -3}

# Friendly names for each swap, used by the audit and the report.
FACTOR_LABELS = {
    "tract_high_minority": "Neighborhood: moved to a high-minority tract",
    "race_white": "Race changed to White",
    "race_black": "Race changed to Black",
    "ethnicity_flip": "Ethnicity flipped (Hispanic / not Hispanic)",
    "sex_flip": "Sex flipped (male / female)",
    "age_flip": "Age moved across 62",
}
FACTOR_GROUP = {
    "tract_high_minority": "neighborhood", "race_white": "race", "race_black": "race",
    "ethnicity_flip": "ethnicity", "sex_flip": "sex", "age_flip": "age",
}


def _add_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Small (about 5%) random changes to income, loan amount, and property value."""
    df = df.copy()
    for col, digits in NOISE_ROUNDING.items():
        if col not in df.columns:
            continue
        num = pd.to_numeric(df[col], errors="coerce")
        mask = num.notna() & (num > 0)
        if not mask.any():
            continue
        noisy = num[mask] * np.exp(rng.normal(0.0, 0.05, int(mask.sum())))
        noisy = noisy.round(digits).clip(lower=1)
        df[col] = df[col].astype(object)
        df.loc[mask, col] = noisy.astype(int) if digits <= 0 else noisy
    return df


def _tract_table(hmda: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in TRACT_COLUMNS if c in hmda.columns]
    t = hmda[cols].dropna(subset=["census_tract"]).drop_duplicates("census_tract").copy()
    t["_pct"] = to_numeric(t["tract_minority_population_percent"])
    t["_inc"] = (to_numeric(t["tract_to_msa_income_percentage"])
                 if "tract_to_msa_income_percentage" in t else np.nan)
    return t.dropna(subset=["_pct"]).reset_index(drop=True)


def _code_style(series: pd.Series | None, yes_no: bool = False):
    """Real HMDA writes 'Yes'/'No' for age_above_62; the synthetic file writes '1'/'2'."""
    if series is None:
        return ("Yes", "No") if yes_no else None
    values = set(series.dropna().astype(str))
    return ("Yes", "No") if {"Yes", "No"} & values else ("1", "2")


def _set(row: dict, col: str, value) -> None:
    if col in row:
        row[col] = value


def generate(cfg: dict, hmda_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build both test sets, write the two CSV files, and return them."""
    rng = np.random.default_rng(int(cfg["random_seed"]))
    features = cfg["model_features"]
    low_thr = float(cfg["low_minority_threshold"])
    high_thr = float(cfg["high_minority_threshold"])

    missing = [f for f in features if f not in hmda_df.columns]
    if missing:
        raise AuditorError("HMDA data is missing model features: " + ", ".join(missing))

    protected = [c for c in protected_column_list(cfg) if c in hmda_df.columns]
    tract_cols = [c for c in TRACT_COLUMNS if c in hmda_df.columns]
    keep = list(dict.fromkeys(features + tract_cols + protected))

    # ── Population set ────────────────────────────────────────────────────────
    n_pop = int(cfg["n_population"])
    pop_src = hmda_df.sample(n=n_pop, replace=len(hmda_df) < n_pop,
                             random_state=int(cfg["random_seed"]))
    pop = _add_noise(pop_src[keep].reset_index(drop=True), rng)
    pop["test_set"] = "population"
    pop["pair_group_id"] = ""
    pop["varied_factor"] = ""
    if "action_taken" in pop_src.columns:          # real past decision, for comparison
        act = to_numeric(pop_src["action_taken"]).to_numpy()
        pop["hist_approved"] = np.where(np.isin(act, [1, 2]), 1.0,
                                        np.where(act == 3, 0.0, np.nan))

    # ── Matched set ───────────────────────────────────────────────────────────
    tracts = _tract_table(hmda_df)
    low_tracts = tracts[tracts["_pct"] < low_thr]
    high_tracts = tracts[tracts["_pct"] > high_thr]
    if low_tracts.empty or high_tracts.empty:
        raise AuditorError(
            f"Need tracts below {low_thr:.0f}% and above {high_thr:.0f}% minority for the "
            "neighborhood swap test; the HMDA data doesn't have both.")

    eligible = hmda_df[hmda_df["derived_race"].isin(KNOWN_RACES)
                       & hmda_df["derived_sex"].isin(["Male", "Female"])]
    if "applicant_age" in hmda_df:
        eligible = eligible[eligible["applicant_age"].astype(str).isin(YOUNG_AGES | OLD_AGES)]
    if len(eligible) < 20:
        eligible = hmda_df                          # tiny files: use whatever we have
    n_base = int(cfg["n_matched_pairs"])
    bases = _add_noise(eligible.sample(n=n_base, replace=len(eligible) < n_base,
                                       random_state=int(cfg["random_seed"]) + 1)[keep]
                       .reset_index(drop=True), rng)

    above62 = _code_style(hmda_df.get("applicant_age_above_62"), yes_no=True)
    rows: list[dict] = []
    for i, base in bases.iterrows():
        group = f"P-{i + 1:04d}"
        low = low_tracts.iloc[int(rng.integers(len(low_tracts)))]
        gap = (high_tracts["_inc"] - low["_inc"]).abs().fillna(np.inf)
        closest = high_tracts.loc[gap.nsmallest(min(5, len(high_tracts))).index]
        high = closest.iloc[int(rng.integers(len(closest)))]

        b = base.to_dict()
        for c in tract_cols:                         # every base sits in a low-minority tract
            b[c] = low[c]

        def variant(name: str, **changes) -> None:
            v = dict(b)
            v.update(changes)
            v.update(test_set="matched", pair_group_id=group, varied_factor=name)
            rows.append(v)

        variant("base")
        variant("tract_high_minority", **{c: high[c] for c in tract_cols})

        race = str(b.get("derived_race", ""))
        for name, label, code in [("race_white", "White", "5"),
                                  ("race_black", "Black or African American", "3")]:
            if race != label:
                ch = {"derived_race": label, "applicant_race_1": code}
                ch.update({f"applicant_race_{k}": np.nan for k in range(2, 6)})
                variant(name, **{k: v for k, v in ch.items() if k in b})

        eth = str(b.get("derived_ethnicity", ""))
        if eth in ("Hispanic or Latino", "Not Hispanic or Latino"):
            hisp = eth != "Hispanic or Latino"
            ch = {"derived_ethnicity": "Hispanic or Latino" if hisp else "Not Hispanic or Latino",
                  "applicant_ethnicity_1": "1" if hisp else "2"}
            variant("ethnicity_flip", **{k: v for k, v in ch.items() if k in b})

        sex = str(b.get("derived_sex", ""))
        if sex in ("Male", "Female"):
            male = sex == "Female"
            ch = {"derived_sex": "Male" if male else "Female", "applicant_sex": "1" if male else "2"}
            variant("sex_flip", **{k: v for k, v in ch.items() if k in b})

        age = str(b.get("applicant_age", ""))
        if age in YOUNG_AGES | OLD_AGES:
            older = age in YOUNG_AGES
            ch = {"applicant_age": "65-74" if older else "35-44",
                  "applicant_age_above_62": above62[0] if older else above62[1]}
            variant("age_flip", **{k: v for k, v in ch.items() if k in b})

    matched = pd.DataFrame(rows)

    # ── Combine, shuffle, assign IDs ──────────────────────────────────────────
    key = pd.concat([pop, matched], ignore_index=True, sort=False)
    key = key.iloc[rng.permutation(len(key))].reset_index(drop=True)
    key.insert(0, "case_id", [f"C-{i:05d}" for i in range(1, len(key) + 1)])
    front = ["case_id", "test_set", "pair_group_id", "varied_factor"]
    key = key[front + [c for c in key.columns if c not in front]]

    apps = key[["case_id"] + features].copy()
    leaked = set(apps.columns) & (set(protected_column_list(cfg))
                                  | {c.replace("-", "_") for c in cfg["post_decision_columns"]})
    if leaked:
        raise AuditorError(f"Refusing to write: protected columns would reach the bank: {leaked}")

    apps_path = resolve_path(cfg["applications_path"])
    key_path = resolve_path(cfg["audit_key_path"])
    apps_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    apps.to_csv(apps_path, index=False)
    key.to_csv(key_path, index=False)

    counts = matched["varied_factor"].value_counts()
    print(f"  Generated {len(key):,} test applicants "
          f"({len(pop):,} population + {len(matched):,} matched in {len(bases):,} groups)")
    print("  Matched variants: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"  Wrote {apps_path}  <- send this to the bank model (Part 1)")
    print(f"  Wrote {key_path}  <- keep this; it has the demographics")
    return {"applications": apps, "audit_key": key}
