"""
audit.py: Module 3. The fairness tests.

Each test is its own function so the dashboard (or a watsonx Orchestrate tool)
can call it directly. run_full_audit() runs them all and saves
output/audit_results.json.

  group_fairness   approval rate by race, ethnicity, sex, age, and neighborhood
                   (population set); gap and disparate impact ratio (four-fifths rule)
  swap_test        matched applicants that differ in ONE factor: how often the
                   decision flips, and how much the approval probability moves
  proxy_detection  how well each model input, on its own, predicts race / sex / age
  summarize        severity per factor and an overall verdict
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.generator import FACTOR_GROUP, FACTOR_LABELS, OLD_AGES, YOUNG_AGES
from src.utils import minority_level, resolve_path, to_numeric, write_json

logger = logging.getLogger("auditor.audit")

DIMENSIONS = {
    "tract_minority_level": "Neighborhood (tract minority level)",
    "derived_race": "Race",
    "derived_ethnicity": "Ethnicity",
    "derived_sex": "Sex",
    "applicant_age": "Age",
}
# Not comparable groups: missing data, or joint applicants with different values.
EXCLUDED_GROUPS = {"Race Not Available", "Ethnicity Not Available", "Sex Not Available",
                   "Free Form Text Only", "Joint", "8888", "9999", "nan", "NA", ""}
FACTORS = {
    "neighborhood": ("tract_minority_level", ["tract_high_minority"]),
    "race": ("derived_race", ["race_white", "race_black"]),
    "ethnicity": ("derived_ethnicity", ["ethnicity_flip"]),
    "sex": ("derived_sex", ["sex_flip"]),
    "age": ("applicant_age", ["age_flip"]),
}
MAIN_COLUMN = {
    "tract_high_minority": "tract_minority_population_percent",
    "race_white": "derived_race", "race_black": "derived_race",
    "ethnicity_flip": "derived_ethnicity", "sex_flip": "derived_sex", "age_flip": "applicant_age",
}
SEVERITY_ORDER = ["none", "low", "medium", "high"]


def _population(df: pd.DataFrame) -> pd.DataFrame:
    pop = df[df.get("test_set", pd.Series("population", index=df.index)) == "population"]
    return pop if len(pop) else df


def _di_severity(ratio: float | None, cfg: dict) -> str:
    if ratio is None or np.isnan(ratio):
        return "none"
    s = cfg["fairness"]["severity"]
    if ratio < s["high"]:
        return "high"
    if ratio < s["medium"]:
        return "medium"
    if ratio < s["low"]:
        return "low"
    return "none"


def _flip_severity(rate: float, cfg: dict) -> str:
    s = cfg["fairness"]["flip_rate_severity"]
    if rate >= s["high"]:
        return "high"
    if rate >= s["medium"]:
        return "medium"
    if rate >= s["low"]:
        return "low"
    return "none"


def _max_sev(*levels: str) -> str:
    return max(levels, key=SEVERITY_ORDER.index) if levels else "none"


# ── 1. Group fairness ─────────────────────────────────────────────────────────

def group_fairness(df: pd.DataFrame, cfg: dict) -> dict:
    """Approval rates by group, and the best-vs-worst gap for each dimension."""
    pop = _population(df).copy()
    if "tract_minority_population_percent" in pop:
        pop["tract_minority_level"] = minority_level(pop["tract_minority_population_percent"], cfg)
    min_n = int(cfg.get("min_group_size", 20))
    threshold = float(cfg["fairness"]["disparate_impact_threshold"])

    rates, gaps = {}, {}
    for dim in DIMENSIONS:
        if dim not in pop:
            continue
        g = pop.groupby(pop[dim].astype(str))["decision"].agg(["count", "mean"])
        rows = [{"group": grp, "n": int(r["count"]), "approval_rate": float(r["mean"]),
                 "included": grp not in EXCLUDED_GROUPS and r["count"] >= min_n}
                for grp, r in g.iterrows()]
        rates[dim] = sorted(rows, key=lambda r: -r["approval_rate"])
        ok = [r for r in rows if r["included"]]
        if len(ok) < 2:
            continue
        best = max(ok, key=lambda r: r["approval_rate"])
        worst = min(ok, key=lambda r: r["approval_rate"])
        di = worst["approval_rate"] / best["approval_rate"] if best["approval_rate"] > 0 else np.nan
        gaps[dim] = {
            "label": DIMENSIONS[dim],
            "best_group": best["group"], "best_rate": best["approval_rate"], "best_n": best["n"],
            "worst_group": worst["group"], "worst_rate": worst["approval_rate"], "worst_n": worst["n"],
            "gap": best["approval_rate"] - worst["approval_rate"],
            "di_ratio": di,
            "flagged": bool(di < threshold) if not np.isnan(di) else False,
            "severity": _di_severity(di, cfg),
            "n_groups_compared": len(ok),
        }
    return {"rates": rates, "gaps": gaps, "min_group_size": min_n}


# ── 2. Swap (counterfactual) test ─────────────────────────────────────────────

def swap_test(df: pd.DataFrame, cfg: dict, max_examples: int = 10) -> dict:
    """Compare each matched variant to its base applicant."""
    if "test_set" not in df:
        return {"summary": [], "examples": []}
    m = df[df["test_set"] == "matched"]
    base = m[m["varied_factor"] == "base"].drop_duplicates("pair_group_id").set_index("pair_group_id")
    var = m[m["varied_factor"] != "base"].copy()
    var = var[var["pair_group_id"].isin(base.index)]
    if var.empty:
        return {"summary": [], "examples": []}

    var["base_decision"] = var["pair_group_id"].map(base["decision"])
    var["base_probability"] = var["pair_group_id"].map(base["probability"])
    var["flipped"] = var["decision"] != var["base_decision"]
    var["delta_prob"] = (pd.to_numeric(var["probability"], errors="coerce")
                         - pd.to_numeric(var["base_probability"], errors="coerce"))

    # Did the bank actually see a difference? Demographic swaps don't change any
    # model input, so a flip there means the model answered identical applications
    # differently: an inconsistency, not evidence about the demographic itself.
    features = [f for f in cfg["model_features"] if f in var.columns and f in base.columns]
    base_inputs = base[features].astype(str)
    var["inputs_identical"] = [
        (base_inputs.loc[g].to_numpy() == row).all()
        for g, row in zip(var["pair_group_id"], var[features].astype(str).to_numpy())
    ]
    var["bias_flip"] = var["flipped"] & ~var["inputs_identical"]
    var["inconsistent_flip"] = var["flipped"] & var["inputs_identical"]

    summary = []
    for factor, g in var.groupby("varied_factor"):
        n, flips = len(g), int(g["flipped"].sum())
        rate = flips / n if n else 0.0
        bias_rate = float(g["bias_flip"].mean()) if n else 0.0
        summary.append({
            "varied_factor": factor,
            "label": FACTOR_LABELS.get(factor, factor),
            "factor_group": FACTOR_GROUP.get(factor, factor),
            "n_pairs": n,
            "n_flipped": flips,
            "flip_rate": rate,
            "approve_to_deny": int(((g["base_decision"] == 1) & (g["decision"] == 0)).sum()),
            "deny_to_approve": int(((g["base_decision"] == 0) & (g["decision"] == 1)).sum()),
            "avg_delta_prob": float(g["delta_prob"].mean()) if g["delta_prob"].notna().any() else None,
            "bank_inputs_changed": bool((~g["inputs_identical"]).any()),
            "n_inconsistent": int(g["inconsistent_flip"].sum()),
            "severity": _flip_severity(bias_rate, cfg),
        })
    summary.sort(key=lambda r: -r["flip_rate"])

    same = var[var["inputs_identical"]]
    consistency = {
        "n_identical_pairs": int(len(same)),
        "n_inconsistent": int(same["flipped"].sum()),
        "inconsistency_rate": float(same["flipped"].mean()) if len(same) else 0.0,
    }
    consistency["severity"] = _flip_severity(consistency["inconsistency_rate"], cfg)

    flipped = var[var["bias_flip"]].copy()
    flipped["_order"] = (flipped["varied_factor"] != "tract_high_minority").astype(int) * 2 \
        + (flipped["decision"] != 0).astype(int)
    examples = []
    for _, r in flipped.sort_values(["_order", "pair_group_id"]).head(max_examples).iterrows():
        col = MAIN_COLUMN.get(r["varied_factor"])
        b = base.loc[r["pair_group_id"]]
        examples.append({
            "pair_group_id": r["pair_group_id"],
            "varied_factor": r["varied_factor"],
            "label": FACTOR_LABELS.get(r["varied_factor"], r["varied_factor"]),
            "changed_column": col,
            "from": b.get(col) if col else None,
            "to": r.get(col) if col else None,
            "base_case_id": b.get("case_id"),
            "variant_case_id": r["case_id"],
            "base_decision": int(r["base_decision"]),
            "variant_decision": int(r["decision"]),
            "base_probability": r["base_probability"],
            "variant_probability": r["probability"],
            "income": r.get("income"), "loan_amount": r.get("loan_amount"),
            "debt_to_income_ratio": r.get("debt_to_income_ratio"),
            "loan_to_value_ratio": r.get("loan_to_value_ratio"),
        })
    return {"summary": summary, "examples": examples, "consistency": consistency}


# ── 3. Proxy detection ────────────────────────────────────────────────────────

def _targets(pop: pd.DataFrame) -> dict[str, pd.Series]:
    t = {}
    if "derived_race" in pop:
        race = pop["derived_race"].astype(str)
        known = ~race.isin(EXCLUDED_GROUPS)
        t["race"] = (race[known] != "White").astype(int)          # 1 = not White
    if "derived_sex" in pop:
        sex = pop["derived_sex"].astype(str)
        t["sex"] = (sex[sex.isin(["Male", "Female"])] == "Female").astype(int)
    if "applicant_age" in pop:
        age = pop["applicant_age"].astype(str)
        t["age"] = age[age.isin(YOUNG_AGES | OLD_AGES)].isin(OLD_AGES).astype(int)  # 1 = 65+
    return t


def proxy_detection(df: pd.DataFrame, cfg: dict) -> dict:
    """For each model input, how well does it alone predict race / sex / age? (AUC)"""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    pop = _population(df)
    features = [f for f in cfg["model_features"] if f in pop]
    flag_at = float(cfg["fairness"]["proxy_auc_flag_threshold"])
    out: dict[str, list] = {}
    for name, y in _targets(pop).items():
        counts = y.value_counts()
        if len(counts) < 2 or counts.min() < 10:
            out[name] = []
            continue
        cv = StratifiedKFold(n_splits=min(5, int(counts.min())), shuffle=True,
                             random_state=int(cfg["random_seed"]))
        ranking = []
        for f in features:
            col = pop.loc[y.index, f]
            num = to_numeric(col)
            x = (num.fillna(num.median()) if num.notna().mean() >= 0.5
                 else pd.Series(pd.factorize(col.astype(str))[0], index=col.index))
            tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20,
                                          random_state=int(cfg["random_seed"]))
            proba = cross_val_predict(tree, x.to_frame(), y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, proba)
            auc = max(auc, 1 - auc)
            ranking.append({"feature": f, "auc": float(auc), "flagged": bool(auc > flag_at)})
        out[name] = sorted(ranking, key=lambda r: -r["auc"])
    return out


# ── 4. Comparison with real past decisions ────────────────────────────────────

def historical_comparison(df: pd.DataFrame, cfg: dict, hmda_df: pd.DataFrame | None = None) -> dict | None:
    """How the model's decisions compare to what lenders actually decided."""
    out: dict = {}
    pop = _population(df)
    if "hist_approved" in pop and pop["hist_approved"].notna().any():
        h = pop.dropna(subset=["hist_approved"]).copy()
        h["level"] = minority_level(h["tract_minority_population_percent"], cfg)
        approved_before = h[h["hist_approved"] == 1]
        by_level = []
        for lvl in ["low", "medium", "high"]:
            g = h[h["level"] == lvl]
            a = approved_before[approved_before["level"] == lvl]
            if len(g):
                by_level.append({"level": lvl, "n": len(g),
                                 "historical_approval": float(g["hist_approved"].mean()),
                                 "model_approval": float(g["decision"].mean()),
                                 "false_denial_rate": float((a["decision"] == 0).mean()) if len(a) else None})
        fdr = [r["false_denial_rate"] for r in by_level if r["false_denial_rate"] is not None]
        out["sample"] = {
            "n": len(h),
            "agreement": float((h["decision"] == h["hist_approved"]).mean()),
            "by_tract_level": by_level,
            "false_denial_gap": (max(fdr) - min(fdr)) if len(fdr) >= 2 else None,
        }
    if hmda_df is not None and "action_taken" in hmda_df:
        act = to_numeric(hmda_df["action_taken"])
        real = hmda_df[act.isin([1, 2, 3])].copy()
        real["approved"] = to_numeric(real["action_taken"]).isin([1, 2]).astype(int)
        real["level"] = minority_level(real["tract_minority_population_percent"], cfg)
        out["hmda"] = {
            "n": len(real),
            "approval_by_tract_level": real.groupby("level")["approved"].mean().to_dict(),
            "approval_by_race": real[~real["derived_race"].isin(EXCLUDED_GROUPS)]
                                .groupby("derived_race")["approved"].mean().to_dict(),
        }
    return out or None


# ── 5. Severity and verdict ───────────────────────────────────────────────────

def summarize(gf: dict, swaps: dict, proxies: dict, cfg: dict) -> tuple[dict, str, str]:
    by_factor = {r["varied_factor"]: r for r in swaps.get("summary", [])}
    race_proxies = [p for p in proxies.get("race", []) if p["flagged"]]
    factors = {}
    for name, (dim, swap_keys) in FACTORS.items():
        reasons = []
        gap = gf["gaps"].get(dim)
        gap_sev = gap["severity"] if gap else "none"
        if gap and gap["flagged"]:
            reasons.append(f"approval ratio {gap['di_ratio']:.2f} ({gap['worst_group']} vs "
                           f"{gap['best_group']}), below the 0.80 guideline")
        swap_rows = [by_factor[k] for k in swap_keys if k in by_factor]
        swap_sev = _max_sev(*[r["severity"] for r in swap_rows]) if swap_rows else "none"
        for r in swap_rows:
            bias_flips = r["n_flipped"] - r["n_inconsistent"]
            if bias_flips:
                reasons.append(f"{bias_flips / r['n_pairs']:.0%} of matched pairs flipped "
                               f"({bias_flips}/{r['n_pairs']}) when only this changed")
        if name == "neighborhood":
            tract_px = [p for p in race_proxies if "tract" in p["feature"]]
            for p in tract_px[:2]:
                reasons.append(f"{p['feature']} predicts race (AUC {p['auc']:.2f})")
        factors[name] = {"rating": _max_sev(gap_sev, swap_sev), "group_gap_severity": gap_sev,
                         "swap_severity": swap_sev, "reasons": reasons}

    strong = any(f["swap_severity"] == "high"
                 or (f["group_gap_severity"] in ("medium", "high")
                     and SEVERITY_ORDER.index(f["swap_severity"]) >= 2)
                 for f in factors.values())
    flagged = [n for n, f in factors.items() if f["rating"] != "none"]
    cons = swaps.get("consistency") or {}
    inconsistent = cons.get("severity", "none") != "none"
    if inconsistent:
        flagged.append(f"consistency ({cons['n_inconsistent']} identical applications got "
                       "different decisions)")
    if strong:
        verdict = "unfair"
        reason = ("Decisions change when only a protected factor or its stand-in changes, "
                  "which is direct evidence of bias: " + ", ".join(flagged) + ".")
    elif flagged:
        verdict = "needs_review"
        reason = ("Disparities found that need human review: " + ", ".join(flagged)
                  + ". Gaps alone don't prove discrimination.")
    else:
        verdict = "fair"
        reason = "No meaningful gaps or decision flips were found in the tested factors."
    return factors, verdict, reason


# ── Run everything ────────────────────────────────────────────────────────────

def run_full_audit(df: pd.DataFrame, cfg: dict, hmda_df: pd.DataFrame | None = None) -> dict:
    """Run every test and save output/audit_results.json."""
    gf = group_fairness(df, cfg)
    swaps = swap_test(df, cfg)
    proxies = proxy_detection(df, cfg)
    hist = historical_comparison(df, cfg, hmda_df)
    factors, verdict, reason = summarize(gf, swaps, proxies, cfg)

    test_set = df.get("test_set", pd.Series("", index=df.index))
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_total": int(len(df)),
        "n_population": int((test_set == "population").sum()),
        "n_matched": int((test_set == "matched").sum()),
        "n_pair_groups": int(df.loc[test_set == "matched", "pair_group_id"].nunique())
        if "pair_group_id" in df else 0,
        "overall_approval_rate": float(df["decision"].mean()),
        "model_features": cfg["model_features"],
        "group_fairness": gf,
        "swap_tests": swaps,
        "consistency": swaps.get("consistency"),
        "proxy_rankings": proxies,
        "historical": hist,
        "factors": factors,
        "verdict": verdict,
        "verdict_reason": reason,
        "thresholds": {
            "disparate_impact": cfg["fairness"]["disparate_impact_threshold"],
            "proxy_auc": cfg["fairness"]["proxy_auc_flag_threshold"],
            "flip_rate": cfg["fairness"]["flip_rate_severity"],
            "low_minority_pct": cfg["low_minority_threshold"],
            "high_minority_pct": cfg["high_minority_threshold"],
        },
        "limitations": [
            "HMDA data has no credit scores, so group gaps show disparities worth review, "
            "not proof of discrimination.",
            "Not every feature correlated with race is unfair; people decide what is justified.",
            "Swapped applicants combine real values and may occasionally be unrealistic.",
            "Joint applicants and groups with missing demographics are excluded from comparisons.",
        ],
    }

    out = resolve_path(cfg["output_dir"]) / "audit_results.json"
    write_json(results, out)

    print(f"  Verdict: {verdict.upper().replace('_', ' ')}")
    for name, f in factors.items():
        print(f"    {name:<13} {f['rating'].upper()}")
    print(f"  Saved {out}")
    return results
