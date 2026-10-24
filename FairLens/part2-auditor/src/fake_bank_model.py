"""
fake_bank_model.py: TEST-ONLY stand-in for Part 1's bank model.

It lets you test the auditor before Part 1 is connected. For the real demo,
decisions come from Part 1 (bank_model.py / the Colab notebook), not from here.

Modes:
  biased      approves on income, debt-to-income and loan-to-value, but cuts approval
              odds in high-minority tracts (the auditor SHOULD flag neighborhood)
  fair        same, with no neighborhood penalty (the auditor should NOT flag neighborhood)
  age_biased  penalizes older housing stock as an age-correlated proxy, if that column
              is in the applications file

Usage (from the fairlenspart2 folder):
  python src/fake_bank_model.py --mode biased --out test_results/biased_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):                       # allow `python src/fake_bank_model.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import AuditorError, read_table, to_numeric  # noqa: E402

MODES = ("biased", "fair", "age_biased")


def _log_z(values: pd.Series) -> pd.Series:
    """Center a positive money column on the log scale (works for dollars or thousands)."""
    logs = np.log(values.where(values > 0))
    return ((logs - logs.median()) / 0.5).fillna(0.0)


def fake_score(apps: pd.DataFrame, mode: str = "biased", seed: int = 42,
               high_minority: float = 70.0) -> pd.DataFrame:
    """Return case_id, decision, probability for every row in apps."""
    if mode not in MODES:
        raise AuditorError(f"Unknown mode '{mode}'. Use one of: {', '.join(MODES)}")
    col = lambda name: to_numeric(apps[name]) if name in apps else pd.Series(np.nan, index=apps.index)

    income = _log_z(col("income"))
    dti = col("debt_to_income_ratio").fillna(36)
    ltv = col("loan_to_value_ratio").fillna(80)
    z = 1.3 + 0.9 * income - 0.06 * (dti - 36) - 0.05 * (ltv - 80)

    if mode == "biased":
        z = z - 1.4 * (col("tract_minority_population_percent") > high_minority)
    elif mode == "age_biased":
        if "tract_median_age_of_housing_units" not in apps:
            raise AuditorError("age_biased mode needs tract_median_age_of_housing_units in the "
                               "applications file; it isn't one of the model features.")
        z = z - 0.04 * (col("tract_median_age_of_housing_units").fillna(40) - 40)

    # Deterministic "noise" from the applicant's own inputs: identical applications always
    # get identical answers, like a real model.
    hashed = pd.util.hash_pandas_object(apps.drop(columns=["case_id"], errors="ignore")
                                        .astype(str), index=False)
    noise = pd.Series((hashed % 10_000).to_numpy() / 10_000 - 0.5, index=apps.index) * 0.6
    prob = 1 / (1 + np.exp(-(z + noise)))
    return pd.DataFrame({"case_id": apps["case_id"].astype(str),
                         "decision": (prob >= 0.5).astype(int),
                         "probability": prob.round(4)})


def main() -> None:
    p = argparse.ArgumentParser(description="TEST-ONLY fake bank model")
    p.add_argument("--mode", default="biased", choices=MODES)
    p.add_argument("--applications", default="data/applications_for_bank.csv")
    p.add_argument("--out", default="test_results/biased_results.csv")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    try:
        apps = read_table(a.applications, dtype={"case_id": str})
        res = fake_score(apps, a.mode, a.seed)
    except AuditorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    print(f"[TEST ONLY] fake bank model ({a.mode}) scored {len(res):,} applicants, "
          f"approval rate {res['decision'].mean():.1%} -> {a.out}")


if __name__ == "__main__":
    main()
