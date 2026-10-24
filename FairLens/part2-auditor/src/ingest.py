"""
ingest.py: Module 2. Load the bank's results and join them to the audit key.

validate() returns plain-English warnings; ingest() stops with an AuditorError
only when the audit can't run at all (no case_id/decision column, no matches).
"""
from __future__ import annotations

import logging

import pandas as pd

from src.utils import AuditorError, read_table, resolve_path

logger = logging.getLogger("auditor.ingest")

REQUIRED = ["case_id", "decision"]


def validate(results_df: pd.DataFrame, audit_key_df: pd.DataFrame) -> list[str]:
    """Check a results file against the audit key. Returns a list of warnings."""
    warnings: list[str] = []
    missing_cols = [c for c in REQUIRED if c not in results_df.columns]
    if missing_cols:
        return [f"Results file is missing required column(s): {', '.join(missing_cols)}. "
                "Expected: case_id, decision (1=approve, 0=deny), probability."]

    ids = results_df["case_id"].astype(str)
    key_ids = set(audit_key_df["case_id"].astype(str))

    dupes = ids[ids.duplicated()]
    if len(dupes):
        warnings.append(f"{len(dupes)} duplicate case_ids in the results (e.g. {dupes.iloc[0]}). "
                        "Only the first row for each is used.")

    decision = pd.to_numeric(results_df["decision"], errors="coerce")
    bad = results_df.loc[~decision.isin([0, 1]), "decision"]
    if len(bad):
        warnings.append(f"{len(bad)} rows have invalid decision values (e.g. {bad.iloc[0]}); "
                        "expected 0 or 1. Those rows are ignored.")

    missing = key_ids - set(ids)
    if missing:
        warnings.append(f"{len(missing)} case_ids from the audit key have no matching row in "
                        f"the results (missing), e.g. {sorted(missing)[0]}. The bank should "
                        "return one row per applicant.")
    extra = set(ids) - key_ids
    if extra:
        warnings.append(f"{len(extra)} case_ids in the results are not in the audit key "
                        f"(e.g. {sorted(extra)[0]}). Was this file made from a different "
                        "applications_for_bank.csv?")

    if "probability" in results_df.columns:
        prob = pd.to_numeric(results_df["probability"], errors="coerce")
        if ((prob < 0) | (prob > 1)).any():
            warnings.append("Some probability values are outside 0-1.")
    return warnings


def ingest(cfg: dict, results_df: pd.DataFrame | None = None,
           audit_key_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return audit key rows joined with the bank's decision and probability."""
    if results_df is None:
        path = resolve_path(cfg["results_path"], must_exist=True)
        if not path.exists():
            raise AuditorError(f"Results file not found: {path}")
        results_df = read_table(path, dtype={"case_id": str})
    if audit_key_df is None:
        path = resolve_path(cfg["audit_key_path"], must_exist=True)
        if not path.exists():
            raise AuditorError(f"Audit key not found: {path}. Run:  python main.py generate")
        audit_key_df = read_table(path, dtype={"case_id": str})

    warnings = validate(results_df, audit_key_df)
    for w in warnings:
        print(f"  WARNING: {w}")
    missing_cols = [c for c in REQUIRED if c not in results_df.columns]
    if missing_cols:
        raise AuditorError(warnings[0])

    res = results_df.copy()
    res["case_id"] = res["case_id"].astype(str)
    res = res.drop_duplicates("case_id", keep="first")
    res["decision"] = pd.to_numeric(res["decision"], errors="coerce")
    res.loc[~res["decision"].isin([0, 1]), "decision"] = pd.NA
    res["probability"] = (pd.to_numeric(res["probability"], errors="coerce")
                          if "probability" in res.columns else pd.NA)
    res = res.dropna(subset=["decision"])
    if res.empty:
        raise AuditorError("No valid decisions (0 or 1) in the results file.")
    res["decision"] = res["decision"].astype(int)

    key = audit_key_df.copy()
    key["case_id"] = key["case_id"].astype(str)
    key = key.drop(columns=[c for c in ("decision", "probability") if c in key.columns])
    merged = key.merge(res[["case_id", "decision", "probability"]], on="case_id", how="inner")
    if merged.empty:
        raise AuditorError("None of the case_ids in the results match the audit key. "
                           "Make sure both files come from the same `python main.py generate` run.")
    print(f"  Matched {len(merged):,} of {len(key):,} applicants "
          f"({merged['decision'].mean():.1%} approved)")
    return merged
