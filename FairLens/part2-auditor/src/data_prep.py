"""
data_prep.py: load the HMDA file and apply the same filter Part 1 uses.

Works with the real HMDA download (hyphenated column names) and with
create_synthetic_hmda.py's file (underscored names), CSV or Excel.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.utils import DEFAULT_FEATURES, AuditorError, read_table, resolve_path, to_numeric

logger = logging.getLogger("auditor.data_prep")

NEEDED_FOR_AUDIT = ["census_tract", "tract_minority_population_percent",
                    "derived_race", "derived_sex"]


def load_hmda(cfg: dict) -> pd.DataFrame:
    """Load and filter the HMDA data named in config.yaml (hmda_path)."""
    path = resolve_path(cfg["hmda_path"], must_exist=True)
    if not path.exists():
        raise AuditorError(
            f"HMDA file not found: {path}\n"
            "  Put the real HMDA file there (or change hmda_path in config.yaml).\n"
            "  For a quick test only, run:  python create_synthetic_hmda.py")

    df = read_table(path, dtype={"census_tract": str})
    print(f"  Loaded {len(df):,} rows from {path.name}")

    for col, value in (cfg.get("hmda_filters") or {}).items():
        col = str(col).replace("-", "_")
        if col in df.columns:
            df = df[to_numeric(df[col]) == float(value)]
    print(f"  {len(df):,} rows after the home-purchase filter")

    # Part 1's bank model needs these; without them the two parts can't work together.
    required = [c for c in DEFAULT_FEATURES + NEEDED_FOR_AUDIT if c not in df.columns]
    if required:
        raise AuditorError("The HMDA file is missing columns the auditor needs: "
                           + ", ".join(required))
    # Other configured features that this file doesn't have are skipped, not fatal.
    absent = [c for c in cfg["model_features"] if c not in df.columns]
    if absent:
        print(f"  Note: skipping {len(absent)} model_features not in this HMDA file: "
              + ", ".join(absent))
        cfg["model_features"] = [c for c in cfg["model_features"] if c in df.columns]
    for c in DEFAULT_FEATURES:                       # always send what Part 1 needs
        if c not in cfg["model_features"]:
            cfg["model_features"].append(c)
    if df.empty:
        raise AuditorError("No rows left after filtering. Check hmda_filters in config.yaml.")
    return df.reset_index(drop=True)
