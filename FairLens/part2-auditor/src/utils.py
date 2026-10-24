"""
utils.py: shared helpers for the auditor.

- AuditorError: the one exception type main.py catches and shows nicely
- load_config(): reads config.yaml and cleans it up
- resolve_path(): turns config paths into real file paths
- read_table(): reads CSV or Excel files
- to_numeric(): converts HMDA values like "20%-<30%" or "Exempt" to numbers
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("auditor")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCEL_TYPES = {".xlsx", ".xlsm", ".xls"}

# Tract-level columns: these always move together when an applicant's tract changes.
TRACT_COLUMNS = [
    "census_tract",
    "tract_minority_population_percent",
    "tract_to_msa_income_percentage",
    "ffiec_msa_md_median_family_income",
    "tract_population",
    "tract_owner_occupied_units",
    "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]

DEFAULT_FEATURES = [
    "income",
    "loan_amount",
    "property_value",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "tract_minority_population_percent",
    "tract_to_msa_income_percentage",
]


class AuditorError(Exception):
    """A problem the user can fix (bad file, missing column, wrong path)."""


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path | None = None) -> dict:
    """Load config.yaml and normalise it so the rest of the code can trust it."""
    try:
        import yaml
    except ImportError as exc:
        raise AuditorError("PyYAML is not installed. Run:  pip install pyyaml") from exc

    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if not path.exists():
        raise AuditorError(f"Config file not found: {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise AuditorError(f"Could not read {path.name}: {exc}") from exc

    cfg.setdefault("random_seed", 42)
    cfg.setdefault("n_population", 600)
    cfg.setdefault("n_matched_pairs", 250)
    cfg.setdefault("high_minority_threshold", 70.0)
    cfg.setdefault("low_minority_threshold", 30.0)
    cfg.setdefault("output_dir", "output")
    cfg.setdefault("min_group_size", 20)
    cfg.setdefault("applications_path", "data/applications_for_bank.csv")
    cfg.setdefault("audit_key_path", "data/audit_key.csv")
    cfg.setdefault("results_path", "data/bank_results.csv")
    cfg.setdefault("hmda_path", "data/hmda_la_2018_lar.csv")
    cfg.setdefault("hmda_filters", {})
    cfg.setdefault("protected_columns", {})
    cfg.setdefault("post_decision_columns", [])
    fair = cfg.setdefault("fairness", {})
    fair.setdefault("disparate_impact_threshold", 0.80)
    fair.setdefault("proxy_auc_flag_threshold", 0.65)
    fair.setdefault("severity", {"high": 0.60, "medium": 0.75, "low": 0.80})
    fair.setdefault("flip_rate_severity", {"high": 0.10, "medium": 0.05, "low": 0.02})
    cfg.setdefault("llm", {})

    # Model features: keep plain column names only, and never allow protected
    # or post-decision columns through (a safety net for config typos).
    raw = cfg.get("model_features") or DEFAULT_FEATURES
    names = [str(f).strip().replace("-", "_") for f in raw if isinstance(f, str) and f.strip()]
    blocked = set(protected_column_list(cfg)) | {c.replace("-", "_") for c in cfg["post_decision_columns"]}
    dropped = [f for f in names if f in blocked]
    if dropped:
        logger.warning("Removed protected/post-decision columns from model_features: %s", dropped)
    cfg["model_features"] = [f for f in names if f not in blocked]
    if not cfg["model_features"]:
        raise AuditorError("config.yaml has no usable model_features.")
    return cfg


def protected_groups(cfg: dict) -> dict[str, list[str]]:
    """{'race': [...], 'ethnicity': [...], 'sex': [...], 'age': [...]} with underscore names."""
    groups = cfg.get("protected_columns") or {}
    return {k: [str(c).replace("-", "_") for c in (v or [])] for k, v in groups.items()}


def protected_column_list(cfg: dict) -> list[str]:
    cols = [c for cols in protected_groups(cfg).values() for c in cols]
    for c in ("derived_race", "derived_ethnicity", "derived_sex", "applicant_age"):
        if c not in cols:
            cols.append(c)
    return cols


def resolve_path(path_value: str | Path, must_exist: bool = False) -> Path:
    """Relative paths work from the project folder, or from where you ran the command."""
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    if must_exist and (Path.cwd() / p).exists():
        return Path.cwd() / p
    return PROJECT_ROOT / p


# ── Files ─────────────────────────────────────────────────────────────────────

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Real HMDA uses hyphens ('applicant_race-1'); we use underscores everywhere."""
    df = df.copy()
    df.columns = [str(c).strip().replace("-", "_") for c in df.columns]
    return df


def read_table(path: str | Path, dtype: dict | None = None) -> pd.DataFrame:
    """Read a CSV or Excel file and standardise its column names."""
    path = Path(path)
    if not path.exists():
        raise AuditorError(f"File not found: {path}")
    if path.suffix.lower() in EXCEL_TYPES:
        try:
            return standardize_columns(pd.read_excel(path, dtype=dtype))
        except ImportError as exc:
            raise AuditorError("Reading Excel needs openpyxl: pip install openpyxl "
                               "(or save the file as CSV).") from exc
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return standardize_columns(pd.read_csv(path, low_memory=False, dtype=dtype,
                                                   encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise AuditorError(f"Could not read {path.name} as CSV or Excel.")


# ── Values ────────────────────────────────────────────────────────────────────

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _text_to_number(text: str) -> float:
    s = str(text).strip()
    if s.lower() in {"", "na", "nan", "none", "exempt"}:
        return np.nan
    nums = [float(n) for n in _NUMBER.findall(s)]
    if not nums:
        return np.nan
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2
    if s.startswith("<"):
        return max(nums[0] - 5, 0.0)
    if s.startswith(">"):
        return nums[0] + 5
    return nums[0]


def to_numeric(series: pd.Series) -> pd.Series:
    """Numbers stay numbers; HMDA ranges become midpoints; 'Exempt'/'NA' become NaN.
    Same rules as Part 1's bank_model.py, so both sides read values identically."""
    out = pd.to_numeric(series, errors="coerce").astype(float)
    needs = out.isna() & series.notna()
    if needs.any():
        raw = series[needs].astype(str)
        lookup = {v: _text_to_number(v) for v in raw.unique()}
        out[needs] = raw.map(lookup)
    return out


def minority_level(pct: pd.Series, cfg: dict) -> pd.Series:
    low, high = float(cfg["low_minority_threshold"]), float(cfg["high_minority_threshold"])
    vals = to_numeric(pct)
    out = pd.Series(np.where(vals < low, "low", np.where(vals > high, "high", "medium")),
                    index=pct.index, dtype=object)
    out[vals.isna()] = np.nan
    return out


# ── JSON ──────────────────────────────────────────────────────────────────────

def to_jsonable(obj: Any) -> Any:
    """Make numpy/pandas values safe for json.dump (NaN becomes null)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def write_json(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")
