"""Generator checks: matched pairs change one factor, tracts stay consistent,
and nothing protected reaches the bank's file. Run:  pytest tests/ -v"""
import numpy as np
import pandas as pd

from src.generator import generate
from src.utils import TRACT_COLUMNS, protected_column_list

FEATURES = ["income", "loan_amount", "property_value", "debt_to_income_ratio",
            "loan_to_value_ratio", "tract_minority_population_percent",
            "tract_to_msa_income_percentage"]


def make_hmda(n=400, seed=0):
    rng = np.random.default_rng(seed)
    tracts = {f"0603700{i:04d}": (pct, 130 - pct) for i, pct in
              enumerate([10, 15, 20, 25, 45, 55, 75, 85, 90, 95])}
    ids = rng.choice(list(tracts), n)
    return pd.DataFrame({
        "census_tract": ids,
        "tract_minority_population_percent": [tracts[t][0] for t in ids],
        "tract_to_msa_income_percentage": [tracts[t][1] for t in ids],
        "income": rng.integers(40, 300, n),
        "loan_amount": rng.integers(100, 900, n) * 1000,
        "property_value": rng.integers(200, 1200, n) * 1000,
        "debt_to_income_ratio": rng.choice(["20%-<30%", "30%-<36%", "40", ">60%"], n),
        "loan_to_value_ratio": rng.uniform(50, 97, n).round(1),
        "derived_race": rng.choice(["White", "Black or African American", "Asian"], n),
        "derived_ethnicity": rng.choice(["Hispanic or Latino", "Not Hispanic or Latino"], n),
        "derived_sex": rng.choice(["Male", "Female"], n),
        "applicant_age": rng.choice(["25-34", "45-54", "65-74"], n),
        "applicant_age_above_62": "No",
        "applicant_race_1": "5", "applicant_sex": "1", "applicant_ethnicity_1": "2",
        "action_taken": rng.choice([1, 3], n, p=[0.8, 0.2]),
    })


def make_cfg(tmp_path):
    return {
        "random_seed": 1, "n_population": 100, "n_matched_pairs": 30,
        "low_minority_threshold": 30.0, "high_minority_threshold": 70.0,
        "model_features": FEATURES, "protected_columns": {}, "post_decision_columns": [],
        "applications_path": str(tmp_path / "apps.csv"),
        "audit_key_path": str(tmp_path / "key.csv"),
    }


def test_bank_file_has_no_protected_columns(tmp_path):
    cfg = make_cfg(tmp_path)
    out = generate(cfg, make_hmda())
    assert list(out["applications"].columns) == ["case_id"] + FEATURES
    assert not set(out["applications"].columns) & set(protected_column_list(cfg))
    assert out["applications"]["case_id"].is_unique


def test_matched_variants_change_only_one_factor(tmp_path):
    cfg = make_cfg(tmp_path)
    key = generate(cfg, make_hmda())["audit_key"]
    m = key[key["test_set"] == "matched"]
    base = m[m["varied_factor"] == "base"].set_index("pair_group_id")
    allowed = {
        "tract_high_minority": set(TRACT_COLUMNS),
        "race_white": {"derived_race", "applicant_race_1"},
        "race_black": {"derived_race", "applicant_race_1"},
        "ethnicity_flip": {"derived_ethnicity", "applicant_ethnicity_1"},
        "sex_flip": {"derived_sex", "applicant_sex"},
        "age_flip": {"applicant_age", "applicant_age_above_62"},
    }
    skip = {"case_id", "varied_factor", "pair_group_id"}
    for _, row in m[m["varied_factor"] != "base"].iterrows():
        b = base.loc[row["pair_group_id"]]
        changed = {c for c in key.columns if c not in skip
                   and str(row[c]) != str(b[c]) and not (pd.isna(row[c]) and pd.isna(b[c]))}
        assert changed, f"{row['varied_factor']} variant changed nothing"
        assert changed <= allowed[row["varied_factor"]], (row["varied_factor"], changed)


def test_tract_columns_stay_consistent(tmp_path):
    key = generate(make_cfg(tmp_path), make_hmda())["audit_key"]
    per_tract = key.groupby("census_tract")[["tract_minority_population_percent",
                                             "tract_to_msa_income_percentage"]].nunique()
    assert (per_tract == 1).all().all()


def test_base_is_low_minority_and_swap_is_high(tmp_path):
    key = generate(make_cfg(tmp_path), make_hmda())["audit_key"]
    m = key[key["test_set"] == "matched"]
    assert (m.loc[m["varied_factor"] == "base", "tract_minority_population_percent"] < 30).all()
    assert (m.loc[m["varied_factor"] == "tract_high_minority",
                  "tract_minority_population_percent"] > 70).all()
