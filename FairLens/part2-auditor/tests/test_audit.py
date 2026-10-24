"""Audit math checks on tiny hand-made data. Run:  pytest tests/ -v"""
import pandas as pd
import pytest

from src.audit import group_fairness, swap_test
from src.ingest import validate

CFG = {
    "min_group_size": 1, "low_minority_threshold": 30.0, "high_minority_threshold": 70.0,
    "model_features": ["income", "tract_minority_population_percent"],
    "fairness": {"disparate_impact_threshold": 0.80,
                 "severity": {"high": 0.60, "medium": 0.75, "low": 0.80},
                 "flip_rate_severity": {"high": 0.10, "medium": 0.05, "low": 0.02}},
}


def test_gap_and_disparate_impact():
    # White: 4 of 5 approved (80%); Black: 2 of 5 approved (40%)
    df = pd.DataFrame({
        "test_set": "population",
        "derived_race": ["White"] * 5 + ["Black or African American"] * 5,
        "decision": [1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    })
    g = group_fairness(df, CFG)["gaps"]["derived_race"]
    assert g["best_group"] == "White" and g["best_rate"] == pytest.approx(0.8)
    assert g["worst_rate"] == pytest.approx(0.4)
    assert g["gap"] == pytest.approx(0.4)
    assert g["di_ratio"] == pytest.approx(0.5)
    assert g["flagged"] and g["severity"] == "high"


def test_flip_rate_and_consistency():
    # 4 groups: neighborhood swap flips in 2 of 4; sex swap never changes model inputs
    rows = []
    for i, (base_d, tract_d, sex_d) in enumerate([(1, 0, 1), (1, 0, 1), (1, 1, 1), (0, 0, 1)]):
        g = f"P-{i}"
        rows += [
            dict(pair_group_id=g, varied_factor="base", decision=base_d, probability=0.6,
                 income=100, tract_minority_population_percent=10),
            dict(pair_group_id=g, varied_factor="tract_high_minority", decision=tract_d,
                 probability=0.4, income=100, tract_minority_population_percent=90),
            dict(pair_group_id=g, varied_factor="sex_flip", decision=sex_d, probability=0.6,
                 income=100, tract_minority_population_percent=10),
        ]
    df = pd.DataFrame(rows).assign(test_set="matched", case_id=lambda d: d.index.astype(str))
    res = swap_test(df, CFG)
    by = {r["varied_factor"]: r for r in res["summary"]}
    assert by["tract_high_minority"]["n_flipped"] == 2
    assert by["tract_high_minority"]["flip_rate"] == pytest.approx(0.5)
    assert by["tract_high_minority"]["approve_to_deny"] == 2
    # Group P-3's base was denied and its sex variant approved: identical inputs, different answer
    assert by["sex_flip"]["n_inconsistent"] == 1
    assert res["consistency"]["n_inconsistent"] == 1


def test_penalty_without_flips_is_still_caught():
    # Everyone stays approved, but the same applicant loses ~20 points of approval chance
    # when only the neighborhood changes: a clear penalty even with zero flips.
    rows = []
    for i in range(40):
        g, p = f"P-{i}", 0.95 - (i % 5) * 0.01
        rows += [
            dict(pair_group_id=g, varied_factor="base", decision=1, probability=p,
                 income=100 + i, tract_minority_population_percent=10),
            dict(pair_group_id=g, varied_factor="tract_high_minority", decision=1,
                 probability=p - 0.20 + (i % 3) * 0.01, income=100 + i,
                 tract_minority_population_percent=90),
        ]
    df = pd.DataFrame(rows).assign(test_set="matched", case_id=lambda d: d.index.astype(str))
    cfg = dict(CFG, fairness=dict(CFG["fairness"], prob_shift_severity={"high": 0.10,
               "medium": 0.05, "low": 0.02}, shift_p_value=0.01))
    row = swap_test(df, cfg)["summary"][0]
    assert row["n_flipped"] == 0 and row["flip_severity"] == "none"
    assert row["avg_shift_when_inputs_changed"] == pytest.approx(-0.19, abs=0.01)
    assert row["shift_severity"] == "high" and row["severity"] == "high"


def test_validate_reports_problems():
    key = pd.DataFrame({"case_id": ["C-1", "C-2", "C-3"]})
    results = pd.DataFrame({"case_id": ["C-1", "C-1", "C-9"], "decision": [1, 5, 0]})
    text = " ".join(validate(results, key)).lower()
    assert "duplicate" in text and "invalid decision" in text
    assert "missing" in text and "not in the audit key" in text
