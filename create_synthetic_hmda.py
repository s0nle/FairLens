"""
create_synthetic_hmda.py
─────────────────────────
Generates a synthetic HMDA-style CSV that matches the column structure
expected by the auditor's generator and data_prep.load_hmda().

Run once before `python main.py generate`:
    python create_synthetic_hmda.py

Output: data/hmda_la_2018_lar.csv  (~5,000 rows)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N    = 5000

rng = np.random.default_rng(SEED)

# ── Census tracts with realistic minority-population percentages ──────────────
TRACTS = {
    # low minority (< 30 %)
    "06037101110": (8.0,  125.0),   # (minority_pct, tract_to_msa_income_pct)
    "06037102000": (12.0, 118.0),
    "06037105000": (22.0, 112.0),
    "06037106110": (28.0, 108.0),
    "06037107000": (18.0, 122.0),
    # medium minority (30–70 %)
    "06037201000": (40.0,  97.0),
    "06037202010": (55.0,  90.0),
    "06037203000": (62.0,  85.0),
    "06037204000": (48.0,  93.0),
    # high minority (> 70 %)
    "06037301000": (75.0,  72.0),
    "06037302000": (82.0,  65.0),
    "06037303010": (90.0,  58.0),
    "06037304000": (88.0,  61.0),
    "06037305000": (95.0,  54.0),
    "06037306000": (78.0,  68.0),
}

tract_ids = list(TRACTS.keys())
# Assign tracts with higher weight to high-minority ones to make the bias signal strong
weights = np.array([
    1, 1, 1, 1, 1,   # low
    2, 2, 2, 2,       # medium
    3, 3, 3, 3, 3, 3  # high
], dtype=float)
weights /= weights.sum()

assigned_tracts = rng.choice(tract_ids, size=N, p=weights)
minority_pcts   = np.array([TRACTS[t][0] for t in assigned_tracts])
tract_income    = np.array([TRACTS[t][1] for t in assigned_tracts])

# ── Income: correlated with tract minority % (historical wealth gap) ──────────
# High-minority tracts have lower average income
base_income = 120_000 - minority_pcts * 800 + rng.normal(0, 20_000, N)
base_income = np.clip(base_income, 25_000, 500_000)

# ── Loan amounts ──────────────────────────────────────────────────────────────
property_value = base_income * rng.uniform(2.5, 6.0, N)
property_value = np.clip(property_value, 80_000, 2_000_000)
loan_amount    = property_value * rng.uniform(0.60, 0.97, N)
loan_amount    = np.clip(loan_amount, 50_000, 1_800_000)

# ── Derived columns ───────────────────────────────────────────────────────────
BUCKET_LTV = ["<20%","20-<30%","30-<40%","40-<50%","50-<60%","60-<70%","70-<80%","80-<90%","90-<100%",">100%"]
ltv_raw    = (loan_amount / property_value) * 100
def _ltv_bucket(v):
    if v < 20:   return "<20%"
    if v < 30:   return "20-<30%"
    if v < 40:   return "30-<40%"
    if v < 50:   return "40-<50%"
    if v < 60:   return "50-<60%"
    if v < 70:   return "60-<70%"
    if v < 80:   return "70-<80%"
    if v < 90:   return "80-<90%"
    if v < 100:  return "90-<100%"
    return ">100%"
ltv_buckets = [_ltv_bucket(v) for v in ltv_raw]

DTI_BUCKETS = ["<20%","20-<30%","30-<36%","36","37","38","39","40","41","42","43","43-<50%","50%-60%",">60%","NA"]
dti_buckets = rng.choice(DTI_BUCKETS[:8], size=N)   # keep it realistic

# ── Race / ethnicity / sex (protected — separate from features) ───────────────
# Race correlated with tract minority % to create proxy signal
race_choices_high = ["Black or African American", "Black or African American",
                     "Hispanic or Latino", "Asian", "White"]
race_choices_low  = ["White", "White", "White", "Asian", "Black or African American"]

derived_race = []
for mp in minority_pcts:
    if mp > 70:
        derived_race.append(rng.choice(race_choices_high))
    elif mp > 30:
        derived_race.append(rng.choice(["White","Black or African American","Asian","Hispanic or Latino"]))
    else:
        derived_race.append(rng.choice(race_choices_low))
derived_race = np.array(derived_race)

# Map derived_race → applicant_race_1 code
race_code_map = {
    "American Indian or Alaska Native":   "1",
    "Asian":                              "2",
    "Black or African American":          "3",
    "Native Hawaiian or Other Pacific Islander": "4",
    "White":                              "5",
    "Hispanic or Latino":                 "5",   # ethnicity handled separately
}
applicant_race_1 = [race_code_map.get(r, "5") for r in derived_race]

derived_ethnicity = np.where(
    derived_race == "Hispanic or Latino",
    "Hispanic or Latino",
    "Not Hispanic or Latino"
)
derived_sex = rng.choice(["Male", "Male", "Female", "Joint"], size=N)
applicant_sex = np.where(derived_sex == "Female", "2", "1")

age_buckets = rng.choice(["25-34","35-44","45-54","55-64","65-74",">74"], size=N,
                         p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05])
age_above_62 = np.where(
    np.isin(age_buckets, ["65-74",">74"]), "1", "2"
)

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    # IDs / geography
    "census_tract":   assigned_tracts,
    "county_code":    "037",
    "state_code":     "06",
    "msa_md":         "31080",
    # Tract-level
    "tract_minority_population_percent":  minority_pcts,
    "tract_to_msa_income_percentage":     tract_income,
    "ffiec_msa_md_median_family_income":  75_000.0,
    "tract_population":                   rng.uniform(2_000, 12_000, N).round(),
    "tract_one_to_four_family_homes":     rng.uniform(300, 3_000, N).round(),
    "tract_median_age_of_housing_units":  rng.uniform(15.0, 60.0, N).round(1),
    # Application
    "loan_type":      rng.choice(["1","2","3"], size=N, p=[0.75, 0.15, 0.10]),
    "loan_purpose":   "1",
    "business_or_commercial_purpose": "2",
    "open_end_line_of_credit": "2",
    "occupancy_type": "1",
    "loan_amount":    loan_amount.round(-3),
    "property_value": property_value.round(-3),
    "income":         base_income.round(-3),
    "loan_to_value_ratio": ltv_buckets,
    "debt_to_income_ratio": dti_buckets,
    "loan_term":      rng.choice([180, 360], size=N, p=[0.20, 0.80]),
    "intro_rate_period": rng.choice([0, 12, 24, 60, 120], size=N, p=[0.60, 0.10, 0.10, 0.10, 0.10]),
    "lien_status":    rng.choice(["1","2"], size=N, p=[0.95, 0.05]),
    "construction_method": rng.choice(["1","2"], size=N, p=[0.95, 0.05]),
    "manufactured_home_secured_property_type":  rng.choice(["1","2","3"], size=N),
    "manufactured_home_land_property_interest": rng.choice(["1","2","5"], size=N),
    "total_units":    rng.choice(["1","2"], size=N, p=[0.95, 0.05]),
    "multifamily_affordable_units": rng.choice([0, 1], size=N, p=[0.95, 0.05]),
    "submission_of_application": rng.choice(["1","2","3"], size=N),
    "initially_payable_to_institution": rng.choice(["1","2"], size=N),
    "aus_1":          rng.choice(["1","2","3"], size=N),
    "applicant_credit_score_type":    rng.choice(["1","2","3"], size=N, p=[0.50, 0.30, 0.20]),
    "co_applicant_credit_score_type": rng.choice(["1","2","9"], size=N, p=[0.30, 0.30, 0.40]),
    # Protected columns
    "derived_race":        derived_race,
    "applicant_race_1":    applicant_race_1,
    "applicant_race_2":    "7",
    "applicant_race_3":    "7",
    "applicant_race_4":    "7",
    "applicant_race_5":    "7",
    "applicant_race_observed":    rng.choice(["1","2","3"], size=N),
    "co_applicant_race_1":        "8",
    "co_applicant_race_2":        "8",
    "co_applicant_race_3":        "8",
    "co_applicant_race_4":        "8",
    "co_applicant_race_5":        "8",
    "co_applicant_race_observed": "3",
    "derived_ethnicity":          derived_ethnicity,
    "applicant_ethnicity_1":      np.where(derived_ethnicity == "Hispanic or Latino", "1", "2"),
    "applicant_ethnicity_2":      "7",
    "applicant_ethnicity_3":      "7",
    "applicant_ethnicity_4":      "7",
    "applicant_ethnicity_5":      "7",
    "applicant_ethnicity_observed": rng.choice(["1","2","3"], size=N),
    "co_applicant_ethnicity_1":   "8",
    "co_applicant_ethnicity_2":   "8",
    "co_applicant_ethnicity_3":   "8",
    "co_applicant_ethnicity_4":   "8",
    "co_applicant_ethnicity_5":   "8",
    "co_applicant_ethnicity_observed": "4",
    "derived_sex":                derived_sex,
    "applicant_sex":               applicant_sex,
    "applicant_sex_observed":      rng.choice(["1","2","3"], size=N),
    "co_applicant_sex":            rng.choice(["1","2","5"], size=N),
    "co_applicant_sex_observed":   rng.choice(["1","2","3"], size=N),
    "applicant_age":               age_buckets,
    "applicant_age_above_62":      age_above_62,
    "co_applicant_age":            rng.choice(["25-34","35-44","45-54","55-64","8888"], size=N),
    "co_applicant_age_above_62":   rng.choice(["1","2","9999"], size=N),
    # Post-decision columns (kept to allow historical comparison)
    "action_taken": rng.choice(["1","3","6"], size=N, p=[0.70, 0.20, 0.10]),
    "denial_reason_1": rng.choice(["1","2","3","7","9"], size=N),
    "denial_reason_2": "9",
    "denial_reason_3": "9",
    "denial_reason_4": "9",
    "interest_rate":   rng.uniform(3.0, 7.5, N).round(3),
    "rate_spread":     rng.choice([float("nan"), 0.5, 1.0, 1.5], size=N),
})

out_path = Path("data/hmda_la_2018_lar.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
print(f"  Columns: {len(df.columns)}")
print(f"  Tracts: {df['census_tract'].nunique()} unique tracts")
print(f"  Minority pct range: {df['tract_minority_population_percent'].min():.0f}% to {df['tract_minority_population_percent'].max():.0f}%")
print(f"  Race distribution:\n{df['derived_race'].value_counts().to_string()}")
