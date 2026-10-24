# Fair Lens Audit Report

_Generated 2026-10-24T06:46:48+00:00 in template mode (no AI credentials or the AI call failed)._

## Overall verdict

**NEEDS REVIEW**: Disparities found that need human review: neighborhood, age. Gaps alone don't prove discrimination.

## What we tested

- Cases audited: 2119 (600 population applicants and 1519 matched applicants in 250 groups)
- Overall approval rate: 92.7%
- Model inputs checked: loan_type, loan_amount, property_value, income, debt_to_income_ratio, loan_to_value_ratio, loan_term, intro_rate_period, lien_status, construction_method, manufactured_home_secured_property_type, manufactured_home_land_property_interest, total_units, multifamily_affordable_units, submission_of_application, initially_payable_to_institution, aus_1, census_tract, tract_minority_population_percent, tract_to_msa_income_percentage, ffiec_msa_md_median_family_income, tract_population, tract_one_to_four_family_homes, tract_median_age_of_housing_units, county_code, state_code, msa_md, co_applicant_credit_score_type, applicant_credit_score_type
- Tests: group fairness, swap tests (one change at a time), proxy detection

## Bias summary by factor

| Factor | Rating | Evidence |
| --- | --- | --- |
| Neighborhood | MEDIUM | 5% of matched pairs flipped (13/250) when only this changed; tract_minority_population_percent predicts race (AUC 0.80); tract_median_age_of_housing_units predicts race (AUC 0.76) |
| Race | NONE | No gap or flips found |
| Ethnicity | NONE | No gap or flips found |
| Sex | NONE | No gap or flips found |
| Age | MEDIUM | approval ratio 0.69 (65-74 vs 55-64), below the 0.80 guideline |

## Key findings

### Group fairness (population set)

| Dimension | Best group | Worst group | Gap | Ratio | Flagged |
| --- | --- | --- | --- | --- | --- |
| Neighborhood (tract minority level) | low (92.9%) | high (91.5%) | 1.4 pts | 0.98 | no |
| Race | White (93.9%) | Asian (84.0%) | 9.9 pts | 0.89 | no |
| Sex | Female (91.9%) | Male (90.6%) | 1.3 pts | 0.99 | no |
| Age | 55-64 (100.0%) | 65-74 (69.2%) | 30.8 pts | 0.69 | YES |

Ratio = worst group's approval rate / best group's. Below 0.80 is flagged (the four-fifths rule).

### Swap tests (matched applicants, one change at a time)

| Change | Pairs | Flipped | Flip rate | Avg probability change | Severity |
| --- | --- | --- | --- | --- | --- |
| Neighborhood: moved to a high-minority tract | 250 | 13 | 5.2% | -0.189 | medium |
| Age moved across 62 | 250 | 0 | 0.0% | +0.000 | none |
| Ethnicity flipped (Hispanic / not Hispanic) | 247 | 0 | 0.0% | +0.000 | none |
| Race changed to Black | 216 | 0 | 0.0% | +0.000 | none |
| Race changed to White | 56 | 0 | 0.0% | +0.000 | none |
| Sex flipped (male / female) | 250 | 0 | 0.0% | +0.000 | none |

### Proxy detection: model inputs that predict race

| Model input | AUC | Flagged |
| --- | --- | --- |
| tract_minority_population_percent | 0.80 | YES |
| tract_median_age_of_housing_units | 0.76 | YES |
| tract_to_msa_income_percentage | 0.76 | YES |
| tract_one_to_four_family_homes | 0.75 | YES |
| income | 0.72 | YES |

AUC 0.5 = no information about race, 1.0 = a perfect stand-in. Above 0.65 is flagged.

## Example cases

- **P-0039** (Neighborhood: moved to a high-minority tract): tract_minority_population_percent 18.7 -> 84.6. Decision went from approved to denied (probability 0.549 -> 0.339); every other input was identical.
- **P-0044** (Neighborhood: moved to a high-minority tract): tract_minority_population_percent 17.37 -> 87.49. Decision went from approved to denied (probability 0.690 -> 0.471); every other input was identical.
- **P-0054** (Neighborhood: moved to a high-minority tract): tract_minority_population_percent 28.68 -> 95.66. Decision went from approved to denied (probability 0.618 -> 0.467); every other input was identical.
- **P-0067** (Neighborhood: moved to a high-minority tract): tract_minority_population_percent 8.0 -> 87.49. Decision went from approved to denied (probability 0.656 -> 0.463); every other input was identical.
- **P-0086** (Neighborhood: moved to a high-minority tract): tract_minority_population_percent 23.06 -> 84.6. Decision went from approved to denied (probability 0.528 -> 0.299); every other input was identical.

## Likely causes

- The model gives weight to neighborhood features. In Los Angeles County, a tract's minority share closely tracks applicants' race, so location acts as a stand-in for race even though race is never an input.
- Some group gaps appear without matching decision flips. These can come from legitimate differences such as income; review before acting.

## Recommended fixes

- Remove tract_minority_population_percent from the model inputs; it has no legitimate role in judging repayment.
- Review other neighborhood inputs (for example tract_to_msa_income_percentage) and keep them only if the bank can justify them; retrain and re-audit.
- Re-run Fair Lens before every new model version goes live, and on a regular schedule to catch drift.
- Keep a person in the loop for denials near the approval threshold.

## Comparison with real past decisions

On 600 population applicants drawn from real records, the model agreed with the lender's actual decision 93.5% of the time.

| Neighborhood | Real approval | Model approval | Model denies past approvals |
| --- | --- | --- | --- |
| low minority | 94.9% | 92.9% | 4.4% |
| medium minority | 95.6% | 92.1% | 5.5% |
| high minority | 96.6% | 91.5% | 5.3% |

Past decisions may themselves be biased, so matching them is not proof of fairness.

## Limitations

- HMDA data has no credit scores, so group gaps show disparities worth review, not proof of discrimination.
- Not every feature correlated with race is unfair; people decide what is justified.
- Swapped applicants combine real values and may occasionally be unrealistic.
- Joint applicants and groups with missing demographics are excluded from comparisons.
