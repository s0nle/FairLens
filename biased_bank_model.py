"""
Simulated bank loan model that LEARNS bias from historical lending decisions,
plus an audit showing how the bias shows up.

Usage:
  python biased_bank_model.py                      # synthetic data (runs offline)
  python biased_bank_model.py --hmda lar.csv       # real public HMDA data

HMDA data: https://ffiec.cfpb.gov/data-browser/  (download a state/year CSV)
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------- data
def synthetic(n=30000, seed=0):
    """Historical decisions with built-in human bias: a direct penalty on
    group B and a 'redlining' penalty via a neighborhood proxy."""
    rng = np.random.default_rng(seed)
    b = rng.random(n) < 0.3
    neighborhood = np.where(b, rng.normal(.6, .2, n), rng.normal(.3, .2, n)).clip(0, 1)
    income = rng.lognormal(np.where(b, 10.8, 11.0), .5)
    credit = rng.normal(np.where(b, 670, 700), 50).clip(300, 850)
    dti = rng.normal(np.where(b, .36, .32), .08).clip(.05, .8)
    loan = income * rng.uniform(.5, 3, n)
    merit = (credit - 650) / 50 + (income / 1e5 - .6) - 6 * (dti - .35)
    repay = (merit + rng.normal(0, 1, n) > 0).astype(int)          # ground truth
    hist = merit - 0.8 * b - 1.5 * neighborhood + rng.normal(0, 1, n)
    df = pd.DataFrame(dict(income=income, credit=credit, dti=dti, loan=loan,
                           neighborhood_risk=neighborhood, repay=repay,
                           approved=(hist > 0).astype(int),
                           group=np.where(b, "B", "A")))
    return df, ["income", "credit", "dti", "loan", "neighborhood_risk"], "A"


def load_hmda(path, max_rows=200_000):
    cols = ["action_taken", "derived_race", "income", "loan_amount",
            "property_value", "debt_to_income_ratio", "loan_to_value_ratio",
            "tract_minority_population_percent", "tract_to_msa_income_percentage"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df = df[df.action_taken.isin([1, 3])]                 # originated vs denied
    df["approved"] = (df.action_taken == 1).astype(int)
    df["dti"] = pd.to_numeric(df.debt_to_income_ratio.astype(str)
                              .str.extract(r"(\d+)")[0], errors="coerce")
    for c in ["income", "loan_amount", "property_value", "loan_to_value_ratio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["group"] = df.derived_race
    df = df[df.group.isin(["White", "Black or African American", "Asian"])]
    feats = ["income", "loan_amount", "property_value", "dti", "loan_to_value_ratio",
             "tract_minority_population_percent", "tract_to_msa_income_percentage"]
    df = df.dropna(subset=feats + ["approved"])
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=0)
    return df, feats, "White"


# --------------------------------------------------------------- audit
def audit(label, test, pred, ref):
    t = test.assign(pred=pred)
    rates = t.groupby("group").pred.mean()
    print(f"\n== {label} ==")
    for g, r in rates.items():
        extra = ""
        if "repay" in t:  # equal opportunity: approval rate among truly creditworthy
            tpr = t[(t.group == g) & (t.repay == 1)].pred.mean()
            extra = f"  | approval rate among creditworthy: {tpr:.1%}"
        print(f"  {g:28s} approval {r:.1%}{extra}")
    for g, r in rates.items():
        if g != ref:
            dir_ = r / rates[ref]
            flag = "  <-- below 0.80 'four-fifths' threshold" if dir_ < .8 else ""
            print(f"  disparate impact ({g} / {ref}): {dir_:.2f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmda")
    a = ap.parse_args()
    df, feats, ref = load_hmda(a.hmda) if a.hmda else synthetic()
    print(f"{len(df):,} applications | groups: {df.group.value_counts().to_dict()}")

    train, test = train_test_split(df, test_size=.3, random_state=0, stratify=df.group)
    groups = sorted(df.group.unique())

    def X(d, with_group):
        x = d[feats].copy()
        if with_group:
            for g in groups:
                if g != ref:
                    x[f"is_{g}"] = (d.group == g).astype(int)
        return x

    # Model 1: sees the protected attribute. Model 2: "fairness through unawareness".
    for name, wg in [("Model WITH group attribute", True),
                     ("Model WITHOUT group attribute (proxies remain)", False)]:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        m.fit(X(train, wg), train.approved)
        audit(name, test, m.predict(X(test, wg)), ref)

    # Proxy test: can the "blind" features predict group membership?
    gm = HistGradientBoostingClassifier().fit(X(train, False), train.group != ref)
    auc = roc_auc_score(test.group != ref, gm.predict_proba(X(test, False))[:, 1])
    print(f"\nProxy check: group predictable from 'neutral' features, AUC = {auc:.2f} "
          f"(0.5 = no leakage, 1.0 = total)")

    # Baseline: what the historical decisions themselves looked like
    audit("Historical decisions (the training labels)", test, test.approved, ref)


if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0]]
    main()
