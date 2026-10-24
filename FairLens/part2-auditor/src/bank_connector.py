"""
bank_connector.py: score applicants locally (used by `python main.py score`).

For the demo, the bank model runs separately in Colab (Part 1) and you move the
files by hand. This module is a shortcut for testing on one computer:
  - if models/bank_model.joblib exists (the file Part 1 saves), it uses that model
  - otherwise it falls back to the TEST-ONLY fake biased model
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.utils import (DEFAULT_RESULTS_FILE, AuditorError, read_table, resolve_path,
                       to_numeric, wants_latest)

logger = logging.getLogger("auditor.bank_connector")


def _predict_with_part1(bundle, apps: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Use Part 1's saved model. Part 1 saves a dict with model, features, threshold."""
    if isinstance(bundle, dict) and "model" in bundle:
        model, features = bundle["model"], bundle.get("features") or cfg["model_features"]
    else:                                            # a bare sklearn model
        model = bundle
        flist = resolve_path(cfg.get("feature_list_path", "models/feature_list.json"))
        features = json.loads(flist.read_text()) if flist.exists() else cfg["model_features"]
    missing = [f for f in features if f not in apps.columns]
    if missing:
        raise AuditorError("Applications file is missing model features: " + ", ".join(missing))
    X = pd.DataFrame({f: to_numeric(apps[f]) for f in features})
    return model.predict_proba(X)[:, 1]


def score(cfg: dict) -> pd.DataFrame:
    """Score applications_for_bank.csv and write bank_results.csv."""
    apps_path = resolve_path(cfg["applications_path"], must_exist=True)
    if not apps_path.exists():
        raise AuditorError(f"{apps_path} not found. Run:  python main.py generate")
    apps = read_table(apps_path, dtype={"case_id": str})

    model_path = resolve_path(cfg.get("model_path", "models/bank_model.joblib"))
    if model_path.exists():
        import joblib
        bundle = joblib.load(model_path)
        prob = _predict_with_part1(bundle, apps, cfg)
        threshold = bundle.get("threshold", 0.5) if isinstance(bundle, dict) else 0.5
        results = pd.DataFrame({"case_id": apps["case_id"],
                                "decision": (prob >= threshold).astype(int),
                                "probability": np.round(prob, 4)})
        print(f"  Scored with Part 1's model ({model_path.name})")
    else:
        from src.fake_bank_model import fake_score
        print(f"  No Part 1 model at {model_path}; using the TEST-ONLY fake biased model.")
        results = fake_score(apps, "biased", int(cfg["random_seed"]),
                             float(cfg["high_minority_threshold"]))

    out = resolve_path(DEFAULT_RESULTS_FILE if wants_latest(cfg["results_path"])
                       else cfg["results_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"  {len(results):,} decisions, approval rate {results['decision'].mean():.1%} -> {out}")
    return results
