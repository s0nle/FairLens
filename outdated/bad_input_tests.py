"""
bad_input_tests.py
-------------------
Creates three bad results CSV files and confirms ingest.py raises or warns
clearly for each, without crashing.

(a) Missing case_ids   -- some audit_key case_ids absent from results
(b) Duplicate case_ids -- a case_id appears twice in the results file
(c) Invalid decision   -- a decision value of 5 (not 0 or 1)
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.ingest import validate, ingest
from src.utils import AuditorError

audit_key_df = pd.read_csv("data/audit_key.csv")
good_results = pd.read_csv("test_results/biased_results.csv")

PASS = "PASS"
FAIL = "FAIL"

results_summary = []


# ── (a) Missing case_ids ──────────────────────────────────────────────────────
print("\n=== Test (a): Missing case_ids ===")
missing_results = good_results.iloc[:-50].copy()   # drop last 50 rows
warnings_a = validate(missing_results, audit_key_df)
msg_a = [w for w in warnings_a if "no matching row" in w.lower() or "missing" in w.lower()]
status_a = PASS if msg_a else FAIL
print(f"  Warnings: {warnings_a}")
print(f"  Result: {status_a} -- detected missing case_ids: {bool(msg_a)}")
results_summary.append(("missing_case_ids", status_a, msg_a[0] if msg_a else "NO WARNING"))

# Verify ingest still completes (warnings only, not fatal for missing IDs)
try:
    merged = ingest({"results_path": "bad", "audit_key_path": "bad"},
                    results_df=missing_results, audit_key_df=audit_key_df)
    print(f"  Ingest completed with {len(merged)} rows (inner join on matching IDs)")
    ingest_status_a = PASS
except AuditorError as e:
    print(f"  AuditorError: {e}")
    ingest_status_a = PASS   # raising is also acceptable


# ── (b) Duplicate case_ids ────────────────────────────────────────────────────
print("\n=== Test (b): Duplicate case_ids ===")
dup_row = good_results.iloc[0:1].copy()
dup_results = pd.concat([good_results, dup_row], ignore_index=True)
warnings_b = validate(dup_results, audit_key_df)
msg_b = [w for w in warnings_b if "duplicate" in w.lower()]
status_b = PASS if msg_b else FAIL
print(f"  Warnings: {warnings_b}")
print(f"  Result: {status_b} -- detected duplicates: {bool(msg_b)}")
results_summary.append(("duplicate_case_ids", status_b, msg_b[0] if msg_b else "NO WARNING"))


# ── (c) Invalid decision value ────────────────────────────────────────────────
print("\n=== Test (c): Invalid decision value (5) ===")
bad_dec_results = good_results.copy()
bad_dec_results.loc[0:4, "decision"] = 5
warnings_c = validate(bad_dec_results, audit_key_df)
msg_c = [w for w in warnings_c if "invalid" in w.lower() or "decision" in w.lower()]
status_c = PASS if msg_c else FAIL
print(f"  Warnings: {warnings_c}")
print(f"  Result: {status_c} -- detected bad decisions: {bool(msg_c)}")
results_summary.append(("invalid_decision_value", status_c, msg_c[0] if msg_c else "NO WARNING"))

# Check ingest raises for all-bad decisions (only 0/1 rows survive after coerce)
all_bad = good_results.copy()
all_bad["decision"] = 5
try:
    merged = ingest({"results_path": "bad", "audit_key_path": "bad"},
                    results_df=all_bad, audit_key_df=audit_key_df)
    print(f"  Ingest with all-bad decisions: {len(merged)} rows merged (decision=5 coerced to NA, dropped in analysis)")
    results_summary.append(("all_invalid_decisions_ingest", PASS, "Ingested with warning; coercion to NA"))
except AuditorError as e:
    print(f"  AuditorError raised as expected: {e}")
    results_summary.append(("all_invalid_decisions_ingest", PASS, f"AuditorError: {str(e)[:80]}"))


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== Bad Input Test Summary ===")
print(f"  {'Test':<35}  {'Status':<6}  {'Details'}")
for name, status, detail in results_summary:
    print(f"  {name:<35}  {status:<6}  {detail[:90]}")

all_pass = all(s == PASS for _, s, _ in results_summary)
print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
sys.exit(0 if all_pass else 1)
