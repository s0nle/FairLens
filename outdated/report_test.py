"""
report_test.py
--------------
Verifies that:
1. report.md and report.html exist after `main.py audit`.
2. The template fallback is used when no watsonx credentials are set.
3. Every key number in report.md matches the number in audit_results.json.
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = "PASS"
FAIL = "FAIL"
results = []


# ── Run the biased audit to generate fresh outputs ───────────────────────────
import subprocess
env = dict(os.environ)
env.pop("WATSONX_API_KEY", None)
env.pop("WATSONX_PROJECT_ID", None)
env.pop("WATSONX_URL", None)

proc = subprocess.run(
    [sys.executable, "main.py", "audit", "--results", "test_results/biased_results.csv"],
    capture_output=True, text=True, env=env
)
print(proc.stdout)
if proc.returncode not in (0, 1):   # exit 1 is acceptable (only stderr INFO)
    print("  stderr:", proc.stderr[:300])


# ── 1. Files exist ────────────────────────────────────────────────────────────
md_path   = Path("output/report.md")
html_path = Path("output/report.html")
json_path = Path("output/audit_results.json")

for path in (md_path, html_path, json_path):
    exists = path.exists()
    results.append((f"{path.name} exists", PASS if exists else FAIL))
    print(f"  {path.name}: {'EXISTS' if exists else 'MISSING'}")


# ── 2. Template fallback was used (no LLM credentials) ───────────────────────
md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
uses_template = "(template mode)" in md_text or "template mode" in md_text
results.append(("template fallback used", PASS if uses_template else FAIL))
print(f"\n  Template fallback: {'YES' if uses_template else 'NO (LLM was called?)'}")


# ── 3. HTML contains body ─────────────────────────────────────────────────────
html_text = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
has_html  = "<html" in html_text and "<body" in html_text
results.append(("report.html is valid HTML", PASS if has_html else FAIL))
print(f"  report.html valid HTML structure: {'YES' if has_html else 'NO'}")


# ── 4. Number consistency: check key numbers from JSON appear in report ───────
if json_path.exists():
    with json_path.open() as f:
        audit = json.load(f)

    n_total     = audit.get("n_total")
    n_pop       = audit.get("n_population")
    n_matched   = audit.get("n_matched")

    # Numbers that must appear in the report
    checks = {
        "n_total in report":   str(n_total) in md_text,
        "n_pop in report":     str(n_pop)   in md_text,
        "n_matched in report": str(n_matched) in md_text,
    }

    # DI ratios from gaps
    gaps = audit.get("group_fairness", {}).get("gaps", {})
    for dim, g in gaps.items():
        di_val = g.get("di_ratio")
        if di_val is None:
            continue
        # The report formats as 2-decimal: look for e.g. "0.60"
        di_str = f"{di_val:.2f}"
        checks[f"DI ratio {dim} ({di_str}) in report"] = di_str in md_text

    for label, ok in checks.items():
        results.append((label, PASS if ok else FAIL))
        print(f"  {'OK' if ok else 'MISS'}: {label}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Report Test Summary ===")
for name, status in results:
    print(f"  {status:<6}  {name}")

all_pass = all(s == PASS for _, s in results)
print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
sys.exit(0 if all_pass else 1)
