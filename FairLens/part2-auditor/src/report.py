"""
report.py: Module 4. The audit report (report.md + report.html).

If watsonx credentials are set (WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL),
IBM Granite writes the plain-English report from the audit numbers. Otherwise, or if
the call fails, a template writes it ("template mode") so the demo never breaks.
Either way, the exact numbers are included as tables so every claim can be checked.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from src.utils import AuditorError, resolve_path, to_jsonable, write_json

logger = logging.getLogger("auditor.report")

VERDICT_TEXT = {"fair": "FAIR", "needs_review": "NEEDS REVIEW", "unfair": "UNFAIR"}


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _num(x, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


# ── Number tables (used in both modes) ────────────────────────────────────────

def _tables(r: dict) -> list[str]:
    L = ["### Group fairness (population set)", "",
         "| Dimension | Best group | Worst group | Gap | Ratio | Flagged |",
         "| --- | --- | --- | --- | --- | --- |"]
    for g in r["group_fairness"]["gaps"].values():
        L.append(f"| {g['label']} | {g['best_group']} ({_pct(g['best_rate'])}) | "
                 f"{g['worst_group']} ({_pct(g['worst_rate'])}) | {g['gap'] * 100:.1f} pts | "
                 f"{_num(g['di_ratio'])} | {'YES' if g['flagged'] else 'no'} |")
    L += ["", "Ratio = worst group's approval rate / best group's. Below 0.80 is flagged "
          "(the four-fifths rule).", "",
          "### Swap tests (matched applicants, one change at a time)", "",
          "| Change | Pairs | Flipped | Flip rate | Approval chance change (same applicant) | Severity |",
          "| --- | --- | --- | --- | --- | --- |"]
    for s in r["swap_tests"]["summary"]:
        d = s.get("avg_shift_when_inputs_changed")
        if d is None:
            change = "0 pts (bank saw no difference)" if not s.get("bank_inputs_changed") else "n/a"
        else:
            p = s.get("shift_p_value")
            sig = "" if p is None else (", p < 0.001" if p < 0.001 else f", p = {p:.3f}")
            change = f"{d * 100:+.1f} pts{sig}"
        L.append(f"| {s['label']} | {s['n_pairs']} | {s['n_flipped']} | {_pct(s['flip_rate'])} | "
                 f"{change} | {s['severity']} |")
    L += ["", "Severity uses whichever is stronger: how many decisions flipped, or how much the "
          "same applicant's approval chance moved (counted only when statistically clear)."]
    cons = r.get("consistency") or {}
    if cons.get("n_identical_pairs"):
        L += ["", f"Consistency: the bank model saw {cons['n_identical_pairs']} pairs of identical "
              f"applications (only race, ethnicity, sex, or age differed, which it never sees) and "
              f"gave different decisions {cons['n_inconsistent']} times "
              f"({_pct(cons['inconsistency_rate'])})."]
    race = r["proxy_rankings"].get("race", [])[:5]
    if race:
        L += ["", "### Proxy detection: model inputs that predict race", "",
              "| Model input | AUC | Flagged |", "| --- | --- | --- |"]
        L += [f"| {p['feature']} | {p['auc']:.2f} | {'YES' if p['flagged'] else 'no'} |" for p in race]
        L += ["", "AUC 0.5 = no information about race, 1.0 = a perfect stand-in. "
              f"Above {r['thresholds']['proxy_auc']:.2f} is flagged."]
    return L


def _examples(r: dict, n: int = 5) -> list[str]:
    ex = r["swap_tests"]["examples"][:n]
    if not ex:
        return ["No decisions flipped in the matched tests."]
    word = {1: "approved", 0: "denied"}
    out = []
    for e in ex:
        change = (f"{e['changed_column']} {e['from']} -> {e['to']}" if e.get("changed_column")
                  else e["label"])
        out.append(f"- **{e['pair_group_id']}** ({e['label']}): {change}. Decision went from "
                   f"{word[e['base_decision']]} to {word[e['variant_decision']]} "
                   f"(probability {_num(e['base_probability'], 3)} -> "
                   f"{_num(e['variant_probability'], 3)}); every other input was identical.")
    return out


# ── Template report ───────────────────────────────────────────────────────────

def _causes_and_fixes(r: dict) -> tuple[list[str], list[str]]:
    f = r["factors"]
    flagged_px = [p["feature"] for p in r["proxy_rankings"].get("race", []) if p["flagged"]]
    causes, fixes = [], []
    if f["neighborhood"]["rating"] != "none":
        causes.append("The model gives weight to neighborhood features. In Los Angeles County, "
                      "a tract's minority share closely tracks applicants' race, so location "
                      "acts as a stand-in for race even though race is never an input.")
        fixes.append("Remove tract_minority_population_percent from the model inputs; it has "
                     "no legitimate role in judging repayment.")
        fixes.append("Review other neighborhood inputs (for example tract_to_msa_income_percentage) "
                     "and keep them only if the bank can justify them; retrain and re-audit.")
    if flagged_px and f["neighborhood"]["rating"] == "none":
        causes.append("Some inputs predict race strongly (" + ", ".join(flagged_px[:3]) + "), "
                      "although the swap tests did not show them changing decisions.")
    for name in ("race", "ethnicity", "sex", "age"):
        if f[name]["swap_severity"] != "none":
            causes.append(f"Decisions changed when only {name} changed, so the model is using "
                          f"{name} information directly or through an input.")
            fixes.append(f"Find and remove the input carrying {name} information, then re-audit.")
    if any(f[n]["group_gap_severity"] != "none" and f[n]["swap_severity"] == "none" for n in f):
        causes.append("Some group gaps appear without matching decision flips. These can come "
                      "from legitimate differences such as income; review before acting.")
    if not causes:
        causes.append("No bias signal was found in the tested factors.")
    fixes += ["Re-run Fair Lens before every new model version goes live, and on a regular "
              "schedule to catch drift.",
              "Keep a person in the loop for denials near the approval threshold."]
    return causes, fixes


def _template_report(r: dict) -> str:
    causes, fixes = _causes_and_fixes(r)
    L = ["# Fair Lens Audit Report", "",
         f"_Generated {r['generated_at']} in template mode (no AI credentials or the AI call failed)._",
         "", "## Overall verdict", "",
         f"**{VERDICT_TEXT[r['verdict']]}**: {r['verdict_reason']}", "",
         "## What we tested", "",
         f"- Cases audited: {r['n_total']} ({r['n_population']} population applicants and "
         f"{r['n_matched']} matched applicants in {r['n_pair_groups']} groups)",
         f"- Overall approval rate: {_pct(r['overall_approval_rate'])}",
         f"- Bank results file: {Path(r['results_file']).name if r.get('results_file') else 'n/a'}",
         "- Model inputs checked: " + ", ".join(r["model_features"]),
         "- Tests: group fairness, swap tests (one change at a time), proxy detection", "",
         "## Bias summary by factor", "",
         "| Factor | Rating | Evidence |", "| --- | --- | --- |"]
    for name, f in r["factors"].items():
        L.append(f"| {name.title()} | {f['rating'].upper()} | "
                 f"{'; '.join(f['reasons']) or 'No gap or flips found'} |")
    L += ["", "## Key findings", ""] + _tables(r)
    L += ["", "## Example cases", ""] + _examples(r)
    L += ["", "## Likely causes", ""] + [f"- {c}" for c in causes]
    L += ["", "## Recommended fixes", ""] + [f"- {x}" for x in fixes]
    L += _historical(r)
    L += ["", "## Limitations", ""] + [f"- {x}" for x in r["limitations"]]
    return "\n".join(L) + "\n"


def _historical(r: dict) -> list[str]:
    h = (r.get("historical") or {}).get("sample")
    if not h:
        return []
    L = ["", "## Comparison with real past decisions", "",
         f"On {h['n']} population applicants drawn from real records, the model agreed with the "
         f"lender's actual decision {_pct(h['agreement'])} of the time.", "",
         "| Neighborhood | Real approval | Model approval | Model denies past approvals |",
         "| --- | --- | --- | --- |"]
    for row in h["by_tract_level"]:
        L.append(f"| {row['level']} minority | {_pct(row['historical_approval'])} | "
                 f"{_pct(row['model_approval'])} | {_pct(row['false_denial_rate'])} |")
    L += ["", "Past decisions may themselves be biased, so matching them is not proof of fairness."]
    return L


# ── AI report (IBM Granite on watsonx.ai) ─────────────────────────────────────

def _llm_summary(r: dict) -> dict:
    return to_jsonable({
        "verdict": r["verdict"], "verdict_reason": r["verdict_reason"],
        "counts": {k: r[k] for k in ("n_total", "n_population", "n_matched", "n_pair_groups")},
        "overall_approval_rate": r["overall_approval_rate"],
        "group_gaps": r["group_fairness"]["gaps"],
        "swap_tests": r["swap_tests"]["summary"],
        "top_race_proxies": r["proxy_rankings"].get("race", [])[:5],
        "example_flipped_cases": r["swap_tests"]["examples"][:5],
        "factors": r["factors"],
        "historical": r.get("historical"),
        "limitations": r["limitations"],
    })


PROMPT = """You are writing a fairness audit report for a bank's leadership team about its
loan-approval AI model. Use ONLY the facts and numbers in the JSON below. Do not invent
numbers, groups, or findings. Write plain English for non-technical readers.

Use these Markdown sections in order:
## Overall verdict
## What we tested
## Key findings
## Example cases
## Likely causes
## Recommended fixes
## Limitations

Keep it under 600 words. Present percentages as percentages.

AUDIT RESULTS JSON:
{data}
"""


def _call_granite(r: dict, cfg: dict) -> str | None:
    key, project = os.getenv("WATSONX_API_KEY"), os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    if not (key and project):
        return None
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        llm = cfg.get("llm", {})
        model = ModelInference(
            model_id=llm.get("model_id", "ibm/granite-3-3-8b-instruct"),
            credentials=Credentials(url=url, api_key=key), project_id=project,
            params={"max_new_tokens": int(llm.get("max_tokens", 1500)),
                    "temperature": float(llm.get("temperature", 0.2))})
        text = model.generate_text(prompt=PROMPT.format(data=json.dumps(_llm_summary(r), indent=1)))
        text = (text or "").strip()
        return text if len(text) > 200 else None
    except Exception as exc:                          # never let the demo break
        logger.warning("Granite call failed, using template instead: %s", exc)
        print(f"  AI report unavailable ({type(exc).__name__}); using template mode.")
        return None


def _ai_report(r: dict, body: str, model_id: str) -> str:
    L = ["# Fair Lens Audit Report", "",
         f"_Generated {r['generated_at']}. Written by IBM Granite ({model_id}) from the audit "
         "numbers below._", "", body.strip(), "", "---", "",
         "## Audit numbers (source: audit_results.json)", "",
         f"- Cases audited: {r['n_total']} ({r['n_population']} population, "
         f"{r['n_matched']} matched in {r['n_pair_groups']} groups)", ""]
    L += _tables(r) + _historical(r)
    return "\n".join(L) + "\n"


# ── HTML ──────────────────────────────────────────────────────────────────────

def _inline(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def markdown_to_html(md: str, title: str = "Fair Lens Audit Report") -> str:
    """Small Markdown converter (headings, lists, tables, bold) with no extra installs."""
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= {"-", " ", ":"} for c in cells):
                    rows.append(cells)
                i += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
                       + "</tr></thead><tbody>" + "".join(
                           "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                           for r in rows[1:]) + "</tbody></table>")
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        m = re.match(r"(#{1,4}) (.*)", line)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{_inline(m.group(2))}</h{n}>")
        elif line.strip() == "---":
            out.append("<hr>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    css = ("body{font-family:'IBM Plex Sans',system-ui,sans-serif;background:#F6F5F0;color:#12151C;"
           "margin:0}main{max-width:900px;margin:0 auto;padding:32px 20px 64px}"
           "h1{background:#A9C5A5;padding:16px 20px;border-radius:10px;font-size:26px}"
           "h2{margin-top:32px;border-bottom:1px solid #E2E4E9;padding-bottom:6px}"
           "table{border-collapse:collapse;width:100%;margin:12px 0;background:#FDFCF9;font-size:14px}"
           "th,td{border:1px solid #E2E4E9;padding:8px 10px;text-align:left}th{background:#EEF0EA}"
           "p,li{line-height:1.6}code{font-family:'IBM Plex Mono',monospace}"
           "@media(max-width:600px){table{display:block;overflow-x:auto}}")
    return (f"<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{html.escape(title)}</title>\n<style>{css}</style>\n</head>\n"
            f"<body>\n<main>\n" + "\n".join(out) + "\n</main>\n</body>\n</html>\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_report(results: dict, cfg: dict) -> dict:
    """Write report.md and report.html to the output folder."""
    if not results or "group_fairness" not in results:
        raise AuditorError("No audit results to report on. Run the audit first.")
    body = _call_granite(results, cfg)
    mode = "ai" if body else "template"
    md = (_ai_report(results, body, cfg.get("llm", {}).get("model_id", "granite"))
          if body else _template_report(results))

    out_dir = resolve_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    page = markdown_to_html(md)

    # Latest report stays at output/report.* ; every run also gets its own dated folder.
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "report.html").write_text(page, encoding="utf-8")

    run_dir = out_dir / "runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    (run_dir / "report.html").write_text(page, encoding="utf-8")
    write_json(results, run_dir / "audit_results.json")
    used = results.get("results_file")
    if used and Path(used).exists():
        shutil.copy2(used, run_dir / "bank_results_used.csv")

    print(f"  Report written ({'IBM Granite' if mode == 'ai' else 'template mode'})")
    print(f"    latest:   {out_dir / 'report.html'}")
    print(f"    this run: {run_dir / 'report.html'}")
    return {"mode": mode, "markdown": md, "run_dir": str(run_dir)}
