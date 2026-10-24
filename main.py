"""
main.py  —  Module 5 / CLI entry point
────────────────────────────────────────
Commands:

  python main.py generate
      Loads HMDA data, generates applications_for_bank.csv and audit_key.csv.

  python main.py score
      Loads Part 1's bank model (or fake fallback), scores applications,
      writes bank_results.csv.

  python main.py audit [--results PATH] [--with-hmda]
      Runs ingest → fairness analysis → report.
      --results PATH   path to bank_results.csv (default: config value)
      --with-hmda      also load real HMDA data for historical comparison

  python main.py run-all [--with-hmda]
      Runs generate → score → audit in sequence.

Each command logs clearly, surfaces errors with friendly messages, and
never exits silently on failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Logging setup (must happen before any src imports) ────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _load_cfg(config_path: str | None = None) -> dict:
    """Load config, printing a friendly error on failure."""
    from src.utils import load_config, AuditorError
    try:
        return load_config(config_path)
    except AuditorError as exc:
        print(f"\n✗ Configuration error:\n  {exc}", file=sys.stderr)
        sys.exit(1)


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> None:
    """Run the applicant generator (Module 1)."""
    from src.utils import AuditorError
    from src.data_prep import load_hmda
    from src.generator import generate

    cfg = _load_cfg(args.config)

    print("\n─── STEP 1: Load HMDA data ───────────────────────────────────────────")
    try:
        hmda_df = load_hmda(cfg)
    except AuditorError as exc:
        print(f"\n✗ Failed to load HMDA data:\n  {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n─── STEP 2: Generate test applicants ────────────────────────────────")
    try:
        generate(cfg, hmda_df=hmda_df)
    except AuditorError as exc:
        print(f"\n✗ Generation failed:\n  {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_score(args: argparse.Namespace) -> None:
    """Score applicants with the bank model (bank_connector / fake fallback)."""
    from src.utils import AuditorError
    from src.bank_connector import score

    cfg = _load_cfg(args.config)

    print("\n─── Scoring applicants ──────────────────────────────────────────────")
    try:
        score(cfg)
    except AuditorError as exc:
        print(f"\n✗ Scoring failed:\n  {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_audit(args: argparse.Namespace) -> None:
    """Run ingest → fairness analysis → report (Modules 2–4)."""
    import pandas as pd
    from src.utils import AuditorError
    from src.ingest import ingest
    from src.audit import run_full_audit
    from src.report import generate_report

    cfg = _load_cfg(args.config)

    # Override results path if provided via --results
    if args.results:
        cfg = dict(cfg)
        cfg["results_path"] = args.results

    print("\n─── STEP 1: Ingest bank results ─────────────────────────────────────")
    try:
        audit_df = ingest(cfg)
    except AuditorError as exc:
        print(f"\n✗ Ingest failed:\n  {exc}", file=sys.stderr)
        sys.exit(1)

    # Optional: load real HMDA for historical comparison
    hmda_df = None
    if getattr(args, "with_hmda", False):
        print("\n─── Loading HMDA data for historical comparison ─────────────────")
        try:
            from src.data_prep import load_hmda
            hmda_df = load_hmda(cfg)
        except AuditorError as exc:
            logger.warning("Could not load HMDA data for historical comparison: %s", exc)
            print(f"  ⚠  Skipping historical comparison: {exc}")

    print("\n─── STEP 2: Run fairness audit ──────────────────────────────────────")
    try:
        audit_results = run_full_audit(audit_df, cfg, hmda_df=hmda_df)
    except AuditorError as exc:
        print(f"\n✗ Audit failed:\n  {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n─── STEP 3: Generate report ─────────────────────────────────────────")
    try:
        generate_report(audit_results, cfg)
    except AuditorError as exc:
        print(f"\n✗ Report generation failed:\n  {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n✓ Audit complete.\n")


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run generate → score → audit in sequence."""
    cmd_generate(args)
    cmd_score(args)
    cmd_audit(args)


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="AI Quality Assurance Auditor — fairness testing for loan-approval models",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config.yaml (default: config.yaml in project root)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    sub.add_parser(
        "generate",
        help="Generate synthetic test applicants from HMDA data",
    )

    # score
    sub.add_parser(
        "score",
        help="Score applications with the bank model (or fake fallback)",
    )

    # audit
    audit_p = sub.add_parser(
        "audit",
        help="Run the full fairness audit on bank results",
    )
    audit_p.add_argument(
        "--results",
        default=None,
        metavar="PATH",
        help="Path to bank_results.csv (overrides config.yaml)",
    )
    audit_p.add_argument(
        "--with-hmda",
        action="store_true",
        dest="with_hmda",
        help="Load real HMDA data to include historical comparison in report",
    )

    # run-all
    run_all_p = sub.add_parser(
        "run-all",
        help="Run generate → score → audit in one command",
    )
    run_all_p.add_argument(
        "--with-hmda",
        action="store_true",
        dest="with_hmda",
        help="Include historical comparison in the audit report",
    )

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser  = _build_parser()
    args    = parser.parse_args()

    dispatch = {
        "generate": cmd_generate,
        "score":    cmd_score,
        "audit":    cmd_audit,
        "run-all":  cmd_run_all,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
