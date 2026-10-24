"""
Fair Lens — Part 1: Simulated Bank Loan Model
==============================================

A stand-in for a real bank's loan-approval AI. It learns from historical
2018 Los Angeles County HMDA mortgage decisions, then scores new applicants
that Part 2 (the Fair Lens auditor) generates.

The ONLY connection to Part 2 is two CSV files, moved by hand:

    Part 2  ──>  applications_for_bank.csv  ──>  Part 1 (this file)
    Part 1  ──>  bank_results.csv           ──>  Part 2

bank_results.csv always has exactly these columns, one row per case_id:
    case_id, decision (1 = approve, 0 = deny), probability (0–1)

The model never sees race, ethnicity, sex, or age. It only uses the seven
features below. Two of them describe the applicant's neighborhood, which is
how bias from historical lending can sneak in — that's what the auditor
is designed to catch.

Run in Google Colab (easiest):
    1. Paste this whole file into a cell and run it.
    2. In a new cell run:  launch_ui()
    Or open Fair_Lens_Part1_Bank_Model.ipynb, which is set up already.

Run on a computer:
    python bank_model.py train --hmda hmda_la_2018.csv
    python bank_model.py score --applications applications_for_bank.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ── Settings ──────────────────────────────────────────────────────────────────

# The model's inputs. Part 2's config.yaml `model_features` should list these.
FEATURES = [
    "income",
    "loan_amount",
    "property_value",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "tract_minority_population_percent",
    "tract_to_msa_income_percentage",
]

MODEL_PATH = "bank_model.joblib"
RESULTS_PATH = "bank_results.csv"
RANDOM_SEED = 42

APPROVE_THRESHOLD = 0.50      # approve when probability >= this

# Optional demo setting. 0.0 = the model only uses what it learned from history.
# Set to e.g. 0.15 to subtract 15 points of approval probability for applicants
# in high-minority tracts, guaranteeing a visible "redlining" bias for the demo.
EXTRA_REDLINING_PENALTY = 0.0
HIGH_MINORITY_PCT = 70.0

# Same HMDA filter Part 2 uses (personal home-purchase loans, primary residence).
HMDA_FILTERS = {
    "loan_purpose": 1,
    "business_or_commercial_purpose": 2,
    "open_end_line_of_credit": 2,
    "occupancy_type": 1,
}
APPROVED_ACTIONS = {1, 2}     # 1 = originated, 2 = approved but not accepted
DENIED_ACTIONS = {3}          # 3 = denied
MAX_TRAINING_ROWS = 200_000

# Columns that must never be used as model inputs (checked as a safety net).
FORBIDDEN_INPUTS = {
    "derived_race", "derived_ethnicity", "derived_sex", "applicant_age",
    "action_taken", "denial_reason_1", "interest_rate", "rate_spread",
}
assert not FORBIDDEN_INPUTS & set(FEATURES), "A protected or post-decision column is in FEATURES"


# ── Cleaning (shared by training AND scoring, so both see identical inputs) ───

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _text_to_number(text: str) -> float:
    """Turn an HMDA value into a number.

    "45000" -> 45000      "36" -> 36           "20%-<30%" -> 25
    "<20%"  -> 15         ">60%" -> 65         "Exempt" / "NA" -> NaN
    """
    s = str(text).strip()
    if s.lower() in {"", "na", "nan", "none", "exempt"}:
        return np.nan
    nums = [float(n) for n in _NUMBER.findall(s)]
    if not nums:
        return np.nan
    if len(nums) >= 2:                       # a range such as "30%-<36%"
        return (nums[0] + nums[1]) / 2
    if s.startswith("<"):
        return max(nums[0] - 5, 0.0)
    if s.startswith(">"):
        return nums[0] + 5
    return nums[0]


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numbers, handling HMDA ranges and text codes."""
    out = pd.to_numeric(series, errors="coerce").astype(float)
    needs_parsing = out.isna() & series.notna()
    if needs_parsing.any():
        raw = series[needs_parsing].astype(str)
        lookup = {value: _text_to_number(value) for value in raw.unique()}
        out[needs_parsing] = raw.map(lookup)
    return out


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Real HMDA uses hyphens ("open-end_line_of_credit"); Part 2 uses underscores."""
    df = df.copy()
    df.columns = [str(c).strip().replace("-", "_") for c in df.columns]
    return df


EXCEL_TYPES = {".xlsx", ".xlsm", ".xls"}
UPLOAD_TYPES = ".csv,.xlsx,.xlsm,.xls"


def read_table(path: str | Path, dtype: dict | None = None) -> pd.DataFrame:
    """Read a CSV or an Excel workbook (first sheet)."""
    if Path(path).suffix.lower() in EXCEL_TYPES:
        try:
            return pd.read_excel(path, dtype=dtype)
        except ImportError as exc:
            raise ValueError("Reading Excel files needs the openpyxl package "
                             "(already installed in Colab). Or save the file as CSV.") from exc
    for encoding in ("utf-8-sig", "latin-1"):          # Excel-saved CSVs are often latin-1
        try:
            return pd.read_csv(path, low_memory=False, dtype=dtype, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Couldn't read {Path(path).name} as CSV or Excel.")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model's features and convert them to numbers."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "The file is missing columns the bank model needs: "
            + ", ".join(missing)
            + ".\nCheck that Part 2's config.yaml `model_features` includes them."
        )
    return pd.DataFrame({c: to_numeric(df[c]) for c in FEATURES}, index=df.index)


# ── Training ──────────────────────────────────────────────────────────────────

def _as_list(paths) -> list:
    return [paths] if isinstance(paths, (str, Path)) else list(paths)


def load_hmda(paths) -> pd.DataFrame:
    """Load one or more HMDA files (CSV or Excel) and keep approved/denied
    home-purchase applications. Several files are stacked into one dataset."""
    frames = []
    for path in _as_list(paths):
        part = standardize_columns(read_table(path))
        print(f"Loaded {len(part):,} rows from {Path(path).name}")
        frames.append(part)
    if not frames:
        raise ValueError("No HMDA files were given.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    if len(frames) > 1:
        print(f"Combined {len(frames)} files: {len(df):,} rows")

    for col, value in HMDA_FILTERS.items():
        if col in df.columns:
            df = df[pd.to_numeric(df[col], errors="coerce") == value]
    if "action_taken" not in df.columns:
        raise ValueError("The HMDA file has no 'action_taken' column, so there are no decisions to learn from.")

    action = pd.to_numeric(df["action_taken"], errors="coerce")
    df = df[action.isin(APPROVED_ACTIONS | DENIED_ACTIONS)].copy()
    df["approved"] = action[df.index].isin(APPROVED_ACTIONS).astype(int)

    if len(df) > MAX_TRAINING_ROWS:
        df = df.sample(MAX_TRAINING_ROWS, random_state=RANDOM_SEED)
    if df["approved"].nunique() < 2:
        raise ValueError("After filtering, the data has only approvals or only denials; the model can't learn.")

    print(f"Kept {len(df):,} home-purchase applications "
          f"({df['approved'].sum():,} approved, {(1 - df['approved']).sum():,} denied)")
    return df


def _minority_level(pct: pd.Series) -> pd.Series:
    return pd.cut(pct, bins=[-np.inf, 30, HIGH_MINORITY_PCT, np.inf],
                  labels=["low (<30%)", "medium", f"high (>{HIGH_MINORITY_PCT:.0f}%)"])


def train(hmda_paths, model_path: str | Path = MODEL_PATH) -> dict:
    """Train the bank model on historical HMDA decisions and save it.

    hmda_paths can be one file or a list of files, e.g.
        train("hmda_2018.csv")
        train(["lenders_part1.xlsx", "lenders_part2.xlsx"])
    """
    df = load_hmda(hmda_paths)
    X = prepare_features(df)
    y = df["approved"]

    # Drop rows where most inputs are unknown; fill the rest during training.
    usable = X.notna().sum(axis=1) >= len(FEATURES) - 2
    X, y = X[usable], y[usable]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    model = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_SEED)),
    ])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= APPROVE_THRESHOLD).astype(int)
    metrics = {
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "auc": round(float(roc_auc_score(y_test, proba)), 4),
        "denial_recall": round(float(recall_score(y_test, pred, pos_label=0)), 4),
    }
    print("\nModel quality on held-out applications:")
    print(f"  accuracy {metrics['accuracy']:.1%} | AUC {metrics['auc']:.3f} | "
          f"catches {metrics['denial_recall']:.1%} of real denials")

    # What the model learned about neighborhoods (no demographics used here).
    level = _minority_level(X_test["tract_minority_population_percent"])
    summary = pd.DataFrame({"historical": y_test, "model": pred, "level": level}) \
        .groupby("level", observed=True)[["historical", "model"]].mean()
    print("\nApproval rate by neighborhood minority level (held-out data):")
    for lvl, row in summary.iterrows():
        print(f"  {str(lvl):<14} real decisions {row['historical']:.1%} | model {row['model']:.1%}")

    bundle = {
        "model": model,
        "features": FEATURES,
        "threshold": APPROVE_THRESHOLD,
        "metrics": metrics,
        "trained_on": ", ".join(Path(p).name for p in _as_list(hmda_paths)),
    }
    joblib.dump(bundle, model_path)
    print(f"\nSaved model to {model_path}")
    return bundle


# ── Scoring (Part 2's file in, results file out) ──────────────────────────────

def score(applications_path: str | Path,
          out_path: str | Path = RESULTS_PATH,
          model_path: str | Path = MODEL_PATH,
          penalty: float = EXTRA_REDLINING_PENALTY,
          threshold: float | None = None) -> pd.DataFrame:
    """Score Part 2's applications_for_bank.csv and write bank_results.csv."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run training first.")
    bundle = joblib.load(model_path)
    threshold = bundle["threshold"] if threshold is None else threshold

    # Read case_id as text so IDs like "00042" come back exactly as Part 2 wrote them.
    apps = standardize_columns(read_table(applications_path, dtype={"case_id": str}))
    if "case_id" not in apps.columns:
        raise ValueError("applications file has no 'case_id' column. Use the file Part 2 generated.")
    dupes = apps["case_id"].duplicated()
    if dupes.any():
        raise ValueError(f"{dupes.sum()} duplicate case_ids in the applications file, e.g. "
                         f"{apps.loc[dupes, 'case_id'].iloc[0]}")

    X = prepare_features(apps)
    n_filled = int(X.isna().sum().sum())
    proba = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]

    if penalty:
        in_high = (X["tract_minority_population_percent"] > HIGH_MINORITY_PCT).to_numpy()
        proba = np.where(in_high, proba - penalty, proba)
    proba = np.clip(proba, 0.0, 1.0)

    results = pd.DataFrame({
        "case_id": apps["case_id"],
        "decision": (proba >= threshold).astype(int),
        "probability": proba.round(4),
    })
    results.to_csv(out_path, index=False)

    # Hand-off check: every case_id in must come back out.
    assert len(results) == len(apps) and results["case_id"].is_unique
    print(f"Scored {len(apps):,} applications -> {len(results):,} results "
          f"(all case_ids returned)")
    print(f"  approved {results['decision'].sum():,} | denied {(1 - results['decision']).sum():,} "
          f"| approval rate {results['decision'].mean():.1%}")
    if n_filled:
        print(f"  filled {n_filled:,} missing values with training medians (no rows dropped)")
    if penalty:
        print(f"  demo redlining penalty applied: -{penalty:.2f} in tracts >{HIGH_MINORITY_PCT:.0f}% minority")
    print(f"Saved {out_path}  ->  upload this file to Fair Lens (Part 2)")
    return results


# ── Google Colab helpers ──────────────────────────────────────────────────────

def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def upload_file(prompt: str) -> str:
    """In Colab, open the upload dialog and return the uploaded file's name."""
    from google.colab import files
    print(prompt)
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No file was uploaded.")
    return next(iter(uploaded))


def upload_files(prompt: str = "Choose one or more files") -> list[str]:
    """In Colab, upload one or more files at once (Ctrl/Cmd-click to pick several)."""
    from google.colab import files
    print(prompt)
    uploaded = files.upload()
    names = list(uploaded)
    print(f"Uploaded {len(names)} file(s): {', '.join(names) if names else 'none'}")
    return names


def download_file(path: str | Path) -> None:
    """In Colab, download a file to your computer."""
    from google.colab import files
    files.download(str(path))


# ── Simple app for Google Colab ───────────────────────────────────────────────

DATA_TYPES = EXCEL_TYPES | {".csv"}


def _data_files() -> list[str]:
    """CSV/Excel files in the current folder (Colab's Files panel), minus our output."""
    return sorted(p.name for p in Path(".").iterdir()
                  if p.is_file() and p.suffix.lower() in DATA_TYPES
                  and p.name != Path(RESULTS_PATH).name)


def _model_files() -> list[str]:
    return sorted(p.name for p in Path(".").iterdir() if p.is_file() and p.suffix == ".joblib")


def _download_link(path: str | Path, label: str) -> str:
    """An HTML link that downloads a small file straight from the notebook output."""
    import base64
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return (f'<a download="{Path(path).name}" href="data:text/csv;base64,{data}" '
            f'style="display:inline-block;padding:8px 16px;background:#2F5D3A;color:#fff;'
            f'border-radius:6px;text-decoration:none;font-weight:600">{label}</a>')


def launch_ui(model_path: str | Path = MODEL_PATH) -> None:
    """Show the Fair Lens bank model app in a Colab/Jupyter notebook."""
    import ipywidgets as w
    from IPython.display import HTML, display

    try:                                                          # lets Colab render widgets
        from google.colab import output
        output.enable_custom_widget_manager()
    except Exception:
        pass

    import shutil

    state: dict = {"apps": None, "results": None}

    def model_status() -> str:
        if Path(model_path).exists():
            b = joblib.load(model_path)
            m = b.get("metrics", {})
            return (f"<b style='color:#1F3D27'>Model ready</b> · trained on {b.get('trained_on', '?')}"
                    f" · accuracy {m.get('accuracy', 0):.0%} · catches {m.get('denial_recall', 0):.0%} of denials")
        return "<b style='color:#8A3508'>No model yet.</b> Train one or load a saved bank_model.joblib."

    wide = w.Layout(width="560px")
    label_style = {"description_width": "140px"}

    # Files already in Colab (dragged into the Files panel or uploaded with a cell)
    refresh_btn = w.Button(description="Refresh file list", icon="refresh")
    files_note = w.HTML()

    # Step 1: model
    status = w.HTML(model_status())
    train_files = w.SelectMultiple(options=[], rows=6, description="Training files",
                                   style=label_style, layout=wide)
    extra_paths = w.Text(description="More paths", style=label_style, layout=wide,
                         placeholder="optional, comma-separated, e.g. /content/drive/MyDrive/hmda.csv")
    train_btn = w.Button(description="Train model", button_style="success", icon="cogs")
    saved_model = w.Dropdown(options=[], description="Saved model", style=label_style,
                             layout=w.Layout(width="400px"))
    load_btn = w.Button(description="Use this model", icon="upload")
    log = w.Output(layout=w.Layout(max_height="220px", overflow="auto"))

    # Step 2: score
    apps_file = w.Dropdown(options=[], description="Applications file", style=label_style,
                           layout=wide)
    penalty = w.FloatSlider(value=EXTRA_REDLINING_PENALTY, min=0, max=0.3, step=0.05,
                            description="Demo bias", readout_format=".2f",
                            style={"description_width": "initial"})
    score_btn = w.Button(description="Approve / deny applicants", button_style="success",
                         icon="check", layout=w.Layout(width="240px"))

    # Results
    show = w.ToggleButtons(options=["All", "Approved", "Denied"], value="All")
    results_out = w.Output()

    def render() -> None:
        results_out.clear_output()
        if state["results"] is None:
            return
        res, apps = state["results"], state["apps"]
        table = res.merge(apps[["case_id"] + FEATURES], on="case_id", how="left")
        table.insert(1, "result", np.where(table["decision"] == 1, "Approved", "Denied"))
        if show.value != "All":
            table = table[table["result"] == show.value]
        n, n_ok = len(res), int(res["decision"].sum())
        with results_out:
            display(HTML(
                f"<div style='font-size:15px;margin:8px 0'>"
                f"<b>{n:,}</b> applicants · <b style='color:#1F3D27'>{n_ok:,} approved</b> · "
                f"<b style='color:#8A3508'>{n - n_ok:,} denied</b> · approval rate "
                f"<b>{n_ok / n:.1%}</b></div>"
                f"<div style='margin:8px 0 12px'>{_download_link(RESULTS_PATH, 'Download bank_results.csv for Fair Lens')}</div>"))
            shown = table.head(500)
            styled = shown.style.hide(axis="index").format({"probability": "{:.3f}"}) \
                .map(lambda v: "background:#E4ECE2;color:#1F3D27;font-weight:600" if v == "Approved"
                     else "background:#FDF0E6;color:#8A3508;font-weight:600", subset=["result"])
            display(styled)
            if len(table) > 500:
                display(HTML(f"<i>Showing the first 500 of {len(table):,} rows. "
                             f"The download has all of them.</i>"))

    def refresh(_=None) -> None:
        files = _data_files()
        keep = [f for f in train_files.value if f in files]
        guess = [f for f in files if any(k in f.lower() for k in ("hmda", "lar", "lender"))]
        train_files.options = files
        train_files.value = tuple(keep or guess)

        prev = apps_file.value
        apps_guess = [f for f in files if "application" in f.lower()]
        apps_file.options = files
        apps_file.value = prev if prev in files else (apps_guess[0] if apps_guess else
                                                      (files[0] if files else None))

        models = _model_files()
        saved_model.options = models
        saved_model.value = (Path(model_path).name if Path(model_path).name in models
                             else (models[0] if models else None))

        files_note.value = (f"<span style='color:#1F3D27'>{len(files)} data file(s) found.</span>"
                            if files else
                            "<span style='color:#8A3508'>No CSV or Excel files yet.</span>")

    def on_train(_=None) -> None:
        with log:
            log.clear_output()
            paths = list(train_files.value) + [p.strip() for p in extra_paths.value.split(",")
                                               if p.strip()]
            if not paths:
                print("Pick at least one training file (Ctrl/Cmd-click to pick several).")
                return
            try:
                print(f"Training on {len(paths)} file(s)… this can take a minute on big files.")
                train(paths, model_path)
                status.value = model_status()
            except Exception as exc:
                print(f"Training failed: {exc}")
        refresh()

    def on_load_model(_=None) -> None:
        with log:
            log.clear_output()
            chosen = saved_model.value
            if not chosen:
                print("No .joblib model files found. Add one to the Files panel, then Refresh.")
                return
            if Path(chosen).resolve() != Path(model_path).resolve():
                shutil.copy(chosen, model_path)
            status.value = model_status()
            print(f"Using saved model {chosen}")

    def on_score(_=None) -> None:
        with log:
            log.clear_output()
            if not Path(model_path).exists():
                print("No model yet. Do step 1 first.")
                return
            if not apps_file.value:
                print("Add applications_for_bank.csv from Fair Lens, click Refresh, then pick it.")
                return
            try:
                state["results"] = score(apps_file.value, RESULTS_PATH, model_path,
                                         penalty=penalty.value)
                state["apps"] = standardize_columns(read_table(
                    apps_file.value, dtype={"case_id": str}))
            except Exception as exc:
                print(f"Scoring failed: {exc}")
                return
        render()

    refresh_btn.on_click(refresh)
    train_btn.on_click(on_train)
    load_btn.on_click(on_load_model)
    score_btn.on_click(on_score)
    show.observe(lambda _: render(), names="value")
    refresh()

    def heading(text: str) -> w.HTML:
        return w.HTML(f"<h3 style='margin:18px 0 6px;color:#12151C'>{text}</h3>")

    display(w.VBox([
        w.HTML("<div style='background:#A9C5A5;padding:14px 18px;border-radius:8px'>"
               "<span style='font-size:20px;font-weight:700;color:#12151C'>Fair Lens</span>"
               "<span style='font-size:16px;color:#2B3030'> · Part 1: Simulated Bank Model</span></div>"),
        w.HTML("<div style='margin-top:12px;font-size:14px'><b>Add your files first:</b> click the "
               "folder icon on Colab's left sidebar and drag your files in (several at once is fine), "
               "or run the upload cell above. Then click <b>Refresh file list</b>.</div>"),
        w.HBox([refresh_btn, files_note]),
        heading("1. Bank model"),
        status,
        w.HTML("<i>Pick one or more HMDA files to train on (Ctrl-click or Cmd-click to pick several). "
               "They're combined into one training set.</i>"),
        train_files,
        extra_paths,
        train_btn,
        w.HBox([saved_model, load_btn]),
        heading("2. Approve / deny Fair Lens applicants"),
        w.HTML("Pick <b>applications_for_bank.csv</b> from Fair Lens (Part 2)."),
        apps_file,
        w.HBox([penalty, w.HTML("&nbsp;<i>0 = learned behavior only. Higher = extra penalty in high-minority tracts (demo).</i>")]),
        score_btn,
        log,
        heading("3. Results"),
        show,
        results_out,
    ]))


# ── Command line ──────────────────────────────────────────────────────────────

def cli() -> None:
    parser = argparse.ArgumentParser(description="Fair Lens Part 1: simulated bank loan model")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train on an HMDA CSV")
    t.add_argument("--hmda", required=True, nargs="+",
                   help="One or more HMDA files (CSV or Excel) with action_taken")
    t.add_argument("--model", default=MODEL_PATH)

    s = sub.add_parser("score", help="Score Part 2's applications_for_bank.csv")
    s.add_argument("--applications", default="applications_for_bank.csv")
    s.add_argument("--out", default=RESULTS_PATH)
    s.add_argument("--model", default=MODEL_PATH)
    s.add_argument("--penalty", type=float, default=EXTRA_REDLINING_PENALTY,
                   help="Demo only: subtract this from approval probability in high-minority tracts")
    s.add_argument("--threshold", type=float, default=None)

    args, _unknown = parser.parse_known_args()   # ignore extra args notebooks pass
    if args.command == "train":
        train(args.hmda, args.model)
    else:
        score(args.applications, args.out, args.model, args.penalty, args.threshold)


def _in_notebook() -> bool:
    """True inside Colab/Jupyter, where the command line belongs to the notebook."""
    return "ipykernel" in sys.modules


if __name__ == "__main__" and not _in_notebook():
    cli()
elif _in_notebook():
    print("Bank model code loaded. In a new cell, run:  launch_ui()")
