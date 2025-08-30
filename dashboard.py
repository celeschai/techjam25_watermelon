# =============================================================================
# Streamlit Dashboard for Business Review Moderation Pipeline
# =============================================================================
# This dashboard provides an interactive web interface for the Gemma-based
# business review moderation pipeline, enabling users to upload CSV files
# and visualize results in real-time.
#
# Purpose:
# - Provide user-friendly interface for the ML pipeline
# - Enable real-time upload and processing of review datasets
# - Visualize policy violations, helpfulness metrics, and temporal trends
# - Offer downloadable results for further analysis
#
# Key Features:
# - Drag-and-drop CSV upload with robust validation
# - Real-time backend processing via gemma_pipeline_dashboard.py
# - Interactive visualizations of policy violations and helpfulness
# - Temporal analysis of reviews over time
# - Comprehensive error handling and user feedback
# - Downloadable processed results
#
# Architecture:
# - Frontend: Streamlit web interface
# - Backend: Python subprocess execution of ML pipeline
# - Data Flow: CSV upload → dashboard.csv → ML processing → dashboard_output.csv → visualization
# - Integration: Seamless connection between UI and ML pipeline
#
# Input Requirements:
# - CSV with columns: user_id, time, rating, text, pics_collapsed, name, category
# - Supports various encodings (UTF-8, Latin-1) for robustness
# - Handles missing data gracefully
#
# Output Features:
# - Policy violation counts and summaries
# - Helpfulness rating distributions
# - Temporal review patterns
# - Processed CSV download with all ML-generated columns
# =============================================================================

import io
import os
import subprocess
import traceback
import pandas as pd
import streamlit as st

# ==============================
# Config / Schemas
# ==============================
INPUT_COLUMNS = [
    "user_id","time","rating","text","pics_collapsed","name","category"
]
OUTPUT_COLUMNS = [
    "review_id","user_id","time","rating","text","pics_collapsed","resp_collapsed",
    "name","description","category","url","image",
    "is_image_ad","is_image_irrelevant","is_text_ad","is_text_irrelevant","is_text_rant",
    "is_review_ad","is_review_irrelevant","helpfulness","sensibility"
]
POLICY_BOOL_COLS = ["is_text_rant","is_review_ad","is_review_irrelevant"]
HELPFULNESS_ORDER = ["very_helpful", "helpful", "not_helpful"]

INPUT_SAVE_PATH = "dashboard.csv"
PIPELINE_OUT_PATH = "dashboard_output.csv"
BACKEND_SCRIPT = "gemma_pipeline_dashboard.py"   # ensure this is available in the same directory or give full path

# ==============================
# Helpers
# ==============================
def _coerce_bool_series(s: pd.Series) -> pd.Series:
    """Turn values like 'True', 'true', 1, '1', 'yes' into True; else False; handles NaN."""
    true_set = {"true", "1", "yes", "y", "t"}
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if pd.isna(x):
            return False
        x_str = str(x).strip().lower()
        return x_str in true_set
    return s.map(to_bool)

def _validate_schema(df: pd.DataFrame, expected_cols: list[str]) -> tuple[bool, list[str]]:
    missing = [c for c in expected_cols if c not in df.columns]
    return (len(missing) == 0, missing)

def _read_csv_resilient(uploaded_bytes: bytes) -> pd.DataFrame | None:
    """Try a few encodings for robustness."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(uploaded_bytes), encoding=enc)
        except Exception:
            continue
    return None

def _ms_to_datetime(ms_series: pd.Series) -> pd.Series:
    """Convert epoch milliseconds to pandas datetime (NaT if invalid) using pd.to_datetime."""
    # Ensure numeric then convert with unit='ms'
    ms = pd.to_numeric(ms_series, errors="coerce")
    # Explicit use of pd.to_datetime as requested
    return pd.to_datetime(ms, unit="ms", errors="coerce")

def _aggregate_by_sensible_scale(dt_series: pd.Series) -> pd.DataFrame:
    """Group review counts by a sensible time scale. Use monthly if span is long; else daily."""
    # Explicitly using pd.to_datetime here as well
    dt = pd.to_datetime(dt_series, errors="coerce")
    dt_valid = dt.dropna()
    if dt_valid.empty:
        return pd.DataFrame(columns=["period", "count"])

    span_days = (dt_valid.max() - dt_valid.min()).days
    if span_days > 365 * 2:
        # group by month for long spans
        key = dt.dt.to_period("M").astype(str)
    elif span_days > 180:
        # group by week for medium spans
        key = dt.dt.to_period("W").astype(str)
    else:
        # group by day for short spans
        key = dt.dt.date.astype(str)

    counts = pd.Series(1, index=dt.index).groupby(key).sum().reset_index()
    counts.columns = ["period", "count"]
    return counts.sort_values("period")

def _save_input_csv(df: pd.DataFrame, path: str = INPUT_SAVE_PATH):
    df.to_csv(path, index=False, encoding="utf-8")

def _run_backend(script_path: str = BACKEND_SCRIPT) -> tuple[bool, str]:
    """
    Run gemma_pipeline.py which should read `dashboard.csv` and produce `pipeline_output.csv`.
    Returns (ok, logs). If script exits non-zero, ok=False with stderr/stdout attached.
    """
    if not os.path.exists(script_path):
        return False, f"Backend script not found: {script_path}"
    # Run `python gemma_pipeline.py` (inherits working directory)
    try:
        completed = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            check=False
        )
        logs = f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        if completed.returncode != 0:
            return False, f"Backend exited with code {completed.returncode}\n{logs}"
        return True, logs
    except Exception as e:
        return False, f"Exception while running backend: {e}"

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="Google Reviews: Policy & Helpfulness Dashboard", layout="wide")
st.title("📥 Google Reviews — CSV Policy Checker & Helpfulness Summary")

with st.expander("Expected input & output schemas", expanded=False):
    st.markdown("**Input CSV (columns, order not required):**")
    st.code(", ".join(INPUT_COLUMNS), language="text")
    st.markdown("**Output CSV (produced by backend):**")
    st.code(", ".join(OUTPUT_COLUMNS), language="text")
    st.markdown(f"- Uploaded CSV is saved as **{INPUT_SAVE_PATH}**")
    st.markdown(f"- Backend script executed: **{BACKEND_SCRIPT}** (must write **{PIPELINE_OUT_PATH}**)")

uploaded = st.file_uploader(
    "Drag & drop or browse your Google Reviews CSV",
    type=["csv"],
    accept_multiple_files=False,
    key="reviews_csv_uploader"
)

error_slot = st.empty()

if uploaded is None:
    st.info("Awaiting CSV upload…")
    st.stop()

# ==============================
# Pipeline with robust error handling
# ==============================
try:
    # 1) Read input CSV (robust)
    raw_bytes = uploaded.getvalue()
    if not raw_bytes:
        st.error("Invalid input: upload failed or file is empty.")
        st.stop()

    in_df = _read_csv_resilient(raw_bytes)
    if in_df is None:
        st.error("Invalid input: unable to read CSV (encoding/format issue).")
        st.stop()

    # 2) Validate input schema
    ok_in, missing_in = _validate_schema(in_df, INPUT_COLUMNS)
    if not ok_in:
        st.error(f"Invalid input: CSV missing required columns: {missing_in}")
        st.stop()

    # 3) Save input as dashboard.csv
    _save_input_csv(in_df, INPUT_SAVE_PATH)

    # 4) Run backend (gemma_pipeline.py)
    with st.spinner("Running backend processing (gemma_pipeline.py)…"):
        ok, backend_logs = _run_backend(BACKEND_SCRIPT)

    with st.expander("Backend logs", expanded=False):
        st.code(backend_logs or "(no logs)", language="text")

    if not ok:
        st.error("Invalid input: backend failed to process the CSV.")
        st.stop()

    # 5) Load pipeline_output.csv
    if not os.path.exists(PIPELINE_OUT_PATH):
        st.error(f"Invalid input: backend did not produce '{PIPELINE_OUT_PATH}'.")
        st.stop()

    out_df = pd.read_csv(PIPELINE_OUT_PATH, encoding="utf-8")

    # 6) Validate output schema
    ok_out, missing_out = _validate_schema(out_df, OUTPUT_COLUMNS)
    if not ok_out:
        st.error(f"Invalid input: backend output missing columns: {missing_out}")
        st.stop()

    # ==============================
    # Panels & Visuals
    # ==============================
    st.subheader("Preview (first 25 rows)")
    st.dataframe(out_df.head(25), use_container_width=True)

    # ---- Policy Violations ----
    st.subheader("Policy Violations")
    for col in POLICY_BOOL_COLS:
        if col not in out_df.columns:
            st.error(f"Output missing '{col}' for policy counting.")
            st.stop()

    pv_counts = {col: _coerce_bool_series(out_df[col]).sum() for col in POLICY_BOOL_COLS}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Text Rants", int(pv_counts["is_text_rant"]))
    with c2:
        st.metric("Review Ads", int(pv_counts["is_review_ad"]))
    with c3:
        st.metric("Review Irrelevant", int(pv_counts["is_review_irrelevant"]))

    # ---- Helpfulness (separate element) ----
    st.subheader("Helpfulness Ratings")
    if "helpfulness" not in out_df.columns:
        st.error("Output missing 'helpfulness' column.")
        st.stop()

    help_norm = out_df["helpfulness"].astype(str).str.strip().str.lower()
    help_counts = help_norm.value_counts()
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        st.metric("Very Helpful", int(help_counts.get("very_helpful", 0)))
    with hc2:
        st.metric("Helpful", int(help_counts.get("helpful", 0)))
    with hc3:
        st.metric("Not Helpful", int(help_counts.get("not_helpful", 0)))

    # ---- Reviews over Time ----
    st.subheader("Reviews Over Time")
    if "time" not in out_df.columns:
        st.error("Output missing 'time' column for timeline.")
        st.stop()

    # Explicitly use pd.to_datetime (ms → datetime)
    dt = _ms_to_datetime(out_df["time"])
    counts = _aggregate_by_sensible_scale(dt)
    if counts.empty:
        st.info("No valid timestamps to plot.")
    else:
        st.line_chart(data=counts.set_index("period")["count"], use_container_width=True)

    # Download processed output
    with st.expander("Download processed CSV", expanded=False):
        try:
            with open(PIPELINE_OUT_PATH, "rb") as f:
                st.download_button(
                    "Download output csv",
                    data=f,
                    file_name="dashboard_download_output.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error("Could not open pipeline_output.csv for download.")
            st.code("".join(traceback.format_exception_only(type(e), e)).strip(), language="text")

except Exception as e:
    error_slot.error("Invalid input: An error occurred during processing.")
    st.code("".join(traceback.format_exception_only(type(e), e)).strip(), language="text")
    with st.expander("Show full traceback (for debugging)"):
        st.code(traceback.format_exc(), language="text")