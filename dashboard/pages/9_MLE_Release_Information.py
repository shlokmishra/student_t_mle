"""MLE release information-loss and privacy-leakage dashboard page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


AUDIT_DIR = Path("results/release_information_audit")
FILES = {
    "report": AUDIT_DIR / "release_information_report.md",
    "coverage": AUDIT_DIR / "diagnostic_coverage.csv",
    "info_summary": AUDIT_DIR / "information_loss_summary.csv",
    "info_by_dataset": AUDIT_DIR / "information_loss_by_dataset.csv",
    "privacy_summary": AUDIT_DIR / "privacy_leakage_summary.csv",
    "privacy_by_case": AUDIT_DIR / "privacy_leakage_by_case.csv",
    "normal_baseline": AUDIT_DIR / "sufficient_baseline_normal.csv",
    "posterior_inputs": AUDIT_DIR / "posterior_summary_inputs.csv",
}


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


st.title("MLE Release Information")
st.caption("Step 4: information loss and latent privacy leakage after sampler correctness is trusted.")

status = pd.DataFrame(
    [{"file": name, "path": str(path), "exists": path.exists()} for name, path in FILES.items()]
)
st.subheader("Audit Status")
st.dataframe(status, use_container_width=True)

report_path = FILES["report"]
if report_path.exists():
    st.markdown(report_path.read_text(encoding="utf-8"))
else:
    st.warning("MLE release information audit has not been generated.")
    st.code(
        "python reporting/diagnostics/analyze_release_information.py "
        "--mle-runset-dir results/final_production_v1 "
        "--release-runset-dir results/release_information_runs "
        "--out-dir results/release_information_audit",
        language="bash",
    )
    st.stop()

coverage = read_csv(str(FILES["coverage"]))
info_summary = read_csv(str(FILES["info_summary"]))
info_by_dataset = read_csv(str(FILES["info_by_dataset"]))
privacy_summary = read_csv(str(FILES["privacy_summary"]))
privacy_by_case = read_csv(str(FILES["privacy_by_case"]))
normal = read_csv(str(FILES["normal_baseline"]))
posterior_inputs = read_csv(str(FILES["posterior_inputs"]))

st.subheader("Coverage")
if not coverage.empty and "available" in coverage.columns:
    missing = coverage[~coverage["available"].astype(bool)]
    if not missing.empty:
        st.warning(f"{len(missing)} Step 4 diagnostics are not available yet.")
st.dataframe(coverage, use_container_width=True)

tab_info, tab_privacy, tab_baseline, tab_inputs, tab_figures = st.tabs(
    ["Information Loss", "Privacy Leakage", "Normal Baseline", "Inputs", "Figures"]
)

with tab_info:
    st.dataframe(info_summary, use_container_width=True)
    st.subheader("Dataset-Level Rows")
    st.dataframe(info_by_dataset, use_container_width=True)

with tab_privacy:
    st.dataframe(privacy_summary, use_container_width=True)
    st.subheader("Case-Level Rows")
    st.dataframe(privacy_by_case, use_container_width=True)

with tab_baseline:
    st.dataframe(normal, use_container_width=True)

with tab_inputs:
    st.dataframe(posterior_inputs, use_container_width=True)

with tab_figures:
    figure_dir = AUDIT_DIR / "figures"
    figures = sorted(figure_dir.glob("*.png"))
    if figures:
        st.dataframe(pd.DataFrame({"figure": [str(path) for path in figures]}), use_container_width=True)
        selected = st.selectbox("preview figure", [str(path) for path in figures])
        st.image(selected)
    else:
        st.info(f"No figures found under {figure_dir}.")
