"""Efficiency audit dashboard page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


AUDIT_DIR = Path("results/final_production_v1_efficiency_audit_cost_first")
FILES = {
    "report": AUDIT_DIR / "efficiency_report.md",
    "summary": AUDIT_DIR / "efficiency_summary.csv",
    "cost_decomposition": AUDIT_DIR / "cost_decomposition.csv",
    "method_winners": AUDIT_DIR / "method_winners.csv",
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


st.title("Efficiency")
st.caption("Final production cost-first Gibbs/RATTLE efficiency audit with correctness caveats attached.")

status = pd.DataFrame(
    [{"file": name, "path": str(path), "exists": path.exists()} for name, path in FILES.items()]
)
st.subheader("Audit Status")
st.dataframe(status, use_container_width=True)

report_path = FILES["report"]
if report_path.exists():
    st.markdown(report_path.read_text(encoding="utf-8"))
else:
    st.warning("Efficiency audit has not been generated.")
    st.code(
        "python reporting/diagnostics/analyze_efficiency.py "
        "--runset-dir results/final_production_v1 "
        "--correctness-dir results/final_production_v1_correctness_audit "
        "--out-dir results/final_production_v1_efficiency_audit_cost_first",
        language="bash",
    )
    st.stop()

winners = read_csv(str(FILES["method_winners"]))
summary = read_csv(str(FILES["summary"]))
cost = read_csv(str(FILES["cost_decomposition"]))

st.subheader("Main Winner Table")
st.info("Efficiency is shown as cost per reliable posterior information. Use caveat-only rows for engineering intuition, not clean headline correctness claims.")
if not cost.empty and "comparison_regime" in cost.columns:
    st.dataframe(cost.groupby(["comparison_regime", "method"], dropna=False).size().reset_index(name="rows"), use_container_width=True)
st.dataframe(winners, use_container_width=True)

tab_cost, tab_winners, tab_figures = st.tabs(["Raw Cost", "Winner Details", "Figures"])

with tab_cost:
    st.dataframe(cost, use_container_width=True)
    st.subheader("Quantile Stability Summary")
    st.dataframe(summary, use_container_width=True)

with tab_winners:
    st.dataframe(winners, use_container_width=True)
    st.subheader("Cost Decomposition")
    st.dataframe(cost, use_container_width=True)

with tab_figures:
    figure_dir = AUDIT_DIR / "figures"
    figures = sorted(figure_dir.glob("*.png"))
    if figures:
        st.dataframe(pd.DataFrame({"figure": [str(path) for path in figures]}), use_container_width=True)
        selected = st.selectbox("preview figure", [str(path) for path in figures])
        st.image(selected)
    else:
        st.info(f"No figures found under {figure_dir}.")
