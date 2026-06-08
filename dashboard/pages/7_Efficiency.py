"""Efficiency audit dashboard page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


AUDIT_DIR = Path("results/efficiency_audit")
FILES = {
    "report": AUDIT_DIR / "efficiency_report.md",
    "summary": AUDIT_DIR / "efficiency_summary.csv",
    "functional_ess": AUDIT_DIR / "functional_ess.csv",
    "cost_decomposition": AUDIT_DIR / "cost_decomposition.csv",
    "method_winners": AUDIT_DIR / "method_winners.csv",
    "rattle_movement": AUDIT_DIR / "rattle_movement_diagnostics.csv",
    "caveats": AUDIT_DIR / "caveat_efficiency_cases.csv",
    "timing": AUDIT_DIR / "timing_warnings.csv",
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
st.caption("Cached Gibbs/RATTLE efficiency audit, filtered by sampler correctness.")

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
        "--correctness-dir results/sampler_correctness_audit "
        "--out-dir results/efficiency_audit",
        language="bash",
    )
    st.stop()

winners = read_csv(str(FILES["method_winners"]))
summary = read_csv(str(FILES["summary"]))
functional = read_csv(str(FILES["functional_ess"]))
cost = read_csv(str(FILES["cost_decomposition"]))
rattle = read_csv(str(FILES["rattle_movement"]))
caveats = read_csv(str(FILES["caveats"]))
timing = read_csv(str(FILES["timing"]))

st.subheader("Main Winner Table")
if not cost.empty and "comparison_regime" in cost.columns:
    st.dataframe(cost.groupby(["comparison_regime", "method"], dropna=False).size().reset_index(name="rows"), use_container_width=True)
st.dataframe(winners, use_container_width=True)

tab_cost, tab_functional, tab_rattle, tab_caveats, tab_figures = st.tabs(
    ["Raw Cost", "Functional ESS", "RATTLE Movement", "Caveats", "Figures"]
)

with tab_cost:
    st.dataframe(cost, use_container_width=True)
    st.subheader("Quantile Stability Summary")
    st.dataframe(summary, use_container_width=True)

with tab_functional:
    st.dataframe(functional, use_container_width=True)

with tab_rattle:
    if not rattle.empty and "high_acceptance_small_move_flag" in rattle.columns:
        flagged = rattle[rattle["high_acceptance_small_move_flag"].astype(bool)]
        if not flagged.empty:
            st.warning(f"{len(flagged)} RATTLE rows have high-acceptance small-move flags.")
            st.dataframe(flagged, use_container_width=True)
        else:
            st.success("No RATTLE high-acceptance small-move flags in the derived cache.")
    st.dataframe(rattle, use_container_width=True)

with tab_caveats:
    st.subheader("Caveat-Only / Excluded Cases")
    st.dataframe(caveats, use_container_width=True)
    st.subheader("Timing Warnings")
    st.dataframe(timing, use_container_width=True)

with tab_figures:
    figure_dir = AUDIT_DIR / "figures"
    figures = sorted(figure_dir.glob("*.png"))
    if figures:
        st.dataframe(pd.DataFrame({"figure": [str(path) for path in figures]}), use_container_width=True)
        selected = st.selectbox("preview figure", [str(path) for path in figures])
        st.image(selected)
    else:
        st.info(f"No figures found under {figure_dir}.")
