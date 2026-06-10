"""Geometry audit dashboard page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


AUDIT_OPTIONS = {
    "final_production_v1": Path("results/final_production_v1_geometry_audit"),
    "student_k1_n50_geometry": Path("results/student_k1_n50_geometry_audit"),
}
available_options = [name for name, path in AUDIT_OPTIONS.items() if path.exists()]
selected_audit = st.selectbox(
    "Geometry audit runset",
    available_options or list(AUDIT_OPTIONS),
    index=0,
)
AUDIT_DIR = AUDIT_OPTIONS[selected_audit]
FILES = {
    "report": AUDIT_DIR / "geometry_report.md",
    "geometry_summary": AUDIT_DIR / "geometry_summary.csv",
    "latent_tail_geometry": AUDIT_DIR / "latent_tail_geometry.csv",
    "geometry_conditioned_posterior": AUDIT_DIR / "geometry_conditioned_posterior.csv",
    "rattle_geometry": AUDIT_DIR / "rattle_geometry_explanation.csv",
    "gibbs_geometry": AUDIT_DIR / "gibbs_geometry_explanation.csv",
    "branch_exploration": AUDIT_DIR / "branch_exploration.csv",
    "rattle_tail_failure": AUDIT_DIR / "rattle_tail_failure_analysis.csv",
    "gibbs_local_move": AUDIT_DIR / "gibbs_local_move_analysis.csv",
    "win_loss": AUDIT_DIR / "geometry_win_loss_table.csv",
    "missing": AUDIT_DIR / "missing_geometry_diagnostics.csv",
    "unresolved": AUDIT_DIR / "unresolved_geometry_cases.csv",
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


st.title("Geometry")
st.caption("Latent-geometry explanations for Gibbs/RATTLE behavior.")
if selected_audit == "student_k1_n50_geometry":
    st.info(
        "Focused Student-t k=1,n=50 Gibbs audit: use this to inspect Cauchy heavy-tail geometry, "
        "branch movement, and score-space collapse diagnostics."
    )

status = pd.DataFrame(
    [{"file": name, "path": str(path), "exists": path.exists()} for name, path in FILES.items()]
)
st.subheader("Audit Status")
st.dataframe(status, use_container_width=True)

report_path = FILES["report"]
if report_path.exists():
    st.markdown(report_path.read_text(encoding="utf-8"))
else:
    st.warning("Geometry audit has not been generated.")
    st.code(
        "python reporting/diagnostics/analyze_geometry.py "
        "--runsets final_production_v1 "
        "--correctness-dir results/final_production_v1_correctness_audit "
        "--efficiency-dir results/final_production_v1_efficiency_audit_cost_first "
        "--out-dir results/final_production_v1_geometry_audit",
        language="bash",
    )
    st.stop()

win_loss = read_csv(str(FILES["win_loss"]))
missing = read_csv(str(FILES["missing"]))
summary = read_csv(str(FILES["geometry_summary"]))
latent = read_csv(str(FILES["latent_tail_geometry"]))
gibbs = read_csv(str(FILES["gibbs_geometry"]))
branch = read_csv(str(FILES["branch_exploration"]))
rattle = read_csv(str(FILES["rattle_geometry"]))
rattle_tail = read_csv(str(FILES["rattle_tail_failure"]))
conditioned = read_csv(str(FILES["geometry_conditioned_posterior"]))
local = read_csv(str(FILES["gibbs_local_move"]))
unresolved = read_csv(str(FILES["unresolved"]))

st.subheader("Geometry Win/Loss")
st.info("Student k=1,n=10 is an unresolved heavy-tail geometry caveat; Laplace RATTLE is not applicable.")
st.dataframe(win_loss, use_container_width=True)

st.subheader("Missing Geometry Diagnostics")
if not missing.empty and "severity" in missing.columns:
    high = missing[missing["severity"].astype(str).isin(["high", "medium"])]
    if not high.empty:
        st.warning(f"{len(high)} medium/high geometry diagnostics are missing.")
st.dataframe(missing, use_container_width=True)

tab_tail, tab_gibbs, tab_rattle, tab_conditioned, tab_figures = st.tabs(
    ["Tail Geometry", "Gibbs", "RATTLE", "Conditioned Posterior", "Figures"]
)

with tab_tail:
    student = summary[summary["model"].astype(str).eq("student_t")] if not summary.empty else pd.DataFrame()
    st.dataframe(student, use_container_width=True)
    st.subheader("Latent Tail Geometry Rows")
    st.dataframe(latent.head(5000), use_container_width=True)

with tab_gibbs:
    st.dataframe(branch, use_container_width=True)
    st.dataframe(gibbs, use_container_width=True)
    st.dataframe(local, use_container_width=True)

with tab_rattle:
    st.dataframe(rattle, use_container_width=True)
    st.dataframe(rattle_tail, use_container_width=True)

with tab_conditioned:
    st.subheader("Geometry-Conditioned Posterior")
    st.dataframe(conditioned, use_container_width=True)
    st.subheader("Unresolved Geometry Cases")
    st.dataframe(unresolved, use_container_width=True)

with tab_figures:
    figure_dir = AUDIT_DIR / "figures"
    figures = sorted(figure_dir.glob("*.png"))
    if figures:
        st.dataframe(pd.DataFrame({"figure": [str(path) for path in figures]}), use_container_width=True)
        selected = st.selectbox("preview figure", [str(path) for path in figures])
        st.image(selected)
    else:
        st.info(f"No figures found under {figure_dir}.")
