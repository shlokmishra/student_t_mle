"""Sampler correctness audit dashboard page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


AUDIT_DIR = Path("results/sampler_correctness_audit")
FILES = {
    "report": AUDIT_DIR / "sampler_correctness_report.md",
    "final_verdicts": AUDIT_DIR / "final_sampler_verdict_table.csv",
    "coverage": AUDIT_DIR / "diagnostic_coverage_table.csv",
    "summary": AUDIT_DIR / "sampler_correctness_summary.csv",
    "suspicious": AUDIT_DIR / "suspicious_sampler_cases.csv",
    "posterior_agreement": AUDIT_DIR / "posterior_agreement.csv",
    "rattle_geometry": AUDIT_DIR / "rattle_geometry_diagnostics.csv",
    "rattle_energy": AUDIT_DIR / "rattle_energy_diagnostics.csv",
    "gibbs_constraints": AUDIT_DIR / "gibbs_constraint_diagnostics.csv",
    "gibbs_branch": AUDIT_DIR / "gibbs_branch_diagnostics.csv",
    "chain_split": AUDIT_DIR / "chain_split_stability.csv",
    "ess_acf": AUDIT_DIR / "ess_autocorrelation_diagnostics.csv",
    "initialization": AUDIT_DIR / "initialization_sensitivity.csv",
    "target_mismatch": AUDIT_DIR / "target_mismatch_diagnostics.csv",
    "missing_outputs": AUDIT_DIR / "missing_outputs.csv",
    "failed_cases": AUDIT_DIR / "failed_cases.csv",
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


st.title("Sampler Correctness")
st.caption("Cached Gibbs/RATTLE correctness audit against raw weighted-MC references.")

status = pd.DataFrame(
    [{"file": name, "path": str(path), "exists": path.exists()} for name, path in FILES.items()]
)
st.subheader("Audit Status")
st.dataframe(status, use_container_width=True)

report_path = FILES["report"]
if report_path.exists():
    st.markdown(report_path.read_text(encoding="utf-8"))
else:
    st.warning("Sampler correctness audit has not been generated.")
    st.code(
        "python reporting/diagnostics/audit_sampler_correctness.py "
        "--runset-dir results/final_production_v1 "
        "--out-dir results/sampler_correctness_audit",
        language="bash",
    )
    st.stop()

final_verdicts = read_csv(str(FILES["final_verdicts"]))
coverage = read_csv(str(FILES["coverage"]))
summary = read_csv(str(FILES["summary"]))
suspicious = read_csv(str(FILES["suspicious"]))
posterior = read_csv(str(FILES["posterior_agreement"]))
rattle = read_csv(str(FILES["rattle_geometry"]))
rattle_energy = read_csv(str(FILES["rattle_energy"]))
gibbs = read_csv(str(FILES["gibbs_constraints"]))
branch = read_csv(str(FILES["gibbs_branch"]))
split = read_csv(str(FILES["chain_split"]))
ess = read_csv(str(FILES["ess_acf"]))
initialization = read_csv(str(FILES["initialization"]))
target = read_csv(str(FILES["target_mismatch"]))
missing_outputs = read_csv(str(FILES["missing_outputs"]))
failed_cases = read_csv(str(FILES["failed_cases"]))

if not failed_cases.empty:
    st.error(f"{len(failed_cases)} failed production cases are present. Do not finalize correctness.")
if not missing_outputs.empty:
    st.error(f"{len(missing_outputs)} required production outputs are missing. Do not finalize correctness.")

st.subheader("Final Verdicts")
if final_verdicts.empty:
    st.warning("Final sampler verdict table is missing.")
else:
    verdict_counts = final_verdicts["verdict"].astype(str).value_counts().to_dict()
    safe_counts = final_verdicts["safe_to_present"].astype(str).value_counts().to_dict()
    c1, c2, c3 = st.columns(3)
    c1.metric("verdict rows", len(final_verdicts))
    c2.metric("clean", verdict_counts.get("clean", 0))
    c3.metric("unsafe/no", safe_counts.get("no", 0))
    st.dataframe(final_verdicts, use_container_width=True)

st.subheader("Diagnostic Coverage")
st.dataframe(coverage, use_container_width=True)

if not summary.empty:
    unresolved = summary[summary["overall_correctness_verdict"].astype(str).eq("unresolved")]
    if not unresolved.empty:
        st.warning(f"{len(unresolved)} sampler verdict rows are unresolved.")
    student_k1 = summary[
        summary["model"].astype(str).eq("student_t")
        & summary["k"].astype(float).eq(1.0)
    ]
    if not student_k1.empty:
        st.info("Student-t k=1 remains a caution zone; use raw weighted-MC, not KDE, as benchmark.")

st.subheader("Suspicious Cases")
if suspicious.empty:
    st.success("No suspicious sampler cases flagged.")
else:
    st.dataframe(suspicious, use_container_width=True)

tab_post, tab_gibbs, tab_rattle, tab_mixing, tab_target = st.tabs(
    ["Posterior", "Gibbs", "RATTLE", "Mixing", "Target"]
)

with tab_post:
    st.dataframe(posterior, use_container_width=True)

with tab_gibbs:
    st.markdown("Constraint residuals, pair-delta preservation, and Student inverse-branch usage.")
    st.dataframe(gibbs, use_container_width=True)
    st.dataframe(branch, use_container_width=True)

with tab_rattle:
    st.markdown("Projection, reverse-check, tangent, and Hamiltonian diagnostics.")
    st.dataframe(rattle, use_container_width=True)
    st.dataframe(rattle_energy, use_container_width=True)

with tab_mixing:
    st.markdown("Split-chain stability, ESS/autocorrelation, and initialization sensitivity.")
    st.dataframe(split, use_container_width=True)
    st.dataframe(ess, use_container_width=True)
    st.dataframe(initialization, use_container_width=True)

with tab_target:
    st.dataframe(target, use_container_width=True)

st.subheader("Figures")
figure_dir = AUDIT_DIR / "figures"
figures = sorted(figure_dir.glob("*.png"))
if figures:
    st.dataframe(pd.DataFrame({"figure": [str(path) for path in figures]}), use_container_width=True)
    selected = st.selectbox("preview figure", [str(path) for path in figures])
    st.image(selected)
else:
    st.info(f"No figures found under {figure_dir}.")
