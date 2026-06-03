"""Decision-ready analysis report page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard_cache import read_cache_csv, require_cache_file, sidebar_cache_controls, show_cache_badge


REPORT_DIR = Path("results/analysis_report")
FILES = {
    "executive_summary.md": REPORT_DIR / "executive_summary.md",
    "posterior_accuracy.csv": REPORT_DIR / "posterior_accuracy.csv",
    "cost_efficiency.csv": REPORT_DIR / "cost_efficiency.csv",
    "method_rankings.csv": REPORT_DIR / "method_rankings.csv",
    "rattle_diagnostics.csv": REPORT_DIR / "rattle_diagnostics.csv",
    "suspicious_cases.csv": REPORT_DIR / "suspicious_cases.csv",
    "multiseed_summary.csv": REPORT_DIR / "multiseed_summary.csv",
    "rattle_stage2_followup.csv": REPORT_DIR / "rattle_stage2_followup.csv",
    "student_score_vs_mle_diagnostics.csv": REPORT_DIR / "student_score_vs_mle_diagnostics.csv",
    "student_k1_n10_diagnostic.md": REPORT_DIR / "student_k1_n10_diagnostic.md",
}


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()


def status_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"file": name, "path": str(path), "exists": path.exists()} for name, path in FILES.items()]
    )


st.title("Analysis Report")
st.caption("Decision-ready posterior accuracy, cost efficiency, RATTLE diagnostics, and follow-up results.")
use_dashboard_cache, dashboard_cache_dir, dashboard_manifest = sidebar_cache_controls("analysis")
show_cache_badge(use_dashboard_cache, dashboard_cache_dir, dashboard_manifest)

if use_dashboard_cache:
    required = [
        "executive_summary_cache.md",
        "posterior_comparison_cache.csv",
        "cost_efficiency_cache.csv",
        "method_rankings_cache.csv",
        "rattle_diagnostics_cache.csv",
        "suspicious_cases_cache.csv",
        "student_k1_n10_diagnostic_cache.csv",
        "figure_index.csv",
    ]
    missing = [name for name in required if require_cache_file(dashboard_cache_dir, name) is None]
    if missing:
        st.stop()
    st.subheader("Report Status")
    st.dataframe(
        pd.DataFrame(
            [{"file": name, "path": str(dashboard_cache_dir / name), "exists": (dashboard_cache_dir / name).exists()} for name in required]
        ),
        use_container_width=True,
    )
    st.markdown((dashboard_cache_dir / "executive_summary_cache.md").read_text(encoding="utf-8"))
    rankings = read_cache_csv(str(dashboard_cache_dir), "method_rankings_cache.csv")
    suspicious = read_cache_csv(str(dashboard_cache_dir), "suspicious_cases_cache.csv")
    accuracy = read_cache_csv(str(dashboard_cache_dir), "posterior_comparison_cache.csv")
    cost = read_cache_csv(str(dashboard_cache_dir), "cost_efficiency_cache.csv")
    rattle = read_cache_csv(str(dashboard_cache_dir), "rattle_diagnostics_cache.csv")
    student_target_diag = read_cache_csv(str(dashboard_cache_dir), "student_k1_n10_diagnostic_cache.csv")
    figures = read_cache_csv(str(dashboard_cache_dir), "figure_index.csv")

    if not suspicious.empty:
        high = suspicious[suspicious["severity"].astype(str).eq("high")]
        if not high.empty:
            st.error(f"{len(high)} high-severity suspicious cases are flagged.")

    st.subheader("Best Method By Model/k/n")
    st.dataframe(rankings[rankings["rank"].eq(1)] if not rankings.empty else rankings, use_container_width=True)
    st.subheader("Suspicious Cases")
    st.dataframe(suspicious, use_container_width=True)
    st.subheader("RATTLE Diagnostics")
    st.dataframe(rattle, use_container_width=True)
    st.subheader("Student k=1,n=10 diagnostic")
    if (dashboard_cache_dir / "student_k1_n10_diagnostic_cache.md").exists():
        st.markdown((dashboard_cache_dir / "student_k1_n10_diagnostic_cache.md").read_text(encoding="utf-8"))
    st.dataframe(student_target_diag, use_container_width=True)
    st.subheader("Posterior Accuracy")
    st.dataframe(accuracy, use_container_width=True)
    st.subheader("Cost Efficiency")
    st.dataframe(cost, use_container_width=True)
    st.subheader("Figures")
    st.dataframe(figures, use_container_width=True)
    if not figures.empty:
        selected = st.selectbox("preview figure", figures["path"].astype(str).tolist())
        st.image(selected)
    st.stop()

st.subheader("Report Status")
st.dataframe(status_table(), use_container_width=True)

summary_path = FILES["executive_summary.md"]
if summary_path.exists():
    st.markdown(summary_path.read_text(encoding="utf-8"))
else:
    st.warning("Analysis report is missing.")
    st.code("python reporting/diagnostics/analyze_full_comparison.py", language="bash")
    st.stop()

rankings = read_csv(str(FILES["method_rankings.csv"]))
suspicious = read_csv(str(FILES["suspicious_cases.csv"]))
accuracy = read_csv(str(FILES["posterior_accuracy.csv"]))
cost = read_csv(str(FILES["cost_efficiency.csv"]))
rattle = read_csv(str(FILES["rattle_diagnostics.csv"]))
multiseed = read_csv(str(FILES["multiseed_summary.csv"]))
stage2 = read_csv(str(FILES["rattle_stage2_followup.csv"]))
student_target_diag = read_csv(str(FILES["student_score_vs_mle_diagnostics.csv"]))

if not suspicious.empty:
    high = suspicious[suspicious["severity"].astype(str).eq("high")]
    if not high.empty:
        st.error(f"{len(high)} high-severity suspicious cases are flagged.")
    medium = suspicious[suspicious["severity"].astype(str).eq("medium")]
    if not medium.empty:
        st.warning(f"{len(medium)} medium-severity suspicious cases are flagged.")

st.subheader("Best Method By Model/k/n")
if rankings.empty:
    st.info("No method rankings available.")
else:
    st.dataframe(rankings[rankings["rank"].eq(1)], use_container_width=True)

st.subheader("Gibbs vs RATTLE")
if cost.empty:
    st.info("No cost table available.")
else:
    compare = cost[~cost["rattle_status"].astype(str).eq("not_applicable")].copy()
    st.dataframe(
        compare[
            [
                "model",
                "k",
                "n",
                "method",
                "ess_per_sec",
                "wall_time_per_ess",
                "wall_time_per_iteration",
                "acceptance_rate",
                "projection_failure_rate",
                "reverse_check_failure_rate",
            ]
        ],
        use_container_width=True,
    )

st.subheader("Suspicious Cases")
st.dataframe(suspicious, use_container_width=True)

st.subheader("RATTLE Diagnostics")
st.dataframe(rattle, use_container_width=True)

if not stage2.empty:
    st.subheader("RATTLE Stage2 Follow-Up")
    st.dataframe(stage2, use_container_width=True)

st.subheader("Student k=1,n=10 diagnostic")
student_diag_md = FILES["student_k1_n10_diagnostic.md"]
if student_diag_md.exists():
    st.markdown(student_diag_md.read_text(encoding="utf-8"))
elif student_target_diag.empty:
    st.info("No Student score-vs-MLE diagnostic is available.")
    st.code(
        "python scripts/run_cost_audit.py --models student_t --methods gibbs rattle --k-values 1 --n-values 10 --num-iterations 10000 --burn-in 2000 --seed 0 --out results/student_k1_n10_target_diag/ --save-latent-diagnostics --latent-diagnostic-thin 10\n"
        "python reporting/diagnostics/diagnose_student_score_vs_mle.py --chain-csv results/student_k1_n10_target_diag/chain_samples.csv --posterior-summaries-csv results/student_k1_n10_target_diag/posterior_summaries.csv --reference-csv reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv --latent-csv results/student_k1_n10_target_diag/latent_x_diagnostics.csv --k-values 1 --n-values 10",
        language="bash",
    )

if not student_target_diag.empty:
    focus = student_target_diag[(student_target_diag["k"].eq(1.0)) & (student_target_diag["n"].eq(10))].copy()
    if not focus.empty:
        summary_rows = []
        for method, group in focus.groupby("method"):
            delta = group["selected_mle_minus_mu_star"].dropna()
            summary_rows.append(
                {
                    "method": method,
                    "target_mismatch_rate": float((group["score_near_zero"] & ~group["selected_mle_near_mu_star"]).mean()),
                    "selected_mle_delta_mean": float(delta.mean()) if not delta.empty else pd.NA,
                    "selected_mle_delta_sd": float(delta.std(ddof=0)) if not delta.empty else pd.NA,
                    "avg_loglik_selected_minus_mu_star": float(group["loglik_selected_minus_mu_star"].mean()),
                    "recommendation": str(group["classification"].iloc[0]),
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

if not multiseed.empty:
    st.subheader("Multi-Seed Robustness")
    st.dataframe(multiseed, use_container_width=True)

st.subheader("Posterior Accuracy")
st.dataframe(accuracy, use_container_width=True)

st.subheader("Figures")
figure_dir = REPORT_DIR / "figures"
figures = sorted(path for path in figure_dir.glob("*.png"))
if figures:
    st.dataframe(pd.DataFrame({"figure": [str(path) for path in figures]}), use_container_width=True)
    selected = st.selectbox("preview figure", [str(path) for path in figures])
    st.image(selected)
else:
    st.info("No figures found.")
