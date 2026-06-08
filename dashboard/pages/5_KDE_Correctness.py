"""KDE correctness audit dashboard page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


AUDIT_DIR = Path("results/kde_correctness_audit")
FILES = {
    "report": AUDIT_DIR / "kde_correctness_report.md",
    "recommendations": AUDIT_DIR / "backend_recommendations.csv",
    "suspicious": AUDIT_DIR / "suspicious_kde_cases.csv",
    "sensitivity": AUDIT_DIR / "backend_sensitivity.csv",
    "seed_stability": AUDIT_DIR / "seed_stability.csv",
}


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()


st.title("KDE Correctness")
st.caption("Cached numerical audit of KDE backends as smoothed posterior-density diagnostics.")

status = pd.DataFrame(
    [{"file": name, "path": str(path), "exists": path.exists()} for name, path in FILES.items()]
)
st.subheader("Audit Status")
st.dataframe(status, use_container_width=True)

report_path = FILES["report"]
if report_path.exists():
    st.markdown(report_path.read_text(encoding="utf-8"))
else:
    st.warning("KDE correctness audit has not been generated.")
    st.code("python reporting/diagnostics/audit_kde_correctness_all_models.py", language="bash")
    st.stop()

recommendations = read_csv(str(FILES["recommendations"]))
suspicious = read_csv(str(FILES["suspicious"]))
sensitivity = read_csv(str(FILES["sensitivity"]))
seed_stability = read_csv(str(FILES["seed_stability"]))

st.subheader("Backend Recommendations")
st.dataframe(recommendations, use_container_width=True)

if not recommendations.empty and "t_abram_status" in recommendations:
    capped = recommendations[recommendations["t_abram_status"].astype(str).str.contains("capped", case=False, na=False)]
    if not capped.empty:
        st.warning("t_abram has capped diagnostic-only rows and is not recommended as a primary backend.")
    else:
        st.info("t_abram is shown only as a tail/backend stress diagnostic, not as a primary recommendation.")

if not recommendations.empty:
    cauchy = recommendations[
        recommendations["model"].astype(str).eq("student_t")
        & recommendations["k"].astype(float).eq(1.0)
        & recommendations["n"].astype(int).eq(10)
    ]
    if not cauchy.empty:
        st.warning("Student-t k=1,n=10 should be interpreted with caution; do not draw conclusions from KDE alone.")

st.subheader("Suspicious KDE Cases")
st.dataframe(suspicious, use_container_width=True)

st.subheader("Backend Sensitivity")
st.dataframe(sensitivity, use_container_width=True)

st.subheader("Seed Stability")
st.dataframe(seed_stability, use_container_width=True)

st.subheader("Figures")
figure_dir = AUDIT_DIR / "figures"
figures = sorted(figure_dir.glob("*.png"))
if figures:
    st.dataframe(pd.DataFrame({"figure": [str(path) for path in figures]}), use_container_width=True)
    selected = st.selectbox("preview figure", [str(path) for path in figures])
    st.image(selected)
else:
    st.info(f"No figures found under {figure_dir}.")
