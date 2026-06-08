"""Streamlit router for the location-model comparison dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(page_title="Location-Model MLE Dashboard", layout="wide")

st.title("Location-Model MLE Comparison Dashboard")
st.caption(
    "Posterior correctness lives on the Posterior Comparison page; exact Gibbs/RATTLE "
    "cost counters live on the Cost Audit page."
)

with st.sidebar:
    st.header("Pages")
    st.caption("Use the page selector above to switch dashboards.")

if hasattr(st, "Page") and hasattr(st, "navigation"):
    page_dir = Path(__file__).resolve().parent / "pages"
    pg = st.navigation(
        [
            st.Page(str(page_dir / "1_Posterior_Comparison.py"), title="Posterior Comparison"),
            st.Page(str(page_dir / "2_Cost_Audit.py"), title="Cost Audit"),
            st.Page(str(page_dir / "3_Model_Validity_Audit.py"), title="Model Validity Audit"),
            st.Page(str(page_dir / "4_Analysis_Report.py"), title="Analysis Report"),
            st.Page(str(page_dir / "5_KDE_Correctness.py"), title="KDE Correctness"),
            st.Page(str(page_dir / "6_Sampler_Correctness.py"), title="Sampler Correctness"),
            st.Page(str(page_dir / "7_Efficiency.py"), title="Efficiency"),
            st.Page(str(page_dir / "8_Geometry.py"), title="Geometry"),
            st.Page(str(page_dir / "9_MLE_Release_Information.py"), title="MLE Release Information"),
        ]
    )
    pg.run()
else:
    st.warning("This Streamlit version does not support explicit navigation. Run with the classic pages sidebar or upgrade Streamlit.")
    st.code("pip install --upgrade streamlit", language="bash")
