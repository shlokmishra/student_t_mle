"""Streamlit router for the location-model comparison dashboard."""

from __future__ import annotations

import streamlit as st


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
    pg = st.navigation(
        [
            st.Page("pages/1_Posterior_Comparison.py", title="Posterior Comparison"),
            st.Page("pages/2_Cost_Audit.py", title="Cost Audit"),
            st.Page("pages/3_Model_Validity_Audit.py", title="Model Validity Audit"),
            st.Page("pages/4_Analysis_Report.py", title="Analysis Report"),
        ]
    )
    pg.run()
else:
    st.warning("This Streamlit version does not support explicit navigation. Run with the classic pages sidebar or upgrade Streamlit.")
    st.code("pip install --upgrade streamlit", language="bash")
