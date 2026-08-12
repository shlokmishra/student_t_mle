"""Model/method validity matrix for posterior and cost comparisons."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.dashboard_cache import read_cache_csv, require_cache_file, sidebar_cache_controls, show_cache_badge
from models.model_registry import MODEL_REGISTRY, model_validity_rows


st.title("Model Validity Audit")
st.caption(
    "This page records which model/method comparisons have a well-defined target, "
    "which support smooth RATTLE constraints, and where caveats block direct deltas."
)
use_dashboard_cache, dashboard_cache_dir, dashboard_manifest = sidebar_cache_controls("validity")
show_cache_badge(use_dashboard_cache, dashboard_cache_dir, dashboard_manifest)

registry_rows = [spec.to_dict() for spec in MODEL_REGISTRY.values()]
registry_df = pd.DataFrame(registry_rows)

st.subheader("Model Registry")
st.dataframe(registry_df, use_container_width=True)

if use_dashboard_cache:
    validity_path = require_cache_file(dashboard_cache_dir, "model_validity_cache.csv")
    if validity_path is None:
        st.stop()
    validity_df = read_cache_csv(str(dashboard_cache_dir), "model_validity_cache.csv")
else:
    validity_df = pd.DataFrame(model_validity_rows())

st.subheader("Validity Matrix")
preferred_columns = [
    "model",
    "k",
    "method",
    "implementation_exists",
    "target_description",
    "mle_convention",
    "target_matches_reference",
    "smooth_constraint",
    "rattle_applicable",
    "tests_passed",
    "warnings",
]
for column in preferred_columns:
    if column not in validity_df.columns:
        validity_df[column] = ""
st.dataframe(validity_df[preferred_columns], use_container_width=True)

laplace_warnings = validity_df[validity_df["model"].eq("laplace") & validity_df["warnings"].astype(str).ne("")]
if not laplace_warnings.empty:
    st.warning("Laplace caveat: exact RATTLE is not applicable because the median constraint is nonsmooth/order-based.")
    st.info(
        "Laplace defaults to odd n=11,21,51, where the sample median is unique and deterministic np.median KDE/raw-MC "
        "matches the Gibbs target. For even n, compare Gibbs only to the median-interval reference."
    )

st.subheader("Implementation Notes")
st.markdown(
    """
- Student Gibbs uses `psi(y)=y/(k+y^2)` with branch weights including `-log|psi'(y)|`.
- Logistic Gibbs uses monotone `psi(y)=tanh(y/2)`, inverse `2*atanh(z)`, and the pushforward Jacobian in `q(z)`.
- Student and Logistic RATTLE use paper-fixed-direction projection with Gram correction enabled by default.
- Laplace defaults to odd `n` so `numpy.median` is the unique sample median and matches Gibbs. Even `n` remains available as a separate median-interval target.
"""
)
