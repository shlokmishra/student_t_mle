import numpy as np

from models.model_registry import LAPLACE_MEDIAN_INTERVAL_TARGET, LAPLACE_NP_MEDIAN_TARGET, model_validity_rows
from reporting.diagnostics.audit_reference_all_models import laplace_interval_reference


def test_laplace_target_metadata_is_explicit():
    assert LAPLACE_NP_MEDIAN_TARGET["target_description"] == "deterministic_np_median_equals_mu_star"
    assert LAPLACE_MEDIAN_INTERVAL_TARGET["target_description"] == "median_interval_contains_mu_star"


def test_laplace_interval_reference_returns_normalized_summary():
    lower = np.array([-1.0, -0.5, 0.0, 0.5])
    upper = np.array([0.0, 0.5, 1.0, 1.5])
    out = laplace_interval_reference(
        lower,
        upper,
        mu_star=0.0,
        prior_mean=0.0,
        prior_std=2.0,
        grid_size=400,
    )
    assert np.isfinite(out["mean"])
    assert np.isfinite(out["sd"])
    assert out["q025"] <= out["q50"] <= out["q975"]
    assert out["marginal_likelihood_estimate"] > 0.0


def test_laplace_even_n_mismatch_is_declared():
    rows = model_validity_rows()
    laplace_gibbs = [row for row in rows if row["model"] == "laplace" and row["method"] == "gibbs"][0]
    assert laplace_gibbs["target_description"] == "median_interval_contains_mu_star"
    assert laplace_gibbs["target_matches_reference"] is False
    assert "not directly comparable" in laplace_gibbs["warnings"]
