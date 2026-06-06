import numpy as np
import pandas as pd
import jax.random as random
from types import SimpleNamespace

from models import loc_laplace
from models.model_registry import LAPLACE_MEDIAN_INTERVAL_TARGET, LAPLACE_NP_MEDIAN_TARGET, model_validity_rows
from reporting.diagnostics import audit_reference_all_models as audit
from reporting.diagnostics.audit_reference_all_models import laplace_interval_reference


def test_laplace_target_metadata_is_explicit():
    assert LAPLACE_NP_MEDIAN_TARGET["target_description"] == "deterministic_median_equals_mu_star"
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


def test_laplace_default_validity_uses_odd_unique_median_target():
    rows = model_validity_rows()
    laplace_gibbs = [row for row in rows if row["model"] == "laplace" and row["method"] == "gibbs"][0]
    assert laplace_gibbs["target_description"] == "deterministic_median_equals_mu_star"
    assert laplace_gibbs["target_matches_reference"] is True
    assert "odd-n default" in laplace_gibbs["warnings"]


def test_laplace_odd_n_gibbs_pins_unique_median():
    params = {
        "n": 11,
        "b": 1.0,
        "num_iterations_T": 8,
        "proposal_std_mu": 0.2,
        "prior_mean": 0.0,
        "prior_std": 10.0,
    }
    out = loc_laplace.run_gibbs(random.PRNGKey(0), mu_star=0.0, params=params, verbose=False)
    x_chain = np.asarray(out["x_chain"])
    assert np.allclose(x_chain[:, params["n"] // 2], 0.0)
    assert np.allclose(np.median(x_chain, axis=1), 0.0)


def test_laplace_odd_n_reference_uses_scalar_median_kde_target(tmp_path):
    out_csv = tmp_path / "laplace_odd_reference.csv"
    density_csv = tmp_path / "laplace_odd_density.csv"
    args = SimpleNamespace(
        models=["laplace"],
        n_values=[11],
        laplace_n_values=[11, 21, 51],
        n_values_explicit=True,
        laplace_n_values_explicit=False,
        k_values=[2.0],
        B_values=[40],
        seeds=[123],
        bandwidths=["scott"],
        mu_star=0.0,
        prior_mean=0.0,
        prior_std=10.0,
        laplace_b=1.0,
        grid_size=120,
        out_csv=out_csv,
        density_out_csv=density_csv,
        include_laplace_np_median_reference=False,
        overwrite=True,
    )
    audit.emit_rows(args)
    df = pd.read_csv(out_csv)
    assert set(df["estimator_type"]) == {"raw_weighted_mc", "kde_grid"}
    assert set(df["backend"]) == {"none", "scott"}
    assert set(df["target_description"]) == {"deterministic_median_equals_mu_star"}
    density = pd.read_csv(density_csv)
    assert set(density["target_description"]) == {"deterministic_median_equals_mu_star"}
    assert set(density["backend"]) == {"scott"}
