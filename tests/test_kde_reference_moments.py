import numpy as np

from kde_ref.moments import raw_weighted_posterior_moments, weighted_quantile


def test_weighted_quantile_basic():
    vals = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 2.0, 1.0])
    q25, q50, q75 = weighted_quantile(vals, weights, [0.25, 0.5, 0.75])
    assert np.isfinite(q25)
    assert q50 == 0.5
    assert q75 == 1.0


def test_raw_weighted_posterior_moments_manual():
    z = np.array([-1.0, 0.0, 1.0])
    mu_star = 2.0
    out = raw_weighted_posterior_moments(z, mu_star, prior_mean=0.0, prior_std=10.0)
    mu_particles = mu_star - z
    weights = np.exp(-0.5 * (mu_particles / 10.0) ** 2)
    probs = weights / weights.sum()
    mean = np.sum(probs * mu_particles)
    var = np.sum(probs * (mu_particles - mean) ** 2)
    np.testing.assert_allclose(out["posterior_mean"], mean)
    np.testing.assert_allclose(out["posterior_var"], var)
    assert 1.0 <= out["weighted_ess"] <= 3.0
