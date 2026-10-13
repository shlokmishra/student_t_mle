import numpy as np
from scipy.special import gammaln, logsumexp

from kde_ref.moments import raw_weighted_posterior_moments, weighted_quantile
from kde_ref.numba_kernels import adaptive_student_logpdf, fixed_student_density
from kde_ref.reference_adapter import STUDENT_MLE_CACHE_VERSION, mle_sample_cache_path


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


def test_numba_student_kernels_match_logsumexp_reference():
    centers = np.array([-1.0, 0.25, 1.5])
    eval_points = np.array([-0.5, 0.0, 0.75])
    bandwidth = 0.8
    nu = 3.0
    log_constant = gammaln((nu + 1.0) / 2.0) - 0.5 * np.log(nu * np.pi) - gammaln(nu / 2.0)
    terms = (
        log_constant
        - 0.5 * (nu + 1.0) * np.log1p(((eval_points[:, None] - centers[None, :]) / bandwidth) ** 2 / nu)
        - np.log(bandwidth)
    )
    expected = np.exp(logsumexp(terms, axis=1) - np.log(centers.size))
    np.testing.assert_allclose(fixed_student_density(eval_points, centers, bandwidth, nu), expected, rtol=1e-12, atol=1e-12)

    bandwidths = np.full(centers.size, bandwidth)
    np.testing.assert_allclose(
        np.exp(adaptive_student_logpdf(eval_points, centers, bandwidths, nu, np.log(1e-300))),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_student_mle_cache_path_cannot_reuse_legacy_score_root_samples(tmp_path):
    path = mle_sample_cache_path(k=1.0, n=10, B=1000, seed=3, audit_dir=tmp_path)
    assert STUDENT_MLE_CACHE_VERSION in path.name
    assert path.name != "student_k1.0_n10_B1000_seed3.npz"
