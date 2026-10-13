import numpy as np
import jax.random as random

from models.loc.student import score_gibbs as loc_student


def _params(iterations=4, n=4, k=2.0, **overrides):
    return {
        "k": k,
        "n": n,
        "num_iterations_T": iterations,
        "proposal_std_mu": 0.2,
        "proposal_std_z": 0.02,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "gibbs_backend": "numba_full",
        **overrides,
    }


def _constraint_residual(x, mu_star, k):
    y = np.asarray(x, dtype=float) - float(mu_star)
    return float(np.sum(y / (float(k) + y * y)))


def test_truncated_normal_inverse_cdf_sampler_stays_in_bounds():
    probs = np.array([1e-6, 0.001, 0.025, 0.5, 0.975, 0.999, 1.0 - 1e-6])
    recovered = np.array([loc_student.norm_cdf(loc_student.norm_ppf(float(p))) for p in probs])
    np.testing.assert_allclose(recovered, probs, rtol=1e-4, atol=1e-7)

    draws = np.array([loc_student.sample_truncated_normal(0.0, 0.1, -0.02, 0.03) for _ in range(200)])
    assert np.all(draws >= -0.02)
    assert np.all(draws <= 0.03)


def test_student_gibbs_smoke_and_preserves_constraint():
    chain = loc_student.run_gibbs(random.PRNGKey(1), 0.0, _params(), verbose=False)
    mus = np.asarray(chain["mu_chain"], dtype=float)
    xs = np.asarray(chain["x_chain"], dtype=float)
    assert mus.shape == (5,)
    assert xs.shape == (5, 4)
    assert np.all(np.isfinite(mus))
    assert np.all(np.isfinite(xs))
    assert chain["gibbs_backend"] == "numba_full"
    assert 0.0 <= float(chain["mu_acceptance_rate"]) <= 1.0
    assert 0.0 <= float(chain["pair_acceptance_rate"]) <= 1.0
    assert 0.0 <= float(chain["z_acceptance_rate"]) <= 1.0
    residuals = [_constraint_residual(row, 0.0, 2.0) for row in xs]
    assert max(abs(val) for val in residuals) < 1e-8


def test_student_gibbs_rejects_legacy_backend():
    with np.testing.assert_raises(ValueError):
        loc_student.run_gibbs(random.PRNGKey(0), 0.0, {**_params(), "gibbs_backend": "jax_scan"}, verbose=False)


def test_student_gibbs_without_x_chain_returns_final_state():
    params = _params(iterations=5, n=6, store_x_chain=False)
    chain = loc_student.run_gibbs(random.PRNGKey(3), 0.0, params, verbose=False)
    mus = np.asarray(chain["mu_chain"], dtype=float)
    x_final = np.asarray(chain["x_final"], dtype=float)
    assert "x_chain" not in chain
    assert mus.shape == (6,)
    assert x_final.shape == (6,)
    assert np.all(np.isfinite(mus))
    assert np.all(np.isfinite(x_final))
    assert abs(_constraint_residual(x_final, 0.0, 2.0)) < 1e-8


def test_full_data_student_mh_uses_the_numba_shared_update():
    params = _params(iterations=5, n=4)
    out = loc_student.run_full_data_mh(
        np.array([0, 17], dtype=np.uint32),
        np.array([-0.5, 0.5, -1.0, 1.0], dtype=np.float64),
        params,
        verbose=False,
    )
    chain = out["mu_chain"]
    assert isinstance(chain, np.ndarray)
    assert chain.dtype == np.float64
    assert chain.shape == (6,)
    assert np.all(np.isfinite(chain))
