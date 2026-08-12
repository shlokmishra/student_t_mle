import numpy as np
import jax.random as random

from models import loc_student


def _params(backend="jax_loop", iterations=4, n=4, k=2.0, **overrides):
    return {
        "k": k,
        "n": n,
        "num_iterations_T": iterations,
        "proposal_std_mu": 0.2,
        "proposal_std_z": 0.02,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "gibbs_backend": backend,
        **overrides,
    }


def _constraint_residual(x, mu_star, k):
    y = np.asarray(x, dtype=float) - float(mu_star)
    return float(np.sum(y / (float(k) + y * y)))


def test_student_pair_update_preserves_score_constraint():
    params = _params()
    x0 = loc_student._initial_x(0.0, 4, 2.0, params)
    x1, pair_acc, z_acc = loc_student._update_x_full(random.PRNGKey(0), x0, 0.1, 0.0, 2.0, 0.02)
    assert int(pair_acc) >= 0
    assert int(z_acc) >= 0
    assert abs(_constraint_residual(np.asarray(x1), 0.0, 2.0)) < 1e-10


def test_numba_truncated_normal_inverse_cdf_sampler_stays_in_bounds():
    probs = np.array([1e-6, 0.001, 0.025, 0.5, 0.975, 0.999, 1.0 - 1e-6])
    recovered = np.array([loc_student._numba_norm_cdf(loc_student._numba_norm_ppf(float(p))) for p in probs])
    np.testing.assert_allclose(recovered, probs, rtol=1e-4, atol=1e-7)

    draws = np.array([loc_student._numba_sample_truncated_normal(0.0, 0.1, -0.02, 0.03) for _ in range(200)])
    assert np.all(draws >= -0.02)
    assert np.all(draws <= 0.03)


def test_student_block_z_pair_update_preserves_score_constraint():
    params = _params()
    x0 = loc_student._initial_x(0.0, 6, 2.0, {**params, "n": 6})
    x1, pair_acc, z_acc, block_acc = loc_student._update_x_full_block_z(random.PRNGKey(0), x0, 0.1, 0.0, 2.0, 0.02)
    assert int(pair_acc) >= 0
    assert int(z_acc) >= 0
    assert bool(block_acc) in {False, True}
    assert abs(_constraint_residual(np.asarray(x1), 0.0, 2.0)) < 1e-10


def test_student_gibbs_backends_smoke_and_preserve_constraint():
    for backend in ["jax_loop", "jax_scan", "jax_scan_block_z", "numba", "numba_full"]:
        chain = loc_student.run_gibbs(random.PRNGKey(1), 0.0, _params(backend=backend), verbose=False)
        mus = np.asarray(chain["mu_chain"], dtype=float)
        xs = np.asarray(chain["x_chain"], dtype=float)
        assert mus.shape == (5,)
        assert xs.shape == (5, 4)
        assert np.all(np.isfinite(mus))
        assert np.all(np.isfinite(xs))
        assert 0.0 <= float(chain["mu_acceptance_rate"]) <= 1.0
        assert 0.0 <= float(chain["pair_acceptance_rate"]) <= 1.0
        assert 0.0 <= float(chain["z_acceptance_rate"]) <= 1.0
        if backend == "jax_scan_block_z":
            assert 0.0 <= float(chain["block_z_acceptance_rate"]) <= 1.0
        residuals = [_constraint_residual(row, 0.0, 2.0) for row in xs]
        assert max(abs(val) for val in residuals) < 1e-8


def test_student_numba_full_without_x_chain_returns_final_state():
    params = _params(backend="numba_full", iterations=5, n=6, k=2.0, store_x_chain=False)
    chain = loc_student.run_gibbs(random.PRNGKey(3), 0.0, params, verbose=False)
    mus = np.asarray(chain["mu_chain"], dtype=float)
    x_final = np.asarray(chain["x_final"], dtype=float)
    assert "x_chain" not in chain
    assert mus.shape == (6,)
    assert x_final.shape == (6,)
    assert np.all(np.isfinite(mus))
    assert np.all(np.isfinite(x_final))
    assert abs(_constraint_residual(x_final, 0.0, 2.0)) < 1e-8


def test_student_numba_full_random_parity_preserves_constraint():
    params = _params(backend="numba_full", iterations=5, n=7, k=2.0, gibbs_pairing_schedule="random_parity")
    chain = loc_student.run_gibbs(random.PRNGKey(4), 0.0, params, verbose=False)
    xs = np.asarray(chain["x_chain"], dtype=float)
    residuals = [_constraint_residual(row, 0.0, 2.0) for row in xs]
    assert max(abs(val) for val in residuals) < 1e-8
    assert chain["gibbs_pairing_schedule"] == "random_parity"


def test_student_numba_full_parallel_pairs_preserve_constraint():
    params = _params(backend="numba_full", iterations=5, n=8, k=2.0, gibbs_pair_parallel=True)
    chain = loc_student.run_gibbs(random.PRNGKey(5), 0.0, params, verbose=False)
    xs = np.asarray(chain["x_chain"], dtype=float)
    residuals = [_constraint_residual(row, 0.0, 2.0) for row in xs]
    assert max(abs(val) for val in residuals) < 1e-8
    assert chain["gibbs_pair_parallel"] is True


def test_student_gibbs_backends_distributional_smoke():
    summaries = {}
    for backend in ["jax_loop", "jax_scan", "jax_scan_block_z", "numba", "numba_full"]:
        means = []
        for seed in [0, 1, 2]:
            params = _params(backend=backend, iterations=20, n=6, k=2.0)
            chain = loc_student.run_gibbs(random.PRNGKey(seed), 0.0, params, verbose=False)
            mus = np.asarray(chain["mu_chain"], dtype=float)
            means.append(float(np.mean(mus[5:])))
        summaries[backend] = float(np.mean(means))
    assert abs(summaries["jax_scan"] - summaries["jax_loop"]) < 1.0
    assert abs(summaries["jax_scan_block_z"] - summaries["jax_loop"]) < 1.0
    assert abs(summaries["numba"] - summaries["jax_loop"]) < 1.0
    assert abs(summaries["numba_full"] - summaries["jax_loop"]) < 1.0
