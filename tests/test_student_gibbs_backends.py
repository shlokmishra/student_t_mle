import numpy as np
import jax.random as random

from models import loc_student


def _params(backend="jax_loop", iterations=4, n=4, k=2.0):
    return {
        "k": k,
        "n": n,
        "num_iterations_T": iterations,
        "proposal_std_mu": 0.2,
        "proposal_std_z": 0.02,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "gibbs_backend": backend,
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


def test_student_gibbs_backends_smoke_and_preserve_constraint():
    for backend in ["jax_loop", "jax_scan", "numba"]:
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
        residuals = [_constraint_residual(row, 0.0, 2.0) for row in xs]
        assert max(abs(val) for val in residuals) < 1e-8


def test_student_gibbs_backends_distributional_smoke():
    summaries = {}
    for backend in ["jax_loop", "jax_scan", "numba"]:
        means = []
        for seed in [0, 1, 2]:
            params = _params(backend=backend, iterations=20, n=6, k=2.0)
            chain = loc_student.run_gibbs(random.PRNGKey(seed), 0.0, params, verbose=False)
            mus = np.asarray(chain["mu_chain"], dtype=float)
            means.append(float(np.mean(mus[5:])))
        summaries[backend] = float(np.mean(means))
    assert abs(summaries["jax_scan"] - summaries["jax_loop"]) < 1.0
    assert abs(summaries["numba"] - summaries["jax_loop"]) < 1.0
