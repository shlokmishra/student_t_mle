import numpy as np
import jax.random as random
import pytest

from models import loc_laplace, loc_laplace_rattle


def _params(**overrides):
    params = {
        "n": 11,
        "b": 1.0,
        "num_iterations_T": 12,
        "proposal_std_mu": 0.2,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "rattle_step_size": 0.03,
        "rattle_num_steps": 2,
        "reverse_check": True,
        "rattle_reverse_position_tol": 1e-2,
        "rattle_reverse_momentum_tol": 1e-2,
        "kink_tol": 1e-8,
        "laplace_rattle_experimental": True,
    }
    params.update(overrides)
    return params


def _side_counts(x, mu_star=0.0):
    x = np.asarray(x, dtype=float)
    return int(np.sum(x < mu_star)), int(np.sum(x == mu_star)), int(np.sum(x > mu_star))


def test_laplace_facet_rattle_even_n_raises():
    with pytest.raises(ValueError, match="only odd n"):
        loc_laplace_rattle.run_rattle(random.PRNGKey(0), 0.0, _params(n=10), verbose=False)


def test_laplace_facet_rattle_requires_experimental_flag():
    params = _params()
    params["laplace_rattle_experimental"] = False
    with pytest.raises(ValueError, match="experimental"):
        loc_laplace_rattle.run_rattle(random.PRNGKey(0), 0.0, params, verbose=False)


def test_laplace_facet_rattle_preserves_fixed_median_facet():
    params = _params(num_iterations_T=8)
    out = loc_laplace_rattle.run_rattle(random.PRNGKey(1), 0.0, params, verbose=False)
    xs = np.asarray(out["x_chain"], dtype=float)
    median_idx = out["median_index"]
    assert median_idx == params["n"] // 2
    assert np.all(xs[:, median_idx] == 0.0)
    assert np.allclose(np.median(xs, axis=1), 0.0)
    for row in xs:
        assert _side_counts(row) == (params["n"] // 2, 1, params["n"] // 2)
    diag = out["projection_diagnostics"]
    assert diag["median_residual"] == 0.0
    assert diag["side_count_failures"] == 0


def test_laplace_facet_trajectory_keeps_median_momentum_zero():
    params = _params(n=11)
    x0 = np.asarray(loc_laplace._initial_x(0.0, params["n"], params), dtype=float)
    median_idx = params["n"] // 2
    momentum = np.ones(params["n"], dtype=float)
    x1, p1, ok, diag = loc_laplace_rattle._facet_trajectory(
        x0,
        momentum,
        mu=0.2,
        mu_star=0.0,
        b=1.0,
        step_size=0.01,
        num_steps=2,
        median_idx=median_idx,
    )
    assert ok
    assert x1[median_idx] == 0.0
    assert p1[median_idx] == 0.0
    assert diag["median_momentum_abs"] == 0.0


def test_laplace_facet_side_boundary_crossing_primitive_detects_chamber_exit():
    x_prev = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    x_new = np.array([-1.0, 0.2, 0.0, -0.1, 1.0])
    crossed, count = loc_laplace_rattle._side_boundary_crossed_between(
        x_prev,
        x_new,
        mu_star=0.0,
        median_idx=2,
    )
    assert crossed is True
    assert count == 2


def test_laplace_facet_rattle_sign0_is_deterministic():
    values = np.array([-2.0, -0.0, 0.0, 3.0])
    np.testing.assert_array_equal(loc_laplace_rattle._sign0(values), np.array([-1.0, 0.0, 0.0, 1.0]))


def test_laplace_facet_rattle_diagnostics_handle_kink_crossings():
    params = _params(num_iterations_T=10, rattle_step_size=0.2, rattle_num_steps=4, kink_tol=0.5)
    out = loc_laplace_rattle.run_rattle(random.PRNGKey(2), 0.0, params, verbose=False)
    diag = out["projection_diagnostics"]
    required = [
        "kink_cross_count",
        "near_kink_count",
        "accept_rate_if_crossed",
        "accept_rate_if_not_crossed",
        "mean_delta_H_if_crossed",
        "mean_delta_H_if_not_crossed",
        "reverse_fail_if_crossed",
        "reverse_fail_if_not_crossed",
        "side_boundary_violation_count",
        "latent_acceptance_rate",
        "delta_h_mean_abs",
        "delta_h_rms",
        "delta_h_max_abs",
        "delta_h_median",
        "delta_h_q95",
        "median_delta_H_if_crossed",
        "median_delta_H_if_not_crossed",
        "q95_delta_H_if_crossed",
        "q95_delta_H_if_not_crossed",
        "side_boundary_cross_count",
        "side_boundary_cross_coordinate_count",
    ]
    for group in ["clean", "kink_only", "boundary_only", "kink_and_boundary"]:
        required.extend([
            f"{group}_proposals",
            f"{group}_accepted",
            f"accept_rate_{group}",
            f"mean_delta_H_{group}",
            f"mean_abs_delta_H_{group}",
            f"median_delta_H_{group}",
            f"q95_delta_H_{group}",
            f"rms_delta_H_{group}",
            f"max_abs_delta_H_{group}",
            f"reverse_fail_rate_{group}",
            f"reverse_fail_rate_checked_{group}",
            f"mean_reverse_position_error_{group}",
            f"mean_reverse_momentum_error_{group}",
        ])
    for key in required:
        assert key in diag
        if "delta_H" in key or key.startswith("delta_h_"):
            assert np.isfinite(float(diag[key])) or np.isnan(float(diag[key]))
        else:
            assert np.isfinite(float(diag[key]))


def test_laplace_facet_rattle_smoke_matches_gibbs_roughly():
    rattle_means = []
    rattle_sds = []
    gibbs_means = []
    gibbs_sds = []
    for seed in [0, 1, 2]:
        rattle_params = _params(num_iterations_T=80, rattle_step_size=0.03, rattle_num_steps=2)
        gibbs_params = {
            "n": 11,
            "b": 1.0,
            "num_iterations_T": 80,
            "proposal_std_mu": 0.2,
            "prior_mean": 0.0,
            "prior_std": 10.0,
        }
        rattle = loc_laplace_rattle.run_rattle(random.PRNGKey(seed), 0.0, rattle_params, verbose=False)
        gibbs = loc_laplace.run_gibbs(random.PRNGKey(seed), 0.0, gibbs_params, verbose=False)
        r_mu = np.asarray(rattle["mu_chain"], dtype=float)[20:]
        g_mu = np.asarray(gibbs["mu_chain"], dtype=float)[20:]
        rattle_means.append(float(np.mean(r_mu)))
        rattle_sds.append(float(np.std(r_mu)))
        gibbs_means.append(float(np.mean(g_mu)))
        gibbs_sds.append(float(np.std(g_mu)))
    assert abs(float(np.mean(rattle_means)) - float(np.mean(gibbs_means))) < 0.5
    assert abs(float(np.mean(rattle_sds)) - float(np.mean(gibbs_sds))) < 0.5
