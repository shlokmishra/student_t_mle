import numpy as np
import jax.random as random

from diagnostics.cost_ledger import CostLedger
from models import loc_logistic_rattle


def _tangent_momentum(x, p, mu_star):
    grad = loc_logistic_rattle._constraint_grad(x, mu_star)
    return p - grad * float(np.dot(grad, p)) / float(np.dot(grad, grad))


def test_logistic_paper_fixed_direction_preserves_constraint_and_tangent_momentum():
    mu_star = 0.0
    mu = 0.1
    x0 = np.array([-0.5, 0.5, -1.0, 1.0])
    p0 = _tangent_momentum(x0, np.array([0.2, -0.1, 0.05, -0.03]), mu_star)
    x1, p1, ok, _ = loc_logistic_rattle._rattle_trajectory(
        x0,
        p0,
        mu,
        mu_star,
        step_size=0.005,
        num_steps=1,
        proj_tol=1e-12,
        proj_max_iters=25,
        grad_tol=1e-12,
        proj_damping=1.0,
        proj_line_search=True,
        proj_init_strategy="trial",
        include_gram_correction=True,
        projection_mode="paper_fixed_direction",
        tangent_tol=1e-9,
    )
    assert ok
    assert abs(loc_logistic_rattle._constraint_value(x1, mu_star)) < 1e-10
    assert abs(float(np.dot(loc_logistic_rattle._constraint_grad(x1, mu_star), p1))) < 1e-9


def test_logistic_reverse_position_check_passes_for_small_step():
    mu_star = 0.0
    mu = 0.1
    x0 = np.array([-0.5, 0.5, -1.0, 1.0])
    p0 = _tangent_momentum(x0, np.array([0.2, -0.1, 0.05, -0.03]), mu_star)
    kwargs = dict(
        mu=mu,
        mu_star=mu_star,
        step_size=0.005,
        num_steps=1,
        proj_tol=1e-12,
        proj_max_iters=25,
        grad_tol=1e-12,
        proj_damping=1.0,
        proj_line_search=True,
        proj_init_strategy="trial",
        include_gram_correction=True,
        projection_mode="paper_fixed_direction",
        tangent_tol=1e-9,
    )
    x1, p1, ok, _ = loc_logistic_rattle._rattle_trajectory(x0, p0, **kwargs)
    assert ok
    x_rev, _, ok_rev, _ = loc_logistic_rattle._rattle_trajectory(x1, -p1, **kwargs)
    assert ok_rev
    assert np.linalg.norm(x_rev - x0) < 1e-6


def test_logistic_gram_correction_changes_potential():
    x = np.array([-0.7, 0.3, -1.1, 1.5])
    mu = 0.2
    mu_star = 0.0
    u0 = loc_logistic_rattle._potential_energy(x, mu, mu_star=mu_star, include_gram_correction=False)
    u1 = loc_logistic_rattle._potential_energy(x, mu, mu_star=mu_star, include_gram_correction=True)
    assert np.isfinite(u0)
    assert np.isfinite(u1)
    assert not np.isclose(u0, u1)


def test_logistic_run_rattle_defaults_and_cost_ledger_metadata():
    params = {
        "n": 4,
        "num_iterations_T": 2,
        "proposal_std_mu": 0.1,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "rattle_step_size": 0.005,
        "rattle_num_steps": 1,
    }
    ledger = CostLedger(method="rattle", model="logistic", n=4, k=np.nan, mu_star=0.0, seed=0)
    out = loc_logistic_rattle.run_rattle(random.PRNGKey(0), 0.0, params, verbose=False, cost_ledger=ledger)
    diag = out["projection_diagnostics"]
    assert diag["include_gram_correction"] is True
    assert diag["projection_mode"] == "paper_fixed_direction"
    assert ledger.counters["projection_mode"] == "paper_fixed_direction"
    assert ledger.counters["gram_correction_enabled"] is True
