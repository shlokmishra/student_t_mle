"""Logistic location constrained-HMC / RATTLE sampler.

The main mode mirrors the paper-style scalar-constraint RATTLE update used by
``models.loc_student_rattle``:

    c(x) = sum_i tanh((x_i - mu_star) / 2) = 0.

For M=I, the default position projection is along the fixed old-position
constraint gradient, with optional legacy normal-Newton projection retained as
an ablation.
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from tqdm import tqdm

from .loc_logistic import (
    get_benchmark_mle_samples,
    get_mle,
    run_full_data_mh,
    sample_data,
    _update_mu_mh,
    _initial_x,
)

EPS_U = 1e-12


def _constraint_value(x, mu_star, cost_ledger=None):
    value = float(np.tanh(0.5 * (np.asarray(x) - mu_star)).sum())
    if cost_ledger is not None:
        cost_ledger.inc("constraint_evals")
        cost_ledger.observe_constraint(value)
    return value


def _constraint_grad(x, mu_star, cost_ledger=None):
    z = np.tanh(0.5 * (np.asarray(x) - mu_star))
    if cost_ledger is not None:
        cost_ledger.inc("constraint_grad_evals")
    return 0.5 * (1.0 - z * z)


def _constraint_gram(x, mu_star, cost_ledger=None):
    if cost_ledger is not None:
        cost_ledger.inc("gram_evals")
    grad = _constraint_grad(x, mu_star, cost_ledger=cost_ledger)
    return float(np.dot(grad, grad))


def _constraint_grad_derivative(x, mu_star):
    z = np.tanh(0.5 * (np.asarray(x) - mu_star))
    grad = 0.5 * (1.0 - z * z)
    return -z * grad


def _grad_log_gram_half(x, mu_star, cost_ledger=None):
    if cost_ledger is not None:
        cost_ledger.inc("gram_grad_evals")
    grad = _constraint_grad(x, mu_star, cost_ledger=cost_ledger)
    grad_deriv = _constraint_grad_derivative(x, mu_star)
    gram = float(np.dot(grad, grad))
    if not np.isfinite(gram) or gram <= 0.0:
        return np.zeros_like(grad)
    return grad * grad_deriv / gram


def _project_momentum(momentum, grad, grad_tol, cost_ledger=None):
    if cost_ledger is not None:
        cost_ledger.inc("projection_evals")
        cost_ledger.inc("momentum_projections")
    grad = np.asarray(grad, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    denom = float(np.dot(grad, grad))
    if not np.isfinite(denom) or denom <= grad_tol:
        return momentum, False
    correction = float(np.dot(grad, momentum)) / denom
    return momentum - correction * grad, True


def _init_diag(x0, mu_star, proposals, cost_ledger=None):
    residual0 = abs(_constraint_value(x0, mu_star, cost_ledger=cost_ledger))
    return {
        "proposals": int(proposals),
        "forward_failures": 0,
        "reverse_failures": 0,
        "initial_momentum_projection_failures": 0,
        "position_projection_failures": 0,
        "momentum_projection_failures": 0,
        "projection_failure_count": 0,
        "reverse_check_failure_count": 0,
        "reverse_projection_solver_failure_count": 0,
        "reverse_position_mismatch_count": 0,
        "reverse_momentum_mismatch_count": 0,
        "reverse_tolerance_only_failure_count": 0,
        "projection_iterations_total": 0,
        "max_projection_residual": residual0,
        "sum_projection_residual": 0.0,
        "num_projection_residuals": 0,
        "max_manifold_residual": residual0,
        "sum_manifold_residual": residual0,
        "num_manifold_residuals": 1,
        "max_reverse_position_error": 0.0,
        "max_reverse_momentum_error": 0.0,
        "sum_reverse_position_error": 0.0,
        "sum_reverse_momentum_error": 0.0,
        "num_reverse_checks": 0,
        "delta_h_sum": 0.0,
        "delta_h_abs_sum": 0.0,
        "delta_h_sq_sum": 0.0,
        "delta_h_max_abs": 0.0,
        "delta_h_count": 0,
        "x_acceptance_rate": 0.0,
    }


def _finalize_diag(diag):
    out = dict(diag)
    n_proj = max(int(out["num_projection_residuals"]), 1)
    n_manifold = max(int(out["num_manifold_residuals"]), 1)
    n_rev = max(int(out["num_reverse_checks"]), 1)
    n_dh = max(int(out["delta_h_count"]), 1)
    out["mean_projection_residual"] = out["sum_projection_residual"] / n_proj
    out["mean_manifold_residual"] = out["sum_manifold_residual"] / n_manifold
    out["mean_reverse_position_error"] = out["sum_reverse_position_error"] / n_rev
    out["mean_reverse_momentum_error"] = out["sum_reverse_momentum_error"] / n_rev
    out["delta_h_mean"] = out["delta_h_sum"] / n_dh
    out["delta_h_mean_abs"] = out["delta_h_abs_sum"] / n_dh
    out["delta_h_rms"] = float(np.sqrt(max(out["delta_h_sq_sum"] / n_dh, 0.0)))
    return out


def _project_to_manifold_newton(
    x,
    mu_star,
    tol,
    max_iters,
    grad_tol,
    damping=1.0,
    line_search=True,
    init_strategy="trial",
    cost_ledger=None,
):
    if cost_ledger is not None:
        cost_ledger.inc("projection_evals")
    x_trial = np.asarray(x, dtype=float).copy()
    residual = abs(_constraint_value(x_trial, mu_star, cost_ledger=cost_ledger))
    if residual <= tol:
        return x_trial, True, 0, residual

    if init_strategy == "linearized":
        grad0 = _constraint_grad(x_trial, mu_star, cost_ledger=cost_ledger)
        denom0 = float(np.dot(grad0, grad0))
        if np.isfinite(denom0) and denom0 > grad_tol:
            x_trial = x_trial - damping * _constraint_value(x_trial, mu_star, cost_ledger=cost_ledger) * grad0 / denom0
            residual = abs(_constraint_value(x_trial, mu_star, cost_ledger=cost_ledger))
            if residual <= tol:
                return x_trial, True, 1, residual

    x_cur = x_trial
    for it in range(1, max_iters + 1):
        grad = _constraint_grad(x_cur, mu_star, cost_ledger=cost_ledger)
        denom = float(np.dot(grad, grad))
        if not np.isfinite(denom) or denom <= grad_tol:
            return x_cur, False, it, residual
        raw_step = _constraint_value(x_cur, mu_star, cost_ledger=cost_ledger) / denom
        alpha = float(damping)
        x_next = x_cur - alpha * raw_step * grad
        next_residual = abs(_constraint_value(x_next, mu_star, cost_ledger=cost_ledger))
        if line_search:
            while alpha > 1e-6 and next_residual > residual:
                alpha *= 0.5
                x_next = x_cur - alpha * raw_step * grad
                next_residual = abs(_constraint_value(x_next, mu_star, cost_ledger=cost_ledger))
        x_cur = x_next
        residual = next_residual
        if residual <= tol:
            return x_cur, True, it, residual
    return x_cur, False, max_iters, residual


def _project_to_manifold_fixed_direction(
    x_tilde,
    direction,
    mu_star,
    tol,
    max_iters,
    grad_tol,
    damping=1.0,
    line_search=True,
    cost_ledger=None,
):
    if cost_ledger is not None:
        cost_ledger.inc("projection_evals")
    x_tilde = np.asarray(x_tilde, dtype=float).copy()
    direction = np.asarray(direction, dtype=float).copy()
    if not np.all(np.isfinite(direction)) or float(np.dot(direction, direction)) <= grad_tol:
        residual = abs(_constraint_value(x_tilde, mu_star, cost_ledger=cost_ledger))
        return x_tilde, False, 0, residual, 0.0

    theta = 0.0
    x_cur = x_tilde
    residual = abs(_constraint_value(x_cur, mu_star, cost_ledger=cost_ledger))
    if residual <= tol:
        return x_cur, True, 0, residual, theta

    for it in range(1, max_iters + 1):
        c_val = _constraint_value(x_cur, mu_star, cost_ledger=cost_ledger)
        grad = _constraint_grad(x_cur, mu_star, cost_ledger=cost_ledger)
        deriv = float(np.dot(grad, direction))
        if not np.isfinite(deriv) or abs(deriv) <= grad_tol:
            return x_cur, False, it, residual, theta
        alpha = float(damping)
        theta_next = theta - alpha * c_val / deriv
        x_next = x_tilde + theta_next * direction
        next_residual = abs(_constraint_value(x_next, mu_star, cost_ledger=cost_ledger))
        if line_search:
            while alpha > 1e-6 and next_residual > residual:
                alpha *= 0.5
                theta_next = theta - alpha * c_val / deriv
                x_next = x_tilde + theta_next * direction
                next_residual = abs(_constraint_value(x_next, mu_star, cost_ledger=cost_ledger))
        theta = theta_next
        x_cur = x_next
        residual = next_residual
        if residual <= tol:
            return x_cur, True, it, residual, theta
    return x_cur, False, max_iters, residual, theta


def _grad_potential(x, mu, mu_star=None, include_gram_correction=False, cost_ledger=None):
    grad = np.tanh(0.5 * (np.asarray(x) - mu))
    if cost_ledger is not None:
        cost_ledger.inc("gradient_evals")
    if include_gram_correction:
        if mu_star is None:
            raise ValueError("mu_star is required when include_gram_correction=True")
        grad = grad + _grad_log_gram_half(x, mu_star, cost_ledger=cost_ledger)
    return grad


def _potential_energy(x, mu, mu_star=None, include_gram_correction=False, cost_ledger=None):
    t = np.asarray(x) - mu
    energy = float(np.sum(t + 2.0 * np.logaddexp(0.0, -t)))
    if cost_ledger is not None:
        cost_ledger.inc("potential_evals")
    if include_gram_correction:
        if mu_star is None:
            raise ValueError("mu_star is required when include_gram_correction=True")
        energy += 0.5 * np.log(max(_constraint_gram(x, mu_star, cost_ledger=cost_ledger), 1e-300))
    return energy


def _hamiltonian(x, momentum, mu, mu_star=None, include_gram_correction=False, cost_ledger=None):
    if cost_ledger is not None:
        cost_ledger.inc("energy_evals")
    return _potential_energy(x, mu, mu_star, include_gram_correction, cost_ledger=cost_ledger) + 0.5 * float(np.dot(momentum, momentum))


def _rattle_trajectory(
    x0,
    momentum0,
    mu,
    mu_star,
    step_size,
    num_steps,
    proj_tol,
    proj_max_iters,
    grad_tol,
    proj_damping,
    proj_line_search,
    proj_init_strategy,
    include_gram_correction=False,
    cost_ledger=None,
    trajectory_kind="forward",
    projection_mode="paper_fixed_direction",
    tangent_tol=1e-8,
):
    x = np.asarray(x0, dtype=float).copy()
    momentum = np.asarray(momentum0, dtype=float).copy()
    residual0 = abs(_constraint_value(x, mu_star, cost_ledger=cost_ledger))
    diag = {
        "position_projection_failures": 0,
        "momentum_projection_failures": 0,
        "projection_iterations": 0,
        "max_projection_residual": residual0,
        "sum_projection_residual": residual0,
        "num_projection_residuals": 1,
    }

    for _ in range(int(num_steps)):
        if cost_ledger is not None:
            cost_ledger.inc("leapfrog_steps")
        if projection_mode == "normal_newton_legacy":
            momentum = momentum - 0.5 * step_size * _grad_potential(
                x, mu, mu_star=mu_star, include_gram_correction=include_gram_correction, cost_ledger=cost_ledger
            )
            momentum, ok = _project_momentum(momentum, _constraint_grad(x, mu_star, cost_ledger=cost_ledger), grad_tol, cost_ledger=cost_ledger)
            if not ok:
                diag["momentum_projection_failures"] += 1
                return x, momentum, False, diag
            x_trial = x + step_size * momentum
            x, ok, n_iter, residual = _project_to_manifold_newton(
                x_trial,
                mu_star,
                proj_tol,
                proj_max_iters,
                grad_tol,
                damping=proj_damping,
                line_search=proj_line_search,
                init_strategy=proj_init_strategy,
                cost_ledger=cost_ledger,
            )
            if cost_ledger is not None:
                cost_ledger.inc("reverse_newton_iters" if trajectory_kind == "reverse" else "forward_newton_iters", n_iter)
                cost_ledger.inc("position_projection_newton_iters", n_iter)
            diag["projection_iterations"] += n_iter
            diag["max_projection_residual"] = max(diag["max_projection_residual"], residual)
            diag["sum_projection_residual"] += residual
            diag["num_projection_residuals"] += 1
            if not ok:
                diag["position_projection_failures"] += 1
                if cost_ledger is not None:
                    cost_ledger.inc("position_projection_failures")
                return x, momentum, False, diag
            grad_new = _constraint_grad(x, mu_star, cost_ledger=cost_ledger)
            momentum, ok = _project_momentum(momentum, grad_new, grad_tol, cost_ledger=cost_ledger)
            if not ok:
                diag["momentum_projection_failures"] += 1
                return x, momentum, False, diag
            momentum = momentum - 0.5 * step_size * _grad_potential(
                x, mu, mu_star=mu_star, include_gram_correction=include_gram_correction, cost_ledger=cost_ledger
            )
            momentum, ok = _project_momentum(momentum, grad_new, grad_tol, cost_ledger=cost_ledger)
            if not ok:
                diag["momentum_projection_failures"] += 1
                return x, momentum, False, diag
        elif projection_mode == "paper_fixed_direction":
            grad_old = _constraint_grad(x, mu_star, cost_ledger=cost_ledger)
            p_half_uncon = momentum - 0.5 * step_size * _grad_potential(
                x, mu, mu_star=mu_star, include_gram_correction=include_gram_correction, cost_ledger=cost_ledger
            )
            x_tilde = x + step_size * p_half_uncon
            x, ok, n_iter, residual, theta = _project_to_manifold_fixed_direction(
                x_tilde,
                grad_old,
                mu_star,
                proj_tol,
                proj_max_iters,
                grad_tol,
                damping=proj_damping,
                line_search=proj_line_search,
                cost_ledger=cost_ledger,
            )
            if cost_ledger is not None:
                cost_ledger.inc("reverse_newton_iters" if trajectory_kind == "reverse" else "forward_newton_iters", n_iter)
                cost_ledger.inc("position_projection_newton_iters", n_iter)
            diag["projection_iterations"] += n_iter
            diag["max_projection_residual"] = max(diag["max_projection_residual"], residual)
            diag["sum_projection_residual"] += residual
            diag["num_projection_residuals"] += 1
            if not ok:
                diag["position_projection_failures"] += 1
                if cost_ledger is not None:
                    cost_ledger.inc("position_projection_failures")
                return x, momentum, False, diag
            p_half = p_half_uncon + (theta / step_size) * grad_old
            p_tilde = p_half - 0.5 * step_size * _grad_potential(
                x, mu, mu_star=mu_star, include_gram_correction=include_gram_correction, cost_ledger=cost_ledger
            )
            grad_new = _constraint_grad(x, mu_star, cost_ledger=cost_ledger)
            momentum, ok = _project_momentum(p_tilde, grad_new, grad_tol, cost_ledger=cost_ledger)
            tangent_abs = abs(float(np.dot(grad_new, momentum))) if ok else np.inf
            residual = abs(_constraint_value(x, mu_star, cost_ledger=cost_ledger))
            diag["max_projection_residual"] = max(diag["max_projection_residual"], residual)
            diag["sum_projection_residual"] += residual
            diag["num_projection_residuals"] += 1
            if not ok or residual > proj_tol or tangent_abs > tangent_tol:
                if residual > proj_tol:
                    diag["position_projection_failures"] += 1
                    if cost_ledger is not None:
                        cost_ledger.inc("position_projection_failures")
                else:
                    diag["momentum_projection_failures"] += 1
                return x, momentum, False, diag
        else:
            raise ValueError(f"Unknown rattle_projection_mode: {projection_mode}")
    return x, momentum, True, diag


def run_rattle(key, mu_star, params, verbose=True, cost_ledger=None):
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    step_size = float(params.get("rattle_step_size", 0.05))
    num_steps = int(params.get("rattle_num_steps", 2))
    proj_tol = float(params.get("rattle_proj_tol", 1e-10))
    proj_max_iters = int(params.get("rattle_proj_max_iters", 25))
    grad_tol = float(params.get("rattle_grad_tol", 1e-12))
    reverse_position_tol = float(params.get("rattle_reverse_position_tol", 5e-3))
    reverse_momentum_tol = float(params.get("rattle_reverse_momentum_tol", 5e-3))
    reverse_check = bool(params.get("reverse_check", True))
    relaxed_position_tol = float(params.get("rattle_relaxed_position_tol", 5e-2))
    relaxed_momentum_tol = float(params.get("rattle_relaxed_momentum_tol", 5e-2))
    proj_damping = float(params.get("rattle_proj_damping", 1.0))
    proj_line_search = bool(params.get("rattle_proj_line_search", True))
    proj_init_strategy = str(params.get("rattle_proj_init_strategy", "trial"))
    include_gram_correction = bool(params.get("rattle_include_gram_correction", True))
    projection_mode = str(params.get("rattle_projection_mode", "paper_fixed_direction"))
    if projection_mode not in {"paper_fixed_direction", "normal_newton_legacy"}:
        raise ValueError(f"Unknown rattle_projection_mode: {projection_mode}")
    tangent_tol = float(params.get("rattle_tangent_tol", 1e-8))
    reverse_check_momentum = bool(params.get("rattle_reverse_check_momentum", projection_mode == "normal_newton_legacy"))
    if cost_ledger is not None:
        cost_ledger.set("projection_mode", projection_mode)
        cost_ledger.set("gram_correction_enabled", include_gram_correction)

    mus = np.zeros(T + 1, dtype=float)
    xs = np.zeros((T + 1, n), dtype=float)
    x0 = np.asarray(_initial_x(mu_star, n, params), dtype=float)
    mus[0] = float(mu_star)
    xs[0, :] = x0
    mu_acc = 0
    x_acc = 0
    diag = _init_diag(x0, mu_star, T, cost_ledger=cost_ledger)

    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="RATTLE (Logistic)")

    for t in iters:
        key, key_mu, key_p, key_u = random.split(key, 4)
        x_cur = xs[t - 1]
        mu_cur = mus[t - 1]
        if cost_ledger is not None:
            cost_ledger.inc("iterations")
            cost_ledger.inc("mu_mh_proposals")
            cost_ledger.inc("prior_logpdf_evals", 2)
            cost_ledger.inc("hmc_proposals")

        mu_new, acc_mu = _update_mu_mh(
            key_mu,
            jnp.asarray(mu_cur),
            jnp.asarray(x_cur),
            params["proposal_std_mu"],
            params["prior_mean"],
            params["prior_std"],
        )
        mu_new = float(mu_new)
        mus[t] = mu_new
        mu_acc += int(acc_mu)
        if cost_ledger is not None:
            cost_ledger.inc("mu_mh_accepts", int(acc_mu))

        momentum0 = np.asarray(random.normal(key_p, shape=(n,)), dtype=float)
        momentum0, ok = _project_momentum(
            momentum0,
            _constraint_grad(x_cur, mu_star, cost_ledger=cost_ledger),
            grad_tol,
            cost_ledger=cost_ledger,
        )
        if not ok:
            xs[t, :] = x_cur
            diag["initial_momentum_projection_failures"] += 1
            diag["projection_failure_count"] += 1
            if cost_ledger is not None:
                cost_ledger.inc("projection_failures")
                cost_ledger.inc("integration_failures")
            continue

        h0 = _hamiltonian(
            x_cur,
            momentum0,
            mu_new,
            mu_star=mu_star,
            include_gram_correction=include_gram_correction,
            cost_ledger=cost_ledger,
        )
        x_prop, momentum_prop, ok, traj_diag = _rattle_trajectory(
            x_cur,
            momentum0,
            mu_new,
            mu_star,
            step_size,
            num_steps,
            proj_tol,
            proj_max_iters,
            grad_tol,
            proj_damping,
            proj_line_search,
            proj_init_strategy,
            include_gram_correction=include_gram_correction,
            cost_ledger=cost_ledger,
            trajectory_kind="forward",
            projection_mode=projection_mode,
            tangent_tol=tangent_tol,
        )
        diag["position_projection_failures"] += traj_diag["position_projection_failures"]
        diag["momentum_projection_failures"] += traj_diag["momentum_projection_failures"]
        diag["projection_iterations_total"] += traj_diag["projection_iterations"]
        diag["max_projection_residual"] = max(diag["max_projection_residual"], traj_diag["max_projection_residual"])
        diag["sum_projection_residual"] += traj_diag["sum_projection_residual"]
        diag["num_projection_residuals"] += traj_diag["num_projection_residuals"]
        if not ok:
            xs[t, :] = x_cur
            diag["forward_failures"] += 1
            diag["projection_failure_count"] += 1
            if cost_ledger is not None:
                cost_ledger.inc("projection_failures")
                cost_ledger.inc("integration_failures")
            continue

        h1 = _hamiltonian(
            x_prop,
            momentum_prop,
            mu_new,
            mu_star=mu_star,
            include_gram_correction=include_gram_correction,
            cost_ledger=cost_ledger,
        )
        delta_h = h1 - h0
        diag["delta_h_sum"] += delta_h
        diag["delta_h_abs_sum"] += abs(delta_h)
        diag["delta_h_sq_sum"] += delta_h * delta_h
        diag["delta_h_max_abs"] = max(diag["delta_h_max_abs"], abs(delta_h))
        diag["delta_h_count"] += 1

        if reverse_check:
            if cost_ledger is not None:
                cost_ledger.inc("reverse_check_attempts")
            x_rev, momentum_rev, ok_rev, rev_diag = _rattle_trajectory(
                x_prop,
                -momentum_prop,
                mu_new,
                mu_star,
                step_size,
                num_steps,
                proj_tol,
                proj_max_iters,
                grad_tol,
                proj_damping,
                proj_line_search,
                proj_init_strategy,
                include_gram_correction=include_gram_correction,
                cost_ledger=cost_ledger,
                trajectory_kind="reverse",
                projection_mode=projection_mode,
                tangent_tol=tangent_tol,
            )
            diag["position_projection_failures"] += rev_diag["position_projection_failures"]
            diag["momentum_projection_failures"] += rev_diag["momentum_projection_failures"]
            diag["projection_iterations_total"] += rev_diag["projection_iterations"]
            diag["max_projection_residual"] = max(diag["max_projection_residual"], rev_diag["max_projection_residual"])
            diag["sum_projection_residual"] += rev_diag["sum_projection_residual"]
            diag["num_projection_residuals"] += rev_diag["num_projection_residuals"]
            reverse_position_error = float(np.linalg.norm(x_rev - x_cur))
            reverse_momentum_error = float(np.linalg.norm(momentum_rev + momentum0))
            diag["max_reverse_position_error"] = max(diag["max_reverse_position_error"], reverse_position_error)
            diag["max_reverse_momentum_error"] = max(diag["max_reverse_momentum_error"], reverse_momentum_error)
            diag["sum_reverse_position_error"] += reverse_position_error
            diag["sum_reverse_momentum_error"] += reverse_momentum_error
            diag["num_reverse_checks"] += 1
            if cost_ledger is not None:
                cost_ledger.set("reverse_position_error", max(float(cost_ledger.counters.get("reverse_position_error", 0.0)), reverse_position_error))
                cost_ledger.set("reverse_momentum_error", max(float(cost_ledger.counters.get("reverse_momentum_error", 0.0)), reverse_momentum_error))
            reverse_ok = ok_rev and reverse_position_error <= reverse_position_tol
            if reverse_check_momentum:
                reverse_ok = reverse_ok and reverse_momentum_error <= reverse_momentum_tol
        else:
            ok_rev = True
            reverse_position_error = 0.0
            reverse_momentum_error = 0.0
            reverse_ok = True

        if not reverse_ok:
            xs[t, :] = x_cur
            diag["reverse_failures"] += 1
            diag["reverse_check_failure_count"] += 1
            if cost_ledger is not None:
                cost_ledger.inc("reverse_check_failures")
            if not ok_rev:
                diag["projection_failure_count"] += 1
                diag["reverse_projection_solver_failure_count"] += 1
                if cost_ledger is not None:
                    cost_ledger.inc("projection_failures")
            else:
                pos_bad = reverse_position_error > reverse_position_tol
                mom_bad = reverse_momentum_error > reverse_momentum_tol
                relaxed_ok = reverse_position_error <= relaxed_position_tol and reverse_momentum_error <= relaxed_momentum_tol
                if relaxed_ok:
                    diag["reverse_tolerance_only_failure_count"] += 1
                else:
                    if pos_bad:
                        diag["reverse_position_mismatch_count"] += 1
                    if mom_bad:
                        diag["reverse_momentum_mismatch_count"] += 1
            continue

        accepted = np.log(float(random.uniform(key_u, minval=EPS_U, maxval=1.0))) < min(0.0, -delta_h)
        if accepted:
            xs[t, :] = x_prop
            x_acc += 1
            if cost_ledger is not None:
                cost_ledger.inc("hmc_accepts")
        else:
            xs[t, :] = x_cur
            if cost_ledger is not None:
                cost_ledger.inc("metropolized_rejections")

        residual_now = abs(_constraint_value(xs[t, :], mu_star, cost_ledger=cost_ledger))
        diag["max_manifold_residual"] = max(diag["max_manifold_residual"], residual_now)
        diag["sum_manifold_residual"] += residual_now
        diag["num_manifold_residuals"] += 1

    diag["x_acceptance_rate"] = x_acc / T
    diag["include_gram_correction"] = include_gram_correction
    diag["projection_mode"] = projection_mode
    diag["reverse_check"] = reverse_check
    if cost_ledger is not None:
        cost_ledger.set("iterations", T)
        cost_ledger.set("mu_mh_accepts", mu_acc)
        cost_ledger.set("hmc_accepts", x_acc)
        cost_ledger.set("projection_mode", projection_mode)
        cost_ledger.set("gram_correction_enabled", include_gram_correction)
        cost_ledger.update_from_projection_diag(_finalize_diag(diag))

    return {
        "mu_chain": mus,
        "x_chain": xs,
        "mu_acceptance_rate": mu_acc / T,
        "x_acceptance_rate": x_acc / T,
        "projection_diagnostics": _finalize_diag(diag),
    }


def sample_representative_states(key, mu_star, params, num_states=50, state_warmup=200, state_spacing=20, verbose=False):
    params = dict(params)
    params["num_iterations_T"] = max(int(state_warmup + state_spacing * max(num_states - 1, 0)), 1)
    chain = run_rattle(key, mu_star, params, verbose=verbose)
    start = min(state_warmup, chain["x_chain"].shape[0] - 1)
    indices = []
    idx = start
    while idx < chain["x_chain"].shape[0] and len(indices) < num_states:
        indices.append(idx)
        idx += state_spacing
    if not indices:
        indices = [chain["x_chain"].shape[0] - 1]
    return [
        {"state_index": int(idx), "x": np.asarray(chain["x_chain"][idx], dtype=float), "mu": float(chain["mu_chain"][idx])}
        for idx in indices
    ]


def evaluate_reversibility_on_states(key, mu_star, params, states):
    params = dict(params)
    step_size = float(params.get("rattle_step_size", 0.05))
    num_steps = int(params.get("rattle_num_steps", 2))
    proj_tol = float(params.get("rattle_proj_tol", 1e-10))
    proj_max_iters = int(params.get("rattle_proj_max_iters", 25))
    grad_tol = float(params.get("rattle_grad_tol", 1e-12))
    reverse_position_tol = float(params.get("rattle_reverse_position_tol", 5e-3))
    proj_damping = float(params.get("rattle_proj_damping", 1.0))
    proj_line_search = bool(params.get("rattle_proj_line_search", True))
    proj_init_strategy = str(params.get("rattle_proj_init_strategy", "trial"))
    include_gram_correction = bool(params.get("rattle_include_gram_correction", True))
    projection_mode = str(params.get("rattle_projection_mode", "paper_fixed_direction"))
    results = []
    base_key = key
    for state in states:
        base_key, key_p = random.split(base_key)
        x_cur = np.asarray(state["x"], dtype=float)
        mu_cur = float(state["mu"])
        p0 = np.asarray(random.normal(key_p, shape=x_cur.shape), dtype=float)
        p0, ok0 = _project_momentum(p0, _constraint_grad(x_cur, mu_star), grad_tol)
        if not ok0:
            results.append({"state_index": int(state["state_index"]), "projection_solver_failure": True})
            continue
        x_prop, p_prop, ok_fwd, _ = _rattle_trajectory(
            x_cur, p0, mu_cur, mu_star, step_size, num_steps, proj_tol, proj_max_iters,
            grad_tol, proj_damping, proj_line_search, proj_init_strategy,
            include_gram_correction=include_gram_correction, projection_mode=projection_mode,
        )
        x_rev, p_rev, ok_rev, _ = _rattle_trajectory(
            x_prop, -p_prop, mu_cur, mu_star, step_size, num_steps, proj_tol, proj_max_iters,
            grad_tol, proj_damping, proj_line_search, proj_init_strategy,
            include_gram_correction=include_gram_correction, projection_mode=projection_mode,
            trajectory_kind="reverse",
        )
        pos_err = float(np.linalg.norm(x_rev - x_cur))
        mom_err = float(np.linalg.norm(p_rev + p0))
        results.append(
            {
                "state_index": int(state["state_index"]),
                "projection_solver_failure": not (ok_fwd and ok_rev),
                "forward_ok": bool(ok_fwd),
                "reverse_ok": bool(ok_rev and pos_err <= reverse_position_tol),
                "position_error": pos_err,
                "momentum_error": mom_err,
                "manifold_residual": abs(_constraint_value(x_prop, mu_star)),
            }
        )
    finite_pos = [r["position_error"] for r in results if "position_error" in r and np.isfinite(r["position_error"])]
    finite_mom = [r["momentum_error"] for r in results if "momentum_error" in r and np.isfinite(r["momentum_error"])]
    return {
        "summary": {
            "num_states": len(results),
            "projection_solver_failures": int(sum(r.get("projection_solver_failure", False) for r in results)),
            "mean_position_error": float(np.mean(finite_pos)) if finite_pos else np.nan,
            "max_position_error": float(np.max(finite_pos)) if finite_pos else np.nan,
            "mean_momentum_error": float(np.mean(finite_mom)) if finite_mom else np.nan,
            "max_momentum_error": float(np.max(finite_mom)) if finite_mom else np.nan,
        },
        "results": results,
    }


def run_reversibility_test(key, mu_star, params, num_states=50, state_warmup=200, state_spacing=20, verbose=False):
    states = sample_representative_states(
        key, mu_star, params, num_states=num_states, state_warmup=state_warmup, state_spacing=state_spacing, verbose=verbose
    )
    return evaluate_reversibility_on_states(key, mu_star, params, states)
