"""
Student-t location model baseline using a constrained HMC / RATTLE-style latent x update.

This keeps the current mu | x MH update comparable to models.loc_student while replacing
the pairwise local x kernel with a full-vector constrained proposal on the manifold

    c(x) = sum_i (x_i - mu_star) / (k + (x_i - mu_star)^2) = 0.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from tqdm import tqdm

from .loc_student import (
    get_mle,
    sample_data,
    get_benchmark_mle_samples,
    run_full_data_mh,
    _update_mu_mh,
)

EPS_U = 1e-12


def _constraint_value(x, mu_star, k):
    y = np.asarray(x) - mu_star
    return float(np.sum(y / (k + y * y)))


def _constraint_grad(x, mu_star, k):
    y = np.asarray(x) - mu_star
    denom = k + y * y
    return (k - y * y) / (denom * denom)


def _project_momentum(momentum, grad, grad_tol):
    grad = np.asarray(grad, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    denom = float(np.dot(grad, grad))
    if not np.isfinite(denom) or denom <= grad_tol:
        return momentum, False
    correction = float(np.dot(grad, momentum)) / denom
    return momentum - correction * grad, True


def _init_diag(x0, mu_star, k, proposals):
    residual0 = abs(_constraint_value(x0, mu_star, k))
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
    mean_sq = out["delta_h_sq_sum"] / n_dh
    out["delta_h_rms"] = float(np.sqrt(max(mean_sq, 0.0)))
    return out


def _project_to_manifold_newton(
    x,
    mu_star,
    k,
    tol,
    max_iters,
    grad_tol,
    damping=1.0,
    line_search=True,
    init_strategy="trial",
):
    x_trial = np.asarray(x, dtype=float).copy()
    residual = abs(_constraint_value(x_trial, mu_star, k))
    if residual <= tol:
        return x_trial, True, 0, residual

    if init_strategy == "linearized":
        grad0 = _constraint_grad(x_trial, mu_star, k)
        denom0 = float(np.dot(grad0, grad0))
        if np.isfinite(denom0) and denom0 > grad_tol:
            step0 = damping * _constraint_value(x_trial, mu_star, k) / denom0
            x_trial = x_trial - step0 * grad0
            residual = abs(_constraint_value(x_trial, mu_star, k))
            if residual <= tol:
                return x_trial, True, 1, residual

    x_cur = x_trial
    for it in range(1, max_iters + 1):
        grad = _constraint_grad(x_cur, mu_star, k)
        denom = float(np.dot(grad, grad))
        if not np.isfinite(denom) or denom <= grad_tol:
            return x_cur, False, it, residual

        raw_step = _constraint_value(x_cur, mu_star, k) / denom
        alpha = float(damping)
        x_next = x_cur - alpha * raw_step * grad
        next_residual = abs(_constraint_value(x_next, mu_star, k))

        if line_search:
            while alpha > 1e-6 and next_residual > residual:
                alpha *= 0.5
                x_next = x_cur - alpha * raw_step * grad
                next_residual = abs(_constraint_value(x_next, mu_star, k))

        x_cur = x_next
        residual = next_residual
        if residual <= tol:
            return x_cur, True, it, residual

    return x_cur, False, max_iters, residual


def _grad_potential(x, mu, k):
    y = np.asarray(x) - mu
    return (k + 1.0) * y / (k + y * y)


def _potential_energy(x, mu, k):
    y = np.asarray(x) - mu
    return float(0.5 * (k + 1.0) * np.sum(np.log1p((y * y) / k)))


def _hamiltonian(x, momentum, mu, k):
    kinetic = 0.5 * float(np.dot(momentum, momentum))
    return _potential_energy(x, mu, k) + kinetic


def _rattle_trajectory(
    x0,
    momentum0,
    mu,
    mu_star,
    k,
    step_size,
    num_steps,
    proj_tol,
    proj_max_iters,
    grad_tol,
    proj_damping,
    proj_line_search,
    proj_init_strategy,
):
    x = np.asarray(x0, dtype=float).copy()
    momentum = np.asarray(momentum0, dtype=float).copy()
    residual0 = abs(_constraint_value(x, mu_star, k))
    diag = {
        "position_projection_failures": 0,
        "momentum_projection_failures": 0,
        "projection_iterations": 0,
        "max_projection_residual": residual0,
        "sum_projection_residual": residual0,
        "num_projection_residuals": 1,
    }

    for _ in range(int(num_steps)):
        momentum = momentum - 0.5 * step_size * _grad_potential(x, mu, k)
        momentum, ok = _project_momentum(momentum, _constraint_grad(x, mu_star, k), grad_tol)
        if not ok:
            diag["momentum_projection_failures"] += 1
            return x, momentum, False, diag

        x_trial = x + step_size * momentum
        x, ok, n_iter, residual = _project_to_manifold_newton(
            x_trial,
            mu_star,
            k,
            proj_tol,
            proj_max_iters,
            grad_tol,
            damping=proj_damping,
            line_search=proj_line_search,
            init_strategy=proj_init_strategy,
        )
        diag["projection_iterations"] += n_iter
        diag["max_projection_residual"] = max(diag["max_projection_residual"], residual)
        diag["sum_projection_residual"] += residual
        diag["num_projection_residuals"] += 1
        if not ok:
            diag["position_projection_failures"] += 1
            return x, momentum, False, diag

        grad_new = _constraint_grad(x, mu_star, k)
        momentum, ok = _project_momentum(momentum, grad_new, grad_tol)
        if not ok:
            diag["momentum_projection_failures"] += 1
            return x, momentum, False, diag

        momentum = momentum - 0.5 * step_size * _grad_potential(x, mu, k)
        momentum, ok = _project_momentum(momentum, grad_new, grad_tol)
        if not ok:
            diag["momentum_projection_failures"] += 1
            return x, momentum, False, diag

    return x, momentum, True, diag


def run_rattle(key, mu_star, params, verbose=True):
    """
    Two-step sampler:
    1. mu | x via the existing MH update from models.loc_student
    2. x | mu, MLE(x)=mu_star via a constrained HMC / RATTLE-style proposal
    """
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    k = float(params["k"])
    step_size = float(params.get("rattle_step_size", 0.05))
    num_steps = int(params.get("rattle_num_steps", 2))
    proj_tol = float(params.get("rattle_proj_tol", 1e-10))
    proj_max_iters = int(params.get("rattle_proj_max_iters", 25))
    grad_tol = float(params.get("rattle_grad_tol", 1e-12))
    reverse_position_tol = float(params.get("rattle_reverse_position_tol", 5e-3))
    reverse_momentum_tol = float(params.get("rattle_reverse_momentum_tol", 5e-3))
    relaxed_position_tol = float(params.get("rattle_relaxed_position_tol", 5e-2))
    relaxed_momentum_tol = float(params.get("rattle_relaxed_momentum_tol", 5e-2))
    proj_damping = float(params.get("rattle_proj_damping", 1.0))
    proj_line_search = bool(params.get("rattle_proj_line_search", True))
    proj_init_strategy = str(params.get("rattle_proj_init_strategy", "trial"))

    mus = np.zeros(T + 1, dtype=float)
    xs = np.zeros((T + 1, n), dtype=float)
    x0 = np.ones(n, dtype=float) * float(mu_star)
    mus[0] = float(mu_star)
    xs[0, :] = x0

    mu_acc = 0
    x_acc = 0
    diag = _init_diag(x0, mu_star, k, T)

    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="RATTLE (Student)")

    for t in iters:
        key, key_mu, key_p, key_u = random.split(key, 4)
        x_cur = xs[t - 1]
        mu_cur = mus[t - 1]

        mu_new, acc_mu = _update_mu_mh(
            key_mu,
            jnp.asarray(mu_cur),
            jnp.asarray(x_cur),
            params["proposal_std_mu"],
            params["prior_mean"],
            params["prior_std"],
            k,
        )
        mu_new = float(mu_new)
        mus[t] = mu_new
        mu_acc += int(acc_mu)

        momentum0 = np.asarray(random.normal(key_p, shape=(n,)), dtype=float)
        momentum0, ok = _project_momentum(momentum0, _constraint_grad(x_cur, mu_star, k), grad_tol)
        if not ok:
            xs[t, :] = x_cur
            diag["initial_momentum_projection_failures"] += 1
            diag["projection_failure_count"] += 1
            residual_cur = abs(_constraint_value(x_cur, mu_star, k))
            diag["max_manifold_residual"] = max(diag["max_manifold_residual"], residual_cur)
            diag["sum_manifold_residual"] += residual_cur
            diag["num_manifold_residuals"] += 1
            continue

        h0 = _hamiltonian(x_cur, momentum0, mu_new, k)
        x_prop, momentum_prop, ok, traj_diag = _rattle_trajectory(
            x_cur,
            momentum0,
            mu_new,
            mu_star,
            k,
            step_size,
            num_steps,
            proj_tol,
            proj_max_iters,
            grad_tol,
            proj_damping,
            proj_line_search,
            proj_init_strategy,
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
            residual_cur = abs(_constraint_value(x_cur, mu_star, k))
            diag["max_manifold_residual"] = max(diag["max_manifold_residual"], residual_cur)
            diag["sum_manifold_residual"] += residual_cur
            diag["num_manifold_residuals"] += 1
            continue

        h1 = _hamiltonian(x_prop, momentum_prop, mu_new, k)
        delta_h = h1 - h0
        diag["delta_h_sum"] += delta_h
        diag["delta_h_abs_sum"] += abs(delta_h)
        diag["delta_h_sq_sum"] += delta_h * delta_h
        diag["delta_h_max_abs"] = max(diag["delta_h_max_abs"], abs(delta_h))
        diag["delta_h_count"] += 1

        x_rev, momentum_rev, ok_rev, rev_diag = _rattle_trajectory(
            x_prop,
            -momentum_prop,
            mu_new,
            mu_star,
            k,
            step_size,
            num_steps,
            proj_tol,
            proj_max_iters,
            grad_tol,
            proj_damping,
            proj_line_search,
            proj_init_strategy,
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
        reverse_ok = (
            ok_rev
            and reverse_position_error <= reverse_position_tol
            and reverse_momentum_error <= reverse_momentum_tol
        )
        if not reverse_ok:
            xs[t, :] = x_cur
            diag["reverse_failures"] += 1
            diag["reverse_check_failure_count"] += 1
            if not ok_rev:
                diag["projection_failure_count"] += 1
                diag["reverse_projection_solver_failure_count"] += 1
            else:
                pos_bad = reverse_position_error > reverse_position_tol
                mom_bad = reverse_momentum_error > reverse_momentum_tol
                relaxed_ok = (
                    reverse_position_error <= relaxed_position_tol
                    and reverse_momentum_error <= relaxed_momentum_tol
                )
                if relaxed_ok:
                    diag["reverse_tolerance_only_failure_count"] += 1
                else:
                    if pos_bad:
                        diag["reverse_position_mismatch_count"] += 1
                    if mom_bad:
                        diag["reverse_momentum_mismatch_count"] += 1
            residual_cur = abs(_constraint_value(x_cur, mu_star, k))
            diag["max_manifold_residual"] = max(diag["max_manifold_residual"], residual_cur)
            diag["sum_manifold_residual"] += residual_cur
            diag["num_manifold_residuals"] += 1
            continue

        log_alpha = min(0.0, -delta_h)
        accepted = np.log(float(random.uniform(key_u, minval=EPS_U, maxval=1.0))) < log_alpha
        if accepted:
            xs[t, :] = x_prop
            x_acc += 1
        else:
            xs[t, :] = x_cur

        residual_now = abs(_constraint_value(xs[t, :], mu_star, k))
        diag["max_manifold_residual"] = max(diag["max_manifold_residual"], residual_now)
        diag["sum_manifold_residual"] += residual_now
        diag["num_manifold_residuals"] += 1

    diag["x_acceptance_rate"] = x_acc / T

    return {
        "mu_chain": mus,
        "x_chain": xs,
        "mu_acceptance_rate": mu_acc / T,
        "x_acceptance_rate": x_acc / T,
        "projection_diagnostics": _finalize_diag(diag),
    }


def run_reversibility_test(
    key,
    mu_star,
    params,
    num_states=50,
    state_warmup=200,
    state_spacing=20,
    verbose=False,
):
    """
    Sample representative (x, mu) states from the Student RATTLE chain and test
    one forward/reverse constrained proposal from each with freshly sampled momentum.
    """
    states = sample_representative_states(
        key,
        mu_star,
        params,
        num_states=num_states,
        state_warmup=state_warmup,
        state_spacing=state_spacing,
        verbose=verbose,
    )
    return evaluate_reversibility_on_states(key, mu_star, params, states)


def sample_representative_states(
    key,
    mu_star,
    params,
    num_states=50,
    state_warmup=200,
    state_spacing=20,
    verbose=False,
):
    """Sample representative (x, mu) states from a short Student RATTLE chain."""
    params = dict(params)
    total_steps = int(state_warmup + state_spacing * max(num_states - 1, 0))
    params["num_iterations_T"] = max(total_steps, 1)
    chain = run_rattle(key, mu_star, params, verbose=verbose)

    start = min(state_warmup, chain["x_chain"].shape[0] - 1)
    indices = []
    idx = start
    while idx < chain["x_chain"].shape[0] and len(indices) < num_states:
        indices.append(idx)
        idx += state_spacing
    if not indices:
        indices = [chain["x_chain"].shape[0] - 1]

    states = []
    for idx in indices:
        states.append({
            "state_index": int(idx),
            "x": np.asarray(chain["x_chain"][idx], dtype=float),
            "mu": float(chain["mu_chain"][idx]),
        })
    return states


def evaluate_reversibility_on_states(key, mu_star, params, states):
    """Evaluate single-proposal reversibility diagnostics on a supplied state set."""
    params = dict(params)
    k = float(params["k"])
    step_size = float(params.get("rattle_step_size", 0.05))
    num_steps = int(params.get("rattle_num_steps", 2))
    proj_tol = float(params.get("rattle_proj_tol", 1e-10))
    proj_max_iters = int(params.get("rattle_proj_max_iters", 25))
    grad_tol = float(params.get("rattle_grad_tol", 1e-12))
    reverse_position_tol = float(params.get("rattle_reverse_position_tol", 5e-3))
    reverse_momentum_tol = float(params.get("rattle_reverse_momentum_tol", 5e-3))
    relaxed_position_tol = float(params.get("rattle_relaxed_position_tol", 5e-2))
    relaxed_momentum_tol = float(params.get("rattle_relaxed_momentum_tol", 5e-2))
    proj_damping = float(params.get("rattle_proj_damping", 1.0))
    proj_line_search = bool(params.get("rattle_proj_line_search", True))
    proj_init_strategy = str(params.get("rattle_proj_init_strategy", "trial"))

    results = []
    base_key = key
    for state in states:
        base_key, key_p = random.split(base_key)
        x_cur = np.asarray(state["x"], dtype=float)
        mu_cur = float(state["mu"])
        momentum0 = np.asarray(random.normal(key_p, shape=x_cur.shape), dtype=float)
        momentum0, ok0 = _project_momentum(momentum0, _constraint_grad(x_cur, mu_star, k), grad_tol)
        if not ok0:
            results.append({
                "state_index": int(state["state_index"]),
                "projection_solver_failure": True,
                "forward_ok": False,
                "reverse_ok": False,
                "position_error": float("nan"),
                "momentum_error": float("nan"),
                "tolerance_only_failure": False,
                "position_mismatch": False,
                "momentum_mismatch": False,
            })
            continue

        x_prop, p_prop, ok_fwd, _ = _rattle_trajectory(
            x_cur,
            momentum0,
            mu_cur,
            mu_star,
            k,
            step_size,
            num_steps,
            proj_tol,
            proj_max_iters,
            grad_tol,
            proj_damping,
            proj_line_search,
            proj_init_strategy,
        )
        if not ok_fwd:
            results.append({
                "state_index": int(state["state_index"]),
                "projection_solver_failure": True,
                "forward_ok": False,
                "reverse_ok": False,
                "position_error": float("nan"),
                "momentum_error": float("nan"),
                "tolerance_only_failure": False,
                "position_mismatch": False,
                "momentum_mismatch": False,
            })
            continue

        x_rev, p_rev, ok_rev, _ = _rattle_trajectory(
            x_prop,
            -p_prop,
            mu_cur,
            mu_star,
            k,
            step_size,
            num_steps,
            proj_tol,
            proj_max_iters,
            grad_tol,
            proj_damping,
            proj_line_search,
            proj_init_strategy,
        )
        pos_err = float(np.linalg.norm(x_rev - x_cur))
        mom_err = float(np.linalg.norm(p_rev + momentum0))
        pos_bad = pos_err > reverse_position_tol
        mom_bad = mom_err > reverse_momentum_tol
        projection_failure = not ok_rev
        tolerance_only = (
            ok_rev
            and (pos_bad or mom_bad)
            and pos_err <= relaxed_position_tol
            and mom_err <= relaxed_momentum_tol
        )
        results.append({
            "state_index": int(state["state_index"]),
            "projection_solver_failure": projection_failure,
            "forward_ok": True,
            "reverse_ok": bool(ok_rev and not pos_bad and not mom_bad),
            "position_error": pos_err,
            "momentum_error": mom_err,
            "tolerance_only_failure": bool(tolerance_only),
            "position_mismatch": bool(ok_rev and pos_bad and not tolerance_only),
            "momentum_mismatch": bool(ok_rev and mom_bad and not tolerance_only),
            "manifold_residual": abs(_constraint_value(x_prop, mu_star, k)),
        })

    def _mean(key_name):
        vals = [r[key_name] for r in results if np.isfinite(r[key_name])]
        return float(np.mean(vals)) if vals else float("nan")

    def _max(key_name):
        vals = [r[key_name] for r in results if np.isfinite(r[key_name])]
        return float(np.max(vals)) if vals else float("nan")

    summary = {
        "num_states": len(results),
        "projection_solver_failures": int(sum(r["projection_solver_failure"] for r in results)),
        "position_mismatch_failures": int(sum(r["position_mismatch"] for r in results)),
        "momentum_mismatch_failures": int(sum(r["momentum_mismatch"] for r in results)),
        "tolerance_only_failures": int(sum(r["tolerance_only_failure"] for r in results)),
        "mean_position_error": _mean("position_error"),
        "max_position_error": _max("position_error"),
        "mean_momentum_error": _mean("momentum_error"),
        "max_momentum_error": _max("momentum_error"),
    }
    return {"summary": summary, "results": results}
