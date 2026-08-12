"""Experimental odd-n Laplace fixed-facet HMC/RATTLE sampler.

This is not a smooth-manifold RATTLE implementation.  It keeps a labelled
median coordinate fixed at mu_star and runs exact nonsmooth HMC on that facet,
rejecting proposals that leave the fixed left/right median inequalities.
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from tqdm import tqdm

from .loc_laplace import (
    get_benchmark_mle_samples,
    get_mle,
    run_full_data_mh,
    sample_data,
    _initial_x,
    _update_mu_mh,
)

EPS_U = 1e-12
DIAG_GROUPS = ("clean", "kink_only", "boundary_only", "kink_and_boundary")


def _sign0(value):
    value = np.asarray(value)
    return np.where(value > 0.0, 1.0, np.where(value < 0.0, -1.0, 0.0))


def _median_index(n: int) -> int:
    n = int(n)
    if n % 2 == 0:
        raise ValueError("Laplace facet-RATTLE supports only odd n")
    return n // 2


def _side_counts(x, mu_star):
    x = np.asarray(x, dtype=float)
    left = int(np.sum(x < float(mu_star)))
    equal = int(np.sum(x == float(mu_star)))
    right = int(np.sum(x > float(mu_star)))
    return left, equal, right


def _side_boundary_violated(x, mu_star, median_idx):
    x = np.asarray(x, dtype=float)
    mu_star = float(mu_star)
    median_idx = int(median_idx)
    return (
        abs(float(x[median_idx]) - mu_star) > 0.0
        or bool(np.any(x[:median_idx] >= mu_star))
        or bool(np.any(x[median_idx + 1 :] <= mu_star))
    )


def _side_boundary_crossed_between(x_prev, x_new, mu_star, median_idx):
    x_prev = np.asarray(x_prev, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    mu_star = float(mu_star)
    median_idx = int(median_idx)
    left_prev = x_prev[:median_idx] - mu_star
    left_new = x_new[:median_idx] - mu_star
    right_prev = x_prev[median_idx + 1 :] - mu_star
    right_new = x_new[median_idx + 1 :] - mu_star
    left_cross = (left_prev < 0.0) & (left_new >= 0.0)
    right_cross = (right_prev > 0.0) & (right_new <= 0.0)
    return bool(np.any(left_cross) or np.any(right_cross)), int(np.sum(left_cross) + np.sum(right_cross))


def _side_count_failed(x, mu_star, median_idx):
    left, equal, right = _side_counts(x, mu_star)
    half = int(median_idx)
    return left != half or equal != 1 or right != half


def _laplace_potential(x, mu, b, cost_ledger=None):
    if cost_ledger is not None:
        cost_ledger.inc("potential_evals")
    return float(np.sum(np.abs(np.asarray(x, dtype=float) - float(mu))) / float(b))


def _grad_potential(x, mu, b, median_idx=None, cost_ledger=None):
    if cost_ledger is not None:
        cost_ledger.inc("gradient_evals")
    grad = _sign0(np.asarray(x, dtype=float) - float(mu)) / float(b)
    if median_idx is not None:
        grad[int(median_idx)] = 0.0
    return np.asarray(grad, dtype=float)


def _project_facet(x, momentum, mu_star, median_idx):
    x = np.asarray(x, dtype=float).copy()
    momentum = np.asarray(momentum, dtype=float).copy()
    x[int(median_idx)] = float(mu_star)
    momentum[int(median_idx)] = 0.0
    return x, momentum


def _hamiltonian(x, momentum, mu, b, median_idx, cost_ledger=None):
    if cost_ledger is not None:
        cost_ledger.inc("energy_evals")
    p = np.asarray(momentum, dtype=float).copy()
    p[int(median_idx)] = 0.0
    return _laplace_potential(x, mu, b, cost_ledger=cost_ledger) + 0.5 * float(np.dot(p, p))


def _init_diag(x0, mu_star, median_idx, proposals):
    side_fail = int(_side_count_failed(x0, mu_star, median_idx))
    diag = {
        "proposals": int(proposals),
        "forward_failures": 0,
        "reverse_failures": 0,
        "side_boundary_violation_count": 0,
        "side_boundary_cross_count": 0,
        "side_boundary_cross_coordinate_count": 0,
        "side_count_failures": side_fail,
        "kink_cross_count": 0,
        "near_kink_count": 0,
        "crossed_proposals": 0,
        "not_crossed_proposals": 0,
        "accepted_crossed": 0,
        "accepted_not_crossed": 0,
        "delta_h_crossed_sum": 0.0,
        "delta_h_not_crossed_sum": 0.0,
        "delta_h_crossed_count": 0,
        "delta_h_not_crossed_count": 0,
        "reverse_fail_crossed_count": 0,
        "reverse_fail_not_crossed_count": 0,
        "reverse_check_failure_count": 0,
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
        "_delta_h_values": [],
        "_delta_h_crossed_values": [],
        "_delta_h_not_crossed_values": [],
        "latent_acceptance_rate": 0.0,
        "median_residual": abs(float(np.asarray(x0)[int(median_idx)]) - float(mu_star)),
        "max_median_momentum_abs": 0.0,
        "experimental_nonsmooth_facet_rattle": True,
    }
    for group in DIAG_GROUPS:
        diag[f"{group}_proposals"] = 0
        diag[f"{group}_accepted"] = 0
        diag[f"{group}_reverse_fail_count"] = 0
        diag[f"{group}_delta_h_sum"] = 0.0
        diag[f"{group}_delta_h_abs_sum"] = 0.0
        diag[f"{group}_delta_h_sq_sum"] = 0.0
        diag[f"{group}_delta_h_max_abs"] = 0.0
        diag[f"{group}_delta_h_count"] = 0
        diag[f"_{group}_delta_h_values"] = []
        diag[f"{group}_reverse_position_error_sum"] = 0.0
        diag[f"{group}_reverse_momentum_error_sum"] = 0.0
        diag[f"{group}_reverse_position_error_max"] = 0.0
        diag[f"{group}_reverse_momentum_error_max"] = 0.0
        diag[f"{group}_reverse_check_count"] = 0
    return diag


def _trajectory_group(kink_crossed, boundary_crossed):
    if kink_crossed and boundary_crossed:
        return "kink_and_boundary"
    if kink_crossed:
        return "kink_only"
    if boundary_crossed:
        return "boundary_only"
    return "clean"


def _record_group_delta_h(diag, group, delta_h):
    diag[f"{group}_delta_h_sum"] += delta_h
    diag[f"{group}_delta_h_abs_sum"] += abs(delta_h)
    diag[f"{group}_delta_h_sq_sum"] += delta_h * delta_h
    diag[f"{group}_delta_h_max_abs"] = max(float(diag[f"{group}_delta_h_max_abs"]), abs(delta_h))
    diag[f"{group}_delta_h_count"] += 1
    diag[f"_{group}_delta_h_values"].append(float(delta_h))


def _quantile_or_nan(values, q):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, q))


def _record_group_reverse_check(diag, group, position_error, momentum_error, reverse_ok):
    diag[f"{group}_reverse_position_error_sum"] += position_error
    diag[f"{group}_reverse_momentum_error_sum"] += momentum_error
    diag[f"{group}_reverse_position_error_max"] = max(
        float(diag[f"{group}_reverse_position_error_max"]), position_error
    )
    diag[f"{group}_reverse_momentum_error_max"] = max(
        float(diag[f"{group}_reverse_momentum_error_max"]), momentum_error
    )
    diag[f"{group}_reverse_check_count"] += 1
    if not reverse_ok:
        diag[f"{group}_reverse_fail_count"] += 1


def _finalize_diag(diag):
    out = dict(diag)
    n_dh = max(int(out["delta_h_count"]), 1)
    n_rev = max(int(out["num_reverse_checks"]), 1)
    crossed = max(int(out["crossed_proposals"]), 1)
    not_crossed = max(int(out["not_crossed_proposals"]), 1)
    crossed_dh = max(int(out["delta_h_crossed_count"]), 1)
    not_crossed_dh = max(int(out["delta_h_not_crossed_count"]), 1)
    out["delta_h_mean"] = out["delta_h_sum"] / n_dh
    out["delta_h_mean_abs"] = out["delta_h_abs_sum"] / n_dh
    out["delta_h_rms"] = float(np.sqrt(max(out["delta_h_sq_sum"] / n_dh, 0.0)))
    out["delta_h_median"] = _quantile_or_nan(out["_delta_h_values"], 0.5)
    out["delta_h_q95"] = _quantile_or_nan(out["_delta_h_values"], 0.95)
    out["delta_h_abs_q95"] = _quantile_or_nan(np.abs(out["_delta_h_values"]), 0.95)
    out["mean_delta_H_if_crossed"] = out["delta_h_crossed_sum"] / crossed_dh
    out["mean_delta_H_if_not_crossed"] = out["delta_h_not_crossed_sum"] / not_crossed_dh
    out["median_delta_H_if_crossed"] = _quantile_or_nan(out["_delta_h_crossed_values"], 0.5)
    out["median_delta_H_if_not_crossed"] = _quantile_or_nan(out["_delta_h_not_crossed_values"], 0.5)
    out["q95_delta_H_if_crossed"] = _quantile_or_nan(out["_delta_h_crossed_values"], 0.95)
    out["q95_delta_H_if_not_crossed"] = _quantile_or_nan(out["_delta_h_not_crossed_values"], 0.95)
    out["max_delta_H_if_crossed"] = float(np.max(out["_delta_h_crossed_values"])) if out["_delta_h_crossed_values"] else np.nan
    out["max_delta_H_if_not_crossed"] = float(np.max(out["_delta_h_not_crossed_values"])) if out["_delta_h_not_crossed_values"] else np.nan
    out["accept_rate_if_crossed"] = out["accepted_crossed"] / crossed
    out["accept_rate_if_not_crossed"] = out["accepted_not_crossed"] / not_crossed
    out["reverse_fail_if_crossed"] = out["reverse_fail_crossed_count"] / crossed
    out["reverse_fail_if_not_crossed"] = out["reverse_fail_not_crossed_count"] / not_crossed
    out["mean_reverse_position_error"] = out["sum_reverse_position_error"] / n_rev
    out["mean_reverse_momentum_error"] = out["sum_reverse_momentum_error"] / n_rev
    for group in DIAG_GROUPS:
        proposals = max(int(out[f"{group}_proposals"]), 1)
        delta_n = max(int(out[f"{group}_delta_h_count"]), 1)
        reverse_n = max(int(out[f"{group}_reverse_check_count"]), 1)
        out[f"accept_rate_{group}"] = out[f"{group}_accepted"] / proposals
        out[f"reverse_fail_rate_{group}"] = out[f"{group}_reverse_fail_count"] / proposals
        out[f"reverse_fail_rate_checked_{group}"] = out[f"{group}_reverse_fail_count"] / reverse_n
        out[f"mean_delta_H_{group}"] = out[f"{group}_delta_h_sum"] / delta_n
        out[f"mean_abs_delta_H_{group}"] = out[f"{group}_delta_h_abs_sum"] / delta_n
        out[f"rms_delta_H_{group}"] = float(np.sqrt(max(out[f"{group}_delta_h_sq_sum"] / delta_n, 0.0)))
        out[f"max_abs_delta_H_{group}"] = out[f"{group}_delta_h_max_abs"]
        out[f"median_delta_H_{group}"] = _quantile_or_nan(out[f"_{group}_delta_h_values"], 0.5)
        out[f"q95_delta_H_{group}"] = _quantile_or_nan(out[f"_{group}_delta_h_values"], 0.95)
        out[f"mean_reverse_position_error_{group}"] = out[f"{group}_reverse_position_error_sum"] / reverse_n
        out[f"mean_reverse_momentum_error_{group}"] = out[f"{group}_reverse_momentum_error_sum"] / reverse_n
        out[f"max_reverse_position_error_{group}"] = out[f"{group}_reverse_position_error_max"]
        out[f"max_reverse_momentum_error_{group}"] = out[f"{group}_reverse_momentum_error_max"]
    for key in list(out):
        if key.startswith("_"):
            out.pop(key)
    return out


def _near_kink(x, mu, kink_tol):
    return bool(np.min(np.abs(np.asarray(x, dtype=float) - float(mu))) < float(kink_tol))


def _facet_trajectory(
    x0,
    momentum0,
    mu,
    mu_star,
    b,
    step_size,
    num_steps,
    median_idx,
    kink_tol=1e-8,
    cost_ledger=None,
):
    x, momentum = _project_facet(x0, momentum0, mu_star, median_idx)
    median_idx = int(median_idx)
    crossed = np.zeros(x.shape[0], dtype=bool)
    boundary_crossed = False
    boundary_cross_coordinate_count = 0
    near_kink_steps = 0
    max_median_momentum_abs = abs(float(momentum[median_idx]))

    for _ in range(int(num_steps)):
        if cost_ledger is not None:
            cost_ledger.inc("leapfrog_steps")
        grad = _grad_potential(x, mu, b, median_idx=median_idx, cost_ledger=cost_ledger)
        momentum = momentum - 0.5 * float(step_size) * grad
        momentum[median_idx] = 0.0
        x_prev = x.copy()
        sign_prev = _sign0(x_prev - float(mu))
        x = x + float(step_size) * momentum
        x[median_idx] = float(mu_star)
        side_crossed, side_cross_count = _side_boundary_crossed_between(x_prev, x, mu_star, median_idx)
        boundary_crossed = boundary_crossed or side_crossed
        boundary_cross_coordinate_count += side_cross_count
        sign_new = _sign0(x - float(mu))
        changed = sign_prev != sign_new
        changed[median_idx] = False
        crossed |= changed
        if _near_kink(x, mu, kink_tol):
            near_kink_steps += 1
        grad = _grad_potential(x, mu, b, median_idx=median_idx, cost_ledger=cost_ledger)
        momentum = momentum - 0.5 * float(step_size) * grad
        momentum[median_idx] = 0.0
        max_median_momentum_abs = max(max_median_momentum_abs, abs(float(momentum[median_idx])))

    x, momentum = _project_facet(x, momentum, mu_star, median_idx)
    ok = not _side_boundary_violated(x, mu_star, median_idx)
    return x, momentum, ok, {
        "kink_cross_count": int(np.sum(crossed)),
        "crossed": bool(np.any(crossed)),
        "near_kink_count": int(near_kink_steps),
        "boundary_crossed": bool(boundary_crossed),
        "side_boundary_cross_coordinate_count": int(boundary_cross_coordinate_count),
        "side_boundary_violated": not ok,
        "median_residual": abs(float(x[median_idx]) - float(mu_star)),
        "median_momentum_abs": abs(float(momentum[median_idx])),
        "max_median_momentum_abs": max_median_momentum_abs,
    }


def run_rattle(key, mu_star, params, verbose=True, cost_ledger=None):
    """Experimental nonsmooth fixed-facet HMC for odd-n Laplace median conditioning."""
    if not bool(params.get("laplace_rattle_experimental", False)):
        raise ValueError("Set laplace_rattle_experimental=True to run experimental Laplace facet-RATTLE")
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    median_idx = _median_index(n)
    b = float(params.get("b", 1.0))
    step_size = float(params.get("rattle_step_size", 0.05))
    num_steps = int(params.get("rattle_num_steps", 2))
    reverse_check = bool(params.get("reverse_check", True))
    reverse_position_tol = float(params.get("rattle_reverse_position_tol", 5e-3))
    reverse_momentum_tol = float(params.get("rattle_reverse_momentum_tol", 5e-3))
    kink_tol = float(params.get("kink_tol", 1e-8))

    mus = np.zeros(T + 1, dtype=float)
    xs = np.zeros((T + 1, n), dtype=float)
    x0 = np.asarray(_initial_x(mu_star, n, params), dtype=float).copy()
    x0[median_idx] = float(mu_star)
    mus[0] = float(mu_star)
    xs[0, :] = x0
    mu_acc = 0
    latent_acc = 0
    diag = _init_diag(x0, mu_star, median_idx, T)

    iters = range(1, T + 1)
    if verbose:
        iters = tqdm(iters, desc="Experimental facet-RATTLE (Laplace)")

    for t in iters:
        key, key_mu, key_p, key_u = random.split(key, 4)
        x_cur = xs[t - 1]
        mu_cur = mus[t - 1]
        if cost_ledger is not None:
            cost_ledger.inc("iterations")
            cost_ledger.inc("mu_mh_proposals")
            cost_ledger.inc("hmc_proposals")

        mu_new, acc_mu = _update_mu_mh(
            key_mu,
            jnp.asarray(mu_cur),
            jnp.asarray(x_cur),
            params["proposal_std_mu"],
            params["prior_mean"],
            params["prior_std"],
            b,
        )
        mu_new = float(mu_new)
        mus[t] = mu_new
        mu_acc += int(acc_mu)
        if cost_ledger is not None:
            cost_ledger.inc("mu_mh_accepts", int(acc_mu))

        if _near_kink(x_cur, mu_new, kink_tol):
            diag["near_kink_count"] += 1

        momentum0 = np.asarray(random.normal(key_p, shape=(n,)), dtype=float).copy()
        momentum0[median_idx] = 0.0
        h0 = _hamiltonian(x_cur, momentum0, mu_new, b, median_idx, cost_ledger=cost_ledger)
        x_prop, momentum_prop, ok, traj_diag = _facet_trajectory(
            x_cur,
            momentum0,
            mu_new,
            mu_star,
            b,
            step_size,
            num_steps,
            median_idx,
            kink_tol=kink_tol,
            cost_ledger=cost_ledger,
        )
        crossed = bool(traj_diag["crossed"])
        boundary_crossed = bool(traj_diag["boundary_crossed"])
        group = _trajectory_group(crossed, boundary_crossed)
        diag[f"{group}_proposals"] += 1
        if boundary_crossed:
            diag["side_boundary_cross_count"] += 1
            diag["side_boundary_cross_coordinate_count"] += int(traj_diag["side_boundary_cross_coordinate_count"])
        if crossed:
            diag["crossed_proposals"] += 1
        else:
            diag["not_crossed_proposals"] += 1
        diag["kink_cross_count"] += int(traj_diag["kink_cross_count"])
        diag["near_kink_count"] += int(traj_diag["near_kink_count"])
        diag["median_residual"] = max(float(diag["median_residual"]), float(traj_diag["median_residual"]))
        diag["max_median_momentum_abs"] = max(
            float(diag["max_median_momentum_abs"]), float(traj_diag["max_median_momentum_abs"])
        )

        if not ok:
            xs[t, :] = x_cur
            diag["forward_failures"] += 1
            diag["side_boundary_violation_count"] += 1
            if _side_count_failed(x_cur, mu_star, median_idx):
                diag["side_count_failures"] += 1
            continue

        h1 = _hamiltonian(x_prop, momentum_prop, mu_new, b, median_idx, cost_ledger=cost_ledger)
        delta_h = h1 - h0
        diag["delta_h_sum"] += delta_h
        diag["delta_h_abs_sum"] += abs(delta_h)
        diag["delta_h_sq_sum"] += delta_h * delta_h
        diag["delta_h_max_abs"] = max(float(diag["delta_h_max_abs"]), abs(delta_h))
        diag["delta_h_count"] += 1
        diag["_delta_h_values"].append(float(delta_h))
        if crossed:
            diag["delta_h_crossed_sum"] += delta_h
            diag["delta_h_crossed_count"] += 1
            diag["_delta_h_crossed_values"].append(float(delta_h))
        else:
            diag["delta_h_not_crossed_sum"] += delta_h
            diag["delta_h_not_crossed_count"] += 1
            diag["_delta_h_not_crossed_values"].append(float(delta_h))
        _record_group_delta_h(diag, group, delta_h)

        reverse_ok = True
        if reverse_check:
            if cost_ledger is not None:
                cost_ledger.inc("reverse_check_attempts")
            x_rev, momentum_rev, ok_rev, rev_diag = _facet_trajectory(
                x_prop,
                -momentum_prop,
                mu_new,
                mu_star,
                b,
                step_size,
                num_steps,
                median_idx,
                kink_tol=kink_tol,
                cost_ledger=cost_ledger,
            )
            reverse_position_error = float(np.linalg.norm(x_rev - x_cur))
            reverse_momentum_error = float(np.linalg.norm(momentum_rev + momentum0))
            diag["max_reverse_position_error"] = max(
                float(diag["max_reverse_position_error"]), reverse_position_error
            )
            diag["max_reverse_momentum_error"] = max(
                float(diag["max_reverse_momentum_error"]), reverse_momentum_error
            )
            diag["sum_reverse_position_error"] += reverse_position_error
            diag["sum_reverse_momentum_error"] += reverse_momentum_error
            diag["num_reverse_checks"] += 1
            diag["near_kink_count"] += int(rev_diag["near_kink_count"])
            reverse_ok = ok_rev and reverse_position_error <= reverse_position_tol and reverse_momentum_error <= reverse_momentum_tol
            _record_group_reverse_check(
                diag,
                group,
                reverse_position_error,
                reverse_momentum_error,
                reverse_ok,
            )
            if not reverse_ok:
                diag["reverse_check_failure_count"] += 1
                diag["reverse_failures"] += 1
                if crossed:
                    diag["reverse_fail_crossed_count"] += 1
                else:
                    diag["reverse_fail_not_crossed_count"] += 1

        if not reverse_ok:
            xs[t, :] = x_cur
            continue

        accepted = np.log(float(random.uniform(key_u, minval=EPS_U, maxval=1.0))) < min(0.0, -delta_h)
        if accepted:
            xs[t, :] = x_prop
            latent_acc += 1
            diag[f"{group}_accepted"] += 1
            if crossed:
                diag["accepted_crossed"] += 1
            else:
                diag["accepted_not_crossed"] += 1
            if cost_ledger is not None:
                cost_ledger.inc("hmc_accepts")
        else:
            xs[t, :] = x_cur
            if cost_ledger is not None:
                cost_ledger.inc("metropolized_rejections")

        if _side_count_failed(xs[t, :], mu_star, median_idx):
            diag["side_count_failures"] += 1
        diag["median_residual"] = max(float(diag["median_residual"]), abs(float(xs[t, median_idx]) - float(mu_star)))

    diag["latent_acceptance_rate"] = latent_acc / max(T, 1)
    out_diag = _finalize_diag(diag)
    if cost_ledger is not None:
        cost_ledger.set("iterations", T)
        cost_ledger.set("mu_mh_accepts", mu_acc)
        cost_ledger.set("hmc_accepts", latent_acc)
        cost_ledger.set("experimental_nonsmooth_facet_rattle", True)
    return {
        "mu_chain": mus,
        "x_chain": xs,
        "mu_acceptance_rate": mu_acc / max(T, 1),
        "x_acceptance_rate": latent_acc / max(T, 1),
        "latent_acceptance_rate": latent_acc / max(T, 1),
        "median_index": median_idx,
        "projection_diagnostics": out_diag,
        "experimental_nonsmooth_facet_rattle": True,
    }
