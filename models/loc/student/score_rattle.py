"""Production Student-t constrained-HMC sampler using paper-style RATTLE.

The latent vector ``x`` is constrained to the Student-t score manifold

    c(x) = sum_i (x_i - mu_star) / (k + (x_i - mu_star)**2) = 0.

The numerical RATTLE kernels below are Numba-compiled and use the scalar,
fixed-old-gradient projection in ``docs/rattle.md``.  Every proposal includes
the reverse projection check required by that algorithm.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from tqdm import tqdm

from models.loc.student.mu_mh import rng_from_key, update_mu_mh_from_variates

EPS_U = 1e-12
GRAM_FLOOR = 1e-300


def _initial_x(mu_star, n, k, params):
    """Construct an exactly score-constrained initial latent state."""
    if "initial_x" in params:
        initial_x = np.asarray(params["initial_x"], dtype=np.float64)
        if initial_x.shape != (n,) or not np.all(np.isfinite(initial_x)):
            raise ValueError("initial_x must be a finite vector with shape (n,)")
        residual = abs(_constraint_value_kernel(initial_x, mu_star, k))
        tolerance = float(params.get("initial_x_constraint_tol", 1e-8))
        if residual > tolerance:
            raise ValueError(
                f"initial_x is not on the score manifold: residual={residual:.3e}"
            )
        return initial_x.copy()
    initialization = str(params.get("initialization", "central"))
    if initialization == "central":
        return np.full(n, float(mu_star), dtype=np.float64)
    if initialization == "tail_heavy":
        amplitude = float(params.get("initialization_tail_amplitude", max(5.0, 5.0 * np.sqrt(k))))
        offsets = np.tile(np.array([amplitude, -amplitude]), n // 2)
    elif initialization == "random":
        rng = np.random.default_rng(int(params.get("initialization_seed", 0)))
        amplitudes = np.exp(rng.normal(0.0, 1.0, size=n // 2))
        offsets = np.empty(2 * (n // 2), dtype=np.float64)
        offsets[0::2] = amplitudes
        offsets[1::2] = -amplitudes
        if n % 2:
            offsets = np.concatenate((offsets, np.zeros(1, dtype=np.float64)))
        rng.shuffle(offsets)
        return float(mu_star) + offsets
    else:
        raise ValueError(f"Unknown initialization: {initialization}")
    if n % 2:
        offsets = np.concatenate((offsets, np.zeros(1, dtype=np.float64)))
    return float(mu_star) + offsets


@njit(cache=True)
def _constraint_value_kernel(x, mu_star, k):
    total = 0.0
    for i in range(x.size):
        y = x[i] - mu_star
        total += y / (k + y * y)
    return total


@njit(cache=True)
def _constraint_grad_kernel(x, mu_star, k):
    grad = np.empty_like(x)
    for i in range(x.size):
        y = x[i] - mu_star
        denom = k + y * y
        grad[i] = (k - y * y) / (denom * denom)
    return grad


@njit(cache=True)
def _constraint_gram_kernel(x, mu_star, k):
    total = 0.0
    for i in range(x.size):
        y = x[i] - mu_star
        denom = k + y * y
        grad = (k - y * y) / (denom * denom)
        total += grad * grad
    return total


@njit(cache=True)
def _grad_potential_kernel(x, mu, mu_star, k):
    """Gradient of Student-t potential plus the coarea Gram correction."""
    gram = _constraint_gram_kernel(x, mu_star, k)
    grad = np.empty_like(x)
    if not math.isfinite(gram) or gram <= 0.0:
        for i in range(x.size):
            y = x[i] - mu
            grad[i] = (k + 1.0) * y / (k + y * y)
        return grad

    for i in range(x.size):
        y_mu = x[i] - mu
        y_star = x[i] - mu_star
        denom = k + y_star * y_star
        constraint_grad = (k - y_star * y_star) / (denom * denom)
        constraint_grad_derivative = -2.0 * y_star * (3.0 * k - y_star * y_star) / (denom * denom * denom)
        grad[i] = (
            (k + 1.0) * y_mu / (k + y_mu * y_mu)
            + constraint_grad * constraint_grad_derivative / gram
        )
    return grad


@njit(cache=True)
def _potential_energy_kernel(x, mu, mu_star, k):
    total = 0.0
    for i in range(x.size):
        y = x[i] - mu
        total += math.log1p((y * y) / k)
    gram = _constraint_gram_kernel(x, mu_star, k)
    return 0.5 * (k + 1.0) * total + 0.5 * math.log(max(gram, GRAM_FLOOR))


@njit(cache=True)
def _project_momentum_kernel(momentum, grad, grad_tol):
    denom = 0.0
    numerator = 0.0
    for i in range(momentum.size):
        denom += grad[i] * grad[i]
        numerator += grad[i] * momentum[i]
    if not math.isfinite(denom) or denom <= grad_tol:
        return momentum.copy(), False

    correction = numerator / denom
    projected = np.empty_like(momentum)
    for i in range(momentum.size):
        projected[i] = momentum[i] - correction * grad[i]
    return projected, True


@njit(cache=True)
def _project_to_manifold_fixed_direction_kernel(
    x_tilde,
    direction,
    mu_star,
    k,
    tol,
    max_iters,
    grad_tol,
    damping,
    line_search,
):
    """Solve c(x_tilde + theta * direction) = 0 for scalar theta."""
    direction_norm_sq = 0.0
    for i in range(direction.size):
        if not math.isfinite(direction[i]):
            residual = abs(_constraint_value_kernel(x_tilde, mu_star, k))
            return x_tilde.copy(), False, 0, residual, 0.0
        direction_norm_sq += direction[i] * direction[i]
    if direction_norm_sq <= grad_tol:
        residual = abs(_constraint_value_kernel(x_tilde, mu_star, k))
        return x_tilde.copy(), False, 0, residual, 0.0

    x_cur = x_tilde.copy()
    x_next = np.empty_like(x_tilde)
    theta = 0.0
    residual = abs(_constraint_value_kernel(x_cur, mu_star, k))
    if residual <= tol:
        return x_cur, True, 0, residual, theta

    for iteration in range(1, max_iters + 1):
        c_value = 0.0
        derivative = 0.0
        for i in range(x_cur.size):
            y = x_cur[i] - mu_star
            denom = k + y * y
            c_value += y / denom
            derivative += ((k - y * y) / (denom * denom)) * direction[i]
        if not math.isfinite(derivative) or abs(derivative) <= grad_tol:
            return x_cur, False, iteration, residual, theta

        raw_step = c_value / derivative
        alpha = damping
        theta_next = theta - alpha * raw_step
        for i in range(x_cur.size):
            x_next[i] = x_tilde[i] + theta_next * direction[i]
        next_residual = abs(_constraint_value_kernel(x_next, mu_star, k))

        if line_search:
            while alpha > 1e-6 and next_residual > residual:
                alpha *= 0.5
                theta_next = theta - alpha * raw_step
                for i in range(x_cur.size):
                    x_next[i] = x_tilde[i] + theta_next * direction[i]
                next_residual = abs(_constraint_value_kernel(x_next, mu_star, k))

        theta = theta_next
        x_cur[:] = x_next
        residual = next_residual
        if residual <= tol:
            return x_cur, True, iteration, residual, theta

    return x_cur, False, max_iters, residual, theta


@njit(cache=True)
def _rattle_trajectory_kernel(
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
    tangent_tol,
):
    """Run the fixed-direction RATTLE map and return compact operational data."""
    x = x0.copy()
    momentum = momentum0.copy()
    max_residual = abs(_constraint_value_kernel(x, mu_star, k))
    projection_iterations = 0
    position_failures = 0
    momentum_failures = 0
    completed_steps = 0

    for _ in range(num_steps):
        grad_old = _constraint_grad_kernel(x, mu_star, k)
        potential_grad = _grad_potential_kernel(x, mu, mu_star, k)
        p_half_unconstrained = momentum - 0.5 * step_size * potential_grad
        x_tilde = x + step_size * p_half_unconstrained
        x, ok, n_iter, residual, theta = _project_to_manifold_fixed_direction_kernel(
            x_tilde,
            grad_old,
            mu_star,
            k,
            proj_tol,
            proj_max_iters,
            grad_tol,
            proj_damping,
            proj_line_search,
        )
        projection_iterations += n_iter
        max_residual = max(max_residual, residual)
        if not ok:
            position_failures += 1
            return (
                x,
                momentum,
                False,
                projection_iterations,
                position_failures,
                momentum_failures,
                max_residual,
                completed_steps,
            )

        p_half = p_half_unconstrained + (theta / step_size) * grad_old
        p_tilde = p_half - 0.5 * step_size * _grad_potential_kernel(x, mu, mu_star, k)
        grad_new = _constraint_grad_kernel(x, mu_star, k)
        momentum, ok = _project_momentum_kernel(p_tilde, grad_new, grad_tol)
        residual = abs(_constraint_value_kernel(x, mu_star, k))
        max_residual = max(max_residual, residual)

        tangent = 0.0
        for i in range(momentum.size):
            tangent += grad_new[i] * momentum[i]
        if not ok or residual > proj_tol or abs(tangent) > tangent_tol:
            if residual > proj_tol:
                position_failures += 1
            else:
                momentum_failures += 1
            return (
                x,
                momentum,
                False,
                projection_iterations,
                position_failures,
                momentum_failures,
                max_residual,
                completed_steps,
            )
        completed_steps += 1

    return (
        x,
        momentum,
        True,
        projection_iterations,
        position_failures,
        momentum_failures,
        max_residual,
        completed_steps,
    )


# Public numerical helpers remain Python-callable for tests and small analyses.
def _constraint_value(x, mu_star, k, cost_ledger=None):
    value = float(_constraint_value_kernel(np.asarray(x, dtype=np.float64), mu_star, k))
    if cost_ledger is not None:
        cost_ledger.inc("constraint_evals")
        cost_ledger.observe_constraint(value)
    return value


def _constraint_grad(x, mu_star, k, cost_ledger=None):
    grad = _constraint_grad_kernel(np.asarray(x, dtype=np.float64), mu_star, k)
    if cost_ledger is not None:
        cost_ledger.inc("constraint_grad_evals")
    return grad


def _constraint_gram(x, mu_star, k, cost_ledger=None):
    gram = float(_constraint_gram_kernel(np.asarray(x, dtype=np.float64), mu_star, k))
    if cost_ledger is not None:
        cost_ledger.inc("gram_evals")
    return gram


def _project_momentum(momentum, grad, grad_tol, cost_ledger=None):
    projected, ok = _project_momentum_kernel(
        np.asarray(momentum, dtype=np.float64), np.asarray(grad, dtype=np.float64), grad_tol
    )
    if cost_ledger is not None:
        cost_ledger.inc("projection_evals")
        cost_ledger.inc("momentum_projections")
    return projected, bool(ok)


def _grad_potential(x, mu, k, mu_star, cost_ledger=None):
    grad = _grad_potential_kernel(np.asarray(x, dtype=np.float64), mu, mu_star, k)
    if cost_ledger is not None:
        cost_ledger.inc("gradient_evals")
        cost_ledger.inc("student_grad_evals", np.asarray(x).size)
        cost_ledger.inc("gram_grad_evals")
    return grad


def _potential_energy(x, mu, k, mu_star, cost_ledger=None):
    energy = float(_potential_energy_kernel(np.asarray(x, dtype=np.float64), mu, mu_star, k))
    if cost_ledger is not None:
        cost_ledger.inc("potential_evals")
        cost_ledger.inc("student_logpdf_evals", np.asarray(x).size)
        cost_ledger.inc("gram_evals")
    return energy


def _hamiltonian(x, momentum, mu, k, mu_star, cost_ledger=None):
    kinetic = 0.5 * float(np.dot(momentum, momentum))
    if cost_ledger is not None:
        cost_ledger.inc("energy_evals")
    return _potential_energy(x, mu, k, mu_star, cost_ledger=cost_ledger) + kinetic


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
    tangent_tol=1e-8,
    cost_ledger=None,
    trajectory_kind="forward",
):
    """Python wrapper around the compiled paper-style RATTLE trajectory."""
    result = _rattle_trajectory_kernel(
        np.asarray(x0, dtype=np.float64),
        np.asarray(momentum0, dtype=np.float64),
        float(mu),
        float(mu_star),
        float(k),
        float(step_size),
        int(num_steps),
        float(proj_tol),
        int(proj_max_iters),
        float(grad_tol),
        float(proj_damping),
        bool(proj_line_search),
        float(tangent_tol),
    )
    x, momentum, ok, n_iter, position_failures, momentum_failures, max_residual, completed_steps = result
    diag = {
        "position_projection_failures": int(position_failures),
        "momentum_projection_failures": int(momentum_failures),
        "projection_iterations": int(n_iter),
        "max_projection_residual": float(max_residual),
        "completed_steps": int(completed_steps),
    }
    if cost_ledger is not None:
        cost_ledger.inc("leapfrog_steps", int(completed_steps))
        cost_ledger.inc("position_projection_newton_iters", int(n_iter))
        cost_ledger.inc(f"{trajectory_kind}_newton_iters", int(n_iter))
        cost_ledger.inc("position_projection_failures", int(position_failures))
    return x, momentum, bool(ok), diag


def _operational_summary(
    *,
    proposals,
    forward_failures,
    reverse_failures,
    initial_momentum_projection_failures,
    position_projection_failures,
    momentum_projection_failures,
    projection_iterations_total,
    max_projection_residual,
    max_manifold_residual,
    max_reverse_position_error,
    delta_h_abs_sum,
    delta_h_max_abs,
    delta_h_count,
    x_acceptance_rate,
):
    """Compact sampler health summary retained for existing callers."""
    count = max(int(delta_h_count), 1)
    return {
        "proposals": int(proposals),
        "forward_failures": int(forward_failures),
        "reverse_failures": int(reverse_failures),
        "initial_momentum_projection_failures": int(initial_momentum_projection_failures),
        "position_projection_failures": int(position_projection_failures),
        "momentum_projection_failures": int(momentum_projection_failures),
        "projection_failure_count": int(forward_failures + initial_momentum_projection_failures),
        "reverse_check_failure_count": int(reverse_failures),
        "projection_iterations_total": int(projection_iterations_total),
        "max_projection_residual": float(max_projection_residual),
        "max_manifold_residual": float(max_manifold_residual),
        "max_reverse_position_error": float(max_reverse_position_error),
        "delta_h_mean_abs": float(delta_h_abs_sum / count),
        "delta_h_max_abs": float(delta_h_max_abs),
        "x_acceptance_rate": float(x_acceptance_rate),
        "include_gram_correction": True,
        "projection_mode": "paper_fixed_direction",
        "reverse_check": True,
    }


def run_rattle(key, mu_star, params, verbose=True, cost_ledger=None):
    """Sample ``mu`` and constrained ``x`` using fully refreshed RATTLE HMC."""
    T = int(params["num_iterations_T"])
    n = int(params["n"])
    k = float(params["k"])
    step_size = float(params.get("rattle_step_size", 0.05))
    num_steps = int(params.get("rattle_num_steps", 2))
    proj_tol = float(params.get("rattle_proj_tol", 1e-10))
    proj_max_iters = int(params.get("rattle_proj_max_iters", 25))
    grad_tol = float(params.get("rattle_grad_tol", 1e-12))
    reverse_position_tol = float(params.get("rattle_reverse_position_tol", 5e-3))
    proj_damping = float(params.get("rattle_proj_damping", 1.0))
    proj_line_search = bool(params.get("rattle_proj_line_search", True))
    tangent_tol = float(params.get("rattle_tangent_tol", 1e-8))

    if T <= 0 or n <= 1 or k <= 0.0 or step_size <= 0.0 or num_steps <= 0:
        raise ValueError("RATTLE requires T > 0, n > 1, k > 0, positive step_size, and num_steps > 0")
    if float(params["proposal_std_mu"]) < 0.0 or float(params["prior_std"]) <= 0.0:
        raise ValueError("RATTLE requires proposal_std_mu >= 0 and prior_std > 0")
    if not bool(params.get("reverse_check", True)):
        raise ValueError("The production RATTLE sampler requires reverse_check=True")
    if not bool(params.get("rattle_include_gram_correction", True)):
        raise ValueError("The production RATTLE sampler requires the coarea Gram correction")
    if str(params.get("rattle_projection_mode", "paper_fixed_direction")) != "paper_fixed_direction":
        raise ValueError("Only the paper_fixed_direction RATTLE projection is supported")

    mus = np.empty(T + 1, dtype=np.float64)
    xs = np.empty((T + 1, n), dtype=np.float64)
    mus[0] = float(mu_star)
    xs[0] = np.asarray(_initial_x(mu_star, n, k, params), dtype=np.float64)
    rng = rng_from_key(key)

    mu_accepts = 0
    x_accepts = 0
    forward_failures = 0
    reverse_failures = 0
    initial_momentum_projection_failures = 0
    position_projection_failures = 0
    momentum_projection_failures = 0
    projection_iterations_total = 0
    max_projection_residual = abs(_constraint_value_kernel(xs[0], mu_star, k))
    max_manifold_residual = max_projection_residual
    max_reverse_position_error = 0.0
    delta_h_abs_sum = 0.0
    delta_h_max_abs = 0.0
    delta_h_count = 0

    if cost_ledger is not None:
        cost_ledger.set("projection_mode", "paper_fixed_direction")
        cost_ledger.set("gram_correction_enabled", True)

    iterations = range(1, T + 1)
    if verbose:
        iterations = tqdm(iterations, desc="RATTLE (Student)")

    for t in iterations:
        x_cur = xs[t - 1]
        mu_cur = mus[t - 1]
        if cost_ledger is not None:
            cost_ledger.inc("iterations")
            cost_ledger.inc("mu_mh_proposals")
            cost_ledger.inc("student_logpdf_evals", 2 * n)
            cost_ledger.inc("prior_logpdf_evals", 2)
            cost_ledger.inc("hmc_proposals")

        mu_new, mu_accepted = update_mu_mh_from_variates(
            mu_cur,
            x_cur,
            params["proposal_std_mu"],
            params["prior_mean"],
            params["prior_std"],
            k,
            float(rng.normal()),
            float(rng.uniform(EPS_U, 1.0)),
        )
        mus[t] = mu_new
        mu_accepts += int(mu_accepted)
        if cost_ledger is not None:
            cost_ledger.inc("mu_mh_accepts", int(mu_accepted))

        momentum0 = rng.standard_normal(n)
        momentum0, momentum_ok = _project_momentum_kernel(
            momentum0, _constraint_grad_kernel(x_cur, mu_star, k), grad_tol
        )
        if not momentum_ok:
            xs[t] = x_cur
            initial_momentum_projection_failures += 1
            max_manifold_residual = max(max_manifold_residual, abs(_constraint_value_kernel(x_cur, mu_star, k)))
            if cost_ledger is not None:
                cost_ledger.inc("projection_failures")
                cost_ledger.inc("integration_failures")
            continue

        h0 = _hamiltonian(x_cur, momentum0, mu_new, k, mu_star, cost_ledger=cost_ledger)
        x_prop, momentum_prop, forward_ok, forward_diag = _rattle_trajectory(
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
            tangent_tol=tangent_tol,
            cost_ledger=cost_ledger,
            trajectory_kind="forward",
        )
        position_projection_failures += forward_diag["position_projection_failures"]
        momentum_projection_failures += forward_diag["momentum_projection_failures"]
        projection_iterations_total += forward_diag["projection_iterations"]
        max_projection_residual = max(max_projection_residual, forward_diag["max_projection_residual"])
        if not forward_ok:
            xs[t] = x_cur
            forward_failures += 1
            max_manifold_residual = max(max_manifold_residual, abs(_constraint_value_kernel(x_cur, mu_star, k)))
            if cost_ledger is not None:
                cost_ledger.inc("projection_failures")
                cost_ledger.inc("integration_failures")
            continue

        h1 = _hamiltonian(x_prop, momentum_prop, mu_new, k, mu_star, cost_ledger=cost_ledger)
        delta_h = h1 - h0
        delta_h_abs_sum += abs(delta_h)
        delta_h_max_abs = max(delta_h_max_abs, abs(delta_h))
        delta_h_count += 1

        if cost_ledger is not None:
            cost_ledger.inc("reverse_check_attempts")
        x_reverse, _, reverse_ok, reverse_diag = _rattle_trajectory(
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
            tangent_tol=tangent_tol,
            cost_ledger=cost_ledger,
            trajectory_kind="reverse",
        )
        position_projection_failures += reverse_diag["position_projection_failures"]
        momentum_projection_failures += reverse_diag["momentum_projection_failures"]
        projection_iterations_total += reverse_diag["projection_iterations"]
        max_projection_residual = max(max_projection_residual, reverse_diag["max_projection_residual"])
        reverse_position_error = float(np.linalg.norm(x_reverse - x_cur))
        max_reverse_position_error = max(max_reverse_position_error, reverse_position_error)
        if not reverse_ok or reverse_position_error > reverse_position_tol:
            xs[t] = x_cur
            reverse_failures += 1
            max_manifold_residual = max(max_manifold_residual, abs(_constraint_value_kernel(x_cur, mu_star, k)))
            if cost_ledger is not None:
                cost_ledger.inc("reverse_check_failures")
                if not reverse_ok:
                    cost_ledger.inc("projection_failures")
            continue

        accepted = math.log(float(rng.uniform(EPS_U, 1.0))) < min(0.0, -delta_h)
        if accepted:
            xs[t] = x_prop
            x_accepts += 1
            if cost_ledger is not None:
                cost_ledger.inc("hmc_accepts")
        else:
            xs[t] = x_cur
            if cost_ledger is not None:
                cost_ledger.inc("metropolized_rejections")
        max_manifold_residual = max(max_manifold_residual, abs(_constraint_value_kernel(xs[t], mu_star, k)))

    x_acceptance_rate = x_accepts / T
    diagnostics = _operational_summary(
        proposals=T,
        forward_failures=forward_failures,
        reverse_failures=reverse_failures,
        initial_momentum_projection_failures=initial_momentum_projection_failures,
        position_projection_failures=position_projection_failures,
        momentum_projection_failures=momentum_projection_failures,
        projection_iterations_total=projection_iterations_total,
        max_projection_residual=max_projection_residual,
        max_manifold_residual=max_manifold_residual,
        max_reverse_position_error=max_reverse_position_error,
        delta_h_abs_sum=delta_h_abs_sum,
        delta_h_max_abs=delta_h_max_abs,
        delta_h_count=delta_h_count,
        x_acceptance_rate=x_acceptance_rate,
    )
    if cost_ledger is not None:
        cost_ledger.set("iterations", T)
        cost_ledger.set("mu_mh_accepts", mu_accepts)
        cost_ledger.set("hmc_accepts", x_accepts)
        cost_ledger.set("reverse_position_error", max_reverse_position_error)
        cost_ledger.update_from_projection_diag(diagnostics)

    return {
        "mu_chain": mus,
        "x_chain": xs,
        "mu_acceptance_rate": mu_accepts / T,
        "x_acceptance_rate": x_acceptance_rate,
        "projection_diagnostics": diagnostics,
    }
