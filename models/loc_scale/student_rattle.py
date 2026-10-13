"""Codimension-two RATTLE for released Student-t location-scale MLEs.

The constrained coordinates are the Student angles from ``docs/loc_scale.tex``.
The potential targets Hausdorff measure on the fixed-resultant manifold and
therefore contains both the coarea Gram correction and the
MLE-conditioning ``abs(det(H_C))`` correction.  Every numerical proposal is
subject to the forward/reverse projection test used by the production
location-only Student RATTLE sampler.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from tqdm import tqdm

from models.loc_scale.student_gibbs import (
    ANGLE_MARGIN,
    EPS_U,
    GEOMETRY_FLOOR,
    _angle_to_y,
    _angles_to_x,
    _det_h_numerator,
    _geometry_sums,
    _log_angle_density,
    _parameter_log_target,
    _score_residual_angles,
    _strict_local_mle,
    _validate_common,
    init_angles,
)
from models.loc.student.mu_mh import rng_from_key


@njit(cache=True)
def _constraint_values(phis, k):
    return _score_residual_angles(phis, k)


@njit(cache=True)
def _constraint_gram_terms(phis):
    gram_00 = 0.0
    gram_01 = 0.0
    gram_11 = 0.0
    for phi in phis:
        sine = math.sin(phi)
        cosine = math.cos(phi)
        gram_00 += sine * sine
        gram_01 -= sine * cosine
        gram_11 += cosine * cosine
    determinant = gram_00 * gram_11 - gram_01 * gram_01
    return gram_00, gram_01, gram_11, determinant


@njit(cache=True)
def _potential_energy_kernel(phis, mu, sigma, mu_star, sigma_star, k):
    if sigma <= 0.0:
        return math.inf
    log_density = 0.0
    for phi in phis:
        value = _log_angle_density(phi, mu, sigma, mu_star, sigma_star, k)
        if not math.isfinite(value):
            return math.inf
        log_density += value

    geometry_sums = _geometry_sums(phis, k)
    if not _strict_local_mle(geometry_sums):
        return math.inf
    det_h = _det_h_numerator(geometry_sums) / sigma_star
    _, _, _, det_gram = _constraint_gram_terms(phis)
    if det_gram <= GEOMETRY_FLOOR or not math.isfinite(det_gram):
        return math.inf
    return -log_density - math.log(abs(det_h)) + 0.5 * math.log(det_gram)


@njit(cache=True)
def _grad_potential_kernel(phis, mu, sigma, mu_star, sigma_star, k):
    """Analytic gradient of likelihood, MLE-Jacobian, and Gram terms."""
    n = phis.size
    gradient = np.empty(n, dtype=np.float64)
    sums = _geometry_sums(phis, k)
    det_h_numerator = _det_h_numerator(sums)
    gram_00, gram_01, gram_11, det_gram = _constraint_gram_terms(phis)
    if (
        sigma <= 0.0
        or not _strict_local_mle(sums)
        or det_gram <= GEOMETRY_FLOOR
        or not math.isfinite(det_gram)
    ):
        gradient[:] = math.nan
        return gradient

    sqrt_k = math.sqrt(k)
    scale_ratio = sigma_star / sigma
    for i in range(n):
        phi = phis[i]
        if not (-math.pi + ANGLE_MARGIN < phi < math.pi - ANGLE_MARGIN):
            gradient[i] = math.nan
            continue
        tangent_half = math.tan(0.5 * phi)
        y = sqrt_k * tangent_half
        dy_dphi = 0.5 * sqrt_k * (1.0 + tangent_half * tangent_half)
        standardized = (mu_star + sigma_star * y - mu) / sigma
        score_standardized = -(k + 1.0) * standardized / (k + standardized * standardized)
        d_log_q = score_standardized * scale_ratio * dy_dphi + tangent_half

        denom = k + y * y
        denom2 = denom * denom
        denom3 = denom2 * denom
        a = (k - y * y) / denom2
        b = 2.0 * k * y / denom2
        a_prime = -2.0 * y * (3.0 * k - y * y) / denom3
        b_prime = 2.0 * k * (k - 3.0 * y * y) / denom3
        d_det_h_dy = (
            a_prime * sums[3]
            + sums[0] * (b + y * b_prime)
            - (a + y * a_prime) * sums[2]
            - sums[1] * b_prime
        )
        d_log_det_h = d_det_h_dy * dy_dphi / det_h_numerator

        sine = math.sin(phi)
        cosine = math.cos(phi)
        d_gram_00 = 2.0 * sine * cosine
        d_gram_01 = sine * sine - cosine * cosine
        d_gram_11 = -2.0 * sine * cosine
        d_det_gram = (
            d_gram_00 * gram_11
            + gram_00 * d_gram_11
            - 2.0 * gram_01 * d_gram_01
        )
        gradient[i] = -d_log_q - d_log_det_h + 0.5 * d_det_gram / det_gram
    return gradient


@njit(cache=True)
def _project_momentum_kernel(momentum, phis, gram_tol):
    gram_00, gram_01, gram_11, determinant = _constraint_gram_terms(phis)
    if determinant <= gram_tol or not math.isfinite(determinant):
        return momentum.copy(), False
    rhs_0 = 0.0
    rhs_1 = 0.0
    for i in range(phis.size):
        rhs_0 -= math.sin(phis[i]) * momentum[i]
        rhs_1 += math.cos(phis[i]) * momentum[i]
    multiplier_0 = (gram_11 * rhs_0 - gram_01 * rhs_1) / determinant
    multiplier_1 = (-gram_01 * rhs_0 + gram_00 * rhs_1) / determinant
    projected = np.empty_like(momentum)
    for i in range(phis.size):
        projected[i] = (
            momentum[i]
            + math.sin(phis[i]) * multiplier_0
            - math.cos(phis[i]) * multiplier_1
        )
    return projected, True


@njit(cache=True)
def _constraint_tangent_norm(momentum, phis):
    residual_0 = 0.0
    residual_1 = 0.0
    for i in range(phis.size):
        residual_0 -= math.sin(phis[i]) * momentum[i]
        residual_1 += math.cos(phis[i]) * momentum[i]
    return math.sqrt(residual_0 * residual_0 + residual_1 * residual_1)


@njit(cache=True)
def _inside_angle_domain(phis):
    for phi in phis:
        if not (-math.pi + ANGLE_MARGIN < phi < math.pi - ANGLE_MARGIN):
            return False
    return True


@njit(cache=True)
def _constraint_norm(phis, k):
    residual_0, residual_1 = _constraint_values(phis, k)
    return math.sqrt(residual_0 * residual_0 + residual_1 * residual_1)


@njit(cache=True)
def _project_position_fixed_direction_kernel(
    trial,
    old_phis,
    k,
    tolerance,
    max_iterations,
    jacobian_tol,
    damping,
    line_search,
):
    """Project with the fixed old constraint-normal directions."""
    current = trial.copy()
    candidate = trial.copy()
    multiplier_0 = 0.0
    multiplier_1 = 0.0
    residual = _constraint_norm(current, k)
    if not _inside_angle_domain(current):
        return current, False, 0, residual, multiplier_0, multiplier_1
    if residual <= tolerance:
        return current, True, 0, residual, multiplier_0, multiplier_1

    for iteration in range(1, max_iterations + 1):
        constraint_0, constraint_1 = _constraint_values(current, k)
        jacobian_00 = 0.0
        jacobian_01 = 0.0
        jacobian_10 = 0.0
        jacobian_11 = 0.0
        for i in range(current.size):
            old_sine = math.sin(old_phis[i])
            old_cosine = math.cos(old_phis[i])
            new_sine = math.sin(current[i])
            new_cosine = math.cos(current[i])
            jacobian_00 += new_sine * old_sine
            jacobian_01 -= new_sine * old_cosine
            jacobian_10 -= new_cosine * old_sine
            jacobian_11 += new_cosine * old_cosine
        determinant = jacobian_00 * jacobian_11 - jacobian_01 * jacobian_10
        if abs(determinant) <= jacobian_tol or not math.isfinite(determinant):
            return current, False, iteration, residual, multiplier_0, multiplier_1

        step_0 = (jacobian_11 * constraint_0 - jacobian_01 * constraint_1) / determinant
        step_1 = (-jacobian_10 * constraint_0 + jacobian_00 * constraint_1) / determinant
        alpha = damping
        next_multiplier_0 = multiplier_0 - alpha * step_0
        next_multiplier_1 = multiplier_1 - alpha * step_1
        for i in range(current.size):
            candidate[i] = (
                trial[i]
                - math.sin(old_phis[i]) * next_multiplier_0
                + math.cos(old_phis[i]) * next_multiplier_1
            )
        next_residual = _constraint_norm(candidate, k)
        domain_ok = _inside_angle_domain(candidate)

        if line_search:
            while alpha > 1e-6 and (not domain_ok or next_residual > residual):
                alpha *= 0.5
                next_multiplier_0 = multiplier_0 - alpha * step_0
                next_multiplier_1 = multiplier_1 - alpha * step_1
                for i in range(current.size):
                    candidate[i] = (
                        trial[i]
                        - math.sin(old_phis[i]) * next_multiplier_0
                        + math.cos(old_phis[i]) * next_multiplier_1
                    )
                next_residual = _constraint_norm(candidate, k)
                domain_ok = _inside_angle_domain(candidate)
        if not domain_ok:
            return current, False, iteration, residual, multiplier_0, multiplier_1

        multiplier_0 = next_multiplier_0
        multiplier_1 = next_multiplier_1
        current[:] = candidate
        residual = next_residual
        if residual <= tolerance:
            return current, True, iteration, residual, multiplier_0, multiplier_1
    return current, False, max_iterations, residual, multiplier_0, multiplier_1


@njit(cache=True)
def _rattle_trajectory_kernel(
    initial_phis,
    initial_momentum,
    mu,
    sigma,
    mu_star,
    sigma_star,
    k,
    step_size,
    num_steps,
    projection_tolerance,
    projection_max_iterations,
    jacobian_tolerance,
    projection_damping,
    projection_line_search,
    tangent_tolerance,
):
    phis = initial_phis.copy()
    momentum = initial_momentum.copy()
    projection_iterations = 0
    position_failures = 0
    momentum_failures = 0
    completed_steps = 0
    max_residual = _constraint_norm(phis, k)

    for _ in range(num_steps):
        potential_gradient = _grad_potential_kernel(phis, mu, sigma, mu_star, sigma_star, k)
        if not np.all(np.isfinite(potential_gradient)):
            return phis, momentum, False, projection_iterations, 1, 0, max_residual, completed_steps
        unconstrained_half = momentum - 0.5 * step_size * potential_gradient
        trial = phis + step_size * unconstrained_half
        old_phis = phis.copy()
        phis, ok, iterations, residual, multiplier_0, multiplier_1 = _project_position_fixed_direction_kernel(
            trial,
            old_phis,
            k,
            projection_tolerance,
            projection_max_iterations,
            jacobian_tolerance,
            projection_damping,
            projection_line_search,
        )
        projection_iterations += iterations
        max_residual = max(max_residual, residual)
        if not ok:
            position_failures += 1
            return (
                phis,
                momentum,
                False,
                projection_iterations,
                position_failures,
                momentum_failures,
                max_residual,
                completed_steps,
            )

        half = np.empty_like(momentum)
        for i in range(momentum.size):
            correction = (
                -math.sin(old_phis[i]) * multiplier_0
                + math.cos(old_phis[i]) * multiplier_1
            ) / step_size
            half[i] = unconstrained_half[i] + correction

        new_gradient = _grad_potential_kernel(phis, mu, sigma, mu_star, sigma_star, k)
        if not np.all(np.isfinite(new_gradient)):
            position_failures += 1
            return (
                phis,
                momentum,
                False,
                projection_iterations,
                position_failures,
                momentum_failures,
                max_residual,
                completed_steps,
            )
        proposed_momentum = half - 0.5 * step_size * new_gradient
        momentum, momentum_ok = _project_momentum_kernel(proposed_momentum, phis, jacobian_tolerance)
        residual = _constraint_norm(phis, k)
        tangent_residual = _constraint_tangent_norm(momentum, phis)
        max_residual = max(max_residual, residual)
        if not momentum_ok or residual > projection_tolerance or tangent_residual > tangent_tolerance:
            momentum_failures += 1
            return (
                phis,
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
        phis,
        momentum,
        True,
        projection_iterations,
        position_failures,
        momentum_failures,
        max_residual,
        completed_steps,
    )


@njit(cache=True)
def _update_parameters_from_variates(
    mu,
    log_sigma,
    x,
    k,
    proposal_std_mu,
    proposal_std_logsigma,
    prior_mean,
    prior_std,
    prior_logsigma_mean,
    prior_logsigma_std,
    normal_mu,
    normal_logsigma,
    uniform,
):
    proposed_mu = mu + proposal_std_mu * normal_mu
    proposed_log_sigma = log_sigma + proposal_std_logsigma * normal_logsigma
    current = _parameter_log_target(
        mu, log_sigma, x, k, prior_mean, prior_std, prior_logsigma_mean, prior_logsigma_std
    )
    proposed = _parameter_log_target(
        proposed_mu,
        proposed_log_sigma,
        x,
        k,
        prior_mean,
        prior_std,
        prior_logsigma_mean,
        prior_logsigma_std,
    )
    accepted = math.log(max(uniform, EPS_U)) < min(0.0, proposed - current)
    if accepted:
        return proposed_mu, proposed_log_sigma, True
    return mu, log_sigma, False


def _potential_energy(phis, mu, sigma, mu_star, sigma_star, k):
    return float(
        _potential_energy_kernel(
            np.asarray(phis, dtype=np.float64), mu, sigma, mu_star, sigma_star, k
        )
    )


def _grad_potential(phis, mu, sigma, mu_star, sigma_star, k):
    return _grad_potential_kernel(
        np.asarray(phis, dtype=np.float64), mu, sigma, mu_star, sigma_star, k
    )


def _project_momentum(momentum, phis, gram_tol=1e-12):
    projected, ok = _project_momentum_kernel(
        np.asarray(momentum, dtype=np.float64), np.asarray(phis, dtype=np.float64), gram_tol
    )
    return projected, bool(ok)


def _rattle_trajectory(
    initial_phis,
    initial_momentum,
    mu,
    sigma,
    mu_star,
    sigma_star,
    k,
    step_size,
    num_steps,
    projection_tolerance=1e-10,
    projection_max_iterations=25,
    jacobian_tolerance=1e-12,
    projection_damping=1.0,
    projection_line_search=True,
    tangent_tolerance=1e-8,
):
    result = _rattle_trajectory_kernel(
        np.asarray(initial_phis, dtype=np.float64),
        np.asarray(initial_momentum, dtype=np.float64),
        float(mu),
        float(sigma),
        float(mu_star),
        float(sigma_star),
        float(k),
        float(step_size),
        int(num_steps),
        float(projection_tolerance),
        int(projection_max_iterations),
        float(jacobian_tolerance),
        float(projection_damping),
        bool(projection_line_search),
        float(tangent_tolerance),
    )
    phis, momentum, ok, iterations, pos_failures, mom_failures, max_residual, completed = result
    diagnostics = {
        "projection_iterations": int(iterations),
        "position_projection_failures": int(pos_failures),
        "momentum_projection_failures": int(mom_failures),
        "max_projection_residual": float(max_residual),
        "completed_steps": int(completed),
    }
    return phis, momentum, bool(ok), diagnostics


def run_rattle(key, mu_star, sigma_star, params, verbose=True):
    """Sample ``(mu, sigma, x)`` with fully refreshed angle-space RATTLE."""
    _validate_common(mu_star, sigma_star, params)
    iterations = int(params["num_iterations_T"])
    n = int(params["n"])
    k = float(params["k"])
    step_size = float(params.get("rattle_step_size", 0.01))
    num_steps = int(params.get("rattle_num_steps", 2))
    projection_tolerance = float(params.get("rattle_proj_tol", 1e-10))
    projection_max_iterations = int(params.get("rattle_proj_max_iters", 25))
    jacobian_tolerance = float(params.get("rattle_grad_tol", 1e-12))
    reverse_position_tolerance = float(params.get("rattle_reverse_position_tol", 5e-3))
    projection_damping = float(params.get("rattle_proj_damping", 1.0))
    projection_line_search = bool(params.get("rattle_proj_line_search", True))
    tangent_tolerance = float(params.get("rattle_tangent_tol", 1e-8))
    if iterations <= 0 or step_size <= 0.0 or num_steps <= 0:
        raise ValueError("RATTLE requires positive iterations, step size, and number of steps")
    if not bool(params.get("reverse_check", True)):
        raise ValueError("Student location-scale RATTLE requires reverse_check=True")
    if not bool(params.get("rattle_include_gram_correction", True)):
        raise ValueError("Student location-scale RATTLE requires the coarea Gram correction")
    if str(params.get("rattle_projection_mode", "paper_fixed_direction")) != "paper_fixed_direction":
        raise ValueError("Only the paper_fixed_direction RATTLE projection is supported")

    proposal_std_mu = float(params.get("proposal_std_mu", 0.3))
    proposal_std_logsigma = float(params.get("proposal_std_logsigma", 0.2))
    prior_mean = float(params.get("prior_mean", 0.0))
    prior_std = float(params.get("prior_std", 5.0))
    prior_logsigma_mean = float(params.get("prior_logsigma_mean", 0.0))
    prior_logsigma_std = float(params.get("prior_logsigma_std", 1.0))
    if proposal_std_mu < 0.0 or proposal_std_logsigma < 0.0:
        raise ValueError("parameter proposal standard deviations must be nonnegative")

    phis = init_angles(n, k)
    mu_chain = np.empty(iterations + 1, dtype=np.float64)
    sigma_chain = np.empty(iterations + 1, dtype=np.float64)
    store_x_chain = bool(params.get("store_x_chain", True))
    if store_x_chain:
        x_chain = np.empty((iterations + 1, n), dtype=np.float64)
        x_chain[0] = _angles_to_x(phis, float(mu_star), float(sigma_star), k)
    mu_chain[0] = float(mu_star)
    sigma_chain[0] = float(sigma_star)
    rng = rng_from_key(key)

    parameter_accepts = 0
    x_accepts = 0
    forward_failures = 0
    reverse_failures = 0
    initial_momentum_failures = 0
    projection_iterations_total = 0
    max_projection_residual = _constraint_norm(phis, k)
    max_reverse_position_error = 0.0
    delta_h_abs_sum = 0.0
    delta_h_max_abs = 0.0
    delta_h_count = 0

    iterator = range(1, iterations + 1)
    if verbose:
        iterator = tqdm(iterator, desc="RATTLE (Student loc-scale)")
    for iteration in iterator:
        x_current = _angles_to_x(phis, float(mu_star), float(sigma_star), k)
        mu, log_sigma, parameter_accepted = _update_parameters_from_variates(
            mu_chain[iteration - 1],
            math.log(sigma_chain[iteration - 1]),
            x_current,
            k,
            proposal_std_mu,
            proposal_std_logsigma,
            prior_mean,
            prior_std,
            prior_logsigma_mean,
            prior_logsigma_std,
            float(rng.normal()),
            float(rng.normal()),
            float(rng.uniform(EPS_U, 1.0)),
        )
        sigma = math.exp(log_sigma)
        mu_chain[iteration] = mu
        sigma_chain[iteration] = sigma
        parameter_accepts += int(parameter_accepted)

        momentum = rng.standard_normal(n)
        momentum, momentum_ok = _project_momentum_kernel(momentum, phis, jacobian_tolerance)
        if not momentum_ok:
            initial_momentum_failures += 1
            if store_x_chain:
                x_chain[iteration] = x_current
            continue

        initial_energy = _potential_energy_kernel(phis, mu, sigma, mu_star, sigma_star, k)
        initial_energy += 0.5 * float(np.dot(momentum, momentum))
        proposed_phis, proposed_momentum, forward_ok, forward_diag = _rattle_trajectory(
            phis,
            momentum,
            mu,
            sigma,
            mu_star,
            sigma_star,
            k,
            step_size,
            num_steps,
            projection_tolerance,
            projection_max_iterations,
            jacobian_tolerance,
            projection_damping,
            projection_line_search,
            tangent_tolerance,
        )
        projection_iterations_total += forward_diag["projection_iterations"]
        max_projection_residual = max(
            max_projection_residual, forward_diag["max_projection_residual"]
        )
        if not forward_ok:
            forward_failures += 1
            if store_x_chain:
                x_chain[iteration] = x_current
            continue

        proposed_energy = _potential_energy_kernel(
            proposed_phis, mu, sigma, mu_star, sigma_star, k
        ) + 0.5 * float(np.dot(proposed_momentum, proposed_momentum))
        delta_h = proposed_energy - initial_energy
        if not math.isfinite(delta_h):
            forward_failures += 1
            if store_x_chain:
                x_chain[iteration] = x_current
            continue
        delta_h_abs_sum += abs(delta_h)
        delta_h_max_abs = max(delta_h_max_abs, abs(delta_h))
        delta_h_count += 1

        reverse_phis, _, reverse_ok, reverse_diag = _rattle_trajectory(
            proposed_phis,
            -proposed_momentum,
            mu,
            sigma,
            mu_star,
            sigma_star,
            k,
            step_size,
            num_steps,
            projection_tolerance,
            projection_max_iterations,
            jacobian_tolerance,
            projection_damping,
            projection_line_search,
            tangent_tolerance,
        )
        projection_iterations_total += reverse_diag["projection_iterations"]
        max_projection_residual = max(
            max_projection_residual, reverse_diag["max_projection_residual"]
        )
        reverse_error = float(np.linalg.norm(reverse_phis - phis))
        max_reverse_position_error = max(max_reverse_position_error, reverse_error)
        if not reverse_ok or reverse_error > reverse_position_tolerance:
            reverse_failures += 1
            if store_x_chain:
                x_chain[iteration] = x_current
            continue

        if math.log(float(rng.uniform(EPS_U, 1.0))) < min(0.0, -delta_h):
            phis = proposed_phis
            x_accepts += 1
        if store_x_chain:
            x_chain[iteration] = _angles_to_x(phis, float(mu_star), float(sigma_star), k)

    x_final = _angles_to_x(phis, float(mu_star), float(sigma_star), k)
    diagnostics = {
        "proposals": iterations,
        "forward_failures": forward_failures,
        "reverse_failures": reverse_failures,
        "initial_momentum_projection_failures": initial_momentum_failures,
        "projection_iterations_total": projection_iterations_total,
        "max_projection_residual": max_projection_residual,
        "max_reverse_position_error": max_reverse_position_error,
        "delta_h_mean_abs": delta_h_abs_sum / max(delta_h_count, 1),
        "delta_h_max_abs": delta_h_max_abs,
        "include_gram_correction": True,
        "include_mle_jacobian": True,
        "projection_mode": "paper_fixed_direction",
        "reverse_check": True,
    }
    output = {
        "mu_chain": mu_chain,
        "sigma_chain": sigma_chain,
        "x_final": x_final,
        "parameter_acceptance_rate": parameter_accepts / iterations,
        "x_acceptance_rate": x_accepts / iterations,
        "projection_diagnostics": diagnostics,
        "mle_selection_condition": "strict_local_maximum; global_uniqueness_assumed",
        "store_x_chain": store_x_chain,
    }
    if store_x_chain:
        output["x_chain"] = x_chain
    return output


__all__ = ["run_rattle"]
