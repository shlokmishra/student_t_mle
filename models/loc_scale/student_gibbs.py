"""Student-t location-scale sampler conditioned on a released joint MLE.

The released statistic is ``(mu_star, sigma_star)`` and the unknown model
parameter is ``(mu, sigma)``.  Following ``docs/loc_scale.tex``, latent data
are represented by angles

    y = (x - mu_star) / sigma_star = sqrt(k) * tan(phi / 2),

so the two released-score constraints reduce to a fixed vector resultant.
Each latent sweep updates disjoint triples.  One angle ``t`` is updated by a
bounded slice sampler using the exact one-dimensional density from the
document; the other two angles are reconstructed in closed form.

The implementation includes the MLE-conditioning ``abs(det(H_C))`` factor
and restricts states to roots that are strict local likelihood maxima.  As in
the derivation, global/selected-MLE uniqueness is an assumption; it is
reported explicitly in the returned metadata.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from models.loc.student.mu_mh import rng_from_key


ANGLE_MARGIN = 1e-10
GEOMETRY_FLOOR = 1e-14
EPS_U = 1e-12


def _validate_common(mu_star, sigma_star, params):
    n = int(params["n"])
    k = float(params["k"])
    if n < 3:
        raise ValueError("Student location-scale sampling requires n >= 3")
    if not math.isfinite(mu_star) or not math.isfinite(sigma_star) or sigma_star <= 0.0:
        raise ValueError("mu_star must be finite and sigma_star must be positive")
    if not math.isfinite(k) or k < 1.0:
        raise ValueError("Student location-scale sampling currently requires k >= 1")
    if float(params.get("prior_std", 5.0)) <= 0.0:
        raise ValueError("prior_std must be positive")
    if float(params.get("prior_logsigma_std", 1.0)) <= 0.0:
        raise ValueError("prior_logsigma_std must be positive")
    if not bool(params.get("include_mle_jacobian", True)):
        raise ValueError("Student location-scale sampling requires the MLE Jacobian")
    if not bool(params.get("require_local_mle", True)):
        raise ValueError("Student location-scale sampling requires the local-MLE condition")


@njit(cache=True)
def _wrap_angle(phi):
    """Wrap a reconstructed angle to the open principal interval."""
    wrapped = (phi + math.pi) % (2.0 * math.pi) - math.pi
    return wrapped


@njit(cache=True)
def _angle_to_y(phi, k):
    return math.sqrt(k) * math.tan(0.5 * phi)


@njit(cache=True)
def _y_to_angle(y, k):
    return 2.0 * math.atan(y / math.sqrt(k))


@njit(cache=True)
def _angles_to_x(phis, mu_star, sigma_star, k):
    x = np.empty(phis.size, dtype=np.float64)
    for i in range(phis.size):
        x[i] = mu_star + sigma_star * _angle_to_y(phis[i], k)
    return x


@njit(cache=True)
def _score_residual_angles(phis, k):
    target_cos = phis.size * (k - 1.0) / (k + 1.0)
    sum_cos = 0.0
    sum_sin = 0.0
    for phi in phis:
        sum_cos += math.cos(phi)
        sum_sin += math.sin(phi)
    return sum_cos - target_cos, sum_sin


def score_residual(x, mu_star, sigma_star, k):
    """Return the two convenient Student score residuals in angle space."""
    y = (np.asarray(x, dtype=np.float64) - float(mu_star)) / float(sigma_star)
    phis = np.asarray([_y_to_angle(float(value), float(k)) for value in y])
    return np.asarray(_score_residual_angles(phis, float(k)), dtype=np.float64)


def init_angles(n, k):
    """Construct a regular, exactly score-constrained initial angle vector."""
    n = int(n)
    k = float(k)
    if n < 3 or k < 1.0:
        raise ValueError("init_angles requires n >= 3 and k >= 1")

    target_mean = (k - 1.0) / (k + 1.0)
    pair_mean = target_mean
    pair_count = n // 2 if n % 2 == 0 else (n - 3) // 2

    room = min(1.0 - pair_mean, pair_mean + 1.0)
    spread = 0.2 * max(room, 0.0)
    if pair_count == 1 or spread == 0.0:
        cosines = np.full(pair_count, pair_mean, dtype=np.float64)
    else:
        cosines = pair_mean + np.linspace(-spread, spread, pair_count)

    phis = []
    for cosine in cosines:
        alpha = math.acos(float(np.clip(cosine, -1.0, 1.0)))
        phis.extend((alpha, -alpha))
    if n % 2:
        # Complete the odd state with a regular three-vector construction of
        # resultant (3 * target_mean, 0); this avoids the excluded angle pi.
        free = math.acos(target_mean)
        pair_1, pair_2, _, feasible = _reconstruct_pair(3.0 * target_mean, 0.0, free)
        if not feasible:
            raise RuntimeError("failed to construct the odd-n Student initial state")
        phis.extend((free, pair_1, pair_2))
    result = np.asarray(phis, dtype=np.float64)
    residual_cos, residual_sin = _score_residual_angles(result, k)
    if abs(residual_cos) > 1e-12 or abs(residual_sin) > 1e-12:
        raise RuntimeError("Student initial-state construction lost the score constraints")
    return result


def init_x(mu_star, sigma_star, n, k):
    """Construct latent data satisfying both released Student score equations."""
    return _angles_to_x(init_angles(n, k), float(mu_star), float(sigma_star), float(k))


@njit(cache=True)
def _geometry_terms_from_y(y, k):
    denom = k + y * y
    denom2 = denom * denom
    a = (k - y * y) / denom2
    b = 2.0 * k * y / denom2
    return a, y * a, b, y * b


@njit(cache=True)
def _geometry_sums(phis, k):
    sums = np.zeros(4, dtype=np.float64)
    for phi in phis:
        y = _angle_to_y(phi, k)
        terms = _geometry_terms_from_y(y, k)
        for j in range(4):
            sums[j] += terms[j]
    return sums


@njit(cache=True)
def _det_h_numerator(sums):
    return sums[0] * sums[3] - sums[1] * sums[2]


@njit(cache=True)
def _strict_local_mle(sums):
    # At a score root the likelihood Hessian is negative definite iff the
    # location curvature is negative and its determinant is positive.
    return sums[0] > GEOMETRY_FLOOR and _det_h_numerator(sums) > GEOMETRY_FLOOR


@njit(cache=True)
def _log_angle_density(phi, mu, sigma, mu_star, sigma_star, k):
    if not (-math.pi + ANGLE_MARGIN < phi < math.pi - ANGLE_MARGIN):
        return -math.inf
    cosine_half = math.cos(0.5 * phi)
    if abs(cosine_half) <= 0.0 or sigma <= 0.0:
        return -math.inf
    y = _angle_to_y(phi, k)
    standardized = (mu_star + sigma_star * y - mu) / sigma
    log_student = -0.5 * (k + 1.0) * math.log1p(standardized * standardized / k)
    log_jacobian = 0.5 * math.log(k) - math.log(2.0) - 2.0 * math.log(abs(cosine_half))
    return log_student + log_jacobian


@njit(cache=True)
def _reconstruct_pair(resultant_x, resultant_y, t):
    vx = resultant_x - math.cos(t)
    vy = resultant_y - math.sin(t)
    radius = math.sqrt(vx * vx + vy * vy)
    if not (GEOMETRY_FLOOR < radius < 2.0 - GEOMETRY_FLOOR):
        return 0.0, 0.0, 0.0, False
    beta = math.atan2(vy, vx)
    gamma = math.acos(max(-1.0, min(1.0, 0.5 * radius)))
    pair_jacobian = abs(math.sin(2.0 * gamma))
    if pair_jacobian <= GEOMETRY_FLOOR:
        return 0.0, 0.0, 0.0, False
    phi_j = _wrap_angle(beta + gamma)
    phi_l = _wrap_angle(beta - gamma)
    if not (
        -math.pi + ANGLE_MARGIN < phi_j < math.pi - ANGLE_MARGIN
        and -math.pi + ANGLE_MARGIN < phi_l < math.pi - ANGLE_MARGIN
    ):
        return 0.0, 0.0, 0.0, False
    return phi_j, phi_l, pair_jacobian, True


@njit(cache=True)
def _triple_log_target(
    t,
    resultant_x,
    resultant_y,
    base_sums,
    mu,
    sigma,
    mu_star,
    sigma_star,
    k,
):
    phi_j, phi_l, pair_jacobian, feasible = _reconstruct_pair(resultant_x, resultant_y, t)
    if not feasible:
        return -math.inf, phi_j, phi_l

    candidate_sums = base_sums.copy()
    for phi in (t, phi_j, phi_l):
        terms = _geometry_terms_from_y(_angle_to_y(phi, k), k)
        for index in range(4):
            candidate_sums[index] += terms[index]
    if not _strict_local_mle(candidate_sums):
        return -math.inf, phi_j, phi_l

    log_density = (
        _log_angle_density(t, mu, sigma, mu_star, sigma_star, k)
        + _log_angle_density(phi_j, mu, sigma, mu_star, sigma_star, k)
        + _log_angle_density(phi_l, mu, sigma, mu_star, sigma_star, k)
    )
    det_h = _det_h_numerator(candidate_sums) / sigma_star
    value = log_density - math.log(pair_jacobian) + math.log(abs(det_h))
    if not math.isfinite(value):
        return -math.inf, phi_j, phi_l
    return value, phi_j, phi_l


@njit(cache=True)
def _slice_update_triple(
    phis,
    index_i,
    index_j,
    index_l,
    mu,
    sigma,
    mu_star,
    sigma_star,
    k,
    max_shrink_steps,
):
    phi_i = phis[index_i]
    old_j = phis[index_j]
    old_l = phis[index_l]
    resultant_x = math.cos(phi_i) + math.cos(old_j) + math.cos(old_l)
    resultant_y = math.sin(phi_i) + math.sin(old_j) + math.sin(old_l)

    full_sums = _geometry_sums(phis, k)
    base_sums = full_sums.copy()
    for phi in (phi_i, old_j, old_l):
        terms = _geometry_terms_from_y(_angle_to_y(phi, k), k)
        for index in range(4):
            base_sums[index] -= terms[index]

    log_current, _, _ = _triple_log_target(
        phi_i, resultant_x, resultant_y, base_sums, mu, sigma, mu_star, sigma_star, k
    )
    if not math.isfinite(log_current):
        return False, 0

    log_slice = log_current + math.log(max(np.random.random(), EPS_U))
    left = -math.pi + ANGLE_MARGIN
    right = math.pi - ANGLE_MARGIN
    for shrink_steps in range(1, max_shrink_steps + 1):
        candidate = left + (right - left) * np.random.random()
        log_candidate, phi_j, phi_l = _triple_log_target(
            candidate, resultant_x, resultant_y, base_sums, mu, sigma, mu_star, sigma_star, k
        )
        if log_candidate >= log_slice:
            phis[index_i] = candidate
            if np.random.random() < 0.5:
                phis[index_j], phis[index_l] = phi_j, phi_l
            else:
                phis[index_j], phis[index_l] = phi_l, phi_j
            return True, shrink_steps
        if candidate < phi_i:
            left = candidate
        else:
            right = candidate
    return False, max_shrink_steps


@njit(cache=True)
def _parameter_log_target(
    mu,
    log_sigma,
    x,
    k,
    prior_mean,
    prior_std,
    prior_logsigma_mean,
    prior_logsigma_std,
):
    sigma = math.exp(log_sigma)
    total = -x.size * log_sigma
    for value in x:
        standardized = (value - mu) / sigma
        total -= 0.5 * (k + 1.0) * math.log1p(standardized * standardized / k)
    z_mu = (mu - prior_mean) / prior_std
    z_sigma = (log_sigma - prior_logsigma_mean) / prior_logsigma_std
    return total - 0.5 * z_mu * z_mu - 0.5 * z_sigma * z_sigma


@njit(cache=True)
def _update_parameters(
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
):
    proposed_mu = mu + proposal_std_mu * np.random.normal()
    proposed_log_sigma = log_sigma + proposal_std_logsigma * np.random.normal()
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
    accepted = math.log(max(np.random.random(), EPS_U)) < min(0.0, proposed - current)
    if accepted:
        return proposed_mu, proposed_log_sigma, True
    return mu, log_sigma, False


@njit(cache=True)
def _run_gibbs_kernel(
    seed,
    initial_phis,
    mu_star,
    sigma_star,
    iterations,
    k,
    proposal_std_mu,
    proposal_std_logsigma,
    prior_mean,
    prior_std,
    prior_logsigma_mean,
    prior_logsigma_std,
    max_slice_shrink_steps,
    store_x_chain,
):
    np.random.seed(seed)
    n = initial_phis.size
    mu_chain = np.empty(iterations + 1, dtype=np.float64)
    sigma_chain = np.empty(iterations + 1, dtype=np.float64)
    if store_x_chain:
        x_chain = np.empty((iterations + 1, n), dtype=np.float64)
    else:
        x_chain = np.empty((0, 0), dtype=np.float64)

    phis = initial_phis.copy()
    mu_chain[0] = mu_star
    sigma_chain[0] = sigma_star
    if store_x_chain:
        x_chain[0] = _angles_to_x(phis, mu_star, sigma_star, k)

    parameter_accepts = 0
    slice_accepts = 0
    slice_attempts = 0
    shrink_steps_total = 0
    max_score_residual = 0.0
    permutation = np.empty(n, dtype=np.int64)

    for iteration in range(1, iterations + 1):
        x = _angles_to_x(phis, mu_star, sigma_star, k)
        mu, log_sigma, accepted = _update_parameters(
            mu_chain[iteration - 1],
            math.log(sigma_chain[iteration - 1]),
            x,
            k,
            proposal_std_mu,
            proposal_std_logsigma,
            prior_mean,
            prior_std,
            prior_logsigma_mean,
            prior_logsigma_std,
        )
        parameter_accepts += int(accepted)
        sigma = math.exp(log_sigma)
        mu_chain[iteration] = mu
        sigma_chain[iteration] = sigma

        permutation[:] = np.random.permutation(n)
        for start in range(0, n - 2, 3):
            accepted_slice, shrink_steps = _slice_update_triple(
                phis,
                permutation[start],
                permutation[start + 1],
                permutation[start + 2],
                mu,
                sigma,
                mu_star,
                sigma_star,
                k,
                max_slice_shrink_steps,
            )
            slice_attempts += 1
            slice_accepts += int(accepted_slice)
            shrink_steps_total += shrink_steps

        residual_cos, residual_sin = _score_residual_angles(phis, k)
        residual = math.sqrt(residual_cos * residual_cos + residual_sin * residual_sin)
        max_score_residual = max(max_score_residual, residual)
        if store_x_chain:
            x_chain[iteration] = _angles_to_x(phis, mu_star, sigma_star, k)

    x_final = _angles_to_x(phis, mu_star, sigma_star, k)
    return (
        mu_chain,
        sigma_chain,
        x_chain,
        x_final,
        parameter_accepts,
        slice_accepts,
        slice_attempts,
        shrink_steps_total,
        max_score_residual,
    )


def run_gibbs(key, mu_star, sigma_star, params, verbose=True):
    """Run Student-t location-scale Metropolis-within-Gibbs.

    ``t`` in the derivation is the first angle of each selected triple.  It is
    updated with a bounded slice step on ``(-pi, pi)``; it is not an additional
    model parameter and therefore has no separate chain.
    """
    _validate_common(mu_star, sigma_star, params)
    iterations = int(params["num_iterations_T"])
    if iterations <= 0:
        raise ValueError("num_iterations_T must be positive")
    if float(params.get("proposal_std_mu", 0.3)) < 0.0 or float(
        params.get("proposal_std_logsigma", 0.2)
    ) < 0.0:
        raise ValueError("parameter proposal standard deviations must be nonnegative")
    max_slice_shrink_steps = int(params.get("slice_max_shrink_steps", 200))
    if max_slice_shrink_steps <= 0:
        raise ValueError("slice_max_shrink_steps must be positive")

    initial = init_angles(int(params["n"]), float(params["k"]))
    if verbose:
        print(f"Gibbs (Student loc-scale): running {iterations} iterations", flush=True)
    result = _run_gibbs_kernel(
        int(rng_from_key(key).integers(0, np.iinfo(np.int32).max)),
        initial,
        float(mu_star),
        float(sigma_star),
        iterations,
        float(params["k"]),
        float(params.get("proposal_std_mu", 0.3)),
        float(params.get("proposal_std_logsigma", 0.2)),
        float(params.get("prior_mean", 0.0)),
        float(params.get("prior_std", 5.0)),
        float(params.get("prior_logsigma_mean", 0.0)),
        float(params.get("prior_logsigma_std", 1.0)),
        max_slice_shrink_steps,
        bool(params.get("store_x_chain", True)),
    )
    if verbose:
        print("Gibbs (Student loc-scale): complete", flush=True)

    (
        mu_chain,
        sigma_chain,
        x_chain,
        x_final,
        parameter_accepts,
        slice_accepts,
        slice_attempts,
        shrink_steps_total,
        max_score_residual,
    ) = result
    output = {
        "mu_chain": mu_chain,
        "sigma_chain": sigma_chain,
        "x_final": x_final,
        "parameter_acceptance_rate": parameter_accepts / iterations,
        "slice_acceptance_rate": slice_accepts / max(slice_attempts, 1),
        "mean_slice_shrink_steps": shrink_steps_total / max(slice_attempts, 1),
        "max_score_residual": max_score_residual,
        "latent_update": "triple_angle_slice",
        "mle_jacobian_included": True,
        "mle_selection_condition": "strict_local_maximum; global_uniqueness_assumed",
        "store_x_chain": bool(params.get("store_x_chain", True)),
    }
    if bool(params.get("store_x_chain", True)):
        output["x_chain"] = x_chain
    return output


__all__ = [
    "init_angles",
    "init_x",
    "run_gibbs",
    "score_residual",
]
