"""Exact local affine-normalization MH-within-Gibbs for 1D location models.

The latent proposal adds isotropic Gaussian noise to the current dataset and
subtracts the proposal's selected location MLE.  The forward and reverse
affine-orbit factors are symmetric and cancel with the coarea Jacobians.  This
implementation is intentionally separate from the older score-preserving
Gibbs kernels.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from models.loc.student.mu_mh import seed_from_key
from models.loc.student.mle import global_mle as student_global_location_mle
from models.loc.student.mle import loglik as _student_location_loglik
from models.loc.student.mle import score as _student_location_score
from models.loc.student.mle import stationary_points as student_location_stationary_points


EPS_U = 1e-15
_MODEL_ALIASES = {
    "gaussian": "normal",
    "normal": "normal",
    "logistic": "logistic",
    "student": "student",
    "student_t": "student",
    "student-t": "student",
}


def _model_name(model):
    name = str(model).lower()
    if name not in _MODEL_ALIASES:
        raise ValueError("affine model must be one of normal, logistic, or student")
    return _MODEL_ALIASES[name]


def _validated_sample(data):
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("affine normalization requires a non-empty 1D sample")
    if not np.all(np.isfinite(x)):
        raise ValueError("affine normalization data must be finite")
    return x


def centered_sum_squares(z):
    """Return ``sum((z - mean(z))**2)`` without assuming ``mean(z) == 0``."""
    values = _validated_sample(z)
    centered = values - float(np.mean(values))
    return float(np.dot(centered, centered))


def student_location_score(mu, data, k):
    """Student location score, retaining the affine module's legacy argument order."""
    return _student_location_score(data, mu, k)


def student_location_loglik(mu, data, k):
    """Student log likelihood, retaining the affine module's legacy argument order."""
    return _student_location_loglik(data, mu, k)


def global_location_mle(data, model, *, k=None, scale=1.0):
    """Selected/global location MLE used throughout the affine experiment."""
    x = _validated_sample(data)
    name = _model_name(model)
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("known model scale must be positive")
    if name == "normal":
        return float(np.mean(x))
    if name == "logistic":
        def score(mu):
            return float(np.sum(np.tanh((x - mu) / (2.0 * scale))))

        if np.ptp(x) == 0.0:
            return float(x[0])
        return float(brentq(score, float(np.min(x)), float(np.max(x)), xtol=1e-13, rtol=1e-14))
    if k is None:
        raise ValueError("student affine normalization requires k")
    if scale != 1.0:
        standardized = x / scale
        return scale * student_global_location_mle(standardized, float(k))
    return student_global_location_mle(x, float(k))


def location_loglik(data, mu, model, *, k=None, scale=1.0):
    """Log p_mu(data), omitting constants independent of the latent state."""
    x = _validated_sample(data)
    name = _model_name(model)
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("known model scale must be positive")
    residual = (x - float(mu)) / scale
    if name == "normal":
        return float(-0.5 * np.dot(residual, residual))
    if name == "logistic":
        return float(np.sum(-residual - 2.0 * np.logaddexp(0.0, -residual)))
    if k is None:
        raise ValueError("student affine normalization requires k")
    k = float(k)
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError("Student degrees of freedom k must be positive")
    return float(-0.5 * (k + 1.0) * np.log1p(residual * residual / k).sum())


def affine_log_acceptance_ratio(x, x_proposed, mu_current, model, *, k=None, scale=1.0):
    """Exact local-MH log ratio after affine normalization."""
    x = _validated_sample(x)
    x_proposed = _validated_sample(x_proposed)
    if x.shape != x_proposed.shape:
        raise ValueError("current and proposed latent states must have the same shape")
    return float(
        location_loglik(x_proposed, mu_current, model, k=k, scale=scale)
        - location_loglik(x, mu_current, model, k=k, scale=scale)
    )


def normalized_affine_proposal(u, mu_star, model, *, k=None, scale=1.0):
    """Normalize an unconstrained vector and map it to the released MLE."""
    unconstrained = _validated_sample(u)
    proposal_mle = global_location_mle(unconstrained, model, k=k, scale=scale)
    z_proposed = unconstrained - proposal_mle
    return float(mu_star) + z_proposed


def get_benchmark_mle_samples(key, params, num_simulations=10000, verbose=False):
    """Simulate selected/global MLE errors for the affine sampler's KDE reference."""
    model = _model_name(params.get("model", params.get("affine_model", "student")))
    n = int(params["n"])
    if n < 1 or int(num_simulations) < 1:
        raise ValueError("n and num_simulations must be positive")
    scale = float(params.get("known_scale", params.get("scale", 1.0)))
    k = float(params["k"]) if model == "student" else None
    rng = _rng_from_key(key)
    if model == "normal":
        data = rng.normal(0.0, scale, size=(int(num_simulations), n))
    elif model == "logistic":
        data = rng.logistic(0.0, scale, size=(int(num_simulations), n))
    else:
        data = scale * rng.standard_t(k, size=(int(num_simulations), n))
    samples = np.empty(int(num_simulations), dtype=np.float64)
    for index, row in enumerate(data):
        samples[index] = global_location_mle(row, model, k=k, scale=scale)
        if verbose and (index + 1) % 10000 == 0:
            print(f"Affine benchmark MLE samples: {index + 1}/{num_simulations}", flush=True)
    return samples


def latent_affine_mh_step(
    x,
    mu_current,
    mu_star,
    model,
    tau_q,
    rng,
    *,
    k=None,
    scale=1.0,
    mle_tol=1e-8,
    return_diagnostics=False,
):
    """Take one exact local affine-normalized latent MH step."""
    current = _validated_sample(x)
    tau_q = float(tau_q)
    if not math.isfinite(tau_q) or tau_q <= 0.0:
        raise ValueError("tau_q must be positive")
    noise = np.asarray(rng.normal(0.0, tau_q, size=current.size), dtype=np.float64)
    u = current + noise
    proposed = normalized_affine_proposal(u, mu_star, model, k=k, scale=scale)
    proposed_mle = global_location_mle(proposed, model, k=k, scale=scale)
    mle_residual = abs(proposed_mle - float(mu_star))
    tolerance = float(mle_tol) * (1.0 + abs(float(mu_star)))
    if not math.isfinite(mle_residual) or mle_residual > tolerance:
        raise RuntimeError(
            f"affine proposal missed the selected-MLE manifold: residual={mle_residual:.3g}"
        )

    log_ratio = affine_log_acceptance_ratio(current, proposed, mu_current, model, k=k, scale=scale)
    accepted = math.log(max(float(rng.uniform()), EPS_U)) < min(0.0, log_ratio)
    state = proposed if accepted else current.copy()
    if not return_diagnostics:
        return state, accepted
    return state, accepted, {
        "log_acceptance_ratio": log_ratio,
        "proposal_mle": proposed_mle,
        "proposal_mle_residual": mle_residual,
        "centered_step_sq": centered_sum_squares(proposed - current),
        "noise_sq": float(np.dot(noise, noise)),
    }


def _mu_logposterior(mu, x, model, prior_mean, prior_std, *, k=None, scale=1.0):
    prior = -0.5 * ((float(mu) - prior_mean) / prior_std) ** 2
    return prior + location_loglik(x, mu, model, k=k, scale=scale)


def _update_mu_mh(mu, x, model, proposal_std, prior_mean, prior_std, rng, *, k=None, scale=1.0):
    candidate = float(mu) + float(proposal_std) * float(rng.normal())
    log_ratio = (
        _mu_logposterior(candidate, x, model, prior_mean, prior_std, k=k, scale=scale)
        - _mu_logposterior(mu, x, model, prior_mean, prior_std, k=k, scale=scale)
    )
    if math.isfinite(log_ratio) and math.log(max(float(rng.uniform()), EPS_U)) < min(0.0, log_ratio):
        return candidate, True
    return float(mu), False


def _rng_from_key(key):
    if isinstance(key, (int, np.integer)):
        return np.random.default_rng(int(key))
    return np.random.default_rng(seed_from_key(key))


def run_gibbs(key, mu_star, params, verbose=True):
    """Run local affine-normalized latent MH followed by the existing-style mu MH."""
    model = _model_name(params.get("model", params.get("affine_model", "student")))
    iterations = int(params["num_iterations_T"])
    n = int(params["n"])
    if iterations < 0 or n < 1:
        raise ValueError("num_iterations_T must be nonnegative and n must be positive")
    tau_q = float(params.get("tau_q", params.get("proposal_std_affine", np.nan)))
    if not math.isfinite(tau_q) or tau_q <= 0.0:
        raise ValueError("params must provide a positive tau_q")
    scale = float(params.get("known_scale", params.get("scale", 1.0)))
    k = float(params["k"]) if model == "student" else None
    prior_mean = float(params["prior_mean"])
    prior_std = float(params["prior_std"])
    proposal_std_mu = float(params["proposal_std_mu"])
    if prior_std <= 0.0 or proposal_std_mu <= 0.0:
        raise ValueError("prior_std and proposal_std_mu must be positive")
    mle_tol = float(params.get("mle_tolerance", 1e-8))
    store_x_chain = bool(params.get("store_x_chain", True))
    rng = _rng_from_key(key)

    x_current = np.full(n, float(mu_star), dtype=np.float64)
    initial_mu = global_location_mle(x_current, model, k=k, scale=scale)
    if abs(initial_mu - float(mu_star)) > mle_tol * (1.0 + abs(float(mu_star))):
        raise RuntimeError("affine Gibbs initializer is not on the selected-MLE manifold")
    mu_current = float(params.get("initial_mu", mu_star))
    mu_chain = np.empty(iterations + 1, dtype=np.float64)
    mu_chain[0] = mu_current
    x_chain = np.empty((iterations + 1, n), dtype=np.float64) if store_x_chain else None
    if x_chain is not None:
        x_chain[0] = x_current

    latent_accepts = 0
    mu_accepts = 0
    max_retained_residual = abs(initial_mu - float(mu_star))
    max_proposal_residual = 0.0
    if verbose:
        print(f"Affine Gibbs ({model}): running {iterations} iterations", flush=True)
    for iteration in range(1, iterations + 1):
        x_current, accepted, diagnostics = latent_affine_mh_step(
            x_current,
            mu_current,
            mu_star,
            model,
            tau_q,
            rng,
            k=k,
            scale=scale,
            mle_tol=mle_tol,
            return_diagnostics=True,
        )
        latent_accepts += int(accepted)
        max_proposal_residual = max(max_proposal_residual, diagnostics["proposal_mle_residual"])
        retained_mle = global_location_mle(x_current, model, k=k, scale=scale)
        max_retained_residual = max(max_retained_residual, abs(retained_mle - float(mu_star)))
        mu_current, accepted_mu = _update_mu_mh(
            mu_current,
            x_current,
            model,
            proposal_std_mu,
            prior_mean,
            prior_std,
            rng,
            k=k,
            scale=scale,
        )
        mu_accepts += int(accepted_mu)
        mu_chain[iteration] = mu_current
        if x_chain is not None:
            x_chain[iteration] = x_current
    if verbose:
        print(f"Affine Gibbs ({model}): complete", flush=True)

    denominator = max(iterations, 1)
    result = {
        "mu_chain": mu_chain,
        "x_final": x_current.copy(),
        "mu_acceptance_rate": mu_accepts / denominator,
        "latent_acceptance_rate": latent_accepts / denominator,
        "latent_acceptance_count": latent_accepts,
        "max_mle_residual": max_retained_residual,
        "max_proposal_mle_residual": max_proposal_residual,
        "affine_model": model,
        "tau_q": tau_q,
        "latent_proposal": "local_gaussian_affine_normalized",
        "mle_definition": "selected_global_location_mle",
        "store_x_chain": store_x_chain,
    }
    if x_chain is not None:
        result["x_chain"] = x_chain
    return result
