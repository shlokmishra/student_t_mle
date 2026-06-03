"""Posterior moment estimators for MLE-conditional location references.

These helpers are used to separate finite MLE-simulation error,
KDE/bandwidth sensitivity, and posterior integration sensitivity.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate, stats


def weighted_quantile(values, weights, quantiles):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if values.size == 0 or np.sum(weights) <= 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, values)


def raw_weighted_posterior_moments(z_samples, mu_star, prior_mean=0.0, prior_std=10.0):
    """Estimate posterior moments directly from simulated centered MLE errors.

    With particles ``mu_j = mu_star - z_j`` and weights proportional to the
    prior density at ``mu_j``, this estimates the posterior induced by the
    empirical distribution of the MLE error samples. It avoids KDE smoothing and
    avoids a posterior integration grid.
    """
    z_samples = np.asarray(z_samples, dtype=float)
    mu_particles = float(mu_star) - z_samples
    log_w = stats.norm.logpdf(mu_particles, loc=prior_mean, scale=prior_std)
    max_log_w = float(np.max(log_w))
    weights = np.exp(log_w - max_log_w)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0 or not np.isfinite(weight_sum):
        raise ValueError("Non-positive or non-finite raw weighted MC weights.")
    probs = weights / weight_sum
    mean = float(np.sum(probs * mu_particles))
    var = float(np.sum(probs * (mu_particles - mean) ** 2))
    q025, q50, q975 = weighted_quantile(mu_particles, weights, [0.025, 0.5, 0.975])
    weighted_ess = float(weight_sum * weight_sum / np.sum(weights * weights))
    # Empirical estimate of integral prior(mu_star - z) f_Z(z) dz.
    normalization_constant = float(np.mean(weights) * np.exp(max_log_w))
    return {
        "posterior_mean": mean,
        "posterior_var": var,
        "posterior_sd": float(np.sqrt(max(var, 0.0))),
        "posterior_q025": float(q025),
        "posterior_q50": float(q50),
        "posterior_q975": float(q975),
        "normalization_constant": normalization_constant,
        "weighted_ess": weighted_ess,
    }


def posterior_grid_bounds(mu_star, prior_mean, prior_std, z_samples, bound_multiplier=5.0):
    z_samples = np.asarray(z_samples, dtype=float)
    prior_radius = float(bound_multiplier) * float(prior_std)
    z_sd = float(np.std(z_samples, ddof=1)) if z_samples.size > 1 else 1.0
    if not np.isfinite(z_sd) or z_sd <= 0:
        z_sd = 1.0
    like_radius = float(bound_multiplier) * z_sd
    lo = min(float(prior_mean) - prior_radius, float(mu_star) - like_radius)
    hi = max(float(prior_mean) + prior_radius, float(mu_star) + like_radius)
    return float(lo), float(hi)


def _log_unnorm_fn(kde_backend, mu_star, prior_mean, prior_std) -> Callable[[np.ndarray], np.ndarray]:
    def log_unnorm(mu):
        mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
        log_prior = stats.norm.logpdf(mu_arr, loc=prior_mean, scale=prior_std)
        log_like = kde_backend.logpdf(float(mu_star) - mu_arr)
        out = log_prior + log_like
        return float(out[0]) if np.isscalar(mu) else out

    return log_unnorm


def _moments_from_grid(mu_grid, log_vals):
    log_vals = np.asarray(log_vals, dtype=float)
    log_max = float(np.max(log_vals))
    vals = np.exp(log_vals - log_max)
    scaled_z0 = float(np.trapezoid(vals, mu_grid))
    if scaled_z0 <= 0 or not np.isfinite(scaled_z0):
        raise ValueError("Grid normalization failed.")
    pdf = vals / scaled_z0
    mean = float(np.trapezoid(mu_grid * pdf, mu_grid))
    second = float(np.trapezoid(mu_grid * mu_grid * pdf, mu_grid))
    var = max(second - mean * mean, 0.0)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[:-1] + pdf[1:]) * np.diff(mu_grid) / 2.0)])
    cdf = cdf / max(cdf[-1], 1e-300)
    q025, q50, q975 = np.interp([0.025, 0.5, 0.975], cdf, mu_grid)
    return {
        "posterior_mean": mean,
        "posterior_var": float(var),
        "posterior_sd": float(np.sqrt(var)),
        "posterior_q025": float(q025),
        "posterior_q50": float(q50),
        "posterior_q975": float(q975),
        "normalization_constant": float(scaled_z0 * np.exp(log_max)),
    }


def kde_grid_posterior_moments(
    kde_backend,
    mu_star,
    prior_mean,
    prior_std,
    z_samples,
    n_grid=4000,
    bound_multiplier=5.0,
):
    lo, hi = posterior_grid_bounds(mu_star, prior_mean, prior_std, z_samples, bound_multiplier)
    mu_grid = np.linspace(lo, hi, int(n_grid))
    log_unnorm = _log_unnorm_fn(kde_backend, mu_star, prior_mean, prior_std)
    out = _moments_from_grid(mu_grid, log_unnorm(mu_grid))
    out["grid_lo"] = lo
    out["grid_hi"] = hi
    return out


def kde_quad_posterior_moments(
    kde_backend,
    mu_star,
    prior_mean,
    prior_std,
    z_samples,
    quantile_grid_size=4000,
    quantile_bound_multiplier=8.0,
):
    """Compute moments by adaptive quadrature and quantiles by a wide grid CDF.

    Quad is used for normalization, first raw moment, and second raw moment.
    Quantiles are reported from a wide grid using the quad normalization so the
    CSV remains comparable across estimator types without doing many nested
    adaptive CDF solves.
    """
    log_unnorm = _log_unnorm_fn(kde_backend, mu_star, prior_mean, prior_std)

    def unnorm(mu):
        val = np.exp(log_unnorm(mu))
        return float(val) if np.isfinite(val) else 0.0

    z0, _ = integrate.quad(unnorm, -np.inf, np.inf, limit=200)
    z1, _ = integrate.quad(lambda mu: mu * unnorm(mu), -np.inf, np.inf, limit=200)
    z2, _ = integrate.quad(lambda mu: mu * mu * unnorm(mu), -np.inf, np.inf, limit=200)
    if z0 <= 0 or not np.isfinite(z0):
        raise ValueError("Quad normalization failed.")
    mean = float(z1 / z0)
    var = float(max(z2 / z0 - mean * mean, 0.0))

    lo, hi = posterior_grid_bounds(mu_star, prior_mean, prior_std, z_samples, quantile_bound_multiplier)
    mu_grid = np.linspace(lo, hi, int(quantile_grid_size))
    vals = np.exp(log_unnorm(mu_grid)) / z0
    vals = np.maximum(vals, 0.0)
    cdf = np.concatenate([[0.0], np.cumsum((vals[:-1] + vals[1:]) * np.diff(mu_grid) / 2.0)])
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    q025, q50, q975 = np.interp([0.025, 0.5, 0.975], cdf, mu_grid)
    return {
        "posterior_mean": mean,
        "posterior_var": var,
        "posterior_sd": float(np.sqrt(var)),
        "posterior_q025": float(q025),
        "posterior_q50": float(q50),
        "posterior_q975": float(q975),
        "normalization_constant": float(z0),
        "grid_lo": lo,
        "grid_hi": hi,
    }


def gaussian_kde_gaussian_prior_moments(kde_backend, mu_star, prior_mean=0.0, prior_std=10.0):
    """Analytic mixture check for scipy Gaussian KDE with Gaussian prior.

    This applies only to the raw Gaussian KDE wrapper used for Scott/Silverman.
    The KDE is a mixture over z_j with common variance h^2. Under
    prior(mu)=N(m0,s0^2), each term gives a Gaussian posterior component for mu.
    """
    if not hasattr(kde_backend, "samples") or not hasattr(kde_backend, "bandwidth"):
        return None
    z = np.asarray(kde_backend.samples, dtype=float)
    h = float(kde_backend.bandwidth)
    if not np.isfinite(h) or h <= 0:
        return None
    obs_means = float(mu_star) - z
    prior_var = float(prior_std) ** 2
    like_var = h * h
    comp_var = 1.0 / (1.0 / prior_var + 1.0 / like_var)
    comp_mean = comp_var * (float(prior_mean) / prior_var + obs_means / like_var)
    log_mix_w = stats.norm.logpdf(obs_means, loc=prior_mean, scale=np.sqrt(prior_var + like_var))
    log_mix_w = log_mix_w - np.max(log_mix_w)
    mix_w = np.exp(log_mix_w)
    mix_w = mix_w / np.sum(mix_w)
    mean = float(np.sum(mix_w * comp_mean))
    second = float(np.sum(mix_w * (comp_var + comp_mean * comp_mean)))
    var = max(second - mean * mean, 0.0)
    q025, q50, q975 = weighted_quantile(comp_mean, mix_w, [0.025, 0.5, 0.975])
    return {
        "posterior_mean": mean,
        "posterior_var": float(var),
        "posterior_sd": float(np.sqrt(var)),
        "posterior_q025": float(q025),
        "posterior_q50": float(q50),
        "posterior_q975": float(q975),
        "normalization_constant": float(np.mean(np.exp(log_mix_w + np.max(log_mix_w)))),
        "weighted_ess": float(1.0 / np.sum(mix_w * mix_w)),
    }
