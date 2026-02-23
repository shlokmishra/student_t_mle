# Build normalized posterior p(theta | hat_theta = mu_star) from MLE samples at theta=0.
# Location family: p(hat_theta | theta=a) = p(hat_theta - a | 0). Posterior propto prior * likelihood.
#
# Bandwidth (bw_method): Scott/Silverman assume roughly Gaussian data. The MLE distribution
# for Student-t/Cauchy is heavy-tailed and non-Gaussian, so these rules often oversmooth
# (too large h → KDE too flat → posterior too diffuse). For moderate n (e.g. n=100), a
# small fixed bandwidth (e.g. 0.001) usually works better; the "right" value can depend on
# n and k (smaller n → MLE more spread → sometimes a slightly larger h is needed).

import numpy as np
import scipy.stats as stats
from scipy.integrate import quad


def _grid_bounds_for_normalisation(prior_mean, prior_std, mu_star, mles, n_sigma=5):
    """
    Choose grid bounds so that prior * KDE has negligible mass outside [lo, hi].
    Prior mass is in prior_mean +/- n_sigma*prior_std.
    Likelihood (KDE of mu_star - mu) has spread in mu of order std(mles) around mu_star.
    """
    mles = np.asarray(mles)
    prior_radius = n_sigma * prior_std
    like_scale = np.std(mles)
    if like_scale <= 0:
        like_radius = 5.0
    else:
        like_radius = n_sigma * like_scale
    lo = min(prior_mean - prior_radius, mu_star - like_radius)
    hi = max(prior_mean + prior_radius, mu_star + like_radius)
    return lo, hi


def validate_posterior_1d(posterior_pdf, lo=-20.0, hi=20.0, n_grid=5000):
    """
    Validate that the 1D posterior integrates to 1 (post = like * prior / norm_const).

    posterior_pdf: callable mu -> density (e.g. from get_normalized_posterior_pdf).
    lo, hi: integration bounds; n_grid: number of points.
    Returns: integral (float), should be close to 1.0.
    """
    mu_grid = np.linspace(lo, hi, n_grid)
    vals = np.maximum(posterior_pdf(mu_grid), 0.0)
    integral = float(np.trapezoid(vals, mu_grid))
    return integral


def get_normalized_posterior_pdf(mu_star, params, mle_samples, verbose=False, use_grid=True, n_grid=4000):
    """
    Return a callable for the normalized posterior density p(theta | hat_theta = mu_star).

    Uses KDE on mle_samples (from theta=0); likelihood at theta is KDE(mu_star - theta); prior from params.
    use_grid: if True, normalisation is done by trapezoid on an auto-chosen grid; if False, use quad.
    n_grid: number of grid points when use_grid=True (ignored otherwise).

    Bandwidth: params["kde_bw_method"] can be "scott", "silverman", or a float. For Student-t/Cauchy
    the MLE distribution is heavy-tailed, so "scott" often oversmooths; a small float (e.g. 0.001)
    is usually needed for n around 100 to get a peaked enough likelihood.
    """
    mles = np.asarray(mle_samples)
    prior_mean = params["prior_mean"]
    prior_std = params["prior_std"]
    bw_method = params.get("kde_bw_method", "scott")
    if verbose:
        print("Fitting KDE to MLE samples using bw_method =", bw_method)
    kde = stats.gaussian_kde(mles, bw_method=bw_method)

    def log_unnorm(mu):
        mu_a = np.atleast_1d(mu)
        log_prior = stats.norm.logpdf(mu_a, loc=prior_mean, scale=prior_std)
        # KDE expects shape (1, n_points) for 1D
        log_like = kde.logpdf((mu_star - mu_a).reshape(1, -1)).ravel()
        out = log_prior + log_like
        return float(out[0]) if np.isscalar(mu) else out

    if use_grid:
        lo, hi = _grid_bounds_for_normalisation(prior_mean, prior_std, mu_star, mles)
        mu_grid = np.linspace(lo, hi, n_grid)
        unnorm_vals = np.exp(log_unnorm(mu_grid))
        integral = float(np.trapezoid(unnorm_vals, mu_grid))
    else:
        integral, _ = quad(lambda mu: np.exp(log_unnorm(mu)), -np.inf, np.inf)

    def normalized_pdf(mu):
        return np.exp(log_unnorm(mu)) / integral

    return normalized_pdf
