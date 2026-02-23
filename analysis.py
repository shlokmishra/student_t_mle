"""
Model-agnostic analysis: posterior mean/variance from KDE pdf, KL(Gibbs || KDE).
"""

import numpy as np
import scipy.stats as stats


def posterior_variance_from_kde(kde_posterior_pdf, mu_grid=None, n_grid=2000):
    """Compute posterior mean and variance from KDE posterior via numerical integration."""
    if mu_grid is None:
        mu_grid = np.linspace(-20, 20, n_grid)
    pdf_vals = np.maximum(kde_posterior_pdf(mu_grid), 1e-20)
    pdf_vals = pdf_vals / np.trapezoid(pdf_vals, mu_grid)
    mean = np.trapezoid(mu_grid * pdf_vals, mu_grid)
    var = np.trapezoid((mu_grid - mean) ** 2 * pdf_vals, mu_grid)
    return float(mean), float(var)


def kl_divergence_estimate(gibbs_samples, kde_posterior_pdf, mu_grid=None, n_grid=2000, eps=1e-10):
    """
    Estimate KL(Gibbs || KDE) on a grid. Approximates Gibbs as p via KDE, KDE posterior as q.
    """
    samples = np.asarray(gibbs_samples).flatten()
    if mu_grid is None:
        lo = max(samples.min() - 3, -50)
        hi = min(samples.max() + 3, 50)
        mu_grid = np.linspace(lo, hi, n_grid)
    gibbs_kde = stats.gaussian_kde(samples, bw_method="scott")
    p_vals = np.maximum(gibbs_kde(mu_grid), eps)
    q_vals = np.maximum(kde_posterior_pdf(mu_grid), eps)
    p_vals = p_vals / np.trapezoid(p_vals, mu_grid)
    kl = np.trapezoid(p_vals * (np.log(p_vals) - np.log(q_vals)), mu_grid)
    return float(kl)
