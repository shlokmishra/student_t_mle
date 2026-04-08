"""
Model-agnostic analysis: posterior summaries, KL(chain || KDE), and ESS utilities.
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


def kl_divergence_estimate(samples, kde_posterior_pdf, mu_grid=None, n_grid=2000, eps=1e-10):
    """
    Estimate KL(chain KDE || reference KDE) on a grid.
    """
    samples = np.asarray(samples).flatten()
    if mu_grid is None:
        lo = max(samples.min() - 3, -50)
        hi = min(samples.max() + 3, 50)
        mu_grid = np.linspace(lo, hi, n_grid)
    chain_kde = stats.gaussian_kde(samples, bw_method="scott")
    p_vals = np.maximum(chain_kde(mu_grid), eps)
    q_vals = np.maximum(kde_posterior_pdf(mu_grid), eps)
    p_vals = p_vals / np.trapezoid(p_vals, mu_grid)
    kl = np.trapezoid(p_vals * (np.log(p_vals) - np.log(q_vals)), mu_grid)
    return float(kl)


def effective_sample_size_1d(samples, max_lag=None):
    """
    Estimate 1D ESS using Geyer's initial positive sequence on autocorrelations.
    """
    x = np.asarray(samples, dtype=float).reshape(-1)
    n = x.size
    if n < 3:
        return float(n)

    x = x - x.mean()
    var = np.dot(x, x) / n
    if not np.isfinite(var) or var <= 0:
        return float(n)

    if max_lag is None:
        max_lag = min(n - 1, max(10, n // 2))
    else:
        max_lag = min(int(max_lag), n - 1)

    acov = np.empty(max_lag + 1, dtype=float)
    acov[0] = var
    for lag in range(1, max_lag + 1):
        acov[lag] = np.dot(x[:-lag], x[lag:]) / (n - lag)

    rho = acov / max(acov[0], 1e-30)
    tau = 1.0
    for k in range(1, max_lag, 2):
        pair_sum = rho[k] + rho[k + 1]
        if not np.isfinite(pair_sum) or pair_sum <= 0:
            break
        tau += 2.0 * pair_sum

    tau = max(tau, 1.0)
    ess = n / tau
    return float(min(max(ess, 1.0), n))


def ess_per_second(samples, elapsed_seconds, max_lag=None):
    """Compute ESS/sec for a 1D chain."""
    ess = effective_sample_size_1d(samples, max_lag=max_lag)
    if elapsed_seconds is None or elapsed_seconds <= 0:
        return ess, float("nan")
    return ess, float(ess / elapsed_seconds)
