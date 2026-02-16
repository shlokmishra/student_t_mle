"""
Analysis utilities: KL divergence, posterior predictive, DP-related metrics.
"""

import numpy as np
import scipy.stats as stats
import jax.random as random

import validation
import utils


def kl_divergence_estimate(
    gibbs_samples,
    kde_posterior_pdf,
    mu_grid=None,
    n_grid=2000,
    eps=1e-10,
):
    """
    Estimate KL(Gibbs || KDE) using a grid.

    Approximates Gibbs samples as p via KDE, uses KDE posterior as q.
    KL(p||q) = integral p(x) log(p(x)/q(x)) dx.
    """
    samples = np.asarray(gibbs_samples).flatten()
    if mu_grid is None:
        lo = max(samples.min() - 3, -50)
        hi = min(samples.max() + 3, 50)
        mu_grid = np.linspace(lo, hi, n_grid)

    gibbs_kde = stats.gaussian_kde(samples, bw_method='scott')
    p_vals = np.maximum(gibbs_kde(mu_grid), eps)
    q_vals = np.maximum(kde_posterior_pdf(mu_grid), eps)

    p_vals = p_vals / np.trapezoid(p_vals, mu_grid)
    kl = np.trapezoid(p_vals * (np.log(p_vals) - np.log(q_vals)), mu_grid)
    return float(kl)


def run_kl_vs_m_study(
    ms,
    k=1.0,
    mu_true=2.0,
    T_gibbs=50000,
    T_kde=50000,
    base_params=None,
    seed=0,
    burnin=1000,
    verbose=True,
):
    """
    Run Gibbs vs KDE for each m, compute KL divergence.
    Returns dict with m, kl, gibbs_variance, kde_variance, etc.
    """
    if base_params is None:
        base_params = validation._default_base_params(mu_true)

    results = {
        'm': [],
        'kl': [],
        'gibbs_variance': [],
        'kde_variance': [],
        'variance_ratio': [],
    }

    key = random.PRNGKey(seed)
    for m in ms:
        if verbose:
            print(f"\n--- m = {m} ---")
        out = validation.run_single_comparison(
            key,
            m=m,
            k=k,
            mu_true=mu_true,
            T_gibbs=T_gibbs,
            T_kde=T_kde,
            base_params=base_params,
            burnin=burnin,
            verbose=verbose,
        )
        key = out['key']

        kl = kl_divergence_estimate(
            out['mu_chain_post_burnin'],
            out['kde_posterior_pdf'],
        )
        ratio = out['gibbs_variance'] / out['kde_variance']

        results['m'].append(m)
        results['kl'].append(kl)
        results['gibbs_variance'].append(out['gibbs_variance'])
        results['kde_variance'].append(out['kde_variance'])
        results['variance_ratio'].append(ratio)

        if verbose:
            print(f"  KL(Gibbs || KDE): {kl:.4f}")
            print(f"  Variance ratio: {ratio:.4f}")

    return results


def posterior_predictive_samples(
    kde_posterior_pdf,
    k,
    n_mu_samples,
    n_predictive_per_mu=1,
    mu_grid=None,
    n_grid=5000,
    seed=None,
):
    """
    Sample from posterior predictive: μ ~ KDE posterior, then y ~ t(μ, k).

    Uses inverse CDF sampling on a grid for μ.
    Returns (mu_samples, y_samples).
    """
    rng = np.random.default_rng(seed)
    if mu_grid is None:
        mu_grid = np.linspace(-20, 20, n_grid)

    pdf_vals = kde_posterior_pdf(mu_grid)
    pdf_vals = np.maximum(pdf_vals, 1e-20)
    pdf_vals = pdf_vals / np.trapezoid(pdf_vals, mu_grid)
    dx = (mu_grid[-1] - mu_grid[0]) / (len(mu_grid) - 1)
    cdf_vals = np.cumsum(pdf_vals) * dx
    cdf_vals = cdf_vals / cdf_vals[-1]

    u = rng.random(n_mu_samples)
    mu_samples = np.interp(u, cdf_vals, mu_grid)

    y_samples = []
    for mu in mu_samples:
        ys = stats.t.rvs(df=k, loc=mu, scale=1, size=n_predictive_per_mu)
        y_samples.extend(ys)
    y_samples = np.array(y_samples)

    return mu_samples, y_samples


def posterior_variance_from_kde(
    kde_posterior_pdf,
    mu_grid=None,
    n_grid=2000,
):
    """Compute posterior mean and variance from KDE posterior via numerical integration."""
    if mu_grid is None:
        mu_grid = np.linspace(-20, 20, n_grid)
    pdf_vals = np.maximum(kde_posterior_pdf(mu_grid), 1e-20)
    pdf_vals = pdf_vals / np.trapezoid(pdf_vals, mu_grid)
    mean = np.trapezoid(mu_grid * pdf_vals, mu_grid)
    var = np.trapezoid((mu_grid - mean) ** 2 * pdf_vals, mu_grid)
    return float(mean), float(var)
