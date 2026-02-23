"""
Analysis utilities: KL divergence, posterior predictive, DP-related metrics.
Supports optional caching via cache_dir parameter.
"""

import numpy as np
import scipy.stats as stats
import jax.random as random
import jax.numpy as jnp

import validation
import utils
import cache as _cache
import jax_gibbs as gs_jax


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
    cache_dir=None,
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
            cache_dir=cache_dir,
            seed_hint=seed,
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


def run_info_loss_sweep(
    ks,
    ms,
    mu_true=2.0,
    T_kde=50000,
    T_fulldata=50000,
    base_params=None,
    seed=0,
    burnin=1000,
    verbose=True,
    cache_dir=None,
):
    """
    Compute information loss ratio: Var(μ|μ*) / Var(μ|x) for each (k, m).

    Var(μ|μ*) comes from the KDE posterior (ground truth for p(μ|μ*)).
    Var(μ|x) comes from standard Metropolis-Hastings on the full data.

    Returns dict suitable for pd.DataFrame with columns:
        k, m, var_mle, var_fulldata, info_loss_ratio
    """
    if base_params is None:
        base_params = validation._default_base_params(mu_true)

    results = {
        'k': [], 'm': [],
        'var_mle': [], 'var_fulldata': [], 'info_loss_ratio': [],
        'mle': [], 'mu_true': [],
    }

    key = random.PRNGKey(seed)
    for k in ks:
        for m in ms:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Info loss: k={k}, m={m}")
                print(f"{'='*60}")

            params = base_params.copy()
            params['k'] = k
            params['m'] = m
            params['num_iterations_T'] = T_fulldata

            # --- Cache paths ---
            kde_path = None
            fulldata_path = None
            if cache_dir:
                kde_path = _cache.cache_path('kde', k, m, T_kde, seed, cache_dir)
                fulldata_path = _cache.cache_path('fulldata', k, m, T_fulldata, seed, cache_dir)

            # Generate data (or load from fulldata cache which has data+mle)
            if cache_dir and _cache.is_cached(fulldata_path):
                if verbose:
                    print(f"  [cache hit] Loading full-data chain from {fulldata_path}")
                cached_fd = _cache.load_fulldata(fulldata_path)
                data = cached_fd['data']
                mle = cached_fd['mle']
                mu_chain_full = cached_fd['mu_chain']
                mu_chain_full_post = mu_chain_full[burnin:]
                var_fulldata = float(mu_chain_full_post.var())
            else:
                key, subkey = random.split(key)
                data = random.t(subkey, df=k, shape=(m,)) + mu_true
                mle = utils.get_mle(data, params)

                if verbose:
                    print(f"  Running full-data MCMC (Var(μ|x))...")
                key, key_full = random.split(key)
                full_results = gs_jax.run_metropolis_x_jax(
                    key_full, jnp.asarray(data), params.copy()
                )
                mu_chain_full = np.array(full_results['mu_chain'])
                mu_chain_full_post = mu_chain_full[burnin:]
                var_fulldata = float(mu_chain_full_post.var())

                if cache_dir:
                    _cache.save_fulldata(fulldata_path, mu_chain_full, data, mle)
                    if verbose:
                        print(f"  [cache save] Full-data chain saved to {fulldata_path}")

            # --- Var(μ|μ*) from KDE posterior ---
            kde_params = params.copy()
            if k == 1.0:
                kde_params['kde_bw_method'] = 0.001

            if cache_dir and _cache.is_cached(kde_path):
                if verbose:
                    print(f"  [cache hit] Loading KDE samples from {kde_path}")
                mle_samples = _cache.load_kde_samples(kde_path)
                kde_pdf = utils.get_normalized_posterior_mle_pdf(
                    mle, kde_params, mle_samples=mle_samples
                )
            else:
                if verbose:
                    print(f"  Computing KDE posterior (Var(μ|μ*))...")
                mle_samples = utils.get_benchmark_mle_samples(kde_params, num_simulations=T_kde)
                kde_pdf = utils.get_normalized_posterior_mle_pdf(
                    mle, kde_params, mle_samples=mle_samples
                )
                if cache_dir:
                    _cache.save_kde_samples(kde_path, mle_samples)
                    if verbose:
                        print(f"  [cache save] KDE samples saved to {kde_path}")

            _, var_mle = posterior_variance_from_kde(kde_pdf)

            ratio = var_mle / var_fulldata if var_fulldata > 0 else np.nan

            results['k'].append(k)
            results['m'].append(m)
            results['var_mle'].append(var_mle)
            results['var_fulldata'].append(var_fulldata)
            results['info_loss_ratio'].append(ratio)
            results['mle'].append(float(mle))
            results['mu_true'].append(mu_true)

            if verbose:
                print(f"  Var(μ|μ*) = {var_mle:.6f}")
                print(f"  Var(μ|x)  = {var_fulldata:.6f}")
                print(f"  Info loss ratio Var(μ|μ*)/Var(μ|x) = {ratio:.4f}")

    return results


def run_kl_vs_m_multi_k(
    ks,
    ms,
    mu_true=2.0,
    T_gibbs=50000,
    T_kde=50000,
    base_params=None,
    seed=0,
    burnin=1000,
    verbose=True,
    cache_dir=None,
):
    """
    Run KL(Gibbs || KDE) study for multiple k values.
    Returns dict suitable for pd.DataFrame with columns: k, m, kl, variance_ratio.
    """
    if base_params is None:
        base_params = validation._default_base_params(mu_true)

    results = {'k': [], 'm': [], 'kl': [], 'gibbs_variance': [], 'kde_variance': [], 'variance_ratio': []}

    key = random.PRNGKey(seed)
    for k in ks:
        for m in ms:
            if verbose:
                print(f"\n--- k={k}, m={m} ---")
            out = validation.run_single_comparison(
                key, m=m, k=k, mu_true=mu_true,
                T_gibbs=T_gibbs, T_kde=T_kde,
                base_params=base_params, burnin=burnin, verbose=verbose,
                cache_dir=cache_dir, seed_hint=seed,
            )
            key = out['key']
            kl = kl_divergence_estimate(out['mu_chain_post_burnin'], out['kde_posterior_pdf'])
            ratio = out['gibbs_variance'] / out['kde_variance']

            results['k'].append(k)
            results['m'].append(m)
            results['kl'].append(kl)
            results['gibbs_variance'].append(out['gibbs_variance'])
            results['kde_variance'].append(out['kde_variance'])
            results['variance_ratio'].append(ratio)

            if verbose:
                print(f"  KL(Gibbs || KDE): {kl:.4f}, Variance ratio: {ratio:.4f}")

    return results
