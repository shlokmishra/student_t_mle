"""
Validation utilities: Gibbs vs KDE comparison, variance sweep.
Supports optional caching via cache_dir parameter.
"""

import time
import numpy as np
import jax.random as random

import jax_gibbs as gs_jax
import utils
import cache as _cache


def _default_base_params(mu_true=2.0):
    return {
        'mu_true': mu_true,
        'prior_mean': 0.0,
        'prior_std': 10.0,
        'proposal_std_mu': 0.9,
        'proposal_std_z': 0.03,
    }


def run_single_comparison(
    key,
    m,
    k,
    mu_true,
    T_gibbs,
    T_kde,
    base_params=None,
    burnin=1000,
    verbose=True,
    cache_dir=None,
    seed_hint=0,
):
    """
    Run Gibbs sampler and KDE posterior for one (m, k) pair.
    Returns Gibbs chain, KDE stats, and timing.

    If cache_dir is set, loads cached Gibbs chain and KDE samples when available,
    and saves them after computing.
    seed_hint is used as part of the cache key (for deterministic runs).
    """
    if base_params is None:
        base_params = _default_base_params(mu_true)
    params = base_params.copy()
    params['k'] = k
    params['m'] = m
    params['num_iterations_T'] = T_gibbs

    # --- Cache paths ---
    gibbs_path = None
    kde_path = None
    if cache_dir:
        gibbs_path = _cache.cache_path('gibbs', k, m, T_gibbs, seed_hint, cache_dir)
        kde_path = _cache.cache_path('kde', k, m, T_kde, seed_hint, cache_dir)

    # --- Gibbs chain ---
    if cache_dir and _cache.is_cached(gibbs_path):
        if verbose:
            print(f"  [cache hit] Loading Gibbs chain from {gibbs_path}")
        cached = _cache.load_gibbs(gibbs_path)
        mu_chain_gibbs = cached['mu_chain']
        data = cached['data']
        mle = cached['mle']
        time_gibbs = 0.0
    else:
        key, subkey = random.split(key)
        data = random.t(subkey, df=k, shape=(m,)) + mu_true
        mle = utils.get_mle(data, params)

        if verbose:
            print(f"  Running Gibbs sampler (T={T_gibbs:,})...")
        key, key_gibbs = random.split(key)
        t0 = time.time()
        gibbs_results = gs_jax.run_gibbs_sampler_mle_jax(key_gibbs, mle, params.copy())
        time_gibbs = time.time() - t0
        mu_chain_gibbs = np.array(gibbs_results['mu_chain'])

        if cache_dir:
            _cache.save_gibbs(gibbs_path, mu_chain_gibbs, data, mle)
            if verbose:
                print(f"  [cache save] Gibbs chain saved to {gibbs_path}")

    burnin = min(burnin, len(mu_chain_gibbs) - 10)
    mu_chain_post_burnin = mu_chain_gibbs[burnin:]

    # --- KDE posterior ---
    if k == 1.0:
        params['kde_bw_method'] = 0.001

    if cache_dir and _cache.is_cached(kde_path):
        if verbose:
            print(f"  [cache hit] Loading KDE samples from {kde_path}")
        mle_samples = _cache.load_kde_samples(kde_path)
        t0 = time.time()
        kde_posterior_pdf = utils.get_normalized_posterior_mle_pdf(
            mle, params, mle_samples=mle_samples
        )
        time_kde = time.time() - t0
    else:
        if verbose:
            print(f"  Computing KDE posterior ({T_kde:,} simulations)...")
        t0 = time.time()
        mle_samples = utils.get_benchmark_mle_samples(params, num_simulations=T_kde)
        kde_posterior_pdf = utils.get_normalized_posterior_mle_pdf(
            mle, params, mle_samples=mle_samples
        )
        time_kde = time.time() - t0

        if cache_dir:
            _cache.save_kde_samples(kde_path, mle_samples)
            if verbose:
                print(f"  [cache save] KDE samples saved to {kde_path}")

    # --- Compute KDE stats ---
    mu_grid = np.linspace(
        mu_chain_post_burnin.min() - 2,
        mu_chain_post_burnin.max() + 2,
        2000,
    )
    kde_pdf_values = kde_posterior_pdf(mu_grid)
    kde_pdf_values = kde_pdf_values / np.trapezoid(kde_pdf_values, mu_grid)
    kde_mean = np.trapezoid(mu_grid * kde_pdf_values, mu_grid)
    kde_variance = np.trapezoid((mu_grid - kde_mean) ** 2 * kde_pdf_values, mu_grid)

    out = {
        'key': key,
        'data': data,
        'mle': mle,
        'mu_chain_gibbs': mu_chain_gibbs,
        'mu_chain_post_burnin': mu_chain_post_burnin,
        'kde_posterior_pdf': kde_posterior_pdf,
        'mu_grid': mu_grid,
        'kde_pdf_values': kde_pdf_values,
        'gibbs_mean': float(mu_chain_post_burnin.mean()),
        'gibbs_variance': float(mu_chain_post_burnin.var()),
        'kde_mean': float(kde_mean),
        'kde_variance': float(kde_variance),
        'time_gibbs': time_gibbs,
        'time_kde': time_kde,
    }
    return out


def run_variance_sweep(
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
    Run Gibbs vs KDE for all (k, m) combinations.
    Returns a dict suitable for pandas DataFrame.
    """
    if base_params is None:
        base_params = _default_base_params(mu_true)
    key = random.PRNGKey(seed)

    results = {
        'k': [],
        'm': [],
        'gibbs_variance': [],
        'kde_variance': [],
        'gibbs_mean': [],
        'kde_mean': [],
        'mle': [],
        'time_gibbs': [],
        'time_kde': [],
    }

    for k in ks:
        for m in ms:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Processing k={k}, m={m}")
                print(f"{'='*60}")
            out = run_single_comparison(
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

            results['k'].append(k)
            results['m'].append(m)
            results['gibbs_variance'].append(out['gibbs_variance'])
            results['kde_variance'].append(out['kde_variance'])
            results['gibbs_mean'].append(out['gibbs_mean'])
            results['kde_mean'].append(out['kde_mean'])
            results['mle'].append(out['mle'])
            results['time_gibbs'].append(out['time_gibbs'])
            results['time_kde'].append(out['time_kde'])

            if verbose:
                ratio = out['gibbs_variance'] / out['kde_variance']
                print(f"  Gibbs variance: {out['gibbs_variance']:.6f}")
                print(f"  KDE variance:   {out['kde_variance']:.6f}")
                print(f"  Variance ratio (Gibbs/KDE): {ratio:.4f}")
                print(f"  Time - Gibbs: {out['time_gibbs']:.2f}s, KDE: {out['time_kde']:.2f}s")

    return results


def run_single_gibbs_kde(
    k,
    m,
    mu_true=2.0,
    T_gibbs=50000,
    T_kde=50000,
    base_params=None,
    data_seed=42,
    gibbs_seed=123,
    burnin=1000,
    cache_dir=None,
):
    """
    Run Gibbs + KDE for a single (k, m) for trace/density comparison.
    Returns dict with chain, KDE pdf, data, mle, etc.
    """
    if base_params is None:
        base_params = _default_base_params(mu_true)
    params = base_params.copy()
    params['k'] = k
    params['m'] = m
    params['num_iterations_T'] = T_gibbs

    # --- Cache paths ---
    gibbs_path = None
    kde_path = None
    if cache_dir:
        gibbs_path = _cache.cache_path('gibbs', k, m, T_gibbs, data_seed, cache_dir)
        kde_path = _cache.cache_path('kde', k, m, T_kde, data_seed, cache_dir)

    # --- Gibbs chain ---
    if cache_dir and _cache.is_cached(gibbs_path):
        print(f"  [cache hit] Loading Gibbs chain from {gibbs_path}")
        cached = _cache.load_gibbs(gibbs_path)
        mu_chain_gibbs = cached['mu_chain']
        data = cached['data']
        mle = cached['mle']
    else:
        key = random.PRNGKey(data_seed)
        key, subkey = random.split(key)
        data = random.t(subkey, df=k, shape=(m,)) + mu_true
        mle = utils.get_mle(data, params)

        key_gibbs = random.PRNGKey(gibbs_seed)
        gibbs_results = gs_jax.run_gibbs_sampler_mle_jax(key_gibbs, mle, params.copy())
        mu_chain_gibbs = np.array(gibbs_results['mu_chain'])

        if cache_dir:
            _cache.save_gibbs(gibbs_path, mu_chain_gibbs, data, mle)
            print(f"  [cache save] Gibbs chain saved to {gibbs_path}")

    burnin = min(burnin, len(mu_chain_gibbs) - 10)
    mu_chain_post_burnin = mu_chain_gibbs[burnin:]

    # --- KDE posterior ---
    if k == 1.0:
        params['kde_bw_method'] = 0.001

    if cache_dir and _cache.is_cached(kde_path):
        print(f"  [cache hit] Loading KDE samples from {kde_path}")
        mle_samples = _cache.load_kde_samples(kde_path)
        kde_posterior_pdf = utils.get_normalized_posterior_mle_pdf(
            mle, params, mle_samples=mle_samples
        )
    else:
        mle_samples = utils.get_benchmark_mle_samples(params, num_simulations=T_kde)
        kde_posterior_pdf = utils.get_normalized_posterior_mle_pdf(
            mle, params, mle_samples=mle_samples
        )
        if cache_dir:
            _cache.save_kde_samples(kde_path, mle_samples)
            print(f"  [cache save] KDE samples saved to {kde_path}")

    return {
        'data': data,
        'mle': mle,
        'mu_chain_gibbs': mu_chain_gibbs,
        'mu_chain_post_burnin': mu_chain_post_burnin,
        'kde_posterior_pdf': kde_posterior_pdf,
        'k': k,
        'm': m,
        'mu_true': mu_true,
        'burnin': burnin,
    }
