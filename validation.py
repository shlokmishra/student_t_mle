"""
Run Gibbs vs KDE for a given model; compare variances via analysis.posterior_variance_from_kde.
"""

import time
import numpy as np
import jax.random as random

from models import loc_student, loc_laplace, loc_logistic
from kde_ref.posterior import get_normalized_posterior_pdf, validate_posterior_1d
from analysis import posterior_variance_from_kde
import cache as _cache

_MODELS = {
    "loc_student": loc_student,
    "loc_laplace": loc_laplace,
    "loc_logistic": loc_logistic,
}


def _default_base_params(mu_true=2.0):
    """Prior: N(prior_mean, prior_std^2). KDE bandwidth: kde_bw_method ('scott', float, or callable)."""
    return {
        "mu_true": mu_true,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "proposal_std_mu": 0.9,
        "proposal_std_z": 0.03,
        "kde_bw_method": "scott",
    }


def _build_params(model, n, mu_true, T_gibbs, base_params, **model_kw):
    base = _default_base_params(mu_true).copy()
    if base_params:
        base.update(base_params)
    base["n"] = n
    base["num_iterations_T"] = T_gibbs
    if model == "student":
        base["k"] = model_kw.get("k", 2.0)
    elif model == "laplace":
        base["b"] = model_kw.get("b", 1.0)
    base.setdefault("kde_bw_method", "scott")
    return base


def _shape_key(model, params):
    if model == "student":
        return (params["k"], params["n"])
    if model == "laplace":
        return (params["b"], params["n"])
    return (params["n"],)


def run_single_comparison(
    model,
    key,
    n,
    mu_true=2.0,
    T_gibbs=5000,
    T_kde=5000,
    T_fulldata=None,
    base_params=None,
    burnin=1000,
    verbose=True,
    cache_dir=None,
    seed_hint=0,
    **model_kw,
):
    """
    Run Gibbs, KDE reference, and (optionally) full-data MH for one model and (n, ...).

    Compares:
      - p(theta | MLE): Gibbs chain and KDE trick
      - p(theta | x): full-data MH chain (if T_fulldata is set)

    model: "loc_student" | "loc_laplace" | "loc_logistic"
    n: sample size (number of observations).
    model_kw: for student pass k=...; for laplace pass b=...; for logistic nothing extra.
    T_fulldata: if set, run (or load) MH on p(theta|x) and add full_data_variance, info_loss_ratio.
    Returns dict with mu_chain_gibbs, mu_chain_post_burnin, kde_posterior_pdf, gibbs_variance,
    kde_variance, and if T_fulldata: mu_chain_fulldata, full_data_variance, info_loss_ratio.
    """
    if model not in _MODELS:
        raise ValueError(f"Unknown model: {model}. Use one of {list(_MODELS)}")
    mod = _MODELS[model]
    params = _build_params(model, n, mu_true, T_gibbs, base_params, **model_kw)
    shape_key = _shape_key(model, params)

    prior_mean = params["prior_mean"]
    prior_std = params["prior_std"]
    gibbs_path = (
        _cache.cache_path("gibbs", model, shape_key, T_gibbs, seed_hint, cache_dir, prior_mean=prior_mean, prior_std=prior_std)
        if cache_dir else None
    )
    # KDE MLE samples do not depend on the prior; use same cache for all priors and apply current prior in get_normalized_posterior_pdf.
    kde_path = (
        _cache.cache_path("kde", model, shape_key, T_kde, seed_hint, cache_dir, include_prior=False)
        if cache_dir else None
    )
    fulldata_path = (
        _cache.cache_path("fulldata", model, shape_key, T_fulldata, seed_hint, cache_dir, prior_mean=prior_mean, prior_std=prior_std)
        if (cache_dir and T_fulldata) else None
    )

    # --- Gibbs chain ---
    gibbs_accept = {}
    if cache_dir and gibbs_path and _cache.is_cached(gibbs_path):
        if verbose:
            print(f"  [cache hit] Loading Gibbs from {gibbs_path}")
        cached = _cache.load_gibbs(gibbs_path)
        mu_chain_gibbs = cached["mu_chain"]
        data = cached["data"]
        mle = cached["mle"]
        time_gibbs = 0.0
        if "mu_acceptance_rate" in cached:
            gibbs_accept["mu"] = cached["mu_acceptance_rate"]
        if "pair_acceptance_rate" in cached:
            gibbs_accept["pair"] = cached["pair_acceptance_rate"]
        if "z_acceptance_rate" in cached:
            gibbs_accept["z"] = cached["z_acceptance_rate"]
    else:
        key, subkey = random.split(key)
        data = np.asarray(mod.sample_data(subkey, params, loc=mu_true))
        mle = mod.get_mle(data, params)
        key, key_gibbs = random.split(key)
        if verbose:
            print(f"  Running Gibbs ({model}, T={T_gibbs})...")
        t0 = time.time()
        gibbs_results = mod.run_gibbs(key_gibbs, float(mle), params.copy(), verbose=verbose)
        time_gibbs = time.time() - t0
        mu_chain_gibbs = np.asarray(gibbs_results["mu_chain"])
        gibbs_accept["mu"] = float(gibbs_results["mu_acceptance_rate"])
        if "pair_acceptance_rate" in gibbs_results:
            gibbs_accept["pair"] = float(gibbs_results["pair_acceptance_rate"])
        if "z_acceptance_rate" in gibbs_results:
            gibbs_accept["z"] = float(gibbs_results["z_acceptance_rate"])
        if cache_dir and gibbs_path:
            _cache.save_gibbs(
                gibbs_path,
                mu_chain_gibbs,
                data,
                mle,
                mu_acceptance_rate=gibbs_accept["mu"],
                pair_acceptance_rate=gibbs_accept.get("pair"),
                z_acceptance_rate=gibbs_accept.get("z"),
                prior_mean=prior_mean,
                prior_std=prior_std,
            )
            if verbose:
                print(f"  [cache save] Gibbs saved to {gibbs_path}")

    burnin = min(burnin, len(mu_chain_gibbs) - 10)
    mu_chain_post_burnin = mu_chain_gibbs[burnin:]

    # --- KDE posterior ---
    kde_params = params.copy()
    # kde_bw_method from params (default "scott"); use a small float (e.g. 0.001) for very heavy tails if needed.

    if cache_dir and kde_path and _cache.is_cached(kde_path):
        if verbose:
            print(f"  [cache hit] Loading KDE samples from {kde_path}")
        kde_loaded = _cache.load_kde_samples(kde_path)
        mle_samples = kde_loaded["mle_samples"]
        kde_posterior_pdf = get_normalized_posterior_pdf(mle, kde_params, mle_samples, verbose=False)
        time_kde = 0.0
    else:
        if verbose:
            print(f"  Computing KDE posterior ({model}, {T_kde} sims)...")
        t0 = time.time()
        mle_samples = mod.get_benchmark_mle_samples(kde_params, num_simulations=T_kde, verbose=False)
        kde_posterior_pdf = get_normalized_posterior_pdf(mle, kde_params, mle_samples, verbose=verbose)
        time_kde = time.time() - t0
        if cache_dir and kde_path:
            _cache.save_kde_samples(kde_path, mle_samples, prior_mean=prior_mean, prior_std=prior_std)
            if verbose:
                print(f"  [cache save] KDE samples saved to {kde_path}")

    kde_mean, kde_var = posterior_variance_from_kde(kde_posterior_pdf)
    gibbs_var = float(mu_chain_post_burnin.var())
    gibbs_mean = float(mu_chain_post_burnin.mean())
    # Validate 1D posterior: post = like * prior / norm_const => integral should be 1
    kde_posterior_integral = validate_posterior_1d(kde_posterior_pdf)

    out = {
        "key": key,
        "model": model,
        "data": data,
        "mle": mle,
        "mu_chain_gibbs": mu_chain_gibbs,
        "mu_chain_post_burnin": mu_chain_post_burnin,
        "kde_posterior_pdf": kde_posterior_pdf,
        "gibbs_mean": gibbs_mean,
        "gibbs_variance": gibbs_var,
        "kde_mean": kde_mean,
        "kde_variance": kde_var,
        "kde_posterior_integral": kde_posterior_integral,
        "time_gibbs": time_gibbs,
        "time_kde": time_kde,
        "gibbs_acceptance": gibbs_accept,
    }

    # --- Full-data MH: p(theta | x) ---
    if T_fulldata is not None:
        params_fd = _build_params(model, n, mu_true, T_fulldata, base_params, **model_kw)
        fulldata_mu_accept = None
        if cache_dir and fulldata_path and _cache.is_cached(fulldata_path):
            if verbose:
                print(f"  [cache hit] Loading full-data chain from {fulldata_path}")
            cached_fd = _cache.load_fulldata(fulldata_path)
            mu_chain_fulldata = cached_fd["mu_chain"]
            time_fulldata = 0.0
            fulldata_mu_accept = cached_fd.get("mu_acceptance_rate")
        else:
            key, key_fd = random.split(key)
            if verbose:
                print(f"  Running full-data MH ({model}, T={T_fulldata})...")
            t0 = time.time()
            fd_results = mod.run_full_data_mh(key_fd, data, params_fd.copy(), verbose=verbose)
            time_fulldata = time.time() - t0
            mu_chain_fulldata = np.asarray(fd_results["mu_chain"])
            fulldata_mu_accept = float(fd_results["mu_acceptance_rate"])
            if cache_dir and fulldata_path:
                _cache.save_fulldata(
                    fulldata_path,
                    mu_chain_fulldata,
                    data,
                    mle,
                    mu_acceptance_rate=fulldata_mu_accept,
                    prior_mean=prior_mean,
                    prior_std=prior_std,
                )
                if verbose:
                    print(f"  [cache save] Full-data chain saved to {fulldata_path}")
            key = key_fd
        burnin_fd = min(burnin, len(mu_chain_fulldata) - 10)
        full_data_var = float(mu_chain_fulldata[burnin_fd:].var())
        full_data_mean = float(mu_chain_fulldata[burnin_fd:].mean())
        info_loss_ratio = (kde_var / full_data_var) if full_data_var > 0 else float("nan")
        out["key"] = key
        out["mu_chain_fulldata"] = mu_chain_fulldata
        out["full_data_mean"] = full_data_mean
        out["full_data_variance"] = full_data_var
        out["info_loss_ratio"] = info_loss_ratio
        out["time_fulldata"] = time_fulldata
        if fulldata_mu_accept is not None:
            out["fulldata_mu_acceptance_rate"] = fulldata_mu_accept

    return out


def run_variance_sweep(
    model,
    key_or_seed,
    ns,
    mu_true=2.0,
    T_gibbs=5000,
    T_kde=5000,
    base_params=None,
    burnin=1000,
    verbose=True,
    cache_dir=None,
    **model_kw,
):
    """Run run_single_comparison for each n; return list of results (e.g. for DataFrame)."""
    key = key_or_seed if hasattr(key_or_seed, "shape") else random.PRNGKey(int(key_or_seed))
    seed = int(key_or_seed) if not hasattr(key_or_seed, "shape") else 0
    results = []
    for n in ns:
        out = run_single_comparison(
            model, key, n,
            mu_true=mu_true, T_gibbs=T_gibbs, T_kde=T_kde,
            base_params=base_params, burnin=burnin, verbose=verbose,
            cache_dir=cache_dir, seed_hint=seed, **model_kw,
        )
        key = out["key"]
        results.append(out)
    return results
