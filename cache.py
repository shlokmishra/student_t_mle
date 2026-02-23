"""
Cache layer: save/load Gibbs chains and KDE MLE samples. Keys include model name and prior.
"""

import os
import numpy as np


def _prior_suffix(prior_mean, prior_std):
    """Filesystem-safe suffix for prior (e.g. prior0_10)."""
    m = str(prior_mean).replace(".", "p").replace("-", "m")
    s = str(prior_std).replace(".", "p").replace("-", "m")
    return f"_prior{m}_{s}"


def cache_path(prefix, model, shape_key, T, seed, cache_dir="cache", prior_mean=0.0, prior_std=10.0, include_prior=True):
    """
    shape_key: for loc_student (k, n), for loc_laplace (b, n), for loc_logistic (n,) only.
    prior_mean, prior_std: used in path only if include_prior=True (Gibbs and fulldata depend on prior; KDE samples do not).
    include_prior: if False, path is prior-independent (use for KDE MLE samples only).
    """
    if len(shape_key) == 2:
        a, n = shape_key
        suffix = f"_{a}_{n}"
    else:
        n = shape_key[0]
        suffix = f"_{n}"
    prior = _prior_suffix(prior_mean, prior_std) if include_prior else ""
    fname = f"{prefix}_model_{model}{suffix}{prior}_T{T}_seed{seed}.npz"
    return os.path.join(cache_dir, fname)


def is_cached(path):
    return os.path.isfile(path)


def save_gibbs(
    path,
    mu_chain,
    data,
    mle,
    mu_acceptance_rate=None,
    pair_acceptance_rate=None,
    z_acceptance_rate=None,
    prior_mean=0.0,
    prior_std=10.0,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs = {
        "mu_chain": np.asarray(mu_chain),
        "data": np.asarray(data),
        "mle": np.float64(mle),
        "prior_mean": np.float64(prior_mean),
        "prior_std": np.float64(prior_std),
    }
    if mu_acceptance_rate is not None:
        kwargs["mu_acceptance_rate"] = np.float64(mu_acceptance_rate)
    if pair_acceptance_rate is not None:
        kwargs["pair_acceptance_rate"] = np.float64(pair_acceptance_rate)
    if z_acceptance_rate is not None:
        kwargs["z_acceptance_rate"] = np.float64(z_acceptance_rate)
    np.savez(path, **kwargs)


def load_gibbs(path):
    d = np.load(path)
    out = {
        "mu_chain": d["mu_chain"],
        "data": d["data"],
        "mle": float(d["mle"]),
    }
    if "prior_mean" in d:
        out["prior_mean"] = float(d["prior_mean"])
    if "prior_std" in d:
        out["prior_std"] = float(d["prior_std"])
    if "mu_acceptance_rate" in d:
        out["mu_acceptance_rate"] = float(d["mu_acceptance_rate"])
    if "pair_acceptance_rate" in d:
        out["pair_acceptance_rate"] = float(d["pair_acceptance_rate"])
    if "z_acceptance_rate" in d:
        out["z_acceptance_rate"] = float(d["z_acceptance_rate"])
    return out


def save_kde_samples(path, mle_samples, prior_mean=0.0, prior_std=10.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        mle_samples=np.asarray(mle_samples),
        prior_mean=np.float64(prior_mean),
        prior_std=np.float64(prior_std),
    )


def load_kde_samples(path):
    d = np.load(path)
    out = {"mle_samples": d["mle_samples"]}
    if "prior_mean" in d:
        out["prior_mean"] = float(d["prior_mean"])
    if "prior_std" in d:
        out["prior_std"] = float(d["prior_std"])
    return out


def save_fulldata(path, mu_chain, data, mle, mu_acceptance_rate=None, prior_mean=0.0, prior_std=10.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs = {
        "mu_chain": np.asarray(mu_chain),
        "data": np.asarray(data),
        "mle": np.float64(mle),
        "prior_mean": np.float64(prior_mean),
        "prior_std": np.float64(prior_std),
    }
    if mu_acceptance_rate is not None:
        kwargs["mu_acceptance_rate"] = np.float64(mu_acceptance_rate)
    np.savez(path, **kwargs)


def load_fulldata(path):
    d = np.load(path)
    out = {
        "mu_chain": d["mu_chain"],
        "data": d["data"],
        "mle": float(d["mle"]),
    }
    if "prior_mean" in d:
        out["prior_mean"] = float(d["prior_mean"])
    if "prior_std" in d:
        out["prior_std"] = float(d["prior_std"])
    if "mu_acceptance_rate" in d:
        out["mu_acceptance_rate"] = float(d["mu_acceptance_rate"])
    return out
