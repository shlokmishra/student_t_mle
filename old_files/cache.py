"""
Cache layer for expensive sampling results.

Stores Gibbs chains, KDE MLE samples, and full-data MCMC chains as .npz files
keyed by (k, m, T, seed). Notebooks check cache first; if hit, skip sampling.
"""

import os
import numpy as np


def cache_path(prefix, k, m, T, seed, cache_dir='cache'):
    """Build .npz filename for a cached result."""
    fname = f"{prefix}_k{k}_m{m}_T{T}_seed{seed}.npz"
    return os.path.join(cache_dir, fname)


def is_cached(path):
    """Check if a cache file exists."""
    return os.path.isfile(path)


# --- Gibbs chain cache ---

def save_gibbs(path, mu_chain, data, mle):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mu_chain=np.asarray(mu_chain), data=np.asarray(data), mle=np.float64(mle))


def load_gibbs(path):
    d = np.load(path)
    return {
        'mu_chain': d['mu_chain'],
        'data': d['data'],
        'mle': float(d['mle']),
    }


# --- KDE MLE samples cache ---

def save_kde_samples(path, mle_samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mle_samples=np.asarray(mle_samples))


def load_kde_samples(path):
    d = np.load(path)
    return d['mle_samples']


# --- Full-data MCMC cache ---

def save_fulldata(path, mu_chain, data, mle):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mu_chain=np.asarray(mu_chain), data=np.asarray(data), mle=np.float64(mle))


def load_fulldata(path):
    d = np.load(path)
    return {
        'mu_chain': d['mu_chain'],
        'data': d['data'],
        'mle': float(d['mle']),
    }
