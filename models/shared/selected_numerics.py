"""Numerical rules for selected-statistic kernels; legacy cutoffs are not reused."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def log_uniform(u):
    if not math.isfinite(u) or u < 0.0 or u >= 1.0:
        raise ValueError('MH uniform must belong to [0,1)')
    return math.log(u) if u > 0.0 else -math.inf


@njit(cache=True)
def mh_accept(log_ratio, u):
    if not math.isfinite(log_ratio):
        raise FloatingPointError('MH ratio unresolved')
    return log_uniform(u) < min(0., log_ratio)


def mle_check_mode(params):
    if 'mle_check' in params and 'n3_endpoint_fast_path' in params:
        raise ValueError('specify mle_check or the legacy n3 option, not both')
    if 'n3_endpoint_fast_path' in params:
        flag = params['n3_endpoint_fast_path']
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError('n3_endpoint_fast_path must be boolean')
        return 'n3' if flag else 'selector'
    mode = params.get('mle_check', 'auto')
    if mode not in ('auto', 'selector'):
        raise ValueError('mle_check must be auto or selector')
    return mode


def validated_run(key, release, params, minimum_n=2):
    """Common boundary validation for new family implementations."""
    x = np.asarray(params['initial_x'])
    if np.iscomplexobj(x):
        raise ValueError('real data required')
    x = np.array(x, dtype=float, copy=True)
    n, count = int(params['n']), int(params['num_iterations_T'])
    if x.ndim != 1 or len(x) != n or n < minimum_n or n != params['n'] or not np.isfinite(x).all():
        raise ValueError('finite matching one-dimensional initial data required')
    if count < 1 or count != params['num_iterations_T']:
        raise ValueError('positive integer iteration count required')
    mu = float(params.get('initial_mu', release)); pm = float(params['prior_mean'])
    ps = float(params['prior_std']); sm = float(params['proposal_std_mu'])
    if not np.isfinite([release, mu, pm, ps, sm]).all() or ps <= 0 or sm < 0:
        raise ValueError('finite locations, positive prior SD, nonnegative proposal SD required')
    if isinstance(key, (int, np.integer)):
        seed = int(key)
    else:
        from models.loc.student.mu_mh import seed_from_key
        seed = seed_from_key(key)
    return x, n, count, mu, pm, ps, sm, seed


@njit(cache=True)
def truncnorm_log_mass(center, sigma, lo, hi):
    if not lo < center < hi or sigma <= 0 or not math.isfinite(sigma):
        raise FloatingPointError('truncated-normal center/scale unresolved')
    value = .5*(math.erf((hi-center)/(math.sqrt(2.)*sigma)) + math.erf((center-lo)/(math.sqrt(2.)*sigma)))
    if value <= 0 or not math.isfinite(value):
        raise FloatingPointError('truncated-normal mass unresolved')
    return math.log(value)
