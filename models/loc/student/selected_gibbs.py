"""Selected-global-MLE Student location MH-within-Gibbs, separate from legacy.

Each joint pair proposal preserves its score sum, proposes BOTH inverse branches
from the score conditional, and corrects the entire move by global selection and
observed curvature. No branch refresh follows a rejection. Regular charts only;
unresolved numerical calculations raise instead of being mathematical rejections.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from numba import njit
from scipy.special import erf, ndtr, ndtri

from models.loc.student.score_gibbs import student_logpdf_no_const
from models.loc.student.mle import global_mle
from models.loc.student.selected_mu import update_mu_mh_from_variates
from models.loc.student.mu_mh import (
    seed_from_key,
    unnormalized_mu_logpdf,
)


class SelectedMLENumericalError(RuntimeError):
    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(f'{reason}: {message}')


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


def _sample(x):
    original = np.asarray(x)
    if np.iscomplexobj(original):
        raise ValueError('data must be real')
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('finite one-dimensional sample with n >= 3 required')
    return x


def score_coordinates(x, t, k, scale=1.):
    y = (np.asarray(x, dtype=np.longdouble)-np.longdouble(t))/np.longdouble(scale)
    z = y/(np.longdouble(k)+y*y)
    if not np.isfinite(z).all():
        raise SelectedMLENumericalError('score_nonfinite', 'unrepresentable score coordinates')
    return np.asarray(z, dtype=float)


def inverse_branches(z, k):
    """All finite inverse branches, in increasing order; no clipping or floors."""
    k = _positive(k, 'k')
    z = float(z)
    bound = 1/(2*math.sqrt(k))
    if not math.isfinite(z) or not -bound < z < bound:
        raise SelectedMLENumericalError('score_chart_boundary', 'score must lie strictly inside its physical support')
    if z == 0.:
        return np.array([0.])
    zz, kk = np.longdouble(z), np.longdouble(k)
    discriminant = 1-4*kk*zz*zz
    if discriminant <= 0:
        raise SelectedMLENumericalError('score_chart_fold', 'inverse branches merge at a fold')
    root = np.sqrt(discriminant)
    inner, outer = float(2*kk*zz/(1+root)), float((1+root)/(2*zz))
    if not math.isfinite(outer) or inner == outer:
        raise SelectedMLENumericalError('inverse_resolution', 'distinct finite inverse branches cannot be represented')
    return np.sort(np.array([inner, outer]))


@njit(cache=True)
def _branch_log_weight(y, loc, k):
    """Exact derivative weight without legacy's additive Jacobian floor."""
    ay = abs(y)
    if ay <= math.sqrt(k):
        r = y*y/k
        if r >= 1.:
            return math.nan
        log_derivative = -math.log(k)+math.log1p(-r)-2*math.log1p(r)
    else:
        r = (math.sqrt(k)/ay)**2
        if r >= 1.:
            return math.nan
        log_derivative = math.log1p(-r)-2*math.log(ay)-2*math.log1p(r)
    residual = y-loc
    if abs(residual) < 1e150 and abs(residual)/math.sqrt(k) < 1e150:
        log_density = student_logpdf_no_const(y, loc, k)
    else:
        log_density = -(k+1)*(math.log(math.hypot(residual, math.sqrt(k)))-.5*math.log(k))
    return log_density-log_derivative


def branch_distribution(z, mu, t, k, scale=1.):
    branches = inverse_branches(z, k)
    loc = (float(mu)-float(t))/float(scale)
    logs = np.array([_branch_log_weight(y, loc, float(k)) for y in branches])
    if not np.isfinite(logs).all():
        raise SelectedMLENumericalError('branch_weight_resolution', 'nonfinite/fold branch weights')
    m = float(logs.max())
    q = m+math.log(float(np.exp(logs-m).sum()))
    return branches, np.exp(logs-q), q


def feasible_interval(delta, k):
    b = 1/(2*math.sqrt(k))
    low, high = max(-b, delta-b), min(b, delta+b)
    if not low < high:
        raise SelectedMLENumericalError('pair_interval', 'pair score interval has no resolvable interior')
    return low, high


def truncnorm_log_mass(center, sigma, low, high):
    # The center is in the interval, so erf subtraction crosses zero and avoids
    # subtracting nearly equal tail CDFs. Sampling still needs representable CDFs.
    if not low < center < high:
        raise SelectedMLENumericalError('proposal_center', 'current score not in pair interval')
    mass = .5*(erf((high-center)/(sigma*math.sqrt(2)))-erf((low-center)/(sigma*math.sqrt(2))))
    if not math.isfinite(mass) or mass <= 0:
        raise SelectedMLENumericalError('proposal_mass', 'truncated-normal mass unresolved')
    return math.log(mass)


def propose_score(z, sigma, low, high, rng):
    log_mass = truncnorm_log_mass(z, sigma, low, high)
    mass = math.exp(log_mass)
    if mass < 32*np.finfo(float).eps:
        raise SelectedMLENumericalError('proposal_resolution', 'inverse CDF cannot resolve this narrow interval')
    p = float(ndtr((low-z)/sigma))+float(rng.random())*mass
    proposed = z+sigma*float(ndtri(p))
    if not math.isfinite(proposed) or not low < proposed < high:
        raise SelectedMLENumericalError('proposal_boundary_rounding', 'inverse CDF rounded onto/outside a chart boundary')
    return proposed, log_mass


@dataclass(frozen=True)
class TargetInfo:
    log_r: float
    curvature: float
    score_residual: float
    selected_mle: float
    selected: bool
    selector_fallback: bool
    root_error_estimate: float
    state_key: tuple


def target_info(x, t, k, *, scale=1., score_tolerance=1e-10, mle_tolerance=1e-10):
    """Numerical global indicator and curvature; selector errors propagate.

    mle_tolerance is an absolute released-location check, NOT a likelihood tie
    tolerance. Near-zero/ambiguous curvature and constraint accuracy raise.
    """
    x = _sample(x)
    k, scale = _positive(k, 'k'), _positive(scale, 'known scale')
    score_tolerance = _positive(score_tolerance, 'score tolerance')
    mle_tolerance = _positive(mle_tolerance, 'MLE tolerance')
    t = float(t)
    if not math.isfinite(t):
        raise ValueError('release must be finite')
    y = (x.astype(np.longdouble)-np.longdouble(t))/np.longdouble(scale)
    denominator = np.longdouble(k)+y*y
    c = np.sum(y/denominator, dtype=np.longdouble)
    terms = (np.longdouble(k)-y*y)/(denominator*denominator)
    h = np.sum(terms, dtype=np.longdouble)
    if not np.isfinite(c) or not np.isfinite(h):
        raise SelectedMLENumericalError('target_nonfinite', 'score or curvature not finite')
    if abs(c) > score_tolerance:
        raise SelectedMLENumericalError('score_constraint', f'score residual {float(abs(c)):.3g}')
    # Keep the selector's original-unit absolute root contract under scale change.
    selected, diag = global_mle(x if scale == 1 else x/scale, k,
                                location_tol=2e-13/scale, return_diagnostics=True)
    selected *= scale
    state_key = (x.tobytes(), t, k, scale)
    if abs(selected-t) > mle_tolerance:
        return TargetInfo(-math.inf, float(h), float(abs(c)), selected, False, diag['fallback'], math.inf, state_key)
    roundoff = 128*np.finfo(np.longdouble).eps*np.sum(np.abs(terms), dtype=np.longdouble)
    if h <= roundoff:
        raise SelectedMLENumericalError('curvature_degenerate', 'selected-state curvature is nonpositive or numerically unresolved')
    error = float(abs(c)/h*scale)
    if error > mle_tolerance:
        raise SelectedMLENumericalError('constraint_location_accuracy', f'score/curvature implies root displacement {error:.3g}')
    return TargetInfo(float(np.log(h)), float(h), float(abs(c)), selected, True, diag['fallback'], error, state_key)


def pair_log_acceptance(x, proposed, pair, mu, t, k, sigma_z, current_info, proposed_info, *, scale=1.):
    """Inspect the exact joint-proposal log ratio and its three components."""
    i, j = pair
    z = score_coordinates(x, t, k, scale)
    zp = score_coordinates(proposed, t, k, scale)
    delta = z[i]+z[j]
    if abs((zp[i]+zp[j])-delta) > 2e-13:
        raise SelectedMLENumericalError('pair_score_sum', 'proposal did not preserve pair score sum')
    if not np.array_equal(np.delete(x, [i, j]), np.delete(proposed, [i, j])):
        raise ValueError('joint pair proposal changed other coordinates')
    low, high = feasible_interval(delta, k)
    qcurrent = branch_distribution(z[i], mu, t, k, scale)[2]+branch_distribution(z[j], mu, t, k, scale)[2]
    qproposed = branch_distribution(zp[i], mu, t, k, scale)[2]+branch_distribution(zp[j], mu, t, k, scale)[2]
    score_ratio = qproposed-qcurrent
    hastings = truncnorm_log_mass(z[i], sigma_z, low, high)-truncnorm_log_mass(zp[i], sigma_z, low, high)
    correction = proposed_info.log_r-current_info.log_r
    return score_ratio+hastings+correction, {'score_log_ratio': score_ratio, 'hastings_log_ratio': hastings,
                                          'correction_log_ratio': correction}


def pair_step(x, pair, mu, t, k, sigma_z, rng, *, scale=1., current_info=None,
              score_tolerance=1e-10, mle_tolerance=1e-10):
    """Joint z/branch MH move; rejection retains the entire current dataset."""
    x = _sample(x)
    sigma_z = _positive(sigma_z, 'score proposal SD')
    if len(pair) != 2 or pair[0] == pair[1] or min(pair) < 0 or max(pair) >= len(x):
        raise ValueError('pair must contain two different valid indices')
    info = current_info or target_info(x, t, k, scale=scale, score_tolerance=score_tolerance, mle_tolerance=mle_tolerance)
    if info.state_key != (x.tobytes(), float(t), float(k), float(scale)):
        raise ValueError('cached correction belongs to a different dataset or target')
    if not info.selected:
        raise ValueError('current state is outside selected-MLE target')
    i, j = pair
    z = score_coordinates(x, t, k, scale)
    delta = z[i]+z[j]
    low, high = feasible_interval(delta, k)
    zi_new, log_mass_current = propose_score(z[i], sigma_z, low, high, rng)
    zj_new = delta-zi_new
    yi, pi, qi = branch_distribution(zi_new, mu, t, k, scale)
    yj, pj, qj = branch_distribution(zj_new, mu, t, k, scale)
    # Proposal probabilities are the score-conditional probabilities, not R-
    # reweighted branch probabilities. The complete joint move is corrected below.
    bi = int(np.searchsorted(np.cumsum(pi), float(rng.random()), side='right'))
    bj = int(np.searchsorted(np.cumsum(pj), float(rng.random()), side='right'))
    if bi >= len(yi) or bj >= len(yj):
        raise SelectedMLENumericalError('branch_probability', 'branch CDF normalization unresolved')
    proposed = x.copy()
    proposed[i], proposed[j] = t+scale*yi[bi], t+scale*yj[bj]
    reconstructed = score_coordinates(proposed, t, k, scale)
    score_error = max(abs(reconstructed[i]-zi_new), abs(reconstructed[j]-zj_new))
    if score_error > 64*np.finfo(float).eps/(2*math.sqrt(k)):
        raise SelectedMLENumericalError('inverse_score_accuracy', 'reconstructed branches lost their proposed score coordinates')
    new_info = target_info(proposed, t, k, scale=scale, score_tolerance=score_tolerance, mle_tolerance=mle_tolerance)
    current_q = branch_distribution(z[i], mu, t, k, scale)[2]+branch_distribution(z[j], mu, t, k, scale)[2]
    score_ratio = qi+qj-current_q
    hastings = log_mass_current-truncnorm_log_mass(zi_new, sigma_z, low, high)
    correction = new_info.log_r-info.log_r
    ratio = score_ratio+hastings+correction
    if math.isnan(ratio) or ratio == math.inf:
        raise SelectedMLENumericalError('acceptance_nonfinite', 'unresolved joint acceptance ratio')
    u = float(rng.random())
    accepted = (math.log(u) if u > 0 else -math.inf) < min(0., ratio)
    # -inf must remain a mathematical rejection, including a representable u=0.
    accepted = bool(accepted and new_info.selected)
    diag = {'accepted': accepted, 'selected_proposal': new_info.selected,
            'selector_fallback': new_info.selector_fallback, 'log_acceptance_ratio': ratio,
            'score_log_ratio': score_ratio, 'hastings_log_ratio': hastings,
            'correction_log_ratio': correction, 'branch_indices': [bi, bj],
            'branch_changed': [bool(abs((x[i]-t)/scale)>math.sqrt(k)) != bool(abs(yi[bi])>math.sqrt(k)),
                               bool(abs((x[j]-t)/scale)>math.sqrt(k)) != bool(abs(yj[bj])>math.sqrt(k))],
            'proposal_score_residual': new_info.score_residual}
    diag['inverse_score_error'] = score_error
    return (proposed, new_info, diag) if accepted else (x.copy(), info, diag)


def run_gibbs(key, mu_star, params, verbose=True):
    """Selected-MLE variant. Requires explicit valid initial_x; local RNG only."""
    x = _sample(params['initial_x']).copy()
    n = int(params['n'])
    if n != params['n'] or len(x) != n:
        raise ValueError('initial_x length must equal n')
    iterations = int(params['num_iterations_T'])
    if iterations != params['num_iterations_T'] or iterations < 0:
        raise ValueError('nonnegative integer number of iterations required')
    k = _positive(params['k'], 'k'); scale = _positive(params.get('known_scale', 1.), 'known scale')
    sigma_z = _positive(params['proposal_std_z'], 'score proposal SD')
    sigma_mu = _positive(params['proposal_std_mu'], 'parameter proposal SD')
    prior_std = _positive(params['prior_std'], 'prior SD'); prior_mean = float(params['prior_mean'])
    mu = float(params.get('initial_mu', mu_star))
    if not math.isfinite(mu) or not math.isfinite(prior_mean):
        raise ValueError('finite initial mu and prior mean required')
    score_tol = _positive(params.get('score_tolerance', 1e-10), 'score tolerance')
    mle_tol = _positive(params.get('mle_tolerance', 1e-10), 'MLE tolerance')
    info = target_info(x, mu_star, k, scale=scale, score_tolerance=score_tol, mle_tolerance=mle_tol)
    if not info.selected:
        raise ValueError('initializer must have the released selected GLOBAL MLE')
    seed = int(key) if isinstance(key, (int, np.integer)) else seed_from_key(key)
    rng = np.random.default_rng(seed)
    store = bool(params.get('store_x_chain', True))
    mus = np.empty(iterations+1); mus[0] = mu
    xs = np.empty((iterations+1, n)) if store else None
    if store:
        xs[0] = x
    counts = dict(pair_attempts=0, pair_accepts=0, global_selection_rejects=0,
                  selector_fallbacks=int(info.selector_fallback), mu_accepts=0, accepted_branch_changes=0)
    max_score, max_mle = info.score_residual, abs(info.selected_mle-mu_star)
    min_curvature = info.curvature
    initial_score_residual = info.score_residual
    initial_c = float(np.sum(score_coordinates(x, mu_star, k, scale), dtype=np.longdouble))
    max_score_drift = max_inverse_score_error = 0.
    start = time.monotonic()
    ceiling = _positive(params.get('wall_ceiling_seconds', 3600.), 'wall ceiling')
    for sweep in range(iterations):
        # Preserve the tested ordinary Student parameter block (in standardized
        # units for known scale), with an unfloored MH uniform.
        noise = float(rng.normal())
        standardized_x = x if scale == 1 else x/scale
        candidate = mu/scale+sigma_mu/scale*noise
        current_log = unnormalized_mu_logpdf(mu/scale, standardized_x, prior_mean/scale, prior_std/scale, k)
        candidate_log = unnormalized_mu_logpdf(candidate, standardized_x, prior_mean/scale, prior_std/scale, k)
        if not math.isfinite(current_log) or not math.isfinite(candidate_log):
            raise SelectedMLENumericalError('parameter_density_nonfinite', 'ordinary parameter MH density unresolved')
        updated, accepted_mu = update_mu_mh_from_variates(
            mu/scale, standardized_x, sigma_mu/scale,
            prior_mean/scale, prior_std/scale, k, noise, float(rng.random()))
        mu = updated*scale
        if not math.isfinite(mu):
            raise SelectedMLENumericalError('parameter_nonfinite', 'parameter update is not finite')
        counts['mu_accepts'] += int(accepted_mu)
        perm = rng.permutation(n)
        for p in range(n//2):
            x, info, diag = pair_step(x, perm[2*p:2*p+2], mu, mu_star, k, sigma_z, rng,
                scale=scale, current_info=info, score_tolerance=score_tol, mle_tolerance=mle_tol)
            counts['pair_attempts'] += 1; counts['pair_accepts'] += int(diag['accepted'])
            counts['global_selection_rejects'] += int(not diag['selected_proposal'])
            counts['selector_fallbacks'] += int(diag['selector_fallback'])
            counts['accepted_branch_changes'] += int(diag['accepted'])*sum(diag['branch_changed'])
            max_score = max(max_score, info.score_residual)
            max_mle = max(max_mle, abs(info.selected_mle-mu_star))
            min_curvature = min(min_curvature, info.curvature)
            max_inverse_score_error = max(max_inverse_score_error, diag['inverse_score_error'])
            max_score_drift = max(max_score_drift, abs(float(np.sum(score_coordinates(x, mu_star, k, scale), dtype=np.longdouble))-initial_c))
        mus[sweep+1] = mu
        if store:
            xs[sweep+1] = x
        if verbose and (sweep+1)%5000 == 0:
            print('selected-MLE Gibbs', seed, sweep+1, round(time.monotonic()-start, 2), flush=True)
        if time.monotonic()-start > ceiling:
            raise TimeoutError(f'selected-MLE chain exceeded wall ceiling at sweep {sweep+1}')
    result = {'mu_chain': mus, 'x_final': x.copy(), 'store_x_chain': store,
              'target': 'selected_global_student_location_mle',
              'estimator': 'hardened_global_likelihood_maximum_exact_ties_lower',
              'kernel': 'sequential_joint_pair_mh_with_global_indicator_and_curvature',
              'gibbs_backend': 'python_pairs_numba_weights_mu', 'seed': seed, 'counts': counts,
              'pair_acceptance_rate': counts['pair_accepts']/counts['pair_attempts'] if counts['pair_attempts'] else None,
              'mu_acceptance_rate': counts['mu_accepts']/iterations if iterations else None,
              'max_score_residual': max_score, 'max_selected_mle_residual': max_mle,
              'initial_score_residual': initial_score_residual, 'max_score_drift': max_score_drift,
              'max_inverse_score_error': max_inverse_score_error,
              'min_retained_curvature': min_curvature, 'seconds': time.monotonic()-start,
              'numerical_policy': 'open score charts; no Jacobian/curvature floor; unresolved computations raise',
              'score_tolerance': score_tol, 'mle_tolerance': mle_tol, 'known_scale': scale}
    if store:
        result['x_chain'] = xs
    return result
