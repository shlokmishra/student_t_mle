"""Selected-global-MLE Student RATTLE using the unchanged score guiding force.

Identity mass in standardized residual coordinates. The complete endpoint MH
ratio includes log J, with full phase reversal checks and lazy global selection.
Legacy score RATTLE and its compiled projection/force kernels are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from numba import njit

from models.loc.student import score_rattle as score_rattle
from models.loc.student.mle import global_mle
from models.loc.student.endpoint import endpoint_certificate, certified_state
from models.shared.selected_numerics import mle_check_mode
from models.loc.student.selected_mu import update_mu_mh_from_variates
from models.loc.student.mu_mh import (
    seed_from_key,
    unnormalized_mu_logpdf,
)


class SelectedRATTLENumericalError(RuntimeError):
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
        raise ValueError('finite real one-dimensional sample with n >= 3 required')
    return x


@njit(cache=True)
def _geometry_kernel(y, k):
    c = j = g = absolute_derivatives = 0.
    for value in y:
        denom = k+value*value
        derivative = (k-value*value)/(denom*denom)
        c += value/denom
        j += derivative
        g += derivative*derivative
        absolute_derivatives += abs(derivative)
    return c, j, g, absolute_derivatives


@dataclass(frozen=True)
class Geometry:
    score: float
    curvature: float
    gram: float
    log_j: float
    state_key: tuple


def geometry(x, t, k, *, scale=1., score_tolerance=1e-10):
    x = _sample(x); k = _positive(k, 'k'); scale = _positive(scale, 'known scale')
    t = float(t)
    if not math.isfinite(t):
        raise ValueError('release must be finite')
    y = (x-t)/scale
    c, j, g, absolute_derivatives = _geometry_kernel(y, k)
    if not all(math.isfinite(v) for v in [c, j, g, absolute_derivatives]):
        raise SelectedRATTLENumericalError('geometry_nonfinite', 'score/curvature/Gram unresolved')
    if abs(c) > score_tolerance:
        raise SelectedRATTLENumericalError('score_constraint', f'score residual {abs(c):.3g}')
    if g <= score_rattle.GRAM_FLOOR:
        raise SelectedRATTLENumericalError('gram_degenerate', 'no regular tangent fiber; Gram floor is not a target')
    error = 128*np.finfo(float).eps*absolute_derivatives
    if abs(j) <= error:
        raise SelectedRATTLENumericalError('curvature_unresolved', 'zero/ambiguous endpoint curvature; singular strata unsupported')
    return Geometry(c, j, g, math.log(j) if j > 0 else -math.inf,
                    (x.tobytes(), t, k, scale))


@dataclass(frozen=True)
class SelectedState:
    geometry: Geometry
    selected_mle: float
    selector_fallback: bool


def selected_state(x, t, k, *, scale=1., mle_tolerance=1e-9, score_tolerance=1e-10,
                   endpoint_geometry=None, selection_diagnostics=None):
    """Return a valid state or None for resolved support failure; errors propagate."""
    g = endpoint_geometry or geometry(x, t, k, scale=scale, score_tolerance=score_tolerance)
    if g.state_key != (np.asarray(x, dtype=float).tobytes(), float(t), float(k), float(scale)):
        raise ValueError('cached geometry belongs to a different dataset or target')
    if g.curvature <= 0:
        return None
    if abs(g.score)/g.curvature*scale > mle_tolerance:
        raise SelectedRATTLENumericalError('constraint_location_accuracy', 'local score/curvature displacement estimate exceeds release tolerance')
    selected, diag = global_mle(np.asarray(x) if scale == 1 else np.asarray(x)/scale,
                               k, location_tol=2e-13/scale, return_diagnostics=True)
    selected *= scale
    if selection_diagnostics is not None:
        selection_diagnostics.update(selected_mle=selected, **diag)
    if abs(selected-t) > mle_tolerance:
        return None
    return SelectedState(g, selected, bool(diag['fallback']))


def reconstruct_endpoint(y, t, scale):
    """Do not retain a rounded physical dataset different from the checked phase point."""
    x = t+scale*y
    recovered = (x-t)/scale
    error = np.abs(recovered-y)
    bound = 64*np.finfo(float).eps*np.maximum(1., np.abs(y))
    if not np.isfinite(x).all() or not np.isfinite(recovered).all() or np.any(error > bound):
        raise SelectedRATTLENumericalError('endpoint_reconstruction', 'release/scale round trip changes the checked phase endpoint')
    return x, float(np.max(error))


@njit(cache=True)
def _audited_trajectory_kernel(y0, p0, mu, k, step_size, num_steps, proj_tol,
                               max_iters, grad_tol, damping, line_search,
                               tangent_tol, transverse_tol):
    """Reuse each legacy RATTLE step; audit its implicit-root regularity."""
    y, p = y0.copy(), p0.copy()
    iterations = position_failures = momentum_failures = completed = 0
    max_score = abs(score_rattle._constraint_value_kernel(y, 0., k))
    max_tangent = 0.; min_transverse = 1.; min_gram = math.inf
    for _ in range(num_steps):
        old_gradient = score_rattle._constraint_grad_kernel(y, 0., k)
        yn, pn, ok, ni, nf, nm, residual, done = score_rattle._rattle_trajectory_kernel(
            y, p, mu, 0., k, step_size, 1, proj_tol, max_iters,
            grad_tol, damping, line_search, tangent_tol)
        iterations += ni; position_failures += nf; momentum_failures += nm
        if not np.isfinite(yn).all() or not np.isfinite(pn).all() or not math.isfinite(residual):
            return yn, pn, 3, iterations, position_failures, momentum_failures, max_score, completed, max_tangent, min_transverse, min_gram
        if not ok:
            return yn, pn, 1, iterations, position_failures, momentum_failures, max_score, completed, max_tangent, min_transverse, min_gram
        new_gradient = score_rattle._constraint_grad_kernel(yn, 0., k)
        old_gram = np.dot(old_gradient, old_gradient)
        new_gram = np.dot(new_gradient, new_gradient)
        transverse = abs(np.dot(old_gradient, new_gradient))/math.sqrt(old_gram*new_gram)
        min_transverse = min(min_transverse, transverse)
        if not math.isfinite(transverse) or transverse <= transverse_tol:
            return yn, pn, 2, iterations, position_failures, momentum_failures, max_score, completed, max_tangent, min_transverse, min_gram
        min_gram = min(min_gram, old_gram, new_gram)
        max_tangent = max(max_tangent, abs(np.dot(new_gradient, pn)))
        max_score = max(max_score, residual)
        completed += done; y, p = yn, pn
    return y, p, 0, iterations, position_failures, momentum_failures, max_score, completed, max_tangent, min_transverse, min_gram


def options(params):
    if not params.get('reverse_check', True) or not params.get('rattle_include_gram_correction', True):
        raise ValueError('selected RATTLE requires reverse checking and the score Gram correction')
    if params.get('rattle_projection_mode', 'paper_fixed_direction') != 'paper_fixed_direction':
        raise ValueError('only shared paper_fixed_direction projection is supported')
    if 'mass_matrix' in params:
        raise ValueError('identity mass in standardized coordinates is the supported momentum convention')
    result = dict(step_size=_positive(params.get('rattle_step_size', .1), 'step size'),
        num_steps=int(params.get('rattle_num_steps', 10)),
        proj_tol=_positive(params.get('rattle_proj_tol', 1e-12), 'projection tolerance'),
        max_iters=int(params.get('rattle_proj_max_iters', 50)),
        grad_tol=_positive(params.get('rattle_grad_tol', 1e-12), 'gradient tolerance'),
        damping=_positive(params.get('rattle_proj_damping', 1.), 'projection damping'),
        line_search=bool(params.get('rattle_proj_line_search', True)),
        tangent_tol=_positive(params.get('rattle_tangent_tol', 1e-10), 'tangent tolerance'),
        transverse_tol=_positive(params.get('rattle_transverse_tol', 1e-10), 'projection transversality tolerance'),
        reverse_position_tol=_positive(params.get('rattle_reverse_position_tol', 1e-8), 'reverse position tolerance'),
        reverse_momentum_tol=_positive(params.get('rattle_reverse_momentum_tol', 1e-8), 'reverse momentum tolerance'))
    if result['num_steps'] <= 0 or result['num_steps'] != params.get('rattle_num_steps', 10):
        raise ValueError('positive integer trajectory length required')
    if result['max_iters'] <= 0 or result['max_iters'] != params.get('rattle_proj_max_iters', 50):
        raise ValueError('positive integer projection iteration limit required')
    return result


def trajectory(y, p, mu, k, opts):
    result = _audited_trajectory_kernel(np.asarray(y, dtype=float), np.asarray(p, dtype=float), float(mu), float(k),
        *(opts[key] for key in ['step_size','num_steps','proj_tol','max_iters','grad_tol','damping','line_search','tangent_tol','transverse_tol']))
    yn, pn, status, ni, nf, nm, residual, done, tangent, transverse, gram = result
    if status == 3:
        raise SelectedRATTLENumericalError('trajectory_nonfinite', 'shared trajectory arithmetic unresolved')
    return yn, pn, status == 0, dict(status=int(status), projection_iterations=int(ni),
        position_failures=int(nf), momentum_failures=int(nm), max_score_residual=residual,
        completed_steps=int(done), max_tangent_residual=tangent, min_transverse=transverse, min_gram=gram)


def phase_proposal(y, p, mu, k, opts):
    """S Phi^L, with the whole (position,momentum) involution domain checked."""
    yp, pp, ok, fd = trajectory(y, p, mu, k, opts)
    diag = dict(forward=fd, reverse=None, reverse_position_error=None, reverse_momentum_error=None,
                rejection='forward_projection' if not ok else None)
    if not ok:
        return yp, -pp, False, diag
    yr, pr, okr, rd = trajectory(yp, -pp, mu, k, opts)
    xe, pe = float(np.linalg.norm(yr-y)), float(np.linalg.norm(pr+p))
    diag.update(reverse=rd, reverse_position_error=xe, reverse_momentum_error=pe)
    valid = okr and math.isfinite(xe) and math.isfinite(pe) and xe <= opts['reverse_position_tol'] and pe <= opts['reverse_momentum_tol']
    if not valid: diag['rejection'] = 'reverse_phase'
    return yp, -pp, bool(valid), diag


def endpoint_log_acceptance(h0, h1, current_geometry, proposed_geometry):
    if not math.isfinite(h0) or not math.isfinite(h1):
        raise SelectedRATTLENumericalError('hamiltonian_nonfinite', 'unresolved score Hamiltonian')
    ratio = -(h1-h0)+proposed_geometry.log_j-current_geometry.log_j
    if math.isnan(ratio) or ratio == math.inf:
        raise SelectedRATTLENumericalError('acceptance_nonfinite', 'unresolved endpoint ratio')
    return ratio


def rattle_step(x, mu, t, k, rng, opts, *, current_state=None, scale=1.,
                mle_tolerance=1e-9, score_tolerance=1e-10, mle_check='selector'):
    x = _sample(x)
    state = current_state or selected_state(x, t, k, scale=scale, mle_tolerance=mle_tolerance, score_tolerance=score_tolerance)
    if state is None: raise ValueError('current state must have the released selected GLOBAL MLE')
    if state.geometry.state_key != (x.tobytes(), float(t), float(k), float(scale)):
        raise ValueError('cached selected state belongs to a different dataset or target')
    if state.geometry.gram <= opts['grad_tol']:
        raise SelectedRATTLENumericalError('tangent_fiber_unresolved', 'current Gram is below shared projection resolution')
    y, mus = (x-t)/scale, (mu-t)/scale
    p, ok = score_rattle._project_momentum_kernel(rng.standard_normal(len(x)),
        score_rattle._constraint_grad_kernel(y, 0., k), opts['grad_tol'])
    diag = dict(accepted=False, selector_called=False, selected_endpoint=None,
                selector_fallback=False, delta_h_score=None, log_acceptance_ratio=None,
                curvature_log_ratio=None, phase=None, rejection=None, proposed_x=None, phase_seconds=0., endpoint_method=None)
    if not ok:
        diag['rejection'] = 'initial_momentum_projection'; return x.copy(), state, diag
    h0 = score_rattle._hamiltonian(y, p, mus, k, 0.)
    phase_start = time.perf_counter()
    yp, pp, valid, phase = phase_proposal(y, p, mus, k, opts)
    diag['phase_seconds'] = time.perf_counter()-phase_start
    diag['phase'] = phase
    if not valid:
        diag['rejection'] = phase['rejection']; return x.copy(), state, diag
    proposed, roundtrip_error = reconstruct_endpoint(yp, t, scale)
    diag['endpoint_roundtrip_error'] = roundtrip_error
    diag['proposed_x'] = proposed
    pg = geometry(proposed, t, k, scale=scale, score_tolerance=score_tolerance)
    h1 = score_rattle._hamiltonian(yp, pp, mus, k, 0.)
    ratio = endpoint_log_acceptance(h0, h1, state.geometry, pg)
    diag.update(delta_h_score=h1-h0, log_acceptance_ratio=ratio,
                curvature_log_ratio=pg.log_j-state.geometry.log_j)
    if pg.curvature < 0:
        diag.update(rejection='nonpositive_curvature', selected_endpoint=False)
        return x.copy(), state, diag
    u = float(rng.random())
    logu = math.log(u) if u > 0 else -math.inf
    if logu >= min(0., ratio):
        diag['rejection'] = 'cheap_metropolis'; return x.copy(), state, diag
    # The indicator can only reduce acceptance. This is the SAME MH uniform;
    # no second acceptance decision and no selector at intermediate positions.
    cert = endpoint_certificate(proposed,t,k,mle_tolerance,pg,mle_check,scale)
    if cert is not None and cert.status == 1:
        diag.update(accepted=True, selected_endpoint=True, endpoint_method=cert.reason)
        return proposed, certified_state(pg,t,cert), diag
    if cert is not None and cert.status == -1:
        diag.update(rejection='global_selection', selected_endpoint=False, endpoint_method=cert.reason)
        return x.copy(), state, diag
    diag['endpoint_method'] = 'selector'
    diag['selector_called'] = True
    selector_diag = {}
    candidate = selected_state(proposed, t, k, scale=scale, mle_tolerance=mle_tolerance,
        score_tolerance=score_tolerance, endpoint_geometry=pg, selection_diagnostics=selector_diag)
    diag['selector_fallback'] = bool(selector_diag.get('fallback', False))
    diag['selected_endpoint'] = candidate is not None
    if candidate is None:
        diag['rejection'] = 'global_selection'; return x.copy(), state, diag
    diag.update(accepted=True)
    return proposed, candidate, diag


def _json_diag(value):
    if isinstance(value, dict): return {k:_json_diag(v) for k,v in value.items()}
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def run_rattle(key, mu_star, params, verbose=True):
    x = _sample(params['initial_x']).copy(); n = int(params['n'])
    count = int(params['num_iterations_T'])
    if n != params['n'] or n != len(x) or count < 1 or count != params['num_iterations_T']:
        raise ValueError('matching integer n and positive integer iteration count required')
    k = _positive(params['k'], 'k'); scale = _positive(params.get('known_scale', 1.), 'known scale')
    prior_std = _positive(params['prior_std'], 'prior SD'); prior_mean = float(params['prior_mean'])
    sigma_mu = float(params['proposal_std_mu']); mu = float(params.get('initial_mu', mu_star))
    if not all(math.isfinite(v) for v in [prior_mean, sigma_mu, mu]) or sigma_mu < 0:
        raise ValueError('finite prior mean, initial mu and nonnegative parameter proposal SD required')
    mle_tol = _positive(params.get('mle_tolerance', 1e-9), 'MLE tolerance')
    score_tol = _positive(params.get('score_tolerance', 1e-10), 'score tolerance')
    opts = options(params); mode = mle_check_mode(params); endpoint_methods = {}
    state = selected_state(x, mu_star, k, scale=scale, mle_tolerance=mle_tol, score_tolerance=score_tol)
    if state is None: raise ValueError('initializer must have the released selected GLOBAL MLE')
    seed = int(key) if isinstance(key, (int, np.integer)) else seed_from_key(key)
    rng = np.random.default_rng(seed)
    mus = np.empty(count+1); mus[0] = mu
    xs = np.empty((count+1,n)); xs[0] = x
    counters = dict(proposals=0, accepted=0, mu_accepts=0, selector_calls=1,
                    selector_fallbacks=int(state.selector_fallback), forward_trajectories=0,
                    reverse_trajectories=0, forward_steps=0, reverse_steps=0,
                    projection_iterations=0, position_projection_failures=0,
                    momentum_projection_failures=0, nonregular_projection_branches=0,
                    initial_momentum_projection=0,
                    forward_projection=0, reverse_phase=0, nonpositive_curvature=0,
                    cheap_metropolis=0, global_selection=0)
    maximum = dict(score=abs(state.geometry.score), selected_mle=abs(state.selected_mle-mu_star),
                   tangent=0., reverse_position=0., reverse_momentum=0., delta_h_score=0.)
    min_curvature = state.geometry.curvature; min_transverse = 1.
    records = []; stride = int(params.get('proposal_record_stride', 0))
    record_limit = int(params.get('proposal_record_limit', 24))
    ceiling = _positive(params.get('wall_ceiling_seconds', 3600.), 'wall ceiling')
    start = time.monotonic(); phase_seconds = 0.
    for iteration in range(count):
        # The correction is independent of mu: preserve ordinary parameter MH,
        # with an unfloored uniform and failure preflight.
        noise = float(rng.normal()); standardized = x/scale
        candidate_mu = mu/scale+sigma_mu/scale*noise
        if not all(math.isfinite(unnormalized_mu_logpdf(v, standardized, prior_mean/scale, prior_std/scale, k))
                   for v in [mu/scale, candidate_mu]):
            raise SelectedRATTLENumericalError('parameter_density_nonfinite', 'ordinary parameter density unresolved')
        mu, accepted_mu = update_mu_mh_from_variates(mu/scale, standardized, sigma_mu/scale,
            prior_mean/scale, prior_std/scale, k, noise, float(rng.random()))
        mu *= scale; counters['mu_accepts'] += int(accepted_mu)
        x, state, diag = rattle_step(x, mu, mu_star, k, rng, opts, current_state=state,
            scale=scale, mle_tolerance=mle_tol, score_tolerance=score_tol, mle_check=mode)
        method=diag['endpoint_method']
        if method: endpoint_methods[method]=endpoint_methods.get(method,0)+1
        counters['proposals'] += 1; counters['accepted'] += int(diag['accepted'])
        counters['selector_calls'] += int(diag['selector_called'])
        counters['selector_fallbacks'] += int(diag['selector_fallback'])
        phase_seconds += diag['phase_seconds']
        if diag['rejection']: counters[diag['rejection']] += 1
        phase = diag['phase']
        if phase:
            for name in ['forward','reverse']:
                d = phase[name]
                if d:
                    counters[name+'_trajectories'] += 1; counters[name+'_steps'] += d['completed_steps']
                    counters['projection_iterations'] += d['projection_iterations']
                    counters['position_projection_failures'] += d['position_failures']
                    counters['momentum_projection_failures'] += d['momentum_failures']
                    counters['nonregular_projection_branches'] += int(d['status'] == 2)
                    maximum['tangent'] = max(maximum['tangent'], d['max_tangent_residual'])
                    min_transverse = min(min_transverse, d['min_transverse'])
            if phase['reverse_position_error'] is not None:
                maximum['reverse_position'] = max(maximum['reverse_position'], phase['reverse_position_error'])
                maximum['reverse_momentum'] = max(maximum['reverse_momentum'], phase['reverse_momentum_error'])
        if diag['delta_h_score'] is not None: maximum['delta_h_score'] = max(maximum['delta_h_score'], abs(diag['delta_h_score']))
        maximum['score'] = max(maximum['score'], abs(state.geometry.score))
        maximum['selected_mle'] = max(maximum['selected_mle'], abs(state.selected_mle-mu_star)+getattr(state,'selected_mle_error_bound',0.))
        min_curvature = min(min_curvature, state.geometry.curvature)
        if stride > 0 and iteration%stride == 0 and diag['proposed_x'] is not None and len(records) < record_limit:
            records.append(dict(iteration=iteration+1, mu=mu, **_json_diag(diag)))
        mus[iteration+1], xs[iteration+1] = mu, x
        if verbose and (iteration+1)%5000 == 0: print('selected-MLE RATTLE', seed, iteration+1, round(time.monotonic()-start,2), flush=True)
        if time.monotonic()-start > ceiling: raise TimeoutError(f'selected RATTLE ceiling at iteration {iteration+1}')
    return dict(mu_chain=mus, x_chain=xs, counts=counters, seconds=time.monotonic()-start,
        phase_seconds=phase_seconds, mle_check=mode, endpoint_methods=endpoint_methods,
        seed=seed, target='selected_global_student_location_mle', estimator='hardened_global_likelihood_maximum_exact_ties_lower',
        guiding_potential='score_coarea_with_gram', endpoint_correction='global_indicator_times_positive_curvature',
        mass='identity_in_standardized_residual_coordinates', reverse_check='full_position_and_momentum',
        x_acceptance_rate=counters['accepted']/count, mu_acceptance_rate=counters['mu_accepts']/count,
        numerical_maxima=maximum, min_retained_curvature=min_curvature, min_projection_transversality=min_transverse,
        proposal_records=records, options=opts, mle_tolerance=mle_tol, score_tolerance=score_tol, known_scale=scale,
        numerical_policy='unresolved selection/curvature/nonfinite values raise; failed regular projection/reversal rejects')
