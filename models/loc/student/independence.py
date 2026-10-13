"""Independent base-data MH for the joint selected Student location-MLE posterior.

Target in E coordinates: product f_0(E_i) * prior(t - T(E)). New iid E
proposals cancel product f_0 in MH. No surface curvature factor belongs here.
Existing samplers and global-MLE selection are not modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from models.loc.student.mle import global_mle
from models.loc.student.mu_mh import seed_from_key


@dataclass(frozen=True)
class State:
    e: np.ndarray
    selected_error_mle: float
    mu: float
    x: np.ndarray
    log_weight: float
    selector_fallback: bool
    target_key: tuple


def _positive(value, name):
    value=float(value)
    if not math.isfinite(value) or value<=0:raise ValueError(name+' must be finite and positive')
    return value


def log_prior_weight(mu, mean, std):
    """Normal log weight with maximum zero; prior normalizer cancels in MH."""
    value=-.5*((float(mu)-mean)/std)**2
    if not math.isfinite(value):raise FloatingPointError('prior weight unresolved')
    return value


def make_state(e, t, k, prior_mean, prior_std, *, scale=1., location_tolerance=2e-13):
    raw=np.asarray(e)
    if np.iscomplexobj(raw):raise ValueError('real base data required')
    e=np.asarray(raw,dtype=float).copy()
    if e.ndim!=1 or not len(e) or not np.isfinite(e).all():raise ValueError('finite nonempty 1D base data required')
    k=_positive(k,'k');scale=_positive(scale,'scale');prior_std=_positive(prior_std,'prior std')
    location_tolerance=_positive(location_tolerance,'MLE location tolerance')
    t=float(t);prior_mean=float(prior_mean)
    if not math.isfinite(t) or not math.isfinite(prior_mean):raise ValueError('finite release/prior mean required')
    standardized=e/scale
    if not np.isfinite(standardized).all():raise FloatingPointError('standardized base data unresolved')
    # Exactly ONE selector invocation; current-state quantities are cached.
    selected,diag=global_mle(standardized,k,location_tol=location_tolerance/scale,return_diagnostics=True)
    selected*=scale;mu=t-selected;x=e+mu
    if not math.isfinite(selected) or not math.isfinite(mu) or not np.isfinite(x).all():
        raise FloatingPointError('selected-MLE reconstruction nonfinite')
    if diag.get('exact_tie',False):
        # Tied modes can exchange order under nonuniform rounding of a common
        # shift. Certify an exact common translation without a second MLE call.
        # This deliberately conservative guard may abort a harmless tied shift.
        from fractions import Fraction
        shifts=[Fraction(float(a))-Fraction(float(b)) for a,b in zip(x,e)]
        if any(v!=shifts[0] for v in shifts[1:]):
            raise FloatingPointError('tied selected-MLE reconstruction has nonuniform rounding')
    # Floating reconstruction must not silently change the latent residual data.
    error=np.abs((x-t)-(e-selected))
    allowance=64*np.finfo(float).eps*np.maximum(scale,np.abs(e-selected))
    if not np.isfinite(error).all() or np.any(error>allowance):
        raise FloatingPointError('selected-MLE reconstruction loses residual precision')
    weight=log_prior_weight(mu,prior_mean,prior_std)
    e.setflags(write=False);x.setflags(write=False)
    return State(e,float(selected),float(mu),x,weight,bool(diag['fallback']),
                 (t,k,prior_mean,prior_std,scale,location_tolerance))


def log_acceptance(current, proposed):
    ratio=proposed.log_weight-current.log_weight
    if not math.isfinite(ratio):raise FloatingPointError('independence acceptance unresolved')
    return min(0.,ratio)


def independence_step(current, rng, t, k, prior_mean, prior_std, *, scale=1., location_tolerance=2e-13):
    if current.target_key != tuple(map(float,(t,k,prior_mean,prior_std,scale,location_tolerance))):
        raise ValueError('cached current state belongs to a different target')
    e=scale*rng.standard_t(k,len(current.e))
    candidate=make_state(e,t,k,prior_mean,prior_std,scale=scale,location_tolerance=location_tolerance)
    u=float(rng.random());logu=math.log(u) if u>0. else -math.inf
    accepted=logu<log_acceptance(current,candidate)
    return candidate if accepted else current,accepted,candidate


def run_independence(key, mu_star, params, verbose=True):
    """Joint chain: mu is reconstructed from E, with no separate parameter update."""
    n=int(params['n']);count=int(params['num_iterations_T'])
    if n<1 or n!=params['n'] or count<1 or count!=params['num_iterations_T']:
        raise ValueError('positive integer n and iteration count required')
    k=_positive(params['k'],'k');scale=_positive(params.get('known_scale',1.),'scale')
    pm=float(params['prior_mean']);ps=_positive(params['prior_std'],'prior std')
    tol=_positive(params.get('location_tolerance',2e-13),'MLE location tolerance')
    seed=int(key) if isinstance(key,(int,np.integer)) else seed_from_key(key)
    rng=np.random.default_rng(seed)
    initial=params.get('initial_e')
    if initial is None:initial=scale*rng.standard_t(k,n)
    state=make_state(initial,mu_star,k,pm,ps,scale=scale,location_tolerance=tol)
    if len(state.e)!=n:raise ValueError('initial_e length must equal n')
    mus=np.empty(count+1);xs=np.empty((count+1,n));log_weights=np.empty(count+1)
    mus[0]=state.mu;xs[0]=state.x;log_weights[0]=state.log_weight
    accepted_flags=np.zeros(count,dtype=bool)
    counts=dict(proposals=0,accepted=0,selector_calls=1,selector_fallbacks=int(state.selector_fallback))
    records=[];stride=int(params.get('proposal_record_stride',0));limit=int(params.get('proposal_record_limit',4))
    start=time.monotonic();ceiling=_positive(params.get('wall_ceiling_seconds',120.),'wall ceiling')
    for it in range(count):
        try:
            state,accepted,candidate=independence_step(state,rng,mu_star,k,pm,ps,scale=scale,location_tolerance=tol)
        except Exception as exc:
            exc.partial=dict(mu_chain=mus[:it+1].copy(),x_chain=xs[:it+1].copy(),log_weight_chain=log_weights[:it+1].copy())
            exc.iteration=it+1
            raise
        accepted_flags[it]=accepted;counts['proposals']+=1;counts['selector_calls']+=1
        counts['accepted']+=int(accepted);counts['selector_fallbacks']+=int(candidate.selector_fallback)
        if stride>0 and it%stride==0 and len(records)<limit:
            records.append(dict(iteration=it+1,accepted=bool(accepted),proposed_e=candidate.e.tolist(),
                                proposed_x=candidate.x.tolist(),proposed_mu=candidate.mu,selected_error_mle=candidate.selected_error_mle))
        mus[it+1]=state.mu;xs[it+1]=state.x;log_weights[it+1]=state.log_weight
        if verbose and (it+1)%3000==0:print('independence MH',seed,it+1,round(time.monotonic()-start,3),flush=True)
        if time.monotonic()-start>ceiling:
            exc=TimeoutError('bounded independence chain exceeded ceiling')
            exc.partial=dict(mu_chain=mus[:it+2].copy(),x_chain=xs[:it+2].copy(),log_weight_chain=log_weights[:it+2].copy())
            exc.iteration=it+1;raise exc
    return dict(mu_chain=mus,x_chain=xs,log_weight_chain=log_weights,accepted_flags=accepted_flags,
                seed=seed,counts=counts,seconds=time.monotonic()-start,proposal_records=records,known_scale=scale,
                target='selected_global_student_location_mle',proposal='independent_standardized_dataset',
                numerical_policy='hardened global selector once per proposal; unresolved failures propagate; no uniform floor or curvature weight')
