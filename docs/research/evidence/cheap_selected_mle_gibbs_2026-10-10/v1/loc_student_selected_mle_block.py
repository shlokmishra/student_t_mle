"""Experimental reversible score blocks with one selected-MLE endpoint correction.

No legacy sampler is changed. Uniform random scan at fixed mu makes K**m
self-adjoint. Numerical inverse branches and weights use regular, untrimmed
charts; unresolved calculations raise, rather than silently changing support.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

from . import loc_student as legacy
from .loc_student_selected_mle import _branch_log_weight
from .loc_student_selected_mle_rattle import geometry, selected_state
from .student_mu_mh import seed_from_key, unnormalized_mu_logpdf, update_mu_mh_from_variates


@njit(cache=True)
def seed_score_rng(seed):
    # Numba's independent thread-local RNG; no Python-global RNG mutation.
    np.random.seed(seed)


@njit(cache=True)
def branch_law(z, loc, k):
    """Sorted finite inverses, lower-branch probability, and score density."""
    bound = 1/(2*math.sqrt(k))
    if not math.isfinite(z) or not -bound < z < bound:
        raise FloatingPointError('score chart boundary unresolved')
    if z == 0.:
        w = _branch_log_weight(0., loc, k)
        if not math.isfinite(w): raise FloatingPointError('zero branch weight unresolved')
        return 0., 0., 1., w
    discriminant = 1-4*k*z*z
    if discriminant <= 0: raise FloatingPointError('inverse fold unresolved')
    root = math.sqrt(discriminant)
    inner = 2*k*z/(1+root); outer = (1+root)/(2*z)
    a, b = min(inner, outer), max(inner, outer)
    if not math.isfinite(a) or not math.isfinite(b) or a == b:
        raise FloatingPointError('inverse branch resolution')
    wa, wb = _branch_log_weight(a, loc, k), _branch_log_weight(b, loc, k)
    if not math.isfinite(wa) or not math.isfinite(wb):
        raise FloatingPointError('inverse branch weights unresolved')
    m = max(wa, wb); total = math.exp(wa-m)+math.exp(wb-m)
    return a, b, math.exp(wa-m)/total, m+math.log(total)


@njit(cache=True)
def score_log_ratio(z, zp, delta, loc, k, sigma):
    bound = 1/(2*math.sqrt(k)); lo, hi = max(-bound, delta-bound), min(bound, delta+bound)
    if not lo < z < hi or not lo < zp < hi:
        raise FloatingPointError('score proposal interval unresolved')
    q = branch_law(z,loc,k)[3]+branch_law(delta-z,loc,k)[3]
    qp = branch_law(zp,loc,k)[3]+branch_law(delta-zp,loc,k)[3]
    lc = legacy.truncnorm_log_normalizer(z,sigma,lo,hi)
    lp = legacy.truncnorm_log_normalizer(zp,sigma,lo,hi)
    ratio = qp-q+lc-lp
    if not math.isfinite(ratio): raise FloatingPointError('score acceptance unresolved')
    return ratio


@njit(cache=True)
def score_pair_kernel(x, i, j, mu, t, k, sigma, max_draws=10000):
    """Marginal-score MH followed by BOTH conditional branches, including rejection."""
    zi, zj = legacy.psi(x[i]-t,k), legacy.psi(x[j]-t,k)
    delta = zi+zj; bound = 1/(2*math.sqrt(k))
    lo, hi = max(-bound,delta-bound), min(bound,delta+bound)
    if not lo < zi < hi: raise FloatingPointError('current score interval unresolved')
    found = False; zp = math.nan; draws = 0
    # Normal rejection has the intended truncated-normal law in exact arithmetic.
    # No inverse-normal approximation, probability clamp or endpoint atom.
    for _ in range(max_draws):
        zp = zi+sigma*np.random.normal(); draws += 1
        if lo < zp < hi: found = True; break
    if not found: raise FloatingPointError('truncated-normal sampling budget exhausted')
    ratio = score_log_ratio(zi,zp,delta,mu-t,k,sigma)
    u = np.random.random(); logu = math.log(u) if u > 0 else -math.inf
    accepted = logu < min(0.,ratio)
    z = zp if accepted else zi; zj_new = delta-z
    a,b,pa,_ = branch_law(z,mu-t,k)
    c,d,pc,_ = branch_law(zj_new,mu-t,k)
    yi = a if np.random.random() < pa else b
    yj = c if np.random.random() < pc else d
    xi, xj = t+yi,t+yj
    error = max(abs(legacy.psi(xi-t,k)-z),abs(legacy.psi(xj-t,k)-zj_new))
    if not math.isfinite(xi) or not math.isfinite(xj) or error > 64*np.finfo(np.float64).eps*bound:
        raise FloatingPointError('inverse score reconstruction unresolved')
    changed = int((abs(x[i]-t)>math.sqrt(k)) != (abs(yi)>math.sqrt(k)))
    changed += int((abs(x[j]-t)>math.sqrt(k)) != (abs(yj)>math.sqrt(k)))
    x[i],x[j] = xi,xj
    return int(accepted),draws,changed,error


@njit(cache=True)
def score_block_kernel(x, mu, t, k, sigma, length):
    proposed = x.copy(); accept = draws = changed = 0; error = 0.
    for _ in range(length):
        i = np.random.randint(len(x)); j = np.random.randint(len(x)-1)
        if j >= i: j += 1
        a,d,b,e = score_pair_kernel(proposed,i,j,mu,t,k,sigma)
        accept += a; draws += d; changed += b; error = max(error,e)
    return proposed,accept,draws,changed,error


def block_step(x, mu, t, k, sigma, length, rng, state, *, mle_tolerance=1e-10, score_tolerance=1e-10):
    x = np.asarray(x,dtype=float)
    if state.geometry.state_key != (x.tobytes(),float(t),float(k),1.):
        raise ValueError('cached selected state does not match block input')
    start = time.perf_counter()
    proposed,a,d,b,e = score_block_kernel(x,mu,t,k,sigma,length)
    seconds = time.perf_counter()-start
    g = geometry(proposed,t,k,score_tolerance=score_tolerance)
    ratio = g.log_j-state.geometry.log_j
    diag = dict(accepted=False,selector_called=False,selector_fallback=False,
        rejection=None,score_accepts=a,normal_draws=d,branch_changes=b,
        max_inverse_error=e,proposal_seconds=seconds,selector_seconds=0.,
        log_correction=ratio,proposed_x=proposed)
    if g.curvature < 0:
        diag['rejection']='nonpositive_curvature';return x.copy(),state,diag
    u = float(rng.random()); logu = math.log(u) if u > 0 else -math.inf
    if logu >= min(0.,ratio):
        diag['rejection']='cheap_metropolis';return x.copy(),state,diag
    diag['selector_called']=True;sd={};start=time.perf_counter()
    candidate = selected_state(proposed,t,k,mle_tolerance=mle_tolerance,
        score_tolerance=score_tolerance,endpoint_geometry=g,selection_diagnostics=sd)
    diag['selector_seconds']=time.perf_counter()-start
    diag['selector_fallback']=bool(sd.get('fallback',False))
    if candidate is None:
        diag['rejection']='global_selection';return x.copy(),state,diag
    diag['accepted']=True
    return proposed,candidate,diag


def run_gibbs(key, mu_star, params, verbose=True):
    """Parameter MH, then a fixed-mu random-scan block and one endpoint MH test."""
    x=np.asarray(params['initial_x'])
    if np.iscomplexobj(x):raise ValueError('real data required')
    x=np.asarray(x,dtype=float).copy();n=int(params['n']);count=int(params['num_iterations_T'])
    length=int(params['block_length']);k=float(params['k']);sigma=float(params['proposal_std_z'])
    mu=float(params.get('initial_mu',mu_star));pm=float(params['prior_mean']);ps=float(params['prior_std']);sm=float(params['proposal_std_mu'])
    if x.ndim!=1 or len(x)!=n or n!=params['n'] or n<3 or not np.isfinite(x).all():raise ValueError('finite matching sample, n>=3 required')
    if count<1 or count!=params['num_iterations_T'] or length<1 or length!=params['block_length']:raise ValueError('positive integer iterations/block length required')
    if not all(math.isfinite(v) for v in [k,sigma,mu,pm,ps,sm]) or min(k,sigma,ps)<=0 or sm<0:raise ValueError('finite positive shape/proposal/prior parameters required')
    if params.get('known_scale',1.)!=1.:raise ValueError('prototype validates unit scale only')
    mt=float(params.get('mle_tolerance',1e-10));st=float(params.get('score_tolerance',1e-10))
    state=selected_state(x,mu_star,k,mle_tolerance=mt,score_tolerance=st)
    if state is None:raise ValueError('initializer must have the released selected global MLE')
    seed=int(key) if isinstance(key,(int,np.integer)) else seed_from_key(key)
    ss=np.random.SeedSequence(seed).spawn(2);rng=np.random.default_rng(ss[0]);score_seed=int(ss[1].generate_state(1)[0]);seed_score_rng(score_seed)
    mus=np.empty(count+1);xs=np.empty((count+1,n));mus[0]=mu;xs[0]=x
    counts=dict(blocks=0,accepted=0,mu_accepts=0,score_pairs=0,score_accepts=0,
        normal_draws=0,branch_changes_proposed=0,branch_changes_accepted=0,selector_calls=1,
        selector_fallbacks=int(state.selector_fallback),nonpositive_curvature=0,cheap_metropolis=0,global_selection=0)
    times=dict(proposal=0.,selector=0.)
    max_score=abs(state.geometry.score);max_mle=abs(state.selected_mle-mu_star);max_inverse=0.
    records=[];stride=int(params.get('proposal_record_stride',0));limit=int(params.get('proposal_record_limit',8))
    ceiling=float(params.get('wall_ceiling_seconds',180.));start=time.monotonic()
    for it in range(count):
        noise=float(rng.normal());candidate=mu+sm*noise
        if not all(math.isfinite(unnormalized_mu_logpdf(v,x,pm,ps,k)) for v in [mu,candidate]):raise FloatingPointError('parameter density unresolved')
        mu,accepted_mu=update_mu_mh_from_variates(mu,x,sm,pm,ps,k,noise,float(rng.uniform(legacy.EPS_U,1.)))
        counts['mu_accepts']+=int(accepted_mu)
        x,state,d=block_step(x,mu,mu_star,k,sigma,length,rng,state,mle_tolerance=mt,score_tolerance=st)
        counts['blocks']+=1;counts['accepted']+=int(d['accepted']);counts['score_pairs']+=length
        for target,source in [('score_accepts','score_accepts'),('normal_draws','normal_draws'),('branch_changes_proposed','branch_changes'),('selector_calls','selector_called'),('selector_fallbacks','selector_fallback')]:counts[target]+=int(d[source])
        if d['accepted']:counts['branch_changes_accepted']+=d['branch_changes']
        if d['rejection']:counts[d['rejection']]+=1
        times['proposal']+=d['proposal_seconds'];times['selector']+=d['selector_seconds']
        max_score=max(max_score,abs(state.geometry.score));max_mle=max(max_mle,abs(state.selected_mle-mu_star));max_inverse=max(max_inverse,d['max_inverse_error'])
        if stride>0 and it%stride==0 and len(records)<limit:
            records.append(dict(iteration=it+1,accepted=d['accepted'],rejection=d['rejection'],selector_called=d['selector_called'],proposed_x=d['proposed_x'].tolist()))
        mus[it+1]=mu;xs[it+1]=x
        if verbose and (it+1)%3000==0:print('selected score block',length,seed,it+1,round(time.monotonic()-start,3),flush=True)
        if time.monotonic()-start>ceiling:raise TimeoutError('bounded selected score-block chain exceeded ceiling')
    return dict(mu_chain=mus,x_chain=xs,counts=counts,seconds=time.monotonic()-start,times=times,
        seed=seed,score_seed=score_seed,target='selected_global_student_location_mle',proposal='random_scan_regular_score_pair_power',
        block_length=length,max_score_residual=max_score,max_selected_mle_residual=max_mle,max_inverse_score_error=max_inverse,
        proposal_records=records,known_scale=1.,numerical_policy='no artificial score/Jacobian/proposal cutoffs; unresolved computation raises; inherited parameter uniform guard retained')
