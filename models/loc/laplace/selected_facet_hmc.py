"""Reflective facet HMC for the selected median; no latent Gibbs updates.

An ordered representation removes redundant labels. Uniform permutations lift
its invariant law to the full exchangeable dataset. This is not smooth RATTLE.
"""
import time
import numpy as np
from .selected import check_state,tangent,trajectory,potential,mu_step
from models.shared.selected_numerics import mh_accept,validated_run


def phase_proposal(q,p,mu,b,h,steps,reverse_tol=1e-8):
    new,pp,events=trajectory(q,p,mu,b,h,steps)
    back,bp,revents=trajectory(new,-pp,mu,b,h,steps)
    error=max(float(np.linalg.norm(back-q)),float(np.linalg.norm(bp+p)))
    return new,-pp,bool(np.isfinite(error) and error<=reverse_tol),events,error


def run_hmc(key,mu_star,params,verbose=True):
    x,n,count,mu,pm,ps,sm,seed=validated_run(key,mu_star,params)
    b=float(params.get('b',1.));h=float(params.get('rattle_step_size',.05));steps=int(params.get('rattle_num_steps',10));rt=float(params.get('reverse_tolerance',1e-8));tol=float(params.get('mle_tolerance',1e-10))
    if not np.isfinite([b,h,rt]).all() or min(b,h,rt)<=0 or steps<1 or steps!=params.get('rattle_num_steps',10):raise ValueError('positive finite Hamiltonian controls required')
    q=np.sort(x);maximum=check_state(q,mu_star,tol);rng=np.random.default_rng(seed)
    if np.any(np.diff(q)<=0):raise ValueError('dispersed strictly ordered initial observations required')
    xs=np.empty((count+1,n));mus=np.empty(count+1);xs[0]=x;mus[0]=mu
    counts=dict(proposals=0,accepted=0,mu_accepts=0,reverse_failures=0,wall_events=0,label_permutations=0);start=time.monotonic();reverse_error=0.
    for it in range(count):
        try:
            mu,a=mu_step(mu,q,sm,pm,ps,b,float(rng.normal()),float(rng.random()));counts['mu_accepts']+=int(a)
            p=tangent(rng.normal(size=n));new,pp,ok,events,error=phase_proposal(q,p,mu,b,h,steps,rt)
            counts['proposals']+=1;counts['wall_events']+=events;reverse_error=max(reverse_error,error)
            if ok:
                check_state(new,mu_star,tol)
                ratio=potential(q,mu,b)+.5*np.dot(p,p)-potential(new,mu,b)-.5*np.dot(pp,pp)
                if mh_accept(ratio,float(rng.random())):q=new;counts['accepted']+=1
            else:counts['reverse_failures']+=1
            maximum=max(maximum,check_state(q,mu_star,tol));xs[it+1]=q[rng.permutation(n)];mus[it+1]=mu;counts['label_permutations']+=1
            if time.monotonic()-start>params.get('wall_ceiling_seconds',180):raise TimeoutError('Laplace HMC ceiling')
        except Exception as exc:
            exc.partial={'mu_chain':mus[:it+1].copy(),'x_chain':xs[:it+1].copy()};exc.iteration=it+1;raise
    return dict(mu_chain=mus,x_chain=xs,target='selected_laplace_midpoint_median',estimator='odd_middle_even_midpoint',kernel='reflective_ordered_facet_hmc',counts=counts,max_selected_mle_residual=maximum,max_reverse_error=reverse_error,seed=seed,seconds=time.monotonic()-start)
