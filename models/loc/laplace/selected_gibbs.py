"""Odd median / even midpoint conditional Gibbs, with exchangeable labels."""
import time
import numpy as np
from .selected import check_state, gibbs_latent, mu_step
from models.shared.selected_numerics import validated_run
from models.loc.student.selected_block import seed_score_rng


def run_gibbs(key,mu_star,params,verbose=True):
    x,n,count,mu,pm,ps,sm,seed=validated_run(key,mu_star,params)
    b=float(params.get('b',1.));tol=float(params.get('mle_tolerance',1e-10))
    if not np.isfinite(b) or b<=0:raise ValueError('positive finite Laplace scale required')
    q=np.sort(x);maximum=check_state(q,mu_star,tol)
    seeds=np.random.SeedSequence(seed).spawn(2);rng=np.random.default_rng(seeds[0]);seed_score_rng(int(seeds[1].generate_state(1)[0]))
    xs=np.empty((count+1,n));mus=np.empty(count+1);xs[0]=x;mus[0]=mu
    counts=dict(updates=0,mu_accepts=0,label_permutations=0);start=time.monotonic()
    for it in range(count):
        try:
            mu,a=mu_step(mu,q,sm,pm,ps,b,float(rng.normal()),float(rng.random()));counts['mu_accepts']+=int(a)
            q=gibbs_latent(q,mu,mu_star,b);maximum=max(maximum,check_state(q,mu_star,tol))
            xs[it+1]=q[rng.permutation(n)];mus[it+1]=mu;counts['updates']+=1;counts['label_permutations']+=1
            if time.monotonic()-start>params.get('wall_ceiling_seconds',180):raise TimeoutError('Laplace Gibbs ceiling')
        except Exception as exc:
            exc.partial={'mu_chain':mus[:it+1].copy(),'x_chain':xs[:it+1].copy()};exc.iteration=it+1;raise
    return dict(mu_chain=mus,x_chain=xs,target='selected_laplace_midpoint_median',estimator='odd_middle_even_midpoint',kernel='exact_latent_conditionals_parameter_mh',counts=counts,max_selected_mle_residual=maximum,seed=seed,seconds=time.monotonic()-start)
