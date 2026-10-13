"""Reversible logistic score blocks with selected-MLE curvature correction."""
import time
import numpy as np
from .selected import check_release, geometry, score_block, mu_step
from models.shared.selected_numerics import mh_accept, validated_run
from models.loc.student.selected_block import seed_score_rng


def run_gibbs(key,mu_star,params,verbose=True):
    x,n,count,mu,pm,ps,sm,seed=validated_run(key,mu_star,params)
    if params.get('known_scale',1.)!=1.:raise ValueError('unit logistic scale required')
    length=int(params.get('block_length',4));sigma=float(params['proposal_std_z'])
    if length<1 or length!=params.get('block_length',4) or not np.isfinite(sigma) or sigma<=0:raise ValueError('positive block length and score SD required')
    tol=float(params.get('mle_tolerance',1e-10));st=float(params.get('score_tolerance',1e-10))
    j,_=check_release(x,mu_star,tol,st)
    seeds=np.random.SeedSequence(seed).spawn(2);rng=np.random.default_rng(seeds[0]);seed_score_rng(int(seeds[1].generate_state(1)[0]))
    xs=np.empty((count+1,n));mus=np.empty(count+1);xs[0]=x;mus[0]=mu
    counts=dict(blocks=0,accepted=0,mu_accepts=0,score_accepts=0,root_brackets=1)
    start=time.monotonic();maxscore=abs(geometry(x,mu_star)[0])
    for it in range(count):
        try:
            mu,a=mu_step(mu,x,sm,pm,ps,float(rng.normal()),float(rng.random()));counts['mu_accepts']+=int(a)
            proposal,inner=score_block(x,mu,mu_star,sigma,length);counts['score_accepts']+=inner
            c,jp,_,_,_=geometry(proposal,mu_star)
            if abs(c)>st:raise FloatingPointError('logistic block score drift')
            if mh_accept(np.log(jp)-np.log(j),float(rng.random())):
                check_release(proposal,mu_star,tol,st);counts['root_brackets']+=1
                x,j=proposal,jp;counts['accepted']+=1
            counts['blocks']+=1;xs[it+1]=x;mus[it+1]=mu;maxscore=max(maxscore,abs(c))
            if time.monotonic()-start>params.get('wall_ceiling_seconds',180):raise TimeoutError('logistic block ceiling')
        except Exception as exc:
            exc.partial={'mu_chain':mus[:it+1].copy(),'x_chain':xs[:it+1].copy()};exc.iteration=it+1;raise
    return dict(mu_chain=mus,x_chain=xs,target='selected_global_logistic_location_mle',estimator='unique_logistic_score_root',kernel='reversible_score_block_curvature_mh',counts=counts,max_score_residual=maxscore,mle_tolerance=tol,seed=seed,seconds=time.monotonic()-start)
