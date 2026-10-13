"""Pure logistic RATTLE with score guiding force and selected-MLE endpoint MH."""
import time
import numpy as np
from .selected import geometry, check_release, trajectory, potential_force, mu_step
from models.shared.selected_numerics import mh_accept, validated_run


def phase_proposal(x,p,mu,t,h,steps,reverse_tol=1e-8):
    q,pp,ok=trajectory(x,p,mu,t,h,steps)
    if not ok:return q,-pp,False
    back,bp,ok=trajectory(q,-pp,mu,t,h,steps)
    return q,-pp,bool(ok and np.linalg.norm(back-x)<=reverse_tol and np.linalg.norm(bp+p)<=reverse_tol)


def run_rattle(key,mu_star,params,verbose=True):
    x,n,count,mu,pm,ps,sm,seed=validated_run(key,mu_star,params)
    if params.get('known_scale',1.)!=1.:raise ValueError('unit logistic scale required')
    h=float(params.get('rattle_step_size',.05));steps=int(params.get('rattle_num_steps',10));rt=float(params.get('reverse_tolerance',1e-8))
    if not np.isfinite([h,rt]).all() or min(h,rt)<=0 or steps<1 or steps!=params.get('rattle_num_steps',10):raise ValueError('positive finite RATTLE controls required')
    tol=float(params.get('mle_tolerance',1e-10));st=float(params.get('score_tolerance',1e-10));j,_=check_release(x,mu_star,tol,st)
    rng=np.random.default_rng(seed);xs=np.empty((count+1,n));mus=np.empty(count+1);xs[0]=x;mus[0]=mu
    counts=dict(proposals=0,accepted=0,projection_or_reverse=0,mu_accepts=0,root_brackets=1)
    start=time.monotonic();maxscore=0.;maxtangent=0.
    for it in range(count):
        try:
            mu,a=mu_step(mu,x,sm,pm,ps,float(rng.normal()),float(rng.random()));counts['mu_accepts']+=int(a)
            _,_,g,_,normal=geometry(x,mu_star);p=rng.normal(size=n);p-=normal*np.dot(normal,p)/g
            q,pp,ok=phase_proposal(x,p,mu,mu_star,h,steps,rt);counts['proposals']+=1
            if ok:
                c,jp,gp,_,nn=geometry(q,mu_star);tangent=abs(np.dot(nn,pp))
                maxscore=max(maxscore,abs(c));maxtangent=max(maxtangent,tangent)
                if abs(c)>st or tangent>1e-10:raise FloatingPointError('logistic phase constraints unresolved')
                h0=potential_force(x,mu,mu_star)[0]+.5*np.dot(p,p)
                h1=potential_force(q,mu,mu_star)[0]+.5*np.dot(pp,pp)
                if mh_accept(h0-h1+np.log(jp)-np.log(j),float(rng.random())):
                    check_release(q,mu_star,tol,st);counts['root_brackets']+=1
                    x,j=q,jp;counts['accepted']+=1
            else:counts['projection_or_reverse']+=1
            xs[it+1]=x;mus[it+1]=mu
            if time.monotonic()-start>params.get('wall_ceiling_seconds',180):raise TimeoutError('logistic RATTLE ceiling')
        except Exception as exc:
            exc.partial={'mu_chain':mus[:it+1].copy(),'x_chain':xs[:it+1].copy()};exc.iteration=it+1;raise
    return dict(mu_chain=mus,x_chain=xs,target='selected_global_logistic_location_mle',estimator='unique_logistic_score_root',kernel='pure_rattle_score_guide_curvature_endpoint',counts=counts,max_score_residual=maxscore,max_tangent_residual=maxtangent,mle_tolerance=tol,seed=seed,seconds=time.monotonic()-start)
