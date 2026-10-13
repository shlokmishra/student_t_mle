"""Bounded, isolated validation of existing location-only selected-MLE code.

Use --config and --phase; source samplers are never modified. Oracle dependency:
SymPy 1.14.0 (see the report for an isolated installation command).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from scripts.selected_mle_reference_oracle import selected_mle_oracle
from models.loc import affine as loc_affine
from models.loc.student import score_gibbs as loc_student
from models.loc.student.mle import global_mle, stationary_points, score, loglik
from kde_ref.moments import raw_weighted_posterior_moments, gaussian_kde_gaussian_prior_moments
from kde_ref.posterior import build_likelihood_kde_backend


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(path):
    return json.loads(path.read_text())


def weighted_quantiles(x, w, probabilities):
    order = np.argsort(x)
    values, weights = np.asarray(x)[order], np.asarray(w)[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return np.interp(probabilities, cumulative, values)


def summarize(mu, weights, latent, case):
    if not np.isfinite(mu).all() or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("invalid reference/chain particles; never silently discard")
    w = weights / weights.sum()
    mean = float(w @ mu)
    q = weighted_quantiles(mu, w, [.05, .5, .95])
    t, width = case['release'], case['tail_width']
    return dict(mean=mean, sd=float(np.sqrt(w @ (mu-mean)**2)),
                q05=float(q[0]), q50=float(q[1]), q95=float(q[2]),
                lower_tail=float(w @ (mu < t-width)), upper_tail=float(w @ (mu > t+width)),
                latent_tail3=float(w @ (np.max(np.abs(latent-t), axis=1) > 3)),
                latent_tail6=float(w @ (np.max(np.abs(latent-t), axis=1) > 6)))


def weight_particles(base, mle, case):
    mu = case['release'] - mle
    x = case['release'] + base - mle[:, None]
    log_w = stats.norm.logpdf(mu, case['prior_mean'], case['prior_std'])
    weights = np.exp(log_w - log_w.max())
    if not np.isfinite(base).all() or not np.isfinite(mle).all():
        raise ValueError("invalid MLE bank; no failed draws may be omitted")
    normalizer = float(np.exp(log_w.max()) * weights.mean())
    return mu, x, weights, normalizer


def bootstrap_se(mu, w, x, case, rng, repetitions):
    rows = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(mu), len(mu))
        rows.append(summarize(mu[indices], w[indices], x[indices], case))
    return {key: float(np.std([r[key] for r in rows], ddof=1)) for key in rows[0]}


def split_rhat(chains):
    half = chains.shape[1] // 2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    def basic(a):
        within = np.mean(np.var(a, axis=1, ddof=1))
        if within == 0:
            return None
        between = a.shape[1] * np.var(a.mean(axis=1), ddof=1)
        return float(np.sqrt(((a.shape[1]-1)*within+between)/a.shape[1]/within))
    def ranked(a):
        ranks = stats.rankdata(a.ravel())
        return stats.norm.ppf((ranks-.375)/(len(ranks)+.25)).reshape(a.shape)
    return max(basic(ranked(split)), basic(ranked(np.abs(split-np.median(split)))))


def distances(chain, particles, weights):
    order = np.argsort(particles)
    particles, weights = particles[order], weights[order]/weights.sum()
    cumulative = np.r_[0., np.cumsum(weights)]
    chain = np.sort(chain)
    grid = np.unique(np.r_[chain, particles])
    d = []
    for side in ('left', 'right'):
        c1 = np.searchsorted(chain, grid, side=side)/len(chain)
        c2 = cumulative[np.searchsorted(particles, grid, side=side)]
        d.append(float(np.max(np.abs(c1-c2))))
    return {'ks': max(d), 'wasserstein': float(stats.wasserstein_distance(chain, particles, v_weights=weights))}


def oracle_error(data, k, oracle, *, all_roots=False):
    result = dict(n=len(data), k=k, data=np.asarray(data).tolist(), oracle=oracle)
    start = time.monotonic()
    try:
        selected = global_mle(data, k)
        result.update(production_selected=selected, score_residual=abs(score(data, selected, k)),
                      location_error=abs(selected-oracle['selected']),
                      loglik_gap=max(0., oracle['loglik']-loglik(data, selected, k)))
        result['location_tolerance'] = max(1e-8, 8*abs(float(np.spacing(oracle['selected']))))
        factor = (k+1)/2  # Production searches at k=1; convert objective units back.
        result['numerical_tie_tolerance'] = 64*np.finfo(float).eps*max(factor, len(data)*factor, abs(oracle['loglik']))
        result['location_disagreement'] = result['location_error'] > result['location_tolerance']
        result['clear_global_failure'] = result['loglik_gap'] > max(1e-9, 4*result['numerical_tie_tolerance'])
        if all_roots:
            roots = stationary_points(data, k)
            expected = np.array([r for r,m in zip(oracle['roots'],oracle['multiplicities']) if m % 2])
            result['production_roots'] = roots.tolist()
            result['oracle_sign_changing_roots'] = expected.tolist()
            result['missing_root_count'] = sum(float(np.min(np.abs(roots-r))) > result['location_tolerance'] for r in expected)
            result['max_returned_root_score'] = max(abs(score(data, r, k)) for r in roots)
    except Exception as exc:
        result.update(production_error=type(exc).__name__+': '+str(exc))
    result['production_seconds'] = time.monotonic()-start
    return result


def selector_audit(cfg, out):
    cases = []
    for k in (1., 2., 3.):
        named = {'three_modes': [-10, 0, 10], 'unequal_clusters': [-10,-10,10,10,10],
                 'outlier': [0,0,100], 'competing_clusters': [0,20,21], 'constant': [3]*3,
                 'tied_n10': [-4]*5+[4]*5, 'six_four': [-4]*6+[4]*4,
                 'far_n3': [0,.25,1e6], 'near_flat_n2': [-float(np.sqrt(k)),float(np.sqrt(k))]}
        for power in (35,45):
            v = [-4.]*5+[4.]*5; v[-1] += 2.**(-power)
            named['near_tie_'+str(power)] = v
        for name, data in named.items():
            cases.append((f'k{k}_{name}', k, np.array(data, dtype=float)))
    rng = np.random.default_rng(cfg['selector_seed'])
    for k,n in ((2.,3),(2.,10),(3.,10)):
        for i in range(cfg['random_selector_cases']):
            cases.append((f'random_{k}_{n}_{i}',k,rng.standard_t(k,n)))
    rows=[]
    for name,k,data in cases:
        try:
            oracle=selected_mle_oracle(data,k)
            row=oracle_error(data,k,oracle,all_roots=True)
            # Determinism/permutation checks do not alter the observed sample.
            row['permutation_error']=abs(global_mle(data[::-1],k)-row.get('production_selected',0))
            shifted = data + 32.
            shifted_oracle=selected_mle_oracle(shifted,k)
            row['shifted_location_error']=abs(global_mle(shifted,k)-shifted_oracle['selected'])
            row['translation_error']=abs(global_mle(shifted,k)-global_mle(data,k)-32.)
        except Exception as exc:
            row={'data':data.tolist(),'k':k,'oracle_error':type(exc).__name__+': '+str(exc)}
        row['case']=name;rows.append(row)
        save(out/'selector_audit.json',rows)
        print('selector',name, row.get('missing_root_count'),row.get('location_error'),flush=True)
    # Extreme inputs diagnose the declared finite-input boundary, not reference cases.
    extreme=[]
    for data in ([0.,1.,1e200],[-1e200,1e200], [0.,1.,2.], [2.**40,2.**40+.25,2.**40+1.5]):
        try:
            val=global_mle(data,2.)
            residual=score(data,val,2.)
            extreme.append({'data':data,'selected':val if np.isfinite(val) else None,'score':residual if np.isfinite(residual) else None})
        except Exception as exc:
            extreme.append({'data':data,'error':type(exc).__name__+': '+str(exc)})
    save(out/'selector_extreme.json',extreme)


def references(cfg, out):
    for index, bank in enumerate(cfg['banks']):
        path=out/(bank['id']+'_reference.npz')
        if path.exists():
            raise FileExistsError(path)
        rng=np.random.default_rng(bank['seed'])
        base=(rng.normal(size=(bank['draws'],bank['n']))*bank.get('scale',1.) if bank['model']=='normal'
              else rng.standard_t(bank['k'],size=(bank['draws'],bank['n'])))
        mles=[]; comparisons=[]; start=time.monotonic()
        for j,data in enumerate(base):
            if bank['model']=='normal':
                mles.append(float(np.mean(data)))
            else:
                try:
                    oracle=selected_mle_oracle(data,bank['k'])
                    mles.append(oracle['selected'])
                    comparison=oracle_error(data,bank['k'],oracle)
                    comparisons.append(comparison)
                    if comparison.get('production_error') or comparison.get('clear_global_failure'):
                        print('REFERENCE SELECTOR DISAGREEMENT',bank['id'],j,flush=True)
                except Exception as exc:
                    save(out/(bank['id']+'_failure.json'),{'index':j,'data':data.tolist(),'error':repr(exc)})
                    raise
            if (j+1)%128==0:
                print('reference',bank['id'],j+1,'seconds',round(time.monotonic()-start,2),flush=True)
            if time.monotonic()-start>cfg['reference_bank_ceiling_seconds']:
                raise TimeoutError('reference bank ceiling; no incomplete bank accepted')
        np.savez_compressed(path,base=base,mle=np.array(mles))
        save(out/(bank['id']+'_selector_comparisons.json'),comparisons)
        save(out/(bank['id']+'_reference_timing.json'),{'seconds':time.monotonic()-start,'draws':len(mles),'failures':0})


def chains(cfg,out):
    for case in cfg['cases']:
        for chain in range(cfg['chains']):
            path=out/(case['id']+f'_chain{chain}.npz')
            if path.exists(): raise FileExistsError(path)
            params=dict(model=case['model'],n=case['n'],k=case.get('k',2.),known_scale=case.get('scale',1.),
                        prior_mean=case['prior_mean'],prior_std=case['prior_std'],num_iterations_T=cfg['iterations'],
                        proposal_std_mu=case['proposal_std_mu'],tau_q=case['tau_q'],store_x_chain=True,
                        initial_mu=case['release']+(-1.5,-.5,.5,1.5)[chain]*case['prior_std'])
            start=time.monotonic()
            result=loc_affine.run_gibbs(case['chain_seed']+chain,case['release'],params,False)
            np.savez_compressed(path,mu=result.pop('mu_chain'),x=result.pop('x_chain'))
            result['x_final']=result['x_final'].tolist()
            result.update(seconds=time.monotonic()-start,params=params,seed=case['chain_seed']+chain)
            save(out/(case['id']+f'_chain{chain}.json'),result)
            print('chain',case['id'],chain,'seconds',round(result['seconds'],2),'accept',result['latent_acceptance_rate'],flush=True)


def analyze(cfg,out):
    results=[]; summary_rows=[]; audit=[]
    for case in cfg['cases']:
        data=np.load(out/(case['bank']+'_reference.npz'))
        mu,x,w,normalizer=weight_particles(data['base'],data['mle'],case)
        ref=summarize(mu,w,x,case)
        ref_se=bootstrap_se(mu,w,x,case,np.random.default_rng(case['chain_seed']+500),cfg['bootstrap_repetitions'])
        helper=raw_weighted_posterior_moments(data['mle'],case['release'],case['prior_mean'],case['prior_std'])
        np.testing.assert_allclose([helper['posterior_mean'],helper['posterior_sd'],helper['normalization_constant']],
                                   [ref['mean'],ref['sd'],normalizer],rtol=1e-12,atol=1e-14)
        traces=[np.load(out/(case['id']+f'_chain{c}.npz')) for c in range(cfg['chains'])]
        cmu=np.array([a['mu'][cfg['burn_in']+1:] for a in traces])
        cx=np.array([a['x'][cfg['burn_in']+1:] for a in traces])
        pooled_mu=cmu.ravel();pooled_x=cx.reshape(-1,case['n'])
        full=summarize(pooled_mu,np.ones(len(pooled_mu)),pooled_x,case)
        batches=[]; per_chain=[]
        for chain in range(cfg['chains']):
            per_chain.append(summarize(cmu[chain],np.ones(cmu.shape[1]),cx[chain],case))
            for lo in range(0,cmu.shape[1]-cfg['batch_length']+1,cfg['batch_length']):
                hi=lo+cfg['batch_length']
                batches.append(summarize(cmu[chain,lo:hi],np.ones(hi-lo),cx[chain,lo:hi],case))
        chain_se={key:max(float(np.std([r[key] for r in batches],ddof=1)/np.sqrt(len(batches))),
                          float(np.std([r[key] for r in per_chain],ddof=1)/np.sqrt(len(per_chain)))) for key in full}
        z={key:float((full[key]-ref[key])/max(1e-15,np.hypot(chain_se[key],ref_se[key]))) for key in full}
        rhat_mu=split_rhat(cmu)
        latent_max=np.max(np.abs(cx-case['release']),axis=2)
        rhat_latent=split_rhat(latent_max)
        mean_se=float(np.sqrt(len(mu)/(len(mu)-1)*np.sum(w*w*(mu-ref['mean'])**2)/w.sum()**2))
        result={'case':case,'reference':ref,'reference_se':ref_se,'reference_mean_influence_se':mean_se,
                'reference_weight_ess':float(w.sum()**2/(w@w)),'normalizer':normalizer,'raw_helper_agrees':True,
                'affine':full,'affine_se':chain_se,'z_difference':z,'per_chain':per_chain,
                'mu_rhat':rhat_mu,'latent_max_rhat':rhat_latent,
                'mean_ess_approx':float(min(len(pooled_mu),full['sd']**2/max(chain_se['mean']**2,1e-30))),
                **distances(pooled_mu,mu,w)}
        # Rejection is an illustration from this same bank, not an independent comparison.
        rej_rng=np.random.default_rng(case['chain_seed']+600)
        accept=rej_rng.random(len(mu))<np.exp(-.5*((mu-case['prior_mean'])/case['prior_std'])**2)
        result['prior_rejection_same_bank']={'accepted':int(accept.sum()),'summary':summarize(mu[accept],np.ones(accept.sum()),x[accept],case)}
        if case['model']=='normal':
            variance=1/(1/case['prior_std']**2+case['n']/case['scale']**2)
            mean=variance*(case['prior_mean']/case['prior_std']**2+case['n']*case['release']/case['scale']**2)
            sd=np.sqrt(variance);lo=case['release']-case['tail_width'];hi=case['release']+case['tail_width']
            analytic=dict(mean=mean,sd=sd,q05=float(stats.norm.ppf(.05,mean,sd)),q50=mean,q95=float(stats.norm.ppf(.95,mean,sd)),lower_tail=float(stats.norm.cdf(lo,mean,sd)),upper_tail=float(stats.norm.sf(hi,mean,sd)))
            result['analytic']=analytic
            result['analytic_z_reference']={key:(ref[key]-value)/max(ref_se[key],1e-15) for key,value in analytic.items()}
            result['analytic_z_affine']={key:(full[key]-value)/max(chain_se[key],1e-15) for key,value in analytic.items()}
            result['gaussian_latent_norm2']={'analytic':case['n']-1,'weighted':float(np.average(np.sum((x-case['release'])**2,axis=1)/case['scale']**2,weights=w)),
                                            'affine':float(np.mean(np.sum((pooled_x-case['release'])**2,axis=1)/case['scale']**2))}
        # Save summaries before potentially expensive exact latent checks.
        results.append(result)
        save(out/'comparison.json',results)
        for key in full:
            summary_rows.append({'case':case['id'],'metric':key,'affine':full[key],'affine_mcse':chain_se[key],
                                 'weighted':ref[key],'weighted_mcse':ref_se[key],'difference_z':z[key],
                                 'analytic':result.get('analytic',{}).get(key,'')})
        with (out/'summary.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(summary_rows[0]));writer.writeheader();writer.writerows(summary_rows)
        # Independently reselect fixed-spaced and extreme retained latent data.
        # Bound exact algebra per state; an unresolved check is retained, never passed.
        def timeout_handler(signum, frame):
            raise TimeoutError('exact latent oracle exceeded 10 seconds')
        for label,states in [('reference',x),('affine',pooled_x)]:
            indices=np.unique(np.r_[np.linspace(0,len(states)-1,cfg['latent_oracle_checks'],dtype=int),np.argmax(np.max(np.abs(states-case['release']),axis=1))])
            for idx in indices:
                row={'case':case['id'],'source':label,'index':int(idx)}
                start=time.monotonic()
                previous=signal.signal(signal.SIGALRM,timeout_handler)
                signal.setitimer(signal.ITIMER_REAL,10.)
                try:
                    estimate=(float(states[idx].mean()) if case['model']=='normal' else selected_mle_oracle(states[idx],case['k'])['selected'])
                    row.update(selected_mle=estimate,residual=abs(estimate-case['release']))
                except Exception as exc:
                    row.update(error=repr(exc),data=states[idx].tolist())
                finally:
                    signal.setitimer(signal.ITIMER_REAL,0.)
                    signal.signal(signal.SIGALRM,previous)
                row['seconds']=time.monotonic()-start
                audit.append(row)
                save(out/'latent_oracle_checks.json',audit)
                if 'error' in row: print('latent unresolved',row,flush=True)
        print('analysis',case['id'],'zmax',max(abs(z[k]) for k in z),'rhat',rhat_mu,rhat_latent,flush=True)
    with (out/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary_rows[0]));writer.writeheader();writer.writerows(summary_rows)


def kde_checks(out):
    # A two-component example separates mixture spread from component centers.
    errors=np.array([-1.,1.]);prior_mean=.7;prior_std=1.3;release=.4
    backend=build_likelihood_kde_backend(errors,{'kde_bw_method':'scott'},verbose=False)
    got=gaussian_kde_gaussian_prior_moments(backend,release,prior_mean,prior_std)
    bandwidth=float(np.sqrt(backend.kde.covariance[0,0]))
    v=1/(1/bandwidth**2+1/prior_std**2)
    means=v*((release-errors)/bandwidth**2+prior_mean/prior_std**2)
    unnormalized=stats.norm.pdf(release-errors,prior_mean,np.sqrt(prior_std**2+bandwidth**2))
    weights=unnormalized/unnormalized.sum()
    quantiles=[brentq(lambda q: float(weights@stats.norm.cdf((q-means)/np.sqrt(v)))-p,-20,20) for p in (.025,.5,.975)]
    save(out/'kde_diagnostic.json',{'helper':got,'correct_mixture_quantiles':quantiles,
                                 'correct_normalizer':float(unnormalized.mean()),'bandwidth':bandwidth,
                                 'note':'Diagnostic only; no KDE enters validation summary or distance.'})


def resolve_latent(cfg,out):
    """Resolve saved oracle timeouts with SymPy's alternative exact algorithm."""
    rows=load(out/'latent_oracle_checks.json')
    results=[]
    cases={c['id']:c for c in cfg['cases']}
    def alarm(signum,frame):
        raise TimeoutError('fast exact isolation exceeded 10 seconds')
    for row in rows:
        if 'error' not in row:
            continue
        result=dict(row)
        case=cases[row['case']]
        previous=signal.signal(signal.SIGALRM,alarm)
        signal.setitimer(signal.ITIMER_REAL,10.)
        start=time.monotonic()
        try:
            oracle=selected_mle_oracle(row['data'],case['k'],fast=True)
            result['resolution']={'algorithm':'SymPy exact intervals fast=True',
                                  'oracle':oracle,'residual':abs(oracle['selected']-case['release'])}
        except Exception as exc:
            result['resolution']={'error':repr(exc)}
        finally:
            signal.setitimer(signal.ITIMER_REAL,0.)
            signal.signal(signal.SIGALRM,previous)
        result['resolution']['seconds']=time.monotonic()-start
        results.append(result)
        save(out/'latent_oracle_resolutions.json',results)
    print('latent resolutions',len(results),'remaining',sum('error' in r['resolution'] for r in results),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['selector','reference','chains','analyze','kde','resolve_latent'],required=True)
    args=parser.parse_args();cfg=load(args.config);out=ROOT/cfg['output']
    out.mkdir(exist_ok=True,parents=True)
    if args.phase=='selector':selector_audit(cfg,out)
    elif args.phase=='reference':references(cfg,out)
    elif args.phase=='chains':chains(cfg,out)
    elif args.phase=='analyze':analyze(cfg,out)
    elif args.phase=='resolve_latent':resolve_latent(cfg,out)
    else:kde_checks(out)


if __name__=='__main__':
    main()
