"""Two frozen n10 cases: new independence MH, unchanged m16 block, direct pilot."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time
from collections import Counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.stats import beta
from models.loc.student import independence as independent
from models.loc.student import selected_block as block
from models.loc.student.mle import global_mle
from scripts.followup_shifted_student_reference import load,save,summary,diagnostics,oracle_check,longest_run,weight_particles,distances
from scripts.validate_student_selected_mle_gibbs import particles,sha


def preflight(cfg,out):
    old=ROOT/cfg['existing_gibbs_evidence'];oldcfg=load(old/'config.json');refs={}
    for case in cfg['cases']:
        _,_,values=particles(case,oldcfg,ROOT/oldcfg['raw_directory'])
        ref=load(old/(case['id']+'_reference.json'))
        actual=summary(values[0],values[2],values[1],case)
        for key in actual:np.testing.assert_allclose(actual[key],ref['summary'][key],atol=1e-12,rtol=1e-12)
        save(out/(case['id']+'_reference.json'),ref);refs[case['id']]=(values,ref)
    sources=[Path(__file__),ROOT/'models/loc/student/independence.py',ROOT/'models/loc/student/selected_block.py',ROOT/'models/loc/student/mle.py',ROOT/'scripts/selected_mle_reference_oracle.py']
    provenance=dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},python=sys.version,cpu_affinity=sorted(os.sched_getaffinity(0)),
        reference_policy='particles() verifies both raw bank hashes and hardened-selector hash; summaries rechecked; original bootstrap uncertainty carried forward')
    save(out/'preflight.json',provenance)
    return refs


def summarize_chains(case,cfg,traces,provenance,reference,reference_values):
    burn=cfg['burn_in'];mus=np.array([r['mu_chain'][burn+1:] for r in traces]);xs=np.array([r['x_chain'][burn+1:] for r in traces])
    maximum=np.max(abs(xs-case['release']),axis=2);full=summary(mus.ravel(),np.ones(mus.size),xs.reshape(-1,case['n']),case)
    per_chain=[];batches=[];movement=[];split=[]
    for i in range(len(mus)):
        a,b=mus[i],xs[i];per_chain.append(summary(a,np.ones(len(a)),b,case))
        half=len(a)//2;split.append([summary(a[sl],np.ones(half),b[sl],case) for sl in (slice(0,half),slice(-half,None))])
        for lo in range(0,len(a),cfg['batch_length']):
            sl=slice(lo,min(lo+cfg['batch_length'],len(a)));batches.append(summary(a[sl],np.ones(len(a[sl])),b[sl],case))
        events=maximum[i]>6;extreme=maximum[i]>cfg['extreme_threshold']
        changed=np.any(b[1:]!=b[:-1],axis=1)
        movement.append(dict(tail6_entries=int(np.sum(~events[:-1]&events[1:])),tail6_exits=int(np.sum(events[:-1]&~events[1:])),
            tail6_longest=longest_run(events),extreme_threshold=cfg['extreme_threshold'],extreme_count=int(extreme.sum()),
            extreme_longest=longest_run(extreme),longest_unchanged_x=longest_run(~changed)+1,maximum_residual=float(maximum[i].max())))
    se={key:max(float(np.std([r[key] for r in batches],ddof=1)/np.sqrt(len(batches))),float(np.std([r[key] for r in per_chain],ddof=1)/np.sqrt(len(mus)))) for key in full}
    for key in ('lower_tail','upper_tail','latent_tail3','latent_tail6'):
        if all(r[key]==0. for r in per_chain) or all(r[key]==1. for r in per_chain):se[key]=None
    # The previously unresolved mu>2 shifted event is outside this bounded comparison.
    if case['id']=='shifted_n10':se['upper_tail']=None
    z={key:(full[key]-reference['summary'][key])/np.hypot(se[key],reference['uncertainty'][key]['se']) if se[key] is not None and reference['uncertainty'][key]['se'] is not None else None for key in full}
    ds=dict(mu=diagnostics(mus),maximum=diagnostics(maximum),tail6=diagnostics((maximum>6).astype(float)))
    seconds=sum(p['seconds'] for p in provenance);counts=Counter()
    for p in provenance:counts.update(p['counts'])
    rmu,rx,rw,_=reference_values
    return dict(case=case,summary=full,mcse=se,difference_mcse=z,per_chain_summary=per_chain,split_summary=split,diagnostics=ds,movement=movement,
        counts=dict(counts),seconds=seconds,microseconds_per_iteration=seconds/(len(mus)*cfg['iterations'])*1e6,
        acceptance=counts['accepted']/(len(mus)*cfg['iterations']),ess_bulk_per_second={key:d['ess_bulk']/seconds if d['ess_bulk'] is not None else None for key,d in ds.items()},
        ess_tail_per_second={key:d['ess_tail']/seconds if d['ess_tail'] is not None else None for key,d in ds.items()},
        parameter_distances=distances(mus.ravel(),rmu,rw),latent_distances=distances(maximum.ravel(),np.max(abs(rx-case['release']),axis=1),rw),
        screen_summary=all(abs(v)<=cfg['difference_mcse_max'] for v in z.values() if v is not None),
        screen_diagnostics=all(d['rhat'] is not None and d['rhat']<=cfg['rhat_max'] and d['ess_bulk'] is not None and d['ess_bulk']>=cfg['ess_min'] and d['ess_tail'] is not None and d['ess_tail']>=cfg['ess_min'] for d in ds.values()))


def chains(cfg,raw,out,refs):
    results=[];checks=[];start=time.monotonic()
    for case in cfg['cases']:
        traces={'independence':[],'block16':[]};prov={'independence':[],'block16':[]}
        for i in range(cfg['chains']):
            previous=load(ROOT/cfg['existing_gibbs_evidence']/(case['id']+f'_chain{i}_provenance.json'))
            x=np.array(previous['initial_x']);mu=previous['initial_mu']
            params={**case,'num_iterations_T':cfg['iterations'],'wall_ceiling_seconds':cfg['chain_ceiling_seconds'],
                'proposal_record_stride':2000,'proposal_record_limit':3}
            ip={**params,'initial_e':x-mu};bp={**params,'initial_x':x,'initial_mu':mu,'block_length':16}
            seed=cfg['seed']+(100 if case['id']=='shifted_n10' else 0)+i
            for method in ('independence','block16'):
                fun=independent.run_independence if method=='independence' else block.run_gibbs
                p=ip if method=='independence' else bp
                fun(seed,case['release'],{**p,'num_iterations_T':32},False)
            for method in (('independence','block16') if i%2==0 else ('block16','independence')):
                stem=case['id']+f'_{method}_chain{i}';path=raw/(stem+'.npz')
                if path.exists():raise FileExistsError(path)
                fun=independent.run_independence if method=='independence' else block.run_gibbs;p=ip if method=='independence' else bp
                try:r=fun(seed,case['release'],p,False)
                except Exception as exc:
                    if hasattr(exc,'partial'):np.savez_compressed(raw/(stem+'_partial.npz'),**exc.partial)
                    save(out/(stem+'_failure.json'),dict(error=repr(exc),iteration=getattr(exc,'iteration',getattr(exc,'block_iteration',None))));raise
                arrays=dict(mu=r['mu_chain'],x=r['x_chain'])
                if method=='independence':arrays.update(log_weight=r['log_weight_chain'],accepted=r['accepted_flags'])
                np.savez_compressed(path,**arrays);traces[method].append(r)
                p={key:v for key,v in r.items() if key not in ('mu_chain','x_chain','log_weight_chain','accepted_flags')}
                p.update(raw_path=str(path.relative_to(ROOT)),sha256=sha(path),initial_x=x.tolist(),initial_mu=mu,initial_e=(x-mu).tolist(),origin=previous['origin'])
                save(out/(stem+'_provenance.json'),p);prov[method].append(p)
                print('chain',stem,round(r['seconds'],3),r['counts']['accepted'],flush=True)
                if i==0:
                    for index in [cfg['burn_in']+1,int(np.argmax(np.max(abs(r['x_chain']-case['release']),axis=1)))]:
                        checks.append(oracle_check(r['x_chain'][index],case,stem+'_retained',index,normalized=True,ceiling=15.))
                    proposal=r['proposal_records'][0]
                    checks.append(oracle_check(proposal['proposed_x'],case,stem+'_proposed',proposal['iteration'],normalized=method=='independence',ceiling=15.))
            if time.monotonic()-start>cfg['total_ceiling_seconds']:raise TimeoutError('chain study ceiling')
        for method in ('independence','block16'):
            values,ref=refs[case['id']];result=summarize_chains(case,cfg,traces[method],prov[method],ref,values);result['method']=method
            results.append(result);save(out/'comparisons.json',results)
            print('summary',case['id'],method,result['acceptance'],result['diagnostics'],result['screen_summary'],result['screen_diagnostics'],flush=True)
        save(out/'oracle_checks.json',checks)
    if any('error' in r or r['selected_location_error']>1e-10 or r.get('released_mle_residual',0)>1e-10 for r in checks):raise ArithmeticError('oracle checks failed; retained evidence')
    save(out/'chains_execution.json',dict(seconds=time.monotonic()-start,chains=16,iterations=16*cfg['iterations'],failures=0,extra_verification_selector_calls=len(checks)))


def direct(cfg,raw,out,refs):
    # Existing construction, one fresh shared base bank for both priors; not a new reference.
    N=cfg['direct_draws'];rng=np.random.default_rng(cfg['direct_seed']);path=raw/'direct_n10.npz'
    if path.exists():raise FileExistsError(path)
    global_mle(np.array([-.3,0.,.3]),2.)
    start=time.perf_counter();base=rng.standard_t(2.,(N,10));mle=np.empty(N);fallbacks=0
    for i,e in enumerate(base):
        try:mle[i],diag=global_mle(e,2.,return_diagnostics=True);fallbacks+=int(diag['fallback'])
        except Exception as exc:save(out/'direct_failure.json',dict(index=i,data=e.tolist(),error=repr(exc)));raise
        if time.perf_counter()-start>cfg['direct_ceiling_seconds']:raise TimeoutError('direct pilot ceiling')
    bank_seconds=time.perf_counter()-start;np.savez_compressed(path,base=base,mle=mle);rows=[]
    for j,case in enumerate(cfg['cases']):
        start=time.perf_counter();mu,x,w,_=weight_particles(base,mle,case)
        accepted=np.random.default_rng(cfg['direct_seed']+100+j).random(N)<np.exp(-.5*((mu-case['prior_mean'])/case['prior_std'])**2)
        process_seconds=time.perf_counter()-start;cost=(bank_seconds+process_seconds)/N
        (rmu,rx,rw,norm),ref=refs[case['id']];a=rw/rw.sum();mean=float(a@rmu);event=np.max(abs(rx-case['release']),axis=1)>6;p=float(a@event)
        meanvar=float(np.sum(a*a*(rmu-mean)**2));eventvar=float(np.sum(a*a*(event-p)**2))
        mu_eff=float(a@((rmu-mean)**2))/meanvar;tail_eff=p*(1-p)/eventvar
        expected=norm*np.sqrt(2*np.pi)*case['prior_std'];count=int(accepted.sum())
        # Stationary independence acceptance estimated from the full fixed bank,
        # E[min(w,w')]/E[w], computed exactly for its empirical weighted distribution.
        lw=-.5*((rmu-case['prior_mean'])/case['prior_std'])**2;ws=np.sort(np.exp(lw));n=len(ws)
        expected_mh=float(np.dot(ws,2*(n-np.arange(n))-1)/(n*n*np.mean(ws)))
        row=dict(case=case,proposals=N,microseconds_per_proposal=cost*1e6,bank_seconds=bank_seconds,processing_seconds=process_seconds,
            selector_calls=N,selector_fallbacks=fallbacks,failures=0,observed_rejection_acceptance=count/N,rejection_accepted=count,
            binomial_ci95=[float(beta.ppf(.025,count,N-count+1)) if count else 0.,float(beta.ppf(.975,count+1,N-count)) if count<N else 1.],
            validated_expected_rejection_acceptance=float(expected),estimated_stationary_independence_acceptance=expected_mh,
            estimated_independent_samples_per_second=expected/cost,observed_independent_samples_per_second=count/(bank_seconds+process_seconds),
            weighted_ess_per_second=float(rw.sum()**2/(rw@rw))/len(rw)/cost,
            weighted_mean_equivalent_per_second=mu_eff/len(rw)/cost,weighted_tail6_equivalent_per_second=tail_eff/len(rw)/cost,
            pilot_weight_ess=float(w.sum()**2/(w@w)),pilot_weighted_summary=summary(mu,w,x,case),
            rejection_summary=summary(mu[accepted],np.ones(count),x[accepted],case) if count else None,
            caveat='Fresh 4096-proposal throughput pilot; reference-based efficiency estimates, not precise shifted rejection validation. Shared bank cost charged in full to each alternative.')
        rows.append(row);np.savez_compressed(raw/(case['id']+'_direct_rejection.npz'),mu=mu[accepted],x=x[accepted])
    save(out/'direct_comparison.json',rows);save(out/'direct_provenance.json',dict(seed=cfg['direct_seed'],raw_path=str(path.relative_to(ROOT)),sha256=sha(path),selector_calls=N,failures=0))


def figures(cfg,raw,out):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    for row,case in enumerate(cfg['cases']):
        for col,method in enumerate(('independence','block16')):
            for i in range(cfg['chains']):
                a=np.load(raw/(case['id']+f'_{method}_chain{i}.npz'));idx=np.arange(0,len(a['mu']),10)
                axes[row,col].plot(idx,np.max(abs(a['x'][idx]-case['release']),axis=1),lw=.5)
            axes[row,col].set_yscale('log');axes[row,col].axvline(cfg['burn_in'],ls=':',color='gray');axes[row,col].set_title(case['id']+' '+method);axes[row,col].set_xlabel('iteration');axes[row,col].set_ylabel('max |X-t|')
    fig.savefig(out/'latent_traces.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();cfg=load(args.config)
    raw=ROOT/cfg['raw_directory'];out=ROOT/cfg['evidence_directory'];raw.mkdir(parents=True,exist_ok=True)
    refs=preflight(cfg,out);chains(cfg,raw,out,refs);direct(cfg,raw,out,refs);figures(cfg,raw,out)
