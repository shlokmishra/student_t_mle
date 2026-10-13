"""One bounded three-length score-block study and direct-simulation cost pilot."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='1'
import numpy as np
from scipy.stats import beta
from models import loc_student_selected_mle_block as selected
from models.student_location_mle import global_mle
from scripts.validate_student_selected_mle_rattle import preflight,starts,old_paths
from scripts.validate_student_selected_mle_gibbs import sha,particles
from scripts.followup_shifted_student_reference import load,save,summary,diagnostics,distances,oracle_check,longest_run,weight_particles


def parameters(case,cfg,x,mu,length,steps):
    return {**case,'initial_x':x,'initial_mu':mu,'block_length':length,'num_iterations_T':steps,
        'mle_tolerance':cfg['mle_tolerance'],'score_tolerance':cfg['score_tolerance'],
        'wall_ceiling_seconds':cfg['chain_ceiling_seconds'],'proposal_record_stride':1000,'proposal_record_limit':6}


def pilot(cfg,raw,out):
    rows=[]
    for case in cfg['cases']:
        _,x,mu,_=list(starts(case,cfg))[1]
        r=selected.run_gibbs(2026103100+case['n'],case['release'],parameters(case,cfg,x,mu,4,400),False)
        r.pop('mu_chain');r.pop('x_chain');rows.append({'case':case['id'],**r});save(out/'pilot.json',rows)
        print('pilot',case['id'],r['seconds'],r['counts'],flush=True)


def chains(cfg,raw,out):
    preflight(cfg,out);checks=[]
    for case in cfg['cases']:
        for length in cfg['block_lengths']:
            for i,x,mu,origin in starts(case,cfg):
                stem=case['id']+f'_m{length}_chain{i}';path=raw/(stem+'.npz')
                if path.exists():raise FileExistsError(path)
                check=oracle_check(x,case,stem+'_initializer',0,normalized=True);checks.append(check);save(out/'initializer_oracle_checks.json',checks)
                if 'error' in check or check['released_mle_residual']>cfg['mle_tolerance']:raise ArithmeticError('initializer failed oracle')
                try:r=selected.run_gibbs(case['chain_seed']+100*length+i,case['release'],parameters(case,cfg,x,mu,length,cfg['iterations']),True)
                except Exception as exc:
                    import traceback
                    save(out/(stem+'_failure.json'),{'error':repr(exc),'traceback':traceback.format_exc(),'initial_x':x.tolist(),'initial_mu':mu});raise
                mus,xs=r.pop('mu_chain'),r.pop('x_chain');np.savez_compressed(path,mu=mus,x=xs)
                save(out/(stem+'_provenance.json'),{**r,'origin':origin,'initial_x':x.tolist(),'initial_mu':mu,'failures':0,'raw_path':str(path.relative_to(ROOT)),'sha256':sha(path)})
                print('complete',stem,r['seconds'],r['counts'],flush=True)


def analyze(cfg,raw,out):
    preflight(cfg,out);old,oraw=old_paths(cfg);ocfg=load(old/'config.json');comparisons=[];rows=[];checks=[]
    gibbs=load(old/'comparisons.json');rattle=load(ROOT/cfg['rattle_evidence']/'comparisons.json')
    for case in cfg['cases']:
        ref=load(out/(case['id']+'_reference.json'));_,_,(rmu,rx,rw,_)=particles(case,ocfg,oraw)
        fixed_g=next(r for r in gibbs if r['case']['id']==case['id']);fixed_r=next(r for r in rattle if r['case']['id']==case['id'])
        gp=[load(old/(case['id']+f'_chain{i}_provenance.json')) for i in range(cfg['chains'])];gs=sum(p['seconds'] for p in gp)
        for length in cfg['block_lengths']:
            traces=[np.load(raw/(case['id']+f'_m{length}_chain{i}.npz')) for i in range(cfg['chains'])]
            provenance=[load(out/(case['id']+f'_m{length}_chain{i}_provenance.json')) for i in range(cfg['chains'])]
            mu=np.array([a['mu'][cfg['burn_in']+1:] for a in traces]);x=np.array([a['x'][cfg['burn_in']+1:] for a in traces]);maximum=np.max(np.abs(x-case['release']),axis=2)
            full=summary(mu.ravel(),np.ones(mu.size),x.reshape(-1,case['n']),case);per_chain=[];batches=[];splits=[];movement=[]
            for i in range(cfg['chains']):
                per_chain.append(summary(mu[i],np.ones(len(mu[i])),x[i],case));half=len(mu[i])//2
                splits.append([summary(mu[i,s],np.ones(half),x[i,s],case) for s in [slice(0,half),slice(-half,None)]])
                for lo in range(0,len(mu[i]),cfg['batch_length']):
                    hi=lo+cfg['batch_length'];batches.append(summary(mu[i,lo:hi],np.ones(hi-lo),x[i,lo:hi],case))
                event=maximum[i]>6;movement.append({'tail6_entries':int(np.sum(~event[:-1]&event[1:])),'tail6_exits':int(np.sum(event[:-1]&~event[1:])), 'longest_tail6_run':longest_run(event),'max_residual':float(maximum[i].max())})
                idx=np.unique(np.r_[np.linspace(0,len(mu[i])-1,cfg['retained_oracle_checks'],dtype=int),np.argmax(maximum[i])])
                for j in idx:checks.append(oracle_check(x[i,j],case,case['id']+f'_m{length}_retained{i}',j,normalized=True))
                for r in provenance[i]['proposal_records']:
                    o=oracle_check(r['proposed_x'],case,case['id']+f'_m{length}_proposed{i}',r['iteration'])
                    if r['accepted']:o['released_mle_residual']=abs(o.get('oracle',{}).get('selected',np.inf)-case['release'])
                    o['block_accepted']=r['accepted'];o['rejection']=r['rejection'];checks.append(o)
            se={key:max(float(np.std([r[key] for r in batches],ddof=1)/np.sqrt(len(batches))),float(np.std([r[key] for r in per_chain],ddof=1)/np.sqrt(cfg['chains']))) for key in full}
            if case['id']=='shifted_n10':se['upper_tail']=None
            for key in ['lower_tail','upper_tail','latent_tail3','latent_tail6']:
                if all(r[key]==0 for r in per_chain) or all(r[key]==1 for r in per_chain):se[key]=None
            diag={'mu':diagnostics(mu),'latent_max':diagnostics(maximum),'latent_tail6':diagnostics((maximum>6).astype(float))};seconds=sum(p['seconds'] for p in provenance)
            for d in diag.values():d.update(ess_bulk_per_second=d['ess_bulk']/seconds if d['ess_bulk'] is not None else None,ess_tail_per_second=d['ess_tail']/seconds if d['ess_tail'] is not None else None)
            z={key:(full[key]-ref['summary'][key])/np.hypot(se[key],ref['uncertainty'][key]['se']) if se[key] is not None and ref['uncertainty'][key]['se'] is not None and np.hypot(se[key],ref['uncertainty'][key]['se'])>0 else None for key in full}
            screens={name:d['rhat'] is not None and d['rhat']<=cfg['rhat_max'] and all(d[k] is not None and d[k]>=cfg['ess_min'] for k in ['ess_bulk','ess_tail']) for name,d in diag.items()}
            screens['summary_comparison']=all(z[key] is not None and abs(z[key])<=cfg['difference_mcse_max'] for key in full if not(case['id']=='shifted_n10' and key=='upper_tail'))
            counts={key:sum(p['counts'][key] for p in provenance) for key in provenance[0]['counts']};timers={key:sum(p['times'][key] for p in provenance) for key in provenance[0]['times']}
            r={'case':case,'block_length':length,'summary':full,'mcse_diagnostic':se,'per_chain':per_chain,'split_summaries':splits,'movement':movement,'diagnostics':diag,'screens':screens,'passes_declared_screens':all(screens.values()),'difference_mcse':z,'retained':mu.size,'seconds':seconds,'counts':counts,'times':timers,'acceptance':counts['accepted']/counts['blocks'], 'selector_calls_per_accepted_block':(counts['selector_calls']-cfg['chains'])/counts['accepted'],'microseconds_per_block':seconds/counts['blocks']*1e6,'microseconds_per_accepted_block':seconds/counts['accepted']*1e6,'selector_microseconds_per_accepted_block':timers['selector']/counts['accepted']*1e6,'selector_microseconds_per_call':timers['selector']/(counts['selector_calls']-cfg['chains'])*1e6,'parameter_distances':distances(mu.ravel(),rmu,rw),'latent_max_distances':distances(maximum.ravel(),np.max(np.abs(rx-case['release']),axis=1),rw),'max_score_residual':max(p['max_score_residual'] for p in provenance),'max_selected_mle_residual':max(p['max_selected_mle_residual'] for p in provenance),'max_inverse_score_error':max(p['max_inverse_score_error'] for p in provenance),'fixed_comparisons':{'gibbs_summary':fixed_g['summary'],'gibbs_ess_bulk_per_second':{k:d['ess_bulk']/gs for k,d in fixed_g['diagnostics'].items()},'rattle_summary':fixed_r['summary'],'rattle_diagnostics':fixed_r['diagnostics']}}
            comparisons.append(r);save(out/'comparisons.json',comparisons);save(out/'oracle_checks.json',checks)
            for key in full:rows.append({'case':case['id'],'length':length,'metric':key,'reference':ref['summary'][key],'reference_mcse':ref['uncertainty'][key]['se'],'block':full[key],'block_mcse':se[key],'difference_mcse':z[key]})
            print('analyze',case['id'],length,diag,screens,flush=True)
    with (out/'summary.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if any('error' in r or r.get('released_mle_residual',0)>cfg['mle_tolerance'] or r.get('selected_location_error',0)>cfg['mle_tolerance'] for r in checks):raise ArithmeticError('oracle audit failed; see records')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13,9),constrained_layout=True)
    for row,case in enumerate(cfg['cases']):
        for col,m in enumerate(cfg['block_lengths']):
            for i in range(cfg['chains']):
                a=np.load(raw/(case['id']+f'_m{m}_chain{i}.npz'));stride=max(1,len(a['mu'])//600)
                axes[row,col].plot(np.arange(len(a['mu']))[::stride],np.max(np.abs(a['x'][::stride]-case['release']),axis=1),lw=.6,label=str(i))
            axes[row,col].set_yscale('log');axes[row,col].axvline(cfg['burn_in'],color='gray',ls=':');axes[row,col].set_title(case['id']+f', m={m}');axes[row,col].set_xlabel('block');axes[row,col].set_ylabel('max residual')
    fig.savefig(out/'latent_traces.png',dpi=140);plt.close(fig)


def direct(cfg,raw,out):
    old,oraw=old_paths(cfg);ocfg=load(old/'config.json');banks={};results=[];checks=[]
    for n in [3,10]:
        spec=next(case for case in cfg['cases'] if case['n']==n);count=cfg['direct_draws'];seed=cfg['direct_seed']+n
        path=raw/f'direct_n{n}.npz'
        if path.exists():raise FileExistsError(path)
        global_mle(np.array([-.4,.4,0.]),2.)
        rng=np.random.default_rng(seed);start=time.perf_counter();base=rng.standard_t(2.,(count,n));mle=np.empty(count);fallbacks=0
        for i,e in enumerate(base):
            try:mle[i],d=global_mle(e,2.,return_diagnostics=True);fallbacks+=int(d['fallback'])
            except Exception as exc:save(out/f'direct_n{n}_failure.json',{'index':i,'data':e.tolist(),'error':repr(exc)});raise
            if time.perf_counter()-start>cfg['direct_ceiling_seconds']:raise TimeoutError('bounded direct pilot')
        seconds=time.perf_counter()-start;np.savez_compressed(path,base=base,mle=mle);banks[n]=(base,mle,seconds)
        for idx in np.unique(np.r_[np.linspace(0,count-1,8,dtype=int),np.argmax(np.max(np.abs(base),axis=1))]):checks.append(oracle_check(base[idx],spec,f'direct_n{n}',idx))
        save(out/f'direct_n{n}_provenance.json',{'draws':count,'seed':seed,'raw_path':str(path.relative_to(ROOT)),'sha256':sha(path),'generation_and_selector_seconds':seconds,'selector_fallbacks':fallbacks,'failures':0})
    for case in cfg['cases']:
        base,mle,bank_seconds=banks[case['n']];start=time.perf_counter();mu,x,w,normalizer=weight_particles(base,mle,case)
        accepted=np.random.default_rng(cfg['direct_seed']+1000+case['n']).random(len(mu))<np.exp(-.5*((mu-case['prior_mean'])/case['prior_std'])**2)
        processing=time.perf_counter()-start;_,_,(rmu,rx,rw,rnorm)=particles(case,ocfg,oraw)
        event=np.max(np.abs(rx-case['release']),axis=1)>6;p=float(rw[event].sum()/rw.sum());wn=rw/rw.sum();var=float(np.sum(wn*wn*(event-p)**2));event_eff=p*(1-p)/var
        ess=float(rw.sum()**2/(rw@rw));fraction=ess/len(rw);expected=rnorm*np.sqrt(2*np.pi)*case['prior_std'];cost=(bank_seconds+processing)/len(mu)
        a=int(accepted.sum());N=len(mu);ci=[float(beta.ppf(.025,a,N-a+1)) if a else 0.,float(beta.ppf(.975,a+1,N-a)) if a<N else 1.]
        result={'case':case,'draws':N,'bank_seconds':bank_seconds,'processing_seconds':processing,'microseconds_per_proposal':cost*1e6,'weighted_pilot_ess':float(w.sum()**2/(w@w)),'validated_bank_weight_ess_fraction':fraction,'estimated_weight_ess_per_second':fraction/cost,'validated_bank_tail6_plugin_effective_fraction':event_eff/len(rw),'estimated_tail6_effective_samples_per_second':event_eff/len(rw)/cost,'rejection_accepted':a,'rejection_acceptance':a/N,'rejection_exact_binomial_ci95':ci,'validated_expected_rejection_acceptance':expected,'estimated_independent_accepted_per_second':expected/cost,'estimated_microseconds_per_independent_accepted':cost/expected*1e6,'summary_pilot_weighted':summary(mu,w,x,case),'limitations':'one warmed throughput pilot, diagnostic SE/efficiency estimates; weighted ESS is not universal per-functional efficiency; reference uncertainty carried forward; centered/shifted use the same n10 standardized cost bank; no historical/current wall-time speedup claim'}
        results.append(result);save(out/'direct_comparison.json',results)
        np.savez_compressed(raw/(case['id']+'_direct_rejection.npz'),mu=mu[accepted],x=x[accepted])
        print('direct',case['id'],cost*1e6,expected/cost, fraction/cost,flush=True)
    save(out/'direct_oracle_checks.json',checks)
    if any('error' in r or r.get('selected_location_error',0)>cfg['mle_tolerance'] for r in checks):raise ArithmeticError('direct pilot oracle failed')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--phase',choices=['pilot','chains','analyze','direct'],required=True);a=ap.parse_args();cfg=load(a.config)
    raw,out=ROOT/cfg['raw_directory'],ROOT/cfg['evidence_directory'];raw.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    paths=['models/loc_student_selected_mle_block.py','models/loc_student.py','models/loc_student_selected_mle.py','models/loc_student_selected_mle_rattle.py','models/student_location_mle.py','models/student_mu_mh.py','scripts/validate_cheap_selected_mle_gibbs.py','scripts/validate_student_selected_mle_gibbs.py','scripts/validate_student_selected_mle_rattle.py','scripts/followup_shifted_student_reference.py']
    save(out/(a.phase+'_execution.json'),{'config_sha256':sha(a.config),'source_sha256':{p:sha(ROOT/p) for p in paths},'python':sys.version,'cpu_affinity':sorted(os.sched_getaffinity(0)),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    globals()[a.phase](cfg,raw,out)
if __name__=='__main__':main()
