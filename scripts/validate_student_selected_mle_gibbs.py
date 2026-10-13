"""Frozen three-case selected-MLE pair-Gibbs validation; no production rewrites."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[name] = '1'
import numpy as np
from models.loc.student import score_gibbs as loc_student
from models.loc.student import selected_gibbs as selected
from scripts.followup_shifted_student_reference import (
    load, save, summary, particle_bootstrap, weighted_quantiles, weight_particles,
    distances, diagnostics, oracle_check, generate_bank, run_affine_chain, longest_run,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def particles(case, cfg, raw):
    if case['n'] == 3:
        a = np.load(raw/'fresh_n3.npz')
        base, mle = a['base'], a['mle']
    else:
        old = ROOT/cfg['validated_shifted_evidence']
        execution = load(old/'references_execution.json')
        from diagnostics.source_provenance import verify_recorded_sources
        verify_recorded_sources(ROOT, execution['source_sha256'], ['models/loc/student/mle.py'])
        bases, mles = [], []
        for stem in ['weighted_a', 'weighted_b']:
            provenance = load(old/(stem+'_provenance.json'))
            path = ROOT/provenance['raw_path']
            if sha(path) != provenance['sha256']:
                raise ArithmeticError('validated raw bank hash mismatch')
            a = np.load(path); bases.append(a['base']); mles.append(a['mle'])
        base, mle = np.concatenate(bases), np.concatenate(mles)
    return base, mle, weight_particles(base, mle, case)


def initial_states(case, cfg, raw):
    _, _, (mu, x, w, _) = particles(case, cfg, raw)
    maximum = np.max(np.abs(x-case['release']), axis=1)
    result = []
    for q in cfg['initial_max_quantiles']:
        value = weighted_quantiles(maximum, w, [q])[0]
        idx = int(np.argmin(np.abs(maximum-value)))
        result.append((x[idx], float(mu[idx]), {'reference_index': idx, 'max_quantile': q}))
    e = np.r_[-12., np.zeros(case['n']-1)]
    x = case['release']+e-selected.global_mle(e, case['k'])
    result.append((x, case['prior_mean'], {'type': 'negative_outlier12'}))
    return result


def references(cfg, raw, out):
    case = cfg['cases'][0]
    base, mle = generate_bank(cfg['fresh_n3_bank'], {'case': case, 'bank_ceiling_seconds': 120.}, raw, out)
    provenance = []
    checks = []
    for case in cfg['cases']:
        base, mle, (mu, x, w, normalizer) = particles(case, cfg, raw)
        estimate = summary(mu, w, x, case)
        if case['id'] == 'shifted_n10':
            saved = load(ROOT/cfg['validated_shifted_evidence']/'weighted_reference.json')
            for metric in estimate:
                np.testing.assert_allclose(estimate[metric], saved['summary'][metric], atol=1e-12, rtol=1e-12)
            uncertainty = saved['uncertainty']
            provenance.append({'case': case['id'], 'source': str(Path(cfg['validated_shifted_evidence'])/'weighted_reference.json'),
                               'source_sha256': sha(ROOT/cfg['validated_shifted_evidence']/'weighted_reference.json'),
                               'uncertainty': 'carried forward unchanged; raw hashes and every summary checked'})
        else:
            uncertainty = particle_bootstrap(mu, w, x, case, cfg['bootstrap_seed']+case['n'], cfg['bootstrap_repetitions'])
            provenance.append({'case': case['id'], 'uncertainty': 'new whole-particle bootstrap',
                               'base_source': 'fresh_n3.npz' if case['n']==3 else 'validated shifted standardized banks, reweighted for centered prior'})
        save(out/(case['id']+'_reference.json'), {'summary': estimate, 'uncertainty': uncertainty,
             'draws': len(mu), 'weight_ess': float(w.sum()**2/(w@w)), 'normalizer': normalizer})
        indices = np.unique(np.r_[np.linspace(0, len(x)-1, cfg['reference_oracle_checks'], dtype=int), np.argmax(w), np.argmax(np.max(np.abs(x-case['release']), axis=1))])
        for idx in indices:
            checks.append(oracle_check(x[idx], case, case['id']+'_reference', idx, normalized=True))
            if case['n'] == 3:
                checks.append(oracle_check(base[idx], case, 'fresh_n3_base', idx))
        print('reference', case['id'], len(mu), flush=True)
    save(out/'reference_provenance.json', provenance)
    save(out/'reference_oracle_checks.json', checks)
    if any('error' in r or r.get('released_mle_residual', 0)>1e-10 for r in checks):
        raise ArithmeticError('independent reference oracle check failed; see saved records')


def chains(cfg, raw, out):
    checks = []
    for case in cfg['cases']:
        for c, (x, mu, origin) in enumerate(initial_states(case, cfg, raw)):
            stem = case['id']+f'_chain{c}'
            path = raw/(stem+'.npz')
            if path.exists(): raise FileExistsError(path)
            check = oracle_check(x, case, stem+'_initializer', 0, normalized=True)
            checks.append(check); save(out/'initializer_oracle_checks.json', checks)
            if 'error' in check or check['released_mle_residual']>1e-10:
                raise ArithmeticError('initializer failed independent oracle check')
            params = {**case, 'num_iterations_T': cfg['iterations'], 'initial_x': x,
                      'initial_mu': mu, 'wall_ceiling_seconds': cfg['chain_ceiling_seconds'],
                      'score_tolerance': cfg['score_tolerance'], 'mle_tolerance': cfg['mle_tolerance']}
            try:
                result = selected.run_gibbs(case['chain_seed']+c, case['release'], params, verbose=True)
            except Exception as exc:
                import traceback
                save(out/(stem+'_failure.json'), {'error': repr(exc), 'traceback': traceback.format_exc(),
                     'seed': case['chain_seed']+c, 'initial_x': x.tolist(), 'initial_mu': mu})
                raise
            mus, xs = result.pop('mu_chain'), result.pop('x_chain')
            result['x_final'] = result['x_final'].tolist()
            np.savez_compressed(path, mu=mus, x=xs)
            save(out/(stem+'_provenance.json'), {**result, 'initial_x': x.tolist(), 'initial_mu': mu,
                 'origin': origin, 'raw_path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'failures': 0})
            print('complete', stem, result['seconds'], flush=True)


def analyze(cfg, raw, out):
    rows, comparisons, oracle = [], [], []
    for case in cfg['cases']:
        ref = load(out/(case['id']+'_reference.json'))
        _, _, (rmu, rx, rw, _) = particles(case, cfg, raw)
        count = len(initial_states(case, cfg, raw))
        traces = [np.load(raw/(case['id']+f'_chain{i}.npz')) for i in range(count)]
        mu = np.array([a['mu'][cfg['burn_in']+1:] for a in traces])
        x = np.array([a['x'][cfg['burn_in']+1:] for a in traces])
        maximum = np.max(np.abs(x-case['release']), axis=2)
        full = summary(mu.ravel(), np.ones(mu.size), x.reshape(-1, case['n']), case)
        chain_rows, batches, splits, movement = [], [], [], []
        for i in range(count):
            chain_rows.append(summary(mu[i], np.ones(len(mu[i])), x[i], case))
            half = len(mu[i])//2
            splits.append([summary(mu[i, s], np.ones(half), x[i, s], case) for s in [slice(0, half), slice(-half, None)]])
            for lo in range(0, len(mu[i]), cfg['batch_length']):
                hi = lo+cfg['batch_length']
                batches.append(summary(mu[i, lo:hi], np.ones(hi-lo), x[i, lo:hi], case))
            event = maximum[i]>6
            movement.append({'tail6_entries': int(np.sum(~event[:-1]&event[1:])),
                             'tail6_exits': int(np.sum(event[:-1]&~event[1:])),
                             'longest_tail6_run': longest_run(event), 'max_residual': float(maximum[i].max())})
            indices = np.unique(np.r_[np.linspace(0, len(mu[i])-1, cfg['retained_oracle_checks'], dtype=int), np.argmax(maximum[i])])
            for idx in indices:
                oracle.append(oracle_check(x[i, idx], case, case['id']+f'_chain{i}', idx, normalized=True))
        se = {key: max(float(np.std([r[key] for r in batches], ddof=1)/np.sqrt(len(batches))),
                       float(np.std([r[key] for r in chain_rows], ddof=1)/np.sqrt(count))) for key in full}
        # The shifted mu>2 tail is too rare for this budget: no normal screening.
        if case['id'] == 'shifted_n10': se['upper_tail'] = None
        diag = {'mu': diagnostics(mu), 'latent_max': diagnostics(maximum), 'latent_tail6': diagnostics((maximum>6).astype(float))}
        z = {}
        for key in full:
            rs, cs = ref['uncertainty'][key]['se'], se[key]
            z[key] = (full[key]-ref['summary'][key])/np.hypot(rs, cs) if rs is not None and cs is not None and np.hypot(rs, cs)>0 else None
            rows.append({'case': case['id'], 'metric': key, 'reference': ref['summary'][key], 'reference_mcse': rs,
                         'gibbs': full[key], 'gibbs_mcse_diagnostic': cs, 'difference_mcse': z[key]})
        screens = {name: all(v is not None and v<=cfg['rhat_max'] for v in [a['rhat']]) and
                    all(v is not None and v>=cfg['ess_min'] for v in [a['ess_bulk'], a['ess_tail']]) for name, a in diag.items()}
        screens['summary_comparison'] = all(abs(v)<=cfg['difference_mcse_max'] for key,v in z.items() if v is not None)
        result = {'case': case, 'summary': full, 'mcse_diagnostic': se, 'per_chain': chain_rows,
                  'split_summaries': splits, 'movement': movement, 'diagnostics': diag, 'screens': screens,
                  'passes_declared_screens': all(screens.values()), 'difference_mcse': z,
                  'retained': int(mu.size), 'rare_upper_events': int(np.sum(mu>case['release']+case['tail_width'])),
                  'parameter_distances': distances(mu.ravel(), rmu, rw),
                  'latent_max_distances': distances(maximum.ravel(), np.max(np.abs(rx-case['release']), axis=1), rw)}
        comparisons.append(result); save(out/'comparisons.json', comparisons)
        print('analyze', case['id'], diag, screens, flush=True)
    save(out/'retained_oracle_checks.json', oracle)
    with (out/'summary.csv').open('w', newline='') as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if any('error' in r or r.get('released_mle_residual', 0)>1e-10 for r in oracle):
        raise ArithmeticError('retained oracle verification failed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for r, case in enumerate(cfg['cases']):
        for i in range(len(cfg['initial_max_quantiles'])+1):
            a = np.load(raw/(case['id']+f'_chain{i}.npz'))
            stride = max(1, len(a['mu'])//1000)
            axes[r,0].plot(np.arange(len(a['mu']))[::stride], a['mu'][::stride], lw=.6, label=str(i))
            axes[r,1].plot(np.arange(len(a['mu']))[::stride], np.max(np.abs(a['x'][::stride]-case['release']), axis=1), lw=.6)
        axes[r,0].set_ylabel(case['id']+' mu'); axes[r,1].set_ylabel('max residual'); axes[r,1].set_yscale('log')
        for ax in axes[r]: ax.axvline(cfg['burn_in'], color='gray', ls=':'); ax.set_xlabel('sweep')
    axes[0,0].legend(title='chain', ncol=4); fig.savefig(out/'traces.png', dpi=150); plt.close(fig)


def benchmark(cfg, raw, out):
    rows = []
    for case in cfg['cases']:
        x, mu, _ = initial_states(case, cfg, raw)[0]
        p = {**case, 'initial_x': x, 'initial_mu': mu, 'store_x_chain': True,
             'num_iterations_T': cfg['benchmark_sweeps'], 'initialization': 'random', 'initialization_seed': 771}
        def run(name, steps):
            pp = {**p, 'num_iterations_T': steps}
            if name == 'selected': return selected.run_gibbs(992, case['release'], pp, False)
            if name == 'legacy': return loc_student.run_gibbs(np.array([0, 992], dtype=np.uint32), case['release'], pp, False)
            return run_affine_chain(x, mu, 992, {**case, 'mle_tolerance': 1e-8}, steps)[1]
        for name in ['legacy', 'affine', 'selected']: run(name, 5)
        timings = {name: [] for name in ['legacy', 'affine', 'selected']}
        for rep in range(3):
            for name in (['legacy', 'affine', 'selected'] if rep%2 == 0 else ['selected', 'affine', 'legacy']):
                start = time.perf_counter(); run(name, cfg['benchmark_sweeps']); timings[name].append(time.perf_counter()-start)
        rows.append({'case': case['id'], 'sweeps': cfg['benchmark_sweeps'], 'seconds': timings,
                     'median_microseconds_per_sweep': {name: float(np.median(v)/cfg['benchmark_sweeps']*1e6) for name,v in timings.items()},
                     'note': 'different targets for legacy; pair sweep has floor(n/2) proposals versus one affine dataset proposal; warmed preliminary throughput only'})
    save(out/'benchmark.json', rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=['references','chains','analyze','benchmark'], required=True)
    args = parser.parse_args(); cfg = load(args.config)
    raw, out = ROOT/cfg['raw_directory'], ROOT/cfg['evidence_directory']
    raw.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    save(out/(args.phase+'_execution.json'), {'config_sha256': sha(args.config), 'python': sys.version,
         'source_sha256': {p: sha(ROOT/p) for p in ['models/loc/student/selected_gibbs.py','models/loc/student/score_gibbs.py','models/loc/student/mle.py','models/loc/student/mu_mh.py','models/loc/affine.py','scripts/followup_shifted_student_reference.py','scripts/validate_student_selected_mle_gibbs.py']},
         'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'cpu_affinity': sorted(os.sched_getaffinity(0))})
    globals()[args.phase](cfg, raw, out)


if __name__ == '__main__': main()
