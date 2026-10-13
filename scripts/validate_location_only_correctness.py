"""Fixed-budget correctness smoke checks, NOT a mixing/performance campaign.

Use --output to choose a NEW evidence directory. Existing evidence is read-only.
Each of ten cases has two methods and two dispersed, 512-iteration chains.
Student banks are verified and reused; other banks contain 4096 independent rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.optimize import brentq
from models.loc.student import selected_block as student_gibbs, selected_rattle as student_rattle
from models.loc.logistic import selected_gibbs as logistic_gibbs, selected_rattle as logistic_rattle
from models.loc.laplace import selected_gibbs as laplace_gibbs, selected_facet_hmc as laplace_hmc
from scripts.validate_student_selected_mle_gibbs import particles
from scripts.followup_shifted_student_reference import (
    summary, particle_bootstrap, weight_particles, weighted_quantiles,
    diagnostics, oracle_check, longest_run,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def convert(a):
        if isinstance(a, np.ndarray): return a.tolist()
        if isinstance(a, np.generic): return a.item()
        raise TypeError(type(a).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n')


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    raw = out / 'raw'
    raw.mkdir()
    old = ROOT / 'docs/research/evidence/student_selected_mle_gibbs_2026-10-10'
    oldcfg = json.loads((old / 'config.json').read_text())
    cases = [{**c, 'family': 'student'} for c in oldcfg['cases']]
    for family, ns in [('logistic', [3, 10]), ('laplace', [3, 4])]:
        settings = [(n, False) for n in ns] + [(n, True) for n in (ns if family == 'laplace' else [10])]
        for n, shift in settings:
            cases.append(dict(id=f'{"shifted" if shift else "centered"}_n{n}', family=family,
                              n=n, release=1.25 if shift else 0., prior_mean=-.75 if shift else 0.,
                              prior_std=.6 if shift else 2., tail_width=.75,
                              proposal_std_mu=.45, proposal_std_z=.1))
    cfg = dict(cases=cases, iterations=512, burn_in=128, chains=2,
               initial_max_quantiles=[.1, .9], bank_rows=4096, bootstrap_repetitions=64,
               seed=2026101217, chain_ceiling_seconds=120, total_ceiling_seconds=600,
               stopping_rule='No extensions or tuning. Abort on numerical failure. Short-chain discrepancies are retained, not convergence passes.')
    write(out / 'config.json', cfg)
    write(out / 'source.json', {str(p.relative_to(ROOT)): sha(p) for folder in ['models', 'kde_ref', 'diagnostics']
                              for p in (ROOT / folder).rglob('*.py')} |
          {str(Path(__file__).relative_to(ROOT)): sha(__file__)})
    write(out / 'environment.json', dict(python=sys.version, numpy=np.__version__,
                                        affinity=sorted(os.sched_getaffinity(0))))
    methods = dict(student=[('block', student_gibbs.run_gibbs), ('rattle', student_rattle.run_rattle)],
                   logistic=[('block', logistic_gibbs.run_gibbs), ('rattle', logistic_rattle.run_rattle)],
                   laplace=[('gibbs', laplace_gibbs.run_gibbs), ('facet_hmc', laplace_hmc.run_hmc)])
    bank_cache, results, oracle_rows = {}, [], []
    start = time.monotonic()
    for ci, case in enumerate(cases):
        stem = case['family'] + '_' + case['id']
        if case['family'] == 'student':
            base, mle, (mu, x, w, normalizer) = particles(case, oldcfg, ROOT / oldcfg['raw_directory'])
            if case['n'] == 3:
                prov = json.loads((old / 'fresh_n3_provenance.json').read_text())
                path = ROOT / oldcfg['raw_directory'] / 'fresh_n3.npz'
                assert sha(path) == prov['sha256']
            reference = json.loads((old / (case['id'] + '_reference.json')).read_text())
            observed = summary(mu, w, x, case)
            for key in observed:
                np.testing.assert_allclose(observed[key], reference['summary'][key], atol=1e-12, rtol=1e-12)
            reference['provenance'] = dict(path=str(old.relative_to(ROOT) / (case['id'] + '_reference.json')),
                                           sha256=sha(old / (case['id'] + '_reference.json')),
                                           status='Raw banks hash-checked; summaries recomputed; uncertainty carried forward.')
        else:
            key = (case['family'], case['n'])
            if key not in bank_cache:
                seed = cfg['seed'] + 100 * ci
                rng = np.random.default_rng(seed)
                base = getattr(rng, case['family'])(size=(cfg['bank_rows'], case['n']))
                if case['family'] == 'logistic':
                    mle = np.array([brentq(lambda t: np.tanh((e-t)/2).sum(), e.min(), e.max(),
                                           xtol=1e-13) for e in base])
                else:
                    mle = np.median(base, axis=1)
                name = f'{key[0]}_n{key[1]}_bank.npz'
                np.savez_compressed(raw / name, base=base, mle=mle)
                bank_cache[key] = base, mle, dict(path='raw/' + name, seed=seed, sha256=sha(raw / name))
            base, mle, provenance = bank_cache[key]
            mu, x, w, normalizer = weight_particles(base, mle, case)
            reference = dict(summary=summary(mu, w, x, case), draws=len(mu),
                             weight_ess=float(w.sum()**2 / np.dot(w, w)), normalizer=normalizer,
                             uncertainty=particle_bootstrap(mu, w, x, case, cfg['seed']+ci, 64),
                             provenance=provenance)
        write(out / (stem + '_reference.json'), reference)
        maximum = np.max(np.abs(x - case['release']), axis=1)
        indices = [int(np.argmin(abs(maximum - value))) for value in weighted_quantiles(maximum, w, [.1, .9])]
        for method, runner in methods[case['family']]:
            traces, telemetry, initials = [], [], []
            for chain, idx in enumerate(indices):
                name = f'{stem}_{method}_{chain}'
                seed = cfg['seed'] + 1000 + 10 * ci + chain
                params = {**case, 'initial_x': x[idx], 'initial_mu': float(mu[idx]),
                          'num_iterations_T': cfg['iterations'], 'block_length': 4 if case['n']==3 else 16,
                          'rattle_step_size': .1 if case['family']=='student' else .05,
                          'rattle_num_steps': 10, 'wall_ceiling_seconds': cfg['chain_ceiling_seconds']}
                result = runner(seed, case['release'], params, False)
                mus, xs = result.pop('mu_chain'), result.pop('x_chain')
                np.savez_compressed(raw / (name + '.npz'), mu=mus, x=xs)
                result.update(seed=seed, reference_index=idx, raw_sha256=sha(raw / (name + '.npz')))
                write(out / (name + '.json'), result)
                traces.append((mus[129:], xs[129:]))
                telemetry.append(result)
                initials.append(dict(index=idx, x=x[idx], mu=mu[idx], max_residual=maximum[idx]))
                if case['family'] == 'student':
                    check = oracle_check(xs[-1], case, name, len(xs)-1, normalized=True)
                    oracle_rows.append(check)
                    write(out / 'oracle_checks.json', oracle_rows)
                    if 'error' in check or check['released_mle_residual']>1e-9:
                        raise ArithmeticError('retained Student endpoint failed independent oracle')
                if time.monotonic()-start > cfg['total_ceiling_seconds']:
                    raise TimeoutError('total smoke ceiling')
            mus = np.array([a[0] for a in traces]); xs = np.array([a[1] for a in traces])
            maxima = np.max(np.abs(xs-case['release']), axis=2)
            diag = dict(mu=diagnostics(mus), latent_max=diagnostics(maxima),
                        latent_tail6=diagnostics((maxima>6).astype(float)))
            row = dict(case=stem, method=method, reference=reference['summary'],
                       summary=summary(mus.ravel(), np.ones(mus.size), xs.reshape(-1,case['n']),case),
                       per_chain=[summary(a, np.ones(len(a)), b, case) for a,b in traces],
                       diagnostics=diag, initials=initials,
                       longest_tail6_run=[longest_run(a>6) for a in maxima],
                       telemetry=telemetry, numerical_failures=0,
                       interpretation='Smoke only. No statistical acceptance threshold or convergence claim.')
            results.append(row)
            write(out / 'comparisons.json', results)
            print(stem, method, 'mean', round(row['summary']['mean'], 4),
                  'tail6', round(row['summary']['latent_tail6'], 4), 'Rhat', diag['mu']['rhat'], flush=True)
    write(out / 'completion.json', dict(seconds=time.monotonic()-start, chains=40, completed=True))
    plot(out, results)


def plot(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), constrained_layout=True)
    names = [r['case']+' / '+r['method'] for r in rows]
    for ax, metric in zip(axes, ['mean', 'latent_tail6']):
        for i, row in enumerate(rows):
            ax.plot(row['reference'][metric], i, 'kx', ms=7)
            ax.plot(row['summary'][metric], i, 'o', color='tab:blue', ms=4)
            ax.plot([c[metric] for c in row['per_chain']], [i,i], '|', color='tab:orange', ms=7)
        ax.set_yticks(range(len(rows)), names if ax is axes[0] else ['']*len(rows))
        ax.invert_yaxis(); ax.set_xlabel(metric)
    axes[0].set_title('x: weighted reference; blue: pooled smoke; orange: each chain')
    axes[1].set_title('512 iterations per chain: discrepancies are expected')
    fig.savefig(out / 'smoke_comparison.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new evidence directory: '+str(args.output))
    try:
        run(args.output)
    except Exception as exc:
        # Save partial chains when available, without discarding the error.
        if args.output.exists():
            if hasattr(exc, 'partial'):
                np.savez_compressed(args.output/'failure_partial.npz', **exc.partial)
            write(args.output/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                                                  iteration=getattr(exc, 'iteration', None)))
        raise
