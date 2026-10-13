"""One-case, bounded selected-MLE reference/exploration study.

Calls production affine updates unchanged; permits dispersed starts only in this
research driver. Raw banks/chains live in ignored tmp; compact evidence is JSON.
Run with --config PATH --phase references|chains|analyze.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
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
from models import loc_affine
from models.student_location_mle import global_mle, score, loglik
from scripts.selected_mle_reference_oracle import selected_mle_oracle
from scripts.validate_selected_mle_reference import (
    save, load, weighted_quantiles, weight_particles, summarize,
    split_rhat, distances,
)


def summary(mu, w, x, case):
    result = summarize(mu, w, x, case)
    maximum = np.max(np.abs(x-case['release']), axis=1)
    q = weighted_quantiles(maximum, w, [.5, .9, .95])
    result.update(max_q50=float(q[0]), max_q90=float(q[1]), max_q95=float(q[2]))
    return result


def particle_bootstrap(mu, w, x, case, seed, repetitions):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        idx = rng.integers(0, len(mu), len(mu))
        draws.append(summary(mu[idx], w[idx], x[idx], case))
    return {
        key: {'se': float(np.std([r[key] for r in draws], ddof=1)),
              'percentile95': np.quantile([r[key] for r in draws], [.025, .975]).tolist()}
        for key in draws[0]
    }


def event_evidence(events):
    """Counts and an exact binomial CI, for independent rejection draws ONLY."""
    n, count = len(events), int(np.sum(events))
    if n == 0:
        return {'n': n, 'events': count, 'interval95': None}
    lo = 0. if count == 0 else float(stats.beta.ppf(.025, count, n-count+1))
    hi = 1. if count == n else float(stats.beta.ppf(.975, count+1, n-count))
    return {'n': n, 'events': count, 'interval95': [lo, hi]}


def run_affine_chain(x0, mu0, seed, case, iterations, *, ceiling=math.inf, progress=False):
    """Same update order/RNG calls as production, with a supplied valid dataset."""
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float, copy=True)
    selected = loc_affine.global_location_mle(x, 'student', k=case['k'])
    tolerance = case['mle_tolerance']*(1+abs(case['release']))
    if abs(selected-case['release']) > tolerance:
        raise ValueError('initial dataset does not have the released selected MLE')
    mu = float(mu0)
    mus = np.empty(iterations+1)
    xs = np.empty((iterations+1, case['n']))
    latent_accept = np.zeros(iterations, dtype=bool)
    mu_accept = np.zeros(iterations, dtype=bool)
    mus[0], xs[0] = mu, x
    max_proposal = max_retained = abs(selected-case['release'])
    start = time.monotonic()
    for i in range(iterations):
        x, latent_accept[i], diag = loc_affine.latent_affine_mh_step(
            x, mu, case['release'], 'student', case['tau_q'], rng,
            k=case['k'], mle_tol=case['mle_tolerance'], return_diagnostics=True)
        max_proposal = max(max_proposal, diag['proposal_mle_residual'])
        retained = loc_affine.global_location_mle(x, 'student', k=case['k'])
        max_retained = max(max_retained, abs(retained-case['release']))
        mu, mu_accept[i] = loc_affine._update_mu_mh(
            mu, x, 'student', case['proposal_std_mu'], case['prior_mean'],
            case['prior_std'], rng, k=case['k'])
        mus[i+1], xs[i+1] = mu, x
        if (i+1) % 5000 == 0 and progress:
            print('chain progress', seed, i+1, round(time.monotonic()-start, 2), flush=True)
        if time.monotonic()-start > ceiling:
            raise TimeoutError(f'chain {seed} exceeded fixed wall ceiling at iteration {i+1}')
    return {'mu': mus, 'x': xs, 'latent_accept': latent_accept, 'mu_accept': mu_accept}, {
        'seconds': time.monotonic()-start,
        'latent_acceptance': float(latent_accept.mean()), 'mu_acceptance': float(mu_accept.mean()),
        'max_proposal_mle_residual': max_proposal, 'max_retained_mle_residual': max_retained,
        'seed': seed, 'initial_mu': mu0, 'initial_x': np.asarray(x0).tolist(),
        'initial_max_residual': float(np.max(np.abs(x0-case['release']))), 'iterations': iterations,
    }


def oracle_check(data, case, label, index, *, normalized=False, ceiling=10.):
    row = {'source': label, 'index': int(index), 'data': np.asarray(data).tolist()}
    start = time.monotonic()
    def alarm(signum, frame):
        raise TimeoutError('independent oracle time ceiling')
    old = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, ceiling)
    try:
        oracle = selected_mle_oracle(data, case['k'], fast=True)
        selected = global_mle(data, case['k'])
        row.update(oracle=oracle, production_selected=selected,
                   selected_location_error=abs(selected-oracle['selected']),
                   loglik_gap=max(0., oracle['loglik']-loglik(data, selected, case['k'])),
                   score_residual=abs(score(data, selected, case['k'])))
        if normalized:
            row['released_mle_residual'] = abs(oracle['selected']-case['release'])
    except Exception as exc:
        row['error'] = repr(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, old)
    row['seconds'] = time.monotonic()-start
    return row


def generate_bank(spec, cfg, raw, out):
    path = raw/(spec['id']+'.npz')
    if path.exists():
        raise FileExistsError('no cache reuse: '+str(path))
    rng = np.random.default_rng(spec['seed'])
    base = rng.standard_t(cfg['case']['k'], (spec['draws'], cfg['case']['n']))
    mle = np.empty(spec['draws'])
    methods, fallback_reasons = Counter(), Counter()
    failures = []
    start = time.monotonic()
    for i, data in enumerate(base):
        try:
            mle[i], diag = global_mle(data, cfg['case']['k'], return_diagnostics=True)
            methods[diag['method']] += 1
            if diag['fallback']:
                fallback_reasons[str(diag.get('fallback_reason'))] += 1
        except Exception as exc:
            failures.append({'index': i, 'data': data.tolist(), 'error': repr(exc)})
            save(out/(spec['id']+'_failures.json'), failures)
            # Abort, rather than dropping or replacing the failed draw.
            raise
        if (i+1) % 16384 == 0:
            print('bank', spec['id'], i+1, round(time.monotonic()-start, 2), flush=True)
        if time.monotonic()-start > cfg['bank_ceiling_seconds']:
            save(out/(spec['id']+'_failures.json'), [{'timeout_after_draws': i+1}])
            raise TimeoutError('fixed bank ceiling')
    np.savez_compressed(path, base=base, mle=mle)
    meta = {**spec, 'seconds': time.monotonic()-start, 'failures': failures,
            'methods': dict(methods), 'fallback_reasons': dict(fallback_reasons),
            'raw_path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    save(out/(spec['id']+'_provenance.json'), meta)
    return base, mle


def references(cfg, raw, out):
    checks = []
    for spec in cfg['weighted_banks']+[cfg['rejection_bank']]:
        base, mle = generate_bank(spec, cfg, raw, out)
        mu, x, w, normalizer = weight_particles(base, mle, cfg['case'])
        start = time.monotonic()
        residual = 0.
        for i, state in enumerate(x):
            try:
                selected = global_mle(state, cfg['case']['k'])
                residual = max(residual, abs(selected-cfg['case']['release']))
                if abs(selected-cfg['case']['release']) > cfg['case']['mle_tolerance']:
                    raise ArithmeticError('normalized particle has wrong selected MLE')
            except Exception as exc:
                save(out/(spec['id']+'_normalization_failure.json'), {'index': i, 'data': state.tolist(), 'error': repr(exc)})
                raise
            if time.monotonic()-start > cfg['bank_ceiling_seconds']:
                raise TimeoutError('fixed normalization-check ceiling')
        save(out/(spec['id']+'_normalization.json'), {'checked': len(x), 'max_selected_mle_residual': residual,
                                                   'failures': 0, 'seconds': time.monotonic()-start})
        if spec == cfg['rejection_bank']:
            rng = np.random.default_rng(cfg['rejection_accept_seed'])
            accepted = rng.random(len(mu)) < np.exp(-.5*((mu-cfg['case']['prior_mean'])/cfg['case']['prior_std'])**2)
            np.savez_compressed(raw/'rejection_accepted.npz', mu=mu[accepted], x=x[accepted])
            save(out/'rejection_acceptance.json', {
                'proposals': len(mu), 'accepted': int(accepted.sum()),
                'rate': float(accepted.mean()),
                'expected_rate_weighted': normalizer*math.sqrt(2*math.pi)*cfg['case']['prior_std']})
        else:
            save(out/(spec['id']+'_summary.json'), {
                'summary': summary(mu, w, x, cfg['case']),
                'uncertainty': particle_bootstrap(mu, w, x, cfg['case'], spec['seed']+100,
                                                  cfg['bootstrap_bank_repetitions']),
                'weight_ess': float(w.sum()**2/(w@w)), 'normalizer': normalizer,
                'maximum_normalized_weight': float(w.max()/w.sum())})
        # Fixed-spaced checks plus highest weight, extreme MLE and largest residual.
        indices = np.unique(np.r_[np.linspace(0, len(base)-1, cfg['oracle_base_checks_per_bank'], dtype=int),
                                  np.argmax(w), np.argmax(np.abs(mle)), np.argmax(np.max(np.abs(x-cfg['case']['release']), axis=1))])
        for i in indices:
            checks.append(oracle_check(base[i], cfg['case'], spec['id'], i))
        latent_indices = np.unique(np.r_[indices[:cfg['oracle_reference_latent_checks']], np.argmax(np.max(np.abs(x-cfg['case']['release']), axis=1))])
        for i in latent_indices:
            checks.append(oracle_check(x[i], cfg['case'], spec['id']+'_normalized', i, normalized=True))
        save(out/'reference_oracle_checks.json', checks)
        print('reference complete', spec['id'], flush=True)


def initial_states(cfg, raw):
    case = cfg['case']
    bank = np.load(raw/(cfg['weighted_banks'][0]['id']+'.npz'))
    mu, x, w, _ = weight_particles(bank['base'], bank['mle'], case)
    maximum = np.max(np.abs(x-case['release']), axis=1)
    states = []
    for q in cfg['initial_reference_max_quantiles']:
        target = float(weighted_quantiles(maximum, w, [q])[0])
        index = int(np.argmin(np.abs(maximum-target)))
        states.append((f'reference_max_q{q}', x[index], float(mu[index]), {'bank_index': index}))
    for name, e, mu0 in cfg['initial_deterministic']:
        e = np.asarray(e, dtype=float)
        normalized = case['release']+e-global_mle(e, case['k'])
        states.append((name, normalized, mu0, {'original_data': e.tolist()}))
    return states


def chains(cfg, raw, out):
    checks = []
    for c, (name, x, mu, origin) in enumerate(initial_states(cfg, raw)):
        checks.append(oracle_check(x, cfg['case'], 'initializer_'+name, c, normalized=True))
        save(out/'initializer_oracle_checks.json', checks)
        if 'error' in checks[-1] or checks[-1]['released_mle_residual'] > 1e-8:
            raise ArithmeticError('independent initializer check failed')
        path = raw/f'chain{c}.npz'
        if path.exists():
            raise FileExistsError(path)
        try:
            arrays, meta = run_affine_chain(x, mu, cfg['chain_seed']+c, cfg['case'], cfg['iterations'],
                                            ceiling=cfg['chain_ceiling_seconds'], progress=True)
        except Exception as exc:
            save(out/f'chain{c}_failure.json', {'error': repr(exc), 'initializer': name, 'origin': origin})
            raise
        np.savez_compressed(path, **arrays)
        save(out/f'chain{c}_provenance.json', {**meta, 'initializer': name, 'origin': origin,
                'failures': 0, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'raw_path': str(path.relative_to(ROOT))})
        print('chain complete', c, name, meta['seconds'], flush=True)


def autocorrelation_ess(chains):
    """Split-chain variance estimator and Geyer initial-positive monotone pairs.

    Diagnostic estimate, not a guarantee, especially for chains failing R-hat.
    """
    half = chains.shape[1]//2
    a = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0).astype(float)
    m, n = a.shape
    within = float(np.var(a, axis=1, ddof=1).mean())
    if within == 0:
        return None
    var_plus = (n-1)/n*within + float(np.var(a.mean(axis=1), ddof=1))
    z = a-a.mean(axis=1, keepdims=True)
    fft_size = 1 << (2*n-1).bit_length()
    fft = np.fft.rfft(z, n=fft_size, axis=1)
    cov = np.fft.irfft(fft*np.conj(fft), n=fft_size, axis=1)[:, :n]/n
    rho = 1-(within-cov.mean(axis=0))/var_plus
    rho[0] = 1.
    pairs = rho[:n//2*2].reshape(-1, 2).sum(axis=1)
    stop = np.flatnonzero(pairs <= 0)
    pairs = pairs[:stop[0]] if len(stop) else pairs
    if not len(pairs):
        return float(m*n)
    pairs = np.minimum.accumulate(pairs)
    tau = max(1., -1+2*float(pairs.sum()))
    return float(min(m*n, m*n/tau))


def diagnostics(chains):
    if np.all(chains == chains.flat[0]):
        return {'rhat': None, 'ess_bulk': None, 'ess_tail': None, 'ess_raw': None,
                'status': 'constant trace; event uncertainty unresolved'}
    # Existing split/folded rank R-hat; bulk ESS uses split-chain rank transform.
    half = chains.shape[1]//2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    ranks = stats.rankdata(split.ravel())
    ranked = stats.norm.ppf((ranks-.375)/(len(ranks)+.25)).reshape(split.shape)
    # Undo split for ESS routine, which performs its own split.
    ranked = np.concatenate([ranked[:len(chains)], ranked[len(chains):]], axis=1)
    qlo, qhi = np.quantile(chains, [.05, .95])
    tail_values = [autocorrelation_ess(chains <= qlo), autocorrelation_ess(chains >= qhi)]
    available = [v for v in tail_values if v is not None]
    # Existing folded R-hat may be undefined for binary folds: use raw binary R-hat.
    if len(np.unique(chains)) == 2:
        within = float(np.var(split, axis=1, ddof=1).mean())
        rhat = float(np.sqrt(((half-1)/half*within+np.var(split.mean(axis=1), ddof=1))/within)) if within else None
    else:
        rhat = split_rhat(chains)
    return {'rhat': rhat, 'ess_bulk': autocorrelation_ess(ranked),
            'ess_tail': min(available) if available else None,
            'ess_raw': autocorrelation_ess(chains)}


def longest_run(mask):
    if not len(mask) or not np.any(mask):
        return 0
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return int(np.max(np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)))


def analyze(cfg, raw, out):
    case = cfg['case']
    bases, mles = [], []
    for spec in cfg['weighted_banks']:
        a = np.load(raw/(spec['id']+'.npz'))
        bases.append(a['base']); mles.append(a['mle'])
    mu, x, w, normalizer = weight_particles(np.concatenate(bases), np.concatenate(mles), case)
    ref = summary(mu, w, x, case)
    uncertainty = particle_bootstrap(mu, w, x, case, cfg['bootstrap_seed'], cfg['bootstrap_repetitions'])
    tail6 = np.max(np.abs(x-case['release']), axis=1) > 6
    # Event-specific influence variance avoids treating total importance ESS as universal.
    influence = {key: float(np.sqrt(np.sum(w*w*(h-float(np.average(h, weights=w)))**2))/w.sum())
                 for key, h in [('mean', mu), ('latent_tail6', tail6.astype(float)), ('upper_tail', (mu>2).astype(float))]}
    reference = {'draws': len(mu), 'summary': ref, 'uncertainty': uncertainty,
                 'influence_se': influence, 'weight_ess': float(w.sum()**2/(w@w)),
                 'normalizer': normalizer, 'maximum_normalized_weight': float(w.max()/w.sum()),
                 'rare_mu_upper_base_events': int(np.sum(mu>2)),
                 'latent_tail6_base_events': int(tail6.sum())}
    save(out/'weighted_reference.json', reference)
    rejection = np.load(raw/'rejection_accepted.npz')
    rmu, rx = rejection['mu'], rejection['x']
    rejection_summary = summary(rmu, np.ones(len(rmu)), rx, case)
    rb = particle_bootstrap(rmu, np.ones(len(rmu)), rx, case, cfg['bootstrap_seed']+1, cfg['bootstrap_repetitions'])
    rej = {'summary': rejection_summary, 'uncertainty': rb,
           'latent_tail6': event_evidence(np.max(np.abs(rx-case['release']), axis=1)>6),
           'rare_mu_upper': event_evidence(rmu>2), **distances(rmu, mu, w)}
    save(out/'independent_rejection_comparison.json', rej)
    count = len(initial_states(cfg, raw))
    traces = [np.load(raw/f'chain{i}.npz') for i in range(count)]
    cmu = np.array([a['mu'][cfg['burn_in']+1:] for a in traces])
    cx = np.array([a['x'][cfg['burn_in']+1:] for a in traces])
    cmax = np.max(np.abs(cx-case['release']), axis=2)
    pooled_mu, pooled_x = cmu.ravel(), cx.reshape(-1, case['n'])
    full = summary(pooled_mu, np.ones(len(pooled_mu)), pooled_x, case)
    per_chain, split_summaries, batches, movements = [], [], [], []
    checks = []
    for i in range(count):
        per_chain.append(summary(cmu[i], np.ones(len(cmu[i])), cx[i], case))
        h = len(cmu[i])//2
        split_summaries.append([summary(cmu[i, s], np.ones(h), cx[i, s], case) for s in [slice(0, h), slice(-h, None)]])
        for lo in range(0, len(cmu[i])-cfg['batch_length']+1, cfg['batch_length']):
            hi = lo+cfg['batch_length']
            batches.append(summary(cmu[i, lo:hi], np.ones(hi-lo), cx[i, lo:hi], case))
        event = cmax[i]>6
        identity = np.argmax(np.abs(cx[i]-case['release']), axis=1)
        signed = (cx[i]-case['release'])[np.arange(len(event)), identity]
        movements.append({'tail6_events': int(event.sum()), 'tail6_entries': int(np.sum(~event[:-1]&event[1:])),
                          'tail6_exits': int(np.sum(event[:-1]&~event[1:])),
                          'longest_tail6_run': longest_run(event), 'longest_no_tail6_run': longest_run(~event),
                          'max_residual_seen': float(cmax[i].max()),
                          'largest_residual_identity_changes': int(np.sum(identity[1:] != identity[:-1])),
                          'tail6_positive_largest_fraction': float(np.mean((signed>6))),
                          'rare_mu_upper_entries': int(np.sum((cmu[i, :-1]<=2)&(cmu[i, 1:]>2))),
                          'rare_mu_upper_longest_run': longest_run(cmu[i]>2)})
        indices = np.unique(np.r_[np.linspace(0, len(cmu[i])-1, cfg['oracle_chain_checks'], dtype=int), np.argmax(cmax[i])])
        for j in indices:
            checks.append(oracle_check(cx[i, j], case, f'chain{i}', j, normalized=True))
    save(out/'retained_chain_oracle_checks.json', checks)
    # Nonzero batch estimates do not establish convergence; include between-chain error.
    se = {}
    for key in full:
        batch_se = float(np.std([r[key] for r in batches], ddof=1)/math.sqrt(len(batches)))
        between_se = float(np.std([r[key] for r in per_chain], ddof=1)/math.sqrt(count))
        se[key] = max(batch_se, between_se)
    if np.sum(cmu>2) == 0:
        se['upper_tail'] = None
    affine = {'summary': full, 'mcse_diagnostic': se, 'per_chain': per_chain,
              'split_summaries': split_summaries, 'movements': movements,
              'diagnostics': {'mu': diagnostics(cmu), 'latent_max': diagnostics(cmax),
                              'latent_tail6': diagnostics((cmax>6).astype(float)),
                              'rare_mu_upper': diagnostics((cmu>2).astype(float))},
              'retained': int(cmu.size), 'rare_mu_upper_events': int(np.sum(cmu>2)),
              'expected_rare_events_if_iid': cmu.size*ref['upper_tail'],
              **distances(pooled_mu, mu, w), 'latent_max_distances': distances(cmax.ravel(), np.max(np.abs(x-case['release']), axis=1), w)}
    save(out/'affine_comparison.json', affine)
    rows = []
    for key in ref:
        rs, cs = uncertainty[key]['se'], se[key]
        rows.append({'metric': key, 'weighted': ref[key], 'weighted_bootstrap_se': rs,
                     'rejection': rejection_summary[key], 'rejection_bootstrap_se': rb[key]['se'],
                     'affine': full[key], 'affine_mcse_diagnostic': cs,
                     'affine_weighted_difference_mcse': (full[key]-ref[key])/math.hypot(rs, cs) if cs is not None and math.hypot(rs, cs)>0 else None})
    with (out/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Compact scientific plot: full traces plus time-window tail occupancy.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for i, a in enumerate(traces):
        maximum = np.max(np.abs(a['x']-case['release']), axis=1)
        stride = max(1, len(maximum)//2000)
        axes[0].plot(np.arange(len(maximum))[::stride], maximum[::stride], lw=.6, label=f'chain {i}')
        axes[1].plot(np.arange(len(maximum))[::stride], a['mu'][::stride], lw=.6)
        batch = cfg['batch_length']; stop = len(cmax[i])//batch*batch
        occupancy = (cmax[i, :stop].reshape(-1, batch)>6).mean(axis=1)
        axes[2].plot(cfg['burn_in']+batch*(np.arange(len(occupancy))+.5), occupancy, lw=1.)
    axes[0].set_yscale('log'); axes[0].set_ylabel('max |X - release|'); axes[0].legend(ncol=3)
    axes[1].set_ylabel('mu'); axes[2].set_ylabel('tail > 6 occupancy'); axes[2].set_xlabel('iteration')
    axes[2].axhline(ref['latent_tail6'], color='black', ls='--', label='weighted reference')
    for ax in axes:
        ax.axvline(cfg['burn_in'], color='gray', ls=':')
    fig.savefig(out/'exploration.png', dpi=150); plt.close(fig)
    print('analysis complete', reference['weight_ess'], affine['diagnostics'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=['references', 'chains', 'analyze'], required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    raw, out = ROOT/cfg['raw_directory'], ROOT/cfg['evidence_directory']
    raw.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    sources = ['models/student_location_mle.py', 'models/loc_affine.py',
               'scripts/selected_mle_reference_oracle.py', 'scripts/validate_selected_mle_reference.py', __file__]
    fingerprint = {str(Path(p).relative_to(ROOT) if Path(p).is_absolute() else p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    save(out/(args.phase+'_execution.json'), {'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
         'source_sha256': fingerprint, 'python': sys.version, 'phase': args.phase,
         'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'sympy', 'numba', 'matplotlib']},
         'cpu_affinity': sorted(os.sched_getaffinity(0)), 'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    globals()[args.phase](cfg, raw, out)


if __name__ == '__main__':
    main()
