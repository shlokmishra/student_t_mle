"""Frozen, bounded location-only comparison. No production sampler changes.

Run --config PATH once; raw paths must be new. --analyze reads existing outputs.
Reference construction, sampling, diagnostics and independent checks are timed
separately. Existing source and evidence remain read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta
from models.loc.student import selected_block as sb, selected_rattle as sr, independence as si
from models.loc.student.mle import global_mle
from models.loc.logistic import selected_gibbs as lg, selected_rattle as lr
from models.loc.laplace import selected_gibbs as pg, selected_facet_hmc as ph
from scripts.validate_location_only_correctness import write, sha
from scripts.validate_student_selected_mle_gibbs import particles
from scripts.validate_student_independence_mh import summarize_chains
from scripts.followup_shifted_student_reference import (
    summary, weight_particles, weighted_quantiles, particle_bootstrap, oracle_check,
)

OLD = ROOT / "docs/research/evidence/student_selected_mle_gibbs_2026-10-10"
RUNNERS = {"student": {"block": sb.run_gibbs, "rattle": sr.run_rattle, "independence": si.run_independence},
           "logistic": {"block": lg.run_gibbs, "rattle": lr.run_rattle},
           "laplace": {"gibbs": pg.run_gibbs, "facet_hmc": ph.run_hmc}}


def load(path):
    return json.loads(Path(path).read_text())


def case_name(case):
    return case["family"] + "_" + case["id"]


def binomial_interval(count, n):
    if n == 0:
        return [None, None]
    return [0. if count == 0 else float(beta.ppf(.025, count, n-count+1)),
            1. if count == n else float(beta.ppf(.975, count+1, n-count))]


def weighted_function_ess(values, weights):
    """IID-equivalent precision for this particular self-normalized IS mean."""
    weights = np.asarray(weights, dtype=float)
    a = weights / weights.sum()
    mean = float(a @ values)
    variance = float(a @ (values-mean)**2)
    mc_variance = float(np.sum(a*a*(values-mean)**2))
    if variance <= 0 or mc_variance <= 0:
        return None
    return variance / mc_variance


def passes_screens(row, cfg):
    """Require resolved evidence: omitted/constant metrics cannot pass."""
    z = row["difference_mcse"]
    agreement = all(z.get(k) is not None and abs(z[k]) <= cfg["difference_mcse_max"]
                    for k in cfg["required_metrics"])
    ds = row["diagnostics"]
    mixing = all(d["rhat"] is not None and d["rhat"] <= cfg["rhat_max"] and
                 all(d[k] is not None and d[k] >= cfg["ess_min"] for k in ("ess_bulk", "ess_tail"))
                 for d in ds.values())
    counts = row["tail6_events_per_chain"]
    resolved = sum(counts) >= cfg["minimum_tail_events"] and all(c > 0 for c in counts)
    return dict(reference_agreement=agreement, mixing=mixing, tail_event_evidence=resolved,
                all_pass=agreement and mixing and resolved)


def bank(family, n, size, seed, ceiling):
    rng = np.random.default_rng(seed)
    warm = time.perf_counter()
    if family == "student":
        global_mle(np.linspace(-.3, .3, n), 2.)
    warm_seconds = time.perf_counter()-warm
    start = time.perf_counter()
    base = rng.standard_t(2., (size, n)) if family == "student" else getattr(rng, family)(size=(size, n))
    mle = np.empty(size)
    fallbacks = 0
    if family == "laplace":
        # Existing direct construction uses array order statistics, not a Python
        # loop per dataset. Preserve that inexpensive implementation.
        mle = np.median(base, axis=1)
        return base, mle, dict(seconds=time.perf_counter()-start, draws=size, seed=seed,
                              estimator_calls=size, selector_fallbacks=0, failures=0,
                              warmup_seconds=warm_seconds, extra_warmup_selector_calls=0)
    for i, e in enumerate(base):
        try:
            if family == "student":
                mle[i], diag = global_mle(e, 2., return_diagnostics=True)
                fallbacks += int(diag["fallback"])
            elif family == "logistic":
                mle[i] = brentq(lambda t: np.tanh((e-t)*.5).sum(), e.min(), e.max(), xtol=1e-13)
            else:
                mle[i] = np.median(e)
        except Exception as exc:
            exc.dataset_index = i
            exc.dataset = e.tolist()
            raise
        if i % 1024 == 0 and time.perf_counter()-start > ceiling:
            raise TimeoutError("independent bank ceiling")
    return base, mle, dict(seconds=time.perf_counter()-start, draws=size, seed=seed,
                          estimator_calls=size, selector_fallbacks=fallbacks, failures=0,
                          warmup_seconds=warm_seconds,
                          extra_warmup_selector_calls=int(family=="student"))


def prepare_references(cfg, raw, out):
    oldcfg = load(OLD / "config.json")
    refs, banks = {}, {}
    start = time.monotonic()
    for ci, case in enumerate(cfg["cases"]):
        name = case_name(case)
        if case["family"] == "student":
            base, mle, values = particles(case, oldcfg, ROOT/oldcfg["raw_directory"])
            if case["n"] == 3:
                prior = load(OLD/"fresh_n3_provenance.json")
                if sha(ROOT/prior["raw_path"]) != prior["sha256"]:
                    raise ArithmeticError("n3 reference bank mismatch")
            ref = load(OLD/(case["id"]+"_reference.json"))
            actual = summary(values[0], values[2], values[1], case)
            for metric in actual:
                np.testing.assert_allclose(actual[metric], ref["summary"][metric], atol=1e-12, rtol=1e-12)
            ref["provenance"] = dict(path=str(OLD.relative_to(ROOT)/(case["id"]+"_reference.json")),
                                     sha256=sha(OLD/(case["id"]+"_reference.json")),
                                     policy="Bank/selector hashes and summaries checked; bootstrap uncertainty carried forward.")
        else:
            key = (case["family"], case["n"])
            if key not in banks:
                base, mle, meta = bank(*key, cfg["reference_draws"], cfg["seed"]+ci,
                                       cfg["reference_ceiling_seconds"])
                path = raw/f"reference_{key[0]}_n{key[1]}.npz"
                np.savez_compressed(path, base=base, mle=mle)
                meta.update(raw_path=str(path.relative_to(ROOT)), sha256=sha(path))
                banks[key] = base, mle, meta
                print("reference bank", key, meta["seconds"], flush=True)
            base, mle, meta = banks[key]
            values = weight_particles(base, mle, case)
            mu, x, w, normalizer = values
            ref = dict(summary=summary(mu, w, x, case), draws=len(mu),
                       weight_ess=float(w.sum()**2/(w@w)), normalizer=normalizer,
                       uncertainty=particle_bootstrap(mu, w, x, case, cfg["seed"]+100+ci,
                                                      cfg["bootstrap_repetitions"]),
                       provenance=meta)
        write(out/(name+"_reference.json"), ref)
        refs[name] = values, ref
    write(out/"reference_execution.json", dict(seconds=time.monotonic()-start))
    return refs


def failure(path, exc, **extra):
    write(path, dict(error=repr(exc), traceback=traceback.format_exc(),
                     iteration=getattr(exc, "iteration", getattr(exc, "block_iteration", None)),
                     dataset_index=getattr(exc, "dataset_index", None),
                     dataset=getattr(exc, "dataset", None), **extra))


def parameters(case, cfg, x, mu, method, count):
    settings = cfg["settings"]
    p = {**case, "initial_x": x, "initial_mu": float(mu), "initial_e": x-mu,
         "num_iterations_T": count, "block_length": settings["student_logistic_block_n3"]
                             if case["n"] == 3 else settings["student_logistic_block_other"],
         "proposal_std_mu": settings["proposal_std_mu"], "proposal_std_z": settings["proposal_std_z"],
         "rattle_step_size": settings["student_rattle_step_size"] if case["family"] == "student"
                             else settings["logistic_laplace_step_size"],
         "rattle_num_steps": settings["rattle_num_steps"],
         "wall_ceiling_seconds": cfg["chain_ceiling_seconds"]}
    return p


def independent_check(x, case, label):
    if case["family"] == "student":
        return oracle_check(x, case, label, 0, normalized=True)
    if case["family"] == "logistic":
        selected = brentq(lambda t: np.tanh((x-t)/2).sum(), x.min(), x.max(), xtol=1e-13)
    else:
        selected = np.median(x)
    return dict(source=label, released_mle_residual=float(abs(selected-case["release"])))


def chain_batch(cfg, raw, out, refs, study_start):
    rows, checks, failures = [], [], []
    for ci, case in enumerate(cfg["cases"]):
        name = case_name(case)
        values, reference = refs[name]
        mu, x, w, _ = values
        maximum = np.max(abs(x-case["release"]), axis=1)
        indices = [int(np.argmin(abs(maximum-v))) for v in
                   weighted_quantiles(maximum, w, cfg["initial_max_quantiles"])]
        initials = [dict(index=i, x=x[i], mu=mu[i], max_residual=maximum[i]) for i in indices]
        write(out/(name+"_initials.json"), initials)
        methods = cfg["methods"][case["family"]]
        traces = {m: [] for m in methods}
        provenance = {m: [] for m in methods}
        failed = set()
        warm = []
        for method in methods:
            try:
                start = time.perf_counter()
                p = parameters(case, cfg, x[indices[0]], mu[indices[0]], method, cfg["warmup_iterations"])
                RUNNERS[case["family"]][method](cfg["seed"]+9000+ci, case["release"], p, False)
                warm.append(dict(method=method, seconds=time.perf_counter()-start))
            except Exception as exc:
                failed.add(method)
                failure(out/(name+"_"+method+"_warmup_failure.json"), exc)
                failures.append(dict(case=name, method=method, phase="warmup", error=repr(exc)))
        write(out/(name+"_warmup.json"), warm)
        for chain, idx in enumerate(indices):
            order = methods[chain % len(methods):] + methods[:chain % len(methods)]
            for method in order:
                if method in failed:
                    continue
                stem = f"{name}_{method}_chain{chain}"
                seed = cfg["seed"]+10000+100*ci+chain
                p = parameters(case, cfg, x[idx], mu[idx], method, cfg["iterations"])
                try:
                    start = time.perf_counter()
                    result = RUNNERS[case["family"]][method](seed, case["release"], p, False)
                    elapsed = time.perf_counter()-start
                    if not np.isfinite(result["x_chain"]).all() or not np.isfinite(result["mu_chain"]).all():
                        raise ArithmeticError("nonfinite retained trace")
                    if case["family"] == "laplace":
                        if np.max(abs(np.median(result["x_chain"], axis=1)-case["release"])) > 1e-10:
                            raise ArithmeticError("retained median constraint")
                    elif case["family"] == "logistic":
                        if np.max(abs(np.tanh((result["x_chain"]-case["release"])/2).sum(axis=1))) > 1e-10:
                            raise ArithmeticError("retained logistic score constraint")
                except Exception as exc:
                    failed.add(method)
                    if hasattr(exc, "partial"):
                        np.savez_compressed(raw/(stem+"_partial.npz"), **exc.partial)
                    failure(out/(stem+"_failure.json"), exc, seed=seed, initial_x=x[idx], initial_mu=mu[idx])
                    failures.append(dict(case=name, method=method, phase="chain", chain=chain, error=repr(exc)))
                    write(out/"failures.json", failures)
                    print("FAILED", stem, repr(exc), flush=True)
                    continue
                arrays = dict(mu=result["mu_chain"], x=result["x_chain"])
                path = raw/(stem+".npz")
                np.savez_compressed(path, **arrays)
                meta = {k: v for k, v in result.items() if k not in
                        ("mu_chain", "x_chain", "log_weight_chain", "accepted_flags")}
                meta.update(loop_seconds=result["seconds"], seconds=elapsed, seed=seed,
                            raw_path=str(path.relative_to(ROOT)), sha256=sha(path))
                if method == "gibbs":  # exact Laplace latent conditional, not an MH acceptance
                    meta["counts"]["accepted"] = cfg["iterations"]
                    meta["acceptance_kind"] = "exact conditional updates"
                write(out/(stem+".json"), meta)
                traces[method].append(result)
                provenance[method].append(meta)
                print("chain", stem, round(elapsed, 3), flush=True)
                if chain == 0:
                    xx = result["x_chain"]
                    for index in [len(xx)-1, int(np.argmax(np.max(abs(xx-case["release"]), axis=1)))]:
                        check = independent_check(xx[index], case, stem+f"_{index}")
                        checks.append(check)
                        write(out/"independent_checks.json", checks)
                        if "error" in check or check["released_mle_residual"] > 1e-9:
                            raise ArithmeticError("independent endpoint verification failed")
                if time.monotonic()-study_start > cfg["total_ceiling_seconds"]:
                    raise TimeoutError("overall study ceiling")
        for method in methods:
            if method in failed:
                rows.append(dict(case=case, method=method, complete=False,
                                 completed_chains=len(traces[method]), eligible_for_ranking=False))
                continue
            row = summarize_chains(case, cfg, traces[method], provenance[method], reference, values)
            row.update(method=method, complete=True, reference=reference["summary"],
                       reference_uncertainty=reference["uncertainty"],
                       telemetry=provenance[method])
            row["tail6_events_per_chain"] = [
                int((np.max(abs(r["x_chain"][cfg["burn_in"]+1:]-case["release"]), axis=1)>6).sum())
                for r in traces[method]]
            row["screens"] = passes_screens(row, cfg)
            row["eligible_for_ranking"] = row["screens"]["all_pass"]
            row["ess_raw_per_second"] = {k: d["ess_raw"]/row["seconds"] if d["ess_raw"] is not None else None
                                         for k, d in row["diagnostics"].items()}
            rows.append(row)
            print("summary", name, method, row["screens"], flush=True)
        write(out/"comparisons.json", rows)
    write(out/"failures.json", failures)
    return rows


def direct_controls(cfg, raw, out, refs):
    banks, rows = {}, []
    for ci, case in enumerate(cfg["cases"]):
        name = case_name(case)
        key = (case["family"], case["n"])
        if key not in banks:
            base, mle, meta = bank(*key, cfg["direct_draws"], cfg["seed"]+20000+ci,
                                   cfg["reference_ceiling_seconds"])
            path = raw/f"direct_{key[0]}_n{key[1]}.npz"
            np.savez_compressed(path, base=base, mle=mle)
            meta.update(raw_path=str(path.relative_to(ROOT)), sha256=sha(path))
            banks[key] = base, mle, meta
        base, mle, meta = banks[key]
        start = time.perf_counter()
        mu, x, w, normalizer = weight_particles(base, mle, case)
        weighted_seconds = meta["seconds"]+time.perf_counter()-start
        weighted_summary = summary(mu, w, x, case)
        event = (np.max(abs(x-case["release"]), axis=1)>6).astype(float)
        weighted_se = particle_bootstrap(mu, w, x, case, cfg["seed"]+30000+ci, 64)
        function_ess = dict(mu=weighted_function_ess(mu, w), tail6=weighted_function_ess(event, w))
        start = time.perf_counter()
        rmu = case["release"]-mle
        u = np.random.default_rng(cfg["seed"]+40000+ci).random(len(mle))
        accept = u < np.exp(-.5*((rmu-case["prior_mean"])/case["prior_std"])**2)
        # Reconstruct accepted rows only; include RNG and array construction in cost.
        rmu = rmu[accept]
        rx = base[accept]+rmu[:, None]
        rejection_seconds = meta["seconds"]+time.perf_counter()-start
        count = int(accept.sum())
        rejection_summary = summary(rmu, np.ones(count), rx, case) if count else None
        events = np.max(abs(rx-case["release"]), axis=1)>6 if count else np.array([], dtype=bool)
        reference = refs[name][1]
        expected_acceptance = reference["normalizer"]*np.sqrt(2*np.pi)*case["prior_std"]
        path = raw/(name+"_rejection.npz")
        np.savez_compressed(path, mu=rmu, x=rx)
        row = dict(case=case, proposals=len(mle), bank=meta, weighted_seconds=weighted_seconds,
                   weighted_summary=weighted_summary, weighted_uncertainty=weighted_se,
                   weighted_ess=float(w.sum()**2/(w@w)),
                   weighted_function_ess=function_ess,
                   weighted_equivalent_per_second={k: v/weighted_seconds if v is not None else None
                                                    for k, v in function_ess.items()},
                   rejection_seconds=rejection_seconds, rejected_proposals=len(mle)-count,
                   rejection_accepted=count, rejection_acceptance=count/len(mle),
                   rejection_acceptance_interval95=binomial_interval(count, len(mle)),
                   reference_expected_acceptance=float(expected_acceptance),
                   rejection_independent_per_second=count/rejection_seconds,
                   rejection_summary=rejection_summary, rejection_tail6_events=int(events.sum()),
                   rejection_tail6_interval95=binomial_interval(int(events.sum()), count),
                   rejection_raw_path=str(path.relative_to(ROOT)), rejection_sha256=sha(path),
                   reference=reference["summary"],
                   cost_policy="Full fresh-bank cost charged separately to each method/prior; diagnostics excluded; no amortized-cache speedup.")
        rows.append(row)
        write(out/"direct_comparisons.json", rows)
        print("direct", name, "accepted", count, "weighted ESS", round(row["weighted_ess"]), flush=True)
    return rows


def figures(cfg, raw, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in load(out/"comparisons.json") if r["complete"]]
    direct = {case_name(r["case"]): r for r in load(out/"direct_comparisons.json")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), constrained_layout=True)
    for ax, metric in zip(axes, ["mu", "tail6"]):
        for i, row in enumerate(rows):
            v = row["ess_raw_per_second"][metric]
            color = "tab:blue" if row["eligible_for_ranking"] else "tab:red"
            if v is not None:
                ax.plot(v, i, "o" if row["eligible_for_ranking"] else "x", color=color, ms=5)
            d = direct[case_name(row["case"])]
            val = d["weighted_equivalent_per_second"][metric]
            if val is not None: ax.plot(val, i, "|", color="tab:green", ms=9)
            ax.plot(d["rejection_independent_per_second"], i, "+", color="gray", ms=6)
        labels = [case_name(r["case"])+" / "+r["method"] for r in rows]
        ax.set_yticks(range(len(rows)), labels if ax is axes[0] else [""]*len(rows))
        ax.invert_yaxis(); ax.set_xscale("log")
        ax.set_xlabel(metric+": effective/independent-equivalent samples per second")
    axes[0].set_title("Blue: passes screens; red: diagnostic only")
    axes[1].set_title("Green: weighted estimate; gray: independent rejection")
    fig.savefig(out/"efficiency.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), constrained_layout=True)
    for ax, metric in zip(axes, ["mean", "latent_tail6"]):
        for i, row in enumerate(rows):
            a,b = row["mcse"][metric], row["reference_uncertainty"][metric]["se"]
            ax.errorbar(row["summary"][metric]-row["reference"][metric], i,
                        xerr=2*np.hypot(a,b) if a is not None and b is not None else None,
                        fmt="o", color="tab:blue" if row["eligible_for_ranking"] else "tab:red", ms=4)
        ax.axvline(0, color="black", lw=.7)
        ax.set_yticks(range(len(rows)), labels if ax is axes[0] else [""]*len(rows))
        ax.invert_yaxis(); ax.set_xlabel(metric+" minus weighted reference (±2 combined MCSE)")
    fig.savefig(out/"reference_agreement.png", dpi=150); plt.close(fig)
    # Show all four latent traces for the previously difficult Student case.
    case = next(c for c in cfg["cases"] if c["family"]=="student" and c["id"]=="centered_n10")
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for ax, method in zip(axes, cfg["methods"]["student"]):
        for i in range(cfg["chains"]):
            path = raw/f"{case_name(case)}_{method}_chain{i}.npz"
            if not path.exists():
                continue
            a = np.load(path)
            idx = np.arange(0, len(a["mu"]), 10)
            ax.plot(idx, np.max(abs(a["x"][idx]-case["release"]), axis=1), lw=.5, label=str(i))
        ax.axvline(cfg["burn_in"], ls=":", color="gray")
        ax.set_yscale("log"); ax.set_ylabel(method+"\nmax |X−t|")
    axes[0].legend(title="chain", ncol=4); axes[-1].set_xlabel("iteration")
    fig.savefig(out/"student_latent_traces.png", dpi=150); plt.close(fig)


def run(config_path):
    cfg = load(config_path)
    out, raw = ROOT/cfg["evidence_directory"], ROOT/cfg["raw_directory"]
    if raw.exists() or (out/"run_started.json").exists():
        raise FileExistsError("Study already started; use fresh config/output locations.")
    raw.mkdir(parents=True)
    start = time.monotonic()
    paths = [p for folder in ["models", "kde_ref", "diagnostics", "scripts", "tests"]
             for p in (ROOT/folder).rglob("*.py")]
    write(out/"run_started.json", dict(config_sha256=sha(config_path),
          source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
          python=sys.version, platform=platform.platform(), cpu_affinity=sorted(os.sched_getaffinity(0)),
          versions={name: importlib.metadata.version(name) for name in ["numpy","scipy","numba","sympy"]},
          start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    try:
        refs = prepare_references(cfg, raw, out)
        rows = chain_batch(cfg, raw, out, refs, start)
        direct_controls(cfg, raw, out, refs)
        figures(cfg, raw, out)
        write(out/"completion.json", dict(seconds=time.monotonic()-start,
              completed_method_cases=sum(r["complete"] for r in rows),
              failed_method_cases=sum(not r["complete"] for r in rows),
              chain_iterations=sum(cfg["iterations"]*cfg["chains"] for r in rows if r["complete"]),
              passes_all_screens=sum(r["eligible_for_ranking"] for r in rows)))
    except Exception as exc:
        failure(out/"study_failure.json", exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analyze", action="store_true", help="Regenerate figures only; no sampling")
    args = parser.parse_args()
    if args.analyze:
        cfg = load(args.config)
        figures(cfg, ROOT/cfg["raw_directory"], ROOT/cfg["evidence_directory"])
    else:
        run(args.config)
