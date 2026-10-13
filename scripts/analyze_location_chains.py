"""Read-only convergence audit of the frozen location performance chains.

No sampling, thinning, changed burn-in, or overwriting existing study evidence.
Output must be a new directory. Uses the study's diagnostic estimators, with
explicit handling of statistics fixed by the Laplace conditioning constraint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np
from scripts.followup_shifted_student_reference import diagnostics, autocorrelation_ess, longest_run
from scripts.validate_selected_mle_reference import weighted_quantiles, weight_particles
from scripts.validate_student_selected_mle_gibbs import particles

PROBS = [.01, .05, .25, .50, .75, .95, .99]
WITHIN_PROBS = [.05, .25, .50, .75, .95]
DEFAULT_STUDY = ROOT / "docs/research/evidence/location_only_performance_2026-10-12"
DEFAULT_OUT = ROOT / "docs/research/evidence/location_chain_diagnostics_2026-10-12"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def observables(mu, x, case):
    """Axes: chain, iteration, observation. Quantiles use linear interpolation."""
    if x.shape[:-1] != mu.shape or x.shape[-1] != case["n"]:
        raise ValueError("incompatible parameter/latent array shapes")
    residual = x - case["release"]
    ordered = np.sort(residual, axis=-1)
    result = {"mu": mu}
    result.update({f"x{i+1}": residual[..., i] for i in range(case["n"])})
    result.update({f"order{i+1}": ordered[..., i] for i in range(case["n"])})
    for p, a in zip(WITHIN_PROBS, np.quantile(residual, WITHIN_PROBS, axis=-1, method="linear")):
        result[f"latent_q{round(100*p):02d}"] = a
    result["maximum"] = np.max(np.abs(residual), axis=-1)
    result["tail6"] = (result["maximum"] > 6).astype(float)
    result["spread"] = ordered[..., -1] - ordered[..., 0]
    return result


def constraint_fixed(name, case):
    return case["family"] == "laplace" and (
        name == "latent_q50" or (case["n"] % 2 == 1 and name == f"order{case['n']//2+1}"))


def scalar_diagnostics(a, *, fixed=False):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or min(a.shape) < 2 or a.shape[1] < 8 or not np.isfinite(a).all():
        raise ValueError("need finite multiple chains with at least eight draws")
    empty = dict(rhat=None, ess_bulk=None, ess_tail=None, ess_raw=None)
    if fixed:
        error = float(np.max(np.abs(a)))
        if error > 1e-10:
            raise ArithmeticError("constraint-fixed latent statistic drifted")
        return dict(**empty, status="constraint_fixed", maximum_constraint_error=error,
                    quantile_ess={})
    if np.all(a == a.flat[0]):
        return dict(**empty, status="unresolved_constant", quantile_ess={})
    if np.all(np.var(a, axis=1) == 0):
        return dict(**empty, status="unresolved_chains_stuck_at_different_values", quantile_ess={})
    d = diagnostics(a)
    d["status"] = "resolved" if all(d[k] is not None for k in empty) else "unresolved"
    # Function ESS for the CDF indicator at each pooled empirical quantile.
    # This is not ESS for the raw value or a precision guarantee for far tails.
    d["quantile_ess"] = {f"q{round(100*p):02d}": autocorrelation_ess(a <= q)
                         for p, q in zip(PROBS, np.quantile(a, PROBS))}
    return d


def summary(a):
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                **{f"q{round(100*p):02d}": float(v) for p, v in zip(PROBS, np.quantile(a, PROBS))})


def acf(a, max_lag=200):
    """Per-chain ACF; constant chains yield NaN and are explicitly unplottable."""
    z = a - np.mean(a, axis=1, keepdims=True)
    n = a.shape[1]
    f = np.fft.rfft(z, n=1 << (2*n-1).bit_length(), axis=1)
    c = np.fft.irfft(f*np.conj(f), axis=1)[:, :min(n, max_lag+1)]
    return np.divide(c, c[:, :1], out=np.full_like(c, np.nan), where=c[:, :1] > 0)


def screen(d, cfg):
    if d["status"] == "constraint_fixed":
        return None
    return d["status"] == "resolved" and d["rhat"] <= cfg["rhat_max"] and all(
        d[k] >= cfg["ess_min"] for k in ("ess_bulk", "ess_tail"))


def inspect_scalar(a, name, case, cfg):
    d = scalar_diagnostics(a, fixed=constraint_fixed(name, case))
    d["screen"] = screen(d, cfg)
    d["pooled"] = summary(a)
    d["per_chain"] = [summary(chain) for chain in a]
    half = a.shape[1]//2
    d["halves"] = [[summary(chain[:half]), summary(chain[-half:])] for chain in a]
    # Report change in means, not a significance test with many dependent tests.
    d["largest_half_mean_change"] = max(abs(v[1]["mean"]-v[0]["mean"]) for v in d["halves"])
    return d


def verify_hashes(expected):
    for path, digest in expected.items():
        if sha(ROOT/path) != digest:
            raise ArithmeticError(f"input provenance mismatch: {path}")


def read_chains(study, case, method, cfg):
    mus, xs, inventory = [], [], []
    name = case["family"] + "_" + case["id"]
    for i in range(cfg["chains"]):
        meta_path = study/f"{name}_{method}_chain{i}.json"
        meta = load(meta_path)
        path = ROOT/meta["raw_path"]
        if sha(path) != meta["sha256"]:
            raise ArithmeticError(f"raw chain hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            mu, x = data["mu"], data["x"]
        if mu.shape != (cfg["iterations"]+1,) or x.shape != (len(mu), case["n"]):
            raise ValueError(f"unexpected full chain shape: {path}")
        if not np.isfinite(mu).all() or not np.isfinite(x).all():
            raise ValueError(f"nonfinite chain: {path}")
        mus.append(mu); xs.append(x)
        inventory.append(dict(case=name, method=method, chain=i, seed=meta["seed"],
                              raw_path=meta["raw_path"], sha256=meta["sha256"],
                              metadata_path=str(meta_path.relative_to(ROOT)),
                              bytes=path.stat().st_size, mu_key="mu", x_key="x",
                              mu_shape=str(mu.shape), x_shape=str(x.shape),
                              retained_start_index=cfg["burn_in"]+1,
                              retained_draws=cfg["iterations"]-cfg["burn_in"]))
    if len(set(r["seed"] for r in inventory)) != cfg["chains"]:
        raise ArithmeticError("chains reuse a seed within the same method")
    return np.stack(mus), np.stack(xs), inventory


def reference_values(case, study):
    ref = load(study/(case["family"]+"_"+case["id"]+"_reference.json"))
    if case["family"] == "student":
        old = ROOT/"docs/research/evidence/student_selected_mle_gibbs_2026-10-10"
        cfg = load(old/"config.json")
        if case["n"] == 3:
            meta = load(old/"fresh_n3_provenance.json")
            verify_hashes({meta["raw_path"]: meta["sha256"]})
        _, _, (mu, x, w, _) = particles(case, cfg, ROOT/cfg["raw_directory"])
    else:
        meta = ref["provenance"]
        verify_hashes({meta["raw_path"]: meta["sha256"]})
        with np.load(ROOT/meta["raw_path"], allow_pickle=False) as a:
            mu, x, w, _ = weight_particles(a["base"], a["mle"], case)
    obs = observables(mu, x, case)
    return {key: {"mean": float(w @ a / w.sum()),
                  **{f"q{round(100*p):02d}": float(v) for p, v in
                     zip(PROBS, weighted_quantiles(a, w, PROBS))}}
            for key, a in obs.items()}


def draw_trace(ax, a, burn, *, symlog=False):
    # Plot EVERY stored iteration. Do not stride away rare spikes.
    for i, chain in enumerate(a):
        ax.plot(chain, lw=.32, alpha=.66, color=COLORS[i], label=f"Chain {i+1}")
    ax.axvspan(0, burn, color=".3", alpha=.10)
    ax.axvline(burn, color=".3", lw=.6, ls=":")
    if symlog:
        ax.set_yscale("symlog", linthresh=2.)
    ax.grid(alpha=.15)


def short_label(d):
    if d["status"] != "resolved":
        return d["status"].replace("_", " ")
    return f"R-hat {d['rhat']:.3f}; ESS bulk/tail {d['ess_bulk']:.0f}/{d['ess_tail']:.0f}"


def plot_case(case, runs, refs, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Disable simplification to preserve all tail excursions even at small width.
    plt.rcParams.update({"path.simplify": False, "font.size": 8})
    name = case["family"]+"_"+case["id"]
    methods = list(runs)
    variables = ["mu", "latent_q05", "latent_q25", "latent_q50", "latent_q75", "latent_q95", "maximum"]
    fig, axes = plt.subplots(len(variables), len(methods), figsize=(6*len(methods), 15), squeeze=False)
    for j, method in enumerate(methods):
        obs, metrics = runs[method]
        for i, key in enumerate(variables):
            ax = axes[i, j]
            draw_trace(ax, obs[key], cfg["burn_in"], symlog=case["family"]=="student" and key!="mu")
            ax.set_title(f"{method} | {key}\n{short_label(metrics[key])}")
            for q in ("q05", "q95"):
                ax.axhline(refs[key][q], color="black", lw=.7, ls="--")
            if i == 0:
                ax.legend(ncol=4, fontsize=7, loc="upper right")
            if i == len(variables)-1:
                ax.set_xlabel("Stored iteration (initial state at 0)")
    fig.suptitle(f"{name}: full traces; grey = original burn-in; dashed = raw reference 5%/95%\n"
                 "Latent values are residuals x-t. Student latent axes use symmetric log scale.")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out/f"{name}_traces.png", dpi=115); plt.close(fig)

    fig, axes = plt.subplots(len(methods), 4, figsize=(18, 3.2*len(methods)), squeeze=False)
    for j, method in enumerate(methods):
        obs, metrics = runs[method]
        for i, key in enumerate(["mu", "latent_q05", "latent_q95"]):
            a = obs[key][:, cfg["burn_in"]+1:]
            for c, chain in enumerate(a):
                axes[j, i].plot(np.arange(1, len(chain)+1), np.cumsum(chain)/np.arange(1, len(chain)+1),
                                color=COLORS[c], lw=.7)
            axes[j, i].axhline(refs[key]["mean"], color="black", ls="--", lw=.8)
            axes[j, i].set_title(f"{method}: running mean of {key}")
            axes[j, i].set_xlabel("Retained draws")
        for c, values in enumerate(acf(obs["maximum"][:, cfg["burn_in"]+1:])):
            axes[j, 3].plot(values, color=COLORS[c], lw=.8, label=f"Chain {c+1}")
        axes[j, 3].set_title(f"{method}: maximum-residual ACF")
        axes[j, 3].set_xlabel("Lag (up to 200)"); axes[j, 3].legend(fontsize=7)
        for ax in axes[j]:
            ax.grid(alpha=.2)
    fig.suptitle(f"{name}: cumulative means and latent autocorrelation (post burn-in)\n"
                 "Dashed lines are finite-bank weighted estimates, not exact values or confidence bands.")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(out/f"{name}_stability.png", dpi=115); plt.close(fig)

    for method, (obs, metrics) in runs.items():
        fig, axes = plt.subplots(case["n"], 2, figsize=(15, max(6, 1.9*case["n"])), squeeze=False)
        for i in range(case["n"]):
            for j, key in enumerate([f"x{i+1}", f"order{i+1}"]):
                draw_trace(axes[i,j], obs[key], cfg["burn_in"], symlog=case["family"]=="student")
                axes[i,j].set_title(f"{key}: {short_label(metrics[key])}")
                if i == 0:
                    axes[i,j].legend(ncol=4, fontsize=7)
        fig.suptitle(f"{name} / {method}: labelled x_i-t (left); ordered x_(i)-t (right)\n"
                     "Full unthinned traces. Grey = burn-in. Student axes use symmetric log scale.")
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(out/f"{name}_{method}_coordinates.png", dpi=110); plt.close(fig)


def group_metrics(metrics, names):
    ds = {k: metrics[k] for k in names if metrics[k]["status"] != "constraint_fixed"}
    if not ds:
        return dict(worst_rhat=None, minimum_bulk_ess=None, minimum_tail_ess=None, failures=[])
    return dict(worst_rhat=max((d["rhat"] for d in ds.values() if d["rhat"] is not None), default=None),
                minimum_bulk_ess=min((d["ess_bulk"] for d in ds.values() if d["ess_bulk"] is not None), default=None),
                minimum_tail_ess=min((d["ess_tail"] for d in ds.values() if d["ess_tail"] is not None), default=None),
                failures=[k for k,d in ds.items() if not d["screen"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    study, out = args.study.resolve(), args.out.resolve()
    cfg = load(study/"config.json")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    frozen = {p:v["sha256"] for p,v in load(study/"artifact_manifest.json").items()}
    frozen.update(load(study/"run_started.json")["source_sha256"])
    for p in (study/"artifact_manifest.json", Path(__file__), ROOT/"tests/test_location_chain_diagnostics.py"):
        frozen[str(p.relative_to(ROOT))] = sha(p)
    verify_hashes(frozen)
    write(out/"analysis_config.json", dict(study=str(study.relative_to(ROOT)),
          burn_in=cfg["burn_in"], retained_start_index=cfg["burn_in"]+1, within_dataset_quantiles=WITHIN_PROBS,
          posterior_quantiles=PROBS, quantile_interpolation="linear", rhat_max=cfg["rhat_max"], ess_min=cfg["ess_min"],
          policy="Exploratory extension of frozen study; no reruns, thinning, tuned burn-in or changed original screens.",
          source_sha256=sha(__file__), reference_bands="Finite-bank weighted posterior 5/95 percentiles, not uncertainty intervals."))
    inventory, scalar_rows, results = [], [], []
    original = {(r["case"]["family"]+"_"+r["case"]["id"],r["method"]):r
                for r in load(study/"comparisons.json")}
    gallery = ["<!doctype html><meta charset='utf-8'><title>Location chain diagnostics</title>",
               "<style>body{max-width:1600px;margin:30px auto;font:16px sans-serif}img{width:100%}summary{cursor:pointer}</style>",
               "<h1>Location chain diagnostics</h1><p>All 92 stored chains; no reruns. Full parameter and latent traces, "
               "within-dataset 5/25/50/75/95% quantiles, all labelled and ordered coordinates. "
               "Grey is the original burn-in. R-hat/ESS use only retained draws. "
               "These are convergence diagnostics, not a convergence proof.</p>"]
    for case in cfg["cases"]:
        name = case["family"]+"_"+case["id"]
        refs = reference_values(case, study)
        write(out/(name+"_reference_quantiles.json"), refs)
        runs = {}
        for method in cfg["methods"][case["family"]]:
            mu, x, rows = read_chains(study, case, method, cfg)
            inventory.extend(rows)
            obs = observables(mu, x, case)
            metrics = {k:inspect_scalar(a[:,cfg["burn_in"]+1:], k, case, cfg) for k,a in obs.items()}
            runs[method] = obs, metrics
            # Confirm original diagnostics have not changed, before extending them.
            old = original[(name,method)]
            for key in ("mu", "maximum", "tail6"):
                for field in ("rhat", "ess_bulk", "ess_tail", "ess_raw"):
                    np.testing.assert_allclose(metrics[key][field], old["diagnostics"][key][field], rtol=1e-12, atol=1e-12)
            groups = {"mu":["mu"], "labels":[f"x{i+1}" for i in range(case["n"])],
                      "orders":[f"order{i+1}" for i in range(case["n"])],
                      "latent_quantiles":[f"latent_q{round(p*100):02d}" for p in WITHIN_PROBS],
                      "all":list(metrics)}
            group = {key:group_metrics(metrics, names) for key,names in groups.items()}
            tail = obs["tail6"][:,cfg["burn_in"]+1:]
            result = dict(case=name, method=method, original_screen=old["screens"]["all_pass"],
                          expanded_screen=not group["all"]["failures"], groups=group,
                          constant_by_constraint=[k for k,d in metrics.items() if d["status"]=="constraint_fixed"],
                          per_chain_tail6_probability=tail.mean(axis=1).tolist(),
                          longest_tail6_residence=[longest_run(a.astype(bool)) for a in tail],
                          metrics=metrics)
            results.append(result)
            for key,d in metrics.items():
                scalar_rows.append(dict(case=name,method=method,variable=key,status=d["status"],
                   rhat=d["rhat"],ess_bulk=d["ess_bulk"],ess_tail=d["ess_tail"],ess_raw=d["ess_raw"],
                   expanded_screen=d["screen"],mean=d["pooled"]["mean"],
                   largest_half_mean_change=d["largest_half_mean_change"],
                   **{f"posterior_{q}":d["pooled"][q] for q in ("q01","q05","q25","q50","q75","q95","q99")},
                   **{f"ess_{q}":d["quantile_ess"].get(q) for q in ("q01","q05","q25","q50","q75","q95","q99")}))
            print(name, method, "expanded failures:", group["all"]["failures"], flush=True)
        plot_case(case, runs, refs, cfg, out)
        gallery.append(f"<h2>{html.escape(name)}</h2><img loading='lazy' src='{name}_traces.png'>"
                       f"<details><summary>Running estimates and autocorrelation</summary><img loading='lazy' src='{name}_stability.png'></details>")
        for method in runs:
            gallery.append(f"<details><summary>{html.escape(method)}: all labelled and ordered latent coordinates</summary>"
                           f"<img loading='lazy' src='{name}_{method}_coordinates.png'></details>")
    csv_write(out/"chain_inventory.csv", inventory)
    csv_write(out/"scalar_diagnostics.csv", scalar_rows)
    write(out/"diagnostics.json", results)
    (out/"index.html").write_text("\n".join(gallery)+"\n")
    verify_hashes(frozen)
    write(out/"completion.json", dict(seconds=time.monotonic()-start, chains=len(inventory),
          compressed_chain_bytes=sum(r["bytes"] for r in inventory),
          method_cases=len(results), scalar_traces=len(scalar_rows), figures=len(list(out.glob("*.png"))),
          original_screen_passes=sum(r["original_screen"] for r in results),
          expanded_screen_passes=sum(r["expanded_screen"] for r in results),
          inputs_verified_unchanged=len(frozen), original_diagnostics_reproduced=True, new_sampling_iterations=0))
    write(out/"input_hashes.json", frozen)
    write(out/"artifact_manifest.json", {str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))
                                        for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(load(out/"completion.json")), flush=True)


if __name__ == "__main__":
    main()
