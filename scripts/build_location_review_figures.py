"""Build two collaborator figures from hashed saved chains; never run a sampler.

Uses original burn-in and full stored arrays. Diagnostic annotations come from
the separately verified expanded audit. Output directory must be new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/"docs/research/evidence/location_only_performance_2026-10-12"
AUDIT = ROOT/"docs/research/evidence/location_chain_diagnostics_2026-10-12"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
METHODS = ["block", "rattle", "independence"]
TITLES = ["Block Gibbs", "RATTLE", "Independence MH"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = {str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__)}
    manifests = {}
    for folder in (STUDY, AUDIT):
        path = folder/"artifact_manifest.json"
        inputs[str(path.relative_to(ROOT))] = sha(path)
        manifests.update(json.loads(path.read_text()))

    def read(path):
        key = str(path.relative_to(ROOT))
        digest = sha(path)
        if key not in manifests or manifests[key]["sha256"] != digest:
            raise ArithmeticError(f"evidence hash mismatch: {key}")
        inputs[key] = digest
        return json.loads(path.read_text())

    cfg = read(STUDY/"config.json")
    rows = read(AUDIT/"diagnostics.json")
    stats = {(r["case"], r["method"]):r for r in rows}
    plt.rcParams.update({"path.simplify": False, "font.size": 9})
    records = []
    for case_id, p in [("centered_n10", .95), ("shifted_n10", .05)]:
        case = next(c for c in cfg["cases"] if c["family"]=="student" and c["id"]==case_id)
        name = "student_"+case_id
        variable = f"latent_q{round(p*100):02d}"
        ref = read(AUDIT/(name+"_reference_quantiles.json"))
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.4), sharex=True, sharey="row")
        for col, method in enumerate(METHODS):
            mu, latent = [], []
            for chain in range(cfg["chains"]):
                meta = read(STUDY/f"{name}_{method}_chain{chain}.json")
                path = ROOT/meta["raw_path"]
                if sha(path) != meta["sha256"]:
                    raise ArithmeticError(f"raw chain hash mismatch: {path}")
                inputs[meta["raw_path"]] = meta["sha256"]
                with np.load(path, allow_pickle=False) as a:
                    m, x = a["mu"], a["x"]
                if m.shape != (cfg["iterations"]+1,) or x.shape != (len(m),case["n"]):
                    raise ValueError("saved array dimensions do not match frozen configuration")
                if not np.isfinite(m).all() or not np.isfinite(x).all():
                    raise ValueError("nonfinite saved state")
                mu.append(m)
                latent.append(np.quantile(x-case["release"],p,axis=1,method="linear"))
            for row, (key, values) in enumerate([("mu",mu),(variable,latent)]):
                d = stats[(name,method)]["metrics"][key]
                pooled = np.asarray(values)[:,cfg["burn_in"]+1:]
                np.testing.assert_allclose(pooled.mean(), d["pooled"]["mean"], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(np.quantile(pooled,.5),d["pooled"]["q50"],rtol=1e-12,atol=1e-12)
                ax = axes[row,col]
                for i, a in enumerate(values):
                    ax.plot(np.arange(len(a)),a,color=COLORS[i],lw=.35,alpha=.66,
                            label=f"Chain {i+1}",rasterized=True)
                ax.axvspan(0,cfg["burn_in"],color=".25",alpha=.09)
                ax.axvline(cfg["burn_in"],color=".4",ls=":",lw=.7)
                for q in ("q05","q95"):
                    ax.axhline(ref[key][q],color="black",ls="--",lw=.8)
                ax.set_title(f"{TITLES[col]}\nR-hat {d['rhat']:.3f}; bulk/tail ESS {d['ess_bulk']:.0f}/{d['ess_tail']:.0f}")
                ax.grid(alpha=.15)
                if row:
                    ax.set_yscale("symlog",linthresh=2.)
                    ax.set_xlabel("Iteration")
                records.append(dict(case=name,method=method,variable=key,rhat=d["rhat"],
                                    ess_bulk=d["ess_bulk"],ess_tail=d["ess_tail"],
                                    pooled_mean=d["pooled"]["mean"],screen=d["screen"]))
        axes[0,0].set_ylabel(r"Parameter $\mu$")
        axes[1,0].set_ylabel(f"Within-dataset {round(100*p)}th percentile of x-t\n(symmetric log scale)")
        title = (r"Centered: $t=0$, prior $N(0,2^2)$" if p==.95 else
                 r"Shifted: $t=1.25$, prior $N(-0.75,0.6^2)$")
        fig.suptitle("Student k=2, n=10 | "+title,fontsize=12)
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=4,loc="lower center",bbox_to_anchor=(.5,.015),frameon=False)
        fig.text(.5,.003,"Every stored iteration shown. Grey: original burn-in. Dashed: finite-bank 5%/95% quantiles, not uncertainty bands.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.08,1,.96))
        for extension in ("png","pdf"):
            fig.savefig(args.out/f"{case_id}.{extension}",dpi=160,
                        metadata={"Creator":"build_location_review_figures.py"} if extension=="pdf" else None)
        plt.close(fig)
    for path, digest in inputs.items():
        if sha(ROOT/path) != digest:
            raise ArithmeticError(f"input changed during rendering: {path}")
    save(args.out/"provenance.json",dict(input_sha256=inputs,annotations=records,
         iterations=cfg["iterations"],burn_in=cfg["burn_in"],retained_start=cfg["burn_in"]+1,
         quantile_method="linear",thinning=1,new_sampling_iterations=0,
         command=f".venv/bin/python scripts/build_location_review_figures.py --out {args.out}",
         outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(args.out.iterdir())}))
    print(f"Saved two figures (PNG/PDF); {len(inputs)} input hashes checked; no sampling.")


if __name__ == "__main__":
    main()
