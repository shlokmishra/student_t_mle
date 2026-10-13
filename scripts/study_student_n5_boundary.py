"""Bounded exact-algebra and deterministic selector audit; no MCMC runs.

Replay with --output pointing to a fresh directory. Existing evidence is never
overwritten. SymPy/mpmath must be supplied by the documented environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import numpy as np
import sympy as sp
from reporting.diagnostics import student_n5_boundary as theory
from scripts.selected_mle_reference_oracle import selected_mle_oracle
from models.loc.student.mle import global_mle, stationary_points, score, StudentMLEError


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def limited(function, seconds=20):
    def alarm(signum, frame):
        raise TimeoutError("deterministic audit operation exceeded 20 seconds")
    old = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    start = time.monotonic()
    try:
        return function(), time.monotonic()-start
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def datasets(ac):
    cases = []
    def add(label, a, perturb_index=None, perturb=0.):
        x = np.array([-a, -a, 0., a, a])
        if perturb_index is not None:
            x[perturb_index] += perturb
        cases.append({"label": label, "a": float(a), "a_hex": float(a).hex(),
                      "data": x.tolist(), "data_hex": [z.hex() for z in x],
                      "perturb_index": perturb_index, "perturb": perturb})
    for a in (1.2, 3., 4., 5.):
        add(f"safe_a_{a}", a)
    for delta in (-1e-8, -1e-12, 1e-12, 1e-8):
        add(f"near_delta_{delta}", ac+delta)
    for name, a in (("binary_predecessor", np.nextafter(ac, -np.inf)),
                    ("rounded_algebraic_parameter", ac),
                    ("binary_successor", np.nextafter(ac, np.inf))):
        add(name, a)
    for index, kind in ((2, "center"), (4, "outer")):
        for power in (30, 45):
            for sign in (-1, 1):
                add(f"asymmetric_{kind}_{sign}_2minus{power}", ac, index, sign*2.**(-power))
    for name, a in (("birth_predecessor", np.nextafter(np.sqrt(5.), -np.inf)),
                    ("birth_successor", np.nextafter(np.sqrt(5.), np.inf))):
        add(name, a)
    return cases


def audit(case):
    x = np.array(case["data"])
    row = dict(case)
    try:
        oracle, seconds = limited(lambda: selected_mle_oracle(x, 1., fast=True))
        row.update(oracle=oracle, oracle_seconds=seconds)
    except Exception as exc:
        row["oracle_error"] = repr(exc)
        return row
    if case["perturb_index"] is None:
        family = theory.symmetric_classification(sp.Rational(case["a"]))
        selected = float(sp.N(family["selected"], 60))
        row["exact_family_oracle"] = {"status": family["status"], "K_sign": int(family["K_sign"]),
                                      "selected": selected,
                                      "matches_rational_oracle": abs(selected-oracle["selected"]) <= 2e-13}
    try:
        (selected, info), seconds = limited(lambda: global_mle(x, 1., return_diagnostics=True))
        row.update(production_selected=selected, production_diagnostics=info,
                   production_seconds=seconds, selected_location_error=abs(selected-oracle["selected"]),
                   selected_score_residual=abs(score(x, selected, 1.)))
        values = [sp.Rational(float(z)) for z in x]
        # High-precision diagnostic differences, separately from the oracle's
        # exact interval global comparison. Never use double LL rounding to
        # decide whether a near-boundary root is the winner.
        left, right = map(sp.Rational, oracle["selected_interval"])
        oracle_mid = (left+right)/2
        objective = lambda m: sum(sp.log(1+(z-m)**2) for z in values)
        row["loglik_loss_80digit_diagnostic"] = str(sp.N(objective(sp.Rational(selected))-objective(oracle_mid), 80))
        row["selected_score_80digit_diagnostic"] = str(sp.N(sum((z-sp.Rational(selected))/(1+(z-sp.Rational(selected))**2) for z in values), 80))
        # Compare the central and negative outer modes even where float LLs
        # round equal. This describes how asymmetric perturbations split ties.
        roots = oracle["roots"]
        row["stationary_loglik_relative_center_80digit"] = [str(sp.N(objective(sp.Rational(roots[len(roots)//2]))-objective(sp.Rational(m)), 80)) for m in roots]
        row["global_winner_matches"] = row["selected_location_error"] <= 2e-13
    except Exception as exc:
        row["production_error"] = repr(exc)
    try:
        roots, seconds = limited(lambda: stationary_points(x, 1.))
        row.update(production_roots=roots.tolist(), enumeration_seconds=seconds,
                   root_count_matches=len(roots) == len(oracle["roots"]))
        row["max_root_location_error"] = float(np.max(abs(roots-np.array(oracle["roots"])))) if len(roots) == len(oracle["roots"]) else None
        row["max_root_score_residual"] = max(abs(score(x, z, 1.)) for z in roots)
    except Exception as exc:
        row["enumeration_error"] = repr(exc)
    return row


def plot(out, ac):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.linspace(2.05, 4.5, 700)
    aa = a*a
    mask = aa >= 5
    vm = (3*aa[mask]-5-2*np.sqrt(aa[mask]*(aa[mask]-5)))/5
    vp = (3*aa[mask]-5+2*np.sqrt(aa[mask]*(aa[mask]-5)))/5
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].plot(a, np.zeros_like(a), color="black", label="central local maximum")
    for sign in (-1, 1):
        axes[0].plot(a[mask], sign*np.sqrt(vp), color="tab:blue", label="outer local maxima" if sign == 1 else None)
        axes[0].plot(a[mask], sign*np.sqrt(vm), color="tab:gray", linestyle=":", label="intervening minima" if sign == 1 else None)
    selected = np.zeros_like(a)
    selected[mask] = np.where(a[mask] < ac, 0., -np.sqrt(vp))
    axes[0].plot(a, selected, color="tab:red", linestyle="--", linewidth=2, label="selected global MLE")
    axes[0].set(xlabel="normalized a", ylabel="stationary location s")
    axes[0].legend(fontsize=8, loc="upper left")
    for z, label, color, linestyle in ((vp, "outer maxima relative to zero", "tab:blue", "-"),
                                      (vm, "intervening minima relative to zero", "tab:gray", ":")):
        p = z*z+2*(1-aa[mask])*z+(1+aa[mask])**2
        gain = 4*np.log1p(aa[mask])-np.log1p(z)-2*np.log(p)
        axes[1].plot(a[mask], gain, label=label, color=color, linestyle=linestyle)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(xlabel="normalized a", ylabel="log likelihood gain (k=1)")
    axes[1].legend(fontsize=8, loc="upper left")
    for axis in axes:
        axis.axvline(np.sqrt(5.), color="gray", linestyle=":")
        axis.axvline(ac, color="tab:red", linestyle="--")
    fig.suptitle("n=5: stationary modes are born before the global selection switches")
    fig.savefig(out/"selection_boundary.png", dpi=180)
    plt.close(fig)
    save(out/"plot_provenance.json", {"display_only": True, "grid": [2.05, 4.5, 700],
         "formula": "outer log gain = 4log(1+A)-log(1+v)-2log(P(v)); k=1",
         "critical_line_source": "certified algebraic root rounded for display"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"docs/research/evidence/student_n5_boundary_2026-10-10")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"algebra.json").exists():
        raise FileExistsError("use a fresh output directory; historical evidence is protected")
    start = time.monotonic()
    cert = theory.algebraic_certificate()
    save(out/"algebra.json", cert)
    ac = float(cert["decimal_approximations"]["a_c"])
    cases = datasets(ac)
    save(out/"config.json", {"cases": cases, "randomness": "none; all datasets deterministic",
         "per_operation_seconds": 20, "post_import_total_ceiling_seconds": 180,
         "location_tolerance": 2e-13, "production_modifications": False,
         "exact_tie_policy": "algebraic oracle only; rounded dataset is separately classified"})
    rows = []
    for case in cases:
        if time.monotonic()-start > 180:
            save(out/"failure.json", {"reason": "bounded task time ceiling", "completed": len(rows)})
            raise TimeoutError("bounded task time ceiling")
        row = audit(case)
        rows.append(row)
        save(out/"selector_audit.json", rows)
        print(case["label"], row.get("production_selected", row.get("production_error")),
              "oracle", row.get("oracle", {}).get("selected"),
              "roots", len(row.get("production_roots", [])), flush=True)
    failure_test = {}
    try:
        global_mle([-4., -4., 0., 4., 4.], 1., location_tol=1e-30)
        failure_test["unexpected_success"] = True
    except StudentMLEError as exc:
        failure_test = {"error": str(exc), "reason": exc.reason, "expected": "output_precision"}
    save(out/"explicit_failure.json", failure_test)
    plot(out, ac)
    sources = [Path(__file__), ROOT/"reporting/diagnostics/student_n5_boundary.py",
               ROOT/"models/loc/student/mle.py", ROOT/"scripts/selected_mle_reference_oracle.py"]
    errors = [{"label": r["label"], "errors": {k: val for k, val in r.items() if k.endswith("error") and val is not None}}
              for r in rows if any(k in r for k in ("oracle_error", "production_error", "enumeration_error"))]
    save(out/"execution.json", {"seconds": time.monotonic()-start, "binary_datasets": len(rows),
          "numerical_failures": errors,
          "wrong_global_selections": [r["label"] for r in rows if r.get("global_winner_matches") is False],
          "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})


if __name__ == "__main__":
    main()
