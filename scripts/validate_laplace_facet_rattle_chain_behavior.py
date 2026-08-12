"""Chain-behavior validation for experimental odd-n Laplace fixed-facet HMC/RATTLE."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import jax.random as random
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import loc_laplace, loc_laplace_rattle
from reporting.diagnostics.audit_reference_all_models import laplace_odd_median_reference


DEFAULT_STEP_GRID = {
    11: [0.03, 0.06, 0.10, 0.15],
    21: [0.015, 0.03, 0.06, 0.10],
    51: [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03],
}


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=_ints, default=[11, 21, 51])
    parser.add_argument("--seeds", type=_ints, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--burnin", type=int, default=2000)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--proposal-std-mu", type=float, default=0.2)
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-std", type=float, default=10.0)
    parser.add_argument("--grid-size", type=int, default=4000)
    parser.add_argument("--kink-tol", type=float, default=1e-8)
    parser.add_argument("--reverse-position-tol", type=float, default=5e-3)
    parser.add_argument("--reverse-momentum-tol", type=float, default=5e-3)
    parser.add_argument("--out-dir", type=Path, default=Path("results/laplace_facet_rattle_chain_behavior"))
    parser.add_argument("--force", action="store_true", help="Rerun cases already present in per_chain_summaries.csv.")
    parser.add_argument("--skip-gibbs", action="store_true")
    return parser.parse_args()


def autocorr_fft(values: np.ndarray, max_lag: int = 200) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.ones(1)
    centered = values - float(np.mean(values))
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return np.ones(1)
    n = centered.size
    size = 1 << (2 * n - 1).bit_length()
    freq = np.fft.rfft(centered, size)
    acov = np.fft.irfft(freq * np.conjugate(freq), size)[:n]
    return acov[: min(int(max_lag) + 1, n)] / denom


def ess_stats(values: np.ndarray, max_lag: int = 200) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or float(np.std(values)) <= 0.0:
        return {
            "ess_mu": np.nan,
            "iact_mu": np.nan,
            "acf_lag1_mu": np.nan,
            "acf_lag5_mu": np.nan,
            "acf_lag10_mu": np.nan,
            "acf_lag50_mu": np.nan,
        }
    acf = autocorr_fft(values, max_lag=max_lag)
    positive = []
    for val in acf[1:]:
        if val <= 0.0:
            break
        positive.append(float(val))
    iact = max(1.0, 1.0 + 2.0 * float(np.sum(positive)))
    return {
        "ess_mu": float(values.size / iact),
        "iact_mu": float(iact),
        "acf_lag1_mu": float(acf[1]) if acf.size > 1 else np.nan,
        "acf_lag5_mu": float(acf[5]) if acf.size > 5 else np.nan,
        "acf_lag10_mu": float(acf[10]) if acf.size > 10 else np.nan,
        "acf_lag50_mu": float(acf[50]) if acf.size > 50 else np.nan,
    }


def chain_behavior_summary(chain: np.ndarray, burnin: int, elapsed_seconds: float) -> dict:
    samples = np.asarray(chain, dtype=float)[int(burnin) :]
    mid = samples.size // 2
    first = samples[:mid]
    second = samples[mid:]
    ess = ess_stats(samples)
    ess_per_sec = float(ess["ess_mu"]) / max(float(elapsed_seconds), 1e-12) if np.isfinite(ess["ess_mu"]) else np.nan
    sd_first = float(np.std(first, ddof=1)) if first.size > 1 else np.nan
    sd_second = float(np.std(second, ddof=1)) if second.size > 1 else np.nan
    return {
        "mean": float(np.mean(samples)),
        "sd": float(np.std(samples, ddof=1)),
        "q025": float(np.quantile(samples, 0.025)),
        "q50": float(np.quantile(samples, 0.5)),
        "q975": float(np.quantile(samples, 0.975)),
        "num_samples": int(samples.size),
        "split1_mean": float(np.mean(first)) if first.size else np.nan,
        "split2_mean": float(np.mean(second)) if second.size else np.nan,
        "split_mean_drift": float(np.mean(second) - np.mean(first)) if first.size and second.size else np.nan,
        "split_abs_mean_drift": abs(float(np.mean(second) - np.mean(first))) if first.size and second.size else np.nan,
        "split_sd_ratio": sd_second / sd_first if sd_first and np.isfinite(sd_first) and sd_first > 0.0 else np.nan,
        "ess_per_sec_mu": ess_per_sec,
        **ess,
    }


def add_reference_diffs(row: dict, reference: dict) -> dict:
    out = dict(row)
    for key in ["mean", "sd", "q025", "q50", "q975"]:
        out[f"analytic_{key}"] = float(reference[key])
        out[f"diff_analytic_{key}"] = float(out[key]) - float(reference[key])
        out[f"abs_diff_analytic_{key}"] = abs(float(out[f"diff_analytic_{key}"]))
    return out


def scalar_diagnostics(diag: dict, iterations: int) -> dict:
    out = {}
    for key, value in diag.items():
        if isinstance(value, (bool, np.bool_)):
            out[key] = int(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
    denom = max(int(iterations), 1)
    for count_key in [
        "side_boundary_violation_count",
        "side_boundary_cross_count",
        "kink_cross_count",
        "near_kink_count",
        "reverse_check_failure_count",
    ]:
        if count_key in out:
            out[count_key.replace("_count", "_rate")] = float(out[count_key]) / denom
    return out


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = sorted(row)
    if exists:
        old = pd.read_csv(path, nrows=0)
        fieldnames = sorted(set(old.columns) | set(fieldnames))
        if list(old.columns) != fieldnames:
            df = pd.read_csv(path)
            for col in fieldnames:
                if col not in df:
                    df[col] = np.nan
            df[fieldnames].to_csv(path, index=False)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, np.nan) for key in fieldnames})


def case_done(existing: pd.DataFrame, method: str, n: int, seed: int, step_size: float, iterations: int, burnin: int) -> bool:
    if existing.empty:
        return False
    step = -1.0 if np.isnan(step_size) else float(step_size)
    have = existing.copy()
    have_step = have["step_size"].fillna(-1.0).astype(float)
    mask = (
        have["method"].astype(str).eq(method)
        & have["n"].astype(int).eq(int(n))
        & have["seed"].astype(int).eq(int(seed))
        & np.isclose(have_step, step)
        & have["iterations"].astype(int).eq(int(iterations))
        & have["burnin"].astype(int).eq(int(burnin))
    )
    return bool(mask.any())


def write_reference(path: Path, references: dict[int, dict], args: argparse.Namespace) -> None:
    rows = []
    for n, reference in references.items():
        rows.append({
            "method": "analytic_odd_median",
            "n": int(n),
            "mu_star": float(args.mu_star),
            "b": float(args.b),
            "prior_mean": float(args.prior_mean),
            "prior_std": float(args.prior_std),
            **{key: reference[key] for key in ["mean", "sd", "q025", "q50", "q975", "marginal_likelihood_estimate"]},
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def aggregate_outputs(out_dir: Path) -> None:
    per_chain_path = out_dir / "per_chain_summaries.csv"
    diagnostics_path = out_dir / "diagnostics.csv"
    if not per_chain_path.exists():
        return
    chains = pd.read_csv(per_chain_path)
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    group_cols = ["method", "n", "step_size"]
    chains["step_size"] = chains["step_size"].fillna(-1.0)
    numeric_cols = [
        "mean",
        "sd",
        "q025",
        "q50",
        "q975",
        "abs_diff_analytic_mean",
        "abs_diff_analytic_sd",
        "ess_mu",
        "ess_per_sec_mu",
        "iact_mu",
        "acf_lag1_mu",
        "acf_lag10_mu",
        "acf_lag50_mu",
        "split_abs_mean_drift",
        "elapsed_seconds",
        "latent_acceptance_rate",
        "mu_acceptance_rate",
    ]
    agg_spec = {}
    for col in numeric_cols:
        if col in chains:
            agg_spec[f"{col}_mean"] = (col, "mean")
            agg_spec[f"{col}_median"] = (col, "median")
            agg_spec[f"{col}_min"] = (col, "min")
            agg_spec[f"{col}_max"] = (col, "max")
    setting = chains.groupby(group_cols, dropna=False).agg(
        chains=("seed", "count"),
        seed_mean_sd=("mean", "std"),
        seed_mean_range=("mean", lambda x: float(np.max(x) - np.min(x))),
        **agg_spec,
    ).reset_index()
    setting["step_size"] = setting["step_size"].replace(-1.0, np.nan)

    if not diagnostics.empty:
        diagnostics["step_size"] = diagnostics["step_size"].fillna(-1.0)
        diag_cols = [
            "side_boundary_violation_rate",
            "side_boundary_cross_rate",
            "kink_cross_rate",
            "near_kink_rate",
            "reverse_check_failure_rate",
            "delta_h_mean",
            "delta_h_median",
            "delta_h_q95",
            "delta_h_max_abs",
            "mean_delta_H_if_crossed",
            "median_delta_H_if_crossed",
            "q95_delta_H_if_crossed",
            "max_delta_H_if_crossed",
            "mean_delta_H_if_not_crossed",
            "median_delta_H_if_not_crossed",
            "q95_delta_H_if_not_crossed",
            "max_delta_H_if_not_crossed",
            "reverse_fail_if_crossed",
            "reverse_fail_if_not_crossed",
            "median_residual",
            "side_count_failures",
        ]
        diag_spec = {f"{col}_mean": (col, "mean") for col in diag_cols if col in diagnostics}
        diag_agg = diagnostics.groupby(["method", "n", "step_size"], dropna=False).agg(**diag_spec).reset_index()
        setting = setting.merge(diag_agg, on=["method", "n", "step_size"], how="left")
        setting["step_size"] = setting["step_size"].replace(-1.0, np.nan)

    setting.to_csv(out_dir / "per_setting_aggregated_summaries.csv", index=False)

    gibbs = setting[setting["method"].eq("gibbs")][["n", "ess_per_sec_mu_median", "ess_mu_median", "abs_diff_analytic_mean_mean", "abs_diff_analytic_sd_mean"]]
    gibbs = gibbs.rename(columns={
        "ess_per_sec_mu_median": "gibbs_ess_per_sec_mu_median",
        "ess_mu_median": "gibbs_ess_mu_median",
        "abs_diff_analytic_mean_mean": "gibbs_abs_diff_analytic_mean_mean",
        "abs_diff_analytic_sd_mean": "gibbs_abs_diff_analytic_sd_mean",
    })
    rattle = setting[setting["method"].eq("experimental_facet_rattle")].copy()
    comparison = rattle.merge(gibbs, on="n", how="left")
    comparison["rattle_over_gibbs_ess_sec"] = comparison["ess_per_sec_mu_median"] / comparison["gibbs_ess_per_sec_mu_median"]
    comparison["rattle_over_gibbs_ess"] = comparison["ess_mu_median"] / comparison["gibbs_ess_mu_median"]
    comparison.to_csv(out_dir / "method_comparison.csv", index=False)


def write_readme(out_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Experimental Laplace Fixed-Facet HMC/RATTLE Chain Behavior",
        "",
        "This is a diagnostic-only local validation run. It does not promote Laplace RATTLE in `model_registry`.",
        "",
        "## Grid",
        "",
        f"- n values: `{args.n_values}`",
        f"- seeds: `{args.seeds}`",
        f"- iterations: `{args.iterations}` with burn-in `{args.burnin}`",
        f"- leapfrog steps: `{args.num_steps}`",
        f"- step-size grid: `{DEFAULT_STEP_GRID}`",
        "",
        "## Command",
        "",
        "```bash",
        "python scripts/validate_laplace_facet_rattle_chain_behavior.py",
        "```",
        "",
        "The script is resumable: existing method/n/seed/step-size rows with matching iteration and burn-in settings are skipped unless `--force` is passed.",
        "",
        "## Outputs",
        "",
        "- `analytic_reference.csv`: analytic odd-n Laplace median posterior reference.",
        "- `per_chain_summaries.csv`: posterior summaries, ESS/autocorrelation, split drift, and timing per chain.",
        "- `diagnostics.csv`: RATTLE latent diagnostics per chain.",
        "- `per_setting_aggregated_summaries.csv`: seed-aggregated summaries by setting.",
        "- `method_comparison.csv`: RATTLE-vs-Gibbs ESS/sec and reference-error comparison.",
        "",
        "## Preliminary Interpretation Template",
        "",
        "Check whether small-step scaling improves `n=51` ESS/sec without making side-boundary rejection dominate. Treat kink diagnostics separately from side-boundary diagnostics; the sampler remains experimental until posterior agreement, ESS, reverse checks, and boundary behavior are all acceptable.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.burnin >= args.iterations:
        raise SystemExit("--burnin must be smaller than --iterations")
    if len(args.seeds) < 3:
        raise SystemExit("Use at least 3 seeds for this validation experiment")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_chain_path = args.out_dir / "per_chain_summaries.csv"
    diagnostics_path = args.out_dir / "diagnostics.csv"
    existing = pd.read_csv(per_chain_path) if per_chain_path.exists() and not args.force else pd.DataFrame()

    references = {}
    for n in args.n_values:
        references[int(n)] = laplace_odd_median_reference(
            n=int(n),
            mu_star=args.mu_star,
            prior_mean=args.prior_mean,
            prior_std=args.prior_std,
            laplace_b=args.b,
            grid_size=args.grid_size,
        )
        print(f"[reference] n={n} mean={references[int(n)]['mean']:.6g} sd={references[int(n)]['sd']:.6g}", flush=True)
    write_reference(args.out_dir / "analytic_reference.csv", references, args)

    base_params = {
        "b": float(args.b),
        "num_iterations_T": int(args.iterations),
        "proposal_std_mu": float(args.proposal_std_mu),
        "prior_mean": float(args.prior_mean),
        "prior_std": float(args.prior_std),
    }

    for n in args.n_values:
        n = int(n)
        step_grid = DEFAULT_STEP_GRID[n]
        for seed in args.seeds:
            seed = int(seed)
            if not args.skip_gibbs and not case_done(existing, "gibbs", n, seed, np.nan, args.iterations, args.burnin):
                params = {**base_params, "n": n}
                start = time.perf_counter()
                out = loc_laplace.run_gibbs(random.PRNGKey(seed), args.mu_star, params, verbose=False)
                elapsed = time.perf_counter() - start
                row = {
                    "method": "gibbs",
                    "n": n,
                    "seed": seed,
                    "step_size": np.nan,
                    "num_steps": np.nan,
                    "iterations": int(args.iterations),
                    "burnin": int(args.burnin),
                    "elapsed_seconds": float(elapsed),
                    "mu_acceptance_rate": float(out["mu_acceptance_rate"]),
                    "latent_acceptance_rate": np.nan,
                }
                row.update(chain_behavior_summary(np.asarray(out["mu_chain"]), args.burnin, elapsed))
                append_row(per_chain_path, add_reference_diffs(row, references[n]))
                existing = pd.read_csv(per_chain_path)
                print(
                    f"[gibbs] n={n} seed={seed} elapsed={elapsed:.2f}s "
                    f"ess={row['ess_mu']:.1f} ess/sec={row['ess_per_sec_mu']:.2f}",
                    flush=True,
                )

            for step_size in step_grid:
                if case_done(existing, "experimental_facet_rattle", n, seed, step_size, args.iterations, args.burnin):
                    print(f"[skip] rattle n={n} seed={seed} eps={step_size:g}", flush=True)
                    continue
                params = {
                    **base_params,
                    "n": n,
                    "rattle_step_size": float(step_size),
                    "rattle_num_steps": int(args.num_steps),
                    "reverse_check": True,
                    "rattle_reverse_position_tol": float(args.reverse_position_tol),
                    "rattle_reverse_momentum_tol": float(args.reverse_momentum_tol),
                    "kink_tol": float(args.kink_tol),
                    "laplace_rattle_experimental": True,
                }
                start = time.perf_counter()
                out = loc_laplace_rattle.run_rattle(random.PRNGKey(seed), args.mu_star, params, verbose=False)
                elapsed = time.perf_counter() - start
                diag = scalar_diagnostics(out["projection_diagnostics"], args.iterations)
                row = {
                    "method": "experimental_facet_rattle",
                    "n": n,
                    "seed": seed,
                    "step_size": float(step_size),
                    "num_steps": int(args.num_steps),
                    "iterations": int(args.iterations),
                    "burnin": int(args.burnin),
                    "elapsed_seconds": float(elapsed),
                    "mu_acceptance_rate": float(out["mu_acceptance_rate"]),
                    "latent_acceptance_rate": float(out["latent_acceptance_rate"]),
                }
                row.update(chain_behavior_summary(np.asarray(out["mu_chain"]), args.burnin, elapsed))
                append_row(per_chain_path, add_reference_diffs(row, references[n]))
                append_row(diagnostics_path, {
                    "method": "experimental_facet_rattle",
                    "n": n,
                    "seed": seed,
                    "step_size": float(step_size),
                    "num_steps": int(args.num_steps),
                    "iterations": int(args.iterations),
                    **diag,
                })
                existing = pd.read_csv(per_chain_path)
                print(
                    f"[rattle] n={n} seed={seed} eps={step_size:g} elapsed={elapsed:.2f}s "
                    f"lat_acc={row['latent_acceptance_rate']:.3f} ess={row['ess_mu']:.1f} "
                    f"ess/sec={row['ess_per_sec_mu']:.2f} side_fail_rate={diag.get('side_boundary_violation_rate', np.nan):.3f} "
                    f"kink_rate={diag.get('kink_cross_rate', np.nan):.3f}",
                    flush=True,
                )

    aggregate_outputs(args.out_dir)
    write_readme(args.out_dir, args)
    metadata = {
        "iterations": int(args.iterations),
        "burnin": int(args.burnin),
        "n_values": [int(n) for n in args.n_values],
        "seeds": [int(seed) for seed in args.seeds],
        "step_grid": DEFAULT_STEP_GRID,
        "experimental_nonsmooth_facet_rattle": True,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] wrote chain-behavior outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
