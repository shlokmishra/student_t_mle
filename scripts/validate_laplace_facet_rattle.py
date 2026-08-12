"""Validate experimental odd-n Laplace facet-RATTLE against Gibbs and analytic reference."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import jax.random as random
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import loc_laplace, loc_laplace_rattle
from reporting.diagnostics.audit_reference_all_models import laplace_odd_median_reference


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=_ints, default=[11, 21, 51])
    parser.add_argument("--seeds", type=_ints, default=[0, 1, 2])
    parser.add_argument("--step-sizes", type=_floats, default=[0.03, 0.06, 0.1, 0.15, 0.2, 0.3])
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--burnin", type=int, default=1000)
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--proposal-std-mu", type=float, default=0.2)
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-std", type=float, default=10.0)
    parser.add_argument("--grid-size", type=int, default=4000)
    parser.add_argument("--kink-tol", type=float, default=1e-8)
    parser.add_argument("--reverse-position-tol", type=float, default=5e-3)
    parser.add_argument("--reverse-momentum-tol", type=float, default=5e-3)
    parser.add_argument("--out-dir", type=Path, default=Path("results/laplace_facet_rattle_validation"))
    parser.add_argument("--skip-gibbs", action="store_true")
    return parser.parse_args()


def chain_summary(chain: np.ndarray, burnin: int) -> dict:
    samples = np.asarray(chain, dtype=float)[int(burnin) :]
    return {
        "mean": float(np.mean(samples)),
        "sd": float(np.std(samples, ddof=1)),
        "q025": float(np.quantile(samples, 0.025)),
        "q50": float(np.quantile(samples, 0.5)),
        "q975": float(np.quantile(samples, 0.975)),
        "num_samples": int(samples.size),
    }


def add_reference_diffs(row: dict, reference: dict) -> dict:
    out = dict(row)
    for key in ["mean", "sd", "q025", "q50", "q975"]:
        out[f"analytic_{key}"] = float(reference[key])
        out[f"diff_analytic_{key}"] = float(out[key]) - float(reference[key])
        out[f"abs_diff_analytic_{key}"] = abs(float(out[f"diff_analytic_{key}"]))
    return out


def scalar_diagnostics(diag: dict) -> dict:
    out = {}
    for key, value in diag.items():
        if isinstance(value, (bool, np.bool_)):
            out[key] = int(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.burnin >= args.iterations:
        raise SystemExit("--burnin must be smaller than --iterations")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    diagnostic_rows = []
    reference_rows = []

    references = {}
    for n in args.n_values:
        reference = laplace_odd_median_reference(
            n=n,
            mu_star=args.mu_star,
            prior_mean=args.prior_mean,
            prior_std=args.prior_std,
            laplace_b=args.b,
            grid_size=args.grid_size,
        )
        references[int(n)] = reference
        reference_rows.append({
            "method": "analytic_odd_median",
            "n": int(n),
            "mu_star": float(args.mu_star),
            "b": float(args.b),
            "prior_mean": float(args.prior_mean),
            "prior_std": float(args.prior_std),
            **{key: reference[key] for key in ["mean", "sd", "q025", "q50", "q975", "marginal_likelihood_estimate"]},
        })
        print(
            f"[reference] n={n} mean={reference['mean']:.6g} sd={reference['sd']:.6g}",
            flush=True,
        )

    base_params = {
        "b": float(args.b),
        "num_iterations_T": int(args.iterations),
        "proposal_std_mu": float(args.proposal_std_mu),
        "prior_mean": float(args.prior_mean),
        "prior_std": float(args.prior_std),
    }

    for n in args.n_values:
        n = int(n)
        for seed in args.seeds:
            seed = int(seed)
            if not args.skip_gibbs:
                params = {**base_params, "n": n}
                start = time.perf_counter()
                gibbs = loc_laplace.run_gibbs(random.PRNGKey(seed), args.mu_star, params, verbose=False)
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
                    "mu_acceptance_rate": float(gibbs["mu_acceptance_rate"]),
                }
                row.update(chain_summary(np.asarray(gibbs["mu_chain"]), args.burnin))
                summary_rows.append(add_reference_diffs(row, references[n]))
                print(
                    f"[gibbs] n={n} seed={seed} elapsed={elapsed:.2f}s "
                    f"mean={row['mean']:.6g} sd={row['sd']:.6g}",
                    flush=True,
                )

            for step_size in args.step_sizes:
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
                rattle = loc_laplace_rattle.run_rattle(random.PRNGKey(seed), args.mu_star, params, verbose=False)
                elapsed = time.perf_counter() - start
                diag = scalar_diagnostics(rattle["projection_diagnostics"])
                row = {
                    "method": "experimental_facet_rattle",
                    "n": n,
                    "seed": seed,
                    "step_size": float(step_size),
                    "num_steps": int(args.num_steps),
                    "iterations": int(args.iterations),
                    "burnin": int(args.burnin),
                    "elapsed_seconds": float(elapsed),
                    "mu_acceptance_rate": float(rattle["mu_acceptance_rate"]),
                    "latent_acceptance_rate": float(rattle["latent_acceptance_rate"]),
                }
                row.update(chain_summary(np.asarray(rattle["mu_chain"]), args.burnin))
                summary_rows.append(add_reference_diffs(row, references[n]))
                diagnostic_rows.append({
                    "n": n,
                    "seed": seed,
                    "step_size": float(step_size),
                    "num_steps": int(args.num_steps),
                    "iterations": int(args.iterations),
                    **diag,
                })
                print(
                    f"[rattle] n={n} seed={seed} eps={step_size:g} L={args.num_steps} "
                    f"elapsed={elapsed:.2f}s latent_acc={row['latent_acceptance_rate']:.3f} "
                    f"mean={row['mean']:.6g} sd={row['sd']:.6g} "
                    f"boundary_cross={diag.get('side_boundary_cross_count', 0):.0f} "
                    f"endpoint_violation={diag.get('side_boundary_violation_count', 0):.0f}",
                    flush=True,
                )

    write_rows(args.out_dir / "analytic_reference.csv", reference_rows)
    write_rows(args.out_dir / "chain_summary.csv", summary_rows)
    write_rows(args.out_dir / "rattle_diagnostics.csv", diagnostic_rows)
    print(f"[done] wrote outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
