"""Tune Student-location Gibbs and RATTLE using acceptance rates only.

The tuner runs independent pilot chains and minimizes absolute distance from
user-supplied target acceptance rates.  It does not calculate or rank by ESS,
ESJD, runtime, or posterior summaries.  RATTLE ``num_steps`` is fixed for a
run: choose it to control trajectory length, then tune ``step_size`` here.

Example:
    .venv/bin/python scripts/tune_student_acceptance.py \
        --n-values 10,50,200,500 --seeds 0,1 \
        --rattle-num-steps 5 --out results/student_acceptance_tuning
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.loc.student import score_gibbs as loc_student
from models.loc.student import score_rattle as loc_student_rattle


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _methods(text: str) -> list[str]:
    methods = [part.strip().lower() for part in text.split(",") if part.strip()]
    unknown = set(methods) - {"gibbs", "rattle"}
    if not methods or unknown:
        raise argparse.ArgumentTypeError(f"methods must be gibbs,rattle; got {sorted(unknown)}")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=_methods, default=["gibbs", "rattle"])
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--n-values", type=_ints, default=[10, 50, 200, 500])
    parser.add_argument("--seeds", type=_ints, default=[0, 1])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-std", type=float, default=10.0)
    parser.add_argument("--initialization", choices=("central", "tail_heavy", "random"), default="central")
    parser.add_argument("--mu-proposal-scales", type=_floats, default=[0.05, 0.1, 0.2, 0.4, 0.8])
    parser.add_argument("--gibbs-z-proposal-scales", type=_floats, default=[0.005, 0.01, 0.02, 0.04, 0.08])
    parser.add_argument("--rattle-step-sizes", type=_floats, default=[0.025, 0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--rattle-num-steps", type=int, default=5)
    parser.add_argument("--target-mu-acceptance", type=float, default=0.44)
    parser.add_argument("--target-gibbs-z-acceptance", type=float, default=0.44)
    parser.add_argument("--target-rattle-acceptance", type=float, default=0.65)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _base_params(args: argparse.Namespace, n: int, proposal_std_mu: float) -> dict:
    return {
        "n": n,
        "k": args.k,
        "num_iterations_T": args.iterations,
        "proposal_std_mu": proposal_std_mu,
        "prior_mean": args.prior_mean,
        "prior_std": args.prior_std,
        "initialization": args.initialization,
        "store_x_chain": False,
    }


def _key(seed: int, n: int) -> np.ndarray:
    return np.array([np.uint32(seed), np.uint32(n)], dtype=np.uint32)


def _run_gibbs(
    args: argparse.Namespace, n: int, seed: int, proposal_std_mu: float, proposal_std_z: float
) -> dict:
    params = {
        **_base_params(args, n, proposal_std_mu),
        "proposal_std_z": proposal_std_z,
        "gibbs_backend": "numba_full",
    }
    result = loc_student.run_gibbs(_key(seed, n), args.mu_star, params, verbose=False)
    return {
        "method": "gibbs",
        "n": n,
        "seed": seed,
        "proposal_std_mu": proposal_std_mu,
        "proposal_std_z": proposal_std_z,
        "rattle_step_size": "",
        "rattle_num_steps": "",
        "mu_acceptance_rate": float(result["mu_acceptance_rate"]),
        "latent_acceptance_rate": float(result["z_acceptance_rate"]),
        "numerically_healthy": True,
        "projection_failure_count": 0,
        "reverse_check_failure_count": 0,
    }


def _run_rattle(
    args: argparse.Namespace, n: int, seed: int, proposal_std_mu: float, step_size: float
) -> dict:
    params = {
        **_base_params(args, n, proposal_std_mu),
        "rattle_step_size": step_size,
        "rattle_num_steps": args.rattle_num_steps,
        "rattle_proj_tol": 1e-10,
        "rattle_proj_max_iters": 25,
        "rattle_reverse_position_tol": 5e-3,
        "rattle_projection_mode": "paper_fixed_direction",
        "rattle_include_gram_correction": True,
        "reverse_check": True,
    }
    result = loc_student_rattle.run_rattle(_key(seed, n), args.mu_star, params, verbose=False)
    diagnostics = result["projection_diagnostics"]
    healthy = (
        diagnostics["projection_failure_count"] == 0
        and diagnostics["reverse_check_failure_count"] == 0
        and diagnostics["max_projection_residual"] <= 1e-8
        and diagnostics["max_reverse_position_error"] <= 5e-3
    )
    return {
        "method": "rattle",
        "n": n,
        "seed": seed,
        "proposal_std_mu": proposal_std_mu,
        "proposal_std_z": "",
        "rattle_step_size": step_size,
        "rattle_num_steps": args.rattle_num_steps,
        "mu_acceptance_rate": float(result["mu_acceptance_rate"]),
        "latent_acceptance_rate": float(result["x_acceptance_rate"]),
        "numerically_healthy": healthy,
        "projection_failure_count": int(diagnostics["projection_failure_count"]),
        "reverse_check_failure_count": int(diagnostics["reverse_check_failure_count"]),
    }


def _aggregate(rows: list[dict], target_mu: float, target_gibbs_z: float, target_rattle: float) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["method"], row["n"], row["proposal_std_mu"],
            row["proposal_std_z"], row["rattle_step_size"], row["rattle_num_steps"],
        )
        grouped[key].append(row)

    output = []
    for key, group in grouped.items():
        method, n, sigma_mu, sigma_z, step_size, num_steps = key
        mean_mu = float(np.mean([row["mu_acceptance_rate"] for row in group]))
        mean_latent = float(np.mean([row["latent_acceptance_rate"] for row in group]))
        latent_target = target_gibbs_z if method == "gibbs" else target_rattle
        output.append(
            {
                "method": method,
                "n": n,
                "proposal_std_mu": sigma_mu,
                "proposal_std_z": sigma_z,
                "rattle_step_size": step_size,
                "rattle_num_steps": num_steps,
                "mean_mu_acceptance_rate": mean_mu,
                "mean_latent_acceptance_rate": mean_latent,
                "target_mu_acceptance_rate": target_mu,
                "target_latent_acceptance_rate": latent_target,
                "acceptance_score": abs(mean_mu - target_mu) + abs(mean_latent - latent_target),
                "all_numerically_healthy": all(row["numerically_healthy"] for row in group),
                "total_projection_failures": sum(row["projection_failure_count"] for row in group),
                "total_reverse_check_failures": sum(row["reverse_check_failure_count"] for row in group),
            }
        )
    return sorted(output, key=lambda row: (row["method"], row["n"], row["acceptance_score"]))


def _select(aggregate: list[dict]) -> list[dict]:
    selected = []
    for method, n in sorted({(row["method"], row["n"]) for row in aggregate}):
        eligible = [
            row for row in aggregate
            if row["method"] == method and row["n"] == n and row["all_numerically_healthy"]
        ]
        if eligible:
            selected.append(min(eligible, key=lambda row: row["acceptance_score"]))
    return selected


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_settings(path: Path, selected: list[dict], args: argparse.Namespace) -> None:
    by_key = {}
    for row in selected:
        setting = {"proposal_std_mu": row["proposal_std_mu"]}
        if row["method"] == "gibbs":
            setting["proposal_std_z"] = row["proposal_std_z"]
        else:
            setting["rattle_step_size"] = row["rattle_step_size"]
            setting["rattle_num_steps"] = row["rattle_num_steps"]
        by_key[f"{row['method']}:student_t:k={args.k:g}:n={row['n']}"] = setting
    payload = {
        "description": "Student-location settings selected only by distance to target acceptance rates.",
        "targets": {
            "mu_acceptance_rate": args.target_mu_acceptance,
            "gibbs_z_acceptance_rate": args.target_gibbs_z_acceptance,
            "rattle_x_acceptance_rate": args.target_rattle_acceptance,
        },
        "by_key": by_key,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _validate(args: argparse.Namespace) -> None:
    if args.iterations <= 0 or args.k <= 0.0 or args.prior_std <= 0.0:
        raise ValueError("iterations, k, and prior_std must be positive")
    if not args.n_values or any(n <= 1 for n in args.n_values):
        raise ValueError("n-values must contain integers greater than one")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    grids = [args.mu_proposal_scales, args.gibbs_z_proposal_scales, args.rattle_step_sizes]
    if any(not grid or any(not math.isfinite(x) or x <= 0.0 for x in grid) for grid in grids):
        raise ValueError("all proposal grids must contain finite positive values")
    if args.rattle_num_steps <= 0:
        raise ValueError("rattle-num-steps must be positive")
    targets = [args.target_mu_acceptance, args.target_gibbs_z_acceptance, args.target_rattle_acceptance]
    if any(not 0.0 < target < 1.0 for target in targets):
        raise ValueError("acceptance targets must lie strictly between zero and one")


def main() -> None:
    args = parse_args()
    _validate(args)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in args.methods:
        for n in args.n_values:
            if method == "gibbs":
                settings = (
                    (sigma_mu, sigma_z)
                    for sigma_mu in args.mu_proposal_scales
                    for sigma_z in args.gibbs_z_proposal_scales
                )
            else:
                settings = (
                    (sigma_mu, step_size)
                    for sigma_mu in args.mu_proposal_scales
                    for step_size in args.rattle_step_sizes
                )
            for first, second in settings:
                for seed in args.seeds:
                    row = (
                        _run_gibbs(args, n, seed, first, second)
                        if method == "gibbs"
                        else _run_rattle(args, n, seed, first, second)
                    )
                    rows.append(row)
                    print(
                        f"{method} n={n} seed={seed} mu_scale={first:g} "
                        f"latent_scale={second:g} mu_acc={row['mu_acceptance_rate']:.3f} "
                        f"latent_acc={row['latent_acceptance_rate']:.3f} "
                        f"healthy={row['numerically_healthy']}",
                        flush=True,
                    )

    aggregate = _aggregate(
        rows, args.target_mu_acceptance, args.target_gibbs_z_acceptance, args.target_rattle_acceptance
    )
    selected = _select(aggregate)
    _write_csv(args.out / "seed_results.csv", rows)
    _write_csv(args.out / "aggregate_results.csv", aggregate)
    _write_csv(args.out / "selected_results.csv", selected)
    _write_settings(args.out / "selected_settings.json", selected, args)
    for row in selected:
        latent_scale = row["proposal_std_z"] if row["method"] == "gibbs" else row["rattle_step_size"]
        print(
            f"selected {row['method']} n={row['n']}: mu_scale={row['proposal_std_mu']:g}, "
            f"latent_scale={latent_scale:g}, mu_acc={row['mean_mu_acceptance_rate']:.3f}, "
            f"latent_acc={row['mean_latent_acceptance_rate']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
