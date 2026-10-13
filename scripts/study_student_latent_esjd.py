"""Study latent acceptance and jump distance for Student-location samplers.

This compares whole-vector x movement per retained iteration.  Alongside raw
x-space ESJD it reports bounded score-space ESJD, because Student-t latent
coordinates can make rare tail/branch jumps dominate a raw second moment.
"""

from __future__ import annotations

import argparse
import csv
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
    return [float(value) for value in text.split(",") if value.strip()]


def _ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seeds", type=_ints, default=[21, 22, 23])
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--proposal-std-mu", type=float, default=1.1)
    parser.add_argument("--gibbs-z-scales", type=_floats, default=[0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.24])
    parser.add_argument("--rattle-step-sizes", type=_floats, default=[0.05, 0.1, 0.2, 0.25, 0.28, 0.3, 0.32, 0.35, 0.4, 0.5, 0.6, 0.8])
    parser.add_argument("--rattle-num-steps", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _params(args: argparse.Namespace) -> dict:
    return {
        "n": args.n,
        "k": args.k,
        "num_iterations_T": args.iterations,
        "proposal_std_mu": args.proposal_std_mu,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "initialization": "central",
        "store_x_chain": True,
    }


def _movement(x_chain: np.ndarray, burn_in: int, mu_star: float, k: float) -> dict:
    x = np.asarray(x_chain[burn_in:], dtype=np.float64)
    dx = np.diff(x, axis=0)
    squared_jump = np.mean(dx * dx, axis=1)
    z = (x - mu_star) / (k + (x - mu_star) ** 2)
    dz = np.diff(z, axis=0)
    score_squared_jump = np.mean(dz * dz, axis=1)
    q99 = float(np.quantile(squared_jump, 0.99))
    return {
        "x_esjd_per_coordinate": float(np.mean(squared_jump)),
        "x_sjd_median": float(np.median(squared_jump)),
        "x_sjd_q95": float(np.quantile(squared_jump, 0.95)),
        "x_esjd_winsorized_q99": float(np.mean(np.minimum(squared_jump, q99))),
        "x_log_jump_mean": float(np.mean(np.log1p(squared_jump))),
        "score_esjd_per_coordinate": float(np.mean(score_squared_jump)),
        "x_move_rate": float(np.mean(squared_jump > 0.0)),
        "x_max_squared_jump": float(np.max(squared_jump)),
    }


def _run_gibbs(args: argparse.Namespace, seed: int, scale: float) -> dict:
    result = loc_student.run_gibbs(
        np.array([seed, args.n], dtype=np.uint32),
        args.mu_star,
        {**_params(args), "proposal_std_z": scale, "gibbs_backend": "numba_full"},
        verbose=False,
    )
    return {
        "method": "gibbs",
        "seed": seed,
        "latent_scale": scale,
        "latent_acceptance_rate": float(result["z_acceptance_rate"]),
        "mh_rejection_rate": 1.0 - float(result["z_acceptance_rate"]),
        "projection_failure_rate": 0.0,
        "reverse_failure_rate": 0.0,
        "delta_h_mean_abs": "",
        **_movement(result["x_chain"], args.burn_in, args.mu_star, args.k),
    }


def _run_rattle(args: argparse.Namespace, seed: int, step_size: float) -> dict:
    result = loc_student_rattle.run_rattle(
        np.array([seed, args.n], dtype=np.uint32),
        args.mu_star,
        {
            **_params(args),
            "rattle_step_size": step_size,
            "rattle_num_steps": args.rattle_num_steps,
            "rattle_projection_mode": "paper_fixed_direction",
            "rattle_include_gram_correction": True,
            "reverse_check": True,
        },
        verbose=False,
    )
    diag = result["projection_diagnostics"]
    projection_rate = float(diag["projection_failure_count"]) / args.iterations
    reverse_rate = float(diag["reverse_check_failure_count"]) / args.iterations
    acceptance = float(result["x_acceptance_rate"])
    return {
        "method": "rattle",
        "seed": seed,
        "latent_scale": step_size,
        "latent_acceptance_rate": acceptance,
        "mh_rejection_rate": max(0.0, 1.0 - acceptance - projection_rate - reverse_rate),
        "projection_failure_rate": projection_rate,
        "reverse_failure_rate": reverse_rate,
        "delta_h_mean_abs": float(diag["delta_h_mean_abs"]),
        **_movement(result["x_chain"], args.burn_in, args.mu_star, args.k),
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["latent_scale"])].append(row)
    numeric_fields = [
        key for key in rows[0]
        if key not in {"method", "seed", "latent_scale", "delta_h_mean_abs"}
    ]
    output = []
    for (method, scale), group in groups.items():
        row = {"method": method, "latent_scale": scale, "seeds": len(group)}
        for field in numeric_fields:
            values = np.asarray([item[field] for item in group], dtype=float)
            row[f"mean_{field}"] = float(np.mean(values))
            row[f"se_{field}"] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        delta_h = [float(item["delta_h_mean_abs"]) for item in group if item["delta_h_mean_abs"] != ""]
        row["mean_delta_h_mean_abs"] = float(np.mean(delta_h)) if delta_h else ""
        output.append(row)
    return sorted(output, key=lambda row: (row["method"], row["latent_scale"]))


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.burn_in < args.iterations:
        raise ValueError("burn-in must be non-negative and smaller than iterations")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scale in args.gibbs_z_scales:
        for seed in args.seeds:
            row = _run_gibbs(args, seed, scale)
            rows.append(row)
            print(f"gibbs sigma_z={scale:g} seed={seed} acc={row['latent_acceptance_rate']:.3f} score_esjd={row['score_esjd_per_coordinate']:.4g}", flush=True)
    for scale in args.rattle_step_sizes:
        for seed in args.seeds:
            row = _run_rattle(args, seed, scale)
            rows.append(row)
            print(f"rattle eps={scale:g} seed={seed} acc={row['latent_acceptance_rate']:.3f} score_esjd={row['score_esjd_per_coordinate']:.4g} fail={row['projection_failure_rate'] + row['reverse_failure_rate']:.4g}", flush=True)
    _write(args.out / "seed_results.csv", rows)
    _write(args.out / "aggregate_results.csv", _aggregate(rows))


if __name__ == "__main__":
    main()
