"""Measure global-MLE occupancy and endpoint crossings on the score manifold.

For every retained transition this records the signed likelihood margin against
the best competing local maximum, raw and score-space jump distance, and the
number of inverse-branch labels that changed.  It deliberately distinguishes
stationary occupancy from transition frequency.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.loc.student import score_gibbs as loc_student
from models.loc.student import score_rattle as loc_student_rattle
from models.loc.student.mle import loglik, stationary_points


def _ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def _floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=None, help="Single-k shorthand overriding --k-values")
    parser.add_argument("--n", type=int, default=None, help="Single-n shorthand overriding --n-values")
    parser.add_argument("--k-values", type=_floats, default=[2.0])
    parser.add_argument("--n-values", type=_ints, default=[10])
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--seeds", type=_ints, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--proposal-std-mu", type=float, default=1.1)
    parser.add_argument("--proposal-std-z", type=float, default=0.16)
    parser.add_argument("--rattle-step-size", type=float, default=0.28)
    parser.add_argument("--rattle-num-steps", type=int, default=5)
    parser.add_argument("--likelihood-tol", type=float, default=1e-8)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _score_derivative(x: np.ndarray, mu: float, k: float) -> float:
    residual = np.asarray(x, dtype=float) - float(mu)
    denominator = k + residual * residual
    return float(np.sum((residual * residual - k) / (denominator * denominator)))


def mle_competitor_margin(
    x: np.ndarray,
    mu_star: float,
    k: float,
    likelihood_tol: float = 1e-8,
) -> dict:
    """Return D=ell(mu_star)-ell(best other local maximum).

    ``D=+inf`` means no competing local maximum exists.  Membership in the
    global-MLE region additionally requires ``mu_star`` itself to be a local
    maximum.  Distinct equal-height maxima have margin zero.
    """
    x = np.asarray(x, dtype=float)
    roots = np.asarray(stationary_points(x, k), dtype=float)
    location_tol = 1e-7 * (1.0 + abs(mu_star) + np.max(np.abs(x)))
    local_maxima = [
        float(root)
        for root in roots
        if _score_derivative(x, float(root), k) < -1e-10
    ]
    mu_star_is_local_max = _score_derivative(x, mu_star, k) < -1e-10
    competitors = [root for root in local_maxima if abs(root - mu_star) > location_tol]
    ll_star = float(loglik(x, mu_star, k))
    if competitors:
        values = np.asarray([loglik(x, root, k) for root in competitors], dtype=float)
        best_index = int(np.argmax(values))
        competitor_mu = competitors[best_index]
        competitor_loglik = float(values[best_index])
        margin = ll_star - competitor_loglik
    else:
        competitor_mu = math.nan
        competitor_loglik = -math.inf
        margin = math.inf
    return {
        "mle_margin": float(margin),
        "mu_star_is_local_max": bool(mu_star_is_local_max),
        "mu_star_is_global_mle": bool(mu_star_is_local_max and margin >= -likelihood_tol),
        "best_competing_mu": float(competitor_mu),
        "loglik_at_mu_star": ll_star,
        "best_competing_loglik": float(competitor_loglik),
        "num_stationary_points": int(roots.size),
        "num_local_maxima": int(len(local_maxima)),
    }


def _chain(args: argparse.Namespace, method: str, k: float, n: int, seed: int) -> dict:
    params = {
        "k": k,
        "n": n,
        "num_iterations_T": args.iterations,
        "proposal_std_mu": args.proposal_std_mu,
        "proposal_std_z": args.proposal_std_z,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "gibbs_backend": "numba_full",
        "initialization": "central",
        "store_x_chain": True,
        "rattle_step_size": args.rattle_step_size,
        "rattle_num_steps": args.rattle_num_steps,
    }
    key = np.array([seed, n], dtype=np.uint32)
    if method == "gibbs":
        return loc_student.run_gibbs(key, args.mu_star, params, verbose=False)
    return loc_student_rattle.run_rattle(key, args.mu_star, params, verbose=False)


def _transition_rows(
    args: argparse.Namespace,
    method: str,
    k: float,
    n: int,
    seed: int,
    chain: dict,
) -> list[dict]:
    states = np.asarray(chain["x_chain"], dtype=float)[args.burn_in :]
    classifications = [
        mle_competitor_margin(x, args.mu_star, k, args.likelihood_tol)
        for x in states
    ]
    rows = []
    sqrt_k = math.sqrt(k)
    for index in range(states.shape[0] - 1):
        before = states[index]
        after = states[index + 1]
        before_class = classifications[index]
        after_class = classifications[index + 1]
        before_valid = before_class["mu_star_is_global_mle"]
        after_valid = after_class["mu_star_is_global_mle"]
        raw_sjd = float(np.mean((after - before) ** 2))
        z_before = (before - args.mu_star) / (k + (before - args.mu_star) ** 2)
        z_after = (after - args.mu_star) / (k + (after - args.mu_star) ** 2)
        rows.append(
            {
                "method": method,
                "k": k,
                "n": n,
                "seed": seed,
                "iteration": args.burn_in + index,
                "mle_valid_before": before_valid,
                "mle_valid_after": after_valid,
                "mle_exit": bool(before_valid and not after_valid),
                "mle_return": bool(not before_valid and after_valid),
                "mle_margin_before": before_class["mle_margin"],
                "mle_margin_after": after_class["mle_margin"],
                "mle_margin_change": after_class["mle_margin"] - before_class["mle_margin"],
                "raw_x_sjd_per_coordinate": raw_sjd,
                "score_sjd_per_coordinate": float(np.mean((z_after - z_before) ** 2)),
                "branch_switch_count": int(np.sum((np.abs(before - args.mu_star) > sqrt_k) != (np.abs(after - args.mu_star) > sqrt_k))),
                "x_changed": bool(raw_sjd > 0.0),
                "num_local_maxima_before": before_class["num_local_maxima"],
                "num_local_maxima_after": after_class["num_local_maxima"],
            }
        )
    return rows


def _mean(rows: list[dict], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows])) if rows else math.nan


def _summary(method: str, k: float, n: int, rows: list[dict]) -> dict:
    at_mle = [row for row in rows if row["mle_valid_before"]]
    outside = [row for row in rows if not row["mle_valid_before"]]
    exits = [row for row in rows if row["mle_exit"]]
    returns = [row for row in rows if row["mle_return"]]
    stays = [row for row in rows if not row["mle_exit"] and not row["mle_return"]]
    return {
        "method": method,
        "k": k,
        "n": n,
        "transitions": len(rows),
        "outside_occupancy": len(outside) / max(len(rows), 1),
        "mle_exit_count": len(exits),
        "mle_exit_probability": len(exits) / max(len(at_mle), 1),
        "mle_return_count": len(returns),
        "mle_return_probability": len(returns) / max(len(outside), 1),
        "mean_score_sjd_all": _mean(rows, "score_sjd_per_coordinate"),
        "mean_score_sjd_exits": _mean(exits, "score_sjd_per_coordinate"),
        "mean_score_sjd_non_crossings": _mean(stays, "score_sjd_per_coordinate"),
        "mean_raw_x_sjd_all": _mean(rows, "raw_x_sjd_per_coordinate"),
        "mean_raw_x_sjd_exits": _mean(exits, "raw_x_sjd_per_coordinate"),
        "mean_raw_x_sjd_non_crossings": _mean(stays, "raw_x_sjd_per_coordinate"),
        "mean_branch_switches_all": _mean(rows, "branch_switch_count"),
        "mean_branch_switches_exits": _mean(exits, "branch_switch_count"),
        "fraction_exits_with_branch_switch": _mean(
            [{"value": row["branch_switch_count"] > 0} for row in exits], "value"
        ),
    }


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.burn_in < args.iterations:
        raise ValueError("burn-in must be non-negative and smaller than iterations")
    args.out.mkdir(parents=True, exist_ok=True)
    k_values = [args.k] if args.k is not None else args.k_values
    n_values = [args.n] if args.n is not None else args.n_values
    rows = []
    chain_rows = []
    seed_summaries = []
    for k in k_values:
        for n in n_values:
            for method in ("gibbs", "rattle"):
                for seed in args.seeds:
                    chain = _chain(args, method, float(k), int(n), seed)
                    method_rows = _transition_rows(args, method, float(k), int(n), seed, chain)
                    rows.extend(method_rows)
                    exits = sum(row["mle_exit"] for row in method_rows)
                    seed_summary = _summary(method, float(k), int(n), method_rows)
                    seed_summary["seed"] = seed
                    seed_summaries.append(seed_summary)
                    chain_row = {
                        "method": method, "k": float(k), "n": int(n), "seed": seed,
                        "mu_acceptance_rate": float(chain["mu_acceptance_rate"]),
                        "latent_acceptance_rate": float(chain["z_acceptance_rate"] if method == "gibbs" else chain["x_acceptance_rate"]),
                        "projection_failure_count": 0,
                        "reverse_check_failure_count": 0,
                    }
                    if method == "rattle":
                        diag = chain["projection_diagnostics"]
                        chain_row["projection_failure_count"] = int(diag["projection_failure_count"])
                        chain_row["reverse_check_failure_count"] = int(diag["reverse_check_failure_count"])
                    chain_rows.append(chain_row)
                    print(
                        f"{method} k={float(k):g} n={int(n)} seed={seed}: "
                        f"{len(method_rows)} transitions, {exits} exits",
                        flush=True,
                    )
    summaries = []
    for k in k_values:
        for n in n_values:
            for method in ("gibbs", "rattle"):
                selected = [
                    row for row in rows
                    if row["method"] == method and row["k"] == float(k) and row["n"] == int(n)
                ]
                summaries.append(_summary(method, float(k), int(n), selected))
    _write(args.out / "transition_results.csv", rows)
    _write(args.out / "summary.csv", summaries)
    _write(args.out / "summary_by_seed.csv", seed_summaries)
    _write(args.out / "chain_health.csv", chain_rows)
    for row in summaries:
        print(row, flush=True)


if __name__ == "__main__":
    main()
