"""Audit the legacy wide-Brent Student location selector against production.

The old selector asked Brent's method for one score root.  For a redescending
Student score, that root need not maximize the likelihood.  Production now
isolates all maximum candidates and selects by likelihood.  This diagnostic
retains the old calculation only as an explicit comparator and independently
checks production results with multi-start EM/IRLS and dense interval search.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import jax.random as random
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.loc.student import score_gibbs as loc_student


def _ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=_floats, default=[1.0, 2.0, 3.0])
    parser.add_argument("--n-values", type=_ints, default=[10, 20, 50])
    parser.add_argument("--seeds", type=_ints, default=[0])
    parser.add_argument("--datasets", type=int, default=10_000)
    parser.add_argument("--likelihood-tol", type=float, default=1e-8)
    parser.add_argument("--dense-points-per-interval", type=int, default=25)
    parser.add_argument(
        "--validation-datasets",
        type=int,
        default=50,
        help="Dense-search cross-checks per grid cell, in addition to every detected failure.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("reporting/diagnostic_outputs/kde_reference_audit/mle_sample_cache"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/latest/student_global_mle_audit"),
    )
    return parser.parse_args()


def loglik(data: np.ndarray, mu: float | np.ndarray, k: float) -> np.ndarray:
    values = np.asarray(mu, dtype=float)
    residual = np.asarray(data, dtype=float)[:, None] - np.atleast_1d(values)[None, :]
    out = -0.5 * (float(k) + 1.0) * np.log1p(residual * residual / float(k)).sum(axis=0)
    return out if values.ndim else out[0]


def score(data: np.ndarray, mu: float, k: float) -> float:
    residual = np.asarray(data, dtype=float) - float(mu)
    return float(np.sum(residual / (float(k) + residual * residual)))


def legacy_wide_brent_score_root(data: np.ndarray, k: float) -> float:
    """Reproduce the removed one-root implementation for audit comparisons."""
    x = np.asarray(data, dtype=float)
    result = root_scalar(
        lambda mu: score(x, mu, k),
        bracket=(float(np.min(x) - 10.0), float(np.max(x) + 10.0)),
        method="brentq",
    )
    if not result.converged:
        raise RuntimeError("Legacy Student score-root calculation did not converge")
    return float(result.root)


def global_mle_multistart(data: np.ndarray, k: float) -> tuple[float, float, float, int]:
    """Find the best Student location mode using deterministic EM/IRLS starts."""
    ordered = np.sort(np.asarray(data, dtype=float))
    midpoints = 0.5 * (ordered[:-1] + ordered[1:])
    starts = np.unique(np.concatenate((ordered, midpoints)))
    start_values = loglik(ordered, starts, k)
    keep = min(16, starts.size)
    locations = starts[np.argpartition(start_values, -keep)[-keep:]].copy()

    for _ in range(500):
        residual = ordered[:, None] - locations[None, :]
        weights = 1.0 / (float(k) + residual * residual)
        updated = np.sum(ordered[:, None] * weights, axis=0) / np.sum(weights, axis=0)
        if np.max(np.abs(updated - locations)) <= 1e-12 * (1.0 + np.max(np.abs(locations))):
            locations = updated
            break
        locations = updated

    values = loglik(ordered, locations, k)
    best = int(np.argmax(values))
    best_mu = float(locations[best])
    return best_mu, float(values[best]), abs(score(ordered, best_mu, k)), int(locations.size)


def global_mle_dense(data: np.ndarray, k: float, points_per_interval: int) -> tuple[float, float]:
    """Independent one-dimensional search with endpoint-dense interval meshes."""
    ordered = np.unique(np.sort(np.asarray(data, dtype=float)))
    if ordered.size == 1:
        return float(ordered[0]), float(loglik(ordered, ordered[0], k))

    angle = np.linspace(0.0, np.pi, max(int(points_per_interval), 9))
    unit = 0.5 * (1.0 - np.cos(angle))
    grid_parts = [ordered]
    for left, right in zip(ordered[:-1], ordered[1:]):
        if right > left:
            grid_parts.append(left + (right - left) * unit)
    grid = np.unique(np.concatenate(grid_parts))
    values = loglik(ordered, grid, k)
    candidates = [(float(grid[int(np.argmax(values))]), float(np.max(values)))]

    peaks = np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])) + 1
    for index in peaks:
        left, right = float(grid[index - 1]), float(grid[index + 1])
        if right <= left:
            continue
        result = minimize_scalar(
            lambda mu: -float(loglik(ordered, mu, k)),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 200},
        )
        candidates.append((float(result.x), -float(result.fun)))
    return max(candidates, key=lambda item: item[1])


def simulation_key(seed: int):
    _, _, key_mle = random.split(random.PRNGKey(int(seed)), 3)
    return key_mle


def audit_datasets(
    data_sets: np.ndarray,
    k: float,
    likelihood_tol: float,
    dense_points: int,
    validation_datasets: int,
    case: dict,
) -> tuple[dict, list[dict]]:
    false_rows: list[dict] = []
    max_score_residual = 0.0
    dense_disagreements = 0
    dense_checked = 0
    for index, data in enumerate(np.asarray(data_sets, dtype=float)):
        params = {"k": float(k), "n": int(data.size)}
        brent_mu = legacy_wide_brent_score_root(data, k)
        brent_ll = float(loglik(data, brent_mu, k))
        global_mu = float(loc_student.get_mle(data, params))
        global_ll = float(loglik(data, global_mu, k))
        global_score = abs(score(data, global_mu, k))
        _, multistart_ll, _, starts = global_mle_multistart(data, k)
        gap = global_ll - brent_ll
        max_score_residual = max(max_score_residual, global_score)
        dense_mu = dense_ll = dense_gap = np.nan
        dense_agrees = True
        if index < validation_datasets or gap > likelihood_tol:
            dense_checked += 1
            dense_mu, dense_ll = global_mle_dense(data, k, dense_points)
            dense_gap = dense_ll - global_ll
            dense_agrees = bool(
                abs(dense_gap) <= max(likelihood_tol, 1e-7)
                and multistart_ll <= global_ll + likelihood_tol
            )
            dense_disagreements += int(not dense_agrees)
        if gap <= likelihood_tol:
            continue

        if not np.isfinite(dense_ll):
            dense_checked += 1
            dense_mu, dense_ll = global_mle_dense(data, k, dense_points)
            dense_gap = dense_ll - global_ll
            dense_agrees = abs(dense_gap) <= max(likelihood_tol, 1e-7)
            dense_disagreements += int(not dense_agrees)
        false_rows.append(
            {
                **case,
                "dataset_index": int(index),
                "brent_mu": brent_mu,
                "global_mu": global_mu,
                "dense_mu": dense_mu,
                "brent_score_abs": abs(score(data, brent_mu, k)),
                "global_score_abs": global_score,
                "brent_loglik": brent_ll,
                "global_loglik": global_ll,
                "dense_loglik": dense_ll,
                "global_minus_brent_loglik": gap,
                "dense_minus_global_loglik": dense_gap,
                "dense_agrees": dense_agrees,
                "num_multistarts": starts,
                "data_min": float(np.min(data)),
                "data_median": float(np.median(data)),
                "data_max": float(np.max(data)),
            }
        )

    gaps = np.asarray([row["global_minus_brent_loglik"] for row in false_rows], dtype=float)
    total = int(len(data_sets))
    summary = {
        **case,
        "num_datasets": total,
        "num_false_brent_mle": int(len(false_rows)),
        "false_rate": float(len(false_rows) / max(total, 1)),
        "max_loglik_gap": float(np.max(gaps)) if gaps.size else 0.0,
        "median_false_loglik_gap": float(np.median(gaps)) if gaps.size else 0.0,
        "dense_validation_checked": int(dense_checked),
        "dense_validation_disagreements": int(dense_disagreements),
        "max_global_score_abs": float(max_score_residual),
    }
    return summary, false_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cache_cases(args: argparse.Namespace) -> list[tuple[Path, dict, np.ndarray, np.ndarray]]:
    cases = []
    for path in sorted(args.cache_dir.glob("student_k*_n*_B*_seed*.npz")):
        loaded = np.load(path)
        k = float(loaded["k"])
        n = int(loaded["n"])
        size = int(loaded["B"])
        seed = int(loaded["seed"])
        data = np.asarray(random.t(simulation_key(seed), df=k, shape=(size, n)), dtype=float)
        cached = np.asarray(loaded["z_samples"], dtype=float)
        cases.append((path, {"source": "physical_cache", "k": k, "n": n, "seed": seed}, data, cached))
    return cases


def markdown_report(args: argparse.Namespace, summaries: list[dict], cache_checks: list[dict]) -> str:
    lines = [
        "# Student wide-Brent versus global-MLE audit",
        "",
        f"Likelihood-gap tolerance: {args.likelihood_tol:g}.",
        "Production globally isolates Student score roots with interval bounds and selects by actual likelihood. Every detected legacy disagreement is independently checked by multi-start EM/IRLS and an endpoint-dense interval search.",
        "",
        "## Simulated benchmark grid",
        "",
        "| k | n | seed | datasets | false Brent MLE | false rate | max loglik gap | dense checks | disagreements |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["source"] != "simulated_grid":
            continue
        lines.append(
            f"| {row['k']:g} | {row['n']} | {row['seed']} | {row['num_datasets']} | "
            f"{row['num_false_brent_mle']} | {100.0 * row['false_rate']:.4f}% | "
            f"{row['max_loglik_gap']:.6g} | {row['dense_validation_checked']} | "
            f"{row['dense_validation_disagreements']} |"
        )
    lines.extend(["", "## Physical cache reconstruction", ""])
    if cache_checks:
        lines.extend(
            [
                "| cache | datasets | cached values reproduced | false Brent MLE | false rate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in cache_checks:
            lines.append(
                f"| `{row['cache_path']}` | {row['num_datasets']} | {row['cached_values_reproduced']} | "
                f"{row['num_false_brent_mle']} | {100.0 * row['false_rate']:.4f}% |"
            )
    else:
        lines.append("No matching physical cache files were found.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A row is false only when the trusted candidate has a strictly higher Student log-likelihood than the wide-Brent score root. Equal-likelihood alternative maximizers are not counted as failures.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.datasets <= 0 or args.dense_points_per_interval < 9:
        raise ValueError("datasets must be positive and dense-points-per-interval must be at least 9")
    args.out.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    false_rows: list[dict] = []
    for k in args.k_values:
        for n in args.n_values:
            for seed in args.seeds:
                case = {"source": "simulated_grid", "k": float(k), "n": int(n), "seed": int(seed)}
                data = np.asarray(random.t(simulation_key(seed), df=float(k), shape=(args.datasets, int(n))), dtype=float)
                summary, failures = audit_datasets(
                    data,
                    float(k),
                    args.likelihood_tol,
                    args.dense_points_per_interval,
                    args.validation_datasets,
                    case,
                )
                summaries.append(summary)
                false_rows.extend(failures)
                print(
                    f"k={float(k):g} n={int(n)} seed={int(seed)}: "
                    f"{summary['num_false_brent_mle']}/{summary['num_datasets']} false",
                    flush=True,
                )

    cache_checks: list[dict] = []
    for path, case, data, cached in cache_cases(args):
        reconstructed = np.asarray(
            [loc_student.get_mle(row, {"k": case["k"], "n": case["n"]}) for row in data],
            dtype=float,
        )
        summary, failures = audit_datasets(
            data,
            case["k"],
            args.likelihood_tol,
            args.dense_points_per_interval,
            min(args.validation_datasets, len(data)),
            case,
        )
        summary["cache_path"] = str(path)
        summary["cached_values_reproduced"] = bool(np.allclose(cached, reconstructed, rtol=0.0, atol=1e-12))
        cache_checks.append(summary)
        false_rows.extend(failures)

    write_csv(args.out / "summary.csv", summaries)
    write_csv(args.out / "false_cases.csv", false_rows)
    write_csv(args.out / "cache_checks.csv", cache_checks)
    (args.out / "report.md").write_text(markdown_report(args, summaries, cache_checks) + "\n", encoding="utf-8")
    (args.out / "settings.json").write_text(
        json.dumps(
            {
                "k_values": args.k_values,
                "n_values": args.n_values,
                "seeds": args.seeds,
                "datasets": args.datasets,
                "likelihood_tol": args.likelihood_tol,
                "dense_points_per_interval": args.dense_points_per_interval,
                "validation_datasets": args.validation_datasets,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
