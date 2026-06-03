"""Reference posterior audit across Student-t, logistic, and Laplace models."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import jax.random as random
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kde_ref.moments import raw_weighted_posterior_moments
from kde_ref.posterior import get_normalized_posterior_pdf
from models import loc_laplace, loc_logistic, loc_student
from models.model_registry import (
    LAPLACE_MEDIAN_INTERVAL_TARGET,
    LAPLACE_NP_MEDIAN_TARGET,
    get_model_spec,
)

OUT_CSV = Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv")
FIELDNAMES = [
    "model",
    "k",
    "n",
    "mu_star",
    "method",
    "estimator_type",
    "backend",
    "mean",
    "var",
    "sd",
    "q025",
    "q50",
    "q975",
    "weighted_ess",
    "marginal_likelihood_estimate",
    "B",
    "seed",
    "mle_convention",
    "target_description",
    "source_file",
]


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["student_t", "logistic", "laplace"], choices=["student_t", "logistic", "laplace"])
    parser.add_argument("--k-values", type=_floats, default=[1.0, 2.0, 3.0])
    parser.add_argument("--n-values", type=_ints, default=[10, 20, 50])
    parser.add_argument("--B-values", type=_ints, default=[100000])
    parser.add_argument("--seeds", type=_ints, default=[123, 456, 789])
    parser.add_argument("--bandwidths", type=_strings, default=["scott", "SJ_transform"])
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-std", type=float, default=10.0)
    parser.add_argument("--laplace-b", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=2500)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_k_values(model: str, k_values: Iterable[float]) -> list[float]:
    return [float(k) for k in k_values] if model == "student_t" else [np.nan]


def simulate_mle_errors(model: str, k: float, n: int, b: int, seed: int, laplace_b: float) -> np.ndarray:
    key = random.PRNGKey(int(seed))
    if model == "student_t":
        return np.asarray(loc_student.get_benchmark_mle_samples(key, {"k": float(k), "n": int(n)}, num_simulations=int(b)), dtype=float)
    if model == "logistic":
        return np.asarray(loc_logistic.get_benchmark_mle_samples(key, {"n": int(n)}, num_simulations=int(b)), dtype=float)
    if model == "laplace":
        return np.asarray(loc_laplace.get_benchmark_mle_samples(key, {"n": int(n), "b": float(laplace_b)}, num_simulations=int(b)), dtype=float)
    raise ValueError(model)


def simulate_laplace_order_stats(n: int, b: int, seed: int, laplace_b: float) -> tuple[np.ndarray, np.ndarray]:
    key = random.PRNGKey(int(seed))
    data = np.asarray(loc_laplace.sample_data(key, {"n": int(n) * int(b), "b": float(laplace_b)}, loc=0.0), dtype=float)
    data = data.reshape(int(b), int(n))
    ordered = np.sort(data, axis=1)
    if int(n) % 2 == 0:
        lower = ordered[:, int(n) // 2 - 1]
        upper = ordered[:, int(n) // 2]
    else:
        lower = ordered[:, int(n) // 2]
        upper = ordered[:, int(n) // 2]
    return lower, upper


def weighted_grid_summary(mu_grid: np.ndarray, weights: np.ndarray) -> dict:
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    integral = float(np.trapezoid(weights, mu_grid))
    if integral <= 0 or not np.isfinite(integral):
        return {key: np.nan for key in ["mean", "var", "sd", "q025", "q50", "q975", "marginal_likelihood_estimate"]}
    density = weights / integral
    mean = float(np.trapezoid(mu_grid * density, mu_grid))
    second = float(np.trapezoid(mu_grid * mu_grid * density, mu_grid))
    var = max(second - mean * mean, 0.0)
    cdf = np.concatenate([[0.0], np.cumsum((density[:-1] + density[1:]) * np.diff(mu_grid) / 2.0)])
    cdf = cdf / max(cdf[-1], 1e-300)
    q025, q50, q975 = np.interp([0.025, 0.5, 0.975], cdf, mu_grid)
    return {
        "mean": mean,
        "var": float(var),
        "sd": float(np.sqrt(var)),
        "q025": float(q025),
        "q50": float(q50),
        "q975": float(q975),
        "marginal_likelihood_estimate": integral,
    }


def laplace_interval_reference(lower: np.ndarray, upper: np.ndarray, mu_star: float, prior_mean: float, prior_std: float, grid_size: int) -> dict:
    lo = float(mu_star - np.quantile(upper, 0.995) - 4.0 * prior_std)
    hi = float(mu_star - np.quantile(lower, 0.005) + 4.0 * prior_std)
    mu_grid = np.linspace(lo, hi, int(grid_size))
    z_grid = float(mu_star) - mu_grid
    likelihood = np.array([np.mean((lower <= z) & (z <= upper)) for z in z_grid], dtype=float)
    prior = stats.norm.pdf(mu_grid, loc=float(prior_mean), scale=float(prior_std))
    out = weighted_grid_summary(mu_grid, prior * likelihood)
    out["weighted_ess"] = np.nan
    return out


def kde_grid_summary(z_samples: np.ndarray, model: str, k: float, n: int, mu_star: float, prior_mean: float, prior_std: float, backend: str, grid_size: int) -> dict:
    params = {
        "k": float(k) if np.isfinite(k) else np.nan,
        "n": int(n),
        "prior_mean": float(prior_mean),
        "prior_std": float(prior_std),
        "kde_bw_method": backend,
    }
    posterior_pdf = get_normalized_posterior_pdf(mu_star, params, z_samples, use_grid=True, n_grid=int(grid_size))
    raw = raw_weighted_posterior_moments(z_samples, mu_star, prior_mean, prior_std)
    width = max(raw["posterior_q975"] - raw["posterior_q025"], raw["posterior_sd"], 1e-3)
    mu_grid = np.linspace(raw["posterior_q025"] - width, raw["posterior_q975"] + width, int(grid_size))
    density = np.maximum(posterior_pdf(mu_grid), 0.0)
    out = weighted_grid_summary(mu_grid, density)
    return out


def row_base(model: str, k: float, n: int, mu_star: float, b: int, seed: int, target: dict | None = None) -> dict:
    spec = get_model_spec(model)
    return {
        "model": model,
        "k": float(k) if np.isfinite(k) else "",
        "n": int(n),
        "mu_star": float(mu_star),
        "B": int(b),
        "seed": int(seed),
        "mle_convention": (target or {}).get("mle_convention", spec.mle_convention),
        "target_description": (target or {}).get("target_description", spec.target_description),
    }


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def emit_rows(args: argparse.Namespace) -> None:
    if args.overwrite and args.out_csv.exists():
        args.out_csv.unlink()
    for model in args.models:
        for k in model_k_values(model, args.k_values):
            for n in args.n_values:
                for b in args.B_values:
                    for seed in args.seeds:
                        z_samples = simulate_mle_errors(model, k, n, b, seed, args.laplace_b)
                        target = LAPLACE_NP_MEDIAN_TARGET if model == "laplace" else None
                        raw = raw_weighted_posterior_moments(z_samples, args.mu_star, args.prior_mean, args.prior_std)
                        append_row(
                            args.out_csv,
                            {
                                **row_base(model, k, n, args.mu_star, b, seed, target),
                                "method": "raw weighted-MC reference",
                                "estimator_type": "raw_weighted_mc",
                                "backend": "none",
                                "mean": raw["posterior_mean"],
                                "var": raw["posterior_var"],
                                "sd": raw["posterior_sd"],
                                "q025": raw["posterior_q025"],
                                "q50": raw["posterior_q50"],
                                "q975": raw["posterior_q975"],
                                "weighted_ess": raw["weighted_ess"],
                                "marginal_likelihood_estimate": raw["normalization_constant"],
                                "source_file": "computed",
                            },
                        )
                        for backend in args.bandwidths:
                            kde = kde_grid_summary(z_samples, model, k, n, args.mu_star, args.prior_mean, args.prior_std, backend, args.grid_size)
                            append_row(
                                args.out_csv,
                                {
                                    **row_base(model, k, n, args.mu_star, b, seed, target),
                                    "method": "KDE smoothed density",
                                    "estimator_type": "kde_grid",
                                    "backend": backend,
                                    **kde,
                                    "weighted_ess": "",
                                    "source_file": "computed",
                                },
                            )
                        if model == "laplace":
                            lower, upper = simulate_laplace_order_stats(n, b, seed, args.laplace_b)
                            interval = laplace_interval_reference(lower, upper, args.mu_star, args.prior_mean, args.prior_std, args.grid_size)
                            append_row(
                                args.out_csv,
                                {
                                    **row_base(model, k, n, args.mu_star, b, seed, LAPLACE_MEDIAN_INTERVAL_TARGET),
                                    "method": "raw MC interval reference",
                                    "estimator_type": "raw_mc_interval_reference",
                                    "backend": "none",
                                    **interval,
                                    "source_file": "computed",
                                },
                            )
                        print(f"completed model={model} n={n} k={k} B={b} seed={seed}")


def main() -> None:
    emit_rows(parse_args())


if __name__ == "__main__":
    main()
