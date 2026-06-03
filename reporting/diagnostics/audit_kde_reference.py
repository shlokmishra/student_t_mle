"""Audit MLE-conditional posterior reference candidates for Student-t location.

This script separates three sources of variation:
  1. finite simulated-MLE sample error via a raw weighted-MC estimator,
  2. KDE backend/bandwidth sensitivity,
  3. posterior integration grid/quad sensitivity.

Smoke run:
    python -m reporting.diagnostics.audit_kde_reference --B-values 200 --seeds 0 --bandwidths SJ_transform,scott --n-grid-values 1000 --bound-multipliers 5 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import jax.random as random
import numpy as np

from kde_ref.moments import (
    gaussian_kde_gaussian_prior_moments,
    kde_grid_posterior_moments,
    kde_quad_posterior_moments,
    raw_weighted_posterior_moments,
)
from kde_ref.posterior import build_likelihood_kde_backend
from models import loc_student

OUT_DIR = Path("reporting/diagnostic_outputs/kde_reference_audit")
OUT_CSV = OUT_DIR / "kde_reference_audit.csv"


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=_floats, default=[2.0], help="Comma-separated Student-t degrees of freedom.")
    parser.add_argument("--n-values", type=_ints, default=[20], help="Comma-separated sample sizes.")
    parser.add_argument("--B-values", type=_ints, default=[1000], help="Comma-separated simulated MLE sample counts.")
    parser.add_argument("--seeds", type=_ints, default=[0], help="Comma-separated deterministic seeds.")
    parser.add_argument("--bandwidths", type=_strings, default=["SJ_transform", "t_abram", "scott", "silverman"], help="Comma-separated KDE backends.")
    parser.add_argument("--n-grid-values", type=_ints, default=[1000, 4000], help="Comma-separated grid sizes for KDE-grid integration.")
    parser.add_argument("--bound-multipliers", type=_floats, default=[5.0, 8.0], help="Comma-separated bound multipliers for KDE-grid integration.")
    parser.add_argument("--use-quad", action="store_true", help="Also compute adaptive quad KDE posterior moments.")
    parser.add_argument("--mu-true", type=float, default=2.0, help="Observed-data generating location.")
    parser.add_argument("--prior-mean", type=float, default=0.0, help="Normal prior mean.")
    parser.add_argument("--prior-std", type=float, default=10.0, help="Normal prior standard deviation.")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV, help="Output CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output CSV.")
    parser.add_argument("--cache-mle-samples", action="store_true", help="Cache simulated centered MLE samples under the output directory.")
    parser.add_argument("--include-gaussian-analytic-check", action="store_true", help="Add analytic mixture rows for Gaussian KDE backends.")
    return parser.parse_args()


FIELDNAMES = [
    "model",
    "k",
    "n",
    "seed",
    "B",
    "mu_star",
    "estimator_type",
    "backend",
    "n_grid",
    "bound_multiplier",
    "grid_lo",
    "grid_hi",
    "posterior_mean",
    "posterior_var",
    "posterior_sd",
    "posterior_q025",
    "posterior_q50",
    "posterior_q975",
    "normalization_constant",
    "bandwidth",
    "weighted_ess",
    "runtime_seconds",
]


def _row_base(k: float, n: int, seed: int, B: int, mu_star: float) -> dict[str, Any]:
    return {
        "model": "loc_student",
        "k": k,
        "n": n,
        "seed": seed,
        "B": B,
        "mu_star": mu_star,
    }


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def _completed(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            keys.add(
                (
                    row["k"],
                    row["n"],
                    row["seed"],
                    row["B"],
                    row["estimator_type"],
                    row["backend"],
                    row["n_grid"],
                    row["bound_multiplier"],
                )
            )
    return keys


def _cache_path(args: argparse.Namespace, k: float, n: int, B: int, seed: int) -> Path:
    return args.out_csv.parent / "mle_sample_cache" / f"student_k{k}_n{n}_B{B}_seed{seed}.npz"


def _simulate_case(args: argparse.Namespace, k: float, n: int, B: int, seed: int) -> tuple[float, np.ndarray]:
    params = {"k": k, "n": n, "num_iterations_T": 1}
    path = _cache_path(args, k, n, B, seed)
    if args.cache_mle_samples and path.exists():
        loaded = np.load(path)
        z_samples = np.asarray(loaded["z_samples"], dtype=float)
    else:
        key = random.PRNGKey(seed)
        key, key_obs, key_mle = random.split(key, 3)
        z_samples = np.asarray(loc_student.get_benchmark_mle_samples(key_mle, params, num_simulations=B, verbose=False), dtype=float)
        if args.cache_mle_samples:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, z_samples=z_samples, k=np.float64(k), n=np.int64(n), B=np.int64(B), seed=np.int64(seed))

    key = random.PRNGKey(seed)
    key, key_obs, _ = random.split(key, 3)
    data = np.asarray(loc_student.sample_data(key_obs, params, loc=args.mu_true), dtype=float)
    mu_star = float(loc_student.get_mle(data, params))
    return mu_star, z_samples


def _emit_raw(args: argparse.Namespace, done: set[tuple[str, ...]], k: float, n: int, seed: int, B: int, mu_star: float, z_samples: np.ndarray) -> None:
    key = (str(k), str(n), str(seed), str(B), "raw_weighted_mc", "none", "", "")
    if key in done:
        return
    start = time.time()
    moments = raw_weighted_posterior_moments(z_samples, mu_star, args.prior_mean, args.prior_std)
    row = {
        **_row_base(k, n, seed, B, mu_star),
        "estimator_type": "raw_weighted_mc",
        "backend": "none",
        "n_grid": "",
        "bound_multiplier": "",
        "grid_lo": "",
        "grid_hi": "",
        **moments,
        "bandwidth": "",
        "runtime_seconds": time.time() - start,
    }
    _append_row(args.out_csv, row)


def _emit_kde_rows(args: argparse.Namespace, done: set[tuple[str, ...]], k: float, n: int, seed: int, B: int, mu_star: float, z_samples: np.ndarray, backend_name: str) -> None:
    params = {
        "k": k,
        "n": n,
        "prior_mean": args.prior_mean,
        "prior_std": args.prior_std,
        "kde_bw_method": backend_name,
    }
    fit_start = time.time()
    backend = build_likelihood_kde_backend(z_samples, params, verbose=False)
    fit_seconds = time.time() - fit_start
    bandwidth = float(getattr(backend, "bandwidth", np.nan))

    for n_grid in args.n_grid_values:
        for bound_multiplier in args.bound_multipliers:
            key = (str(k), str(n), str(seed), str(B), "kde_grid", backend_name, str(n_grid), str(bound_multiplier))
            if key in done:
                continue
            start = time.time()
            moments = kde_grid_posterior_moments(
                backend,
                mu_star,
                args.prior_mean,
                args.prior_std,
                z_samples,
                n_grid=n_grid,
                bound_multiplier=bound_multiplier,
            )
            row = {
                **_row_base(k, n, seed, B, mu_star),
                "estimator_type": "kde_grid",
                "backend": backend_name,
                "n_grid": n_grid,
                "bound_multiplier": bound_multiplier,
                **moments,
                "bandwidth": bandwidth,
                "weighted_ess": "",
                "runtime_seconds": fit_seconds + time.time() - start,
            }
            _append_row(args.out_csv, row)

    if args.use_quad:
        key = (str(k), str(n), str(seed), str(B), "kde_quad", backend_name, "", "")
        if key not in done:
            start = time.time()
            moments = kde_quad_posterior_moments(
                backend,
                mu_star,
                args.prior_mean,
                args.prior_std,
                z_samples,
                quantile_grid_size=max(args.n_grid_values),
                quantile_bound_multiplier=max(args.bound_multipliers),
            )
            row = {
                **_row_base(k, n, seed, B, mu_star),
                "estimator_type": "kde_quad",
                "backend": backend_name,
                "n_grid": "",
                "bound_multiplier": "",
                **moments,
                "bandwidth": bandwidth,
                "weighted_ess": "",
                "runtime_seconds": fit_seconds + time.time() - start,
            }
            _append_row(args.out_csv, row)

    if args.include_gaussian_analytic_check and backend_name.lower() in {"scott", "silverman"}:
        key = (str(k), str(n), str(seed), str(B), "gaussian_kde_analytic", backend_name, "", "")
        if key not in done:
            start = time.time()
            moments = gaussian_kde_gaussian_prior_moments(backend, mu_star, args.prior_mean, args.prior_std)
            if moments is not None:
                row = {
                    **_row_base(k, n, seed, B, mu_star),
                    "estimator_type": "gaussian_kde_analytic",
                    "backend": backend_name,
                    "n_grid": "",
                    "bound_multiplier": "",
                    "grid_lo": "",
                    "grid_hi": "",
                    **moments,
                    "bandwidth": bandwidth,
                    "runtime_seconds": fit_seconds + time.time() - start,
                }
                _append_row(args.out_csv, row)


def main() -> None:
    args = parse_args()
    if args.overwrite and args.out_csv.exists():
        args.out_csv.unlink()
    done = _completed(args.out_csv)
    for k in args.k_values:
        for n in args.n_values:
            for B in args.B_values:
                for seed in args.seeds:
                    mu_star, z_samples = _simulate_case(args, k, n, B, seed)
                    _emit_raw(args, done, k, n, seed, B, mu_star, z_samples)
                    for backend in args.bandwidths:
                        _emit_kde_rows(args, done, k, n, seed, B, mu_star, z_samples, backend)
                    print(f"completed k={k} n={n} B={B} seed={seed}")


if __name__ == "__main__":
    main()
