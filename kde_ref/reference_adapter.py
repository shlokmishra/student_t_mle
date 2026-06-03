"""Adapters for Step 1 KDE/reference posterior comparisons.

This module is intentionally thin: it reuses the existing posterior,
moment, and audit machinery and only normalizes shapes for diagnostics,
dashboards, and later reference comparisons.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import jax.random as random
import numpy as np
import pandas as pd

from kde_ref.moments import (
    kde_grid_posterior_moments,
    kde_quad_posterior_moments,
    posterior_grid_bounds,
    raw_weighted_posterior_moments,
)
from kde_ref.posterior import build_likelihood_kde_backend, get_normalized_posterior_pdf
from models import loc_student


DEFAULT_AUDIT_DIR = Path("reporting/diagnostic_outputs/kde_reference_audit")
DEFAULT_STEP1_CSV = DEFAULT_AUDIT_DIR / "step1_k2_n10_20_50.csv"
DEFAULT_BACKENDS = ("scott", "SJ_transform", "t_abram")


def _as_list(values: Iterable[int | float | str] | int | float | str) -> list:
    if isinstance(values, (str, int, float)):
        return [values]
    return list(values)


def _csv_arg(values: Iterable[int | float | str] | int | float | str) -> str:
    return ",".join(str(value) for value in _as_list(values))


def _read_csvs(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "source_file" not in frame.columns:
            frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def discover_audit_csvs(audit_dir: Path = DEFAULT_AUDIT_DIR) -> list[Path]:
    """Return audit CSVs from the standard output directory."""
    if not audit_dir.exists():
        return []
    return sorted(audit_dir.glob("*.csv"))


def audit_command(
    *,
    k_values: Iterable[float] = (2.0,),
    n_values: Iterable[int] = (10, 20, 50),
    B_values: Iterable[int] = (100000,),
    seeds: Iterable[int] = (123,),
    backends: Iterable[str] = DEFAULT_BACKENDS,
    out_csv: Path = DEFAULT_STEP1_CSV,
    use_quad: bool = False,
    cache_mle_samples: bool = True,
    overwrite: bool = False,
) -> list[str]:
    """Build the existing audit command with this repo's comma-list CLI."""
    cmd = [
        sys.executable,
        "-m",
        "reporting.diagnostics.audit_kde_reference",
        "--k-values",
        _csv_arg(k_values),
        "--n-values",
        _csv_arg(n_values),
        "--B-values",
        _csv_arg(B_values),
        "--seeds",
        _csv_arg(seeds),
        "--bandwidths",
        _csv_arg(backends),
        "--out-csv",
        str(out_csv),
    ]
    if cache_mle_samples:
        cmd.append("--cache-mle-samples")
    if use_quad:
        cmd.append("--use-quad")
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def load_or_run_kde_audit(
    *,
    k_values: Iterable[float] = (2.0,),
    n_values: Iterable[int] = (10, 20, 50),
    B_values: Iterable[int] = (100000,),
    seeds: Iterable[int] = (123,),
    backends: Iterable[str] = DEFAULT_BACKENDS,
    out_csv: Path = DEFAULT_STEP1_CSV,
    run_if_missing: bool = False,
    use_quad: bool = False,
    cache_mle_samples: bool = True,
) -> pd.DataFrame:
    """Load matching audit rows, optionally running the existing audit first."""
    paths = [out_csv] if out_csv.exists() else discover_audit_csvs(out_csv.parent)
    df = _read_csvs(paths)
    matching = filter_reference_summary(
        df,
        k_values=k_values,
        n_values=n_values,
        B_values=B_values,
        seeds=seeds,
        backends=backends,
    )
    if not matching.empty or not run_if_missing:
        return common_summary_schema(matching)

    cmd = audit_command(
        k_values=k_values,
        n_values=n_values,
        B_values=B_values,
        seeds=seeds,
        backends=backends,
        out_csv=out_csv,
        use_quad=use_quad,
        cache_mle_samples=cache_mle_samples,
    )
    subprocess.run(cmd, check=True)
    return common_summary_schema(pd.read_csv(out_csv))


def filter_reference_summary(
    df: pd.DataFrame,
    *,
    k_values: Iterable[float] | None = None,
    n_values: Iterable[int] | None = None,
    B_values: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    backends: Iterable[str] | None = None,
    mu_star: float | None = None,
) -> pd.DataFrame:
    """Filter audit rows while preserving raw weighted-MC rows."""
    if df.empty:
        return df
    out = df.copy()
    if k_values is not None:
        out = out[out["k"].astype(float).isin([float(x) for x in _as_list(k_values)])]
    if n_values is not None:
        out = out[out["n"].astype(int).isin([int(x) for x in _as_list(n_values)])]
    if B_values is not None:
        out = out[out["B"].astype(int).isin([int(x) for x in _as_list(B_values)])]
    if seeds is not None:
        out = out[out["seed"].astype(int).isin([int(x) for x in _as_list(seeds)])]
    if backends is not None and "backend" in out:
        allowed = {str(x) for x in _as_list(backends)} | {"none", ""}
        out = out[out["backend"].fillna("").astype(str).isin(allowed)]
    if mu_star is not None and "mu_star" in out:
        out = out[np.isclose(out["mu_star"].astype(float), float(mu_star))]
    return out.reset_index(drop=True)


def common_summary_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return audit summaries in the dashboard/reference comparison schema."""
    columns = [
        "method",
        "estimator_type",
        "backend",
        "n",
        "k",
        "mu_star",
        "mean",
        "var",
        "sd",
        "q025",
        "q50",
        "q975",
        "marginal_likelihood_estimate",
        "posterior_integral_check",
        "weighted_ess",
        "B",
        "seed",
        "grid_size",
        "bound_multiplier",
        "grid_lo",
        "grid_hi",
        "bandwidth",
        "source_file",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(
        {
            "method": np.where(
                df["estimator_type"].eq("raw_weighted_mc"),
                "raw weighted-MC reference",
                "KDE smoothed density",
            ),
            "estimator_type": df["estimator_type"],
            "backend": df["backend"].fillna(""),
            "n": df["n"],
            "k": df["k"],
            "mu_star": df["mu_star"],
            "mean": df["posterior_mean"],
            "var": df["posterior_var"],
            "sd": df["posterior_sd"],
            "q025": df["posterior_q025"],
            "q50": df["posterior_q50"],
            "q975": df["posterior_q975"],
            "marginal_likelihood_estimate": df["normalization_constant"],
            "posterior_integral_check": df.get("posterior_integral_check", np.nan),
            "weighted_ess": df.get("weighted_ess", np.nan),
            "B": df["B"],
            "seed": df["seed"],
            "grid_size": df.get("n_grid", np.nan),
            "bound_multiplier": df.get("bound_multiplier", np.nan),
            "grid_lo": df.get("grid_lo", np.nan),
            "grid_hi": df.get("grid_hi", np.nan),
            "bandwidth": df.get("bandwidth", np.nan),
            "source_file": df.get("source_file", ""),
        }
    )
    return out[columns]


def load_reference_summary(
    csv_path: Path | None = None,
    *,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    **filters,
) -> pd.DataFrame:
    """Read audit CSV output and return rows in a common schema."""
    paths = [csv_path] if csv_path is not None else discover_audit_csvs(audit_dir)
    df = _read_csvs([Path(path) for path in paths if path is not None])
    return common_summary_schema(filter_reference_summary(df, **filters))


def mle_sample_cache_path(
    *,
    k: float,
    n: int,
    B: int,
    seed: int,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
) -> Path:
    return audit_dir / "mle_sample_cache" / f"student_k{float(k)}_n{int(n)}_B{int(B)}_seed{int(seed)}.npz"


def load_or_simulate_mle_errors(
    *,
    k: float,
    n: int,
    B: int,
    seed: int,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    cache_mle_samples: bool = True,
) -> np.ndarray:
    """Load cached centered MLE errors or simulate them with the model helper."""
    path = mle_sample_cache_path(k=k, n=n, B=B, seed=seed, audit_dir=audit_dir)
    if path.exists():
        return np.asarray(np.load(path)["z_samples"], dtype=float)

    params = {"k": float(k), "n": int(n), "num_iterations_T": 1}
    key = random.PRNGKey(int(seed))
    _, _, key_mle = random.split(key, 3)
    z_samples = np.asarray(
        loc_student.get_benchmark_mle_samples(key_mle, params, num_simulations=int(B), verbose=False),
        dtype=float,
    )
    if cache_mle_samples:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            z_samples=z_samples,
            k=np.float64(k),
            n=np.int64(n),
            B=np.int64(B),
            seed=np.int64(seed),
        )
    return z_samples


def build_posterior_density_grid(
    *,
    z_samples: np.ndarray | None = None,
    k: float = 2.0,
    n: int = 20,
    mu_star: float = 0.0,
    prior_mean: float = 0.0,
    prior_std: float = 10.0,
    B: int = 1000,
    seed: int = 0,
    backends: Iterable[str] = DEFAULT_BACKENDS,
    grid_size: int = 1000,
    bound_multiplier: float = 5.0,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
) -> pd.DataFrame:
    """Build normalized KDE posterior density curves for selected backends."""
    if z_samples is None:
        z_samples = load_or_simulate_mle_errors(k=k, n=n, B=B, seed=seed, audit_dir=audit_dir)
    z_samples = np.asarray(z_samples, dtype=float)
    lo, hi = posterior_grid_bounds(mu_star, prior_mean, prior_std, z_samples, bound_multiplier)
    mu_grid = np.linspace(lo, hi, int(grid_size))
    rows = []
    for backend_name in backends:
        params = {
            "k": float(k),
            "n": int(n),
            "prior_mean": float(prior_mean),
            "prior_std": float(prior_std),
            "kde_bw_method": backend_name,
        }
        posterior_pdf, info = get_normalized_posterior_pdf(
            mu_star,
            params,
            z_samples,
            use_grid=True,
            n_grid=int(grid_size),
            return_info=True,
        )
        density = np.maximum(posterior_pdf(mu_grid), 0.0)
        cdf = np.concatenate([[0.0], np.cumsum((density[:-1] + density[1:]) * np.diff(mu_grid) / 2.0)])
        posterior_integral_check = float(cdf[-1])
        if posterior_integral_check > 0:
            cdf = cdf / posterior_integral_check
        rows.append(
            pd.DataFrame(
                {
                    "method": "KDE smoothed density",
                    "estimator_type": "kde_density",
                    "backend": backend_name,
                    "n": int(n),
                    "k": float(k),
                    "mu_star": float(mu_star),
                    "mu": mu_grid,
                    "density": density,
                    "cdf": cdf,
                    "marginal_likelihood_estimate": info["normalization_constant"],
                    "posterior_integral_check": posterior_integral_check,
                    "seed": int(seed),
                    "B": int(B),
                    "source_file": "computed",
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "method",
                "estimator_type",
                "backend",
                "n",
                "k",
                "mu_star",
                "mu",
                "density",
                "cdf",
                "marginal_likelihood_estimate",
                "posterior_integral_check",
                "seed",
                "B",
                "source_file",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def build_reference_summaries_from_samples(
    *,
    z_samples: np.ndarray,
    k: float,
    n: int,
    mu_star: float,
    prior_mean: float,
    prior_std: float,
    B: int,
    seed: int,
    backends: Iterable[str] = DEFAULT_BACKENDS,
    include_raw: bool = True,
    include_kde_grid: bool = True,
    include_kde_quad: bool = False,
    grid_size: int = 1000,
    bound_multiplier: float = 5.0,
) -> pd.DataFrame:
    """Compute Step 1 summaries from already available MLE-error samples."""
    z_samples = np.asarray(z_samples, dtype=float)
    rows: list[dict] = []
    base = {"n": int(n), "k": float(k), "mu_star": float(mu_star), "B": int(B), "seed": int(seed)}
    if include_raw:
        moments = raw_weighted_posterior_moments(z_samples, mu_star, prior_mean, prior_std)
        rows.append(_summary_row(base, "raw weighted-MC reference", "raw_weighted_mc", "none", moments))

    for backend_name in backends:
        params = {
            "k": float(k),
            "n": int(n),
            "prior_mean": float(prior_mean),
            "prior_std": float(prior_std),
            "kde_bw_method": backend_name,
        }
        backend = build_likelihood_kde_backend(z_samples, params, verbose=False)
        if include_kde_grid:
            moments = kde_grid_posterior_moments(
                backend,
                mu_star,
                prior_mean,
                prior_std,
                z_samples,
                n_grid=grid_size,
                bound_multiplier=bound_multiplier,
            )
            rows.append(
                _summary_row(
                    base,
                    "KDE smoothed density",
                    "kde_grid",
                    backend_name,
                    moments,
                    grid_size=grid_size,
                    bound_multiplier=bound_multiplier,
                    bandwidth=getattr(backend, "bandwidth", np.nan),
                )
            )
        if include_kde_quad:
            moments = kde_quad_posterior_moments(
                backend,
                mu_star,
                prior_mean,
                prior_std,
                z_samples,
                quantile_grid_size=grid_size,
                quantile_bound_multiplier=bound_multiplier,
            )
            rows.append(
                _summary_row(
                    base,
                    "KDE smoothed density",
                    "kde_quad",
                    backend_name,
                    moments,
                    grid_size=grid_size,
                    bound_multiplier=bound_multiplier,
                    bandwidth=getattr(backend, "bandwidth", np.nan),
                )
            )
    return pd.DataFrame(rows)


def _summary_row(
    base: dict,
    method: str,
    estimator_type: str,
    backend: str,
    moments: dict,
    *,
    grid_size: int | float = np.nan,
    bound_multiplier: float = np.nan,
    bandwidth: float = np.nan,
) -> dict:
    return {
        **base,
        "method": method,
        "estimator_type": estimator_type,
        "backend": backend,
        "mean": moments["posterior_mean"],
        "var": moments["posterior_var"],
        "sd": moments["posterior_sd"],
        "q025": moments["posterior_q025"],
        "q50": moments["posterior_q50"],
        "q975": moments["posterior_q975"],
        "marginal_likelihood_estimate": moments["normalization_constant"],
        "posterior_integral_check": moments.get("posterior_integral_check", np.nan),
        "weighted_ess": moments.get("weighted_ess", np.nan),
        "grid_size": grid_size,
        "bound_multiplier": bound_multiplier,
        "grid_lo": moments.get("grid_lo", np.nan),
        "grid_hi": moments.get("grid_hi", np.nan),
        "bandwidth": bandwidth,
        "source_file": moments.get("source_file", "computed"),
    }
