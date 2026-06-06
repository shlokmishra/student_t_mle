"""Streamlit dashboard for KDE/reference posterior Step 1 comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from dashboard_cache import read_cache_csv, require_cache_file, sidebar_cache_controls, show_cache_badge
from kde_ref.moments import posterior_grid_bounds
from kde_ref.reference_adapter import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_BACKENDS,
    DEFAULT_STEP1_CSV,
    audit_command,
    build_posterior_density_grid,
    build_reference_summaries_from_samples,
    load_or_simulate_mle_errors,
    load_reference_summary,
    mle_sample_cache_path,
)

STUDENT_N_VALUES = (10, 20, 50)
LAPLACE_N_VALUES = (11, 21, 51)
SUMMARY_COLUMNS = [
    "method",
    "estimator_type",
    "backend",
    "n",
    "mean",
    "sd",
    "var",
    "q025",
    "q50",
    "q975",
    "marginal_likelihood_estimate",
    "posterior_integral_check",
    "weighted_ess",
    "source_file",
    "plot_grid_lo",
    "plot_grid_hi",
    "plot_grid_size",
    "loaded_from_audit_csv",
    "used_cached_mle_samples",
    "recomputed_in_ui",
    "audit_csv_path",
    "mle_cache_path",
]
DIFFERENCE_COLUMNS = [
    "method",
    "estimator_type",
    "backend",
    "n",
    "k",
    "mu_star",
    "B",
    "seed",
    "delta_mean",
    "delta_sd",
    "rel_sd_error",
    "delta_q025",
    "delta_q50",
    "delta_q975",
    "marginal_likelihood_estimate",
    "posterior_integral_check",
    "source_file",
    "plot_grid_lo",
    "plot_grid_hi",
    "plot_grid_size",
    "loaded_from_audit_csv",
    "used_cached_mle_samples",
    "recomputed_in_ui",
    "audit_csv_path",
    "mle_cache_path",
]
EXPORT_STATUS_COLUMNS = [
    "loaded_from_audit_csv",
    "used_cached_mle_samples",
    "recomputed_in_ui",
    "audit_csv_path",
    "mle_cache_path",
]
DEFAULT_SAMPLER_RESULTS_DIR = Path("results/cost_audit")
ANALYSIS_REPORT_PATH = Path("results/analysis_report/executive_summary.md")
RATTLE_SETTINGS_PATH = Path("results/rattle_tuning/recommended_rattle_settings.json")
DEFAULT_ALL_MODEL_REFERENCE_CSV = Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv")
ALL_MODEL_REFERENCE_SMOKE_CSV = Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models_smoke.csv")
REFERENCE_LEVEL_PATHS = {
    "smoke": ALL_MODEL_REFERENCE_SMOKE_CSV,
    "full": DEFAULT_ALL_MODEL_REFERENCE_CSV,
}
SAMPLER_LEVEL_PATHS = {
    "smoke": Path("results/cost_audit_smoke"),
    "medium": Path("results/cost_audit_medium"),
    "full": Path("results/cost_audit"),
    "multiseed": Path("results/cost_audit_multiseed"),
}
SAMPLER_AUDIT_COMMAND = (
    "python scripts/run_cost_audit.py --methods gibbs rattle --n-values 10,20,50 "
    "--laplace-n-values 11,21,51 --k 2 --mu-star 0 --num-iterations 10000 --burn-in 2000 --seed 0 --out results/cost_audit/"
)
ALL_MODEL_REFERENCE_COMMAND = (
    "python reporting/diagnostics/audit_reference_all_models.py --models student_t logistic laplace "
    "--k-values 1,2,3 --n-values 10,20,50 --laplace-n-values 11,21,51 --B-values 100000 --seeds 123,456,789 "
    "--bandwidths scott,SJ_transform --out-csv reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"
)


def auto_best_reference_path() -> Path:
    for level in ["full", "smoke"]:
        path = REFERENCE_LEVEL_PATHS[level]
        if path.exists():
            return path
    return DEFAULT_ALL_MODEL_REFERENCE_CSV


def auto_best_sampler_dir() -> Path:
    for level in ["full", "medium", "smoke"]:
        path = SAMPLER_LEVEL_PATHS[level]
        if (path / "cost_ledger.csv").exists():
            return path
    return DEFAULT_SAMPLER_RESULTS_DIR


def rattle_settings_status(path: Path = RATTLE_SETTINGS_PATH) -> str:
    if not path.exists():
        return "missing"
    data = json.loads(path.read_text(encoding="utf-8"))
    statuses = sorted({str(row.get("status", "")) for row in data.get("settings", []) if row.get("status")})
    return ", ".join(statuses) if statuses else "unknown"


@st.cache_data(show_spinner=False)
def cached_z_samples(k: float, n: int, B: int, seed: int) -> np.ndarray:
    return load_or_simulate_mle_errors(k=k, n=n, B=B, seed=seed, audit_dir=DEFAULT_AUDIT_DIR)


@st.cache_data(show_spinner=False)
def cached_density_grid(
    z_samples: np.ndarray,
    k: float,
    n: int,
    mu_star: float,
    prior_mean: float,
    prior_std: float,
    B: int,
    seed: int,
    backends: tuple[str, ...],
    grid_size: int,
    bound_multiplier: float,
) -> pd.DataFrame:
    return build_posterior_density_grid(
        z_samples=z_samples,
        k=k,
        n=n,
        mu_star=mu_star,
        prior_mean=prior_mean,
        prior_std=prior_std,
        B=B,
        seed=seed,
        backends=backends,
        grid_size=grid_size,
        bound_multiplier=bound_multiplier,
    )


@st.cache_data(show_spinner=False)
def cached_summaries(
    z_samples: np.ndarray,
    k: float,
    n: int,
    mu_star: float,
    prior_mean: float,
    prior_std: float,
    B: int,
    seed: int,
    backends: tuple[str, ...],
    include_kde_grid: bool,
    include_kde_quad: bool,
    grid_size: int,
    bound_multiplier: float,
) -> pd.DataFrame:
    return build_reference_summaries_from_samples(
        z_samples=z_samples,
        k=k,
        n=n,
        mu_star=mu_star,
        prior_mean=prior_mean,
        prior_std=prior_std,
        B=B,
        seed=seed,
        backends=backends,
        include_raw=True,
        include_kde_grid=include_kde_grid,
        include_kde_quad=include_kde_quad,
        grid_size=grid_size,
        bound_multiplier=bound_multiplier,
    )


@st.cache_data(show_spinner=False)
def cached_audit_summary(
    csv_path: str,
    k: float,
    ns: tuple[int, ...],
    B: int,
    seed: int,
    backends: tuple[str, ...],
    mu_star: float,
) -> pd.DataFrame:
    return load_reference_summary(
        csv_path=Path(csv_path),
        k_values=[float(k)],
        n_values=[int(n) for n in ns],
        B_values=[int(B)],
        seeds=[int(seed)],
        backends=backends,
        mu_star=float(mu_star),
    )


def selected_backends() -> tuple[str, ...]:
    chosen = []
    for backend in DEFAULT_BACKENDS:
        label = "KDE t_abram, capped diagnostic" if backend == "t_abram" else f"KDE {backend}"
        default = backend != "t_abram"
        if st.sidebar.checkbox(label, value=default, key=f"backend_{backend}"):
            chosen.append(backend)
    return tuple(chosen)


def plot_xlim_from_raw(summary_df: pd.DataFrame, density_df: pd.DataFrame, show_full_grid: bool) -> tuple[float, float]:
    if show_full_grid:
        return float(density_df["mu"].min()), float(density_df["mu"].max())

    raw = summary_df[summary_df["estimator_type"].eq("raw_weighted_mc")]
    if raw.empty:
        return float(density_df["mu"].min()), float(density_df["mu"].max())

    plot_los = raw["q025"] - 0.25 * (raw["q975"] - raw["q025"])
    plot_his = raw["q975"] + 0.25 * (raw["q975"] - raw["q025"])
    return float(plot_los.min()), float(plot_his.max())


def with_plot_metadata(df: pd.DataFrame, plot_lo: float, plot_hi: float, plot_grid_size: int) -> pd.DataFrame:
    out = df.copy()
    out["plot_grid_lo"] = float(plot_lo)
    out["plot_grid_hi"] = float(plot_hi)
    out["plot_grid_size"] = int(plot_grid_size)
    return out


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out


MODEL_LABELS = {"student_t": "Student-t", "logistic": "Logistic", "laplace": "Laplace"}
METHOD_LABELS = {
    "raw_weighted_mc": "Raw weighted-MC",
    "raw_mc_interval_reference": "Raw interval reference",
    "kde_grid": "KDE smoothed density",
    "kde_quad": "KDE quadrature summary",
    "sampler": "Sampler",
}


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model))


def method_label(value: str) -> str:
    return METHOD_LABELS.get(str(value), str(value))


def display_backend_label(row: pd.Series) -> str:
    backend = str(row.get("backend", ""))
    if backend == "median_interval":
        return "median interval reference"
    if backend == "t_abram" and bool(row.get("density_sample_capped", False)):
        b_used = row.get("B_used", row.get("density_sample_size", np.nan))
        if pd.notna(b_used):
            return f"t_abram, plot-only B={int(float(b_used))} cap"
        return "t_abram, plot-only cap"
    return backend


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model" in out.columns:
        out["model"] = out["model"].map(model_label)
    if "estimator_type" in out.columns:
        out["method"] = out["estimator_type"].map(method_label)
    elif "method" in out.columns:
        out["method"] = out["method"].map(method_label)
    if "backend" in out.columns:
        out["backend"] = out.apply(display_backend_label, axis=1)
    if "density_note" in out.columns and "note" not in out.columns:
        out["note"] = out["density_note"]
    return out


def default_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return ensure_columns(display_table(df), columns)[columns]


def with_status_metadata(df: pd.DataFrame, status_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for column in EXPORT_STATUS_COLUMNS:
            out[column] = np.nan
        return out

    status_cols = ["n", *EXPORT_STATUS_COLUMNS]
    out = out.drop(columns=EXPORT_STATUS_COLUMNS, errors="ignore")
    return out.merge(status_df[status_cols], on="n", how="left")


def make_plot_density_grid(
    density_df: pd.DataFrame,
    *,
    plot_lo: float,
    plot_hi: float,
    plot_grid_size: int,
    show_full_grid: bool,
) -> pd.DataFrame:
    if show_full_grid:
        return with_plot_metadata(density_df, float(density_df["mu"].min()), float(density_df["mu"].max()), len(density_df["mu"].unique()))

    rows = []
    plot_grid = np.linspace(float(plot_lo), float(plot_hi), int(plot_grid_size))
    for _, part in density_df.groupby(["n", "backend"], sort=False):
        part = part.sort_values("mu")
        row = part.iloc[0].to_dict()
        density = np.interp(plot_grid, part["mu"], part["density"], left=0.0, right=0.0)
        cdf = np.interp(plot_grid, part["mu"], part["cdf"], left=0.0, right=1.0)
        frame = pd.DataFrame(
            {
                **{key: row[key] for key in part.columns if key not in {"mu", "density", "cdf"}},
                "mu": plot_grid,
                "density": density,
                "cdf": cdf,
            }
        )
        rows.append(with_plot_metadata(frame, float(plot_lo), float(plot_hi), int(plot_grid_size)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=density_df.columns)


def add_plot_labels(density_df: pd.DataFrame, overlay_mode: str) -> pd.DataFrame:
    out = density_df.copy()
    sampler_mask = out.get("estimator_type", pd.Series("", index=out.index)).eq("sampler_density")
    if overlay_mode in {"compare n values for fixed backend", "compare n values for fixed method"}:
        out["plot_label"] = out.get("backend", pd.Series("", index=out.index)).astype(str) + " n=" + out["n"].astype(str)
        out.loc[sampler_mask, "plot_label"] = out.loc[sampler_mask, "method"].astype(str) + " n=" + out.loc[sampler_mask, "n"].astype(str)
    elif overlay_mode == "compare seeds for fixed density type":
        out["plot_label"] = "seed=" + out.get("seed", pd.Series("", index=out.index)).astype(str)
    else:
        out["plot_label"] = out.apply(display_backend_label, axis=1)
        out.loc[sampler_mask, "plot_label"] = out.loc[sampler_mask, "method"].astype(str)
    return out


def posterior_mass_xlim(summary_df: pd.DataFrame) -> tuple[float, float] | None:
    if summary_df.empty or not {"q025", "q975"}.issubset(summary_df.columns):
        return None
    raw = summary_df[
        summary_df.get("estimator_type", pd.Series("", index=summary_df.index)).astype(str).isin(
            ["raw_weighted_mc", "raw_mc_interval_reference"]
        )
    ].copy()
    if raw.empty:
        return None
    raw["q025"] = pd.to_numeric(raw["q025"], errors="coerce")
    raw["q975"] = pd.to_numeric(raw["q975"], errors="coerce")
    raw = raw[np.isfinite(raw["q025"]) & np.isfinite(raw["q975"])]
    raw = raw[raw["q975"] > raw["q025"]]
    if raw.empty:
        return None
    width = raw["q975"] - raw["q025"]
    lo = float((raw["q025"] - 0.25 * width).min())
    hi = float((raw["q975"] + 0.25 * width).max())
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else None


def plot_density_overlay(
    plot_density_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    overlay_mode: str,
    raw_marker_mode: str,
    selected_marker_n: int,
    show_log_density: bool,
    visible_xlim: tuple[float, float] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = add_plot_labels(plot_density_df, overlay_mode)
    if {"plot_grid_lo", "plot_grid_hi"}.issubset(plot_density_df.columns):
        lo_values = pd.to_numeric(plot_density_df["plot_grid_lo"], errors="coerce")
        hi_values = pd.to_numeric(plot_density_df["plot_grid_hi"], errors="coerce")
        finite_lo = lo_values[np.isfinite(lo_values)]
        finite_hi = hi_values[np.isfinite(hi_values)]
        plot_lo = float(finite_lo.min()) if not finite_lo.empty else float(plot_density_df["mu"].min())
        plot_hi = float(finite_hi.max()) if not finite_hi.empty else float(plot_density_df["mu"].max())
    else:
        plot_lo = float(plot_density_df["mu"].min())
        plot_hi = float(plot_density_df["mu"].max())

    for label, part in plot_df.groupby("plot_label", sort=False):
        y = np.log(np.maximum(part["density"].to_numpy(dtype=float), 1e-300)) if show_log_density else part["density"]
        is_sampler = part.get("estimator_type", pd.Series(dtype=str)).astype(str).eq("sampler_density").any()
        if is_sampler and not show_log_density:
            ax.fill_between(part["mu"], y, alpha=0.22, step="mid", label=str(label))
            ax.plot(part["mu"], y, alpha=0.45, linewidth=1.0)
        else:
            ax.plot(part["mu"], y, label=str(label), linewidth=2 if not is_sampler else 1.4, alpha=0.9 if not is_sampler else 0.65)

    if raw_marker_mode != "none" and not summary_df.empty:
        raw = summary_df[
            summary_df["estimator_type"].isin(["raw_weighted_mc", "raw_mc_interval_reference"])
        ]
        if raw_marker_mode == "selected n only":
            raw = raw[raw["n"].astype(int).eq(int(selected_marker_n))]
        raw = raw.sort_values(["estimator_type", "n"]).drop_duplicates(
            subset=[col for col in ["estimator_type", "n", "k", "mu_star", "seed"] if col in raw.columns]
        )
        legend_labels_used: set[str] = set()
        for _, row in raw.iterrows():
            if overlay_mode in {"compare n values for fixed backend", "compare n values for fixed method"}:
                suffix = f" n={int(row['n'])}"
            elif overlay_mode == "compare seeds for fixed density type":
                suffix = f" seed={int(row['seed'])}" if pd.notna(row.get("seed", np.nan)) else ""
            else:
                suffix = ""
            marker_label = "interval reference" if row.get("estimator_type", "") == "raw_mc_interval_reference" else "raw"
            mean_label = f"{marker_label} mean{suffix}"
            interval_label = f"{marker_label} 95% interval{suffix}"
            median_label = f"{marker_label} median{suffix}"
            ax.axvline(row["mean"], color="black", linestyle="-", linewidth=1.2, label=mean_label if mean_label not in legend_labels_used else None)
            legend_labels_used.add(mean_label)
            ax.axvline(row["q025"], color="black", linestyle=":", linewidth=1.0, label=interval_label if interval_label not in legend_labels_used else None)
            legend_labels_used.add(interval_label)
            ax.axvline(row["q975"], color="black", linestyle=":", linewidth=1.0)
            ax.axvline(row["q50"], color="black", linestyle="--", linewidth=1.0, label=median_label if median_label not in legend_labels_used else None)
            legend_labels_used.add(median_label)

    if visible_xlim is not None and np.isfinite(visible_xlim[0]) and np.isfinite(visible_xlim[1]) and visible_xlim[1] > visible_xlim[0]:
        plot_lo, plot_hi = visible_xlim
    ax.set_xlim(plot_lo, plot_hi)
    ax.set_xlabel("mu")
    ax.set_ylabel("log posterior density" if show_log_density else "posterior density")
    ax.set_title("Posterior density")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def density_diagnostics(density_df: pd.DataFrame, plot_density_df: pd.DataFrame, grid_size: int, bound_multiplier: float) -> pd.DataFrame:
    rows = []
    for keys, part in density_df.groupby(["n", "backend"], sort=False):
        n_value, backend = keys
        plot_part = plot_density_df[(plot_density_df["n"].eq(n_value)) & (plot_density_df["backend"].eq(backend))]
        mu = plot_part["mu"].to_numpy(dtype=float)
        dx = float(np.nanmedian(np.diff(mu))) if mu.size > 1 else np.nan
        rows.append(
            {
                "n": n_value,
                "backend": backend,
                "marginal_likelihood_estimate": float(part["marginal_likelihood_estimate"].iloc[0]),
                "posterior_integral_check": float(np.trapezoid(part["density"], part["mu"])),
                "B": int(part["B"].iloc[0]),
                "seed": int(part["seed"].iloc[0]),
                "computational_grid_size": int(grid_size),
                "plot_grid_size": int(plot_part["plot_grid_size"].iloc[0]) if not plot_part.empty else np.nan,
                "plot_grid_lo": float(plot_part["plot_grid_lo"].iloc[0]) if not plot_part.empty else np.nan,
                "plot_grid_hi": float(plot_part["plot_grid_hi"].iloc[0]) if not plot_part.empty else np.nan,
                "bound_multiplier": float(bound_multiplier),
                "plot_dx": dx,
            }
        )
    return pd.DataFrame(rows)


def difference_from_raw(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=DIFFERENCE_COLUMNS)

    keys = ["n", "k", "mu_star", "B", "seed"]
    raw = summary_df[summary_df["estimator_type"].eq("raw_weighted_mc")]
    kde = summary_df[~summary_df["estimator_type"].eq("raw_weighted_mc")]
    if raw.empty or kde.empty:
        return pd.DataFrame(columns=DIFFERENCE_COLUMNS)

    raw_cols = keys + ["mean", "sd", "q025", "q50", "q975"]
    merged = kde.merge(raw[raw_cols], on=keys, suffixes=("", "_raw"), how="inner")
    merged["delta_mean"] = merged["mean"] - merged["mean_raw"]
    merged["delta_sd"] = merged["sd"] - merged["sd_raw"]
    merged["rel_sd_error"] = merged["delta_sd"] / merged["sd_raw"].replace(0.0, np.nan)
    merged["delta_q025"] = merged["q025"] - merged["q025_raw"]
    merged["delta_q50"] = merged["q50"] - merged["q50_raw"]
    merged["delta_q975"] = merged["q975"] - merged["q975_raw"]
    return merged


def tail_diagnostics(density_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    raw = summary_df[summary_df["estimator_type"].eq("raw_weighted_mc")]
    if raw.empty or density_df.empty:
        return pd.DataFrame(
            columns=["backend", "n", "left_tail_at_raw_q025", "right_tail_at_raw_q975", "central_mass_raw_95_interval"]
        )

    rows = []
    raw_by_n = {int(row.n): row for row in raw.itertuples()}
    for keys, part in density_df.groupby(["n", "backend"], sort=False):
        n_value, backend = keys
        raw_row = raw_by_n.get(int(n_value))
        if raw_row is None:
            continue
        part = part.sort_values("mu")
        cdf_q025 = float(np.interp(raw_row.q025, part["mu"], part["cdf"], left=0.0, right=1.0))
        cdf_q975 = float(np.interp(raw_row.q975, part["mu"], part["cdf"], left=0.0, right=1.0))
        rows.append(
            {
                "backend": backend,
                "n": int(n_value),
                "left_tail_at_raw_q025": cdf_q025,
                "right_tail_at_raw_q975": 1.0 - cdf_q975,
                "central_mass_raw_95_interval": cdf_q975 - cdf_q025,
            }
        )
    return pd.DataFrame(rows)


def attach_density_checks(summary_df: pd.DataFrame, density_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or density_df.empty:
        return summary_df

    checks = (
        density_df.groupby(["n", "backend"], as_index=False)
        .agg(
            posterior_integral_check=("posterior_integral_check", "first"),
            density_marginal_likelihood_estimate=("marginal_likelihood_estimate", "first"),
        )
    )
    out = summary_df.drop(columns=["posterior_integral_check"], errors="ignore").merge(
        checks[["n", "backend", "posterior_integral_check"]],
        on=["n", "backend"],
        how="left",
    )
    return out


def selected_summary_rows(summary_df: pd.DataFrame, *, show_raw: bool, show_kde_grid: bool, show_kde_quad: bool) -> pd.DataFrame:
    keep = pd.Series(False, index=summary_df.index)
    if show_raw:
        keep = keep | summary_df["estimator_type"].eq("raw_weighted_mc")
    if show_kde_grid:
        keep = keep | summary_df["estimator_type"].eq("kde_grid")
    if show_kde_quad:
        keep = keep | summary_df["estimator_type"].eq("kde_quad")
    return summary_df[keep].reset_index(drop=True)


def summarize_status(values: pd.Series) -> str:
    unique = [str(value) for value in values.dropna().unique()]
    if not unique:
        return "unavailable"
    if len(unique) == 1:
        return unique[0]
    return "mixed"


@st.cache_data(show_spinner=False)
def cached_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()


def rename_likely_columns(df: pd.DataFrame, mapping: dict[str, tuple[str, ...]]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    warnings = []
    lower_to_original = {str(column).lower(): column for column in out.columns}
    for canonical, candidates in mapping.items():
        if canonical in out.columns:
            continue
        for candidate in candidates:
            found = lower_to_original.get(candidate.lower())
            if found is not None:
                out = out.rename(columns={found: canonical})
                warnings.append(f"Using {found!r} as {canonical!r}.")
                break
    return out, warnings


def load_sampler_outputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    chain = cached_csv(str(results_dir / "chain_samples.csv"))
    summaries = cached_csv(str(results_dir / "posterior_summaries.csv"))
    diagnostics = cached_csv(str(results_dir / "diagnostic_summary.csv"))
    ledger = cached_csv(str(results_dir / "cost_ledger.csv"))
    warnings = []
    chain, chain_warnings = rename_likely_columns(
        chain,
        {
            "method": ("sampler", "algorithm"),
            "n": ("sample_size", "num_observations"),
            "k": ("df", "student_df"),
            "mu_star": ("mle", "mu_hat", "observed_mle"),
            "seed": ("rng_seed",),
            "iteration": ("iter", "draw", "draw_id"),
            "mu": ("mu_sample", "sample_mu", "theta"),
        },
    )
    warnings.extend(chain_warnings)
    return chain, summaries, diagnostics, ledger, warnings


def sampler_smoke_status(ledger: pd.DataFrame, chain: pd.DataFrame) -> str:
    n_values = set()
    if not ledger.empty and "n" in ledger:
        n_values.update(ledger["n"].dropna().astype(int).tolist())
    if not chain.empty and "n" in chain:
        n_values.update(chain["n"].dropna().astype(int).tolist())
    iterations = []
    if not ledger.empty and "iterations" in ledger:
        iterations = ledger["iterations"].dropna().astype(float).tolist()
    if iterations and min(iterations) < 1000:
        return "smoke"
    expected_n_values = set(STUDENT_N_VALUES) | set(LAPLACE_N_VALUES)
    if n_values and not n_values.issubset(expected_n_values):
        return "smoke"
    if not iterations and not chain.empty and len(chain) < 1000:
        return "smoke"
    return "full_or_unknown"


def filter_sampler_chain(
    chain: pd.DataFrame,
    *,
    selected_ns: tuple[int, ...],
    k: float,
    mu_star: float,
    seed: int,
    selected_methods: set[str],
) -> pd.DataFrame:
    required = {"method", "n", "mu"}
    if chain.empty or not required.issubset(chain.columns):
        return pd.DataFrame()
    out = chain.copy()
    out = out[out["method"].astype(str).str.lower().isin(selected_methods)]
    out = out[out["n"].astype(int).isin([int(n) for n in selected_ns])]
    if "k" in out.columns and np.isfinite(k):
        out = out[np.isclose(out["k"].astype(float), float(k))]
    if "mu_star" in out.columns:
        out = out[np.isclose(out["mu_star"].astype(float), float(mu_star))]
    if "seed" in out.columns:
        out = out[out["seed"].astype(int).eq(int(seed))]
    if "is_burn_in" in out.columns:
        out = out[~out["is_burn_in"].astype(bool)]
    return out.reset_index(drop=True)


def sampler_density_frames(
    chain: pd.DataFrame,
    *,
    plot_lo: float,
    plot_hi: float,
    plot_grid_size: int,
    source_file: Path,
) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    rows = []
    grid = np.linspace(float(plot_lo), float(plot_hi), int(plot_grid_size))
    for (method, n_value), part in chain.groupby(["method", "n"], sort=False):
        samples = part["mu"].to_numpy(dtype=float)
        samples = samples[np.isfinite(samples)]
        if samples.size < 2:
            continue
        try:
            kde = stats.gaussian_kde(samples)
            density = np.asarray(kde(grid), dtype=float)
            density_method = "gaussian_kde_scott"
        except Exception:
            bins = min(max(int(np.sqrt(samples.size)), 20), 80)
            hist, edges = np.histogram(samples, bins=bins, range=(float(plot_lo), float(plot_hi)), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            density = np.interp(grid, centers, hist, left=0.0, right=0.0)
            density_method = "histogram_fallback"
        integral = float(np.trapezoid(density, grid))
        if integral > 0:
            density = density / integral
        cdf = np.concatenate([[0.0], np.cumsum((density[:-1] + density[1:]) * np.diff(grid) / 2.0)])
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        first = part.iloc[0]
        rows.append(
            pd.DataFrame(
                {
                    "method": str(method),
                    "estimator_type": "sampler_density",
                    "backend": str(method).lower(),
                    "n": int(n_value),
                    "k": float(first["k"]) if "k" in part.columns else np.nan,
                    "mu_star": float(first["mu_star"]) if "mu_star" in part.columns else np.nan,
                    "mu": grid,
                    "density": density,
                    "cdf": cdf,
                    "marginal_likelihood_estimate": np.nan,
                    "posterior_integral_check": float(np.trapezoid(density, grid)),
                    "density_method": density_method,
                    "density_note": "Display-only KDE smoothing of Gibbs/RATTLE chain samples.",
                    "seed": int(first["seed"]) if "seed" in part.columns else np.nan,
                    "B": np.nan,
                    "source_file": str(source_file),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sampler_summary_from_chains(chain: pd.DataFrame, *, source_file: Path) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    rows = []
    for (method, n_value), part in chain.groupby(["method", "n"], sort=False):
        samples = part["mu"].to_numpy(dtype=float)
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            continue
        first = part.iloc[0]
        mean = float(np.mean(samples))
        var = float(np.var(samples))
        rows.append(
            {
                "method": str(method),
                "estimator_type": "sampler",
                "backend": str(method).lower(),
                "n": int(n_value),
                "k": float(first["k"]) if "k" in part.columns else np.nan,
                "mu_star": float(first["mu_star"]) if "mu_star" in part.columns else np.nan,
                "seed": int(first["seed"]) if "seed" in part.columns else np.nan,
                "mean": mean,
                "var": var,
                "sd": float(np.sqrt(max(var, 0.0))),
                "q025": float(np.quantile(samples, 0.025)),
                "q50": float(np.quantile(samples, 0.5)),
                "q975": float(np.quantile(samples, 0.975)),
                "ess_mu": np.nan,
                "ess_per_sec": np.nan,
                "acceptance_rate": np.nan,
                "source_file": str(source_file),
            }
        )
    return pd.DataFrame(rows)


def sampler_summary_from_audit(summaries: pd.DataFrame, diagnostics: pd.DataFrame, *, selected_methods: set[str], source_file: Path) -> pd.DataFrame:
    if summaries.empty or "method" not in summaries.columns:
        return pd.DataFrame()
    out = summaries.copy()
    out = out[out["method"].astype(str).str.lower().isin(selected_methods)]
    rename = {
        "mean_mu": "mean",
        "var_mu": "var",
        "sd_mu": "sd",
        "q025_mu": "q025",
        "q50_mu": "q50",
        "q975_mu": "q975",
    }
    out = out.rename(columns={key: value for key, value in rename.items() if key in out.columns})
    if not diagnostics.empty:
        diag_cols = [col for col in ["method", "n", "k", "mu_star", "seed", "ess_mu", "ess_per_sec", "acceptance_rate"] if col in diagnostics.columns]
        merge_keys = [col for col in ["method", "n", "k", "mu_star", "seed"] if col in out.columns and col in diagnostics.columns]
        if merge_keys and diag_cols:
            out = out.drop(columns=["ess_per_sec", "acceptance_rate"], errors="ignore").merge(
                diagnostics[diag_cols].drop_duplicates(),
                on=merge_keys,
                how="left",
            )
    out["estimator_type"] = "sampler"
    out["backend"] = out["method"].astype(str).str.lower()
    out["source_file"] = str(source_file)
    return out


def comparison_summary_table(reference_rows: pd.DataFrame, sampler_rows: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    columns = [
        "method",
        "n",
        "k",
        "mu_star",
        "seed",
        "mean",
        "sd",
        "var",
        "q025",
        "q50",
        "q975",
        "ess_mu",
        "ess_per_sec",
        "acceptance_rate",
        "source_file",
        "delta_mean",
        "delta_sd",
        "rel_sd_error",
        "delta_q025",
        "delta_q50",
        "delta_q975",
    ]
    rows = []
    if not reference_rows.empty:
        ref = reference_rows.copy()
        ref["method"] = np.where(
            ref["estimator_type"].isin(["raw_weighted_mc", "raw_mc_interval_reference"]),
            "raw_weighted_mc",
            "kde_" + ref["backend"].astype(str),
        )
        rows.append(ref)
    if not sampler_rows.empty:
        rows.append(sampler_rows)
    if not rows:
        return pd.DataFrame(columns=columns), "none"
    out = pd.concat(rows, ignore_index=True)
    out = ensure_columns(out, columns[:-6])
    baseline = out[out["method"].astype(str).eq("raw_weighted_mc")]
    baseline_label = "raw_weighted_mc"
    if baseline.empty:
        baseline = out[out["method"].astype(str).eq("kde_scott")]
        baseline_label = "kde_scott_fallback"
    if baseline.empty:
        return ensure_columns(out, columns)[columns], "none"
    key_cols = [col for col in ["n", "k", "mu_star", "seed"] if col in out.columns and col in baseline.columns]
    base_cols = key_cols + ["mean", "sd", "q025", "q50", "q975"]
    merged = out.merge(baseline[base_cols], on=key_cols, how="left", suffixes=("", "_baseline"))
    merged["delta_mean"] = merged["mean"] - merged["mean_baseline"]
    merged["delta_sd"] = merged["sd"] - merged["sd_baseline"]
    merged["rel_sd_error"] = merged["delta_sd"] / merged["sd_baseline"].replace(0.0, np.nan)
    merged["delta_q025"] = merged["q025"] - merged["q025_baseline"]
    merged["delta_q50"] = merged["q50"] - merged["q50_baseline"]
    merged["delta_q975"] = merged["q975"] - merged["q975_baseline"]
    return ensure_columns(merged, columns)[columns], baseline_label


st.title("Posterior Comparison")
st.caption(
    "Raw weighted-MC is the moment/quantile reference. "
    "KDE curves are smoothed density estimates for visualization/backend sensitivity."
)
use_dashboard_cache, dashboard_cache_dir, dashboard_manifest = sidebar_cache_controls("posterior")
show_cache_badge(use_dashboard_cache, dashboard_cache_dir, dashboard_manifest)

if use_dashboard_cache:
    reference_path = require_cache_file(dashboard_cache_dir, "reference_cache.csv")
    posterior_density_path = require_cache_file(dashboard_cache_dir, "posterior_density_cache.csv")
    sampler_density_path = require_cache_file(dashboard_cache_dir, "sampler_density_cache.csv")
    density_status_path = require_cache_file(dashboard_cache_dir, "density_cache_status.csv")
    comparison_path = require_cache_file(dashboard_cache_dir, "posterior_comparison_cache.csv")
    cost_path = require_cache_file(dashboard_cache_dir, "cost_efficiency_cache.csv")
    figures_path = require_cache_file(dashboard_cache_dir, "figure_index.csv")
    if not all([reference_path, posterior_density_path, sampler_density_path, density_status_path, comparison_path, cost_path, figures_path]):
        st.stop()

    reference_cache = read_cache_csv(str(dashboard_cache_dir), "reference_cache.csv")
    posterior_density_cache = read_cache_csv(str(dashboard_cache_dir), "posterior_density_cache.csv")
    sampler_density_cache = read_cache_csv(str(dashboard_cache_dir), "sampler_density_cache.csv")
    density_status_cache = read_cache_csv(str(dashboard_cache_dir), "density_cache_status.csv")
    comparison_cache = read_cache_csv(str(dashboard_cache_dir), "posterior_comparison_cache.csv")
    cost_cache = read_cache_csv(str(dashboard_cache_dir), "cost_efficiency_cache.csv")
    figures_cache = read_cache_csv(str(dashboard_cache_dir), "figure_index.csv")
    reference_seed_values = sorted(reference_cache["seed"].dropna().astype(int).unique().tolist()) if "seed" in reference_cache.columns else [123]
    sampler_seed_values = sorted(sampler_density_cache["seed"].dropna().astype(int).unique().tolist()) if "seed" in sampler_density_cache.columns else [0]

    with st.sidebar:
        st.header("Cached Posterior View")
        model_choice_cached = st.selectbox(
            "model",
            ["student_t", "logistic", "laplace"],
            index=0,
            format_func=model_label,
            key="cached_model",
        )
        k_cached = st.selectbox("k", [1.0, 2.0, 3.0], index=1, disabled=model_choice_cached != "student_t", key="cached_k")
        n_options_cached = list(LAPLACE_N_VALUES if model_choice_cached == "laplace" else STUDENT_N_VALUES)
        cached_overlay_mode = st.radio(
            "overlay mode",
            ["compare methods for fixed n", "compare n values for fixed method", "compare seeds for fixed density type"],
            index=0,
            key="cached_overlay_mode",
        )
        n_cached = st.selectbox(
            "n",
            n_options_cached,
            index=1,
            disabled=cached_overlay_mode == "compare n values for fixed method",
            key="cached_n",
        )
        reference_seed_cached = st.selectbox(
            "reference seed (raw/KDE)",
            reference_seed_values,
            index=reference_seed_values.index(123) if 123 in reference_seed_values else 0,
            key="cached_reference_seed",
        )
        sampler_seed_cached = st.selectbox(
            "sampler seed (Gibbs/RATTLE)",
            sampler_seed_values,
            index=sampler_seed_values.index(0) if 0 in sampler_seed_values else 0,
            key="cached_sampler_seed",
        )
        st.caption("Reference seeds affect raw/KDE curves; sampler seeds affect Gibbs/RATTLE curves.")
        laplace_selected = model_choice_cached == "laplace"
        laplace_even_cached = laplace_selected and int(n_cached) % 2 == 0
        show_scott_cached = st.checkbox("KDE scott", value=not laplace_even_cached, disabled=laplace_even_cached, key="cached_scott")
        show_sj_cached = st.checkbox("KDE SJ_transform", value=not laplace_even_cached, disabled=laplace_even_cached, key="cached_sj")
        show_tabram_cached = st.checkbox("KDE t_abram, capped diagnostic", value=False, disabled=laplace_even_cached, key="cached_tabram")
        if laplace_even_cached:
            st.caption("Laplace reference curve: median interval contains mu_star.")
        elif laplace_selected:
            st.caption("Laplace odd-n reference: deterministic sample median KDE.")
        seed_density_family_cached = None
        if cached_overlay_mode == "compare seeds for fixed density type":
            if laplace_even_cached:
                seed_family_options = ["median_interval", "gibbs"]
            elif laplace_selected:
                seed_family_options = ["KDE scott", "KDE SJ_transform", "KDE t_abram", "gibbs"]
            else:
                seed_family_options = ["KDE scott", "KDE SJ_transform", "KDE t_abram", "gibbs", "rattle"]
            seed_density_family_cached = st.selectbox(
                "density type",
                seed_family_options,
                index=0,
                format_func=lambda value: "median interval reference" if value == "median_interval" else str(value),
                key="cached_seed_density_family",
            )
        show_gibbs_cached = st.checkbox("Gibbs", value=True, key="cached_gibbs")
        show_rattle_cached = st.checkbox("RATTLE", value=model_choice_cached != "laplace", disabled=model_choice_cached == "laplace", key="cached_rattle")
        cached_raw_marker_mode = st.selectbox(
            "Raw reference markers",
            ["none", "selected n only", "all n"],
            index=1,
            key="cached_raw_marker_mode",
        )
        show_log_density_cached = st.checkbox("Show log density", value=False, key="cached_log_density")
        show_full_grid_cached = st.checkbox("Show full computational grid", value=False, key="cached_show_full_grid")

    if model_choice_cached == "laplace" and int(n_cached) % 2 == 0:
        st.info("Laplace deterministic np.median KDE/raw-MC references are hidden by default because they target a different even-n convention.")
        st.warning("Laplace exact RATTLE is not applicable because the median/order constraint is nonsmooth.")
        st.info("Laplace Gibbs is compared to median_interval_contains_mu_star.")
    elif model_choice_cached == "laplace":
        st.info("Laplace odd-n comparison uses the unique sample median target; deterministic median KDE/raw-MC and Gibbs are directly comparable.")
    if model_choice_cached == "student_t" and float(k_cached) == 1.0 and int(n_cached) == 10:
        st.warning("Student k=1,n=10 unresolved: see Analysis Report for score-root vs selected-MLE diagnostics.")

    selected_backends_cached = []
    if show_scott_cached:
        selected_backends_cached.append("scott")
    if show_sj_cached:
        selected_backends_cached.append("SJ_transform")
    if show_tabram_cached:
        selected_backends_cached.append("t_abram")
    if model_choice_cached == "laplace" and int(n_cached) % 2 == 0:
        selected_backends_cached = ["median_interval"]
    sampler_methods_cached = []
    if show_gibbs_cached:
        sampler_methods_cached.append("gibbs")
    if show_rattle_cached:
        sampler_methods_cached.append("rattle")
    if cached_overlay_mode == "compare seeds for fixed density type" and seed_density_family_cached is not None:
        if str(seed_density_family_cached).startswith("KDE "):
            selected_backends_cached = [str(seed_density_family_cached).replace("KDE ", "", 1)]
            sampler_methods_cached = []
        elif str(seed_density_family_cached) == "median_interval":
            selected_backends_cached = ["median_interval"]
            sampler_methods_cached = []
        else:
            selected_backends_cached = []
            sampler_methods_cached = [str(seed_density_family_cached)]
    fixed_family_cached = None
    if cached_overlay_mode == "compare n values for fixed method":
        selected_families = [f"KDE {backend}" for backend in selected_backends_cached] + [method for method in sampler_methods_cached]
        if selected_families:
            default_family = "gibbs" if "gibbs" in selected_families else selected_families[0]
            with st.sidebar:
                fixed_family_cached = st.selectbox(
                    "fixed method/backend",
                    selected_families,
                    index=selected_families.index(default_family),
                    key="cached_fixed_family",
                )
            if fixed_family_cached.startswith("KDE "):
                selected_backends_cached = [fixed_family_cached.replace("KDE ", "", 1)]
                sampler_methods_cached = []
            else:
                selected_backends_cached = []
                sampler_methods_cached = [fixed_family_cached]

    selected_ns_cached = n_options_cached if cached_overlay_mode == "compare n values for fixed method" else [int(n_cached)]
    compare_seed_overlay_cached = cached_overlay_mode == "compare seeds for fixed density type"

    def cached_filter(df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        if df.empty or "model" not in df.columns:
            return pd.DataFrame()
        out = df[df["model"].astype(str).eq(model_choice_cached)].copy()
        if "n" in out.columns:
            out = out[out["n"].astype(int).isin([int(n) for n in selected_ns_cached])]
        if model_choice_cached == "student_t" and "k" in out.columns:
            out = out[np.isclose(out["k"].astype(float), float(k_cached))]
        if seed is not None and "seed" in out.columns:
            seed_values = pd.to_numeric(out["seed"], errors="coerce")
            if seed_values.notna().any():
                out = out[np.isclose(seed_values, int(seed))]
        return out

    reference_view = cached_filter(reference_cache, seed=None if compare_seed_overlay_cached else int(reference_seed_cached))
    visible_xlim_cached = None if show_full_grid_cached else posterior_mass_xlim(reference_view)
    kde_view = cached_filter(posterior_density_cache, seed=None if compare_seed_overlay_cached else int(reference_seed_cached))
    if not selected_backends_cached:
        kde_view = pd.DataFrame()
    elif not kde_view.empty:
        kde_view = kde_view[kde_view["backend"].astype(str).isin(selected_backends_cached)]
    missing_kde_backends = []
    if selected_backends_cached:
        cached_kde_backends = set(kde_view.get("backend", pd.Series(dtype=str)).dropna().astype(str).unique()) if not kde_view.empty else set()
        missing_kde_backends = [backend for backend in selected_backends_cached if backend not in cached_kde_backends]
        if missing_kde_backends:
            density_kind = "median-interval reference density grid" if laplace_even_cached else "KDE density grid"
            st.warning(
                f"Missing {density_kind} for this exact selection: "
                f"model={model_label(model_choice_cached)}, "
                f"k={k_cached if model_choice_cached == 'student_t' else 'n/a'}, "
                f"n={','.join(map(str, selected_ns_cached))}, "
                f"reference seed={'all' if compare_seed_overlay_cached else int(reference_seed_cached)}, "
                f"backend(s)={', '.join(missing_kde_backends)}."
            )
        elif not kde_view.empty:
            kde_b = pd.to_numeric(kde_view.get("B", pd.Series(dtype=float)), errors="coerce").dropna()
            kde_b_used = pd.to_numeric(kde_view.get("B_used", kde_view.get("B", pd.Series(dtype=float))), errors="coerce").dropna()
            b_text = int(kde_b.max()) if not kde_b.empty else "unknown"
            b_used_text = int(kde_b_used.max()) if not kde_b_used.empty else b_text
            density_kind = "median-interval density cache" if laplace_even_cached else "KDE density cache"
            st.caption(
                f"Selected {density_kind} ready: B={b_text}, B_used={b_used_text}, "
                f"reference seed={'all' if compare_seed_overlay_cached else int(reference_seed_cached)}."
            )
    tabram_view = cached_filter(posterior_density_cache, seed=None if compare_seed_overlay_cached else int(reference_seed_cached))
    tabram_view = tabram_view[tabram_view.get("backend", pd.Series(dtype=str)).astype(str).eq("t_abram")] if not tabram_view.empty else tabram_view
    tabram_capped = bool(
        not tabram_view.empty
        and "density_sample_capped" in tabram_view.columns
        and tabram_view["density_sample_capped"].fillna(False).astype(bool).any()
    )
    if tabram_capped:
        st.warning("t_abram is adaptive and expensive; cached t_abram curve is capped for visualization only.")
    sampler_view = cached_filter(sampler_density_cache, seed=None if compare_seed_overlay_cached else int(sampler_seed_cached))
    if sampler_methods_cached and not sampler_view.empty:
        sampler_view = sampler_view[sampler_view["method"].astype(str).isin(sampler_methods_cached)]
    else:
        sampler_view = pd.DataFrame()

    plot_parts = [part for part in [kde_view, sampler_view] if not part.empty]
    plot_ready = pd.concat(plot_parts, ignore_index=True) if plot_parts else pd.DataFrame()
    if not plot_ready.empty:
        if show_full_grid_cached:
            st.warning("Showing the full computational grid; narrow posteriors can look compressed in this view.")
        elif visible_xlim_cached is not None:
            st.caption("Visible x-axis uses the raw weighted-MC 95% posterior-mass zoom; cached density curves still come from the full computational/reference grid.")
        notes = sorted(
            {
                str(note)
                for note in plot_ready.get("density_note", pd.Series(dtype=str)).dropna().astype(str)
                if str(note).strip()
            }
        )
        for note in notes:
            st.caption(note)
        if "plot_grid_lo" not in plot_ready.columns:
            plot_ready["plot_grid_lo"] = float(plot_ready["mu"].min())
        if "plot_grid_hi" not in plot_ready.columns:
            plot_ready["plot_grid_hi"] = float(plot_ready["mu"].max())
        if "plot_grid_size" not in plot_ready.columns:
            plot_ready["plot_grid_size"] = int(plot_ready["mu"].nunique())
        st.pyplot(
            plot_density_overlay(
                plot_ready,
                reference_view,
                overlay_mode=(
                    "compare seeds for fixed density type"
                    if cached_overlay_mode == "compare seeds for fixed density type"
                    else "compare n values for fixed backend"
                    if cached_overlay_mode == "compare n values for fixed method"
                    else "compare backends for fixed n"
                ),
                raw_marker_mode=cached_raw_marker_mode,
                selected_marker_n=int(n_cached),
                show_log_density=bool(show_log_density_cached),
                visible_xlim=visible_xlim_cached,
            ),
            clear_figure=True,
        )
    else:
        st.warning("No selected density curves are available for this exact cache selection.")
        available_density = cached_filter(posterior_density_cache)
        available_sampler = cached_filter(sampler_density_cache)
        available_rows = []
        if not available_density.empty:
            available_rows.append(
                available_density[["model", "k", "n", "seed", "backend"]]
                .drop_duplicates()
                .assign(cache_type="KDE density grid")
            )
        if not available_sampler.empty:
            available_rows.append(
                available_sampler[["model", "k", "n", "seed", "method"]]
                .drop_duplicates()
                .rename(columns={"method": "backend"})
                .assign(cache_type="sampler density")
            )
        if available_rows:
            st.caption("Available cached density selections for the current model/k/n filter:")
            st.dataframe(display_table(pd.concat(available_rows, ignore_index=True)), use_container_width=True, hide_index=True)
        else:
            st.error("No plot-ready density cache exists for this model/k/n filter.")
        st.dataframe(cached_filter(density_status_cache, seed=int(reference_seed_cached)), use_container_width=True)
        st.code("python scripts/prepare_dashboard_cache.py", language="bash")

    st.subheader("Reference Summaries")
    st.caption("Raw weighted-MC is the summary benchmark. KDE curves are smoothed visualization/backend sensitivity diagnostics.")
    ref_default_columns = ["method", "backend", "mean", "sd", "q025", "q50", "q975", "B", "B_used", "note"]
    st.dataframe(default_columns(reference_view, ref_default_columns), use_container_width=True, hide_index=True)
    with st.expander("Show full reference table", expanded=False):
        st.dataframe(display_table(reference_view), use_container_width=True, hide_index=True)

    st.subheader("Posterior Accuracy Deltas")
    comparison_view = cached_filter(comparison_cache, seed=int(sampler_seed_cached))
    if tabram_capped and "backend" in comparison_view.columns:
        comparison_view = comparison_view[~comparison_view["backend"].astype(str).eq("t_abram")]
    delta_columns = ["method", "backend", "delta_mean", "delta_sd", "rel_sd_error", "delta_q025", "delta_q50", "delta_q975", "ess_per_sec"]
    st.dataframe(default_columns(comparison_view, delta_columns), use_container_width=True, hide_index=True)
    with st.expander("Show full posterior accuracy table", expanded=False):
        st.dataframe(display_table(comparison_view), use_container_width=True, hide_index=True)

    st.subheader("Cost Efficiency")
    st.dataframe(cached_filter(cost_cache), use_container_width=True)

    st.subheader("Density Cache Status")
    density_status_view = cached_filter(density_status_cache, seed=int(reference_seed_cached))
    b_values = pd.to_numeric(density_status_view.get("B", pd.Series(dtype=float)), errors="coerce").dropna()
    b_used_values = pd.to_numeric(density_status_view.get("B_used", pd.Series(dtype=float)), errors="coerce").dropna()
    b_text = int(b_values.max()) if not b_values.empty else "unknown"
    b_used_text = int(b_used_values.max()) if not b_used_values.empty else b_text
    capped_text = bool(density_status_view.get("t_abram_capped", pd.Series(dtype=bool)).fillna(False).astype(bool).any()) if not density_status_view.empty else False
    st.info(f"Current density cache: B={b_text}; B_used={b_used_text}; t_abram capped={capped_text}.")
    if b_values.empty or float(b_values.max()) < 100000:
        st.warning("Preview density cache only.")
    st.dataframe(display_table(density_status_view), use_container_width=True, hide_index=True)
    st.stop()

st.subheader("Sampler Outputs")
st.caption("Open Cost Audit page for exact Gibbs/RATTLE cost counters.")
st.caption("Use the sidebar page selector: Cost Audit")

with st.sidebar:
    st.header("Audit CSV")
    reference_level = st.selectbox("reference data level", ["auto-best", "smoke", "full", "custom"], index=0)
    selected_reference_path = auto_best_reference_path() if reference_level == "auto-best" else REFERENCE_LEVEL_PATHS.get(reference_level, DEFAULT_ALL_MODEL_REFERENCE_CSV)
    all_model_reference_csv_path = st.text_input("All-model reference CSV path", value=str(selected_reference_path))
    audit_csv_path = st.text_input("Audit CSV path", value=str(DEFAULT_STEP1_CSV))

    st.header("Model")
    model_choice = st.selectbox("model", ["student_t", "logistic", "laplace"], format_func=lambda x: {"student_t": "Student", "logistic": "Logistic", "laplace": "Laplace"}[x])
    model_n_values = LAPLACE_N_VALUES if model_choice == "laplace" else STUDENT_N_VALUES

    st.header("Compare")
    overlay_mode = st.radio(
        "overlay mode",
        ["compare backends for fixed n", "compare n values for fixed backend"],
    )

    st.header("Case")
    if overlay_mode == "compare backends for fixed n":
        selected_ns = (st.selectbox("n", model_n_values, index=1),)
        st.header("KDE Backends")
        backends = selected_backends()
    else:
        selected_ns = model_n_values
        fixed_backend = st.selectbox("fixed backend", DEFAULT_BACKENDS, index=0)
        backends = (fixed_backend,)

    if model_choice == "student_t":
        k = st.selectbox("k", [1.0, 2.0, 3.0], index=1)
    else:
        k = np.nan
    mu_star = st.number_input("mu_star", value=0.0, step=0.1)
    prior_mean = st.number_input("prior_mean", value=0.0, step=0.5)
    prior_std = st.number_input("prior_std", min_value=0.1, value=10.0, step=0.5)
    B = st.number_input("B", min_value=50, value=100000, step=50)
    seed = st.number_input("seed", min_value=0, value=123, step=1)
    allow_debug_recompute = st.checkbox("Allow small debug recomputation in UI", value=int(B) <= 5000)

    st.header("Estimators")
    show_raw = st.checkbox("raw weighted-MC summaries", value=True)
    show_density = st.checkbox("KDE density curves", value=True)
    show_kde_grid = st.checkbox("KDE-grid summaries", value=True)
    show_kde_quad = st.checkbox("KDE-quad summaries", value=False)

    st.header("Samplers")
    sampler_level = st.selectbox("sampler results level", ["auto-best", "smoke", "medium", "full", "multiseed", "custom"], index=0)
    selected_sampler_dir = auto_best_sampler_dir() if sampler_level == "auto-best" else SAMPLER_LEVEL_PATHS.get(sampler_level, DEFAULT_SAMPLER_RESULTS_DIR)
    sampler_results_dir = st.text_input("Sampler results directory", value=str(selected_sampler_dir))
    show_gibbs_overlay = st.checkbox("Gibbs", value=True)
    show_rattle_overlay = st.checkbox("RATTLE", value=model_choice != "laplace", disabled=model_choice == "laplace")
    if model_choice == "laplace":
        st.caption("Exact RATTLE not applicable: Laplace median constraint is nonsmooth/order-based.")

    st.header("Plot")
    show_full_grid = st.checkbox("Show full computational grid", value=False)
    show_log_density = st.checkbox("Show log density", value=False)
    raw_marker_mode = st.selectbox("Raw reference markers", ["none", "selected n only", "all n"], index=1)
    selected_marker_n = st.selectbox("raw marker n", model_n_values, index=model_n_values.index(selected_ns[0]))

    st.header("Grid")
    grid_size = st.slider("computational_grid_size", min_value=2000, max_value=12000, value=2000, step=250)
    plot_grid_size = st.slider("plot_grid_size", min_value=500, max_value=8000, value=2000, step=250)
    bound_multiplier = st.slider("bound_multiplier", min_value=3.0, max_value=12.0, value=5.0, step=0.5)


if overlay_mode == "compare backends for fixed n":
    selected_marker_n = selected_ns[0]

status_cols_top = st.columns(4)
status_cols_top[0].metric("reference level", reference_level)
status_cols_top[1].metric("sampler level", sampler_level)
status_cols_top[2].metric("RATTLE settings", rattle_settings_status())
status_cols_top[3].metric("analysis report", "found" if ANALYSIS_REPORT_PATH.exists() else "missing")

if model_choice != "student_t":
    st.warning("Using generalized all-model reference CSV for non-Student models. The UI will not run reference audits automatically.")
    if model_choice == "laplace" and any(int(n) % 2 == 0 for n in selected_ns):
        st.error("Laplace Gibbs and np.median KDE/raw-MC references are not directly comparable for even n.")
        st.warning("Laplace exact RATTLE is not applicable because the median/order constraint is nonsmooth.")
        st.info("Laplace Gibbs is compared only to median_interval_contains_mu_star.")
    all_ref_path = Path(all_model_reference_csv_path)
    if not all_ref_path.exists():
        st.info("Generalized reference CSV is missing.")
        st.code(ALL_MODEL_REFERENCE_COMMAND, language="bash")
    ref_df = pd.read_csv(all_ref_path) if all_ref_path.exists() else pd.DataFrame()
    if not ref_df.empty:
        ref_df = ref_df[ref_df["model"].astype(str).eq(model_choice)]
        ref_df = ref_df[ref_df["n"].astype(int).isin([int(n) for n in selected_ns])]
        if model_choice == "student_t" and "k" in ref_df.columns:
            ref_df = ref_df[np.isclose(ref_df["k"].astype(float), float(k))]
        if "mu_star" in ref_df.columns:
            ref_df = ref_df[np.isclose(ref_df["mu_star"].astype(float), float(mu_star))]
        if "seed" in ref_df.columns:
            ref_df = ref_df[ref_df["seed"].astype(int).eq(int(seed))]
        ref_df = ref_df.rename(columns={"marginal_likelihood_estimate": "marginal_likelihood_estimate"})
        ref_df["source_file"] = str(all_ref_path)
        ref_df["B"] = ref_df.get("B", np.nan)
        ref_df["weighted_ess"] = ref_df.get("weighted_ess", np.nan)
    sampler_methods = {"gibbs"} | ({"rattle"} if show_rattle_overlay else set())
    sampler_dir = Path(sampler_results_dir)
    sampler_chain, sampler_audit_summaries, sampler_diagnostics, sampler_ledger, sampler_warnings = load_sampler_outputs(sampler_dir)
    for warning in sampler_warnings:
        st.warning(warning)
    filtered_sampler_chain = filter_sampler_chain(
        sampler_chain,
        selected_ns=tuple(int(n) for n in selected_ns),
        k=float(k) if np.isfinite(k) else np.nan,
        mu_star=float(mu_star),
        seed=int(seed),
        selected_methods=sampler_methods,
    )
    if "model" in filtered_sampler_chain.columns:
        filtered_sampler_chain = filtered_sampler_chain[filtered_sampler_chain["model"].astype(str).eq(model_choice)]
    sampler_summary_rows = sampler_summary_from_audit(
        sampler_audit_summaries,
        sampler_diagnostics,
        selected_methods=sampler_methods,
        source_file=sampler_dir / "posterior_summaries.csv",
    )
    if not sampler_summary_rows.empty and "model" in sampler_summary_rows.columns:
        sampler_summary_rows = sampler_summary_rows[sampler_summary_rows["model"].astype(str).eq(model_choice)]
    if sampler_summary_rows.empty:
        sampler_summary_rows = sampler_summary_from_chains(filtered_sampler_chain, source_file=sampler_dir / "chain_samples.csv")

    plot_lo = float(ref_df["q025"].min() - 0.25 * (ref_df["q975"].max() - ref_df["q025"].min())) if not ref_df.empty else float(mu_star) - 1.0
    plot_hi = float(ref_df["q975"].max() + 0.25 * (ref_df["q975"].max() - ref_df["q025"].min())) if not ref_df.empty else float(mu_star) + 1.0
    sampler_density_df = sampler_density_frames(
        filtered_sampler_chain,
        plot_lo=plot_lo,
        plot_hi=plot_hi,
        plot_grid_size=int(plot_grid_size),
        source_file=sampler_dir / "chain_samples.csv",
    )
    sampler_density_df = with_plot_metadata(sampler_density_df, plot_lo, plot_hi, int(plot_grid_size)) if not sampler_density_df.empty else sampler_density_df
    if not sampler_density_df.empty:
        st.pyplot(
            plot_density_overlay(
                sampler_density_df,
                ref_df.rename(columns={"weighted_ess": "weighted_ess"}),
                overlay_mode=overlay_mode,
                raw_marker_mode=raw_marker_mode,
                selected_marker_n=int(selected_marker_n),
                show_log_density=bool(show_log_density),
            ),
            clear_figure=True,
        )
    elif sampler_methods:
        st.warning("No Gibbs/RATTLE samples found.")
        st.code(SAMPLER_AUDIT_COMMAND, language="bash")

    comparison_ref_df = ref_df
    if model_choice == "laplace" and any(int(n) % 2 == 0 for n in selected_ns):
        interval_ref = ref_df[ref_df["estimator_type"].astype(str).eq("raw_mc_interval_reference")]
        if not interval_ref.empty:
            comparison_ref_df = interval_ref
            st.info("Laplace sampler deltas use the median-interval reference target.")
        else:
            st.warning("Laplace median-interval reference is unavailable; sampler deltas are suppressed.")
            sampler_summary_rows = pd.DataFrame()
    elif model_choice == "laplace":
        st.info("Laplace odd-n sampler deltas use the deterministic unique sample median reference.")
    method_comparison_df, method_comparison_baseline = comparison_summary_table(comparison_ref_df, sampler_summary_rows)
    st.subheader("Sampler vs Reference Summary")
    st.dataframe(method_comparison_df, use_container_width=True)
    st.stop()

if not backends:
    st.warning("Select at least one KDE backend.")
    st.stop()

if int(B) < 20000:
    st.warning("Debug run only: B is small, do not interpret backend differences scientifically.")

audit_path = Path(audit_csv_path)
audit_csv_exists = audit_path.exists()
audit_summary = pd.DataFrame()
if audit_csv_exists:
    audit_summary = cached_audit_summary(
        str(audit_path),
        float(k),
        tuple(int(n) for n in selected_ns),
        int(B),
        int(seed),
        backends,
        float(mu_star),
    )

summary_frames = []
density_frames = []
grid_bounds = []
skipped_density_cases = []
status_rows = []
loaded_from_audit_csv = bool(not audit_summary.empty)

for n in selected_ns:
    cache_path = mle_sample_cache_path(k=float(k), n=int(n), B=int(B), seed=int(seed), audit_dir=DEFAULT_AUDIT_DIR)
    used_cached_mle_samples = cache_path.exists()
    large_missing_cache = int(B) > 20000 and not used_cached_mle_samples
    may_recompute_in_ui = (not used_cached_mle_samples) and bool(allow_debug_recompute) and not large_missing_cache
    can_load_samples = used_cached_mle_samples or may_recompute_in_ui

    if used_cached_mle_samples:
        mle_sample_status = "loaded from cache"
        density_status = "computed from cached samples"
    elif large_missing_cache:
        mle_sample_status = "cache missing; refused large-B UI recomputation"
        density_status = "unavailable: Large-B reference must be generated from CLI."
    elif may_recompute_in_ui:
        mle_sample_status = "newly simulated in UI"
        density_status = "computed from newly simulated debug samples"
    else:
        mle_sample_status = "cache missing; UI recomputation disabled"
        density_status = "unavailable: UI recomputation disabled"

    status_rows.append(
        {
            "n": int(n),
            "audit_csv_path": str(audit_path),
            "audit_csv_exists": bool(audit_csv_exists),
            "loaded_from_audit_csv": loaded_from_audit_csv,
            "summary_status": "loaded from audit CSV" if loaded_from_audit_csv else "recomputed in UI" if can_load_samples else "unavailable",
            "mle_cache_path": str(cache_path),
            "used_cached_mle_samples": bool(used_cached_mle_samples),
            "recomputed_in_ui": bool(may_recompute_in_ui),
            "mle_sample_status": mle_sample_status,
            "density_status": density_status,
        }
    )

    if not can_load_samples:
        skipped_density_cases.append(f"n={int(n)} cache={cache_path} status={density_status}")
        continue

    z_samples = cached_z_samples(float(k), int(n), int(B), int(seed))
    if audit_summary.empty:
        summary_frames.append(
            cached_summaries(
                z_samples,
                float(k),
                int(n),
                float(mu_star),
                float(prior_mean),
                float(prior_std),
                int(B),
                int(seed),
                backends,
                show_kde_grid,
                show_kde_quad,
                int(grid_size),
                float(bound_multiplier),
            )
        )
    density_frames.append(
        cached_density_grid(
            z_samples,
            float(k),
            int(n),
            float(mu_star),
            float(prior_mean),
            float(prior_std),
            int(B),
            int(seed),
            backends,
            int(grid_size),
            float(bound_multiplier),
        )
    )
    grid_bounds.append(posterior_grid_bounds(float(mu_star), float(prior_mean), float(prior_std), z_samples, float(bound_multiplier)))

density_df = pd.concat(density_frames, ignore_index=True) if density_frames else pd.DataFrame()
status_df = pd.DataFrame(status_rows)
if audit_summary.empty:
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
else:
    summary_df = audit_summary.copy()
summary_df = attach_density_checks(summary_df, density_df)
if summary_df.empty:
    comparison_rows = pd.DataFrame()
    display_rows = pd.DataFrame()
else:
    comparison_rows = selected_summary_rows(summary_df, show_raw=True, show_kde_grid=show_kde_grid, show_kde_quad=show_kde_quad)
    display_rows = selected_summary_rows(summary_df, show_raw=show_raw, show_kde_grid=show_kde_grid, show_kde_quad=show_kde_quad)

if density_df.empty:
    raw_for_bounds = summary_df[summary_df["estimator_type"].eq("raw_weighted_mc")]
    if raw_for_bounds.empty:
        plot_lo, plot_hi = float(mu_star) - 1.0, float(mu_star) + 1.0
    else:
        plot_lo = float((raw_for_bounds["q025"] - 0.25 * (raw_for_bounds["q975"] - raw_for_bounds["q025"])).min())
        plot_hi = float((raw_for_bounds["q975"] + 0.25 * (raw_for_bounds["q975"] - raw_for_bounds["q025"])).max())
    lo, hi = plot_lo, plot_hi
    actual_plot_grid_size = int(plot_grid_size)
    plot_density_df = pd.DataFrame()
else:
    lo = min(bound[0] for bound in grid_bounds)
    hi = max(bound[1] for bound in grid_bounds)
    plot_lo, plot_hi = plot_xlim_from_raw(summary_df, density_df, bool(show_full_grid))
    actual_plot_grid_size = len(density_df["mu"].unique()) if show_full_grid else int(plot_grid_size)
    plot_density_df = make_plot_density_grid(
        density_df,
        plot_lo=plot_lo,
        plot_hi=plot_hi,
        plot_grid_size=actual_plot_grid_size,
        show_full_grid=bool(show_full_grid),
    )

sampler_methods = set()
if show_gibbs_overlay:
    sampler_methods.add("gibbs")
if show_rattle_overlay:
    sampler_methods.add("rattle")
sampler_dir = Path(sampler_results_dir)
sampler_chain, sampler_audit_summaries, sampler_diagnostics, sampler_ledger, sampler_warnings = load_sampler_outputs(sampler_dir)
for warning in sampler_warnings:
    st.warning(warning)

sampler_density_df = pd.DataFrame()
sampler_summary_rows = pd.DataFrame()
if sampler_methods:
    missing_sampler_files = [
        name
        for name in ["chain_samples.csv", "posterior_summaries.csv", "diagnostic_summary.csv", "cost_ledger.csv"]
        if not (sampler_dir / name).exists()
    ]
    if missing_sampler_files:
        st.warning("No Gibbs/RATTLE samples found." if "chain_samples.csv" in missing_sampler_files else "Some sampler audit files are missing.")
        st.code(SAMPLER_AUDIT_COMMAND, language="bash")
    smoke_status = sampler_smoke_status(sampler_ledger, sampler_chain)
    if smoke_status == "smoke":
        st.warning("Smoke run only. Do not interpret posterior accuracy or cost scientifically.")

    filtered_sampler_chain = filter_sampler_chain(
        sampler_chain,
        selected_ns=tuple(int(n) for n in selected_ns),
        k=float(k),
        mu_star=float(mu_star),
        seed=int(seed),
        selected_methods=sampler_methods,
    )
    sampler_density_df = sampler_density_frames(
        filtered_sampler_chain,
        plot_lo=plot_lo,
        plot_hi=plot_hi,
        plot_grid_size=actual_plot_grid_size,
        source_file=sampler_dir / "chain_samples.csv",
    )
    sampler_summary_rows = sampler_summary_from_audit(
        sampler_audit_summaries,
        sampler_diagnostics,
        selected_methods=sampler_methods,
        source_file=sampler_dir / "posterior_summaries.csv",
    )
    if not sampler_summary_rows.empty:
        sampler_summary_rows = sampler_summary_rows[
            sampler_summary_rows["n"].astype(int).isin([int(n) for n in selected_ns])
        ]
        if "k" in sampler_summary_rows.columns:
            sampler_summary_rows = sampler_summary_rows[np.isclose(sampler_summary_rows["k"].astype(float), float(k))]
        if "mu_star" in sampler_summary_rows.columns:
            sampler_summary_rows = sampler_summary_rows[np.isclose(sampler_summary_rows["mu_star"].astype(float), float(mu_star))]
        if "seed" in sampler_summary_rows.columns:
            sampler_summary_rows = sampler_summary_rows[sampler_summary_rows["seed"].astype(int).eq(int(seed))]
    if sampler_summary_rows.empty:
        sampler_summary_rows = sampler_summary_from_chains(filtered_sampler_chain, source_file=sampler_dir / "chain_samples.csv")

display_rows = with_plot_metadata(display_rows, plot_lo, plot_hi, actual_plot_grid_size)
comparison_rows = with_plot_metadata(comparison_rows, plot_lo, plot_hi, actual_plot_grid_size)
difference_df = with_plot_metadata(difference_from_raw(comparison_rows), plot_lo, plot_hi, actual_plot_grid_size)
if not sampler_density_df.empty:
    sampler_density_df = with_plot_metadata(sampler_density_df, plot_lo, plot_hi, actual_plot_grid_size)
plot_density_for_plot = (
    pd.concat([frame for frame in [plot_density_df, sampler_density_df] if not frame.empty], ignore_index=True)
    if any(not frame.empty for frame in [plot_density_df, sampler_density_df])
    else pd.DataFrame()
)
density_export_parts = [frame for frame in [density_df, sampler_density_df] if not frame.empty]
density_export_source = pd.concat(density_export_parts, ignore_index=True) if density_export_parts else pd.DataFrame()
density_export_df = with_plot_metadata(density_export_source, plot_lo, plot_hi, actual_plot_grid_size)
display_rows = with_status_metadata(display_rows, status_df)
difference_df = with_status_metadata(difference_df, status_df)
density_export_df = with_status_metadata(density_export_df, status_df)
summary_export_df = ensure_columns(display_rows, SUMMARY_COLUMNS)
difference_export_df = ensure_columns(difference_df, DIFFERENCE_COLUMNS)
method_comparison_df, method_comparison_baseline = comparison_summary_table(comparison_rows, sampler_summary_rows)
density_export_df = ensure_columns(
    density_export_df,
    [
        "marginal_likelihood_estimate",
        "posterior_integral_check",
        "source_file",
        "plot_grid_lo",
        "plot_grid_hi",
        "plot_grid_size",
        *EXPORT_STATUS_COLUMNS,
    ],
)

top = st.columns(4)
top[0].metric("n", ",".join(str(n) for n in selected_ns))
top[1].metric("k", f"{float(k):g}")
top[2].metric("B", int(B))
top[3].metric("visible x", f"[{plot_lo:.2f}, {plot_hi:.2f}]")
st.caption(f"Computational grid covers [{lo:.2f}, {hi:.2f}] before any plot zoom.")

st.subheader("Cache And Computation Status")
st.caption(f"audit_csv_path: {audit_path}")
status_metrics = st.columns(4)
status_metrics[0].metric("audit CSV exists", "yes" if audit_csv_exists else "no")
status_metrics[1].metric("summaries", "audit CSV" if loaded_from_audit_csv else "recomputed/UI" if not summary_df.empty else "unavailable")
status_metrics[2].metric("MLE samples", summarize_status(status_df["mle_sample_status"]))
status_metrics[3].metric("density curves", summarize_status(status_df["density_status"]))
st.dataframe(status_df, use_container_width=True)

if any("Large-B reference must be generated from CLI." in value for value in status_df["density_status"].astype(str)):
    st.warning("Large-B reference must be generated from CLI.")

if skipped_density_cases:
    st.info("Audit summaries loaded without recomputing large-B density curves because MLE-error cache files are missing.")
    for case in skipped_density_cases:
        st.caption(case)

if density_df.empty:
    diag = pd.DataFrame()
else:
    diag = density_diagnostics(density_df, plot_density_df, int(grid_size), float(bound_multiplier))
    raw_sd = summary_df[summary_df["estimator_type"].eq("raw_weighted_mc")][["n", "sd"]]
    dx_warnings = diag.merge(raw_sd, on="n", how="left")
    dx_warnings = dx_warnings[dx_warnings["plot_dx"] > dx_warnings["sd"] / 20.0]
    if not dx_warnings.empty:
        cases = ", ".join(f"n={int(row.n)} {row.backend}" for row in dx_warnings.itertuples())
        st.warning(f"Plot grid spacing is coarse relative to raw weighted-MC sd for: {cases}. Increase plot_grid_size.")

any_sampler_overlay = bool(sampler_methods)
if (show_density or any_sampler_overlay) and not plot_density_for_plot.empty:
    st.pyplot(
        plot_density_overlay(
            plot_density_for_plot,
            summary_df,
            overlay_mode=overlay_mode,
            raw_marker_mode=raw_marker_mode,
            selected_marker_n=int(selected_marker_n),
            show_log_density=bool(show_log_density),
        ),
        clear_figure=True,
    )
elif show_density or any_sampler_overlay:
    st.info("Requested posterior density curves are unavailable until matching inputs or artifacts are present.")

st.subheader("Summary")
summary_default_columns = ["method", "backend", "mean", "sd", "q025", "q50", "q975", "B", "B_used", "note"]
st.dataframe(default_columns(summary_export_df, summary_default_columns), use_container_width=True, hide_index=True)
with st.expander("Show full reference table", expanded=False):
    st.dataframe(display_table(summary_export_df[SUMMARY_COLUMNS]), use_container_width=True, hide_index=True)

st.subheader("Sampler vs Reference Summary")
if method_comparison_baseline == "kde_scott_fallback":
    st.warning("Raw weighted-MC baseline is unavailable; deltas use KDE scott as a fallback.")
elif method_comparison_baseline == "none":
    st.info("No baseline is available for method deltas.")
st.dataframe(method_comparison_df, use_container_width=True)

st.subheader("Difference From Raw Weighted-MC")
difference_default_columns = ["method", "backend", "delta_mean", "delta_sd", "rel_sd_error", "delta_q025", "delta_q50", "delta_q975", "ess_per_sec"]
st.dataframe(default_columns(difference_export_df, difference_default_columns), use_container_width=True, hide_index=True)
with st.expander("Show full posterior accuracy table", expanded=False):
    st.dataframe(display_table(difference_export_df[DIFFERENCE_COLUMNS]), use_container_width=True, hide_index=True)

st.subheader("Diagnostics")
grid_diag_cols = ["n", "backend", "estimator_type", "grid_lo", "grid_hi", "bandwidth"]
summary_diag = summary_df[summary_df["estimator_type"].isin(["kde_grid", "kde_quad"])]
summary_diag = summary_diag[[col for col in grid_diag_cols if col in summary_diag.columns]]
if not summary_diag.empty and not diag.empty:
    diag = diag.merge(summary_diag, on=["n", "backend"], how="left")
b_values_diag = pd.to_numeric(density_export_df.get("B", pd.Series(dtype=float)), errors="coerce").dropna()
b_used_diag = pd.to_numeric(density_export_df.get("B_used", pd.Series(dtype=float)), errors="coerce").dropna()
tabram_capped_diag = bool(
    "density_sample_capped" in density_export_df.columns
    and density_export_df["density_sample_capped"].fillna(False).astype(bool).any()
)
b_text_diag = int(b_values_diag.max()) if not b_values_diag.empty else "unknown"
b_used_text_diag = int(b_used_diag.max()) if not b_used_diag.empty else b_text_diag
st.info(f"Current density cache: B={b_text_diag}; B_used={b_used_text_diag}; t_abram capped={tabram_capped_diag}.")
if b_values_diag.empty or float(b_values_diag.max()) < 100000:
    st.warning("Preview density cache only.")
st.dataframe(diag, use_container_width=True, hide_index=True)

st.subheader("Tail Diagnostics")
tail_df = with_plot_metadata(tail_diagnostics(density_df, summary_df), plot_lo, plot_hi, actual_plot_grid_size)
st.dataframe(tail_df, use_container_width=True)

if not audit_path.exists() or audit_summary.empty:
    cmd = audit_command(
        k_values=[float(k)],
        n_values=model_n_values,
        B_values=[100000],
        seeds=[123],
        backends=DEFAULT_BACKENDS,
        out_csv=audit_path,
    )
    message = "Audit CSV path does not exist." if not audit_path.exists() else "No matching audit CSV row for the current controls."
    st.caption(message)
    st.code(" ".join(cmd), language="bash")

exports = st.columns(3)
exports[0].download_button(
    "Download summary CSV",
    data=summary_export_df.to_csv(index=False).encode("utf-8"),
    file_name="kde_reference_summary.csv",
    mime="text/csv",
)
exports[1].download_button(
    "Download difference CSV",
    data=difference_export_df.to_csv(index=False).encode("utf-8"),
    file_name="kde_reference_differences.csv",
    mime="text/csv",
)
exports[2].download_button(
    "Download density grid CSV",
    data=density_export_df.to_csv(index=False).encode("utf-8"),
    file_name="kde_reference_density_grid.csv",
    mime="text/csv",
)
