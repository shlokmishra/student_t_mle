"""Numerical KDE correctness audit for Student-t, logistic, and odd-n Laplace.

This is a cache-only postprocessor. It does not run samplers, modify KDE
backends, or regenerate simulated MLE samples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKENDS = ["scott", "SJ_transform", "t_abram"]
PRIMARY_BACKENDS = ["scott", "SJ_transform"]
CASE_COLUMNS = ["model", "k_key", "n", "mu_star", "ref_seed"]
CASE_DISPLAY_COLUMNS = ["model", "k", "n", "mu_star", "ref_seed"]
QUANTILES = {
    "q01": 0.01,
    "q025": 0.025,
    "q05": 0.05,
    "q50": 0.5,
    "q95": 0.95,
    "q975": 0.975,
    "q99": 0.99,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"),
    )
    parser.add_argument(
        "--density-cache",
        type=Path,
        default=Path("results/dashboard_cache/posterior_density_cache.csv"),
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path("results/dashboard_cache/cache_manifest.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/kde_correctness_audit/"))
    return parser.parse_args()


def k_key(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "__NA__").astype(str)


def finite_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def target_cases(df: pd.DataFrame) -> pd.DataFrame:
    student = df["model"].eq("student_t") & df["k"].isin([1.0, 2.0, 3.0]) & df["n"].isin([10, 20, 50])
    logistic = df["model"].eq("logistic") & df["n"].isin([10, 20, 50])
    laplace = df["model"].eq("laplace") & df["n"].isin([11, 21, 51])
    return df[student | logistic | laplace].copy()


def density_group_quantiles(group: pd.DataFrame) -> dict[str, float]:
    ordered = group.sort_values("mu")
    mu = ordered["mu"].to_numpy(float)
    density = np.maximum(ordered["density"].to_numpy(float), 0.0)
    if mu.size < 2 or not np.isfinite(density).any():
        return {name: np.nan for name in QUANTILES}
    integral = float(np.trapezoid(density, mu))
    if integral <= 0.0 or not math.isfinite(integral):
        return {name: np.nan for name in QUANTILES}
    cdf = np.concatenate([[0.0], np.cumsum((density[:-1] + density[1:]) * np.diff(mu) / 2.0)])
    cdf = np.maximum.accumulate(np.clip(cdf / max(cdf[-1], 1e-300), 0.0, 1.0))
    return {name: float(np.interp(prob, cdf, mu)) for name, prob in QUANTILES.items()}


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not args.reference_csv.exists():
        raise FileNotFoundError(args.reference_csv)
    ref = pd.read_csv(args.reference_csv)
    ref = target_cases(ref)
    ref["ref_seed"] = ref["seed"].astype(int)
    ref["k_key"] = k_key(ref["k"])

    if args.density_cache.exists():
        density = pd.read_csv(args.density_cache)
        density = target_cases(density)
        density["ref_seed"] = density["seed"].astype(int)
        density["k_key"] = k_key(density["k"])
    else:
        density = pd.DataFrame()

    manifest = {}
    if args.cache_manifest.exists():
        manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    return ref, density, manifest


def density_summary(density: pd.DataFrame) -> pd.DataFrame:
    if density.empty:
        return pd.DataFrame()
    rows = []
    for key, group in density.groupby(CASE_COLUMNS + ["backend"], dropna=False):
        first = group.iloc[0]
        ordered = group.sort_values("mu")
        mu = ordered["mu"].to_numpy(float)
        dens = np.maximum(ordered["density"].to_numpy(float), 0.0)
        integral = float(np.trapezoid(dens, mu)) if mu.size > 1 else np.nan
        q = density_group_quantiles(ordered)
        rows.append(
            {
                **dict(zip(CASE_COLUMNS + ["backend"], key, strict=True)),
                "B_used": int(finite_float(first.get("B_used", first.get("B")), 0)),
                "density_sample_size": int(finite_float(first.get("density_sample_size", first.get("B_used", first.get("B"))), 0)),
                "sample_cap_applied": normalize_bool(first.get("density_sample_capped", False)),
                "posterior_integral_check": finite_float(first.get("posterior_integral_check", integral)),
                "marginal_likelihood_density": finite_float(first.get("marginal_likelihood_estimate")),
                "density_note": "" if pd.isna(first.get("density_note", "")) else str(first.get("density_note", "")),
                **q,
            }
        )
    return pd.DataFrame(rows)


def make_summary(ref: pd.DataFrame, dens_sum: pd.DataFrame) -> pd.DataFrame:
    raw = ref[ref["estimator_type"].eq("raw_weighted_mc")].copy()
    raw_cols = CASE_COLUMNS + ["mean", "sd", "q025", "q50", "q975"]
    raw = raw[raw_cols].rename(columns={col: f"raw_{col}" for col in ["mean", "sd", "q025", "q50", "q975"]})
    for name in ["q01", "q05", "q95", "q99"]:
        raw[f"raw_{name}"] = np.nan

    kde = ref[ref["backend"].isin(BACKENDS)].copy()
    kde = kde.merge(raw, on=CASE_COLUMNS, how="left")
    if not dens_sum.empty:
        keep = CASE_COLUMNS + [
            "backend",
            "B_used",
            "sample_cap_applied",
            "posterior_integral_check",
            "density_note",
            "q01",
            "q05",
            "q95",
            "q99",
        ]
        kde = kde.merge(dens_sum[keep], on=CASE_COLUMNS + ["backend"], how="left", suffixes=("", "_density"))
    else:
        kde["B_used"] = kde["B"]
        kde["sample_cap_applied"] = False
        kde["posterior_integral_check"] = np.nan
        kde["density_note"] = ""
        for name in ["q01", "q05", "q95", "q99"]:
            kde[name] = np.nan

    kde["B_full"] = kde["B"].astype(int)
    kde["B_used"] = kde["B_used"].fillna(kde["B_full"]).astype(int)
    kde["sample_cap_applied"] = kde["sample_cap_applied"].map(normalize_bool)
    kde["delta_mean"] = kde["mean"] - kde["raw_mean"]
    kde["delta_sd"] = kde["sd"] - kde["raw_sd"]
    kde["rel_sd_error"] = kde["delta_sd"].abs() / kde["raw_sd"].replace(0, np.nan)
    for name in QUANTILES:
        raw_name = f"raw_{name}"
        if raw_name not in kde:
            kde[raw_name] = np.nan
        kde[f"delta_{name}"] = kde[name] - kde[raw_name]

    warnings = []
    for _, row in kde.iterrows():
        parts = []
        if row["backend"] == "t_abram" and (row["B_used"] < row["B_full"] or bool(row["sample_cap_applied"])):
            parts.append("t_abram capped diagnostic only")
        if pd.isna(row.get("raw_q01")) or pd.isna(row.get("raw_q05")):
            parts.append("raw q01/q05/q95/q99 unavailable in cached raw summary")
        if finite_float(row.get("posterior_integral_check")) and abs(float(row["posterior_integral_check"]) - 1.0) > 0.01:
            parts.append("posterior integral check away from 1")
        warnings.append("; ".join(parts))
    kde["warning"] = warnings

    columns = [
        "model",
        "k",
        "n",
        "mu_star",
        "ref_seed",
        "backend",
        "estimator_type",
        "B_full",
        "B_used",
        "sample_cap_applied",
        "mean",
        "sd",
        "q01",
        "q025",
        "q05",
        "q50",
        "q95",
        "q975",
        "q99",
        "raw_mean",
        "raw_sd",
        "raw_q01",
        "raw_q025",
        "raw_q05",
        "raw_q50",
        "raw_q95",
        "raw_q975",
        "raw_q99",
        "delta_mean",
        "delta_sd",
        "rel_sd_error",
        "delta_q01",
        "delta_q025",
        "delta_q05",
        "delta_q50",
        "delta_q95",
        "delta_q975",
        "delta_q99",
        "posterior_integral_check",
        "marginal_likelihood_estimate",
        "target_description",
        "warning",
    ]
    kde = kde.sort_values(["model", "k_key", "n", "ref_seed", "backend"])
    return kde[[col for col in columns if col in kde]]


def pairwise_backend_sensitivity(summary: pd.DataFrame, density: pd.DataFrame) -> pd.DataFrame:
    density_lookup = {}
    if not density.empty:
        for key, group in density.groupby(CASE_COLUMNS + ["backend"], dropna=False):
            ordered = group.sort_values("mu")
            density_lookup[key] = (ordered["mu"].to_numpy(float), ordered["density"].to_numpy(float))

    rows = []
    pairs = [("scott", "SJ_transform"), ("scott", "t_abram"), ("SJ_transform", "t_abram")]
    for case, group in summary.groupby(CASE_DISPLAY_COLUMNS, dropna=False):
        by_backend = {row["backend"]: row for _, row in group.iterrows()}
        raw_width = finite_float(group["raw_q975"].iloc[0] - group["raw_q025"].iloc[0])
        raw_sd = finite_float(group["raw_sd"].iloc[0])
        case_key = (
            case[0],
            "__NA__" if pd.isna(case[1]) else str(case[1]),
            case[2],
            case[3],
            case[4],
        )
        for left, right in pairs:
            if left not in by_backend or right not in by_backend:
                continue
            a = by_backend[left]
            b = by_backend[right]
            grid_diff = np.nan
            iae = np.nan
            d1 = density_lookup.get((*case_key, left))
            d2 = density_lookup.get((*case_key, right))
            if d1 is not None and d2 is not None and d1[0].size > 1 and d2[0].size > 1:
                lo = max(float(np.nanmin(d1[0])), float(np.nanmin(d2[0])))
                hi = min(float(np.nanmax(d1[0])), float(np.nanmax(d2[0])))
                if hi > lo:
                    grid = np.linspace(lo, hi, min(2500, max(d1[0].size, d2[0].size)))
                    y1 = np.interp(grid, d1[0], d1[1])
                    y2 = np.interp(grid, d2[0], d2[1])
                    diff = np.abs(y1 - y2)
                    grid_diff = float(np.nanmax(diff))
                    iae = float(np.trapezoid(diff, grid))
            abs_sd = abs(float(a["sd"]) - float(b["sd"]))
            rel_sd = abs_sd / max((float(a["sd"]) + float(b["sd"])) / 2.0, 1e-300)
            q025_diff = abs(finite_float(a.get("q025")) - finite_float(b.get("q025")))
            q975_diff = abs(finite_float(a.get("q975")) - finite_float(b.get("q975")))
            rel_q = max(q025_diff, q975_diff) / max(raw_width, raw_sd, 1e-300)
            warning = "none"
            if rel_sd > 0.05 or rel_q > 0.05:
                warning = "serious"
            elif rel_sd > 0.02 or rel_q > 0.02:
                warning = "mild"
            rows.append(
                {
                    "model": case[0],
                    "k": case[1],
                    "n": case[2],
                    "mu_star": case[3],
                    "ref_seed": case[4],
                    "backend_pair": f"{left} vs {right}",
                    "backend_left": left,
                    "backend_right": right,
                    "abs_mean_diff": abs(float(a["mean"]) - float(b["mean"])),
                    "abs_sd_diff": abs_sd,
                    "rel_sd_diff": rel_sd,
                    "abs_q025_diff": q025_diff,
                    "abs_q50_diff": abs(finite_float(a.get("q50")) - finite_float(b.get("q50"))),
                    "abs_q975_diff": q975_diff,
                    "abs_q01_diff": abs(finite_float(a.get("q01")) - finite_float(b.get("q01"))),
                    "abs_q99_diff": abs(finite_float(a.get("q99")) - finite_float(b.get("q99"))),
                    "max_density_diff_on_grid": grid_diff,
                    "integrated_abs_density_diff": iae,
                    "raw_width95": raw_width,
                    "quantile_diff_over_raw_width95": rel_q,
                    "warning": warning,
                }
            )
    return pd.DataFrame(rows)


def seed_stability(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in summary.groupby(["model", "k", "n", "backend"], dropna=False):
        raw_width = (group["raw_q975"] - group["raw_q025"]).mean()
        vals_sd = group["sd"].astype(float)
        cv_sd = float(vals_sd.std(ddof=0) / max(vals_sd.mean(), 1e-300))
        q_spread = max(
            float(group[col].max() - group[col].min())
            for col in ["q025", "q50", "q975"]
            if col in group
        )
        warning = "none"
        if cv_sd > 0.05 or q_spread / max(raw_width, 1e-300) > 0.05:
            warning = "serious"
        elif cv_sd > 0.02 or q_spread / max(raw_width, 1e-300) > 0.02:
            warning = "mild"
        rows.append(
            {
                "model": key[0],
                "k": key[1],
                "n": key[2],
                "backend": key[3],
                "num_seeds": int(group["ref_seed"].nunique()),
                "mean_of_mean": float(group["mean"].mean()),
                "sd_of_mean": float(group["mean"].std(ddof=0)),
                "range_mean": float(group["mean"].max() - group["mean"].min()),
                "mean_of_sd": float(vals_sd.mean()),
                "sd_of_sd": float(vals_sd.std(ddof=0)),
                "cv_sd": cv_sd,
                "range_q025": float(group["q025"].max() - group["q025"].min()),
                "range_q50": float(group["q50"].max() - group["q50"].min()),
                "range_q975": float(group["q975"].max() - group["q975"].min()),
                "range_q01": float(group["q01"].max() - group["q01"].min()) if group["q01"].notna().any() else np.nan,
                "range_q99": float(group["q99"].max() - group["q99"].min()) if group["q99"].notna().any() else np.nan,
                "raw_width95_mean": float(raw_width),
                "quantile_spread_over_raw_width95": float(q_spread / max(raw_width, 1e-300)),
                "warning": warning,
            }
        )
    return pd.DataFrame(rows)


def cdf_at(mu: np.ndarray, cdf: np.ndarray, x: float) -> float:
    if not math.isfinite(x) or mu.size == 0:
        return np.nan
    if x <= mu[0]:
        return 0.0
    if x >= mu[-1]:
        return 1.0
    return float(np.interp(x, mu, cdf))


def normalized_cdf(mu: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, float, bool]:
    if mu.size < 2:
        return np.full_like(mu, np.nan), np.nan, False
    density = np.maximum(density, 0.0)
    integral = float(np.trapezoid(density, mu))
    if integral <= 0.0 or not math.isfinite(integral):
        return np.full_like(mu, np.nan), integral, False
    cdf = np.concatenate([[0.0], np.cumsum((density[:-1] + density[1:]) * np.diff(mu) / 2.0)])
    raw_final = float(cdf[-1] / integral)
    cdf = np.maximum.accumulate(np.clip(cdf / max(cdf[-1], 1e-300), 0.0, 1.0))
    return cdf, raw_final, bool(np.all(np.diff(cdf) >= -1e-10))


def density_grid_audits(density: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if density.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    summary_lookup = {
        (row["model"], "__NA__" if pd.isna(row["k"]) else str(row["k"]), row["n"], row["mu_star"], row["ref_seed"], row["backend"]): row
        for _, row in summary.iterrows()
    }
    tail_rows = []
    cdf_rows = []
    symmetry_rows = []
    mode_rows = []
    for key, group in density.groupby(CASE_COLUMNS + ["backend"], dropna=False):
        ordered = group.sort_values("mu")
        mu = ordered["mu"].to_numpy(float)
        dens = ordered["density"].to_numpy(float)
        cdf, final_cdf, monotone = normalized_cdf(mu, dens)
        row = summary_lookup.get((*key,))
        if row is None:
            continue
        q = {name: float(np.interp(prob, cdf, mu)) if np.isfinite(cdf).any() else np.nan for name, prob in QUANTILES.items()}
        raw_q025 = finite_float(row.get("raw_q025"))
        raw_q975 = finite_float(row.get("raw_q975"))
        raw_q01 = finite_float(row.get("raw_q01"))
        raw_q99 = finite_float(row.get("raw_q99"))
        raw_q50 = finite_float(row.get("raw_q50"))
        raw_sd = finite_float(row.get("raw_sd"))
        central95 = cdf_at(mu, cdf, raw_q975) - cdf_at(mu, cdf, raw_q025)
        central98 = cdf_at(mu, cdf, raw_q99) - cdf_at(mu, cdf, raw_q01)
        p_2sd = cdf_at(mu, cdf, raw_q50 - 2.0 * raw_sd) + 1.0 - cdf_at(mu, cdf, raw_q50 + 2.0 * raw_sd)
        p_3sd = cdf_at(mu, cdf, raw_q50 - 3.0 * raw_sd) + 1.0 - cdf_at(mu, cdf, raw_q50 + 3.0 * raw_sd)
        tail_warning = "none"
        if math.isfinite(central95):
            err95 = abs(central95 - 0.95)
            if err95 > 0.07:
                tail_warning = "serious"
            elif err95 > 0.03:
                tail_warning = "mild"
        if not math.isfinite(central98):
            tail_warning = "missing_raw_98_quantiles" if tail_warning == "none" else tail_warning
        tail_rows.append(
            {
                "model": key[0],
                "k": np.nan if key[1] == "__NA__" else float(key[1]),
                "n": key[2],
                "mu_star": key[3],
                "ref_seed": key[4],
                "backend": key[5],
                "p_mu_lt_raw_q025": cdf_at(mu, cdf, raw_q025),
                "p_mu_gt_raw_q975": 1.0 - cdf_at(mu, cdf, raw_q975),
                "p_mu_lt_raw_q01": cdf_at(mu, cdf, raw_q01),
                "p_mu_gt_raw_q99": 1.0 - cdf_at(mu, cdf, raw_q99),
                "p_abs_gt_2_raw_sd": p_2sd,
                "p_abs_gt_3_raw_sd": p_3sd,
                "central_mass_raw_95_interval": central95,
                "central_mass_raw_98_interval": central98,
                "tail_mass_error_95": abs(central95 - 0.95) if math.isfinite(central95) else np.nan,
                "tail_mass_error_98": abs(central98 - 0.98) if math.isfinite(central98) else np.nan,
                "warning": tail_warning,
            }
        )
        raw_width = max(raw_q975 - raw_q025, raw_sd, 1e-300)
        max_q_error = np.nanmax([abs(q["q025"] - raw_q025), abs(q["q50"] - raw_q50), abs(q["q975"] - raw_q975)])
        cdf_warning = "none"
        if not monotone or abs(final_cdf - 1.0) > 0.01 or max_q_error / raw_width > 0.05:
            cdf_warning = "serious"
        elif max_q_error / raw_width > 0.02:
            cdf_warning = "mild"
        cdf_rows.append(
            {
                "model": key[0],
                "k": np.nan if key[1] == "__NA__" else float(key[1]),
                "n": key[2],
                "mu_star": key[3],
                "ref_seed": key[4],
                "backend": key[5],
                "cdf_monotone": monotone,
                "final_cdf": final_cdf,
                **q,
                "raw_q025": raw_q025,
                "raw_q50": raw_q50,
                "raw_q975": raw_q975,
                "max_quantile_error_over_raw_width95": max_q_error / raw_width,
                "warning": cdf_warning,
            }
        )
        sym95 = abs((q["q025"] + q["q975"]) / 2.0 - float(key[3]))
        sym98 = abs((q["q01"] + q["q99"]) / 2.0 - float(key[3]))
        asym = max(abs(finite_float(row.get("mean")) - float(key[3])), abs(q["q50"] - float(key[3])), sym95, sym98)
        sym_warning = "serious" if asym / raw_width > 0.05 else "mild" if asym / raw_width > 0.02 else "none"
        symmetry_rows.append(
            {
                "model": key[0],
                "k": np.nan if key[1] == "__NA__" else float(key[1]),
                "n": key[2],
                "mu_star": key[3],
                "ref_seed": key[4],
                "backend": key[5],
                "abs_mean_minus_mu_star": abs(finite_float(row.get("mean")) - float(key[3])),
                "abs_median_minus_mu_star": abs(q["q50"] - float(key[3])),
                "abs_q_sym_95": sym95,
                "abs_q_sym_98": sym98,
                "asymmetry_over_raw_width95": asym / raw_width,
                "warning": sym_warning,
            }
        )
        max_density = float(np.nanmax(dens)) if dens.size else np.nan
        threshold = max_density * 0.01
        maxima = []
        if dens.size >= 3 and math.isfinite(max_density):
            for idx in range(1, dens.size - 1):
                if dens[idx] > dens[idx - 1] and dens[idx] >= dens[idx + 1] and dens[idx] >= threshold:
                    left_min = float(np.nanmin(dens[max(0, idx - 50) : idx + 1]))
                    right_min = float(np.nanmin(dens[idx : min(dens.size, idx + 51)]))
                    prominence = dens[idx] - max(left_min, right_min)
                    if prominence >= max_density * 0.005:
                        maxima.append((float(mu[idx]), float(dens[idx]), float(prominence)))
        mode_warning = "serious" if len(maxima) > 2 else "mild" if len(maxima) > 1 else "none"
        mode_rows.append(
            {
                "model": key[0],
                "k": np.nan if key[1] == "__NA__" else float(key[1]),
                "n": key[2],
                "mu_star": key[3],
                "ref_seed": key[4],
                "backend": key[5],
                "num_local_maxima": len(maxima),
                "peak_locations": ";".join(f"{item[0]:.6g}" for item in maxima),
                "peak_heights": ";".join(f"{item[1]:.6g}" for item in maxima),
                "peak_prominences": ";".join(f"{item[2]:.6g}" for item in maxima),
                "warning": mode_warning,
            }
        )
    return pd.DataFrame(tail_rows), pd.DataFrame(cdf_rows), pd.DataFrame(symmetry_rows), pd.DataFrame(mode_rows)


def monotonic_n(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in summary.groupby(["model", "k", "backend", "ref_seed"], dropna=False):
        ns = [11, 21, 51] if key[0] == "laplace" else [10, 20, 50]
        by_n = {int(row["n"]): row for _, row in group.iterrows()}
        selected = [by_n.get(n) for n in ns]
        if any(row is None for row in selected):
            continue
        sds = [float(row["sd"]) for row in selected]
        widths = [float(row["q975"] - row["q025"]) for row in selected]
        mono_sd = bool(sds[0] >= sds[1] >= sds[2])
        mono_width = bool(widths[0] >= widths[1] >= widths[2])
        rows.append(
            {
                "model": key[0],
                "k": key[1],
                "backend": key[2],
                "ref_seed": key[3],
                "n_values": ",".join(map(str, ns)),
                "sd_n10_or_11": sds[0],
                "sd_n20_or_21": sds[1],
                "sd_n50_or_51": sds[2],
                "width95_n10_or_11": widths[0],
                "width95_n20_or_21": widths[1],
                "width95_n50_or_51": widths[2],
                "monotone_sd_decrease": mono_sd,
                "monotone_width95_decrease": mono_width,
                "warning": "none" if mono_sd and mono_width else "serious",
            }
        )
    return pd.DataFrame(rows)


def backend_recommendations(
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    stability: pd.DataFrame,
    tail: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    score_rows = summary[summary["backend"].isin(PRIMARY_BACKENDS)].copy()
    score_rows["q_error"] = (
        score_rows["delta_q025"].abs().fillna(0.0)
        + score_rows["delta_q50"].abs().fillna(0.0)
        + score_rows["delta_q975"].abs().fillna(0.0)
    ) / (3.0 * (score_rows["raw_q975"] - score_rows["raw_q025"]).abs().replace(0, np.nan))
    score_rows["integral_error"] = (score_rows["posterior_integral_check"] - 1.0).abs().fillna(0.0)
    for key, group in summary.groupby(["model", "k", "n"], dropna=False):
        rec = {"model": key[0], "k": key[1], "n": key[2]}
        t_rows = group[group["backend"].eq("t_abram")]
        t_capped = bool((t_rows["B_used"] < t_rows["B_full"]).any() or t_rows["sample_cap_applied"].map(bool).any()) if not t_rows.empty else False
        t_status = "not_available" if t_rows.empty else "capped_diagnostic_only" if t_capped else "available_uncapped_tail_stress_test"
        case_scores = {}
        for backend in PRIMARY_BACKENDS:
            rows_b = score_rows[
                score_rows["model"].eq(key[0])
                & score_rows["n"].eq(key[2])
                & score_rows["backend"].eq(backend)
                & (score_rows["k"].isna() if pd.isna(key[1]) else score_rows["k"].eq(key[1]))
            ]
            stab = stability[
                stability["model"].eq(key[0])
                & stability["n"].eq(key[2])
                & stability["backend"].eq(backend)
                & (stability["k"].isna() if pd.isna(key[1]) else stability["k"].eq(key[1]))
            ]
            tail_b = tail[
                tail["model"].eq(key[0])
                & tail["n"].eq(key[2])
                & tail["backend"].eq(backend)
                & (tail["k"].isna() if pd.isna(key[1]) else tail["k"].eq(key[1]))
            ]
            seed_penalty = float(stab["cv_sd"].max()) if not stab.empty else np.nan
            tail_penalty = float(tail_b["tail_mass_error_95"].mean()) if not tail_b.empty else np.nan
            case_scores[backend] = float(
                rows_b["rel_sd_error"].mean()
                + rows_b["q_error"].mean()
                + rows_b["integral_error"].mean()
                + (0.0 if np.isnan(seed_penalty) else seed_penalty)
                + (0.0 if np.isnan(tail_penalty) else tail_penalty)
            )
        sens_case = sensitivity[
            sensitivity["model"].eq(key[0])
            & sensitivity["n"].eq(key[2])
            & sensitivity["backend_pair"].eq("scott vs SJ_transform")
            & (sensitivity["k"].isna() if pd.isna(key[1]) else sensitivity["k"].eq(key[1]))
        ]
        backend_disagree = bool((sens_case["warning"].isin(["serious"]).any()) if not sens_case.empty else False)
        stab_case = stability[
            stability["model"].eq(key[0])
            & stability["n"].eq(key[2])
            & stability["backend"].isin(PRIMARY_BACKENDS)
            & (stability["k"].isna() if pd.isna(key[1]) else stability["k"].eq(key[1]))
        ]
        seed_instability = bool(stab_case["warning"].isin(["serious"]).any()) if not stab_case.empty else False
        tail_case = tail[
            tail["model"].eq(key[0])
            & tail["n"].eq(key[2])
            & tail["backend"].isin(PRIMARY_BACKENDS)
            & (tail["k"].isna() if pd.isna(key[1]) else tail["k"].eq(key[1]))
        ]
        tail_instability = bool(tail_case["warning"].isin(["serious"]).any()) if not tail_case.empty else False
        forced_caution = key[0] == "student_t" and finite_float(key[1]) == 1.0 and int(key[2]) == 10
        if backend_disagree or seed_instability or tail_instability or forced_caution:
            primary = "no_single_kde_recommended"
            secondary = ""
        else:
            ordered = sorted(case_scores, key=lambda name: case_scores[name])
            primary, secondary = ordered[0], ordered[1]
        reason = []
        if forced_caution:
            reason.append("Student k=1,n=10 Cauchy-like case requires caution")
        if backend_disagree:
            reason.append("Scott and SJ_transform disagree materially")
        if seed_instability:
            reason.append("seed instability exceeds serious threshold")
        if tail_instability:
            reason.append("tail mass audit exceeds serious threshold")
        if not reason:
            reason.append(f"{primary} has smaller combined score among uncapped primary candidates")
        rows.append(
            {
                **rec,
                "recommended_primary_backend": primary,
                "secondary_backend": secondary,
                "t_abram_status": t_status,
                "combined_score_scott": case_scores.get("scott", np.nan),
                "combined_score_SJ_transform": case_scores.get("SJ_transform", np.nan),
                "backend_disagreement_flag": backend_disagree,
                "seed_instability_flag": seed_instability,
                "tail_instability_flag": tail_instability,
                "recommendation_reason": "; ".join(reason),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "k", "n"])


def suspicious_cases(*tables: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for audit, table in tables:
        if table.empty or "warning" not in table:
            continue
        flagged = table[~table["warning"].astype(str).isin(["", "none"])]
        flagged = flagged[~flagged["warning"].astype(str).eq("missing_raw_98_quantiles")]
        for _, row in flagged.iterrows():
            severity = "high" if str(row["warning"]) == "serious" else "medium"
            metric = ""
            value = np.nan
            for candidate in ["rel_sd_diff", "cv_sd", "tail_mass_error_95", "max_quantile_error_over_raw_width95", "asymmetry_over_raw_width95", "num_local_maxima"]:
                if candidate in row and pd.notna(row[candidate]):
                    metric = candidate
                    value = row[candidate]
                    break
            rows.append(
                {
                    "severity": severity,
                    "audit": audit,
                    "model": row.get("model"),
                    "k": row.get("k"),
                    "n": row.get("n"),
                    "ref_seed": row.get("ref_seed", np.nan),
                    "backend": row.get("backend", row.get("backend_pair", "")),
                    "reason": row.get("warning"),
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    suspicious: pd.DataFrame,
    manifest: dict,
) -> None:
    no_single = int(recommendations["recommended_primary_backend"].eq("no_single_kde_recommended").sum())
    capped = int(summary[summary["backend"].eq("t_abram")]["sample_cap_applied"].map(bool).sum())
    high = int(suspicious["severity"].eq("high").sum()) if not suspicious.empty else 0
    lines = [
        "# KDE Correctness Audit",
        "",
        "## 1. Executive summary",
        f"- Audited {summary[['model', 'k', 'n', 'mu_star', 'ref_seed']].drop_duplicates().shape[0]} cached posterior cases across Scott, SJ_transform, and t_abram KDE grids.",
        f"- {no_single} model/k/n groups are marked `no_single_kde_recommended`.",
        f"- {high} high-severity suspicious audit rows were found.",
        f"- t_abram capped rows: {capped}.",
        "",
        "## 2. What KDE is and is not used for",
        "Raw weighted-MC is the posterior-summary benchmark. KDE is used as a smoothed density diagnostic for visualization and backend sensitivity, not as ground truth. If KDE backends disagree materially, KDE should not be used to make scientific conclusions.",
        "",
        "## 3. Student-t results by k",
        "Student-t cases are available for k=1,2,3 and n=10,20,50. The k=1 family is the main stress case; Student k=1,n=10 should be interpreted with caution because Cauchy-like likelihood geometry can make smooth density summaries backend-sensitive.",
        "",
        "## 4. Logistic results",
        "Logistic cases are audited at n=10,20,50. Use the recommendation table to choose Scott or SJ_transform only when raw-summary agreement, backend sensitivity, seed stability, and tail mass checks are all acceptable.",
        "",
        "## 5. Laplace odd-n results",
        "Laplace scalar median KDE comparisons use odd n values 11,21,51. The even-n interval target is separate and is intentionally not mixed into this KDE audit.",
        "",
        "## 6. Cauchy k=1 warning",
        "Student k=1,n=10 is forced to `no_single_kde_recommended` unless future cached metrics are unexpectedly stable enough to justify a specific smoothed backend. Treat figures for this case as diagnostic views, not scientific evidence by themselves.",
        "",
        "## 7. t_abram capped diagnostic note",
        "Capped t_abram is only a visual/tail diagnostic and is never recommended as a primary backend. Uncapped t_abram may still be useful as a tail stress test, but final primary recommendations are restricted to Scott and SJ_transform.",
        "",
        "## 8. Backend recommendations",
        recommendations.to_markdown(index=False),
        "",
        "## 9. Suspicious cases",
        suspicious.head(50).to_markdown(index=False) if not suspicious.empty else "No suspicious cases were flagged by the configured thresholds.",
        "",
        "## 10. Dashboard suggestions",
        "Show backend recommendations, suspicious cases, t_abram cap status, the Student k=1,n=10 caution, and paths to generated figures. The dashboard should read these generated files and should not recompute the audit.",
        "",
        "## Cache provenance",
        f"- Data level: {manifest.get('data_level', 'unknown')}",
        f"- Cache created at: {manifest.get('created_at', 'unknown')}",
    ]
    (out_dir / "kde_correctness_report.md").write_text("\n".join(lines), encoding="utf-8")


def case_label(row: pd.Series) -> str:
    k = "na" if pd.isna(row.get("k")) else f"k={float(row['k']):g}"
    return f"{row['model']} {k} n={int(row['n'])}"


def save_figures(
    out_dir: Path,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    stability: pd.DataFrame,
    tail: pd.DataFrame,
    density: pd.DataFrame,
    monotonic: pd.DataFrame,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    agg = summary.groupby(["model", "k", "n", "backend"], dropna=False)["rel_sd_error"].mean().reset_index()
    pivot = agg.pivot_table(index=["model", "k", "n"], columns="backend", values="rel_sd_error", dropna=False)
    plt.figure(figsize=(8, max(4, 0.32 * len(pivot))))
    plt.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="mean rel sd error vs raw")
    plt.yticks(range(len(pivot)), [f"{idx[0]} k={idx[1]} n={idx[2]}" for idx in pivot.index], fontsize=7)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "backend_summary_error_heatmap.png", dpi=160)
    plt.close()

    stab = stability.pivot_table(index=["model", "k", "n"], columns="backend", values="cv_sd", dropna=False)
    plt.figure(figsize=(8, max(4, 0.32 * len(stab))))
    plt.imshow(stab.fillna(0.0).to_numpy(), aspect="auto", cmap="magma")
    plt.colorbar(label="cv(sd) across seeds")
    plt.yticks(range(len(stab)), [f"{idx[0]} k={idx[1]} n={idx[2]}" for idx in stab.index], fontsize=7)
    plt.xticks(range(len(stab.columns)), stab.columns, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "seed_stability_heatmap.png", dpi=160)
    plt.close()

    tail_agg = tail.groupby(["model", "k", "n", "backend"], dropna=False)["tail_mass_error_95"].mean().reset_index()
    tail_pivot = tail_agg.pivot_table(index=["model", "k", "n"], columns="backend", values="tail_mass_error_95", dropna=False)
    plt.figure(figsize=(8, max(4, 0.32 * len(tail_pivot))))
    plt.imshow(tail_pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="plasma")
    plt.colorbar(label="mean |KDE mass(raw 95%) - 0.95|")
    plt.yticks(range(len(tail_pivot)), [f"{idx[0]} k={idx[1]} n={idx[2]}" for idx in tail_pivot.index], fontsize=7)
    plt.xticks(range(len(tail_pivot.columns)), tail_pivot.columns, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "tail_probability_error_heatmap.png", dpi=160)
    plt.close()

    representative = [
        ("student_t", 1.0, 10),
        ("student_t", 2.0, 20),
        ("student_t", 3.0, 20),
        ("logistic", np.nan, 20),
        ("laplace", np.nan, 21),
    ]
    for model, k, n in representative:
        subset = density[density["model"].eq(model) & density["n"].eq(n)]
        subset = subset[subset["k"].isna()] if pd.isna(k) else subset[subset["k"].eq(k)]
        if subset.empty:
            continue
        seed = sorted(subset["ref_seed"].unique())[0]
        subset = subset[subset["ref_seed"].eq(seed)]
        plt.figure(figsize=(7, 4))
        for backend, group in subset.groupby("backend"):
            ordered = group.sort_values("mu")
            plt.plot(ordered["mu"], ordered["density"], label=backend, linewidth=1.5)
        k_label = "na" if pd.isna(k) else f"{float(k):g}"
        plt.title(f"Backend density overlay: {model} k={k_label} n={n} seed={seed}")
        plt.xlabel("mu")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        fname = f"density_overlay_{model}_k{k_label}_n{n}.png"
        plt.savefig(fig_dir / fname, dpi=160)
        plt.close()

    cdf_subset = density[density["model"].eq("student_t") & density["k"].eq(1.0) & density["n"].eq(10)]
    if not cdf_subset.empty:
        seed = sorted(cdf_subset["ref_seed"].unique())[0]
        cdf_subset = cdf_subset[cdf_subset["ref_seed"].eq(seed)]
        plt.figure(figsize=(7, 4))
        for backend, group in cdf_subset.groupby("backend"):
            ordered = group.sort_values("mu")
            cdf, _, _ = normalized_cdf(ordered["mu"].to_numpy(float), ordered["density"].to_numpy(float))
            plt.plot(ordered["mu"], cdf, label=backend)
        plt.title(f"CDF overlay: student_t k=1 n=10 seed={seed}")
        plt.xlabel("mu")
        plt.ylabel("cdf")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "cdf_overlay_student_t_k1_n10.png", dpi=160)
        plt.close()

    if not monotonic.empty:
        for (model, backend), group in monotonic.groupby(["model", "backend"]):
            plt.figure(figsize=(7, 4))
            for _, row in group.iterrows():
                ns = [int(part) for part in str(row["n_values"]).split(",")]
                sds = [row["sd_n10_or_11"], row["sd_n20_or_21"], row["sd_n50_or_51"]]
                label_k = "k=na" if pd.isna(row["k"]) else f"k={float(row['k']):g}"
                plt.plot(ns, sds, marker="o", label=f"{label_k} seed={int(row['ref_seed'])}", alpha=0.8)
            plt.title(f"n-monotonicity: {model} {backend}")
            plt.xlabel("n")
            plt.ylabel("posterior sd")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(fig_dir / f"n_monotonicity_{model}_{backend}.png", dpi=160)
            plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref, density, manifest = load_inputs(args)
    dens_sum = density_summary(density)
    summary = make_summary(ref, dens_sum)
    sensitivity = pairwise_backend_sensitivity(summary, density)
    stability = seed_stability(summary)
    tail, cdf_quantile, symmetry, mode = density_grid_audits(density, summary)
    monotonic = monotonic_n(summary)
    recommendations = backend_recommendations(summary, sensitivity, stability, tail)
    suspicious = suspicious_cases(
        ("backend_sensitivity", sensitivity),
        ("seed_stability", stability),
        ("tail_probability", tail),
        ("cdf_quantile", cdf_quantile),
        ("symmetry", symmetry),
        ("monotonic_n", monotonic),
        ("mode_bump", mode),
    )

    outputs = {
        "kde_correctness_summary.csv": summary,
        "backend_sensitivity.csv": sensitivity,
        "seed_stability.csv": stability,
        "tail_probability_audit.csv": tail,
        "cdf_quantile_audit.csv": cdf_quantile,
        "symmetry_audit.csv": symmetry,
        "monotonic_n_audit.csv": monotonic,
        "mode_bump_audit.csv": mode,
        "backend_recommendations.csv": recommendations,
        "suspicious_kde_cases.csv": suspicious,
    }
    for name, table in outputs.items():
        table.to_csv(args.out_dir / name, index=False)
    write_report(args.out_dir, summary, recommendations, suspicious, manifest)
    save_figures(args.out_dir, summary, sensitivity, stability, tail, density, monotonic)
    print(f"Wrote KDE correctness audit to {args.out_dir}")


if __name__ == "__main__":
    main()
