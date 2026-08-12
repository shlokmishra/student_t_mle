"""Analyze latent geometry diagnostics across named cached runsets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "geometry_audit" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results") / "geometry_audit" / ".cache"))
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.run_registry import load_common_run_outputs, load_run_registry, resolve_runset_paths


GROUP_KEYS = ["runset", "model", "k_key", "n", "method"]
METHOD_KEYS = ["model", "k_key", "n", "method"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-registry", type=Path, default=Path("configs/analysis_run_registry.yaml"))
    parser.add_argument("--runsets", nargs="+", default=["final_production_v1"])
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"),
    )
    parser.add_argument("--correctness-dir", type=Path, default=Path("results/sampler_correctness_audit/"))
    parser.add_argument("--efficiency-dir", type=Path, default=Path("results/efficiency_audit/"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/geometry_audit/"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def k_key(value) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def add_k_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "k" not in out.columns:
        out["k"] = np.nan
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    out["k_key"] = out["k"].map(k_key)
    if "initialization" not in out.columns:
        out["initialization"] = "unspecified"
    out["initialization"] = out["initialization"].fillna("unspecified").astype(str)
    return out


def numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def x_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith("x_") and col.split("_", 1)[1].isdigit()]
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def entropy_concentration(weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(np.abs(weights)))
    if total <= 0:
        return np.nan, np.nan
    p = np.abs(weights) / total
    entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-300))))
    concentration = float(np.max(p))
    return entropy, concentration


def student_geometry(x: np.ndarray, mu_star: float, k: float) -> dict:
    y = x - mu_star
    abs_y = np.abs(y)
    z = y / (k + y * y)
    sqrt_k = float(np.sqrt(k))
    tail_count = int(np.sum(abs_y > sqrt_k))
    tail_2_count = int(np.sum(abs_y > 2.0 * sqrt_k))
    extreme_count = int(np.sum(abs_y > 5.0 * sqrt_k))
    far_tail_count = int(np.sum(abs_y > 20.0 * sqrt_k))
    entropy, concentration = entropy_concentration(z)
    abs_z = np.abs(z)
    abs_z_sum = float(np.sum(abs_z))
    if extreme_count > 0:
        geom_class = "extreme_tail"
    elif tail_count == 0:
        geom_class = "no_tail"
    elif tail_count == 1:
        geom_class = "one_tail"
    else:
        geom_class = "multiple_tail"
    positive = int(np.sum(z > 0))
    negative = int(np.sum(z < 0))
    return {
        "score_residual": float(np.sum(z)),
        "max_abs_y": float(np.max(abs_y)),
        "mean_abs_y": float(np.mean(abs_y)),
        "q90_abs_y": float(np.quantile(abs_y, 0.90)),
        "q95_abs_y": float(np.quantile(abs_y, 0.95)),
        "num_extreme_y_gt_sqrt_k": tail_count,
        "num_extreme_y_gt_2sqrt_k": tail_2_count,
        "num_extreme_y_gt_5sqrt_k": extreme_count,
        "num_extreme_y_gt_20sqrt_k": far_tail_count,
        "fraction_y_gt_sqrt_k": float(tail_count / len(x)),
        "fraction_y_gt_2sqrt_k": float(tail_2_count / len(x)),
        "fraction_y_gt_5sqrt_k": float(extreme_count / len(x)),
        "fraction_y_gt_20sqrt_k": float(far_tail_count / len(x)),
        "max_abs_z": float(np.max(np.abs(z))),
        "sum_abs_z": abs_z_sum,
        "score_collapse_ratio": float(np.max(abs_y) / max(float(np.max(abs_z)), 1e-300)),
        "z_entropy": entropy,
        "fraction_score_largest_abs_z": concentration,
        "central_branch_count": int(np.sum(abs_y < sqrt_k)),
        "tail_branch_count": tail_count,
        "tail_branch_fraction": float(tail_count / len(x)),
        "z_positive_count": positive,
        "z_negative_count": negative,
        "z_sign_balance": float((positive - negative) / len(x)),
        "latent_geometry_class": geom_class,
        "geometry_family": "student_tail_branch",
    }


def logistic_geometry(x: np.ndarray, mu_star: float) -> dict:
    y = x - mu_star
    abs_y = np.abs(y)
    z = np.tanh(y / 2.0)
    saturation_count = int(np.sum(np.abs(z) > 0.95))
    if saturation_count == 0:
        geom_class = "unsaturated"
    elif saturation_count == 1:
        geom_class = "one_saturated"
    else:
        geom_class = "multiple_saturated"
    positive = int(np.sum(z > 0))
    negative = int(np.sum(z < 0))
    return {
        "score_residual": float(np.sum(z)),
        "max_abs_y": float(np.max(abs_y)),
        "mean_abs_y": float(np.mean(abs_y)),
        "q90_abs_y": float(np.quantile(abs_y, 0.90)),
        "q95_abs_y": float(np.quantile(abs_y, 0.95)),
        "saturation_count": saturation_count,
        "saturation_fraction": float(saturation_count / len(x)),
        "max_abs_z": float(np.max(np.abs(z))),
        "z_positive_count": positive,
        "z_negative_count": negative,
        "z_sign_balance": float((positive - negative) / len(x)),
        "latent_geometry_class": geom_class,
        "geometry_family": "logistic_saturation",
    }


def laplace_geometry(x: np.ndarray, mu_star: float) -> dict:
    y = x - mu_star
    below = int(np.sum(x < mu_star))
    equal = int(np.sum(np.isclose(x, mu_star, atol=1e-10, rtol=0.0)))
    above = int(np.sum(x > mu_star))
    median_residual = float(np.median(x) - mu_star)
    if abs(median_residual) <= 1e-8 and len(x) % 2 == 1:
        geom_class = "standard_odd_median"
    else:
        geom_class = "tail_heavy_latent"
    return {
        "median_residual": median_residual,
        "count_below_mu_star": below,
        "count_equal_mu_star": equal,
        "count_above_mu_star": above,
        "max_abs_y": float(np.max(np.abs(y))),
        "mean_abs_y": float(np.mean(np.abs(y))),
        "q90_abs_y": float(np.quantile(np.abs(y), 0.90)),
        "q95_abs_y": float(np.quantile(np.abs(y), 0.95)),
        "latent_geometry_class": geom_class,
        "geometry_family": "laplace_order_median",
    }


def latent_geometry_rows(latent: pd.DataFrame, runset: str) -> pd.DataFrame:
    if latent.empty:
        return pd.DataFrame()
    latent = add_k_key(latent)
    cols = x_columns(latent)
    rows: list[dict] = []
    for row in latent.itertuples(index=False):
        data = row._asdict()
        model = str(data.get("model"))
        n = int(data.get("n"))
        method = str(data.get("method"))
        k = float(data.get("k")) if not pd.isna(data.get("k")) else np.nan
        mu_star = float(data.get("mu_star", 0.0))
        base = {
            "runset": runset,
            "model": model,
            "k": k,
            "k_key": k_key(k),
            "n": n,
            "method": method,
            "seed": int(data.get("seed", 0)),
            "initialization": str(data.get("initialization", "unspecified")),
            "iteration": int(data.get("iteration", 0)),
            "mu": float(data.get("mu", np.nan)),
            "mu_star": mu_star,
        }
        if cols:
            xs = np.asarray([data[col] for col in cols[:n]], dtype=float)
            xs = xs[np.isfinite(xs)]
            if xs.size == 0:
                continue
            base["num_x"] = int(xs.size)
            if model == "student_t" and np.isfinite(k):
                geom = student_geometry(xs, mu_star, k)
            elif model == "logistic":
                geom = logistic_geometry(xs, mu_star)
            elif model == "laplace":
                geom = laplace_geometry(xs, mu_star)
            else:
                geom = {"latent_geometry_class": "unsupported_model", "geometry_family": "unsupported"}
        else:
            max_abs_y = float(data.get("x_abs_max", data.get("max_abs_y", np.nan)))
            x_sd = float(data.get("x_sd", np.nan))
            constraint = float(data.get("constraint_residual", np.nan))
            cached_class = data.get("latent_tail_geometry_class", data.get("latent_geometry_class", np.nan))
            cached_class = str(cached_class) if pd.notna(cached_class) else ""
            tail_fraction = float(data.get("tail_fraction", np.nan))
            extreme_tail_fraction = float(data.get("extreme_tail_fraction", np.nan))
            gram_value = float(data.get("gram_value", np.nan))
            base["num_x"] = n
            if model == "student_t" and np.isfinite(k):
                sqrt_k = float(np.sqrt(k))
                if cached_class:
                    geom_class = cached_class
                elif np.isfinite(max_abs_y) and max_abs_y > 5.0 * sqrt_k:
                    geom_class = "extreme_tail"
                elif np.isfinite(max_abs_y) and max_abs_y > sqrt_k:
                    geom_class = "multiple_tail"
                else:
                    geom_class = "no_tail"
                geom = {
                    "score_residual": constraint,
                    "max_abs_y": max_abs_y,
                    "mean_abs_y": np.nan,
                    "q90_abs_y": np.nan,
                    "q95_abs_y": np.nan,
                    "tail_branch_fraction": tail_fraction,
                    "extreme_tail_fraction": extreme_tail_fraction,
                    "latent_geometry_class": geom_class,
                    "geometry_family": "student_tail_branch_summary",
                    "x_sd": x_sd,
                    "gram_value": gram_value,
                }
            elif model == "logistic":
                geom = {
                    "score_residual": constraint,
                    "max_abs_y": max_abs_y,
                    "mean_abs_y": np.nan,
                    "q90_abs_y": np.nan,
                    "q95_abs_y": np.nan,
                    "saturation_fraction": tail_fraction,
                    "extreme_tail_fraction": extreme_tail_fraction,
                    "latent_geometry_class": cached_class or "logistic_summary",
                    "geometry_family": "logistic_saturation_summary",
                    "x_sd": x_sd,
                    "gram_value": gram_value,
                }
            elif model == "laplace":
                geom = {
                    "median_residual": float(data.get("median_minus_mu_star", 0.0)),
                    "max_abs_y": max_abs_y,
                    "mean_abs_y": np.nan,
                    "q90_abs_y": np.nan,
                    "q95_abs_y": np.nan,
                    "latent_geometry_class": cached_class or ("standard_odd_median" if n % 2 == 1 else "laplace_summary"),
                    "geometry_family": "laplace_order_median_summary",
                    "x_sd": x_sd,
                    "gram_value": gram_value,
                }
            else:
                geom = {"latent_geometry_class": "unsupported_summary", "geometry_family": "unsupported"}
        rows.append({**base, **geom})
    return pd.DataFrame(rows)


def summarize_geometry(latent_geom: pd.DataFrame) -> pd.DataFrame:
    if latent_geom.empty:
        return pd.DataFrame()
    agg_map = {
        "score_residual": ["mean", "std", "max"],
        "max_abs_y": ["mean", "median", "max"],
        "mean_abs_y": ["mean"],
        "q95_abs_y": ["mean", "max"],
        "tail_branch_fraction": ["mean", "max"],
        "saturation_fraction": ["mean", "max"],
        "fraction_score_largest_abs_z": ["mean", "max"],
        "max_abs_z": ["mean", "max"],
        "sum_abs_z": ["mean"],
        "score_collapse_ratio": ["mean", "max"],
        "fraction_y_gt_sqrt_k": ["mean", "max"],
        "fraction_y_gt_2sqrt_k": ["mean", "max"],
        "fraction_y_gt_5sqrt_k": ["mean", "max"],
        "fraction_y_gt_20sqrt_k": ["mean", "max"],
        "num_extreme_y_gt_20sqrt_k": ["mean", "max"],
        "median_residual": ["mean", "max"],
    }
    available = {metric: funcs for metric, funcs in agg_map.items() if metric in latent_geom.columns}
    summary = latent_geom.groupby(GROUP_KEYS, dropna=False).agg(available).reset_index()
    summary.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in summary.columns]
    counts = latent_geom.groupby(GROUP_KEYS + ["latent_geometry_class"], dropna=False).size().reset_index(name="class_count")
    totals = latent_geom.groupby(GROUP_KEYS, dropna=False).size().reset_index(name="num_states")
    dominant = counts.sort_values("class_count", ascending=False).drop_duplicates(GROUP_KEYS).rename(
        columns={"latent_geometry_class": "dominant_geometry_class", "class_count": "dominant_class_count"}
    )
    out = summary.merge(totals, on=GROUP_KEYS, how="left").merge(
        dominant[GROUP_KEYS + ["dominant_geometry_class", "dominant_class_count"]],
        on=GROUP_KEYS,
        how="left",
    )
    out["dominant_class_fraction"] = out["dominant_class_count"] / out["num_states"].replace(0, np.nan)
    return out


def posterior_summary(values: pd.Series) -> dict:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {key: np.nan for key in ["mean", "sd", "q025", "q50", "q975", "ess_proxy"]}
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "q025": float(np.quantile(arr, 0.025)),
        "q50": float(np.quantile(arr, 0.50)),
        "q975": float(np.quantile(arr, 0.975)),
        "ess_proxy": float(arr.size),
    }


def geometry_conditioned_posterior(latent_geom: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    if latent_geom.empty:
        return pd.DataFrame()
    reference = add_k_key(reference)
    raw = reference[reference.get("estimator_type", "").astype(str).eq("raw_weighted_mc")].copy()
    raw = raw[["model", "k_key", "n", "sd"]].drop_duplicates(["model", "k_key", "n"]).rename(columns={"sd": "raw_sd"})
    rows: list[dict] = []
    full = {}
    for keys, part in latent_geom.groupby(GROUP_KEYS, dropna=False):
        full[keys] = posterior_summary(part["mu"])
    for keys, part in latent_geom.groupby(GROUP_KEYS + ["latent_geometry_class"], dropna=False):
        runset, model, kk, n, method, geom_class = keys
        stats = posterior_summary(part["mu"])
        full_stats = full[(runset, model, kk, n, method)]
        row = {
            "runset": runset,
            "model": model,
            "k": np.nan if kk == "NA" else float(kk),
            "k_key": kk,
            "n": int(n),
            "method": method,
            "latent_geometry_class": geom_class,
            "num_samples": int(len(part)),
            "fraction_of_chain": float(len(part) / len(latent_geom[(latent_geom["runset"].eq(runset)) & (latent_geom["model"].eq(model)) & (latent_geom["k_key"].eq(kk)) & (latent_geom["n"].eq(n)) & (latent_geom["method"].eq(method))])),
            "mu_mean": stats["mean"],
            "mu_sd": stats["sd"],
            "q025": stats["q025"],
            "q50": stats["q50"],
            "q975": stats["q975"],
            "ess_proxy": stats["ess_proxy"],
            "delta_mean": stats["mean"] - full_stats["mean"],
            "delta_sd": stats["sd"] - full_stats["sd"],
            "delta_q025": stats["q025"] - full_stats["q025"],
            "delta_q975": stats["q975"] - full_stats["q975"],
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(raw, on=["model", "k_key", "n"], how="left")
    out["flag_mean_shift_gt_015_raw_sd"] = out["delta_mean"].abs() > 0.15 * out["raw_sd"].abs()
    out["flag_sd_shift_gt_15pct"] = out["delta_sd"].abs() > 0.15 * out["mu_sd"].replace(0, np.nan).abs()
    out["flag_low_class_fraction"] = out["fraction_of_chain"] < 0.05
    return out


def class_transitions(latent_geom: pd.DataFrame) -> pd.DataFrame:
    if latent_geom.empty:
        return pd.DataFrame()
    rows = []
    for keys, part in latent_geom.sort_values("iteration").groupby(GROUP_KEYS + ["seed", "initialization"], dropna=False):
        runset, model, kk, n, method, seed, initialization = keys
        classes = part["latent_geometry_class"].astype(str).to_numpy()
        if classes.size < 2:
            continue
        changes = classes[1:] != classes[:-1]
        for from_cls in sorted(set(classes)):
            mask = classes[:-1] == from_cls
            denom = int(np.sum(mask))
            if denom == 0:
                continue
            for to_cls in sorted(set(classes)):
                count = int(np.sum(mask & (classes[1:] == to_cls)))
                rows.append(
                    {
                        "runset": runset,
                        "model": model,
                        "k": np.nan if kk == "NA" else float(kk),
                        "k_key": kk,
                        "n": int(n),
                        "method": method,
                        "seed": int(seed),
                        "initialization": initialization,
                        "from_class": from_cls,
                        "to_class": to_cls,
                        "transition_count": count,
                        "transition_probability": float(count / denom),
                    }
                )
        rows.append(
            {
                "runset": runset,
                "model": model,
                "k": np.nan if kk == "NA" else float(kk),
                "k_key": kk,
                "n": int(n),
                "method": method,
                "seed": int(seed),
                "initialization": initialization,
                "from_class": "__summary__",
                "to_class": "__changed__",
                "transition_count": int(np.sum(changes)),
                "transition_probability": float(np.mean(changes)),
            }
        )
    return pd.DataFrame(rows)


def summarize_branch_diagnostics(branch: pd.DataFrame) -> pd.DataFrame:
    if branch.empty:
        return branch
    branch = add_k_key(branch)
    group_cols = [col for col in ["runset", "model", "k_key", "n", "method", "seed", "initialization"] if col in branch.columns]
    if "branch_pair" in branch.columns and "frequency" in branch.columns:
        pivot = (
            branch.pivot_table(
                index=group_cols,
                columns="branch_pair",
                values="frequency",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
    else:
        pivot = branch[group_cols].drop_duplicates().copy() if group_cols else pd.DataFrame(index=[0])
    rate = pd.DataFrame()
    if "branch_switching_rate" in branch.columns:
        rate = branch.groupby(group_cols, dropna=False)["branch_switching_rate"].mean().reset_index()
    out = pivot.merge(rate, on=group_cols, how="left") if not rate.empty and not pivot.empty else (rate if pivot.empty else pivot)
    if out.empty:
        return out
    for pair in ["lower/lower", "lower/upper", "upper/lower", "upper/upper"]:
        if pair not in out.columns:
            out[pair] = 0.0
    out["branch_diagnostic_available"] = True
    out["tail_tail_pair_fraction"] = out["upper/upper"]
    out["mixed_pair_fraction"] = out["lower/upper"] + out["upper/lower"]
    out["central_central_pair_fraction"] = out["lower/lower"]
    out["branch_imbalance"] = (out["upper/upper"] - out["lower/lower"]).abs()
    out["note"] = "Branch diagnostics loaded from runset branch_diagnostics.csv."
    return out


def gibbs_explanations(
    latent_geom: pd.DataFrame,
    correctness_dir: Path,
    runset_branch: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    branch_frames = []
    baseline_branch = read_csv(correctness_dir / "gibbs_branch_diagnostics.csv")
    if not baseline_branch.empty:
        baseline_branch["runset"] = "correctness_audit"
        branch_frames.append(baseline_branch)
    if runset_branch is not None and not runset_branch.empty:
        branch_frames.append(runset_branch)
    raw_branch = pd.concat(branch_frames, ignore_index=True, sort=False) if branch_frames else pd.DataFrame()
    branch = summarize_branch_diagnostics(raw_branch) if not raw_branch.empty else pd.DataFrame()
    trans = class_transitions(latent_geom[latent_geom["method"].eq("gibbs")]) if not latent_geom.empty else pd.DataFrame()
    if trans.empty:
        local = pd.DataFrame()
    else:
        summary = trans[trans["from_class"].eq("__summary__")].copy()
        summary["local_trapping_score"] = 1.0 - summary["transition_probability"]
        summary["local_trapping_status"] = np.select(
            [summary["local_trapping_score"] >= 0.95, summary["local_trapping_score"] >= 0.80],
            ["high", "moderate"],
            default="low",
        )
        local = summary.rename(columns={"transition_probability": "class_switching_rate"})[
            ["runset", "model", "k", "k_key", "n", "method", "seed", "initialization", "class_switching_rate", "local_trapping_score", "local_trapping_status"]
        ]
    if branch.empty and local.empty:
        return pd.DataFrame(), branch, local
    expl = local.copy()
    if not branch.empty and not expl.empty:
        merge_keys = [key for key in ["runset", "model", "k_key", "n", "method", "seed", "initialization"] if key in expl.columns and key in branch.columns]
        expl = expl.merge(
            branch[
                [
                    col
                    for col in [
                        *merge_keys,
                        "branch_diagnostic_available",
                        "branch_switching_rate",
                        "branch_imbalance",
                        "tail_tail_pair_fraction",
                        "mixed_pair_fraction",
                        "central_central_pair_fraction",
                        "note",
                    ]
                    if col in branch.columns
                ]
            ],
            on=merge_keys,
            how="left",
        )
    elif not branch.empty:
        expl = branch.copy()
        expl["runset"] = "correctness_audit"
    if not expl.empty:
        expl["geometry_explanation"] = np.where(
            expl.get("local_trapping_status", "").astype(str).eq("high"),
            "Cached latent classes have high diagonal persistence; inspect pair-delta and branch diagnostics before claiming fast latent exploration.",
            "Cached latent classes switch often enough in thinned diagnostics; pair-delta and branch diagnostics provide the exact Gibbs checks.",
        )
    return expl, branch, local


def rattle_explanations(
    latent_geom: pd.DataFrame,
    correctness_dir: Path,
    efficiency_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geom = read_csv(correctness_dir / "rattle_geometry_diagnostics.csv")
    energy = read_csv(correctness_dir / "rattle_energy_diagnostics.csv")
    movement = read_csv(efficiency_dir / "rattle_movement_diagnostics.csv")
    for name in ["geom", "energy", "movement"]:
        frame = locals()[name]
        if not frame.empty:
            locals()[name] = add_k_key(frame)
    geom = add_k_key(geom) if not geom.empty else pd.DataFrame()
    energy = add_k_key(energy) if not energy.empty else pd.DataFrame()
    movement = add_k_key(movement) if not movement.empty else pd.DataFrame()
    agg = summarize_geometry(latent_geom[latent_geom["method"].eq("rattle")]) if not latent_geom.empty else pd.DataFrame()
    base = geom.copy() if not geom.empty else pd.DataFrame()
    if not base.empty and not agg.empty:
        base = base.merge(
            agg[["model", "k_key", "n", "method", "max_abs_y_mean", "max_abs_y_max", "dominant_geometry_class", "tail_branch_fraction_mean"]],
            on=["model", "k_key", "n", "method"],
            how="left",
        )
    if not base.empty and not energy.empty:
        cols = ["model", "k_key", "n", "method", "seed", "energy_diagnostic_available", "mean_delta_H", "q95_delta_H", "acceptance_rate", "note"]
        energy_keys = ["model", "k_key", "n", "method"]
        if "seed" in base.columns and "seed" in energy.columns:
            energy_keys.append("seed")
        base = base.merge(energy[[col for col in cols if col in energy.columns]], on=energy_keys, how="left", suffixes=("", "_energy"))
    if not base.empty and not movement.empty:
        cols = ["model", "k_key", "n", "method", "esjd_mu_per_sec", "median_abs_delta_mu", "high_acceptance_small_move_flag"]
        base = base.merge(movement[[col for col in cols if col in movement.columns]].groupby(["model", "k_key", "n", "method"], dropna=False).median(numeric_only=True).reset_index(), on=["model", "k_key", "n", "method"], how="left")
    if not base.empty:
        base["geometry_explanation"] = np.where(
            base.get("energy_diagnostic_available", False).fillna(False).astype(bool),
            "Energy diagnostics available; inspect delta_H/tail relationships.",
            "Projection/reversibility diagnostics are available, but delta_H and x-move tail relationships are not cached.",
        )
    tail_failure = pd.DataFrame()
    if not base.empty:
        tail_failure = base[
            [
                col
                for col in [
                    "model",
                    "k",
                    "k_key",
                    "n",
                    "method",
                    "rattle_status",
                    "max_abs_y_mean",
                    "max_abs_y_max",
                    "tail_branch_fraction_mean",
                    "energy_diagnostic_available",
                    "mean_delta_H",
                    "q95_delta_H",
                    "forward_newton_iters_per_proposal",
                    "reverse_newton_iters_per_proposal",
                    "projection_failure_rate",
                    "reverse_check_failure_rate",
                    "acceptance_rate",
                    "esjd_mu_per_sec",
                    "high_acceptance_small_move_flag",
                    "geometry_explanation",
                ]
                if col in base.columns
            ]
        ].copy()
        tail_failure["tail_failure_relationship_available"] = tail_failure.get("energy_diagnostic_available", False).fillna(False).astype(bool)
        tail_failure["recommendation"] = np.where(
            tail_failure["tail_failure_relationship_available"],
            "Use cached relationship diagnostics.",
            "Use final production diagnostics with per-proposal delta_H, Newton iterations, Gram, and tail-state links.",
        )
    return base, tail_failure


def geometry_win_loss(
    geom_summary: pd.DataFrame,
    correctness_dir: Path,
    efficiency_dir: Path,
) -> pd.DataFrame:
    verdicts = add_k_key(read_csv(correctness_dir / "final_sampler_verdict_table.csv"))
    winners = add_k_key(read_csv(efficiency_dir / "method_winners.csv"))
    if verdicts.empty:
        return pd.DataFrame()
    rows = []
    summary_lookup = geom_summary.set_index(GROUP_KEYS) if not geom_summary.empty else None
    for _, row in verdicts.iterrows():
        model = row["model"]
        kk = row["k_key"]
        n = int(row["n"])
        method = row["method"]
        gtext = "latent diagnostics missing"
        if summary_lookup is not None:
            matches = geom_summary[
                geom_summary["model"].eq(model)
                & geom_summary["k_key"].eq(kk)
                & geom_summary["n"].eq(n)
                & geom_summary["method"].eq(method)
            ]
            if not matches.empty:
                grow = matches.iloc[0]
                gtext = f"dominant={grow.get('dominant_geometry_class', 'unknown')}; max_abs_y_mean={grow.get('max_abs_y_mean', np.nan):.3g}"
        winner = "not_ranked"
        if not winners.empty:
            w = winners[(winners["model"].eq(model)) & (winners["k_key"].eq(kk)) & (winners["n"].eq(n))]
            if not w.empty:
                winner = str(w.iloc[0].get("recommended_efficiency_winner", "not_ranked"))
        if model == "logistic":
            reason = "smooth latent geometry; RATTLE global moves often help" if method == "rattle" else "smooth latent geometry; Gibbs is correct but usually less efficient in cache"
        elif model == "student_t" and kk == "1":
            reason = "heavy-tail branch geometry and target sensitivity; unresolved/caveat regime"
        elif model == "student_t":
            reason = "manageable tails; RATTLE wins at larger n when movement is adequate" if method == "rattle" else "local Gibbs moves are correct but can be slower as n grows"
        elif model == "laplace":
            reason = "RATTLE not applicable due nonsmooth median/order geometry" if method == "rattle" else "odd-n scalar median geometry supports Gibbs-only baseline"
        else:
            reason = "geometry explanation unavailable"
        caveat = row.get("main_warning", "")
        if str(row.get("safe_to_present", "")) != "yes":
            recommended = "caveat_or_hide"
        elif winner == "rattle" and method == "rattle":
            recommended = "presentation_ok"
        elif winner == "gibbs" and method == "gibbs":
            recommended = "presentation_ok"
        elif winner == "gibbs_only" and method == "gibbs":
            recommended = "presentation_ok"
        else:
            recommended = "supporting_context"
        rows.append(
            {
                "model": model,
                "k": row.get("k", np.nan),
                "k_key": kk,
                "n": n,
                "method": method,
                "correctness_verdict": row.get("verdict", ""),
                "efficiency_winner": winner,
                "geometry_summary": gtext,
                "likely_reason_for_win_or_failure": reason,
                "caveat": caveat,
                "recommended_status": recommended,
            }
        )
    return pd.DataFrame(rows)


def missing_rows(
    run_outputs: list[dict],
    latent_geom: pd.DataFrame,
    correctness_dir: Path,
) -> pd.DataFrame:
    rows = []
    for out in run_outputs:
        rows.extend(out["missing"])
    has_transition = any(not out["tables"].get("transition_diagnostics", pd.DataFrame()).empty for out in run_outputs)
    has_branch = any(not out["tables"].get("branch_diagnostics", pd.DataFrame()).empty for out in run_outputs)
    energy_tables = [out["tables"].get("rattle_energy_diagnostics", pd.DataFrame()) for out in run_outputs]
    has_energy_link = any(
        (not table.empty)
        and any(col in table.columns and pd.to_numeric(table[col], errors="coerce").notna().any() for col in ["delta_H_mean_abs", "delta_h_mean_abs", "delta_H_max_abs", "delta_h_max_abs"])
        for table in energy_tables
    )
    if latent_geom.empty or not latent_geom["model"].eq("logistic").any():
        rows.append(
            {
                "runset": "all_runsets",
                "diagnostic": "logistic_latent_geometry",
                "path": "latent_x_diagnostics.csv",
                "severity": "medium",
                "message": "No Logistic latent geometry diagnostics found; smooth-geometry explanation relies on correctness/efficiency summaries.",
            }
        )
    if latent_geom.empty or not latent_geom["model"].eq("laplace").any():
        rows.append(
            {
                "runset": "all_runsets",
                "diagnostic": "laplace_odd_n_latent_geometry",
                "path": "latent_x_diagnostics.csv",
                "severity": "medium",
                "message": "No odd-n Laplace latent geometry diagnostics found; Laplace geometry is limited to applicability/correctness logic.",
            }
        )
    for diagnostic, path, present in [
        ("exact_gibbs_pair_delta_transition_diagnostics", "transition_diagnostics.csv", has_transition),
        ("exact_student_branch_labels", "branch_diagnostics.csv", has_branch),
    ]:
        if present:
            continue
        rows.append(
            {
                "runset": "all_runsets",
                "diagnostic": diagnostic,
                "path": path,
                "severity": "high",
                "message": "Required for exact Gibbs pair/branch transition claims; final production should include this diagnostic.",
            }
        )
    energy = read_csv(correctness_dir / "rattle_energy_diagnostics.csv")
    baseline_energy = (not energy.empty) and energy.get("energy_diagnostic_available", pd.Series(dtype=bool)).fillna(False).astype(bool).any()
    if not baseline_energy and not has_energy_link:
        rows.append(
            {
                "runset": "all_runsets",
                "diagnostic": "rattle_delta_H_tail_relationship",
                "path": "rattle_energy_diagnostics.csv",
                "severity": "high",
                "message": "delta_H and tail-state diagnostics were not cached; cannot test whether RATTLE avoids tails due Hamiltonian/Newton errors.",
            }
        )
    return pd.DataFrame(rows).drop_duplicates()


def unresolved_cases(win_loss: pd.DataFrame, conditioned: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not win_loss.empty:
        unresolved = win_loss[
            win_loss["correctness_verdict"].astype(str).isin(["unresolved", "pass_with_warning"])
            | win_loss["recommended_status"].astype(str).eq("caveat_or_hide")
        ]
        rows.extend(unresolved.to_dict(orient="records"))
    if not conditioned.empty:
        flagged = conditioned[
            conditioned[["flag_mean_shift_gt_015_raw_sd", "flag_sd_shift_gt_15pct", "flag_low_class_fraction"]]
            .fillna(False)
            .any(axis=1)
        ]
        for _, row in flagged.iterrows():
            rows.append(
                {
                    "model": row["model"],
                    "k": row["k"],
                    "k_key": row["k_key"],
                    "n": row["n"],
                    "method": row["method"],
                    "correctness_verdict": "geometry_conditioned_flag",
                    "efficiency_winner": "",
                    "geometry_summary": row["latent_geometry_class"],
                    "likely_reason_for_win_or_failure": "Posterior summary changes across latent geometry class or class is rare.",
                    "caveat": "geometry-conditioned posterior flag",
                    "recommended_status": "targeted_validation_needed",
                }
            )
    if not missing.empty:
        for _, row in missing[missing["severity"].isin(["high", "medium"])].iterrows():
            rows.append(
                {
                    "model": "",
                    "k": np.nan,
                    "k_key": "",
                    "n": np.nan,
                    "method": "",
                    "correctness_verdict": "missing_diagnostic",
                    "efficiency_winner": "",
                    "geometry_summary": row["diagnostic"],
                    "likely_reason_for_win_or_failure": row["message"],
                    "caveat": row["path"],
                    "recommended_status": "targeted_validation_needed",
                }
            )
    return pd.DataFrame(rows).drop_duplicates()


def write_figures(
    out_dir: Path,
    latent_geom: pd.DataFrame,
    conditioned: pd.DataFrame,
    transitions: pd.DataFrame,
    rattle_tail: pd.DataFrame,
    branch: pd.DataFrame,
    win_loss: pd.DataFrame,
) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(fig, name: str) -> None:
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    student = latent_geom[latent_geom["model"].eq("student_t")] if not latent_geom.empty else pd.DataFrame()
    if not student.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = []
        vals = []
        for keys, part in student.groupby(["k_key", "n", "method"], dropna=False):
            labels.append(f"k={keys[0]} n={keys[1]} {keys[2]}")
            vals.append(part["max_abs_y"].to_numpy(dtype=float))
        ax.boxplot(vals, tick_labels=labels, showfliers=False)
        ax.set_ylabel("max |x_i - mu_star|")
        ax.set_title("Student tail geometry")
        ax.tick_params(axis="x", rotation=70, labelsize=7)
        save(fig, "student_tail_geometry_histogram.png")

        fig, ax = plt.subplots(figsize=(9, 5))
        occ = student.groupby(["k_key", "n", "method"], dropna=False)["tail_branch_fraction"].mean().reset_index()
        ax.bar(np.arange(len(occ)), occ["tail_branch_fraction"])
        ax.set_xticks(np.arange(len(occ)))
        ax.set_xticklabels([f"k={r.k_key} n={r.n} {r.method}" for r in occ.itertuples()], rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("mean tail branch fraction")
        save(fig, "student_branch_occupancy.png")

        cauchy = student[student["k_key"].astype(str).isin(["1", "1.0"])].copy()
        if not cauchy.empty and {"max_abs_y", "max_abs_z"}.issubset(cauchy.columns):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(cauchy["max_abs_y"], cauchy["max_abs_z"], s=12, alpha=0.45)
            ax.set_xscale("log")
            ax.set_xlabel("max |x_i - mu_star|")
            ax.set_ylabel("max |z_i|, z=y/(1+y^2)")
            ax.set_title("Cauchy tail size versus bounded score coordinate")
            save(fig, "student_k1_score_collapse_scatter.png")

    if not transitions.empty:
        summary = transitions[transitions["from_class"].ne("__summary__")]
        if not summary.empty:
            class_order = ["central", "mixed_tail", "extreme_tail", "tail_dominated"]
            counts = (
                summary.groupby(["from_class", "to_class"], dropna=False)["transition_count"]
                .sum()
                .unstack(fill_value=0)
                .reindex(index=class_order, columns=class_order, fill_value=0)
            )
            row_totals = counts.sum(axis=1)
            pivot = counts.div(row_totals.replace(0, np.nan), axis=0)

            fig, ax = plt.subplots(figsize=(6.8, 5.4))
            values = pivot.to_numpy(dtype=float)
            im = ax.imshow(values, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(col).replace("_", "\n") for col in pivot.columns], rotation=0)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([f"{str(idx).replace('_', ' ')}  (n={int(row_totals.get(idx, 0))})" for idx in pivot.index])
            ax.set_xlabel("next latent class")
            ax.set_ylabel("current latent class")
            ax.set_title("Row-normalized pooled latent-class transitions")
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    val = values[i, j]
                    text = "--" if not np.isfinite(val) else f"{val:.2f}"
                    color = "white" if np.isfinite(val) and val >= 0.55 else "#1f1f1f"
                    ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
            fig.colorbar(im, ax=ax, label="P(next class | current class)")
            save(fig, "latent_geometry_class_transition_heatmap.png")

    if not conditioned.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        data = []
        labels = []
        for keys, part in latent_geom.groupby(["latent_geometry_class"], dropna=False):
            data.append(part["mu"].dropna().to_numpy(dtype=float))
            labels.append(str(keys))
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.set_ylabel("mu")
            ax.tick_params(axis="x", rotation=45)
            save(fig, "geometry_conditioned_mu_posterior_boxplot.png")

    if not rattle_tail.empty:
        for xcol, name, ylabel in [
            ("mean_delta_H", "rattle_delta_H_vs_max_abs_y.png", "mean delta_H"),
            ("forward_newton_iters_per_proposal", "rattle_newton_iters_vs_max_abs_y.png", "forward Newton iters/proposal"),
            ("max_abs_constraint_residual", "rattle_gram_vs_max_abs_y.png", "constraint residual proxy"),
        ]:
            if xcol in rattle_tail.columns and "max_abs_y_mean" in rattle_tail.columns and rattle_tail[xcol].notna().any():
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(rattle_tail["max_abs_y_mean"], rattle_tail[xcol])
                ax.set_xlabel("mean max_abs_y")
                ax.set_ylabel(ylabel)
                save(fig, name)

    if not branch.empty and ("approx_middle_fraction" in branch.columns or "branch_switching_rate" in branch.columns):
        fig, ax = plt.subplots(figsize=(7, 4))
        branch = add_k_key(branch)
        branch_plot = branch.copy()
        if "approx_middle_fraction" in branch_plot.columns:
            y = branch_plot["approx_middle_fraction"]
            ylabel = "approx middle branch fraction"
        else:
            branch_plot = branch_plot.groupby(["k_key", "n"], dropna=False)["branch_switching_rate"].mean().reset_index()
            y = branch_plot["branch_switching_rate"]
            ylabel = "mean branch switching rate"
        labels = [f"k={r.k_key} n={r.n}" for r in branch_plot.itertuples()]
        ax.bar(np.arange(len(branch_plot)), y)
        ax.set_xticks(np.arange(len(branch_plot)))
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        save(fig, "gibbs_branch_switching_rate_by_case.png")

    if not win_loss.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = win_loss.groupby(["model", "recommended_status"], dropna=False).size().unstack(fill_value=0)
        im = ax.imshow(counts.to_numpy(dtype=float))
        ax.set_xticks(np.arange(len(counts.columns)))
        ax.set_xticklabels(counts.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(counts.index)))
        ax.set_yticklabels(counts.index)
        fig.colorbar(im, ax=ax, label="case count")
        save(fig, "geometry_win_loss_summary_heatmap.png")
    return paths


def frame_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def write_report(
    out_dir: Path,
    runsets: list[str],
    latent_geom: pd.DataFrame,
    summary: pd.DataFrame,
    conditioned: pd.DataFrame,
    rattle_exp: pd.DataFrame,
    gibbs_exp: pd.DataFrame,
    win_loss: pd.DataFrame,
    missing: pd.DataFrame,
    unresolved: pd.DataFrame,
    figures: list[str],
) -> None:
    lines = [
        "# Geometry Audit",
        "",
        "This audit explains sampler behavior using cached latent geometry where available. It does not run samplers, change transition logic, or use KDE as a benchmark. Raw weighted-MC remains the posterior-summary benchmark.",
        "",
        "## Runsets",
        "",
        "- " + ", ".join(runsets),
        "",
        "## Available Geometry",
        "",
    ]
    if latent_geom.empty:
        lines.append("No per-state latent geometry diagnostics were available.")
    else:
        lines.append(f"Per-state latent geometry rows: `{len(latent_geom)}`.")
        lines.append(frame_md(summary.head(20)))
    lines.extend(["", "## Geometry-Conditioned Posterior", "", frame_md(conditioned.head(30))])
    lines.extend(["", "## RATTLE Geometry Explanation", "", frame_md(rattle_exp.head(30))])
    lines.extend(["", "## Gibbs Geometry Explanation", "", frame_md(gibbs_exp.head(30))])
    lines.extend(["", "## Geometry Win/Loss Table", "", frame_md(win_loss)])
    lines.extend(["", "## Missing Diagnostics", "", frame_md(missing)])
    lines.extend(["", "## Unresolved Geometry Cases", "", frame_md(unresolved.head(60))])
    lines.extend(
        [
            "",
            "## Current Supported Explanations",
            "",
            "- Final production diagnostics are the intended source for thinned latent geometry, Gibbs pair-delta checks, branch usage, RATTLE delta_H, Newton, Gram, and reversibility summaries.",
            "- Student-t geometry is summarized through latent tail classes and max-|x_i-mu_star| diagnostics.",
            "- Logistic efficiency/correctness patterns are interpreted as smooth-geometry cases when latent diagnostics are available.",
            "- RATTLE explanations should use projection/reversibility, tangent residual, delta_H, Newton, Gram, and tail-state diagnostics together.",
            "- Gibbs explanations should use pair-delta preservation, branch usage, and latent-class switching together.",
            "- Student-t k=1,n=10 remains unresolved unless final production diagnostics supply stronger evidence.",
            "- Laplace RATTLE remains not applicable; odd-n Laplace Gibbs is the presentation baseline.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in figures)
    (out_dir / "geometry_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    registry = load_run_registry(args.run_registry)

    outputs = []
    latent_frames = []
    branch_frames = []
    for runset_name in args.runsets:
        runset = resolve_runset_paths(runset_name, registry)
        out = load_common_run_outputs(runset)
        outputs.append(out)
        latent = out["tables"].get("latent_x_diagnostics", pd.DataFrame())
        if latent.empty:
            latent = out["tables"].get("latent_diagnostics", pd.DataFrame())
        if latent.empty:
            latent = out["tables"].get("geometry_diagnostics", pd.DataFrame())
        if not latent.empty:
            latent_frames.append(latent_geometry_rows(latent, runset.label))
        branch = out["tables"].get("branch_diagnostics", pd.DataFrame())
        if not branch.empty:
            branch = branch.copy()
            branch["runset"] = runset.label
            branch_frames.append(branch)

    latent_geom = pd.concat(latent_frames, ignore_index=True, sort=False) if latent_frames else pd.DataFrame()
    runset_branch = pd.concat(branch_frames, ignore_index=True, sort=False) if branch_frames else pd.DataFrame()
    reference = read_csv(args.reference_csv)
    geom_summary = summarize_geometry(latent_geom)
    conditioned = geometry_conditioned_posterior(latent_geom, reference)
    gibbs_exp, branch, local_move = gibbs_explanations(latent_geom, args.correctness_dir, runset_branch)
    rattle_exp, rattle_tail = rattle_explanations(latent_geom, args.correctness_dir, args.efficiency_dir)
    win_loss = geometry_win_loss(geom_summary, args.correctness_dir, args.efficiency_dir)
    missing = missing_rows(outputs, latent_geom, args.correctness_dir)
    unresolved = unresolved_cases(win_loss, conditioned, missing)
    transitions = class_transitions(latent_geom)

    latent_geom.to_csv(args.out_dir / "latent_tail_geometry.csv", index=False)
    transitions.to_csv(args.out_dir / "latent_tail_geometry_class_transitions.csv", index=False)
    geom_summary.to_csv(args.out_dir / "geometry_summary.csv", index=False)
    conditioned.to_csv(args.out_dir / "geometry_conditioned_posterior.csv", index=False)
    rattle_exp.to_csv(args.out_dir / "rattle_geometry_explanation.csv", index=False)
    gibbs_exp.to_csv(args.out_dir / "gibbs_geometry_explanation.csv", index=False)
    branch.to_csv(args.out_dir / "branch_exploration.csv", index=False)
    rattle_tail.to_csv(args.out_dir / "rattle_tail_failure_analysis.csv", index=False)
    local_move.to_csv(args.out_dir / "gibbs_local_move_analysis.csv", index=False)
    win_loss.to_csv(args.out_dir / "geometry_win_loss_table.csv", index=False)
    missing.to_csv(args.out_dir / "missing_geometry_diagnostics.csv", index=False)
    unresolved.to_csv(args.out_dir / "unresolved_geometry_cases.csv", index=False)

    figures = write_figures(args.out_dir, latent_geom, conditioned, transitions, rattle_tail, branch, win_loss)
    write_report(
        args.out_dir,
        args.runsets,
        latent_geom,
        geom_summary,
        conditioned,
        rattle_exp,
        gibbs_exp,
        win_loss,
        missing,
        unresolved,
        figures,
    )
    manifest = {
        "runsets": args.runsets,
        "rows": {
            "latent_tail_geometry": int(len(latent_geom)),
            "geometry_summary": int(len(geom_summary)),
            "geometry_conditioned_posterior": int(len(conditioned)),
            "rattle_geometry_explanation": int(len(rattle_exp)),
            "gibbs_geometry_explanation": int(len(gibbs_exp)),
            "geometry_win_loss_table": int(len(win_loss)),
            "missing_geometry_diagnostics": int(len(missing)),
        },
        "figures": figures,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
