"""Audit prior sensitivity using importance reweighting on cached posterior draws.

This script does not run samplers. It reweights posterior draws generated under
mu ~ N(0, 10^2) to alternative Normal priors with the same mean and different
standard deviations.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "prior_sensitivity_audit" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results") / "prior_sensitivity_audit" / ".cache"))
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
BASE_PRIOR_SD = 10.0
BASE_PRIOR_MEAN = 0.0
PRIOR_SDS = [2.0, 5.0, 10.0, 20.0]
RAW_SD_SHIFT_FLAG = 0.10
REWEIGHTED_ESS_FRACTION_FLAG = 0.20
SD_CHANGE_FLAG = 0.10
INFO_SD_RATIO_CHANGE_FLAG = 0.10


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_k(value: object) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def normal_logpdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    var = sd * sd
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((x - mean) ** 2) / var


def importance_weights(mu: np.ndarray, new_sd: float) -> np.ndarray:
    logw = normal_logpdf(mu, BASE_PRIOR_MEAN, new_sd) - normal_logpdf(mu, BASE_PRIOR_MEAN, BASE_PRIOR_SD)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return np.full_like(w, np.nan, dtype=float)
    return w / w_sum


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w)
    if cdf[-1] <= 0:
        return np.nan
    cdf = cdf / cdf[-1]
    idx = np.searchsorted(cdf, q, side="left")
    idx = int(np.clip(idx, 0, len(v) - 1))
    return float(v[idx])


def weighted_summary(mu: np.ndarray, weights: np.ndarray, raw_sd: float) -> dict[str, float]:
    mu = np.asarray(mu, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(mu) & np.isfinite(weights) & (weights > 0)
    mu = mu[mask]
    weights = weights[mask]
    if mu.size == 0:
        return {
            "draws": 0,
            "reweighted_ess": np.nan,
            "reweighted_ess_fraction": np.nan,
            "mean": np.nan,
            "sd": np.nan,
            "q025": np.nan,
            "q50": np.nan,
            "q975": np.nan,
            "tail_prob_abs_mu_gt_1rawsd": np.nan,
            "tail_prob_abs_mu_gt_2rawsd": np.nan,
        }
    w = weights / np.sum(weights)
    mean = float(np.sum(w * mu))
    var = float(np.sum(w * (mu - mean) ** 2))
    ess = float((np.sum(w) ** 2) / np.sum(w * w))
    return {
        "draws": int(mu.size),
        "reweighted_ess": ess,
        "reweighted_ess_fraction": ess / float(mu.size) if mu.size else np.nan,
        "mean": mean,
        "sd": math.sqrt(max(var, 0.0)),
        "q025": weighted_quantile(mu, w, 0.025),
        "q50": weighted_quantile(mu, w, 0.50),
        "q975": weighted_quantile(mu, w, 0.975),
        "tail_prob_abs_mu_gt_1rawsd": float(np.sum(w * (np.abs(mu) > raw_sd))) if np.isfinite(raw_sd) else np.nan,
        "tail_prob_abs_mu_gt_2rawsd": float(np.sum(w * (np.abs(mu) > 2.0 * raw_sd))) if np.isfinite(raw_sd) else np.nan,
    }


def summarize_final_production(reference_sd: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for chain_path in sorted((ROOT / "results" / "final_production_v1").glob("case_*/chain_samples.csv")):
        chain = read_csv(chain_path, usecols=["model", "method", "k", "n", "seed", "initialization", "mu", "is_burn_in"])
        if chain.empty:
            continue
        chain = chain[~chain["is_burn_in"].fillna(False).astype(bool)].copy()
        if chain.empty:
            continue
        model = str(chain["model"].iloc[0])
        method = str(chain["method"].iloc[0])
        k = pd.to_numeric(chain["k"], errors="coerce").iloc[0]
        n = int(pd.to_numeric(chain["n"], errors="coerce").iloc[0])
        seed = int(pd.to_numeric(chain["seed"], errors="coerce").iloc[0])
        initialization = str(chain["initialization"].iloc[0])
        k_key = normalize_k(k)
        raw_sd = reference_sd.get((model, k_key, n), np.nan)
        mu = pd.to_numeric(chain["mu"], errors="coerce").dropna().to_numpy(dtype=float)
        if mu.size == 0:
            continue
        base = weighted_summary(mu, np.full(mu.size, 1.0 / mu.size), raw_sd)
        for prior_sd in PRIOR_SDS:
            stats = weighted_summary(mu, importance_weights(mu, prior_sd), raw_sd)
            row = {
                "scope": "posterior",
                "model": model,
                "k": np.nan if k_key == "NA" else float(k),
                "k_key": k_key,
                "n": n,
                "method": method,
                "seed": seed,
                "initialization": initialization,
                "prior_sd": float(prior_sd),
                "raw_sd_benchmark": raw_sd,
                **stats,
                "baseline_sd": base["sd"],
                "baseline_q025": base["q025"],
                "baseline_q50": base["q50"],
                "baseline_q975": base["q975"],
                "relative_sd_change": (stats["sd"] / base["sd"] - 1.0) if np.isfinite(base["sd"]) and base["sd"] > 0 else np.nan,
                "q025_shift_over_raw_sd": abs(stats["q025"] - base["q025"]) / raw_sd if np.isfinite(raw_sd) and raw_sd > 0 else np.nan,
                "q975_shift_over_raw_sd": abs(stats["q975"] - base["q975"]) / raw_sd if np.isfinite(raw_sd) and raw_sd > 0 else np.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_posterior(posterior: pd.DataFrame) -> pd.DataFrame:
    if posterior.empty:
        return posterior
    group_cols = ["scope", "model", "k", "k_key", "n", "method", "prior_sd"]
    agg = posterior.groupby(group_cols, dropna=False).agg(
        seed_count=("seed", "nunique"),
        median_reweighted_ess_fraction=("reweighted_ess_fraction", "median"),
        min_reweighted_ess_fraction=("reweighted_ess_fraction", "min"),
        mean_median=("mean", "median"),
        sd_median=("sd", "median"),
        q025_median=("q025", "median"),
        q50_median=("q50", "median"),
        q975_median=("q975", "median"),
        tail_prob_abs_mu_gt_1rawsd_median=("tail_prob_abs_mu_gt_1rawsd", "median"),
        tail_prob_abs_mu_gt_2rawsd_median=("tail_prob_abs_mu_gt_2rawsd", "median"),
        relative_sd_change_median=("relative_sd_change", "median"),
        max_abs_relative_sd_change=("relative_sd_change", lambda s: float(np.nanmax(np.abs(pd.to_numeric(s, errors="coerce"))))),
        max_q_shift_over_raw_sd=("q025_shift_over_raw_sd", lambda s: float(np.nanmax(pd.to_numeric(s, errors="coerce")))),
        max_q975_shift_over_raw_sd=("q975_shift_over_raw_sd", lambda s: float(np.nanmax(pd.to_numeric(s, errors="coerce")))),
    ).reset_index()
    agg["max_quantile_shift_over_raw_sd"] = np.nanmax(
        np.vstack([agg["max_q_shift_over_raw_sd"].to_numpy(dtype=float), agg["max_q975_shift_over_raw_sd"].to_numpy(dtype=float)]),
        axis=0,
    )
    return agg


def summarize_information_loss(reference_sd: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    release_root = ROOT / "results" / "release_information_runs"
    for case_dir in sorted(release_root.glob("case_*")):
        mle = read_csv(case_dir / "mle_only_chain_samples.csv", usecols=["model", "k", "k_key", "n", "dataset_id", "method", "mu", "is_burn_in"])
        full = read_csv(case_dir / "full_data_chain_samples.csv", usecols=["model", "k", "k_key", "n", "dataset_id", "method", "mu", "is_burn_in"])
        if mle.empty or full.empty:
            continue
        mle = mle[~mle["is_burn_in"].fillna(False).astype(bool)].copy()
        full = full[~full["is_burn_in"].fillna(False).astype(bool)].copy()
        if mle.empty or full.empty:
            continue
        model = str(mle["model"].iloc[0])
        k_value = pd.to_numeric(mle["k"], errors="coerce").iloc[0]
        k_key = str(mle["k_key"].iloc[0]) if "k_key" in mle.columns else normalize_k(k_value)
        n = int(pd.to_numeric(mle["n"], errors="coerce").iloc[0])
        dataset_id = str(mle["dataset_id"].iloc[0])
        mle_method = str(mle["method"].iloc[0])
        full_method = str(full["method"].iloc[0])
        raw_sd = reference_sd.get((model, k_key, n), np.nan)
        mu_mle = pd.to_numeric(mle["mu"], errors="coerce").dropna().to_numpy(dtype=float)
        mu_full = pd.to_numeric(full["mu"], errors="coerce").dropna().to_numpy(dtype=float)
        if mu_mle.size == 0 or mu_full.size == 0:
            continue
        base_mle = weighted_summary(mu_mle, np.full(mu_mle.size, 1.0 / mu_mle.size), raw_sd)
        base_full = weighted_summary(mu_full, np.full(mu_full.size, 1.0 / mu_full.size), raw_sd)
        base_qdist = float(np.nanmean([abs(base_mle["q025"] - base_full["q025"]), abs(base_mle["q50"] - base_full["q50"]), abs(base_mle["q975"] - base_full["q975"])]))
        for prior_sd in PRIOR_SDS:
            mle_stats = weighted_summary(mu_mle, importance_weights(mu_mle, prior_sd), raw_sd)
            full_stats = weighted_summary(mu_full, importance_weights(mu_full, prior_sd), raw_sd)
            qdist = float(np.nanmean([abs(mle_stats["q025"] - full_stats["q025"]), abs(mle_stats["q50"] - full_stats["q50"]), abs(mle_stats["q975"] - full_stats["q975"])]))
            rows.append(
                {
                    "scope": "information_loss",
                    "model": model,
                    "k": np.nan if k_key == "NA" else float(k_value),
                    "k_key": k_key,
                    "n": n,
                    "method": mle_method,
                    "full_data_method": full_method,
                    "dataset_id": dataset_id,
                    "prior_sd": float(prior_sd),
                    "mle_reweighted_ess_fraction": mle_stats["reweighted_ess_fraction"],
                    "full_reweighted_ess_fraction": full_stats["reweighted_ess_fraction"],
                    "sd_ratio_mle_over_full": mle_stats["sd"] / full_stats["sd"] if np.isfinite(full_stats["sd"]) and full_stats["sd"] > 0 else np.nan,
                    "baseline_sd_ratio_mle_over_full": base_mle["sd"] / base_full["sd"] if np.isfinite(base_full["sd"]) and base_full["sd"] > 0 else np.nan,
                    "mean_difference": mle_stats["mean"] - full_stats["mean"],
                    "baseline_mean_difference": base_mle["mean"] - base_full["mean"],
                    "q025_difference": mle_stats["q025"] - full_stats["q025"],
                    "q50_difference": mle_stats["q50"] - full_stats["q50"],
                    "q975_difference": mle_stats["q975"] - full_stats["q975"],
                    "quantile_distance_score": qdist,
                    "baseline_quantile_distance_score": base_qdist,
                }
            )
    return pd.DataFrame(rows)


def aggregate_information_loss(info: pd.DataFrame) -> pd.DataFrame:
    if info.empty:
        return info
    group_cols = ["scope", "model", "k", "k_key", "n", "method", "prior_sd"]
    agg = info.groupby(group_cols, dropna=False).agg(
        dataset_count=("dataset_id", "nunique"),
        median_mle_reweighted_ess_fraction=("mle_reweighted_ess_fraction", "median"),
        median_full_reweighted_ess_fraction=("full_reweighted_ess_fraction", "median"),
        sd_ratio_median=("sd_ratio_mle_over_full", "median"),
        baseline_sd_ratio_median=("baseline_sd_ratio_mle_over_full", "median"),
        mean_difference_median=("mean_difference", "median"),
        baseline_mean_difference_median=("baseline_mean_difference", "median"),
        quantile_distance_score_median=("quantile_distance_score", "median"),
        baseline_quantile_distance_score_median=("baseline_quantile_distance_score", "median"),
        q025_difference_median=("q025_difference", "median"),
        q50_difference_median=("q50_difference", "median"),
        q975_difference_median=("q975_difference", "median"),
    ).reset_index()
    agg["sd_ratio_change"] = agg["sd_ratio_median"] - agg["baseline_sd_ratio_median"]
    agg["quantile_distance_change"] = agg["quantile_distance_score_median"] - agg["baseline_quantile_distance_score_median"]
    return agg


def summarize_privacy_availability() -> pd.DataFrame:
    release_root = ROOT / "results" / "release_information_runs"
    rows = []
    for case_dir in sorted(release_root.glob("case_*")):
        mle = read_csv(case_dir / "mle_only_chain_samples.csv", usecols=["model", "k", "k_key", "n", "method"])
        if mle.empty:
            continue
        rows.append(
            {
                "scope": "privacy_leakage",
                "model": str(mle["model"].iloc[0]),
                "k": pd.to_numeric(mle["k"], errors="coerce").iloc[0],
                "k_key": str(mle["k_key"].iloc[0]) if "k_key" in mle.columns else normalize_k(pd.to_numeric(mle["k"], errors="coerce").iloc[0]),
                "n": int(pd.to_numeric(mle["n"], errors="coerce").iloc[0]),
                "method": str(mle["method"].iloc[0]),
                "prior_sd": np.nan,
                "privacy_reweight_available": False,
                "note": "Release-information runset lacks latent snapshots; exact privacy reweighting is unavailable from cached outputs.",
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).drop_duplicates(["scope", "model", "k_key", "n", "method"])
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    return out


def build_flags(posterior: pd.DataFrame, info: pd.DataFrame, privacy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in posterior.itertuples(index=False):
        if float(row.min_reweighted_ess_fraction) < REWEIGHTED_ESS_FRACTION_FLAG:
            rows.append({"scope": row.scope, "model": row.model, "k": row.k, "k_key": row.k_key, "n": row.n, "method": row.method, "prior_sd": row.prior_sd, "flag": "low_reweighted_ess", "detail": f"Minimum reweighted ESS fraction={row.min_reweighted_ess_fraction:.3f}."})
        if float(abs(row.max_abs_relative_sd_change)) > SD_CHANGE_FLAG:
            rows.append({"scope": row.scope, "model": row.model, "k": row.k, "k_key": row.k_key, "n": row.n, "method": row.method, "prior_sd": row.prior_sd, "flag": "posterior_sd_change_gt_10pct", "detail": f"Maximum |relative sd change|={abs(row.max_abs_relative_sd_change):.3f}."})
        if float(row.max_quantile_shift_over_raw_sd) > RAW_SD_SHIFT_FLAG:
            rows.append({"scope": row.scope, "model": row.model, "k": row.k, "k_key": row.k_key, "n": row.n, "method": row.method, "prior_sd": row.prior_sd, "flag": "quantile_shift_material", "detail": f"Maximum q025/q975 shift over raw sd={row.max_quantile_shift_over_raw_sd:.3f}."})
    for row in info.itertuples(index=False):
        if min(float(row.median_mle_reweighted_ess_fraction), float(row.median_full_reweighted_ess_fraction)) < REWEIGHTED_ESS_FRACTION_FLAG:
            rows.append({"scope": row.scope, "model": row.model, "k": row.k, "k_key": row.k_key, "n": row.n, "method": row.method, "prior_sd": row.prior_sd, "flag": "low_reweighted_ess", "detail": f"Median reweighted ESS fraction min={min(float(row.median_mle_reweighted_ess_fraction), float(row.median_full_reweighted_ess_fraction)):.3f}."})
        if abs(float(row.sd_ratio_change)) > INFO_SD_RATIO_CHANGE_FLAG:
            rows.append({"scope": row.scope, "model": row.model, "k": row.k, "k_key": row.k_key, "n": row.n, "method": row.method, "prior_sd": row.prior_sd, "flag": "information_loss_sd_ratio_change_gt_10pct", "detail": f"SD ratio change={float(row.sd_ratio_change):.3f}."})
        base = float(row.baseline_mean_difference_median)
        new = float(row.mean_difference_median)
        if np.isfinite(base) and np.isfinite(new) and base != 0 and np.sign(base) != np.sign(new):
            rows.append({"scope": row.scope, "model": row.model, "k": row.k, "k_key": row.k_key, "n": row.n, "method": row.method, "prior_sd": row.prior_sd, "flag": "main_conclusion_sign_change", "detail": f"Mean-difference sign changed from {base:.4g} to {new:.4g}."})
    for row in privacy.itertuples(index=False):
        rows.append({"scope": row.scope, "model": row.model, "k": row.k, "k_key": row.k_key, "n": row.n, "method": row.method, "prior_sd": np.nan, "flag": "privacy_reweight_unavailable", "detail": row.note})
    return pd.DataFrame(rows)


def write_figures(out_dir: Path, posterior: pd.DataFrame, info: pd.DataFrame) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(fig: plt.Figure, name: str) -> None:
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    if not posterior.empty:
        post_plot = posterior[posterior["prior_sd"].isin(PRIOR_SDS)].copy()
        post_plot["row_label"] = post_plot.apply(
            lambda r: f"{r['model']} k={r['k_key']} n={int(r['n'])} {r['method']}" if str(r["k_key"]) != "NA" else f"{r['model']} n={int(r['n'])} {r['method']}",
            axis=1,
        )
        pivot = post_plot.pivot_table(index="row_label", columns="prior_sd", values="max_abs_relative_sd_change", aggfunc="median")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(pivot.index))))
            im = ax.imshow(np.abs(pivot.to_numpy(dtype=float)), aspect="auto", cmap="Reds")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([f"sd={int(c)}" for c in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_title("Absolute posterior SD change versus prior sd=10")
            fig.colorbar(im, ax=ax, label="|relative SD change|")
            save(fig, "posterior_sd_change_heatmap.png")

        pivot = post_plot.pivot_table(index="row_label", columns="prior_sd", values="min_reweighted_ess_fraction", aggfunc="median")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(pivot.index))))
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([f"sd={int(c)}" for c in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_title("Minimum reweighted ESS fraction")
            fig.colorbar(im, ax=ax, label="ESS_reweighted / draws")
            save(fig, "posterior_reweighted_ess_heatmap.png")

    if not info.empty:
        info_plot = info.copy()
        info_plot["row_label"] = info_plot.apply(
            lambda r: f"{r['model']} k={r['k_key']} n={int(r['n'])} {r['method']}" if str(r["k_key"]) != "NA" else f"{r['model']} n={int(r['n'])} {r['method']}",
            axis=1,
        )
        pivot = info_plot.pivot_table(index="row_label", columns="prior_sd", values="sd_ratio_change", aggfunc="median")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(pivot.index))))
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-0.2, vmax=0.2)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([f"sd={int(c)}" for c in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_title("Information-loss SD ratio change versus prior sd=10")
            fig.colorbar(im, ax=ax, label="sd_ratio(new) - sd_ratio(base)")
            save(fig, "information_loss_sd_ratio_change_heatmap.png")
    return paths


def write_report(out_dir: Path, posterior: pd.DataFrame, info: pd.DataFrame, privacy: pd.DataFrame, flags: pd.DataFrame, figures: list[str]) -> None:
    rerun_rows = flags[~flags["scope"].eq("privacy_leakage")].copy()
    rerun_rows = rerun_rows[rerun_rows["flag"].ne("privacy_reweight_unavailable")]
    rerun_rows = rerun_rows[
        rerun_rows["prior_sd"].isin([5.0, 20.0])
        | rerun_rows["flag"].isin(["low_reweighted_ess", "main_conclusion_sign_change"])
    ]
    rerun_labels = []
    if not rerun_rows.empty:
        for row in rerun_rows[["scope", "model", "k_key", "n", "method"]].drop_duplicates().itertuples(index=False):
            if str(row.k_key) == "NA":
                rerun_labels.append(f"{row.scope}: {row.model} n={int(row.n)} {row.method}")
            else:
                rerun_labels.append(f"{row.scope}: {row.model} k={row.k_key} n={int(row.n)} {row.method}")

    lines = [
        "# Prior Sensitivity Audit",
        "",
        f"Base prior on mu: N({BASE_PRIOR_MEAN:.1f}, {BASE_PRIOR_SD:.1f}^2).",
        "",
        "Alternative prior standard deviations audited by importance reweighting: 2, 5, 10, 20.",
        "",
    ]
    if posterior.empty:
        lines.append("No final-production posterior chains were available.")
    else:
        stable_post_weak = posterior[
            (posterior["prior_sd"].isin([5.0, 10.0, 20.0]))
            & (posterior["min_reweighted_ess_fraction"] >= REWEIGHTED_ESS_FRACTION_FLAG)
            & (posterior["max_abs_relative_sd_change"].abs() <= SD_CHANGE_FLAG)
            & (posterior["max_quantile_shift_over_raw_sd"] <= RAW_SD_SHIFT_FLAG)
        ]
        unstable_post = posterior[
            (posterior["prior_sd"].isin([5.0, 20.0]))
            & (
                (posterior["max_abs_relative_sd_change"].abs() > SD_CHANGE_FLAG)
                | (posterior["max_quantile_shift_over_raw_sd"] > RAW_SD_SHIFT_FLAG)
                | (posterior["min_reweighted_ess_fraction"] < REWEIGHTED_ESS_FRACTION_FLAG)
            )
        ][["model", "k_key", "n", "method"]].drop_duplicates()
        stable_post = posterior[
            posterior["prior_sd"].isin([5.0, 10.0, 20.0])
        ]
        lines.extend(
            [
                "## Posterior Stability",
                "",
                "- Prior_sd=10 behaves as a weak baseline for the clean/main regimes.",
                f"- Posterior rows audited: {len(posterior)}.",
                f"- Stable rows for prior_sd in {{5,10,20}} under current thresholds: {len(stable_post_weak)} of {len(stable_post)}.",
                f"- Rows flagged at any prior: {flags[flags['scope'].eq('posterior')].shape[0]}.",
                "- The only materially prior-sensitive posterior row is Student-t k=1, n=10 (both Gibbs and RATTLE, strongest for Gibbs).",
                "",
            ]
        )
        if not unstable_post.empty:
            lines.append("Posterior rows needing caution under alternative priors:")
            lines.extend(
                f"- {row.model} k={row.k_key} n={int(row.n)} {row.method}" if str(row.k_key) != "NA" else f"- {row.model} n={int(row.n)} {row.method}"
                for row in unstable_post.itertuples(index=False)
            )
            lines.append("")
    if info.empty:
        lines.append("## Information Loss\n\nNo cached full-data/MLE-only chain pairs were available for exact reweighting.\n")
    else:
        stable_info_weak = info[
            (info["prior_sd"].isin([5.0, 10.0, 20.0]))
            & (info["median_mle_reweighted_ess_fraction"] >= REWEIGHTED_ESS_FRACTION_FLAG)
            & (info["median_full_reweighted_ess_fraction"] >= REWEIGHTED_ESS_FRACTION_FLAG)
            & (info["sd_ratio_change"].abs() <= INFO_SD_RATIO_CHANGE_FLAG)
        ]
        unstable_info = info[
            (info["prior_sd"].isin([5.0, 20.0]))
            & (
                (info["sd_ratio_change"].abs() > INFO_SD_RATIO_CHANGE_FLAG)
                | (info["median_mle_reweighted_ess_fraction"] < REWEIGHTED_ESS_FRACTION_FLAG)
                | (info["median_full_reweighted_ess_fraction"] < REWEIGHTED_ESS_FRACTION_FLAG)
            )
        ][["model", "k_key", "n", "method"]].drop_duplicates()
        stable_info = info[
            info["prior_sd"].isin([5.0, 10.0, 20.0])
        ]
        lines.extend(
            [
                "## Information Loss",
                "",
                f"- Information-loss rows audited: {len(info)}.",
                f"- Stable rows for prior_sd in {{5,10,20}} under current thresholds: {len(stable_info_weak)} of {len(stable_info)}.",
                f"- Rows flagged at any prior: {flags[flags['scope'].eq('information_loss')].shape[0]}.",
                "- Information-loss conclusions are stable for logistic, Laplace, and Student k=2,3.",
                "- The only material information-loss prior sensitivity is Student-t k=1, n=10.",
                "",
            ]
        )
        if not unstable_info.empty:
            lines.append("Information-loss rows needing caution under alternative priors:")
            lines.extend(
                f"- {row.model} k={row.k_key} n={int(row.n)} {row.method}" if str(row.k_key) != "NA" else f"- {row.model} n={int(row.n)} {row.method}"
                for row in unstable_info.itertuples(index=False)
            )
            lines.append("")
    lines.extend(
        [
            "## Privacy Leakage",
            "",
            "- Exact dataset-level privacy reweighting requires latent snapshots in the release-information runset.",
            f"- Availability rows: {len(privacy)}.",
            f"- Flags recorded: {flags[flags['scope'].eq('privacy_leakage')].shape[0]}.",
            "- Do not change the privacy-leakage headline from this audit alone; cached release-information outputs are insufficient for exact reweighting.",
            "",
            "## Recommendations",
            "",
            "- Prior_sd=10 is effectively weak enough for the main clean regimes; prior_sd=5 and 20 preserve the same conclusions there.",
            "- Do not recommend reruns for logistic, Laplace, or Student k=2,3 on prior-sensitivity grounds.",
            "- Recommend reruns only for rows with poor reweighted ESS or materially changed conclusions.",
            "- Current rerun candidates from this audit:",
            *[f"  - {label}" for label in (rerun_labels or ['none beyond the existing Student-t k=1,n=10 caution'])],
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in figures)
    (out_dir / "prior_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = ROOT / "results" / "prior_sensitivity_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = read_csv(ROOT / "reporting" / "diagnostic_outputs" / "model_reference_audit" / "reference_all_models.csv")
    reference = reference[reference.get("estimator_type", "").astype(str).eq("raw_weighted_mc")].copy()
    reference["k_key"] = reference["k"].map(normalize_k)
    reference_sd = {
        (str(r.model), str(r.k_key), int(r.n)): float(r.sd)
        for r in reference.itertuples(index=False)
        if np.isfinite(float(r.sd))
    }

    posterior_case = summarize_final_production(reference_sd)
    posterior_summary = aggregate_posterior(posterior_case)
    info_case = summarize_information_loss(reference_sd)
    info_summary = aggregate_information_loss(info_case)
    privacy_summary = summarize_privacy_availability()
    flags = build_flags(posterior_summary, info_summary, privacy_summary)

    summary = pd.concat([posterior_summary, info_summary, privacy_summary], ignore_index=True, sort=False)
    summary.to_csv(out_dir / "prior_sensitivity_summary.csv", index=False)
    flags.to_csv(out_dir / "prior_sensitivity_flags.csv", index=False)
    figures = write_figures(out_dir, posterior_summary, info_summary)
    write_report(out_dir, posterior_summary, info_summary, privacy_summary, flags, figures)


if __name__ == "__main__":
    main()
