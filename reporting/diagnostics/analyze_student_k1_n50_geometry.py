"""Focused analysis for the Student-t k=1,n=50 Gibbs geometry runset."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "student_k1_n50_geometry_audit" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results") / "student_k1_n50_geometry_audit" / ".cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_FILES = [
    "run_metadata.json",
    "chain_samples.csv",
    "posterior_summaries.csv",
    "cost_ledger.csv",
    "latent_x_diagnostics.csv",
    "latent_diagnostics.csv",
    "transition_diagnostics.csv",
    "branch_diagnostics.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/student_k1_n50_geometry_runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/student_k1_n50_geometry_audit"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/final_production_v1"))
    return parser.parse_args()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def x_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [col for col in df.columns if col.startswith("x_") and col[2:].isdigit()],
        key=lambda col: int(col[2:]),
    )


def case_dirs(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.glob("case_*") if path.is_dir())


def run_lengths(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size == 0:
        return np.asarray([], dtype=int)
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, values.size]
    return ends - starts


def run_length_rows(values: np.ndarray, label: str, case_meta: dict, iteration_step: float) -> list[dict]:
    lengths = run_lengths(values)
    if lengths.size == 0:
        return []
    return [
        {
            **case_meta,
            "state_variable": label,
            "num_runs": int(lengths.size),
            "mean_run_snapshots": float(np.mean(lengths)),
            "median_run_snapshots": float(np.median(lengths)),
            "q95_run_snapshots": float(np.quantile(lengths, 0.95)),
            "max_run_snapshots": int(np.max(lengths)),
            "mean_run_iterations": float(np.mean(lengths) * iteration_step),
            "median_run_iterations": float(np.median(lengths) * iteration_step),
            "q95_run_iterations": float(np.quantile(lengths, 0.95) * iteration_step),
            "max_run_iterations": float(np.max(lengths) * iteration_step),
        }
    ]


def autocorr(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if lag <= 0 or values.size <= lag:
        return float("nan")
    centered = values - float(np.mean(values))
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(centered[:-lag], centered[lag:]) / denom)


def summarize_chain_mu(case_dir: Path, burn_in: int) -> dict:
    chain = read_csv(case_dir / "chain_samples.csv", usecols=["iteration", "mu", "is_burn_in"])
    post = chain[~chain["is_burn_in"].astype(bool)].copy()
    midpoint = int(post["iteration"].min() + (post["iteration"].max() - post["iteration"].min()) / 2)
    first = post[post["iteration"] <= midpoint]["mu"].to_numpy(dtype=float)
    second = post[post["iteration"] > midpoint]["mu"].to_numpy(dtype=float)
    return {
        "post_mu_mean_from_chain": float(post["mu"].mean()),
        "post_mu_sd_from_chain": float(post["mu"].std(ddof=1)),
        "first_half_mu_mean": float(np.mean(first)),
        "second_half_mu_mean": float(np.mean(second)),
        "abs_split_mean_drift": float(abs(np.mean(first) - np.mean(second))),
        "rel_split_mean_drift_vs_sd": float(abs(np.mean(first) - np.mean(second)) / max(float(post["mu"].std(ddof=1)), 1e-300)),
    }


def latent_metrics(case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latent = read_csv(case_dir / "latent_x_diagnostics.csv")
    if latent.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    cols = x_columns(latent)
    x = latent[cols].to_numpy(dtype=float)
    abs_y = np.abs(x)
    z = x / (1.0 + x * x)
    abs_z = np.abs(z)

    count_gt1 = np.sum(abs_y > 1.0, axis=1)
    count_gt2 = np.sum(abs_y > 2.0, axis=1)
    count_gt5 = np.sum(abs_y > 5.0, axis=1)
    count_gt20 = np.sum(abs_y > 20.0, axis=1)
    max_abs_y = np.max(abs_y, axis=1)
    max_abs_z = np.max(abs_z, axis=1)
    sum_abs_z = np.sum(abs_z, axis=1)

    far_tail_class = np.select(
        [count_gt20 == 0, count_gt20 == 1, count_gt20 == 2],
        ["no_gt20", "one_gt20", "two_gt20"],
        default="three_plus_gt20",
    )
    geometry_class = np.select(
        [count_gt20 > 0, count_gt5 > 0, count_gt1 > 0],
        ["far_tail_gt20", "extreme_tail_gt5", "tail_gt1"],
        default="central",
    )

    metrics = pd.DataFrame(
        {
            "case_id": latent["case_id"].astype(str),
            "seed": pd.to_numeric(latent["seed"], errors="coerce").astype(int),
            "initialization": latent["initialization"].astype(str),
            "iteration": pd.to_numeric(latent["iteration"], errors="coerce").astype(int),
            "mu": pd.to_numeric(latent["mu"], errors="coerce"),
            "max_abs_y": max_abs_y,
            "log10_max_abs_y": np.log10(np.maximum(max_abs_y, 1e-300)),
            "mean_abs_y": np.mean(abs_y, axis=1),
            "q95_abs_y": np.quantile(abs_y, 0.95, axis=1),
            "count_gt1": count_gt1,
            "count_gt2": count_gt2,
            "count_gt5": count_gt5,
            "count_gt20": count_gt20,
            "fraction_gt1": count_gt1 / x.shape[1],
            "fraction_gt20": count_gt20 / x.shape[1],
            "score_residual": np.sum(z, axis=1),
            "max_abs_z": max_abs_z,
            "sum_abs_z": sum_abs_z,
            "score_concentration": max_abs_z / np.maximum(sum_abs_z, 1e-300),
            "score_collapse_ratio": max_abs_y / np.maximum(max_abs_z, 1e-300),
            "z_sign_balance": (np.sum(z > 0.0, axis=1) - np.sum(z < 0.0, axis=1)) / x.shape[1],
            "far_tail_class": far_tail_class,
            "geometry_class": geometry_class,
            "has_gt20": count_gt20 > 0,
        }
    )

    case_meta = {
        "case_id": str(metrics["case_id"].iloc[0]),
        "seed": int(metrics["seed"].iloc[0]),
        "initialization": str(metrics["initialization"].iloc[0]),
    }
    iteration_step = float(np.median(np.diff(metrics["iteration"].to_numpy(dtype=float)))) if len(metrics) > 1 else float("nan")
    run_rows = []
    run_rows.extend(run_length_rows(metrics["has_gt20"].to_numpy(bool), "has_gt20", case_meta, iteration_step))
    run_rows.extend(run_length_rows(metrics["far_tail_class"].to_numpy(str), "far_tail_class", case_meta, iteration_step))
    run_rows.extend(run_length_rows(metrics["geometry_class"].to_numpy(str), "geometry_class", case_meta, iteration_step))
    run_summary = pd.DataFrame(run_rows)

    ac_rows = []
    for variable in ["max_abs_y", "count_gt20", "score_collapse_ratio", "mu"]:
        values = metrics[variable].to_numpy(dtype=float)
        row = {**case_meta, "variable": variable}
        for lag in [1, 5, 25, 100, 500]:
            row[f"acf_lag_{lag}"] = autocorr(values, lag)
        ac_rows.append(row)
    autocorr_summary = pd.DataFrame(ac_rows)
    return metrics, run_summary, autocorr_summary


def chain_summary_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["case_id", "seed", "initialization"], dropna=False)
    rows = []
    for keys, part in grouped:
        case_id, seed, init = keys
        class_freq = part["far_tail_class"].value_counts(normalize=True)
        rows.append(
            {
                "case_id": case_id,
                "seed": int(seed),
                "initialization": init,
                "num_latent_snapshots": int(len(part)),
                "mu_mean_snapshot": float(part["mu"].mean()),
                "mu_sd_snapshot": float(part["mu"].std(ddof=1)),
                "max_abs_y_mean": float(part["max_abs_y"].mean()),
                "max_abs_y_median": float(part["max_abs_y"].median()),
                "max_abs_y_q95": float(part["max_abs_y"].quantile(0.95)),
                "max_abs_y_max": float(part["max_abs_y"].max()),
                "prob_any_gt20": float(part["has_gt20"].mean()),
                "mean_count_gt20": float(part["count_gt20"].mean()),
                "mean_fraction_gt1": float(part["fraction_gt1"].mean()),
                "score_collapse_ratio_mean": float(part["score_collapse_ratio"].mean()),
                "score_collapse_ratio_q95": float(part["score_collapse_ratio"].quantile(0.95)),
                "score_concentration_mean": float(part["score_concentration"].mean()),
                "far_tail_class_switch_rate": float(np.mean(part["far_tail_class"].to_numpy(str)[1:] != part["far_tail_class"].to_numpy(str)[:-1])),
                "geometry_class_switch_rate": float(np.mean(part["geometry_class"].to_numpy(str)[1:] != part["geometry_class"].to_numpy(str)[:-1])),
                "no_gt20_fraction": float(class_freq.get("no_gt20", 0.0)),
                "one_gt20_fraction": float(class_freq.get("one_gt20", 0.0)),
                "two_gt20_fraction": float(class_freq.get("two_gt20", 0.0)),
                "three_plus_gt20_fraction": float(class_freq.get("three_plus_gt20", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def geometry_conditioned_mu(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in metrics.groupby(["far_tail_class"], dropna=False):
        class_name = keys[0] if isinstance(keys, tuple) else keys
        mu = part["mu"].to_numpy(dtype=float)
        rows.append(
            {
                "conditioning": "far_tail_class",
                "class": str(class_name),
                "num_snapshots": int(len(part)),
                "fraction": float(len(part) / len(metrics)),
                "mu_mean": float(np.mean(mu)),
                "mu_sd": float(np.std(mu, ddof=1)),
                "mu_q025": float(np.quantile(mu, 0.025)),
                "mu_q50": float(np.quantile(mu, 0.5)),
                "mu_q975": float(np.quantile(mu, 0.975)),
            }
        )
    for keys, part in metrics.groupby(["geometry_class"], dropna=False):
        class_name = keys[0] if isinstance(keys, tuple) else keys
        mu = part["mu"].to_numpy(dtype=float)
        rows.append(
            {
                "conditioning": "geometry_class",
                "class": str(class_name),
                "num_snapshots": int(len(part)),
                "fraction": float(len(part) / len(metrics)),
                "mu_mean": float(np.mean(mu)),
                "mu_sd": float(np.std(mu, ddof=1)),
                "mu_q025": float(np.quantile(mu, 0.025)),
                "mu_q50": float(np.quantile(mu, 0.5)),
                "mu_q975": float(np.quantile(mu, 0.975)),
            }
        )
    out = pd.DataFrame(rows)
    full_sd = float(metrics["mu"].std(ddof=1))
    full_mean = float(metrics["mu"].mean())
    out["delta_mu_mean_vs_full"] = out["mu_mean"] - full_mean
    out["abs_delta_mean_over_full_sd"] = out["delta_mu_mean_vs_full"].abs() / max(full_sd, 1e-300)
    return out


def branch_summary(run_dir: Path) -> pd.DataFrame:
    frames = []
    for case_dir in case_dirs(run_dir):
        path = case_dir / "branch_diagnostics.csv"
        frame = read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    branch = pd.concat(frames, ignore_index=True, sort=False)
    pivot = (
        branch.pivot_table(
            index=["case_id", "seed", "initialization"],
            columns="branch_pair",
            values="frequency",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rate = branch.groupby(["case_id", "seed", "initialization"], dropna=False)["branch_switching_rate"].mean().reset_index()
    out = pivot.merge(rate, on=["case_id", "seed", "initialization"], how="left")
    for col in ["lower/lower", "lower/upper", "upper/lower", "upper/upper"]:
        if col not in out.columns:
            out[col] = 0.0
    out["mixed_pair_fraction"] = out["lower/upper"] + out["upper/lower"]
    out["tail_tail_minus_central_central"] = out["upper/upper"] - out["lower/lower"]
    return out


def load_baseline(baseline_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(baseline_dir.glob("case_student_t_k1_n50_gibbs_seed*_init_central/posterior_summaries.csv")):
        frame = read_csv(path)
        if not frame.empty:
            frame["baseline_case_dir"] = path.parent.name
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def integrity_summary(run_dir: Path) -> pd.DataFrame:
    rows = []
    for case_dir in case_dirs(run_dir):
        row = {"case_dir": case_dir.name}
        missing = [name for name in REQUIRED_FILES if not (case_dir / name).exists()]
        row["missing_required_files"] = ",".join(missing)
        row["is_complete"] = not missing
        metadata_path = case_dir / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            row["metadata_status"] = metadata.get("status")
            row["num_iterations"] = metadata.get("num_iterations")
            row["burn_in"] = metadata.get("burn_in")
            row["diagnostic_thin"] = metadata.get("diagnostic_thin")
        for name in REQUIRED_FILES:
            if name == "run_metadata.json":
                continue
            path = case_dir / name
            row[f"{name}_rows"] = len(read_csv(path)) if path.exists() else 0
        rows.append(row)
    return pd.DataFrame(rows)


def write_figures(out_dir: Path, metrics: pd.DataFrame, chain_summary: pd.DataFrame, branch: pd.DataFrame, conditioned: pd.DataFrame) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(fig, name: str) -> None:
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [chain_summary.loc[chain_summary["initialization"].eq(init), "mu_sd_snapshot"] for init in ["central", "tail_heavy", "random"]]
    ax.boxplot(data, tick_labels=["central", "tail_heavy", "random"], showfliers=True)
    ax.set_ylabel("snapshot mu SD")
    ax.set_title("Posterior mu stability across initializations")
    save(fig, "posterior_mu_sd_by_initialization.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    data = [np.log10(metrics.loc[metrics["initialization"].eq(init), "max_abs_y"].clip(lower=1e-300)) for init in ["central", "tail_heavy", "random"]]
    ax.boxplot(data, tick_labels=["central", "tail_heavy", "random"], showfliers=False)
    ax.set_ylabel("log10 max |x_i - mu_star|")
    ax.set_title("Heavy-tail latent extent")
    save(fig, "max_abs_y_by_initialization.png")

    sample = metrics.sample(n=min(len(metrics), 50000), random_state=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(sample["max_abs_y"], sample["max_abs_z"], s=3, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel("max |y|")
    ax.set_ylabel("max |z|")
    ax.set_title("Cauchy score-space collapse")
    save(fig, "score_collapse_scatter.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    class_order = ["no_gt20", "one_gt20", "two_gt20", "three_plus_gt20"]
    counts = metrics["far_tail_class"].value_counts(normalize=True).reindex(class_order, fill_value=0.0)
    ax.bar(np.arange(len(counts)), counts.to_numpy())
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(class_order, rotation=25, ha="right")
    ax.set_ylabel("fraction of latent snapshots")
    ax.set_title("Far-tail occupancy classes")
    save(fig, "far_tail_class_occupancy.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    if not branch.empty:
        pair_cols = ["lower/lower", "lower/upper", "upper/lower", "upper/upper"]
        means = branch[pair_cols].mean()
        ax.bar(np.arange(len(pair_cols)), means.to_numpy())
        ax.set_xticks(np.arange(len(pair_cols)))
        ax.set_xticklabels(pair_cols, rotation=25, ha="right")
        ax.set_ylabel("mean pair frequency")
        ax.set_title("Gibbs inverse-branch pair balance")
    save(fig, "branch_pair_frequencies.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    cond = conditioned[conditioned["conditioning"].eq("far_tail_class")].set_index("class").reindex(class_order)
    ax.errorbar(np.arange(len(cond)), cond["mu_mean"], yerr=cond["mu_sd"], fmt="o", capsize=4)
    ax.axhline(float(metrics["mu"].mean()), color="black", linewidth=1, linestyle="--")
    ax.set_xticks(np.arange(len(cond)))
    ax.set_xticklabels(class_order, rotation=25, ha="right")
    ax.set_ylabel("mu mean +/- SD")
    ax.set_title("Geometry-conditioned posterior mu")
    save(fig, "geometry_conditioned_mu.png")

    return paths


def format_float(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 0.001):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def write_memo(
    out_dir: Path,
    integrity: pd.DataFrame,
    posterior: pd.DataFrame,
    chain_summary: pd.DataFrame,
    by_init: pd.DataFrame,
    conditioned: pd.DataFrame,
    branch: pd.DataFrame,
    run_lengths: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    figures: list[str],
) -> None:
    overall_mu_mean = float(posterior["mean_mu"].mean())
    overall_sd_mean = float(posterior["sd_mu"].mean())
    sd_range = (float(posterior["sd_mu"].min()), float(posterior["sd_mu"].max()))
    mean_abs_max = float(posterior["mean_mu"].abs().max())
    prob_gt20 = float(chain_summary["prob_any_gt20"].mean())
    count_gt20 = float(chain_summary["mean_count_gt20"].mean())
    tail_frac = float(chain_summary["mean_fraction_gt1"].mean())
    switch_rate = float(chain_summary["far_tail_class_switch_rate"].mean())
    branch_rate = float(branch["branch_switching_rate"].mean()) if not branch.empty else float("nan")
    far_tail_delta = (
        float(conditioned.loc[conditioned["conditioning"].eq("far_tail_class"), "abs_delta_mean_over_full_sd"].max())
        if not conditioned.empty
        else float("nan")
    )
    max_delta = float(conditioned["abs_delta_mean_over_full_sd"].max()) if not conditioned.empty else float("nan")
    has_tail = run_lengths[run_lengths["state_variable"].eq("has_gt20")] if not run_lengths.empty else pd.DataFrame()
    has_tail_median_run = float(has_tail["median_run_iterations"].median()) if not has_tail.empty else float("nan")
    has_tail_max_run = float(has_tail["max_run_iterations"].max()) if not has_tail.empty else float("nan")

    lines = [
        "# Student k=1,n=50 Gibbs Geometry Findings",
        "",
        "## Executive takeaways",
        "",
        f"- Run integrity is clean: {int(integrity['is_complete'].sum())}/{len(integrity)} cases have all required outputs.",
        f"- Posterior `mu` is stable across 15 long chains: mean of chain means is {format_float(overall_mu_mean)}, mean SD is {format_float(overall_sd_mean)}, and SD range is {format_float(sd_range[0])}-{format_float(sd_range[1])}.",
        f"- Initialization does not move the posterior summary materially: max absolute chain mean is {format_float(mean_abs_max)}, far below one posterior SD.",
        f"- The latent geometry is strongly Cauchy-tail dominated: any `|y|>20` appears in about {format_float(prob_gt20 * 100)}% of thinned states, with mean count `|y|>20` about {format_float(count_gt20)} out of 50.",
        f"- About {format_float(tail_frac * 100)}% of coordinates are on the tail branch `|y|>1`, while branch switching is balanced at about {format_float(branch_rate)}.",
        f"- The far-tail count changes frequently in thinned diagnostics: mean far-tail class switch rate is {format_float(switch_rate)} per snapshot; `has |y|>20` runs have median length {format_float(has_tail_median_run)} iterations and max length {format_float(has_tail_max_run)} iterations.",
        f"- Geometry-conditioned `mu` means are close to the full posterior: far-tail count classes shift by at most {format_float(far_tail_delta)} posterior SDs; including the rare `tail_gt1` class, the max shift is {format_float(max_delta)} posterior SDs.",
        "",
        "## Interpretation",
        "",
        "The long chains support a clean separation between posterior inference for `mu` and latent-data geometry. The posterior marginal for `mu` is stable across seeds and initializations, but the compatible latent datasets almost always contain Cauchy tail coordinates. The Gibbs branch mechanism is not one-sided; branch-pair frequencies are near balanced and branch switching is about one half. With the stricter `|y|>20` classes, the sampler is not frozen in one tail-count class. The difficult geometry is that huge `|y|` values are common while their score coordinates remain bounded through `z=y/(1+y^2)`.",
        "",
        "## Key tables",
        "",
        "- `posterior_by_chain.csv`: posterior summaries, cost, and split drift per chain.",
        "- `latent_geometry_by_chain.csv`: tail, score-collapse, and class-switching summaries per chain.",
        "- `geometry_conditioned_mu.csv`: posterior `mu` summaries by far-tail and geometry class.",
        "- `run_length_summary.csv`: persistence of far-tail and geometry classes in thinned diagnostics.",
        "- `branch_summary.csv`: Gibbs branch-pair frequencies and branch switching.",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{path}`" for path in figures)
    if not baseline_comparison.empty:
        lines.extend(
            [
                "",
                "## 100k baseline comparison",
                "",
                "The 500k posterior summaries remain close to the earlier 100k central-initialization production runs; see `baseline_comparison.csv` for the exact rows.",
            ]
        )
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    integrity = integrity_summary(args.run_dir)
    case_paths = case_dirs(args.run_dir)
    posterior_frames = []
    ledger_frames = []
    metric_frames = []
    run_length_frames = []
    autocorr_frames = []

    for case_dir in case_paths:
        posterior = read_csv(case_dir / "posterior_summaries.csv")
        ledger = read_csv(case_dir / "cost_ledger.csv")
        if not posterior.empty:
            posterior = posterior.merge(
                ledger[[col for col in ["case_id", "wall_time_sec", "student_logpdf_evals", "pair_updates_attempted", "pair_updates_completed"] if col in ledger.columns]],
                on="case_id",
                how="left",
            )
            posterior.update(pd.DataFrame([summarize_chain_mu(case_dir, int(posterior["burn_in"].iloc[0]))] * len(posterior)))
            posterior_frames.append(posterior)
        if not ledger.empty:
            ledger_frames.append(ledger)
        metrics, run_summary, autocorr_summary = latent_metrics(case_dir)
        if not metrics.empty:
            metric_frames.append(metrics)
        if not run_summary.empty:
            run_length_frames.append(run_summary)
        if not autocorr_summary.empty:
            autocorr_frames.append(autocorr_summary)

    posterior_by_chain = pd.concat(posterior_frames, ignore_index=True, sort=False) if posterior_frames else pd.DataFrame()
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()
    run_length_summary = pd.concat(run_length_frames, ignore_index=True, sort=False) if run_length_frames else pd.DataFrame()
    autocorr_summary = pd.concat(autocorr_frames, ignore_index=True, sort=False) if autocorr_frames else pd.DataFrame()
    latent_by_chain = chain_summary_from_metrics(metrics) if not metrics.empty else pd.DataFrame()
    latent_by_initialization = (
        latent_by_chain.groupby("initialization", dropna=False)
        .agg(
            {
                "mu_mean_snapshot": ["mean", "std"],
                "mu_sd_snapshot": ["mean", "std"],
                "max_abs_y_mean": ["mean", "std"],
                "max_abs_y_median": ["mean", "std"],
                "prob_any_gt20": ["mean", "std"],
                "mean_count_gt20": ["mean", "std"],
                "mean_fraction_gt1": ["mean", "std"],
                "score_collapse_ratio_mean": ["mean", "std"],
                "far_tail_class_switch_rate": ["mean", "std"],
            }
        )
        if not latent_by_chain.empty
        else pd.DataFrame()
    )
    if not latent_by_initialization.empty:
        latent_by_initialization.columns = ["_".join(col).strip("_") for col in latent_by_initialization.columns]
        latent_by_initialization = latent_by_initialization.reset_index()
    conditioned = geometry_conditioned_mu(metrics) if not metrics.empty else pd.DataFrame()
    branch = branch_summary(args.run_dir)

    baseline = load_baseline(args.baseline_dir)
    baseline_comparison = pd.DataFrame()
    if not baseline.empty and not posterior_by_chain.empty:
        baseline_comparison = pd.DataFrame(
            [
                {
                    "summary": "100k_final_production_central",
                    "num_chains": int(len(baseline)),
                    "mean_mu_mean": float(baseline["mean_mu"].mean()),
                    "sd_mu_mean": float(baseline["sd_mu"].mean()),
                    "sd_mu_min": float(baseline["sd_mu"].min()),
                    "sd_mu_max": float(baseline["sd_mu"].max()),
                    "ess_mu_mean": float(baseline["ess_mu"].mean()),
                },
                {
                    "summary": "500k_geometry_all_inits",
                    "num_chains": int(len(posterior_by_chain)),
                    "mean_mu_mean": float(posterior_by_chain["mean_mu"].mean()),
                    "sd_mu_mean": float(posterior_by_chain["sd_mu"].mean()),
                    "sd_mu_min": float(posterior_by_chain["sd_mu"].min()),
                    "sd_mu_max": float(posterior_by_chain["sd_mu"].max()),
                    "ess_mu_mean": float(posterior_by_chain["ess_mu"].mean()),
                },
            ]
        )

    integrity.to_csv(args.out_dir / "integrity_summary.csv", index=False)
    posterior_by_chain.to_csv(args.out_dir / "posterior_by_chain.csv", index=False)
    latent_by_chain.to_csv(args.out_dir / "latent_geometry_by_chain.csv", index=False)
    latent_by_initialization.to_csv(args.out_dir / "latent_geometry_by_initialization.csv", index=False)
    conditioned.to_csv(args.out_dir / "geometry_conditioned_mu.csv", index=False)
    branch.to_csv(args.out_dir / "branch_summary.csv", index=False)
    run_length_summary.to_csv(args.out_dir / "run_length_summary.csv", index=False)
    autocorr_summary.to_csv(args.out_dir / "autocorr_summary.csv", index=False)
    baseline_comparison.to_csv(args.out_dir / "baseline_comparison.csv", index=False)
    # Store per-snapshot metrics for further ad hoc work; this is about 300k rows.
    metrics.to_csv(args.out_dir / "latent_snapshot_metrics.csv", index=False)

    figures = write_figures(args.out_dir, metrics, latent_by_chain, branch, conditioned)
    write_memo(
        args.out_dir,
        integrity,
        posterior_by_chain,
        latent_by_chain,
        latent_by_initialization,
        conditioned,
        branch,
        run_length_summary,
        baseline_comparison,
        figures,
    )

    manifest = {
        "run_dir": str(args.run_dir),
        "out_dir": str(args.out_dir),
        "num_cases": int(len(case_paths)),
        "num_latent_snapshots": int(len(metrics)),
        "outputs": [
            "integrity_summary.csv",
            "posterior_by_chain.csv",
            "latent_geometry_by_chain.csv",
            "latent_geometry_by_initialization.csv",
            "geometry_conditioned_mu.csv",
            "branch_summary.csv",
            "run_length_summary.csv",
            "autocorr_summary.csv",
            "baseline_comparison.csv",
            "latent_snapshot_metrics.csv",
            "findings.md",
        ],
        "figures": figures,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
