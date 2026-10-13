"""Prepare extra meeting figures from cached final-production audits.

This script does not run samplers and does not recompute reference/KDE data.
It reads cached final-production efficiency and geometry CSVs and writes small
presentation-facing figures plus source summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "meeting_pack" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "results" / "meeting_pack" / ".cache"))

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--efficiency-dir", type=Path, default=Path("results/final_production_v1_efficiency_audit_cost_first"))
    parser.add_argument("--geometry-dir", type=Path, default=Path("results/final_production_v1_geometry_audit"))
    parser.add_argument("--release-info-dir", type=Path, default=Path("results/final_production_v1_release_information_audit_100"))
    parser.add_argument("--previous-release-info-dir", type=Path, default=Path("results/final_production_v1_release_information_audit"))
    parser.add_argument("--meeting-dir", type=Path, default=Path("results/meeting_pack"))
    parser.add_argument("--image-dir", type=Path, default=Path("docs/presentation/Images/final"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def k_label(k_key: object) -> str:
    if pd.isna(k_key) or str(k_key) == "nan":
        return ""
    value = str(k_key).replace(".0", "")
    return f" k={value}"


def regime_label(row: pd.Series) -> str:
    model = str(row["model"])
    if model == "student_t":
        return f"Student{k_label(row.get('k_key'))}, n={int(row['n'])}"
    if model == "logistic":
        return f"Logistic, n={int(row['n'])}"
    if model == "laplace":
        return f"Laplace, n={int(row['n'])}"
    return f"{model}, n={int(row['n'])}"


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def median_abs_deviation(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def smooth_density(values: np.ndarray, xmin: float | None = None, xmax: float | None = None, bins: int = 160, bandwidth_bins: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    lo = float(arr.min()) if xmin is None else float(xmin)
    hi = float(arr.max()) if xmax is None else float(xmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([]), np.array([])
    hist, edges = np.histogram(arr, bins=bins, range=(lo, hi), density=True)
    radius = max(1, int(np.ceil(3 * bandwidth_bins)))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / bandwidth_bins) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(hist, kernel, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, smoothed


def selected_information_loss_mad_summary() -> pd.DataFrame:
    selected_specs = [
        ("normal_known_var", "nan", 20, "Normal\nknown var", "#6B8793"),
        ("logistic", "nan", 20, "Logistic\nn=20", "#2E7D32"),
        ("student_t", "2", 20, "Student\nk=2, n=20", "#43A047"),
        ("student_t", "3", 20, "Student\nk=3, n=20", "#66BB6A"),
        ("student_t", "1", 10, "Student\nk=1, n=10", "#C62828"),
    ]
    keep = {(model, kk, int(n)): (label, color) for model, kk, n, label, color in selected_specs}
    rows: list[dict[str, object]] = []
    release_root = ROOT / "results" / "release_information_runs"
    for case_dir in sorted(release_root.glob("case_*")):
        mle_path = case_dir / "mle_only_chain_samples.csv"
        full_path = case_dir / "full_data_chain_samples.csv"
        if not mle_path.exists() or not full_path.exists():
            continue
        mle = read_csv(mle_path)
        full = read_csv(full_path)
        if mle.empty or full.empty:
            continue
        model = str(mle["model"].iloc[0])
        k_value = pd.to_numeric(mle["k"], errors="coerce").iloc[0] if "k" in mle.columns else np.nan
        k_key = "nan" if pd.isna(k_value) else str(int(float(k_value))) if float(k_value).is_integer() else f"{float(k_value):g}"
        n = int(pd.to_numeric(mle["n"], errors="coerce").iloc[0])
        key = (model, k_key, n)
        if key not in keep:
            continue
        if "is_burn_in" in mle.columns:
            mle = mle[~mle["is_burn_in"].fillna(False).astype(bool)]
        if "is_burn_in" in full.columns:
            full = full[~full["is_burn_in"].fillna(False).astype(bool)]
        mu_mle = pd.to_numeric(mle["mu"], errors="coerce").dropna().to_numpy(dtype=float)
        mu_full = pd.to_numeric(full["mu"], errors="coerce").dropna().to_numpy(dtype=float)
        if mu_mle.size == 0 or mu_full.size == 0:
            continue
        mad_mle = median_abs_deviation(mu_mle)
        mad_full = median_abs_deviation(mu_full)
        rows.append(
            {
                "model": model,
                "k_key": k_key,
                "n": n,
                "label": keep[key][0].replace("\n", " "),
                "color": keep[key][1],
                "dataset_id": str(mle["dataset_id"].iloc[0]) if "dataset_id" in mle.columns else case_dir.name,
                "mad_mle_only": mad_mle,
                "mad_full_data": mad_full,
                "mad_ratio_mle_over_full": mad_mle / mad_full if np.isfinite(mad_full) and mad_full > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def efficiency_outputs(efficiency_dir: Path, meeting_dir: Path, image_dir: Path) -> list[str]:
    summary = read_csv(efficiency_dir / "efficiency_summary.csv")
    if summary.empty:
        return []
    summary = summary.copy()
    summary["display_regime"] = summary.apply(regime_label, axis=1)
    summary["method_label"] = summary["method"].str.upper()

    comparable = summary[
        (
            (summary["model"].eq("logistic"))
            | (summary["model"].eq("student_t") & summary["k_key"].astype(str).isin(["2", "2.0", "3", "3.0"]))
        )
        & summary["method"].isin(["gibbs", "rattle"])
    ].copy()
    comparable["family"] = comparable.apply(
        lambda r: "Logistic" if r["model"] == "logistic" else f"Student k={str(r['k_key']).replace('.0', '')}",
        axis=1,
    )
    comparable["n"] = pd.to_numeric(comparable["n"], errors="coerce")
    comparable["sec_per_iteration_median"] = pd.to_numeric(comparable["sec_per_iteration_median"], errors="coerce")
    comparable["ess_mu_per_sec_median"] = pd.to_numeric(comparable["ess_mu_per_sec_median"], errors="coerce")
    comparable["pair_updates_completed_per_iter"] = pd.to_numeric(comparable["pair_updates_completed_per_iter"], errors="coerce")
    comparable["projection_evals_per_iter"] = pd.to_numeric(comparable["projection_evals_per_iter"], errors="coerce")
    comparable["forward_newton_iters_per_iter"] = pd.to_numeric(comparable["forward_newton_iters_per_iter"], errors="coerce")

    out_cols = [
        "model",
        "k_key",
        "n",
        "method",
        "comparison_regime",
        "safe_to_present",
        "sec_per_iteration_median",
        "ess_mu_per_sec_median",
        "wall_time_per_ess_mu_median",
        "pair_updates_completed_per_iter",
        "pair_grid_evals_per_iter",
        "projection_evals_per_iter",
        "gram_evals_per_iter",
        "forward_newton_iters_per_iter",
        "reverse_newton_iters_per_iter",
    ]
    cost_summary = comparable[[c for c in out_cols if c in comparable.columns]].sort_values(["model", "k_key", "n", "method"])
    cost_summary.to_csv(meeting_dir / "cost_decomposition_summary.csv", index=False)

    paths: list[str] = []
    families = ["Logistic", "Student k=2", "Student k=3"]
    colors = {"gibbs": "#1f77b4", "rattle": "#d62728"}
    markers = {"gibbs": "o", "rattle": "s"}

    for y_col, ylabel, title, filename, yscale in [
        ("sec_per_iteration_median", "seconds / iteration", "Raw cost per iteration", "sec_per_iteration_by_n.png", "linear"),
        ("ess_mu_per_sec_median", "ESS(mu) / second", "Posterior information per second", "ess_per_sec_by_n.png", "log"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=False)
        for ax, family in zip(axes, families, strict=True):
            part = comparable[comparable["family"].eq(family)]
            for method, group in part.groupby("method"):
                group = group.sort_values("n")
                ax.plot(group["n"], group[y_col], marker=markers.get(method, "o"), color=colors.get(method), label=method.upper(), linewidth=2)
            ax.set_title(family)
            ax.set_xlabel("n")
            ax.grid(alpha=0.25)
            if yscale == "log":
                ax.set_yscale("log")
            if not part.empty:
                ax.set_xticks(sorted(part["n"].dropna().unique()))
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(frameon=False, loc="best")
        fig.suptitle(title)
        out = image_dir / filename
        save(fig, out)
        paths.append(str(out))

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.5), sharex=False)
    panel_specs = [
        ("sec_per_iteration_median", "seconds / iteration", "Raw cost", "linear"),
        ("ess_mu_per_sec_median", "ESS(mu) / second", "Information rate", "log"),
    ]
    for row_idx, (y_col, ylabel, row_title, yscale) in enumerate(panel_specs):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx, col_idx]
            part = comparable[comparable["family"].eq(family)]
            for method, group in part.groupby("method"):
                group = group.sort_values("n")
                ax.plot(group["n"], group[y_col], marker=markers.get(method, "o"), color=colors.get(method), label=method.upper(), linewidth=2)
            if row_idx == 0:
                ax.set_title(family)
            ax.set_xlabel("n")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if yscale == "log":
                ax.set_yscale("log")
            if not part.empty:
                ax.set_xticks(sorted(part["n"].dropna().unique()))
            ax.grid(alpha=0.25)
            if col_idx == 2:
                ax.text(1.03, 0.5, row_title, transform=ax.transAxes, rotation=90, va="center", fontsize=10)
    axes[0, -1].legend(frameon=False, loc="best")
    fig.suptitle("Cost decomposition: raw iteration cost and ESS/sec")
    out = image_dir / "efficiency_cost_decomposition_combined.png"
    save(fig, out)
    paths.append(str(out))

    return paths


def geometry_outputs(geometry_dir: Path, meeting_dir: Path, image_dir: Path) -> list[str]:
    latent = read_csv(geometry_dir / "latent_tail_geometry.csv")
    conditioned = read_csv(geometry_dir / "geometry_conditioned_posterior.csv")
    if latent.empty:
        return []

    latent = latent.copy()
    latent["k_key"] = latent["k_key"].astype(str).str.replace(".0", "", regex=False)
    reps = latent[
        latent["model"].eq("student_t")
        & (
            (latent["k_key"].eq("3") & latent["n"].eq(20))
            | (latent["k_key"].eq("1") & latent["n"].eq(10))
        )
        & latent["method"].isin(["gibbs", "rattle"])
    ].copy()
    if reps.empty:
        return []
    reps["regime"] = "Student k=" + reps["k_key"].astype(str) + ", n=" + reps["n"].astype(int).astype(str)
    reps["method_label"] = reps["method"].str.upper()

    occ = (
        reps.groupby(["regime", "method", "latent_geometry_class"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = occ.groupby(["regime", "method"], dropna=False)["count"].transform("sum")
    occ["fraction"] = occ["count"] / totals
    occ.to_csv(meeting_dir / "representative_geometry_class_occupancy.csv", index=False)

    class_order = ["central", "mixed_tail", "extreme_tail", "tail_dominated"]
    colors = {
        "central": "#4C78A8",
        "mixed_tail": "#72B7B2",
        "extreme_tail": "#F58518",
        "tail_dominated": "#B279A2",
    }
    labels = []
    plot_rows = []
    for regime in ["Student k=3, n=20", "Student k=1, n=10"]:
        for method in ["gibbs", "rattle"]:
            labels.append(f"{regime}\n{method.upper()}")
            plot_rows.append((regime, method))

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    bottoms = np.zeros(len(plot_rows))
    x = np.arange(len(plot_rows))
    for cls in class_order:
        vals = []
        for regime, method in plot_rows:
            val = occ[
                occ["regime"].eq(regime)
                & occ["method"].eq(method)
                & occ["latent_geometry_class"].astype(str).eq(cls)
            ]["fraction"]
            vals.append(float(val.iloc[0]) if not val.empty else 0.0)
        ax.bar(x, vals, bottom=bottoms, label=cls, color=colors.get(cls))
        bottoms += np.asarray(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of thinned latent states")
    ax.set_ylim(0, 1)
    ax.set_title("Representative latent geometry classes")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    out1 = image_dir / "representative_geometry_classes.png"
    save(fig, out1)

    paths = [str(out1)]

    max_abs = reps[["regime", "method", "seed", "initialization", "iteration", "max_abs_y"]].copy()
    max_abs["log10_max_abs_y"] = np.log10(max_abs["max_abs_y"].clip(lower=1e-8))
    density_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    regime_order = ["Student k=3, n=20", "Student k=1, n=10"]
    method_colors = {"gibbs": "#1f77b4", "rattle": "#d62728"}
    x_lo = float(max_abs["log10_max_abs_y"].min())
    x_hi = float(max_abs["log10_max_abs_y"].max())
    for ax, regime in zip(axes, regime_order, strict=True):
        subset = max_abs[max_abs["regime"].eq(regime)].copy()
        for method in ["gibbs", "rattle"]:
            vals = subset[subset["method"].eq(method)]["log10_max_abs_y"].to_numpy(dtype=float)
            centers, density = smooth_density(vals, xmin=x_lo, xmax=x_hi, bins=180, bandwidth_bins=2.2)
            if centers.size == 0:
                continue
            ax.plot(centers, density, linewidth=2.2, color=method_colors.get(method), label=method.upper())
            ax.fill_between(centers, 0, density, color=method_colors.get(method), alpha=0.14)
            density_rows.extend(
                {
                    "regime": regime,
                    "method": method,
                    "log10_max_abs_y": float(xv),
                    "density": float(yv),
                }
                for xv, yv in zip(centers, density, strict=True)
            )
        ax.set_title(regime)
        ax.set_xlabel(r"$\log_{10}(\max_i |x_i-\mu^\star|)$")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("smoothed density over thinned latent states")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle(r"Representative Student regimes: latent extreme size via $\max_i |x_i-\mu^\star|$")
    out_max = image_dir / "representative_max_abs_y_density.png"
    save(fig, out_max)
    pd.DataFrame(density_rows).to_csv(meeting_dir / "representative_max_abs_y_density.csv", index=False)
    paths.append(str(out_max))

    if not conditioned.empty:
        cond = conditioned.copy()
        cond["k_key"] = cond["k_key"].astype(str).str.replace(".0", "", regex=False)
        cond = cond[
            cond["model"].eq("student_t")
            & (
                (cond["k_key"].eq("3") & cond["n"].eq(20))
                | (cond["k_key"].eq("1") & cond["n"].eq(10))
            )
            & cond["method"].isin(["gibbs", "rattle"])
        ].copy()
        if not cond.empty:
            cond["regime"] = "Student k=" + cond["k_key"].astype(str) + ", n=" + cond["n"].astype(int).astype(str)
            cond.to_csv(meeting_dir / "representative_geometry_conditioned_posterior.csv", index=False)
            fig, ax = plt.subplots(figsize=(9.2, 4.2))
            cond = cond.sort_values(["regime", "method", "latent_geometry_class"])
            x = np.arange(len(cond))
            ax.errorbar(
                x,
                cond["mu_mean"],
                yerr=1.96 * cond["mu_sd"].fillna(0) / np.sqrt(cond["num_samples"].clip(lower=1)),
                fmt="o",
                color="#1F4E79",
                ecolor="#888888",
                capsize=2,
            )
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    f"{r.regime}\n{str(r.method).upper()} {r.latent_geometry_class}\nfrac={r.fraction_of_chain:.2f}"
                    for r in cond.itertuples()
                ],
                rotation=70,
                ha="right",
                fontsize=7,
            )
            ax.set_ylabel("geometry-conditioned mean(mu)")
            ax.set_title("Posterior mean by latent geometry class")
            out2 = image_dir / "representative_geometry_conditioned_mu.png"
            save(fig, out2)
            paths.append(str(out2))
    return paths


def release_information_outputs(release_info_dir: Path, previous_release_info_dir: Path, meeting_dir: Path, image_dir: Path) -> list[str]:
    info = read_csv(release_info_dir / "information_loss_summary.csv")
    if info.empty:
        return []

    info = info.copy()
    info["k_key"] = info["k_key"].astype(str).str.replace(".0", "", regex=False)
    selected = selected_information_loss_mad_summary()
    if not selected.empty:
        wasserstein = info[["model", "k_key", "n", "wasserstein_mu_median"]].copy()
        wasserstein["k_key"] = wasserstein["k_key"].astype(str).str.replace(".0", "", regex=False)
        selected = selected.merge(wasserstein, on=["model", "k_key", "n"], how="left")
        selected = (
            selected.groupby(["model", "k_key", "n", "label", "color"], dropna=False)
            .agg(
                mad_ratio_mle_over_full_median=("mad_ratio_mle_over_full", "median"),
                mad_ratio_mle_over_full_mad=("mad_ratio_mle_over_full", lambda s: float(np.median(np.abs(pd.to_numeric(s, errors="coerce") - np.median(pd.to_numeric(s, errors="coerce")))))),
                wasserstein_mu_median=("wasserstein_mu_median", "median"),
                dataset_count=("dataset_id", "nunique"),
            )
            .reset_index()
        )
    selected.to_csv(meeting_dir / "information_loss_selected_summary_100.csv", index=False)

    same_release = False
    if previous_release_info_dir.exists() and release_info_dir.exists():
        try:
            same_release = previous_release_info_dir.samefile(release_info_dir)
        except OSError:
            same_release = previous_release_info_dir == release_info_dir
    else:
        same_release = previous_release_info_dir == release_info_dir
    if not same_release:
        prev = read_csv(previous_release_info_dir / "information_loss_summary.csv")
        if not prev.empty:
            prev = prev.copy()
            prev["k_key"] = prev["k_key"].astype(str).str.replace(".0", "", regex=False)
            keys = ["model", "k_key", "n"]
            cols = ["sd_ratio_mle_over_full_median", "wasserstein_mu_median"]
            comp = prev[keys + cols].merge(info[keys + cols], on=keys, how="inner", suffixes=("_old30", "_new100"))
            for col in cols:
                comp[f"{col}_delta"] = comp[f"{col}_new100"] - comp[f"{col}_old30"]
            comp.to_csv(meeting_dir / "information_loss_30_vs_100_comparison.csv", index=False)

    if selected.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), gridspec_kw={"width_ratios": [1.05, 0.95]})
    x = np.arange(len(selected))
    tick_labels = [
        r["label"].replace(" ", "\n", 1) if r["label"].startswith("Normal") else r["label"].replace(", ", ",\n")
        for _, r in selected.iterrows()
    ]

    ax = axes[0]
    bars = ax.bar(x, selected["mad_ratio_mle_over_full_median"], color=selected["color"])
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("median MAD ratio\nMLE-only / full-data")
    ax.set_ylim(0.8, max(1.08, float(selected["mad_ratio_mle_over_full_median"].max()) + 0.05))
    ax.set_title("Robust central spread")
    for bar, row in zip(bars, selected.itertuples(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{row.mad_ratio_mle_over_full_median:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1]
    bars = ax.bar(x, selected["wasserstein_mu_median"], color=selected["color"])
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("median Wasserstein distance")
    ax.set_ylim(0, max(0.13, float(selected["wasserstein_mu_median"].max()) + 0.02))
    ax.set_title("Shape and tail difference")
    for bar, row in zip(bars, selected.itertuples(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{row.wasserstein_mu_median:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("Information loss: robust spread stays close while shape differences grow in hard regimes")
    fig.text(
        0.01,
        0.01,
        "MAD = median absolute deviation of posterior draws. Wasserstein distance compares the full posterior shape. Values summarize cached simulated datasets per regime.",
        fontsize=9,
        color="#555555",
    )
    out = image_dir / "information_loss_selected_bars.png"
    save(fig, out)
    return [str(out)]


def main() -> None:
    args = parse_args()
    args.meeting_dir.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    figures.extend(efficiency_outputs(args.efficiency_dir, args.meeting_dir, args.image_dir))
    figures.extend(geometry_outputs(args.geometry_dir, args.meeting_dir, args.image_dir))
    figures.extend(release_information_outputs(args.release_info_dir, args.previous_release_info_dir, args.meeting_dir, args.image_dir))
    manifest = {
        "figures": figures,
        "tables": [
            str(args.meeting_dir / "cost_decomposition_summary.csv"),
            str(args.meeting_dir / "representative_geometry_class_occupancy.csv"),
            str(args.meeting_dir / "representative_max_abs_y_density.csv"),
            str(args.meeting_dir / "representative_geometry_conditioned_posterior.csv"),
            str(args.meeting_dir / "information_loss_selected_summary_100.csv"),
            str(args.meeting_dir / "information_loss_30_vs_100_comparison.csv"),
        ],
    }
    (args.meeting_dir / "evidence_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
