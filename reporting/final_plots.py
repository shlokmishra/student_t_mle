"""Plot final comparison figures from locked summary data."""

from __future__ import annotations

import math
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reporting.final_results import LOCKED_RESULTS, write_summary_outputs


STATUS_COLORS = {
    "defensible": "#2a9d8f",
    "not defensible": "#e76f51",
    "not attempted": "#9aa0a6",
}

METHOD_COLORS = {
    "gibbs": "#264653",
    "kde": "#f4a261",
    "rattle": "#2a9d8f",
    "full_data_mh": "#6d597a",
}

KDE_AUDIT_PATH = Path("artifacts/final_comparison/kde_bandwidth_audit.json")


def _normal_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    std = max(math.sqrt(var), 1e-8)
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def _find_model_entry(model_key: str) -> dict:
    for entry in LOCKED_RESULTS:
        if entry["model"] == model_key:
            return entry
    raise KeyError(model_key)


def _load_kde_audit_payload() -> dict | None:
    if not KDE_AUDIT_PATH.exists():
        return None
    with KDE_AUDIT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_posterior_overlays(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    kde_audit = _load_kde_audit_payload()
    plot_payload = {}
    if kde_audit:
        plot_payload = kde_audit.get("plot_payload", {})

    selected = [
        ("loc_logistic", "Logistic", 20),
        ("loc_student_k3", "Student-3", 20),
        ("loc_student_k2", "Student-2", 20),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for ax, (model_key, title, n_value) in zip(axes, selected):
        if model_key in plot_payload:
            curves = plot_payload[model_key]
            grid = np.asarray(curves["grid"], dtype=float)
            ax.plot(grid, np.asarray(curves["gibbs_density"], dtype=float), color=METHOD_COLORS["gibbs"], linewidth=2.0, label="gibbs")
            ax.plot(grid, np.asarray(curves["kde_density"], dtype=float), color=METHOD_COLORS["kde"], linewidth=2.0, label=f"kde ({curves['kde_bw_method']})")
            ax.plot(grid, np.asarray(curves["rattle_density"], dtype=float), color=METHOD_COLORS["rattle"], linewidth=2.0, label="rattle")
            ax.plot(grid, np.asarray(curves["full_data_density"], dtype=float), color=METHOD_COLORS["full_data_mh"], linewidth=2.0, label="full data mh")
            ax.set_title(f"{title} (n={n_value})")
        else:
            entry = _find_model_entry(model_key)
            run = next(run for run in entry["runs"] if run["n"] == n_value)
            means = [
                metrics["posterior_mean"]
                for metrics in run["methods"].values()
                if "posterior_mean" in metrics and "posterior_var" in metrics
            ]
            vars_ = [
                metrics["posterior_var"]
                for metrics in run["methods"].values()
                if "posterior_mean" in metrics and "posterior_var" in metrics
            ]
            lo = min(means) - 4.0 * math.sqrt(max(vars_))
            hi = max(means) + 4.0 * math.sqrt(max(vars_))
            grid = np.linspace(lo, hi, 500)

            for method in ["gibbs", "kde", "rattle", "full_data_mh"]:
                metrics = run["methods"].get(method)
                if not metrics or "posterior_mean" not in metrics or "posterior_var" not in metrics:
                    continue
                ax.plot(
                    grid,
                    _normal_pdf(grid, metrics["posterior_mean"], metrics["posterior_var"]),
                    label=method.replace("_", " "),
                    color=METHOD_COLORS[method],
                    linewidth=2.0,
                )
            ax.set_title(f"{title} (n={n_value})")
        ax.set_xlabel("mu")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Normal approximation density")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Posterior overlay using audited KDE settings", fontsize=13)
    fig.tight_layout()

    path = out_path / "posterior_overlay.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ess_per_second(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    labels = []
    gibbs_vals = []
    rattle_vals = []
    full_vals = []
    for entry in LOCKED_RESULTS:
        if not entry["runs"]:
            continue
        labels.append(entry["label"])
        gibbs_vals.append(
            np.mean(
                [
                    run["methods"].get("gibbs", {}).get("ess_per_sec", np.nan)
                    for run in entry["runs"]
                ]
            )
        )
        rattle_vals.append(
            np.mean(
                [
                    run["methods"].get("rattle", {}).get("ess_per_sec", np.nan)
                    for run in entry["runs"]
                ]
            )
        )
        full_vals.append(
            np.mean(
                [
                    run["methods"].get("full_data_mh", {}).get("ess_per_sec", np.nan)
                    for run in entry["runs"]
                ]
            )
        )

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, gibbs_vals, width=width, color=METHOD_COLORS["gibbs"], label="Gibbs")
    ax.bar(x, rattle_vals, width=width, color=METHOD_COLORS["rattle"], label="RATTLE")
    ax.bar(x + width, full_vals, width=width, color=METHOD_COLORS["full_data_mh"], label="Full-data MH")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ESS/sec")
    ax.set_title("Runtime-adjusted efficiency")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = out_path / "ess_per_sec_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_baseline_fragility(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    kde_audit = _load_kde_audit_payload()
    audit_lookup = {}
    if kde_audit:
        audit_lookup = {entry["model"]: entry for entry in kde_audit.get("model_audits", [])}

    labels = [entry["label"] for entry in LOCKED_RESULTS]
    status_colors = [STATUS_COLORS[entry["baseline_status"]] for entry in LOCKED_RESULTS]
    reverse_vals = []
    var_gap_vals = []
    for entry in LOCKED_RESULTS:
        representative = None
        for run in entry["runs"]:
            if run["n"] == 20:
                representative = run
                break
        if representative is None and entry["runs"]:
            representative = entry["runs"][0]

        if representative and "rattle" in representative["methods"] and "kde" in representative["methods"]:
            rattle = representative["methods"]["rattle"]
            audit_entry = audit_lookup.get(entry["model"])
            if audit_entry:
                kde_var = audit_entry["recommended"]["posterior_var"]
            else:
                kde_var = representative["methods"]["kde"].get("posterior_var")
            reverse_vals.append(100.0 * rattle.get("reverse_fail_rate", np.nan))
            rattle_var = rattle.get("posterior_var")
            var_gap_vals.append(100.0 * abs(rattle_var - kde_var) / kde_var)
        else:
            reverse_vals.append(np.nan)
            var_gap_vals.append(np.nan)

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.bar(x, np.ones(len(labels)), color=status_colors)
    for idx, entry in enumerate(LOCKED_RESULTS):
        ax1.text(
            idx,
            0.5,
            entry["baseline_status"],
            ha="center",
            va="center",
            color="white" if entry["baseline_status"] != "not attempted" else "black",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_yticks([])
    ax1.set_title("Baseline status and fragility at representative n=20")

    width = 0.35
    ax2.bar(x - width / 2, reverse_vals, width=width, color="#457b9d", label="Reverse fail %")
    ax2.bar(x + width / 2, var_gap_vals, width=width, color="#e9c46a", label="Variance gap vs KDE %")
    ax2.set_ylabel("Percent")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis="y", alpha=0.2)
    ax2.legend(frameon=False)
    fig.tight_layout()

    path = out_path / "baseline_fragility.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_figures(out_dir: str | Path = "artifacts/final_comparison") -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    write_summary_outputs(out_path)
    return {
        "posterior_overlay": plot_posterior_overlays(out_path),
        "ess_per_sec": plot_ess_per_second(out_path),
        "baseline_fragility": plot_baseline_fragility(out_path),
    }


if __name__ == "__main__":
    paths = generate_all_figures()
    for key, value in paths.items():
        print(f"{key}: {value}")
