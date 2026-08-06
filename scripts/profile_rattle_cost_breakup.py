"""Profile Student-t RATTLE cost by algorithm block across n.

This is a local diagnostic profiler.  It does not modify sampler behavior and
does not write to canonical production outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import jax.random as random
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.cost_ledger import CostLedger
from models import loc_student_rattle


@dataclass
class ExclusiveProfiler:
    """Collect exclusive wall time for nested wrapped calls."""

    totals: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    stack: list[dict[str, Any]] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)

    @contextmanager
    def phase(self, name: str):
        self.phases.append(name)
        try:
            yield
        finally:
            self.phases.pop()

    @contextmanager
    def section(self, name: str):
        section_name = f"{self.phases[-1]}:{name}" if self.phases else name
        frame = {"name": section_name, "child_time": 0.0, "start": time.perf_counter()}
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - float(frame["start"])
            self.stack.pop()
            exclusive = max(elapsed - float(frame["child_time"]), 0.0)
            self.totals[section_name] = self.totals.get(section_name, 0.0) + exclusive
            self.calls[section_name] = self.calls.get(section_name, 0) + 1
            if self.stack:
                self.stack[-1]["child_time"] += elapsed


class MonkeyPatch:
    def __init__(self) -> None:
        self._originals: list[tuple[Any, str, Any]] = []

    def set(self, obj: Any, name: str, value: Any) -> None:
        self._originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self) -> None:
        for obj, name, original in reversed(self._originals):
            setattr(obj, name, original)
        self._originals.clear()


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def effective_sample_size(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n <= 1:
        return float(n)
    centered = values - np.mean(values)
    var = float(np.dot(centered, centered) / n)
    if var <= 0.0 or not np.isfinite(var):
        return float(n)
    autocorr_sum = 0.0
    for lag in range(1, n):
        acov = float(np.dot(centered[:-lag], centered[lag:]) / n)
        rho = acov / var
        if rho <= 0.0:
            break
        autocorr_sum += rho
    return float(max(n / (1.0 + 2.0 * autocorr_sum), 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=_floats, default=[2.0])
    parser.add_argument("--n-values", type=_ints, default=[10, 20, 50, 100, 200, 500, 1000, 2000])
    parser.add_argument("--seeds", type=_ints, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("results/rattle_cost_profile_v1"))
    parser.add_argument("--proposal-std-mu", type=float, default=0.3)
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-std", type=float, default=10.0)
    parser.add_argument("--rattle-step-size", type=float, default=0.05)
    parser.add_argument("--rattle-num-steps", type=int, default=2)
    parser.add_argument("--rattle-proj-tol", type=float, default=1e-10)
    parser.add_argument("--rattle-proj-max-iters", type=int, default=25)
    parser.add_argument("--rattle-grad-tol", type=float, default=1e-12)
    parser.add_argument("--rattle-tangent-tol", type=float, default=1e-8)
    parser.add_argument("--rattle-reverse-position-tol", type=float, default=5e-3)
    parser.add_argument("--rattle-reverse-momentum-tol", type=float, default=5e-3)
    parser.add_argument("--rattle-proj-damping", type=float, default=1.0)
    parser.add_argument("--rattle-proj-line-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rattle-proj-init-strategy", default="trial")
    parser.add_argument(
        "--rattle-projection-mode",
        choices=["paper_fixed_direction", "normal_newton_legacy"],
        default="paper_fixed_direction",
    )
    parser.add_argument("--rattle-include-gram-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reverse-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initialization", choices=["central", "tail_heavy", "random"], default="central")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def base_params(args: argparse.Namespace, k: float, n: int, seed: int, iterations: int | None = None) -> dict[str, Any]:
    return {
        "n": int(n),
        "k": float(k),
        "num_iterations_T": int(args.iterations if iterations is None else iterations),
        "proposal_std_mu": float(args.proposal_std_mu),
        "prior_mean": float(args.prior_mean),
        "prior_std": float(args.prior_std),
        "rattle_step_size": float(args.rattle_step_size),
        "rattle_num_steps": int(args.rattle_num_steps),
        "rattle_proj_tol": float(args.rattle_proj_tol),
        "rattle_proj_max_iters": int(args.rattle_proj_max_iters),
        "rattle_grad_tol": float(args.rattle_grad_tol),
        "rattle_tangent_tol": float(args.rattle_tangent_tol),
        "rattle_reverse_position_tol": float(args.rattle_reverse_position_tol),
        "rattle_reverse_momentum_tol": float(args.rattle_reverse_momentum_tol),
        "rattle_proj_damping": float(args.rattle_proj_damping),
        "rattle_proj_line_search": bool(args.rattle_proj_line_search),
        "rattle_proj_init_strategy": str(args.rattle_proj_init_strategy),
        "rattle_projection_mode": str(args.rattle_projection_mode),
        "rattle_include_gram_correction": bool(args.rattle_include_gram_correction),
        "reverse_check": bool(args.reverse_check),
        "initialization": str(args.initialization),
        "initialization_seed": int(seed),
    }


def _wrap(profiler: ExclusiveProfiler, name: str, fn: Callable) -> Callable:
    def wrapped(*args, **kwargs):
        with profiler.section(name):
            return fn(*args, **kwargs)

    return wrapped


def _trajectory_wrapper(profiler: ExclusiveProfiler, fn: Callable) -> Callable:
    def wrapped(*args, **kwargs):
        kind = str(kwargs.get("trajectory_kind", "forward"))
        if len(args) >= 17:
            kind = str(args[16])
        phase = "reverse check" if kind == "reverse" else "forward proposal"
        with profiler.phase(phase):
            with profiler.section("trajectory_scaffolding"):
                return fn(*args, **kwargs)

    return wrapped


def install_profiler(profiler: ExclusiveProfiler) -> MonkeyPatch:
    patch = MonkeyPatch()
    patch.set(loc_student_rattle, "_update_mu_mh", _wrap(profiler, "mu_mh_update", loc_student_rattle._update_mu_mh))
    patch.set(
        loc_student_rattle,
        "_project_momentum",
        _wrap(profiler, "momentum_projection", loc_student_rattle._project_momentum),
    )
    patch.set(
        loc_student_rattle,
        "_project_to_manifold_fixed_direction",
        _wrap(profiler, "position_projection_solver", loc_student_rattle._project_to_manifold_fixed_direction),
    )
    patch.set(
        loc_student_rattle,
        "_project_to_manifold_newton",
        _wrap(profiler, "position_projection_solver", loc_student_rattle._project_to_manifold_newton),
    )
    patch.set(loc_student_rattle, "_grad_potential", _wrap(profiler, "potential_gradient", loc_student_rattle._grad_potential))
    patch.set(
        loc_student_rattle,
        "_potential_energy",
        _wrap(profiler, "potential_energy", loc_student_rattle._potential_energy),
    )
    patch.set(loc_student_rattle, "_hamiltonian", _wrap(profiler, "hamiltonian_wrapper", loc_student_rattle._hamiltonian))
    patch.set(
        loc_student_rattle,
        "_constraint_value",
        _wrap(profiler, "constraint_value", loc_student_rattle._constraint_value),
    )
    patch.set(
        loc_student_rattle,
        "_constraint_grad",
        _wrap(profiler, "constraint_gradient", loc_student_rattle._constraint_grad),
    )
    patch.set(loc_student_rattle, "_constraint_gram", _wrap(profiler, "constraint_gram", loc_student_rattle._constraint_gram))
    patch.set(
        loc_student_rattle,
        "_grad_log_gram_half",
        _wrap(profiler, "gram_correction_gradient", loc_student_rattle._grad_log_gram_half),
    )
    patch.set(
        loc_student_rattle,
        "_rattle_trajectory",
        _trajectory_wrapper(profiler, loc_student_rattle._rattle_trajectory),
    )
    return patch


def run_profiled_case(args: argparse.Namespace, k: float, n: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    params = base_params(args, k, n, seed)
    ledger = CostLedger(
        method="rattle",
        model="student_t",
        n=int(n),
        k=float(k),
        mu_star=float(args.mu_star),
        seed=int(seed),
        iterations=int(args.iterations),
    )
    profiler = ExclusiveProfiler()
    patch = install_profiler(profiler)
    started = time.perf_counter()
    try:
        chain = loc_student_rattle.run_rattle(
            random.PRNGKey(seed),
            args.mu_star,
            params,
            verbose=False,
            cost_ledger=ledger,
        )
    finally:
        elapsed = time.perf_counter() - started
        patch.restore()

    diag = dict(chain.get("projection_diagnostics", {}))
    counters = dict(ledger.counters)
    mus = np.asarray(chain.get("mu_chain", []), dtype=float)
    burn_in = min(max(int(args.burn_in), 0), max(mus.size - 1, 0))
    post = mus[burn_in:]
    ess_mu = effective_sample_size(post)
    measured = float(sum(profiler.totals.values()))
    other = max(float(elapsed - measured), 0.0)
    if other > 0.0:
        profiler.totals["other_overhead"] = profiler.totals.get("other_overhead", 0.0) + other
        profiler.calls["other_overhead"] = profiler.calls.get("other_overhead", 0) + 1

    iterations = max(int(args.iterations), 1)
    block_rows = []
    for block, seconds in sorted(profiler.totals.items()):
        block_rows.append(
            {
                "model": "student_t",
                "k": float(k),
                "n": int(n),
                "seed": int(seed),
                "iterations": int(args.iterations),
                "block": block,
                "calls": int(profiler.calls.get(block, 0)),
                "elapsed_sec": float(seconds),
                "sec_per_iteration": float(seconds) / iterations,
                "share_of_total": float(seconds) / max(elapsed, 1e-12),
            }
        )

    row = {
        "model": "student_t",
        "k": float(k),
        "n": int(n),
        "seed": int(seed),
        "iterations": int(args.iterations),
        "elapsed_sec": float(elapsed),
        "sec_per_iteration": float(elapsed) / iterations,
        "measured_block_sec": measured,
        "other_overhead_sec": other,
        "burn_in": int(args.burn_in),
        "ess_mu": float(ess_mu),
        "ess_mu_per_sec": float(ess_mu) / max(elapsed, 1e-12),
        "posterior_mean_mu": float(np.mean(post)) if post.size else np.nan,
        "posterior_sd_mu": float(np.std(post)) if post.size else np.nan,
        "mu_acceptance_rate": float(chain.get("mu_acceptance_rate", np.nan)),
        "x_acceptance_rate": float(chain.get("x_acceptance_rate", np.nan)),
        "projection_failure_count": int(diag.get("projection_failure_count", counters.get("projection_failures", 0))),
        "reverse_check_failure_count": int(diag.get("reverse_check_failure_count", counters.get("reverse_check_failures", 0))),
        "projection_failure_rate": float(diag.get("projection_failure_count", counters.get("projection_failures", 0)))
        / max(float(counters.get("hmc_proposals", iterations)), 1.0),
        "reverse_check_failure_rate": float(diag.get("reverse_check_failure_count", counters.get("reverse_check_failures", 0)))
        / max(float(counters.get("hmc_proposals", iterations)), 1.0),
        "projection_iterations_per_iteration": float(diag.get("projection_iterations_total", np.nan)) / iterations,
        "max_projection_residual": float(diag.get("max_projection_residual", np.nan)),
        "max_manifold_residual": float(diag.get("max_manifold_residual", np.nan)),
        "max_reverse_position_error": float(diag.get("max_reverse_position_error", np.nan)),
        "max_reverse_momentum_error": float(diag.get("max_reverse_momentum_error", np.nan)),
        "student_grad_evals_per_iteration": float(counters.get("student_grad_evals", np.nan)) / iterations,
        "constraint_grad_evals_per_iteration": float(counters.get("constraint_grad_evals", np.nan)) / iterations,
        "constraint_grad_coordinate_evals_per_iteration": float(counters.get("constraint_grad_evals", np.nan)) * int(n) / iterations,
        "constraint_value_coordinate_evals_per_iteration": float(counters.get("constraint_evals", np.nan)) * int(n) / iterations,
        "student_logpdf_evals_per_iteration": float(counters.get("student_logpdf_evals", np.nan)) / iterations,
        "projection_evals_per_iteration": float(counters.get("projection_evals", np.nan)) / iterations,
        "leapfrog_steps_per_iteration": float(counters.get("leapfrog_steps", np.nan)) / iterations,
        "projection_mode": str(args.rattle_projection_mode),
        "gram_correction_enabled": bool(args.rattle_include_gram_correction),
        "reverse_check": bool(args.reverse_check),
        "rattle_step_size": float(args.rattle_step_size),
        "rattle_num_steps": int(args.rattle_num_steps),
    }
    return row, block_rows


def summarize_cases(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "elapsed_sec",
        "sec_per_iteration",
        "mu_acceptance_rate",
        "x_acceptance_rate",
        "ess_mu",
        "ess_mu_per_sec",
        "posterior_sd_mu",
        "projection_failure_rate",
        "reverse_check_failure_rate",
        "projection_iterations_per_iteration",
        "max_projection_residual",
        "max_manifold_residual",
        "max_reverse_position_error",
        "max_reverse_momentum_error",
        "student_grad_evals_per_iteration",
        "constraint_grad_evals_per_iteration",
        "constraint_grad_coordinate_evals_per_iteration",
        "constraint_value_coordinate_evals_per_iteration",
        "student_logpdf_evals_per_iteration",
        "projection_evals_per_iteration",
        "leapfrog_steps_per_iteration",
    ]
    summary = rows.groupby(["k", "n"], as_index=False)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary["num_seeds"] = rows.groupby(["k", "n"]).size().to_numpy()
    return summary


def summarize_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    summary = (
        blocks.groupby(["k", "n", "block"], as_index=False)
        .agg(
            sec_per_iteration_mean=("sec_per_iteration", "mean"),
            sec_per_iteration_std=("sec_per_iteration", "std"),
            share_of_total_mean=("share_of_total", "mean"),
            calls_mean=("calls", "mean"),
        )
        .sort_values(["k", "n", "block"])
    )
    return summary


def block_group(block: str) -> str:
    if ":" in block:
        phase, _ = block.split(":", 1)
        if phase in {"forward proposal", "reverse check"}:
            return phase
    if block == "mu_mh_update":
        return "mu MH"
    if block == "trajectory_scaffolding":
        return "trajectory scaffolding"
    if block == "position_projection_solver":
        return "position projection"
    if block == "momentum_projection":
        return "momentum projection"
    if block in {"potential_gradient", "potential_energy", "hamiltonian_wrapper"}:
        return "potential/Hamiltonian"
    if block in {"constraint_value", "constraint_gradient", "constraint_gram", "gram_correction_gradient"}:
        return "constraint/Gram"
    return "other overhead"


def grouped_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    out = blocks.copy()
    out["block_group"] = out["block"].map(block_group)
    return (
        out.groupby(["k", "n", "seed", "block_group"], as_index=False)
        .agg(sec_per_iteration=("sec_per_iteration", "sum"), share_of_total=("share_of_total", "sum"))
    )


def write_figures(out_dir: Path, case_summary: pd.DataFrame, block_summary: pd.DataFrame, grouped: pd.DataFrame) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []

    for k, sub in grouped.groupby("k"):
        pivot = (
            sub.groupby(["n", "block_group"], as_index=False)["sec_per_iteration"]
            .mean()
            .pivot(index="n", columns="block_group", values="sec_per_iteration")
            .fillna(0.0)
            .sort_index()
        )
        order = [
            "mu MH",
            "forward proposal",
            "reverse check",
            "potential/Hamiltonian",
            "constraint/Gram",
            "position projection",
            "momentum projection",
            "trajectory scaffolding",
            "other overhead",
        ]
        pivot = pivot[[col for col in order if col in pivot.columns]]
        ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 5), width=0.82)
        ax.set_title(f"Student-t RATTLE cost breakup, k={float(k):g}")
        ax.set_xlabel("n")
        ax.set_ylabel("seconds per iteration")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        plt.tight_layout()
        path = fig_dir / f"rattle_cost_breakup_stacked_k{float(k):g}.png"
        plt.savefig(path, dpi=160)
        plt.close()
        figures.append(str(path))

    for k, sub in case_summary.groupby("k"):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(sub["n"], sub["sec_per_iteration_mean"], marker="o", label="total wall time")
        group_summary = grouped[grouped["k"].eq(k)].groupby(["n", "block_group"], as_index=False)["sec_per_iteration"].mean()
        proposal = group_summary[group_summary["block_group"].eq("forward proposal")]
        reverse = group_summary[group_summary["block_group"].eq("reverse check")]
        overhead = group_summary[group_summary["block_group"].isin(["other overhead", "mu MH"])]
        ax.plot(proposal.groupby("n")["sec_per_iteration"].sum().index, proposal.groupby("n")["sec_per_iteration"].sum().values, marker="s", label="forward proposal")
        ax.plot(reverse.groupby("n")["sec_per_iteration"].sum().index, reverse.groupby("n")["sec_per_iteration"].sum().values, marker="^", label="reverse check")
        ax.plot(overhead.groupby("n")["sec_per_iteration"].sum().index, overhead.groupby("n")["sec_per_iteration"].sum().values, marker="d", label="fixed overhead + mu MH")
        ax.set_xscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("seconds per iteration")
        ax.set_title(f"RATTLE timing components vs n, k={float(k):g}")
        ax.legend()
        plt.tight_layout()
        path = fig_dir / f"rattle_total_vs_components_k{float(k):g}.png"
        plt.savefig(path, dpi=160)
        plt.close(fig)
        figures.append(str(path))

    for k, sub in grouped.groupby("k"):
        group_summary = sub.groupby(["n", "block_group"], as_index=False)["sec_per_iteration"].mean()
        pivot = (
            group_summary[group_summary["block_group"].isin(["forward proposal", "reverse check"])]
            .pivot(index="n", columns="block_group", values="sec_per_iteration")
            .fillna(0.0)
            .sort_index()
        )
        fig, ax = plt.subplots(figsize=(8, 4.8))
        if "forward proposal" in pivot:
            ax.plot(pivot.index, pivot["forward proposal"], marker="o", label="forward proposal")
        if "reverse check" in pivot:
            ax.plot(pivot.index, pivot["reverse check"], marker="s", label="reverse check")
        ax.set_xscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("seconds per iteration")
        ax.set_title(f"Forward proposal vs reverse check cost, k={float(k):g}")
        ax.legend()
        plt.tight_layout()
        path = fig_dir / f"rattle_forward_vs_reverse_check_k{float(k):g}.png"
        plt.savefig(path, dpi=160)
        plt.close(fig)
        figures.append(str(path))

    for k, sub in case_summary.groupby("k"):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(sub["n"], sub["student_grad_evals_per_iteration_mean"], marker="o", label="Student gradient coordinates")
        ax.plot(sub["n"], sub["student_logpdf_evals_per_iteration_mean"], marker="s", label="Student logpdf coordinates")
        ax.plot(
            sub["n"],
            sub["constraint_grad_coordinate_evals_per_iteration_mean"],
            marker="^",
            label="constraint-gradient coordinates",
        )
        ax.plot(
            sub["n"],
            sub["constraint_value_coordinate_evals_per_iteration_mean"],
            marker="d",
            label="constraint-value coordinates",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("coordinate evaluations per iteration")
        ax.set_title(f"Operation counts per iteration, k={float(k):g}")
        ax.legend()
        plt.tight_layout()
        path = fig_dir / f"rattle_operation_counts_k{float(k):g}.png"
        plt.savefig(path, dpi=160)
        plt.close(fig)
        figures.append(str(path))

    for k, sub in case_summary.groupby("k"):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(sub["n"], sub["projection_iterations_per_iteration_mean"], marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("Newton iterations per sampler iteration")
        ax.set_title(f"Position projection work vs n, k={float(k):g}")
        plt.tight_layout()
        path = fig_dir / f"rattle_projection_iterations_k{float(k):g}.png"
        plt.savefig(path, dpi=160)
        plt.close(fig)
        figures.append(str(path))

    for k, sub in case_summary.groupby("k"):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(sub["n"], sub["ess_mu_per_sec_mean"], marker="o", label="ESS(mu)/sec")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("ESS(mu) per second")
        ax.set_title(f"Diagnostic local ESS/sec vs n, k={float(k):g}")
        ax.legend()
        plt.tight_layout()
        path = fig_dir / f"rattle_ess_per_sec_k{float(k):g}.png"
        plt.savefig(path, dpi=160)
        plt.close(fig)
        figures.append(str(path))

    (fig_dir / "figure_index.json").write_text(json.dumps({"figures": figures}, indent=2), encoding="utf-8")
    return figures


def _markdown_table(df: pd.DataFrame, cols: list[str], floatfmt: str = ".6g") -> str:
    if df.empty:
        return "_No rows._"
    return df[cols].to_markdown(index=False, floatfmt=floatfmt)


def write_report(
    args: argparse.Namespace,
    case_summary: pd.DataFrame,
    block_summary: pd.DataFrame,
    grouped_summary: pd.DataFrame,
    figures: list[str],
) -> None:
    report = args.out / "rattle_cost_profile_report.md"
    key_cols = [
        "k",
        "n",
        "sec_per_iteration_mean",
        "projection_iterations_per_iteration_mean",
        "student_grad_evals_per_iteration_mean",
        "student_logpdf_evals_per_iteration_mean",
        "constraint_grad_coordinate_evals_per_iteration_mean",
        "constraint_value_coordinate_evals_per_iteration_mean",
        "constraint_grad_evals_per_iteration_mean",
        "projection_evals_per_iteration_mean",
        "ess_mu_mean",
        "ess_mu_per_sec_mean",
        "x_acceptance_rate_mean",
        "projection_failure_rate_mean",
        "reverse_check_failure_rate_mean",
    ]
    block_cols = ["k", "n", "block_group", "sec_per_iteration_mean", "share_of_total_mean"]
    top_blocks = grouped_summary.sort_values(["k", "n", "sec_per_iteration_mean"], ascending=[True, True, False])

    lines = [
        "# RATTLE Cost Breakup Profile",
        "",
        "## Settings",
        "",
        f"- model: Student-t location",
        f"- k values: {','.join(str(k) for k in args.k_values)}",
        f"- n values: {','.join(str(n) for n in args.n_values)}",
        f"- seeds: {','.join(str(seed) for seed in args.seeds)}",
        f"- iterations: {int(args.iterations)}",
        f"- mass matrix: M = I",
        f"- projection_mode: {args.rattle_projection_mode}",
        f"- Gram correction enabled: {bool(args.rattle_include_gram_correction)}",
        f"- reverse check enabled: {bool(args.reverse_check)}",
        f"- step size: {float(args.rattle_step_size)}",
        f"- leapfrog steps: {int(args.rattle_num_steps)}",
        "",
        "## Interpretation",
        "",
        "The implementation performs vector work that scales with n, especially Student-t gradient and log-density element evaluations.  The local wall-clock curve can still grow sublinearly over this range because fixed Python/JAX/PRNG/bookkeeping overhead and a nearly constant number of scalar projection/Newton calls dominate the small-to-moderate n regime.",
        "",
        "The reverse projection check is included as its own timed group.  It reruns a RATTLE trajectory from the momentum-reversed proposal and is therefore a real part of per-iteration cost whenever `reverse_check=True`.",
        "",
        "This profile is local timing evidence, not a claim that RATTLE is dimension-free.",
        "",
        "The ESS/sec values below are short-run local diagnostics.  They are useful for discussing how cost and mixing jointly affect efficiency, but production efficiency claims should use long-chain/canonical cost-audit rows after correctness checks pass.",
        "",
        "## Case Summary",
        "",
        _markdown_table(case_summary, key_cols),
        "",
        "## Grouped Cost Breakup",
        "",
        _markdown_table(top_blocks, block_cols),
        "",
        "## Tuned Parameters",
        "",
        "- `rattle_step_size`",
        "- `rattle_num_steps`",
        "- `rattle_proj_tol`",
        "- `rattle_proj_max_iters`",
        "- `rattle_grad_tol`",
        "- `rattle_reverse_position_tol`",
        "- `rattle_reverse_momentum_tol`",
        "- `rattle_tangent_tol`",
        "- `rattle_proj_damping`",
        "- `rattle_proj_line_search`",
        "- `rattle_proj_init_strategy`",
        "- `rattle_projection_mode`",
        "- `rattle_include_gram_correction`",
        "- `reverse_check`",
        "",
        "## Held Fixed Here",
        "",
        "- `M = I`",
        "- fresh Gaussian momentum each iteration, projected to the tangent space",
        "- no persistent GHMC momentum and no friction `gamma`",
        "- `mu_star = 0`",
        "- central initialization",
        "- prior `N(0, 10^2)`",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{path}`" for path in figures)
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.warmup:
        print("[warmup] start untimed RATTLE warm-up", flush=True)
        for k in args.k_values:
            for n in args.n_values:
                warm_params = base_params(args, float(k), int(n), int(args.seeds[0]), iterations=2)
                loc_student_rattle.run_rattle(
                    random.PRNGKey(int(args.seeds[0])),
                    args.mu_star,
                    warm_params,
                    verbose=False,
                    cost_ledger=None,
                )
        print("[warmup] done", flush=True)

    total = len(args.k_values) * len(args.n_values) * len(args.seeds)
    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    case_idx = 0
    for k in args.k_values:
        for n in args.n_values:
            for seed in args.seeds:
                case_idx += 1
                print(f"[{case_idx}/{total}] start k={float(k):g} n={int(n)} seed={int(seed)}", flush=True)
                row, blocks = run_profiled_case(args, float(k), int(n), int(seed))
                rows.append(row)
                block_rows.extend(blocks)
                print(
                    f"[{case_idx}/{total}] done sec/iter={row['sec_per_iteration']:.6g} "
                    f"x_acc={row['x_acceptance_rate']:.3f} "
                    f"proj_fail={row['projection_failure_rate']:.3g} "
                    f"rev_fail={row['reverse_check_failure_rate']:.3g}",
                    flush=True,
                )

    rows_df = pd.DataFrame(rows)
    blocks_df = pd.DataFrame(block_rows)
    case_summary = summarize_cases(rows_df)
    block_summary = summarize_blocks(blocks_df)
    grouped = grouped_blocks(blocks_df)
    grouped_summary = (
        grouped.groupby(["k", "n", "block_group"], as_index=False)
        .agg(
            sec_per_iteration_mean=("sec_per_iteration", "mean"),
            sec_per_iteration_std=("sec_per_iteration", "std"),
            share_of_total_mean=("share_of_total", "mean"),
        )
        .sort_values(["k", "n", "block_group"])
    )

    rows_df.to_csv(args.out / "rattle_cost_profile_rows.csv", index=False)
    blocks_df.to_csv(args.out / "rattle_cost_profile_blocks.csv", index=False)
    case_summary.to_csv(args.out / "rattle_cost_profile_summary.csv", index=False)
    block_summary.to_csv(args.out / "rattle_cost_profile_block_summary.csv", index=False)
    grouped_summary.to_csv(args.out / "rattle_cost_profile_grouped_summary.csv", index=False)
    figures = write_figures(args.out, case_summary, block_summary, grouped)
    write_report(args, case_summary, block_summary, grouped_summary, figures)
    print(f"wrote profiling outputs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
