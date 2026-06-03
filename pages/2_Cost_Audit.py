"""Streamlit page for Gibbs vs RATTLE cost-audit details."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard_cache import read_cache_csv, require_cache_file, sidebar_cache_controls, show_cache_badge


OUT_DIR = Path("results/cost_audit")
RUN_DIRS = {
    "smoke": Path("results/cost_audit_smoke"),
    "medium": Path("results/cost_audit_medium"),
    "full": Path("results/cost_audit"),
    "multiseed": Path("results/cost_audit_multiseed"),
}
REFERENCE_AUDIT_DEFAULT = Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv")
ANALYSIS_REPORT = Path("results/analysis_report/executive_summary.md")
RATTLE_SETTINGS = Path("results/rattle_tuning/recommended_rattle_settings.json")

RUN_COMMAND = (
    "python scripts/run_cost_audit.py --methods gibbs rattle --n-values 10,20,50 "
    "--k 2 --mu-star 0 --num-iterations 10000 --burn-in 2000 --seed 0 --out results/cost_audit/"
)

GIBBS_COLUMNS = [
    "model",
    "method",
    "n",
    "k",
    "mu_star",
    "seed",
    "mu_mh_proposals",
    "mu_mh_accepts",
    "pair_updates_attempted",
    "pair_updates_completed",
    "pair_grid_evals",
    "pair_inverse_branch_evals",
    "pair_weight_evals",
    "pair_rejections",
    "sweep_count",
]

RATTLE_COLUMNS = [
    "model",
    "method",
    "n",
    "k",
    "mu_star",
    "seed",
    "hmc_proposals",
    "hmc_accepts",
    "leapfrog_steps",
    "forward_newton_iters",
    "reverse_newton_iters",
    "momentum_projections",
    "reverse_check_attempts",
    "reverse_check_failures",
    "energy_evals",
    "integration_failures",
    "projection_mode",
    "gram_correction_enabled",
    "position_projection_newton_iters",
    "position_projection_failures",
    "reverse_position_error",
    "reverse_momentum_error",
]

IDENTITY_COLUMNS = ["model", "method", "n", "k", "mu_star", "seed", "run_status", "target_description"]
HEADLINE_COST_COLUMNS = [
    "model",
    "k",
    "n",
    "method",
    "ess_per_sec",
    "wall_time_per_ess",
    "acceptance_rate",
    "projection_failure_rate",
    "reverse_check_failure_rate",
    "run_status",
]
NORMALIZED_DEFAULT_COLUMNS = [
    "model",
    "k",
    "n",
    "method",
    "wall_time_per_iteration",
    "wall_time_per_ess",
    "constraint_evals_per_iteration",
    "projection_evals_per_iteration",
    "leapfrog_steps_per_iteration",
    "reverse_check_fail_rate",
]
HEADLINE_EXTRA_COLUMNS = [
    "seed",
    "iterations",
    "ess_mu",
    "wall_time_sec",
    "wall_time_per_iteration",
    "student_logpdf_evals_per_iteration",
    "student_grad_evals_per_iteration",
    "newton_iters_per_iteration",
    "pair_grid_evals_per_iteration",
    "projection_mode",
    "gram_correction_enabled",
    "rattle_status",
]
RAW_LEDGER_FRONT_COLUMNS = [
    "model",
    "method",
    "n",
    "k",
    "seed",
    "iterations",
    "ess_mu",
    "ess_per_sec",
    "acceptance_rate",
    "wall_time_sec",
    "run_status",
    "rattle_status",
    "projection_mode",
    "gram_correction_enabled",
    "student_logpdf_evals",
    "student_grad_evals",
    "constraint_evals",
    "constraint_grad_evals",
    "gram_evals",
    "projection_evals",
    "projection_failures",
    "pair_grid_evals",
    "hmc_proposals",
    "hmc_accepts",
    "leapfrog_steps",
]


@st.cache_data(show_spinner=False)
def read_csv_if_exists(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out


def reorder_columns(df: pd.DataFrame, priority: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    front = [column for column in priority if column in df.columns]
    rest = [column for column in df.columns if column not in front]
    return df[front + rest]


def coerce_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "wall_time_per_ess": "cost_per_effective_sample_sec",
        "wall_time_per_iteration": "wall_time_per_iteration_sec",
        "reverse_check_failure_rate": "reverse_check_fail_rate",
    }
    for canonical, alias in aliases.items():
        if canonical not in out.columns and alias in out.columns:
            out[canonical] = out[alias]
    return out


def available_seed_options(*frames: pd.DataFrame) -> list[str]:
    seeds: set[int] = set()
    for frame in frames:
        if frame.empty or "seed" not in frame.columns:
            continue
        values = pd.to_numeric(frame["seed"], errors="coerce").dropna().astype(int)
        seeds.update(values.tolist())
    return ["all"] + [str(seed) for seed in sorted(seeds)]


def apply_filters(df: pd.DataFrame, model_filter: str, method_filter: str, n_filter: str, k_filter: str, seed_filter: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if model_filter != "all" and "model" in out.columns:
        out = out[out["model"].astype(str).eq(model_filter)]
    if method_filter != "both" and "method" in out.columns:
        out = out[out["method"].astype(str).eq(method_filter)]
    if n_filter != "all" and "n" in out.columns:
        out = out[out["n"].astype(int).eq(int(n_filter))]
    if k_filter != "all" and "k" in out.columns:
        out = out[np.isclose(out["k"].astype(float), float(k_filter))]
    if seed_filter != "all" and "seed" in out.columns:
        out = out[out["seed"].astype(int).eq(int(seed_filter))]
    return out.reset_index(drop=True)


def normalized_cost_table(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    out = ledger.copy()
    needed = [
        "iterations",
        "wall_time_sec",
        "ess_mu",
        "student_logpdf_evals",
        "student_grad_evals",
        "constraint_evals",
        "constraint_grad_evals",
        "gram_evals",
        "projection_evals",
        "projection_failures",
        "forward_newton_iters",
        "reverse_newton_iters",
        "leapfrog_steps",
        "reverse_check_failures",
        "reverse_check_attempts",
        "hmc_proposals",
        "pair_grid_evals",
        "acceptance_rate",
    ]
    out = ensure_columns(out, needed)
    iterations = out["iterations"].replace(0, np.nan)
    ess = out["ess_mu"].replace(0, np.nan)
    reverse_attempts = out["reverse_check_attempts"].where(out["reverse_check_attempts"] > 0, out["hmc_proposals"])

    out["wall_time_per_iteration"] = out["wall_time_sec"] / iterations
    out["wall_time_per_ess"] = out["wall_time_sec"] / ess
    out["student_logpdf_evals_per_iteration"] = out["student_logpdf_evals"] / iterations
    out["student_grad_evals_per_iteration"] = out["student_grad_evals"] / iterations
    out["constraint_evals_per_iteration"] = out["constraint_evals"] / iterations
    out["constraint_grad_evals_per_iteration"] = out["constraint_grad_evals"] / iterations
    out["gram_evals_per_iteration"] = out["gram_evals"] / iterations
    out["projection_evals_per_iteration"] = out["projection_evals"] / iterations
    out["newton_iters_per_iteration"] = (out["forward_newton_iters"] + out["reverse_newton_iters"]) / iterations
    out["leapfrog_steps_per_iteration"] = out["leapfrog_steps"] / iterations
    out["reverse_check_fail_rate"] = out["reverse_check_failures"] / reverse_attempts.replace(0, np.nan)
    out["reverse_check_failure_rate"] = out["reverse_check_fail_rate"]
    out["projection_failure_rate"] = out["projection_failures"] / out["projection_evals"].replace(0, np.nan)
    out["pair_grid_evals_per_iteration"] = out["pair_grid_evals"] / iterations

    columns = [
        "model",
        "method",
        "n",
        "k",
        "mu_star",
        "seed",
        "run_status",
        "iterations",
        "wall_time_sec",
        "ess_mu",
        "ess_per_sec",
        "acceptance_rate",
        "wall_time_per_iteration",
        "wall_time_per_ess",
        "student_logpdf_evals_per_iteration",
        "student_grad_evals_per_iteration",
        "constraint_evals_per_iteration",
        "constraint_grad_evals_per_iteration",
        "gram_evals_per_iteration",
        "projection_evals_per_iteration",
        "newton_iters_per_iteration",
        "leapfrog_steps_per_iteration",
        "reverse_check_fail_rate",
        "reverse_check_failure_rate",
        "projection_failure_rate",
        "pair_grid_evals_per_iteration",
        "projection_mode",
        "gram_correction_enabled",
        "rattle_status",
        "target_description",
    ]
    return ensure_columns(out, columns)[columns]


def add_series_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    labels = []
    for row in out.itertuples(index=False):
        model = str(getattr(row, "model", ""))
        method = str(getattr(row, "method", ""))
        k = getattr(row, "k", np.nan)
        if model == "student_t" and pd.notna(k):
            labels.append(f"Student k={float(k):g} {method}")
        else:
            labels.append(f"{model} {method}")
    out["series_label"] = labels
    return out


def headline_cost_table(ledger: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty and normalized.empty:
        return pd.DataFrame(columns=HEADLINE_COST_COLUMNS)
    base_cols = [column for column in IDENTITY_COLUMNS + ["iterations", "wall_time_sec", "ess_mu", "ess_per_sec", "acceptance_rate", "projection_mode", "gram_correction_enabled", "rattle_status"] if column in ledger.columns]
    base = ledger[base_cols].copy() if not ledger.empty else pd.DataFrame()
    norm_cols = [
        column
        for column in [
            "model",
            "method",
            "n",
            "k",
            "mu_star",
            "seed",
            "wall_time_per_iteration",
            "wall_time_per_ess",
            "student_logpdf_evals_per_iteration",
            "student_grad_evals_per_iteration",
            "constraint_evals_per_iteration",
            "projection_evals_per_iteration",
            "newton_iters_per_iteration",
            "leapfrog_steps_per_iteration",
            "pair_grid_evals_per_iteration",
            "reverse_check_fail_rate",
            "projection_failure_rate",
        ]
        if column in normalized.columns
    ]
    if base.empty:
        out = normalized.copy()
    else:
        keys = [column for column in ["model", "method", "n", "k", "mu_star", "seed"] if column in base.columns and column in normalized.columns]
        out = base.merge(normalized[norm_cols], on=keys, how="left") if keys and not normalized.empty else base
    return reorder_columns(ensure_columns(out, HEADLINE_COST_COLUMNS + HEADLINE_EXTRA_COLUMNS), HEADLINE_COST_COLUMNS + HEADLINE_EXTRA_COLUMNS)


def cost_summary_cards(ledger: pd.DataFrame, normalized: pd.DataFrame, suspicious: pd.DataFrame | None = None) -> None:
    cols = st.columns(6)
    if not normalized.empty and "ess_per_sec" in normalized.columns:
        finite_ess = normalized[np.isfinite(pd.to_numeric(normalized["ess_per_sec"], errors="coerce"))]
    else:
        finite_ess = pd.DataFrame()
    if finite_ess.empty:
        best_ess = "unavailable"
    else:
        row = finite_ess.sort_values("ess_per_sec", ascending=False).iloc[0]
        best_ess = f"{row.get('model', '')} {row.get('method', '')} n={int(row.get('n', 0))}"
    if not normalized.empty and "wall_time_per_ess" in normalized.columns:
        finite_cost = normalized[np.isfinite(pd.to_numeric(normalized["wall_time_per_ess"], errors="coerce"))]
    else:
        finite_cost = pd.DataFrame()
    if finite_cost.empty:
        best_cost = "unavailable"
    else:
        row = finite_cost.sort_values("wall_time_per_ess", ascending=True).iloc[0]
        best_cost = f"{row.get('model', '')} {row.get('method', '')} n={int(row.get('n', 0))}"
    rattle = ledger[ledger.get("method", pd.Series(dtype=str)).astype(str).eq("rattle")] if not ledger.empty else pd.DataFrame()
    acceptance_warnings = 0
    if not rattle.empty and "acceptance_rate" in rattle.columns:
        rates = pd.to_numeric(rattle["acceptance_rate"], errors="coerce")
        acceptance_warnings = int(((rates < 0.5) | (rates > 0.995)).sum())
    projection_failures = int(pd.to_numeric(ledger.get("projection_failures", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not ledger.empty else 0
    reverse_failures = int(pd.to_numeric(ledger.get("reverse_check_failures", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not ledger.empty else 0
    suspicious_count = int(len(suspicious)) if suspicious is not None and not suspicious.empty else 0
    cols[0].metric("Best ESS/sec", best_ess)
    cols[1].metric("Lowest cost/ESS", best_cost)
    cols[2].metric("RATTLE acceptance warnings", acceptance_warnings)
    cols[3].metric("Projection failures", projection_failures)
    cols[4].metric("Reverse-check failures", reverse_failures)
    cols[5].metric("Suspicious cases", suspicious_count)


def plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str | None = None, method_filter: str = "both", chart: str = "points") -> None:
    if df.empty or metric not in df.columns:
        st.info(f"No data available for {title}.")
        return
    plot_df = df.copy()
    if method_filter != "both":
        plot_df = plot_df[plot_df["method"].astype(str).eq(method_filter)]
    plot_df = plot_df[np.isfinite(plot_df[metric])]
    if plot_df.empty:
        st.info(f"No finite values available for {title}.")
        return
    plot_df = add_series_label(plot_df)

    if chart == "bars":
        labels = plot_df["series_label"].drop_duplicates().tolist()
        n_values = sorted(plot_df["n"].dropna().astype(int).unique())
        x = np.arange(len(n_values))
        width = 0.8 / max(len(labels), 1)
        fig, ax = plt.subplots(figsize=(10.5, 5.4))
        for index, label in enumerate(labels):
            part = plot_df[plot_df["series_label"].eq(label)]
            values = [
                float(part[part["n"].astype(int).eq(int(n_value))][metric].mean())
                if not part[part["n"].astype(int).eq(int(n_value))].empty
                else np.nan
                for n_value in n_values
            ]
            ax.bar(x + (index - (len(labels) - 1) / 2.0) * width, values, width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n_value) for n_value in n_values])
    else:
        fig, ax = plt.subplots(figsize=(10.5, 5.4))
        for label, part in plot_df.groupby("series_label", sort=False):
            part = part.sort_values("n")
            ax.plot(part["n"], part[metric], marker="o", linewidth=0, markersize=7, label=str(label))
            if len(part["n"].dropna().unique()) > 1:
                ax.plot(part["n"], part[metric], linewidth=1.2, alpha=0.45)
        ax.set_xticks(sorted(plot_df["n"].dropna().astype(int).unique()))

    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel or metric)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def acf(values: np.ndarray, max_lag: int = 50) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.array([0]), np.array([1.0])
    centered = values - values.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0:
        return np.array([0]), np.array([1.0])
    max_lag = min(max_lag, values.size - 1)
    lags = np.arange(max_lag + 1)
    vals = np.array([1.0 if lag == 0 else float(np.dot(centered[:-lag], centered[lag:]) / denom) for lag in lags])
    return lags, vals


def plot_chain_diagnostics(chain: pd.DataFrame, reference_summary: pd.DataFrame, ledger: pd.DataFrame | None = None) -> None:
    if chain.empty:
        st.info("chain_samples.csv is missing; chain plots are unavailable.")
        return

    st.subheader("Chain Diagnostics")
    show_multiple = st.checkbox("Show multiple chains", value=False)
    models = list(chain["model"].dropna().astype(str).unique()) if "model" in chain.columns else ["student_t"]
    model_index = models.index("student_t") if "student_t" in models else 0
    methods = list(chain["method"].dropna().astype(str).unique())
    method_index = methods.index("gibbs") if "gibbs" in methods else 0
    n_values = sorted(chain["n"].dropna().astype(int).unique())
    n_index = n_values.index(20) if 20 in n_values else 0
    seed_values = sorted(chain["seed"].dropna().astype(int).unique()) if "seed" in chain.columns else [0]
    seed_index = seed_values.index(0) if 0 in seed_values else 0
    k_values = sorted(chain["k"].dropna().astype(float).unique()) if "k" in chain.columns else [2.0]
    k_index = list(k_values).index(2.0) if 2.0 in list(k_values) else 0
    col1, col2, col3, col4, col5 = st.columns(5)
    chosen_model = col1.selectbox("chain model", models, index=model_index)
    chosen_k = col2.selectbox("chain k", k_values, index=k_index, disabled=chosen_model != "student_t")
    chosen_n = col3.selectbox("chain n", n_values, index=n_index)
    chosen_method = col4.selectbox("chain method", methods, index=method_index)
    chosen_seed = col5.selectbox("chain seed", seed_values, index=seed_index)
    part = chain.copy()
    if "model" in part.columns:
        part = part[part["model"].astype(str).eq(str(chosen_model))]
    if "k" in part.columns and chosen_model == "student_t":
        part = part[np.isclose(part["k"].astype(float), float(chosen_k))]
    part = part[part["n"].astype(int).eq(int(chosen_n))]
    if not show_multiple:
        part = part[part["method"].astype(str).eq(chosen_method)]
        if "seed" in part.columns:
            part = part[part["seed"].astype(int).eq(int(chosen_seed))]
    if part.empty:
        st.info("No matching chain samples.")
        return

    post = part[~part["is_burn_in"].astype(bool)] if "is_burn_in" in part.columns else part
    ledger_part = pd.DataFrame()
    if ledger is not None and not ledger.empty:
        ledger_part = ledger.copy()
        if "model" in ledger_part.columns:
            ledger_part = ledger_part[ledger_part["model"].astype(str).eq(str(chosen_model))]
        if "k" in ledger_part.columns and chosen_model == "student_t":
            ledger_part = ledger_part[np.isclose(ledger_part["k"].astype(float), float(chosen_k))]
        ledger_part = ledger_part[ledger_part["n"].astype(int).eq(int(chosen_n))]
        if not show_multiple:
            ledger_part = ledger_part[ledger_part["method"].astype(str).eq(chosen_method)]
            if "seed" in ledger_part.columns:
                ledger_part = ledger_part[ledger_part["seed"].astype(int).eq(int(chosen_seed))]
    summary_cols = st.columns(5)
    summary_cols[0].metric("mean", f"{post['mu'].mean():.3g}")
    summary_cols[1].metric("sd", f"{post['mu'].std():.3g}")
    ess_value = pd.to_numeric(ledger_part.get("ess_mu", pd.Series(dtype=float)), errors="coerce").mean() if not ledger_part.empty else np.nan
    ess_sec_value = pd.to_numeric(ledger_part.get("ess_per_sec", pd.Series(dtype=float)), errors="coerce").mean() if not ledger_part.empty else np.nan
    accept_value = pd.to_numeric(ledger_part.get("acceptance_rate", pd.Series(dtype=float)), errors="coerce").mean() if not ledger_part.empty else np.nan
    summary_cols[2].metric("ESS", "n/a" if not np.isfinite(ess_value) else f"{ess_value:.3g}")
    summary_cols[3].metric("ESS/sec", "n/a" if not np.isfinite(ess_sec_value) else f"{ess_sec_value:.3g}")
    summary_cols[4].metric("acceptance", "n/a" if not np.isfinite(accept_value) else f"{accept_value:.3g}")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    group_cols = [col for col in ["method", "seed"] if col in part.columns]
    if show_multiple and group_cols:
        for label, sub in part.groupby(group_cols, sort=False):
            ax.plot(sub["iteration"], sub["mu"], linewidth=0.9, alpha=0.75, label=str(label))
        ax.legend(loc="best", fontsize=8)
    else:
        ax.plot(part["iteration"], part["mu"], linewidth=1)
    ax.set_title(f"Trace of mu: {chosen_model}, k={chosen_k:g}, n={chosen_n}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mu")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    lags, acf_vals = acf(post["mu"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(lags, acf_vals, width=0.8)
    ax.set_title(f"ACF of mu: {chosen_method}, n={chosen_n}")
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for method, method_part in part.groupby("method", sort=False):
        samples = method_part[~method_part["is_burn_in"].astype(bool)]["mu"] if "is_burn_in" in method_part.columns else method_part["mu"]
        ax.hist(samples, bins=min(40, max(12, int(np.sqrt(len(samples))))), density=True, alpha=0.28, label=str(method))
    if not reference_summary.empty:
        raw = reference_summary[
            reference_summary["estimator_type"].astype(str).eq("raw_weighted_mc")
            & reference_summary["n"].astype(int).eq(int(chosen_n))
        ]
        for _, row in raw.iterrows():
            ax.axvline(row["posterior_mean"], color="black", linestyle="-", linewidth=1.2, label="raw weighted-MC mean")
            ax.axvline(row["posterior_q025"], color="black", linestyle=":", linewidth=1.0, label="raw weighted-MC 95% interval")
            ax.axvline(row["posterior_q975"], color="black", linestyle=":", linewidth=1.0)
    ax.set_title(f"Posterior density overlay: n={chosen_n}")
    ax.set_xlabel("mu")
    ax.set_ylabel("density")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def load_reference_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def csv_status(paths: dict[str, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "file": name,
                "path": str(path),
                "status": "found" if path.exists() else "missing",
            }
            for name, path in paths.items()
        ]
    )


def values_status(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return "unavailable"
    values = sorted(df[column].dropna().unique())
    if not values:
        return "unavailable"
    return ", ".join(str(int(value)) if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer() else str(value) for value in values)


def smoke_status(ledger: pd.DataFrame, chain: pd.DataFrame) -> str:
    if not ledger.empty and "run_status" in ledger.columns:
        statuses = sorted(set(ledger["run_status"].dropna().astype(str)))
        if statuses:
            return ", ".join(statuses)
    n_values = set()
    if not ledger.empty and "n" in ledger.columns:
        n_values.update(ledger["n"].dropna().astype(int).tolist())
    if not chain.empty and "n" in chain.columns:
        n_values.update(chain["n"].dropna().astype(int).tolist())
    if not ledger.empty and "iterations" in ledger.columns and ledger["iterations"].dropna().astype(float).min() < 1000:
        return "smoke"
    if n_values and n_values != {10, 20, 50}:
        return "smoke"
    if chain.empty and ledger.empty:
        return "missing"
    return "full_or_unknown"


def auto_best_results_dir() -> Path:
    for label in ["full", "medium", "smoke"]:
        path = RUN_DIRS[label]
        if (path / "cost_ledger.csv").exists():
            return path
    return OUT_DIR


def rattle_settings_status(path: Path = RATTLE_SETTINGS) -> str:
    if not path.exists():
        return "missing"
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    statuses = sorted({str(row.get("status", "")) for row in data.get("settings", []) if row.get("status")})
    return ", ".join(statuses) if statuses else "unknown"


st.title("Gibbs vs RATTLE Cost Audit")
st.caption("This page compares computational cost. Posterior correctness should still be judged on the KDE/raw weighted-MC reference page.")
use_dashboard_cache, dashboard_cache_dir, dashboard_manifest = sidebar_cache_controls("cost")
show_cache_badge(use_dashboard_cache, dashboard_cache_dir, dashboard_manifest)

if use_dashboard_cache:
    ledger_path = require_cache_file(dashboard_cache_dir, "cost_ledger_cache.csv")
    cost_path = require_cache_file(dashboard_cache_dir, "cost_efficiency_cache.csv")
    summaries_path = require_cache_file(dashboard_cache_dir, "posterior_summaries_cache.csv")
    chain_path = require_cache_file(dashboard_cache_dir, "chain_samples_thinned_cache.csv")
    if not all([ledger_path, cost_path, summaries_path, chain_path]):
        st.stop()

    ledger = read_cache_csv(str(dashboard_cache_dir), "cost_ledger_cache.csv")
    normalized = coerce_cost_columns(read_cache_csv(str(dashboard_cache_dir), "cost_efficiency_cache.csv"))
    summaries = read_cache_csv(str(dashboard_cache_dir), "posterior_summaries_cache.csv")
    chains = read_cache_csv(str(dashboard_cache_dir), "chain_samples_thinned_cache.csv")
    suspicious = read_cache_csv(str(dashboard_cache_dir), "suspicious_cases_cache.csv")

    with st.sidebar:
        st.header("Cached Filters")
        model_filter = st.selectbox("model", ["all", "student_t", "logistic", "laplace"], index=0)
        method_filter = st.selectbox("method", ["both", "gibbs", "rattle"], index=0)
        n_filter = st.selectbox("n", ["all", "10", "20", "50"], index=0)
        k_filter = st.selectbox("k", ["all", "1.0", "2.0", "3.0"], index=0)
        seed_filter = st.selectbox("seed", available_seed_options(ledger, normalized, summaries, chains), index=0)
        show_raw = st.checkbox("show raw counters", value=False)
        show_normalized = st.checkbox("show normalized counters", value=True)

    ledger_f = apply_filters(ledger, model_filter, method_filter, n_filter, k_filter, str(seed_filter))
    normalized_f = apply_filters(normalized, model_filter, method_filter, n_filter, k_filter, str(seed_filter))
    summaries_f = apply_filters(summaries, model_filter, method_filter, n_filter, k_filter, str(seed_filter))
    chains_f = apply_filters(chains, model_filter, method_filter, n_filter, k_filter, str(seed_filter))
    suspicious_f = apply_filters(suspicious, model_filter, method_filter, n_filter, k_filter, str(seed_filter)) if not suspicious.empty else suspicious
    headline_f = headline_cost_table(ledger_f, normalized_f)

    st.subheader("Cost Conclusions")
    cost_summary_cards(ledger_f, normalized_f, suspicious_f)

    st.subheader("Cached Audit Data Status")
    status_cols = st.columns(5)
    status_cols[0].metric("results", "dashboard cache")
    status_cols[1].metric("methods", values_status(ledger_f if not ledger_f.empty else chains_f, "method"))
    status_cols[2].metric("n values", values_status(ledger_f if not ledger_f.empty else chains_f, "n"))
    status_cols[3].metric("seeds", values_status(ledger_f if not ledger_f.empty else chains_f, "seed"))
    status_cols[4].metric("run status", smoke_status(ledger_f, chains_f))

    st.subheader("Cost Overview")
    st.caption("Main comparison columns are kept at the front so model/method/n stay visible while scanning.")
    st.dataframe(headline_f[HEADLINE_COST_COLUMNS], use_container_width=True, hide_index=True)

    if show_normalized:
        st.subheader("Normalized Cost Table")
        st.dataframe(ensure_columns(normalized_f, NORMALIZED_DEFAULT_COLUMNS)[NORMALIZED_DEFAULT_COLUMNS], use_container_width=True, hide_index=True)

    st.subheader("Suspicious Cases")
    if suspicious_f.empty:
        st.info("No suspicious cases for the selected filters.")
    else:
        st.dataframe(reorder_columns(suspicious_f, ["model", "method", "n", "k", "seed", "warning", "reason", "metric", "value"]), use_container_width=True, hide_index=True)

    if show_raw:
        with st.expander("Raw Cost Ledger", expanded=False):
            st.dataframe(reorder_columns(ledger_f, RAW_LEDGER_FRONT_COLUMNS), use_container_width=True, hide_index=True)

    with st.expander("Gibbs-Specific Counters", expanded=False):
        gibbs_table = ensure_columns(ledger_f[ledger_f.get("method", pd.Series(dtype=str)).astype(str).eq("gibbs")], GIBBS_COLUMNS)
        st.dataframe(gibbs_table[GIBBS_COLUMNS], use_container_width=True, hide_index=True)

    with st.expander("RATTLE-Specific Counters", expanded=False):
        rattle_table = ensure_columns(ledger_f[ledger_f.get("method", pd.Series(dtype=str)).astype(str).eq("rattle")], RATTLE_COLUMNS)
        st.dataframe(rattle_table[RATTLE_COLUMNS], use_container_width=True, hide_index=True)

    with st.expander("Posterior Summaries", expanded=False):
        st.dataframe(reorder_columns(summaries_f, ["model", "method", "n", "k", "seed", "mean_mu", "sd_mu", "q025_mu", "q50_mu", "q975_mu", "ess_mu", "ess_per_sec", "acceptance_rate"]), use_container_width=True, hide_index=True)

    st.subheader("Cost Plots")
    tab_ess, tab_cost, tab_accept, tab_diag = st.tabs(["ESS/sec", "Cost per ESS", "Acceptance", "Constraint/projection diagnostics"])
    with tab_ess:
        plot_metric(normalized_f, "ess_per_sec", "ESS/sec by n", "ESS/sec", chart="bars")
    with tab_cost:
        plot_metric(normalized_f, "wall_time_per_ess", "wall time per ESS by n", "seconds")
        plot_metric(normalized_f, "wall_time_per_iteration", "wall time per iteration by n", "seconds")
    with tab_accept:
        plot_metric(normalized_f, "acceptance_rate", "acceptance rate by n", "acceptance rate", chart="bars")
    with tab_diag:
        plot_metric(normalized_f, "projection_evals_per_iteration", "projection evals per iteration by n")
        plot_metric(normalized_f, "constraint_evals_per_iteration", "constraint evals per iteration by n")

    plot_chain_diagnostics(chains_f, pd.DataFrame(), ledger_f)
    st.stop()

with st.sidebar:
    st.header("Data")
    results_level = st.selectbox("results directory selector", ["auto-best", "smoke", "medium", "full", "multiseed", "custom"], index=0)
    default_dir = auto_best_results_dir() if results_level == "auto-best" else RUN_DIRS.get(results_level, OUT_DIR)
    results_dir = Path(st.text_input("results directory", value=str(default_dir)))
    st.header("Filters")
    model_filter = st.selectbox("model", ["all", "student_t", "logistic", "laplace"], index=0)
    method_filter = st.selectbox("method", ["both", "gibbs", "rattle"], index=0)
    n_filter = st.selectbox("n", ["all", "10", "20", "50"], index=0)
    k_filter = st.selectbox("k", ["all", "1.0", "2.0", "3.0"], index=0)
    seed_filter = st.selectbox("seed", ["all", "0", "123", "456", "789"], index=0)
    show_raw = st.checkbox("show raw counters", value=False)
    show_normalized = st.checkbox("show normalized counters", value=True)
    st.header("Reference Overlay")
    overlay_reference = st.checkbox("overlay posterior reference from KDE audit CSV", value=False)
    reference_csv = st.text_input("reference audit CSV", value=str(REFERENCE_AUDIT_DEFAULT))

LEDGER_PATH = results_dir / "cost_ledger.csv"
SUMMARY_PATH = results_dir / "posterior_summaries.csv"
DIAG_PATH = results_dir / "diagnostic_summary.csv"
CHAIN_PATH = results_dir / "chain_samples.csv"

paths = {
    "cost_ledger.csv": LEDGER_PATH,
    "posterior_summaries.csv": SUMMARY_PATH,
    "diagnostic_summary.csv": DIAG_PATH,
    "chain_samples.csv": CHAIN_PATH,
}
missing = [name for name, path in paths.items() if not path.exists()]

ledger = read_csv_if_exists(str(LEDGER_PATH))
summaries = read_csv_if_exists(str(SUMMARY_PATH))
diagnostics = read_csv_if_exists(str(DIAG_PATH))
chains = read_csv_if_exists(str(CHAIN_PATH))
suspicious = read_csv_if_exists("results/analysis_report/suspicious_cases.csv")

st.subheader("Audit Data Status")
status_cols = st.columns(7)
status_cols[0].metric("results directory", str(results_dir))
status_cols[1].metric("methods", values_status(ledger if not ledger.empty else chains, "method"))
status_cols[2].metric("n values", values_status(ledger if not ledger.empty else chains, "n"))
status_cols[3].metric("seeds", values_status(ledger if not ledger.empty else chains, "seed"))
status_cols[4].metric("run status", smoke_status(ledger, chains))
status_cols[5].metric("RATTLE settings", rattle_settings_status())
status_cols[6].metric("analysis report", "found" if ANALYSIS_REPORT.exists() else "missing")
st.dataframe(csv_status(paths), use_container_width=True)

if missing:
    st.warning("Some cost-audit files are missing. The audit will not run automatically.")
    st.code(RUN_COMMAND, language="bash")

if "smoke" in smoke_status(ledger, chains):
    st.warning("Smoke run only. Do not interpret posterior accuracy or cost scientifically.")

ledger_f = apply_filters(ledger, model_filter, method_filter, n_filter, k_filter, str(seed_filter))
summaries_f = apply_filters(summaries, model_filter, method_filter, n_filter, k_filter, str(seed_filter))
diagnostics_f = apply_filters(diagnostics, model_filter, method_filter, n_filter, k_filter, str(seed_filter))
chains_f = apply_filters(chains, model_filter, method_filter, n_filter, k_filter, str(seed_filter))
suspicious_f = apply_filters(suspicious, model_filter, method_filter, n_filter, k_filter, str(seed_filter)) if not suspicious.empty else suspicious
normalized = coerce_cost_columns(normalized_cost_table(ledger_f))
headline = headline_cost_table(ledger_f, normalized)

st.subheader("Cost Conclusions")
cost_summary_cards(ledger_f, normalized, suspicious_f)

st.subheader("Cost Overview")
st.caption("Main comparison columns are kept at the front so model/method/n stay visible while scanning.")
st.dataframe(headline[HEADLINE_COST_COLUMNS], use_container_width=True, hide_index=True)

if show_raw:
    with st.expander("Raw Cost Ledger", expanded=False):
        st.dataframe(reorder_columns(ledger_f, RAW_LEDGER_FRONT_COLUMNS), use_container_width=True, hide_index=True)

if show_normalized:
    st.subheader("Normalized Cost Table")
    st.dataframe(ensure_columns(normalized, NORMALIZED_DEFAULT_COLUMNS)[NORMALIZED_DEFAULT_COLUMNS], use_container_width=True, hide_index=True)

st.subheader("Suspicious Cases")
if suspicious_f.empty:
    st.info("No suspicious cases for the selected filters.")
else:
    st.dataframe(reorder_columns(suspicious_f, ["model", "method", "n", "k", "seed", "warning", "reason", "metric", "value"]), use_container_width=True, hide_index=True)

with st.expander("Gibbs-Specific Counters", expanded=False):
    gibbs_table = ensure_columns(ledger_f[ledger_f.get("method", pd.Series(dtype=str)).astype(str).eq("gibbs")], GIBBS_COLUMNS)
    st.dataframe(gibbs_table[GIBBS_COLUMNS], use_container_width=True, hide_index=True)

with st.expander("RATTLE-Specific Counters", expanded=False):
    rattle_table = ensure_columns(ledger_f[ledger_f.get("method", pd.Series(dtype=str)).astype(str).eq("rattle")], RATTLE_COLUMNS)
    st.dataframe(rattle_table[RATTLE_COLUMNS], use_container_width=True, hide_index=True)

with st.expander("Posterior Summaries", expanded=False):
    st.dataframe(reorder_columns(summaries_f, ["model", "method", "n", "k", "seed", "mean_mu", "sd_mu", "q025_mu", "q50_mu", "q975_mu", "ess_mu", "ess_per_sec", "acceptance_rate"]), use_container_width=True, hide_index=True)

if not diagnostics_f.empty:
    st.subheader("Diagnostic Summary")
    st.dataframe(reorder_columns(diagnostics_f, ["model", "method", "n", "k", "seed", "ess_per_sec", "cost_per_effective_sample_sec", "wall_time_per_iteration_sec", "acceptance_rate", "projection_failure_rate", "reverse_check_failure_rate"]), use_container_width=True, hide_index=True)

st.subheader("Cost Plots")
tab_ess, tab_cost, tab_accept, tab_diag = st.tabs(["ESS/sec", "Cost per ESS", "Acceptance", "Constraint/projection diagnostics"])
with tab_ess:
    plot_metric(normalized, "ess_per_sec", "ESS/sec by n", "ESS/sec", chart="bars")
with tab_cost:
    plot_metric(normalized, "wall_time_per_ess", "wall time per ESS by n", "seconds")
    plot_metric(normalized, "wall_time_per_iteration", "wall time per iteration by n", "seconds")
with tab_accept:
    plot_metric(normalized, "acceptance_rate", "acceptance rate by n", "acceptance rate", chart="bars")
with tab_diag:
    plot_metric(normalized, "constraint_evals_per_iteration", "constraint evals per iteration by n")
    plot_metric(normalized, "projection_evals_per_iteration", "projection evals per iteration by n")
    plot_metric(ledger_f, "reverse_check_failures", "reverse check failures by n for RATTLE", "failures", method_filter="rattle", chart="bars")
    plot_metric(normalized, "pair_grid_evals_per_iteration", "pair grid evals per iteration by n for Gibbs", method_filter="gibbs", chart="bars")

reference_summary = load_reference_summary(Path(reference_csv)) if overlay_reference else pd.DataFrame()
plot_chain_diagnostics(chains_f, reference_summary, ledger_f)

exports = st.columns(3)
exports[0].download_button(
    "Download raw cost ledger CSV",
    data=ledger_f.to_csv(index=False).encode("utf-8"),
    file_name="cost_ledger_filtered.csv",
    mime="text/csv",
)
exports[1].download_button(
    "Download normalized cost table CSV",
    data=normalized.to_csv(index=False).encode("utf-8"),
    file_name="normalized_cost_table.csv",
    mime="text/csv",
)
exports[2].download_button(
    "Download posterior summaries CSV",
    data=summaries_f.to_csv(index=False).encode("utf-8"),
    file_name="posterior_summaries_filtered.csv",
    mime="text/csv",
)
