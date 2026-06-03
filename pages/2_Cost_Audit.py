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


def apply_filters(df: pd.DataFrame, model_filter: str, method_filter: str, n_filter: str, k_filter: str, seed_filter: int) -> pd.DataFrame:
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
    if "seed" in out.columns:
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
    out["projection_failure_rate"] = out["projection_failures"] / out["projection_evals"].replace(0, np.nan)
    out["pair_grid_evals_per_iteration"] = out["pair_grid_evals"] / iterations

    columns = [
        "method",
        "model",
        "n",
        "k",
        "mu_star",
        "seed",
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
        "projection_failure_rate",
        "acceptance_rate",
        "ess_mu",
        "ess_per_sec",
        "pair_grid_evals_per_iteration",
    ]
    return ensure_columns(out, columns)[columns]


def plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str | None = None, method_filter: str = "both") -> None:
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

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for method, part in plot_df.groupby("method", sort=False):
        part = part.sort_values("n")
        ax.plot(part["n"], part[metric], marker="o", linewidth=2, label=str(method))
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel or metric)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
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


def plot_chain_diagnostics(chain: pd.DataFrame, reference_summary: pd.DataFrame) -> None:
    if chain.empty:
        st.info("chain_samples.csv is missing; chain plots are unavailable.")
        return

    st.subheader("Chain Diagnostics")
    methods = list(chain["method"].dropna().astype(str).unique())
    n_values = sorted(chain["n"].dropna().astype(int).unique())
    col1, col2 = st.columns(2)
    chosen_method = col1.selectbox("chain method", methods, index=0)
    chosen_n = col2.selectbox("chain n", n_values, index=0)
    part = chain[(chain["method"].astype(str).eq(chosen_method)) & (chain["n"].astype(int).eq(int(chosen_n)))]
    if part.empty:
        st.info("No matching chain samples.")
        return

    post = part[~part["is_burn_in"].astype(bool)] if "is_burn_in" in part.columns else part
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(part["iteration"], part["mu"], linewidth=1)
    ax.set_title(f"Trace of mu: {chosen_method}, n={chosen_n}")
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
    for method, method_part in chain[chain["n"].astype(int).eq(int(chosen_n))].groupby("method", sort=False):
        samples = method_part[~method_part["is_burn_in"].astype(bool)]["mu"] if "is_burn_in" in method_part.columns else method_part["mu"]
        ax.hist(samples, bins=40, density=True, alpha=0.35, label=str(method))
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
    normalized = read_cache_csv(str(dashboard_cache_dir), "cost_efficiency_cache.csv")
    summaries = read_cache_csv(str(dashboard_cache_dir), "posterior_summaries_cache.csv")
    chains = read_cache_csv(str(dashboard_cache_dir), "chain_samples_thinned_cache.csv")

    with st.sidebar:
        st.header("Cached Filters")
        model_filter = st.selectbox("model", ["all", "student_t", "logistic", "laplace"], index=0)
        method_filter = st.selectbox("method", ["both", "gibbs", "rattle"], index=0)
        n_filter = st.selectbox("n", ["all", "10", "20", "50"], index=0)
        k_filter = st.selectbox("k", ["all", "1.0", "2.0", "3.0"], index=0)
        seed_filter = st.number_input("seed", min_value=0, value=0, step=1)
        show_raw = st.checkbox("show raw counters", value=True)
        show_normalized = st.checkbox("show normalized counters", value=True)

    ledger_f = apply_filters(ledger, model_filter, method_filter, n_filter, k_filter, int(seed_filter))
    normalized_f = apply_filters(normalized, model_filter, method_filter, n_filter, k_filter, int(seed_filter))
    summaries_f = apply_filters(summaries, model_filter, method_filter, n_filter, k_filter, int(seed_filter))
    chains_f = apply_filters(chains, model_filter, method_filter, n_filter, k_filter, int(seed_filter))

    st.subheader("Cached Audit Data Status")
    status_cols = st.columns(5)
    status_cols[0].metric("results", "dashboard cache")
    status_cols[1].metric("methods", values_status(ledger_f if not ledger_f.empty else chains_f, "method"))
    status_cols[2].metric("n values", values_status(ledger_f if not ledger_f.empty else chains_f, "n"))
    status_cols[3].metric("seeds", values_status(ledger_f if not ledger_f.empty else chains_f, "seed"))
    status_cols[4].metric("run status", smoke_status(ledger_f, chains_f))

    if show_raw:
        st.subheader("Raw Cost Ledger")
        st.dataframe(ledger_f, use_container_width=True)
    if show_normalized:
        st.subheader("Normalized Cost Table")
        st.dataframe(normalized_f, use_container_width=True)

    st.subheader("Gibbs-Specific Counters")
    gibbs_table = ensure_columns(ledger_f[ledger_f.get("method", pd.Series(dtype=str)).astype(str).eq("gibbs")], GIBBS_COLUMNS)
    st.dataframe(gibbs_table[GIBBS_COLUMNS], use_container_width=True)

    st.subheader("RATTLE-Specific Counters")
    rattle_table = ensure_columns(ledger_f[ledger_f.get("method", pd.Series(dtype=str)).astype(str).eq("rattle")], RATTLE_COLUMNS)
    st.dataframe(rattle_table[RATTLE_COLUMNS], use_container_width=True)

    st.subheader("Posterior Summaries")
    st.dataframe(summaries_f, use_container_width=True)

    st.subheader("Cost Plots")
    plot_cols = st.columns(2)
    with plot_cols[0]:
        plot_metric(normalized_f, "ess_per_sec", "ESS/sec vs n", "ESS/sec")
    with plot_cols[1]:
        plot_metric(normalized_f, "wall_time_per_iteration", "wall time per iteration vs n", "seconds")
    with plot_cols[0]:
        plot_metric(normalized_f, "wall_time_per_ess", "wall time per ESS vs n", "seconds")
    with plot_cols[1]:
        plot_metric(normalized_f, "acceptance_rate", "acceptance rate vs n", "acceptance rate")

    plot_chain_diagnostics(chains_f, pd.DataFrame())
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
    seed_filter = st.number_input("seed", min_value=0, value=0, step=1)
    show_raw = st.checkbox("show raw counters", value=True)
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

ledger_f = apply_filters(ledger, model_filter, method_filter, n_filter, k_filter, int(seed_filter))
summaries_f = apply_filters(summaries, model_filter, method_filter, n_filter, k_filter, int(seed_filter))
diagnostics_f = apply_filters(diagnostics, model_filter, method_filter, n_filter, k_filter, int(seed_filter))
chains_f = apply_filters(chains, model_filter, method_filter, n_filter, k_filter, int(seed_filter))
normalized = normalized_cost_table(ledger_f)

if show_raw:
    st.subheader("Raw Cost Ledger")
    st.dataframe(ledger_f, use_container_width=True)

if show_normalized:
    st.subheader("Normalized Cost Table")
    st.dataframe(normalized, use_container_width=True)

st.subheader("Gibbs-Specific Counters")
gibbs_table = ensure_columns(ledger_f[ledger_f.get("method", pd.Series(dtype=str)).astype(str).eq("gibbs")], GIBBS_COLUMNS)
st.dataframe(gibbs_table[GIBBS_COLUMNS], use_container_width=True)

st.subheader("RATTLE-Specific Counters")
rattle_table = ensure_columns(ledger_f[ledger_f.get("method", pd.Series(dtype=str)).astype(str).eq("rattle")], RATTLE_COLUMNS)
st.dataframe(rattle_table[RATTLE_COLUMNS], use_container_width=True)

st.subheader("Posterior Summaries")
st.dataframe(summaries_f, use_container_width=True)

if not diagnostics_f.empty:
    st.subheader("Diagnostic Summary")
    st.dataframe(diagnostics_f, use_container_width=True)

st.subheader("Cost Plots")
plot_cols = st.columns(2)
with plot_cols[0]:
    plot_metric(normalized, "ess_per_sec", "ESS/sec vs n", "ESS/sec")
with plot_cols[1]:
    plot_metric(normalized, "wall_time_per_iteration", "wall time per iteration vs n", "seconds")
with plot_cols[0]:
    plot_metric(normalized, "wall_time_per_ess", "wall time per ESS vs n", "seconds")
with plot_cols[1]:
    plot_metric(normalized, "acceptance_rate", "acceptance rate vs n", "acceptance rate")
with plot_cols[0]:
    plot_metric(normalized, "constraint_evals_per_iteration", "constraint evals per iteration vs n")
with plot_cols[1]:
    plot_metric(normalized, "projection_evals_per_iteration", "projection evals per iteration vs n")
with plot_cols[0]:
    plot_metric(ledger_f, "reverse_check_failures", "reverse check failures vs n for RATTLE", "failures", method_filter="rattle")
with plot_cols[1]:
    plot_metric(normalized, "pair_grid_evals_per_iteration", "pair grid evals per iteration vs n for Gibbs", method_filter="gibbs")

reference_summary = load_reference_summary(Path(reference_csv)) if overlay_reference else pd.DataFrame()
plot_chain_diagnostics(chains_f, reference_summary)

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
