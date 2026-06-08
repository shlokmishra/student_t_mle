"""Analyze cached Gibbs/RATTLE efficiency conditional on sampler correctness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "efficiency_audit" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results") / "efficiency_audit" / ".cache"))
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KEYS = ["model", "k_key", "n", "method"]
SEED_KEYS = ["model", "k_key", "n", "method", "seed", "initialization"]
PRESENTABLE = {"yes", "caveat_only"}

from diagnostics.run_registry import Runset, load_common_run_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runset-dir", type=Path, default=Path("results/final_production_v1/"))
    parser.add_argument("--cost-dir", type=Path, default=Path("results/cost_audit/"))
    parser.add_argument("--correctness-dir", type=Path, default=Path("results/sampler_correctness_audit/"))
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"),
    )
    parser.add_argument(
        "--reference-density-csv",
        type=Path,
        default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models_density_grid.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/efficiency_audit/"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_k_value(value) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def add_k_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "k" not in out.columns:
        out["k"] = np.nan
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    out["k_key"] = out["k"].map(normalize_k_value)
    if "seed" not in out.columns:
        out["seed"] = 0
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").fillna(0).astype(int)
    if "initialization" not in out.columns:
        out["initialization"] = "unspecified"
    out["initialization"] = out["initialization"].fillna("unspecified").astype(str)
    if "mu_star" not in out.columns:
        out["mu_star"] = 0.0
    return out


def numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index)
    return df[column].astype(bool)


def load_cost_table(cost_dir: Path, name: str) -> pd.DataFrame:
    primary = read_csv(cost_dir / name)
    primary["cache_source"] = str(cost_dir) if not primary.empty else ""
    if primary.empty:
        frames = []
        for path in sorted(cost_dir.glob(f"case_*/{name}")):
            frame = read_csv(path)
            if frame.empty:
                continue
            frame["cache_source"] = f"{cost_dir}/case_*"
            frame["case_dir"] = str(path.parent)
            frames.append(frame)
        if frames:
            primary = pd.concat(frames, ignore_index=True, sort=False)

    # The final correctness table uses odd-n Laplace rows. The main requested
    # cost cache contains even-n Laplace rows, while an existing rerun cache
    # under results/results/cost_audit holds the odd-n Laplace runs.
    supplement_dir = ROOT / "results" / "results" / "cost_audit"
    supplement = read_csv(supplement_dir / name)
    if supplement.empty:
        return primary
    supplement = supplement[
        supplement.get("model", pd.Series(dtype=str)).astype(str).eq("laplace")
        & supplement.get("n", pd.Series(dtype=float)).isin([11, 21, 51])
    ].copy()
    if supplement.empty:
        return primary
    supplement["cache_source"] = str(supplement_dir)
    if primary.empty:
        return supplement
    return pd.concat([primary, supplement], ignore_index=True, sort=False)


def attach_correctness(df: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = add_k_key(df)
    verdict_cols = KEYS + ["verdict", "evidence_strength", "main_reason", "main_warning", "safe_to_present"]
    verdict_cols = [col for col in verdict_cols if col in verdicts.columns]
    out = out.merge(verdicts[verdict_cols].drop_duplicates(KEYS), on=KEYS, how="left")
    out["safe_to_present"] = out["safe_to_present"].fillna("no_correctness_filter")
    out["verdict"] = out["verdict"].fillna("no_correctness_filter")
    out["efficiency_set"] = np.select(
        [
            out["safe_to_present"].eq("yes"),
            out["safe_to_present"].eq("caveat_only"),
            out["safe_to_present"].eq("no") | out["verdict"].isin(["unresolved", "not_applicable"]),
        ],
        ["main_efficiency_set", "caveat_efficiency_set", "excluded"],
        default="excluded",
    )
    out["comparison_regime"] = out.apply(comparison_regime, axis=1)
    return out


def comparison_regime(row: pd.Series) -> str:
    model = str(row.get("model", ""))
    method = str(row.get("method", ""))
    n = int(row.get("n")) if pd.notna(row.get("n")) else -1
    k = pd.to_numeric(pd.Series([row.get("k")]), errors="coerce").iloc[0]
    safe = str(row.get("safe_to_present", ""))
    if model == "student_t" and pd.notna(k) and float(k) == 1.0 and n == 10:
        return "diagnostic_only"
    if safe == "caveat_only":
        return "caveat"
    if safe != "yes":
        return "excluded"
    if model == "logistic" and n in {10, 20, 50} and method in {"gibbs", "rattle"}:
        return "main_claim"
    if model == "student_t" and pd.notna(k) and float(k) in {2.0, 3.0} and n in {20, 50} and method in {"gibbs", "rattle"}:
        return "main_claim"
    if model == "laplace" and n in {11, 21, 51} and method == "gibbs":
        return "main_claim"
    return "auxiliary"


def raw_reference(reference: pd.DataFrame, density: pd.DataFrame | None = None) -> pd.DataFrame:
    ref = add_k_key(reference)
    if "estimator_type" not in ref.columns:
        return pd.DataFrame()
    ref = ref[ref["estimator_type"].astype(str).eq("raw_weighted_mc")].copy()
    cols = ["model", "k_key", "n", "mu_star", "mean", "sd", "q025", "q50", "q975", "weighted_ess"]
    out = ref[[col for col in cols if col in ref.columns]].drop_duplicates(["model", "k_key", "n"])
    for q in ["q01", "q05", "q95", "q99"]:
        if q not in out.columns:
            out[q] = np.nan
    if density is not None and not density.empty and {"backend", "mu", "cdf"}.issubset(density.columns):
        dens = add_k_key(density)
        dens = dens[dens["backend"].astype(str).eq("scott")].copy()
        rows = []
        for keys, group in dens.groupby(["model", "k_key", "n", "mu_star", "seed"], dropna=False):
            g = group.sort_values("mu")
            grid = pd.to_numeric(g["mu"], errors="coerce").to_numpy(float)
            cdf = pd.to_numeric(g["cdf"], errors="coerce").to_numpy(float)
            ok = np.isfinite(grid) & np.isfinite(cdf)
            if ok.sum() < 2:
                continue
            grid = grid[ok]
            cdf = np.maximum.accumulate(cdf[ok])
            if cdf[-1] > 0:
                cdf = cdf / cdf[-1]
            rows.append(
                {
                    "model": keys[0],
                    "k_key": keys[1],
                    "n": int(keys[2]),
                    "mu_star": keys[3],
                    "q01": float(np.interp(0.01, cdf, grid)),
                    "q05": float(np.interp(0.05, cdf, grid)),
                    "q95": float(np.interp(0.95, cdf, grid)),
                    "q99": float(np.interp(0.99, cdf, grid)),
                }
            )
        tails = pd.DataFrame(rows)
        if not tails.empty:
            tails = tails.groupby(["model", "k_key", "n", "mu_star"], dropna=False).mean(numeric_only=True).reset_index()
            out = out.merge(tails, on=["model", "k_key", "n", "mu_star"], how="left", suffixes=("", "_density"))
            for q in ["q01", "q05", "q95", "q99"]:
                out[q] = out[f"{q}_density"].combine_first(out[q])
                out = out.drop(columns=[f"{q}_density"], errors="ignore")
    return out


def frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def coalesce_numeric(df: pd.DataFrame, columns: list[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(default, index=df.index, dtype=float)
    for column in columns:
        if column in df.columns:
            out = out.combine_first(pd.to_numeric(df[column], errors="coerce"))
    return out


def cost_decomposition(ledger: pd.DataFrame, diagnostics: pd.DataFrame, verdicts: pd.DataFrame, summaries: pd.DataFrame | None = None) -> pd.DataFrame:
    ledger = attach_correctness(ledger, verdicts)
    if ledger.empty:
        return pd.DataFrame()
    out = ledger.copy()
    if summaries is not None and not summaries.empty:
        summaries = add_k_key(summaries)
        summary_cols = SEED_KEYS + [
            col
            for col in ["ess_mu", "ess_per_sec", "acceptance_rate", "mean_mu", "sd_mu"]
            if col in summaries.columns
        ]
        out = out.merge(summaries[summary_cols].drop_duplicates(SEED_KEYS), on=SEED_KEYS, how="left", suffixes=("", "_summary"))
        for col in ["ess_mu", "ess_per_sec", "acceptance_rate"]:
            if f"{col}_summary" in out.columns:
                out[col] = numeric(out, col).combine_first(numeric(out, f"{col}_summary"))
    iters = numeric(out, "iterations").where(numeric(out, "iterations").gt(0), numeric(out, "num_iterations"))
    iters = iters.replace(0, np.nan)
    burn = numeric(out, "burn_in", 0).fillna(0)
    wall = coalesce_numeric(out, ["wall_time_sec", "wall_time_seconds", "elapsed_seconds", "runtime_seconds"])
    ess = numeric(out, "ess_mu").replace(0, np.nan)

    out["iterations"] = iters
    out["wall_time_sec"] = wall
    out["post_burnin_samples"] = (iters - burn).clip(lower=0)
    out["sec_per_iteration"] = wall / iters
    out["iterations_per_sec"] = iters / wall.replace(0, np.nan)
    out["wall_time_per_post_burnin_sample"] = wall / out["post_burnin_samples"].replace(0, np.nan)
    out["ess_mu_per_sec"] = numeric(out, "ess_per_sec").fillna(ess / wall.replace(0, np.nan))
    out["wall_time_per_ess_mu"] = wall / ess

    for source, target in [
        ("mu_mh_proposals", "mu_mh_proposals_per_iter"),
        ("pair_updates_attempted", "pair_updates_attempted_per_iter"),
        ("pair_updates_completed", "pair_updates_completed_per_iter"),
        ("pair_grid_evals", "pair_grid_evals_per_iter"),
        ("pair_inverse_branch_evals", "pair_inverse_branch_evals_per_iter"),
        ("pair_weight_evals", "pair_weight_evals_per_iter"),
        ("hmc_proposals", "hmc_proposals_per_iter"),
        ("leapfrog_steps", "leapfrog_steps_per_iter"),
        ("constraint_evals", "constraint_evals_per_iter"),
        ("constraint_grad_evals", "constraint_grad_evals_per_iter"),
        ("gram_evals", "gram_evals_per_iter"),
        ("projection_evals", "projection_evals_per_iter"),
        ("forward_newton_iters", "forward_newton_iters_per_iter"),
        ("reverse_newton_iters", "reverse_newton_iters_per_iter"),
    ]:
        out[target] = numeric(out, source, 0).fillna(0) / iters

    proj_evals = numeric(out, "projection_evals").replace(0, np.nan)
    hmc_props = numeric(out, "hmc_proposals").replace(0, np.nan)
    out["projection_failure_rate"] = numeric(out, "projection_failures", 0).fillna(0) / proj_evals
    out["reverse_check_fail_rate"] = numeric(out, "reverse_check_failures", 0).fillna(0) / hmc_props

    if not diagnostics.empty:
        diagnostics = add_k_key(diagnostics)
        diag_cols = SEED_KEYS + ["projection_failure_rate", "reverse_check_failure_rate"]
        diag_cols = [col for col in diag_cols if col in diagnostics.columns]
        out = out.drop(columns=["projection_failure_rate_diag", "reverse_check_failure_rate_diag"], errors="ignore").merge(
            diagnostics[diag_cols].rename(
                columns={
                    "projection_failure_rate": "projection_failure_rate_diag",
                    "reverse_check_failure_rate": "reverse_check_failure_rate_diag",
                }
            ),
            on=SEED_KEYS,
            how="left",
        )
        out["projection_failure_rate"] = out["projection_failure_rate_diag"].combine_first(out["projection_failure_rate"])
        out["reverse_check_fail_rate"] = out["reverse_check_failure_rate_diag"].combine_first(out["reverse_check_fail_rate"])

    ordered = [
        "model",
        "k",
        "k_key",
        "n",
        "method",
        "seed",
        "initialization",
        "safe_to_present",
        "verdict",
        "efficiency_set",
        "comparison_regime",
        "wall_time_sec",
        "iterations",
        "post_burnin_samples",
        "sec_per_iteration",
        "iterations_per_sec",
        "wall_time_per_post_burnin_sample",
        "acceptance_rate",
        "ess_mu",
        "ess_mu_per_sec",
        "wall_time_per_ess_mu",
        "mu_mh_proposals_per_iter",
        "pair_updates_attempted_per_iter",
        "pair_updates_completed_per_iter",
        "pair_grid_evals_per_iter",
        "pair_inverse_branch_evals_per_iter",
        "pair_weight_evals_per_iter",
        "hmc_proposals_per_iter",
        "leapfrog_steps_per_iter",
        "constraint_evals_per_iter",
        "constraint_grad_evals_per_iter",
        "gram_evals_per_iter",
        "projection_evals_per_iter",
        "forward_newton_iters_per_iter",
        "reverse_newton_iters_per_iter",
        "reverse_check_fail_rate",
        "projection_failure_rate",
        "cache_source",
        "source_file",
        "main_warning",
    ]
    for col in ordered:
        if col not in out.columns:
            out[col] = np.nan
    return out[ordered]


def autocorr_fft(values: np.ndarray, max_lag: int = 1000) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.ones(1)
    centered = values - values.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0:
        return np.ones(1)
    n = centered.size
    size = 1 << (2 * n - 1).bit_length()
    freq = np.fft.rfft(centered, size)
    acov = np.fft.irfft(freq * np.conjugate(freq), size)[:n]
    acf = acov / denom
    return acf[: min(max_lag + 1, acf.size)]


def ess_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or float(np.nanstd(values)) <= 0:
        return {"ess": np.nan, "iact": np.nan, "lag1_acf": np.nan, "lag10_acf": np.nan}
    acf = autocorr_fft(values)
    positive = []
    for val in acf[1:]:
        if val <= 0:
            break
        positive.append(float(val))
    iact = max(1.0, 1.0 + 2.0 * np.sum(positive))
    return {
        "ess": float(values.size / iact),
        "iact": float(iact),
        "lag1_acf": float(acf[1]) if acf.size > 1 else np.nan,
        "lag10_acf": float(acf[10]) if acf.size > 10 else np.nan,
    }


def tail_functional(values: np.ndarray, threshold: float, side: str) -> np.ndarray:
    if side == "left":
        return (values < threshold).astype(float)
    return (values > threshold).astype(float)


def functional_ess(chain: pd.DataFrame, cost: pd.DataFrame, reference: pd.DataFrame, verdicts: pd.DataFrame, density: pd.DataFrame | None = None) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    chain = attach_correctness(chain, verdicts)
    ref = raw_reference(reference, density).rename(
        columns={"sd": "raw_sd", "q025": "raw_q025", "q975": "raw_q975", "q01": "raw_q01", "q05": "raw_q05", "q95": "raw_q95", "q99": "raw_q99"}
    )
    cost_cols = SEED_KEYS + ["wall_time_sec"]
    cost_time = add_k_key(cost)[cost_cols].drop_duplicates(SEED_KEYS)
    chain = chain[~bool_series(chain, "is_burn_in")].copy()
    chain = chain.merge(
        ref[["model", "k_key", "n", "raw_sd", "raw_q025", "raw_q975", "raw_q01", "raw_q05", "raw_q95", "raw_q99"]],
        on=["model", "k_key", "n"],
        how="left",
    )
    chain = chain.merge(cost_time, on=SEED_KEYS, how="left")

    rows: list[dict] = []
    for keys, part in chain.groupby(SEED_KEYS, dropna=False, sort=False):
        model, k_key, n, method, seed, initialization = keys
        vals = part["mu"].to_numpy(dtype=float)
        wall = float(part["wall_time_sec"].dropna().iloc[0]) if part["wall_time_sec"].notna().any() else np.nan
        base = {
            "model": model,
            "k": np.nan if k_key == "NA" else float(k_key),
            "k_key": k_key,
            "n": int(n),
            "method": method,
            "seed": int(seed),
            "initialization": initialization,
            "safe_to_present": part["safe_to_present"].iloc[0],
            "verdict": part["verdict"].iloc[0],
            "comparison_regime": part["comparison_regime"].iloc[0],
            "draws": int(vals.size),
            "wall_time_sec": wall,
        }
        functionals = {
            "mu": (vals, "available", np.nan),
            "mu_squared": (vals**2, "available", np.nan),
        }
        for name, quantile, side in [
            ("left_tail_025", "raw_q025", "left"),
            ("right_tail_975", "raw_q975", "right"),
            ("left_tail_05", "raw_q05", "left"),
            ("right_tail_95", "raw_q95", "right"),
            ("left_tail_01", "raw_q01", "left"),
            ("right_tail_99", "raw_q99", "right"),
        ]:
            threshold = float(part[quantile].dropna().iloc[0]) if quantile in part.columns and part[quantile].notna().any() else np.nan
            if np.isfinite(threshold):
                indicator = tail_functional(vals, threshold, side)
                rate = float(indicator.mean())
                status = "unstable_indicator" if rate <= 0.005 or rate >= 0.995 else "available"
                functionals[name] = (indicator, status, threshold)
            else:
                functionals[name] = (np.array([], dtype=float), "missing_raw_quantile", np.nan)

        for functional, (series, status, threshold) in functionals.items():
            row = {**base, "functional": functional, "threshold": threshold, "status": status}
            if status == "missing_raw_quantile":
                row.update({"ess": np.nan, "ess_per_sec": np.nan, "iact": np.nan, "lag1_acf": np.nan, "lag10_acf": np.nan})
            elif status == "unstable_indicator":
                row.update({"ess": np.nan, "ess_per_sec": np.nan, "iact": np.nan, "lag1_acf": np.nan, "lag10_acf": np.nan})
            else:
                stats = ess_stats(series)
                row.update(stats)
                row["ess_per_sec"] = stats["ess"] / wall if np.isfinite(wall) and wall > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def efficiency_summary(chain: pd.DataFrame, cost: pd.DataFrame, reference: pd.DataFrame, verdicts: pd.DataFrame, density: pd.DataFrame | None = None) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    chain = attach_correctness(chain, verdicts)
    ref = raw_reference(reference, density).rename(columns={"mean": "raw_mean", "sd": "raw_sd", "q025": "raw_q025", "q50": "raw_q50", "q975": "raw_q975"})
    cost_cols = SEED_KEYS + ["wall_time_sec", "ess_mu", "ess_mu_per_sec", "wall_time_per_ess_mu", "acceptance_rate"]
    cost_small = add_k_key(cost)[cost_cols].drop_duplicates(SEED_KEYS)
    chain = chain[~bool_series(chain, "is_burn_in")].copy()
    chain = chain.merge(ref[["model", "k_key", "n", "raw_sd"]], on=["model", "k_key", "n"], how="left")
    chain = chain.merge(cost_small, on=SEED_KEYS, how="left")
    rows = []
    for keys, part in chain.groupby(SEED_KEYS, dropna=False, sort=False):
        model, k_key, n, method, seed, initialization = keys
        vals = part["mu"].to_numpy(dtype=float)
        chunks = np.array_split(vals, 4)
        row = {
            "model": model,
            "k": np.nan if k_key == "NA" else float(k_key),
            "k_key": k_key,
            "n": int(n),
            "method": method,
            "seed": int(seed),
            "initialization": initialization,
            "safe_to_present": part["safe_to_present"].iloc[0],
            "verdict": part["verdict"].iloc[0],
            "efficiency_set": part["efficiency_set"].iloc[0],
            "comparison_regime": part["comparison_regime"].iloc[0],
            "draws": int(vals.size),
            "wall_time_sec": part["wall_time_sec"].iloc[0],
            "ess_mu": part["ess_mu"].iloc[0],
            "ess_mu_per_sec": part["ess_mu_per_sec"].iloc[0],
            "wall_time_per_ess_mu": part["wall_time_per_ess_mu"].iloc[0],
            "acceptance_rate": part["acceptance_rate"].iloc[0],
        }
        raw_sd = float(part["raw_sd"].dropna().iloc[0]) if part["raw_sd"].notna().any() else np.nan
        metrics = {"mean": [], "sd": [], "q01": [], "q05": [], "q50": [], "q95": [], "q99": []}
        for i, chunk in enumerate(chunks, start=1):
            stats = {
                "mean": float(np.mean(chunk)),
                "sd": float(np.std(chunk, ddof=1)),
                "q01": float(np.quantile(chunk, 0.01)),
                "q05": float(np.quantile(chunk, 0.05)),
                "q50": float(np.quantile(chunk, 0.50)),
                "q95": float(np.quantile(chunk, 0.95)),
                "q99": float(np.quantile(chunk, 0.99)),
            }
            for metric, value in stats.items():
                row[f"chunk{i}_{metric}"] = value
                metrics[metric].append(value)
        denom = raw_sd if np.isfinite(raw_sd) and raw_sd > 0 else np.nan
        row["max_chunk_mean_diff_over_raw_sd"] = (max(metrics["mean"]) - min(metrics["mean"])) / denom
        row["max_chunk_sd_rel_diff"] = (max(metrics["sd"]) - min(metrics["sd"])) / np.nanmean(metrics["sd"])
        for q in ["q01", "q05", "q95", "q99"]:
            row[f"max_chunk_{q}_diff_over_raw_sd"] = (max(metrics[q]) - min(metrics[q])) / denom
        rows.append(row)
    return pd.DataFrame(rows)


def rattle_movement_diagnostics(chain: pd.DataFrame, cost: pd.DataFrame, reference: pd.DataFrame, verdicts: pd.DataFrame, density: pd.DataFrame | None = None) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    chain = attach_correctness(chain, verdicts)
    chain = chain[~bool_series(chain, "is_burn_in")].copy()
    ref = raw_reference(reference, density).rename(columns={"sd": "raw_sd"})
    chain = chain.merge(ref[["model", "k_key", "n", "raw_sd"]], on=["model", "k_key", "n"], how="left")
    cost_small = add_k_key(cost)[SEED_KEYS + ["wall_time_sec", "acceptance_rate", "ess_mu_per_sec"]].drop_duplicates(SEED_KEYS)
    chain = chain.merge(cost_small, on=SEED_KEYS, how="left")
    rows = []
    for keys, part in chain.groupby(SEED_KEYS, dropna=False, sort=False):
        model, k_key, n, method, seed, initialization = keys
        vals = part["mu"].to_numpy(dtype=float)
        deltas = np.diff(vals)
        abs_delta = np.abs(deltas)
        wall = float(part["wall_time_sec"].dropna().iloc[0]) if part["wall_time_sec"].notna().any() else np.nan
        esjd = float(np.mean(deltas**2)) if deltas.size else np.nan
        rows.append(
            {
                "model": model,
                "k": np.nan if k_key == "NA" else float(k_key),
                "k_key": k_key,
                "n": int(n),
                "method": method,
                "seed": int(seed),
                "initialization": initialization,
                "safe_to_present": part["safe_to_present"].iloc[0],
                "verdict": part["verdict"].iloc[0],
                "comparison_regime": part["comparison_regime"].iloc[0],
                "mean_abs_delta_mu": float(np.mean(abs_delta)) if abs_delta.size else np.nan,
                "median_abs_delta_mu": float(np.median(abs_delta)) if abs_delta.size else np.nan,
                "q95_abs_delta_mu": float(np.quantile(abs_delta, 0.95)) if abs_delta.size else np.nan,
                "esjd_mu": esjd,
                "esjd_mu_per_sec": esjd / wall if np.isfinite(wall) and wall > 0 else np.nan,
                "acceptance_rate": part["acceptance_rate"].iloc[0],
                "ess_mu_per_sec": part["ess_mu_per_sec"].iloc[0],
                "raw_sd": part["raw_sd"].iloc[0],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    gibbs_esjd = out[out["method"].eq("gibbs")][["model", "k_key", "n", "seed", "initialization", "esjd_mu_per_sec"]].rename(columns={"esjd_mu_per_sec": "gibbs_esjd_mu_per_sec"})
    out = out.merge(gibbs_esjd, on=["model", "k_key", "n", "seed", "initialization"], how="left")
    small_relative = out["esjd_mu_per_sec"] < 0.5 * out["gibbs_esjd_mu_per_sec"]
    tiny_median = out["median_abs_delta_mu"] < 0.01 * out["raw_sd"]
    out["high_acceptance_small_move_flag"] = out["method"].eq("rattle") & (out["acceptance_rate"] >= 0.995) & (small_relative | tiny_median)
    return out[out["method"].eq("rattle")].reset_index(drop=True)


def timing_warnings(cost: pd.DataFrame, functional: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        {
            "warning_type": "wall_time_scope_uncertain",
            "severity": "medium",
            "detail": "Cached ledger wall_time_sec is used as recorded; the cache does not prove JAX compile/warmup/cache-loading time is excluded.",
            "recommended_action": "Treat absolute seconds as approximate; emphasize within-cache comparisons and ESS/sec ratios.",
        },
        {
            "warning_type": "tiny_warmup_check_not_run",
            "severity": "info",
            "detail": "No new sampler timing warmup was run; this step is derived from cached audit outputs only.",
            "recommended_action": "Run a targeted no-compile timing microbenchmark later if absolute runtime claims matter.",
        },
        {
            "warning_type": "tail_quantiles_missing",
            "severity": "info",
            "detail": "q01/q05/q95/q99 are taken from the reference density grid when available; raw weighted-MC summary CSV still carries q025/q50/q975 directly.",
            "recommended_action": "For exact raw weighted-MC tail quantiles in tables, regenerate reference summaries with q01/q05/q95/q99.",
        },
    ]
    if cost.get("cache_source", pd.Series(dtype=str)).astype(str).str.contains("results/results/cost_audit", regex=False).any():
        rows.append(
            {
                "warning_type": "mixed_cache_source_for_laplace",
                "severity": "info",
                "detail": "Odd-n Laplace timing rows were supplemented from the existing results/results/cost_audit cache to match final correctness rows.",
                "recommended_action": "Keep Laplace as Gibbs-only baseline; do not compare Laplace RATTLE.",
            }
        )
    if not functional.empty and functional["status"].eq("unstable_indicator").any():
        rows.append(
            {
                "warning_type": "unstable_tail_indicator",
                "severity": "low",
                "detail": "Some tail indicator ESS values were marked not meaningful because the indicator was almost always 0 or 1.",
                "recommended_action": "Use quantile split stability alongside functional ESS for tail behavior.",
            }
        )
    return pd.DataFrame(rows)


def caveat_cases(summary: pd.DataFrame, suspicious: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = summary[summary["safe_to_present"].eq("caveat_only") | summary["safe_to_present"].eq("no")].copy()
    if suspicious.empty:
        out["suspicious_issue_count"] = 0
        return out
    suspicious = add_k_key(suspicious)
    counts = suspicious.groupby(KEYS, dropna=False).size().reset_index(name="suspicious_issue_count")
    return out.merge(counts, on=KEYS, how="left").assign(suspicious_issue_count=lambda d: d["suspicious_issue_count"].fillna(0).astype(int))


def method_winners(cost: pd.DataFrame, functional: pd.DataFrame, summary: pd.DataFrame, rattle_moves: pd.DataFrame) -> pd.DataFrame:
    main = cost[cost["safe_to_present"].eq("yes") & cost["comparison_regime"].eq("main_claim")].copy()
    if main.empty:
        return pd.DataFrame()
    func_available = functional[functional["status"].eq("available")].copy()
    func_min = (
        func_available.groupby(SEED_KEYS, dropna=False)["ess_per_sec"].min().reset_index(name="min_functional_ess_per_sec")
        if not func_available.empty
        else pd.DataFrame(columns=SEED_KEYS + ["min_functional_ess_per_sec"])
    )
    main = main.merge(func_min, on=SEED_KEYS, how="left")
    agg = main.groupby(KEYS, dropna=False).agg(
        k=("k", "first"),
        ess_mu_per_sec=("ess_mu_per_sec", "median"),
        wall_time_per_ess_mu=("wall_time_per_ess_mu", "median"),
        min_functional_ess_per_sec=("min_functional_ess_per_sec", "median"),
        projection_failure_rate=("projection_failure_rate", "max"),
        reverse_check_fail_rate=("reverse_check_fail_rate", "max"),
        main_warning=("main_warning", lambda s: "; ".join(sorted(set(str(x) for x in s.dropna() if str(x) not in {"", "nan", "none"}))) or "none"),
    ).reset_index()
    if rattle_moves.empty or "high_acceptance_small_move_flag" not in rattle_moves.columns:
        move_flags = pd.DataFrame(columns=KEYS + ["high_acceptance_small_move_flag"])
    else:
        move_flags = rattle_moves.groupby(KEYS, dropna=False)["high_acceptance_small_move_flag"].max().reset_index()
    agg = agg.merge(move_flags, on=KEYS, how="left")
    agg["high_acceptance_small_move_flag"] = agg["high_acceptance_small_move_flag"].where(
        agg["high_acceptance_small_move_flag"].notna(), False
    ).astype(bool)

    rows = []
    for keys, part in agg.groupby(["model", "k_key", "n"], dropna=False):
        model, k_key, n = keys
        methods = set(part["method"].astype(str))
        if model == "laplace":
            rows.append(
                {
                    "model": model,
                    "k": np.nan,
                    "n": int(n),
                    "winner_by_mu_ess_per_sec": "gibbs",
                    "winner_by_tail_min_ess_per_sec": "gibbs",
                    "winner_by_wall_time_per_ess": "gibbs",
                    "recommended_efficiency_winner": "gibbs_only",
                    "reason": "Laplace is a Gibbs-only baseline under the correctness filter.",
                    "caveats": "RATTLE is not applicable for Laplace.",
                }
            )
            continue
        if not {"gibbs", "rattle"}.issubset(methods):
            only = ", ".join(sorted(methods))
            rows.append(
                {
                    "model": model,
                    "k": np.nan if k_key == "NA" else float(k_key),
                    "n": int(n),
                    "winner_by_mu_ess_per_sec": "not_comparable",
                    "winner_by_tail_min_ess_per_sec": "not_comparable",
                    "winner_by_wall_time_per_ess": "not_comparable",
                    "recommended_efficiency_winner": "not_comparable",
                    "reason": f"Only one correctness-filtered method is available: {only}.",
                    "caveats": "No Gibbs/RATTLE winner claim.",
                }
            )
            continue

        wide = part.set_index("method")
        gibbs = wide.loc["gibbs"]
        rattle = wide.loc["rattle"]

        def winner_high(metric: str) -> str:
            g, r = float(gibbs[metric]), float(rattle[metric])
            if not np.isfinite(g) or not np.isfinite(r):
                return "insufficient_data"
            ratio = max(g, r) / min(g, r) if min(g, r) > 0 else np.inf
            if ratio < 1.2:
                return "tie/practically similar"
            return "gibbs" if g > r else "rattle"

        def winner_low(metric: str) -> str:
            g, r = float(gibbs[metric]), float(rattle[metric])
            if not np.isfinite(g) or not np.isfinite(r):
                return "insufficient_data"
            ratio = max(g, r) / min(g, r) if min(g, r) > 0 else np.inf
            if ratio < 1.2:
                return "tie/practically similar"
            return "gibbs" if g < r else "rattle"

        mu_winner = winner_high("ess_mu_per_sec")
        tail_winner = winner_high("min_functional_ess_per_sec")
        wall_winner = winner_low("wall_time_per_ess_mu")
        recommended = mu_winner
        caveats = []
        if rattle["high_acceptance_small_move_flag"]:
            caveats.append("RATTLE high-acceptance small-move flag.")
            if recommended == "rattle":
                recommended = "rattle_wins_but_conservative"
        if gibbs["main_warning"] != "none" or rattle["main_warning"] != "none":
            caveats.append("Correctness warning present; avoid a clean winner claim.")
        if rattle["projection_failure_rate"] > 0 or rattle["reverse_check_fail_rate"] > 0:
            caveats.append("RATTLE failure counters are nonzero.")
        rows.append(
            {
                "model": model,
                "k": np.nan if k_key == "NA" else float(k_key),
                "n": int(n),
                "winner_by_mu_ess_per_sec": mu_winner,
                "winner_by_tail_min_ess_per_sec": tail_winner,
                "winner_by_wall_time_per_ess": wall_winner,
                "recommended_efficiency_winner": recommended,
                "reason": f"Median ESS/sec: Gibbs={gibbs['ess_mu_per_sec']:.3g}, RATTLE={rattle['ess_mu_per_sec']:.3g}.",
                "caveats": " ".join(caveats) if caveats else "none",
            }
        )
    return pd.DataFrame(rows)


def label(row: pd.Series) -> str:
    k = "" if row.get("k_key", "NA") == "NA" else f" k={row['k_key']}"
    return f"{row['model']}{k} n={int(row['n'])} {row['method']}"


def write_figures(out_dir: Path, cost: pd.DataFrame, functional: pd.DataFrame, rattle_moves: pd.DataFrame, caveats: pd.DataFrame) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(fig, name: str) -> None:
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    present = cost[cost["safe_to_present"].isin(PRESENTABLE)].copy()
    if not present.empty:
        for metric, name, ylabel in [
            ("ess_mu_per_sec", "ess_per_sec_mu_vs_n.png", "ESS/sec for mu"),
            ("wall_time_per_ess_mu", "wall_time_per_ess_vs_n.png", "wall time / ESS(mu)"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))
            for (model, k_key, method), part in present.groupby(["model", "k_key", "method"], dropna=False):
                grouped = part.groupby("n")[metric].median().reset_index()
                ax.plot(grouped["n"], grouped[metric], marker="o", label=f"{model} k={k_key} {method}")
            ax.set_xlabel("n")
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.legend(fontsize=7, ncols=2)
            save(fig, name)

        fig, ax = plt.subplots(figsize=(9, 5))
        comp_cols = [
            "mu_mh_proposals_per_iter",
            "pair_grid_evals_per_iter",
            "hmc_proposals_per_iter",
            "leapfrog_steps_per_iter",
            "constraint_evals_per_iter",
            "projection_evals_per_iter",
            "gram_evals_per_iter",
            "forward_newton_iters_per_iter",
            "reverse_newton_iters_per_iter",
        ]
        bars = present.groupby(["model", "k_key", "n", "method"], dropna=False)[comp_cols].median().reset_index().head(24)
        bottom = np.zeros(len(bars))
        x = np.arange(len(bars))
        for col in comp_cols:
            vals = bars[col].fillna(0).to_numpy(dtype=float)
            if np.nanmax(vals) <= 0:
                continue
            ax.bar(x, vals, bottom=bottom, label=col.replace("_per_iter", ""))
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([label(r) for _, r in bars.iterrows()], rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("counter per iteration")
        ax.legend(fontsize=6, ncols=2)
        save(fig, "cost_decomposition_stacked_bars.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(present["acceptance_rate"], present["ess_mu_per_sec"], c=present["method"].map({"gibbs": 0, "rattle": 1}), alpha=0.75)
        ax.set_xlabel("acceptance rate")
        ax.set_ylabel("ESS/sec for mu")
        ax.set_yscale("log")
        save(fig, "acceptance_rate_vs_ess_per_sec.png")

    func = functional[functional["status"].eq("available")].copy()
    if not func.empty:
        min_func = func.groupby(["model", "k_key", "n", "method"], dropna=False)["ess_per_sec"].min().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        for (model, k_key, method), part in min_func.groupby(["model", "k_key", "method"], dropna=False):
            grouped = part.groupby("n")["ess_per_sec"].median().reset_index()
            ax.plot(grouped["n"], grouped["ess_per_sec"], marker="o", label=f"{model} k={k_key} {method}")
        ax.set_xlabel("n")
        ax.set_ylabel("minimum functional ESS/sec")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncols=2)
        save(fig, "minimum_functional_ess_per_sec_vs_n.png")

    if not rattle_moves.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for (model, k_key), part in rattle_moves.groupby(["model", "k_key"], dropna=False):
            grouped = part.groupby("n")["esjd_mu_per_sec"].median().reset_index()
            ax.plot(grouped["n"], grouped["esjd_mu_per_sec"], marker="o", label=f"{model} k={k_key}")
        ax.set_xlabel("n")
        ax.set_ylabel("RATTLE ESJD/sec for mu")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncols=2)
        save(fig, "rattle_esjd_per_sec_vs_n.png")

    if not caveats.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        subset = caveats.copy().head(25)
        subset["case"] = subset.apply(lambda r: f"{r['model']} k={r.get('k_key', 'NA')} n={int(r['n'])} {r['method']}", axis=1)
        ax.barh(subset["case"], subset.get("suspicious_issue_count", 0))
        ax.set_xlabel("suspicious issue count")
        ax.set_ylabel("caveat/excluded case")
        save(fig, "caveat_only_cases.png")
    return paths


def write_report(
    out_dir: Path,
    cost: pd.DataFrame,
    functional: pd.DataFrame,
    winners: pd.DataFrame,
    rattle_moves: pd.DataFrame,
    timing: pd.DataFrame,
    caveats: pd.DataFrame,
    figures: list[str],
) -> None:
    verdict_counts = cost.groupby(["efficiency_set", "method"], dropna=False).size().reset_index(name="rows")
    high_move_count = int(rattle_moves.get("high_acceptance_small_move_flag", pd.Series(dtype=bool)).sum()) if not rattle_moves.empty else 0
    clean_winners = winners[~winners.get("recommended_efficiency_winner", pd.Series(dtype=str)).astype(str).isin(["not_comparable"])].copy()
    lines = [
        "# Efficiency Audit",
        "",
        "Efficiency is conditional on sampler correctness. Winner claims use only rows marked `safe_to_present == yes` in `final_sampler_verdict_table.csv`; caveat-only rows are analyzed separately. Raw weighted-MC remains the posterior-summary benchmark; KDE is only a visualization diagnostic.",
        "",
        "## 1. Executive Summary",
        "",
        f"- Correctness-filtered cost rows by set/method: {verdict_counts.to_dict(orient='records')}.",
        f"- Main winner rows produced: {len(winners)}.",
        f"- RATTLE high-acceptance small-move flags: {high_move_count}.",
        "- Student-t k=1,n=10 is excluded from winner claims.",
        "- Laplace RATTLE is not applicable; Laplace appears only as a Gibbs-only baseline.",
        "",
        "## 2. Regimes Included/Excluded",
        "",
        "- Main efficiency set: `safe_to_present == yes`.",
        "- Caveat efficiency set: `safe_to_present == caveat_only`.",
        "- Excluded: unresolved, not applicable, or absent from the correctness filter.",
        "",
        "## 3. Main Efficiency Winners",
        "",
    ]
    if winners.empty:
        lines.append("No winner table could be produced from cached inputs.")
    else:
        lines.append(frame_to_markdown(winners))
    lines.extend(["", "## 4. Gibbs Cost Decomposition", ""])
    gibbs = cost[cost["method"].eq("gibbs") & cost["safe_to_present"].isin(PRESENTABLE)]
    if gibbs.empty:
        lines.append("No presentable Gibbs rows found.")
    else:
        cols = ["model", "k", "n", "ess_mu_per_sec", "wall_time_per_ess_mu", "mu_mh_proposals_per_iter", "pair_grid_evals_per_iter", "pair_updates_completed_per_iter"]
        lines.append(frame_to_markdown(gibbs.groupby(["model", "k_key", "n"], dropna=False)[cols[3:]].median().reset_index()))
    lines.extend(["", "## 5. RATTLE Cost Decomposition", ""])
    rat = cost[cost["method"].eq("rattle") & cost["safe_to_present"].isin(PRESENTABLE)]
    if rat.empty:
        lines.append("No presentable RATTLE rows found.")
    else:
        cols = ["ess_mu_per_sec", "wall_time_per_ess_mu", "hmc_proposals_per_iter", "leapfrog_steps_per_iter", "projection_evals_per_iter", "reverse_check_fail_rate", "projection_failure_rate"]
        lines.append(frame_to_markdown(rat.groupby(["model", "k_key", "n"], dropna=False)[cols].median().reset_index()))
    lines.extend(["", "## 6. Functional ESS and Tail Efficiency", ""])
    if functional.empty:
        lines.append("Functional ESS could not be computed.")
    else:
        status_counts = functional["status"].value_counts(dropna=False).to_dict()
        lines.append(f"Functional ESS status counts: `{status_counts}`.")
        lines.append("Tail indicators include q025/q975 from raw summaries and q01/q05/q95/q99 when the reference density grid is available.")
    lines.extend(["", "## 7. RATTLE High-Acceptance Movement Analysis", ""])
    if rattle_moves.empty:
        lines.append("No RATTLE movement rows found.")
    else:
        cols = ["model", "k", "n", "seed", "acceptance_rate", "median_abs_delta_mu", "esjd_mu_per_sec", "gibbs_esjd_mu_per_sec", "high_acceptance_small_move_flag"]
        lines.append(frame_to_markdown(rattle_moves[cols]))
    lines.extend(["", "## 8. Timing Fairness Warnings", ""])
    lines.append(frame_to_markdown(timing) if not timing.empty else "No timing warnings.")
    lines.extend(["", "## 9. Caveat-Only Cases", ""])
    lines.append(frame_to_markdown(caveats) if not caveats.empty else "No caveat-only cases.")
    lines.extend(
        [
            "",
            "## 10. Recommended Next Runs",
            "",
            "- Add q01/q05/q95/q99 to raw weighted-MC reference summaries if exact 1%/5% tail ESS claims are needed.",
            "- Run a tiny no-compile timing microbenchmark later if absolute wall-time claims matter.",
            "- Keep Student-t k=1,n=10 diagnostic-only until target/mixing mismatch is resolved.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in figures)
    (out_dir / "efficiency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    verdicts = add_k_key(read_csv(args.correctness_dir / "final_sampler_verdict_table.csv"))
    suspicious = read_csv(args.correctness_dir / "suspicious_sampler_cases.csv")
    reference = read_csv(args.reference_csv)
    density = read_csv(args.reference_density_csv)
    if args.runset_dir.exists() and any(args.runset_dir.glob("case_*")):
        runset = Runset(
            name="final_production_v1",
            run_dir=args.runset_dir,
            reference_csv=args.reference_csv,
            label="final_production_v1",
        )
        outputs = load_common_run_outputs(runset)
        ledger = outputs["tables"].get("cost_ledger", pd.DataFrame())
        diagnostics = outputs["tables"].get("diagnostic_summary", pd.DataFrame())
        summaries = outputs["tables"].get("posterior_summaries", pd.DataFrame())
        chains = outputs["tables"].get("chain_samples", pd.DataFrame())
    else:
        ledger = load_cost_table(args.cost_dir, "cost_ledger.csv")
        diagnostics = load_cost_table(args.cost_dir, "diagnostic_summary.csv")
        summaries = load_cost_table(args.cost_dir, "posterior_summaries.csv")
        chains = load_cost_table(args.cost_dir, "chain_samples.csv")

    cost = cost_decomposition(ledger, diagnostics, verdicts, summaries)
    summary = efficiency_summary(chains, cost, reference, verdicts, density)
    func = functional_ess(chains, cost, reference, verdicts, density)
    rattle_moves = rattle_movement_diagnostics(chains, cost, reference, verdicts, density)
    timing = timing_warnings(cost, func, args)
    caveats = caveat_cases(add_k_key(verdicts), suspicious)
    winners = method_winners(cost, func, summary, rattle_moves)

    cost.to_csv(args.out_dir / "cost_decomposition.csv", index=False)
    summary.to_csv(args.out_dir / "efficiency_summary.csv", index=False)
    func.to_csv(args.out_dir / "functional_ess.csv", index=False)
    winners.to_csv(args.out_dir / "method_winners.csv", index=False)
    rattle_moves.to_csv(args.out_dir / "rattle_movement_diagnostics.csv", index=False)
    timing.to_csv(args.out_dir / "timing_warnings.csv", index=False)
    caveats.to_csv(args.out_dir / "caveat_efficiency_cases.csv", index=False)

    figures = write_figures(args.out_dir, cost, func, rattle_moves, caveats)
    write_report(args.out_dir, cost, func, winners, rattle_moves, timing, caveats, figures)

    manifest = {
        "outputs": [
            "efficiency_report.md",
            "efficiency_summary.csv",
            "functional_ess.csv",
            "cost_decomposition.csv",
            "method_winners.csv",
            "rattle_movement_diagnostics.csv",
            "timing_warnings.csv",
            "caveat_efficiency_cases.csv",
        ],
        "figures": figures,
        "rows": {
            "cost_decomposition": int(len(cost)),
            "efficiency_summary": int(len(summary)),
            "functional_ess": int(len(func)),
            "method_winners": int(len(winners)),
            "rattle_movement_diagnostics": int(len(rattle_moves)),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
