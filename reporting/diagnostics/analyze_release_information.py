"""Analyze MLE-release information loss and latent privacy leakage.

Step 4 consumes trusted MLE-only sampler outputs plus optional dataset-level
full-data posterior outputs. It does not run samplers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "release_information_audit" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results") / "release_information_audit" / ".cache"))
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.run_registry import Runset, load_common_run_outputs


KEYS = ["model", "k_key", "n"]
CASE_KEYS = ["model", "k_key", "n", "dataset_id"]
SUMMARY_KEYS = CASE_KEYS + ["conditioning", "method", "seed", "initialization"]
TAIL_THRESHOLDS = [2.0, 3.0, 5.0, 10.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mle-runset-dir", type=Path, default=Path("results/final_production_v1/"))
    parser.add_argument("--release-runset-dir", type=Path, default=Path("results/release_information_runs/"))
    parser.add_argument("--correctness-dir", type=Path, default=Path("results/sampler_correctness_audit/"))
    parser.add_argument("--efficiency-dir", type=Path, default=Path("results/efficiency_audit/"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/release_information_audit/"))
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-sd", type=float, default=10.0)
    parser.add_argument("--normal-sigma", type=float, default=1.0)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def normalize_k_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


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
    if "method" not in out.columns:
        out["method"] = "unknown"
    return out


def ensure_dataset_id(df: pd.DataFrame) -> pd.DataFrame:
    out = add_k_key(df)
    if "dataset_id" not in out.columns:
        pieces = []
        for row in out.itertuples(index=False):
            data = row._asdict()
            mu_star = data.get("mu_star", 0.0)
            seed = data.get("seed", 0)
            init = data.get("initialization", "unspecified")
            pieces.append(f"mu_star_{float(mu_star):.6g}_seed_{int(seed)}_init_{init}")
        out["dataset_id"] = pieces
    out["dataset_id"] = out["dataset_id"].astype(str)
    return out


def finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def posterior_summary(values: np.ndarray) -> dict[str, float]:
    arr = finite(values)
    if arr.size == 0:
        return {key: np.nan for key in ["mean", "sd", "q01", "q05", "q025", "q50", "q95", "q975", "q99"]}
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q025": float(np.quantile(arr, 0.025)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "q975": float(np.quantile(arr, 0.975)),
        "q99": float(np.quantile(arr, 0.99)),
    }


def empirical_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    a = finite(a)
    b = finite(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    grid = np.linspace(0.0, 1.0, min(max(min(a.size, b.size), 2), 5000))
    return float(np.mean(np.abs(np.quantile(a, grid) - np.quantile(b, grid))))


def empirical_ks(a: np.ndarray, b: np.ndarray) -> float:
    a = finite(a)
    b = finite(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    return float(stats.ks_2samp(a, b).statistic)


def model_label(row: pd.Series) -> str:
    if str(row.get("k_key", "NA")) == "NA":
        return str(row.get("model", ""))
    return f"{row.get('model')} k={row.get('k_key')}"


def predictive_tail_probability(model: str, k_key: str, mu_values: np.ndarray, mu_star: float, threshold: float) -> float:
    mu = finite(mu_values)
    if mu.size == 0:
        return np.nan
    left = mu_star - threshold
    right = mu_star + threshold
    if model == "student_t":
        df = float(k_key)
        probs = stats.t.cdf(left, df=df, loc=mu, scale=1.0) + stats.t.sf(right, df=df, loc=mu, scale=1.0)
    elif model == "logistic":
        probs = stats.logistic.cdf(left, loc=mu, scale=1.0) + stats.logistic.sf(right, loc=mu, scale=1.0)
    elif model == "laplace":
        probs = stats.laplace.cdf(left, loc=mu, scale=1.0) + stats.laplace.sf(right, loc=mu, scale=1.0)
    elif model == "normal_known_var":
        probs = stats.norm.cdf(left, loc=mu, scale=1.0) + stats.norm.sf(right, loc=mu, scale=1.0)
    else:
        return np.nan
    return float(np.mean(probs))


def prior_prob_max_gt(model: str, k_key: str, n: int, threshold: float, mu_star: float = 0.0) -> float:
    left = mu_star - threshold
    right = mu_star + threshold
    if model == "student_t":
        df = float(k_key)
        inside = stats.t.cdf(right, df=df, loc=mu_star, scale=1.0) - stats.t.cdf(left, df=df, loc=mu_star, scale=1.0)
    elif model == "logistic":
        inside = stats.logistic.cdf(right, loc=mu_star, scale=1.0) - stats.logistic.cdf(left, loc=mu_star, scale=1.0)
    elif model == "laplace":
        inside = stats.laplace.cdf(right, loc=mu_star, scale=1.0) - stats.laplace.cdf(left, loc=mu_star, scale=1.0)
    elif model == "normal_known_var":
        inside = stats.norm.cdf(right, loc=mu_star, scale=1.0) - stats.norm.cdf(left, loc=mu_star, scale=1.0)
    else:
        return np.nan
    inside = float(np.clip(inside, 0.0, 1.0))
    return float(1.0 - inside**int(n))


def read_case_table(case_dir: Path, names: list[str]) -> pd.DataFrame:
    for name in names:
        table = read_csv(case_dir / name)
        if not table.empty:
            return table
    return pd.DataFrame()


def case_metadata(case_dir: Path) -> dict[str, Any]:
    meta = read_json(case_dir / "run_metadata.json")
    meta.update(read_json(case_dir / "case_metadata.json"))
    meta.setdefault("case_id", case_dir.name.removeprefix("case_"))
    meta.setdefault("output_dir", str(case_dir))
    return meta


def attach_metadata(df: pd.DataFrame, meta: dict[str, Any], conditioning: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["case_id", "dataset_id", "model", "k", "n", "method", "seed", "initialization", "mu_star"]:
        value = meta.get(col)
        if col not in out.columns:
            out[col] = value
        elif value is not None:
            out[col] = out[col].fillna(value)
    if "dataset_id" not in out.columns or out["dataset_id"].isna().all():
        out["dataset_id"] = meta.get("dataset_id", meta.get("data_seed", meta.get("case_id", "unknown_dataset")))
    out["conditioning"] = conditioning
    return ensure_dataset_id(out)


def load_release_runset(release_dir: Path) -> dict[str, pd.DataFrame]:
    chains: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    observed: list[pd.DataFrame] = []
    latent: list[pd.DataFrame] = []
    if not release_dir.exists():
        return {"chains": pd.DataFrame(), "summaries": pd.DataFrame(), "observed_data": pd.DataFrame(), "latent": pd.DataFrame()}
    for case_dir in sorted(release_dir.glob("case_*")):
        meta = case_metadata(case_dir)
        for conditioning, chain_names, summary_names in [
            ("full_data", ["full_data_chain_samples.csv", "chain_samples_full_data.csv"], ["full_data_posterior_summaries.csv", "posterior_summaries_full_data.csv"]),
            ("mle_only", ["mle_only_chain_samples.csv", "chain_samples_mle_only.csv", "chain_samples.csv"], ["mle_only_posterior_summaries.csv", "posterior_summaries_mle_only.csv", "posterior_summaries.csv"]),
        ]:
            chain = attach_metadata(read_case_table(case_dir, chain_names), meta, conditioning)
            if not chain.empty:
                chains.append(chain)
            summary = attach_metadata(read_case_table(case_dir, summary_names), meta, conditioning)
            if not summary.empty:
                summaries.append(summary)
        data = read_case_table(case_dir, ["observed_data.csv", "data.csv", "x_observed.csv"])
        if not data.empty:
            data = attach_metadata(data, meta, "observed_data")
            observed.append(data)
        lat = attach_metadata(
            read_case_table(case_dir, ["latent_diagnostics.csv", "geometry_diagnostics.csv", "latent_privacy_diagnostics.csv"]),
            meta,
            "mle_only",
        )
        if not lat.empty:
            latent.append(lat)
    return {
        "chains": pd.concat(chains, ignore_index=True, sort=False) if chains else pd.DataFrame(),
        "summaries": pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame(),
        "observed_data": pd.concat(observed, ignore_index=True, sort=False) if observed else pd.DataFrame(),
        "latent": pd.concat(latent, ignore_index=True, sort=False) if latent else pd.DataFrame(),
    }


def load_mle_runset(mle_dir: Path) -> dict[str, pd.DataFrame]:
    if not mle_dir.exists():
        return {"chains": pd.DataFrame(), "summaries": pd.DataFrame(), "latent": pd.DataFrame()}
    outputs = load_common_run_outputs(
        Runset(
            name="final_production_v1",
            run_dir=mle_dir,
            reference_csv=None,
            label="final_production_v1",
            optional=True,
        )
    )
    chains = outputs["tables"].get("chain_samples", pd.DataFrame())
    summaries = outputs["tables"].get("posterior_summaries", pd.DataFrame())
    latent = outputs["tables"].get("latent_diagnostics", pd.DataFrame())
    if latent.empty:
        latent = outputs["tables"].get("geometry_diagnostics", pd.DataFrame())
    for frame in [chains, summaries, latent]:
        if not frame.empty:
            frame["conditioning"] = "mle_only"
    return {
        "chains": ensure_dataset_id(chains) if not chains.empty else chains,
        "summaries": ensure_dataset_id(summaries) if not summaries.empty else summaries,
        "latent": ensure_dataset_id(latent) if not latent.empty else latent,
    }


def summarize_chains(chains: pd.DataFrame) -> pd.DataFrame:
    if chains.empty or "mu" not in chains.columns:
        return pd.DataFrame()
    chains = ensure_dataset_id(chains)
    if "is_burn_in" in chains.columns:
        chains = chains[~chains["is_burn_in"].fillna(False).astype(bool)].copy()
    rows = []
    for keys, part in chains.groupby(SUMMARY_KEYS, dropna=False):
        model, kk, n, dataset_id, conditioning, method, seed, initialization = keys
        vals = part["mu"].to_numpy(dtype=float)
        base = {
            "model": model,
            "k": np.nan if kk == "NA" else float(kk),
            "k_key": kk,
            "n": int(n),
            "dataset_id": dataset_id,
            "conditioning": conditioning,
            "method": method,
            "seed": int(seed),
            "initialization": initialization,
            "draws": int(finite(vals).size),
            "mu_star": float(pd.to_numeric(part.get("mu_star", pd.Series([0.0])), errors="coerce").dropna().iloc[0]) if part.get("mu_star") is not None and pd.to_numeric(part.get("mu_star"), errors="coerce").notna().any() else 0.0,
        }
        stats_row = {**base, **posterior_summary(vals)}
        for threshold in TAIL_THRESHOLDS:
            stats_row[f"predictive_tail_prob_gt_{threshold:g}"] = predictive_tail_probability(
                str(model), str(kk), vals, stats_row["mu_star"], threshold
            )
        rows.append(stats_row)
    return pd.DataFrame(rows)


def normalize_summary_columns(summaries: pd.DataFrame) -> pd.DataFrame:
    if summaries.empty:
        return summaries
    out = ensure_dataset_id(summaries)
    renames = {
        "mean_mu": "mean",
        "sd_mu": "sd",
        "q01_mu": "q01",
        "q05_mu": "q05",
        "q025_mu": "q025",
        "q50_mu": "q50",
        "q95_mu": "q95",
        "q975_mu": "q975",
        "q99_mu": "q99",
    }
    out = out.rename(columns={key: val for key, val in renames.items() if key in out.columns})
    for col in ["conditioning", "dataset_id", "method", "seed", "initialization", "mu_star"]:
        if col not in out.columns:
            out[col] = "mle_only" if col == "conditioning" else np.nan
    if "q05" not in out.columns and {"q025", "q50"}.issubset(out.columns):
        out["q05"] = np.nan
    if "q95" not in out.columns and {"q975", "q50"}.issubset(out.columns):
        out["q95"] = np.nan
    if "q01" not in out.columns:
        out["q01"] = np.nan
    if "q99" not in out.columns:
        out["q99"] = np.nan
    return out


def combine_summaries(chains: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    from_chains = summarize_chains(chains)
    summaries = normalize_summary_columns(summaries)
    if summaries.empty:
        return from_chains
    keep = SUMMARY_KEYS + ["mu_star", "draws", "mean", "sd", "q01", "q05", "q025", "q50", "q95", "q975", "q99"]
    for col in keep:
        if col not in summaries.columns:
            summaries[col] = np.nan
    summaries = summaries[keep].copy()
    if from_chains.empty:
        return summaries
    merged = pd.concat([from_chains, summaries], ignore_index=True, sort=False)
    return merged.sort_values("draws", ascending=False, na_position="last").drop_duplicates(SUMMARY_KEYS)


def chain_lookup(chains: pd.DataFrame) -> dict[tuple, np.ndarray]:
    if chains.empty or "mu" not in chains.columns:
        return {}
    chains = ensure_dataset_id(chains)
    if "is_burn_in" in chains.columns:
        chains = chains[~chains["is_burn_in"].fillna(False).astype(bool)].copy()
    out = {}
    for keys, part in chains.groupby(SUMMARY_KEYS, dropna=False):
        out[keys] = part["mu"].to_numpy(dtype=float)
    return out


def information_loss_table(combined_summary: pd.DataFrame, chains: pd.DataFrame) -> pd.DataFrame:
    if combined_summary.empty:
        return pd.DataFrame()
    full = combined_summary[combined_summary["conditioning"].eq("full_data")].copy()
    mle = combined_summary[combined_summary["conditioning"].eq("mle_only")].copy()
    if full.empty or mle.empty:
        return pd.DataFrame()
    lookup = chain_lookup(chains)
    rows = []
    for _, full_row in full.iterrows():
        candidates = mle[
            mle["model"].eq(full_row["model"])
            & mle["k_key"].eq(full_row["k_key"])
            & mle["n"].eq(full_row["n"])
            & mle["dataset_id"].eq(full_row["dataset_id"])
        ]
        for _, mle_row in candidates.iterrows():
            row = {
                "model": full_row["model"],
                "k": full_row["k"],
                "k_key": full_row["k_key"],
                "n": int(full_row["n"]),
                "dataset_id": full_row["dataset_id"],
                "mle_only_method": mle_row["method"],
                "full_data_method": full_row["method"],
                "mean_difference": mle_row["mean"] - full_row["mean"],
                "sd_ratio_mle_over_full": mle_row["sd"] / full_row["sd"] if full_row["sd"] else np.nan,
                "interval_95_width_ratio": (mle_row["q975"] - mle_row["q025"]) / (full_row["q975"] - full_row["q025"]) if full_row["q975"] > full_row["q025"] else np.nan,
                "interval_90_width_ratio": (mle_row["q95"] - mle_row["q05"]) / (full_row["q95"] - full_row["q05"]) if full_row["q95"] > full_row["q05"] else np.nan,
            }
            for q in ["q01", "q05", "q025", "q50", "q95", "q975", "q99"]:
                row[f"{q}_difference"] = mle_row.get(q, np.nan) - full_row.get(q, np.nan)
            for threshold in TAIL_THRESHOLDS:
                col = f"predictive_tail_prob_gt_{threshold:g}"
                if col in mle_row.index and col in full_row.index:
                    row[f"{col}_difference"] = mle_row[col] - full_row[col]
            full_key = tuple(full_row[key] for key in SUMMARY_KEYS)
            mle_key = tuple(mle_row[key] for key in SUMMARY_KEYS)
            row["wasserstein_mu"] = empirical_wasserstein(lookup.get(full_key, np.array([])), lookup.get(mle_key, np.array([])))
            row["ks_mu"] = empirical_ks(lookup.get(full_key, np.array([])), lookup.get(mle_key, np.array([])))
            quantile_diffs = [abs(row.get(f"{q}_difference", np.nan)) for q in ["q05", "q50", "q95"]]
            row["quantile_distance_score"] = float(np.nanmean(quantile_diffs))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_information_loss(info: pd.DataFrame) -> pd.DataFrame:
    if info.empty:
        return pd.DataFrame()
    metrics = [
        "mean_difference",
        "sd_ratio_mle_over_full",
        "interval_95_width_ratio",
        "interval_90_width_ratio",
        "wasserstein_mu",
        "ks_mu",
        "quantile_distance_score",
    ]
    available = [col for col in metrics if col in info.columns]
    out = info.groupby(KEYS + ["mle_only_method"], dropna=False)[available].agg(["mean", "median", "max", "std"]).reset_index()
    out.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in out.columns]
    out["model_label"] = out.apply(model_label, axis=1)
    return out


def observed_outlier_rows(observed: pd.DataFrame) -> pd.DataFrame:
    if observed.empty:
        return pd.DataFrame()
    observed = ensure_dataset_id(observed)
    rows = []
    value_cols = [col for col in observed.columns if col.startswith("x_") and col.split("_", 1)[1].isdigit()]
    for keys, part in observed.groupby(CASE_KEYS, dropna=False):
        model, kk, n, dataset_id = keys
        mu_star = float(pd.to_numeric(part.get("mu_star", pd.Series([0.0])), errors="coerce").dropna().iloc[0]) if "mu_star" in part.columns else 0.0
        if value_cols:
            vals = part[value_cols].to_numpy(dtype=float).ravel()
        elif "x" in part.columns:
            vals = part["x"].to_numpy(dtype=float)
        else:
            continue
        abs_dev = np.abs(finite(vals) - mu_star)
        if abs_dev.size == 0:
            continue
        row = {"model": model, "k": np.nan if kk == "NA" else float(kk), "k_key": kk, "n": int(n), "dataset_id": dataset_id, "actual_max_abs": float(np.max(abs_dev))}
        for threshold in TAIL_THRESHOLDS:
            row[f"actual_count_gt_{threshold:g}"] = int(np.sum(abs_dev > threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def privacy_leakage_table(latent: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    if latent.empty:
        return pd.DataFrame()
    latent = ensure_dataset_id(latent)
    max_col = None
    for candidate in ["x_abs_max", "max_abs_y", "max_abs_x_minus_mu_star", "latent_max_abs"]:
        if candidate in latent.columns:
            max_col = candidate
            break
    if max_col is None:
        return pd.DataFrame()
    actual = observed_outlier_rows(observed)
    rows = []
    for keys, part in latent.groupby(CASE_KEYS + ["method", "seed", "initialization"], dropna=False):
        model, kk, n, dataset_id, method, seed, initialization = keys
        max_abs = pd.to_numeric(part[max_col], errors="coerce").dropna().to_numpy(dtype=float)
        if max_abs.size == 0:
            continue
        base = {
            "model": model,
            "k": np.nan if kk == "NA" else float(kk),
            "k_key": kk,
            "n": int(n),
            "dataset_id": dataset_id,
            "method": method,
            "seed": int(seed),
            "initialization": initialization,
            "posterior_mean_max_abs": float(np.mean(max_abs)),
            "posterior_q50_max_abs": float(np.quantile(max_abs, 0.50)),
            "posterior_q95_max_abs": float(np.quantile(max_abs, 0.95)),
        }
        actual_row = actual[
            actual["model"].eq(model)
            & actual["k_key"].eq(kk)
            & actual["n"].eq(int(n))
            & actual["dataset_id"].eq(dataset_id)
        ]
        for threshold in TAIL_THRESHOLDS:
            prior = prior_prob_max_gt(str(model), str(kk), int(n), threshold)
            posterior = float(np.mean(max_abs > threshold))
            row = {
                **base,
                "threshold": threshold,
                "prior_prob_M_gt_c": prior,
                "posterior_prob_M_gt_c_given_mle": posterior,
                "leakage_probability_shift": posterior - prior if np.isfinite(prior) else np.nan,
            }
            if not actual_row.empty:
                row["actual_max_abs"] = actual_row.iloc[0]["actual_max_abs"]
                row["actual_count_gt_c"] = actual_row.iloc[0].get(f"actual_count_gt_{threshold:g}", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_privacy(leakage: pd.DataFrame) -> pd.DataFrame:
    if leakage.empty:
        return pd.DataFrame()
    metrics = [
        "prior_prob_M_gt_c",
        "posterior_prob_M_gt_c_given_mle",
        "leakage_probability_shift",
        "posterior_mean_max_abs",
        "posterior_q95_max_abs",
    ]
    out = leakage.groupby(KEYS + ["method", "threshold"], dropna=False)[metrics].agg(["mean", "median", "max", "std"]).reset_index()
    out.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in out.columns]
    out["model_label"] = out.apply(model_label, axis=1)
    return out


def normal_sufficient_baseline(prior_sd: float, sigma: float) -> pd.DataFrame:
    rows = []
    prior_var = prior_sd**2
    for n in [10, 20, 50]:
        post_var = 1.0 / (1.0 / prior_var + n / sigma**2)
        post_sd = float(np.sqrt(post_var))
        width90 = float(stats.norm.ppf(0.95) - stats.norm.ppf(0.05)) * post_sd
        width95 = float(stats.norm.ppf(0.975) - stats.norm.ppf(0.025)) * post_sd
        rows.append(
            {
                "model": "normal_known_var",
                "k": np.nan,
                "k_key": "NA",
                "n": n,
                "posterior_sd_full_data": post_sd,
                "posterior_sd_mle_only": post_sd,
                "sd_ratio_mle_over_full": 1.0,
                "interval_90_width_ratio": 1.0,
                "interval_95_width_ratio": 1.0,
                "wasserstein_mu": 0.0,
                "ks_mu": 0.0,
                "reason": "Sample mean is sufficient for Normal location with known variance.",
                "width90": width90,
                "width95": width95,
            }
        )
    return pd.DataFrame(rows)


def diagnostic_coverage(combined: pd.DataFrame, info: pd.DataFrame, latent: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if combined.empty:
        rows.append({"diagnostic": "posterior_summaries", "available": False, "rows": 0, "note": "No posterior summaries or chains found."})
    else:
        for conditioning in ["mle_only", "full_data"]:
            part = combined[combined["conditioning"].eq(conditioning)]
            rows.append({"diagnostic": f"{conditioning}_posterior", "available": not part.empty, "rows": int(len(part)), "note": "" if not part.empty else "Needed for information-loss comparison."})
    rows.append({"diagnostic": "information_loss_pairs", "available": not info.empty, "rows": int(len(info)), "note": "" if not info.empty else "Requires matched full_data and mle_only posteriors for the same simulated datasets."})
    rows.append({"diagnostic": "latent_privacy_diagnostics", "available": not latent.empty, "rows": int(len(latent)), "note": "" if not latent.empty else "Requires latent_diagnostics or geometry_diagnostics with x_abs_max/max_abs_y."})
    rows.append({"diagnostic": "observed_data", "available": not observed.empty, "rows": int(len(observed)), "note": "Optional, needed to compare posterior latent beliefs to actual dataset outliers."})
    return pd.DataFrame(rows)


def frame_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def write_figures(out_dir: Path, info_summary: pd.DataFrame, leakage_summary: pd.DataFrame, info_rows: pd.DataFrame) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(fig, name: str) -> None:
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    if not info_summary.empty:
        for metric, name, title in [
            ("sd_ratio_mle_over_full_median", "sd_ratio_heatmap.png", "MLE-only / full-data posterior SD"),
            ("wasserstein_mu_median", "wasserstein_heatmap.png", "Wasserstein distance"),
            ("interval_95_width_ratio_median", "interval_width_ratio_heatmap.png", "95% interval width ratio"),
        ]:
            if metric not in info_summary.columns:
                continue
            pivot = info_summary.pivot_table(index="model_label", columns="n", values=metric, aggfunc="median")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
            save(fig, name)

    if not leakage_summary.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, part in leakage_summary.groupby("model_label", dropna=False):
            grouped = part.groupby("threshold")["leakage_probability_shift_median"].median().reset_index()
            ax.plot(grouped["threshold"], grouped["leakage_probability_shift_median"], marker="o", label=label)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("outlier threshold c")
        ax.set_ylabel("posterior minus prior P(M > c)")
        ax.legend(fontsize=7)
        save(fig, "privacy_leakage_probability_shift.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        for label, part in leakage_summary.groupby("model_label", dropna=False):
            grouped = part.groupby("threshold")["posterior_prob_M_gt_c_given_mle_median"].median().reset_index()
            ax.plot(grouped["threshold"], grouped["posterior_prob_M_gt_c_given_mle_median"], marker="o", label=label)
        ax.set_xlabel("outlier threshold c")
        ax.set_ylabel("posterior P(M > c | MLE)")
        ax.legend(fontsize=7)
        save(fig, "posterior_extreme_probability_by_threshold.png")

    if not info_rows.empty:
        examples = info_rows.sort_values("quantile_distance_score", ascending=False).head(12)
        if not examples.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = [f"{r.model} k={r.k_key} n={int(r.n)}" for r in examples.itertuples()]
            ax.barh(labels, examples["quantile_distance_score"])
            ax.set_xlabel("mean absolute q05/q50/q95 shift")
            save(fig, "representative_information_loss_cases.png")
    return paths


def write_report(out_dir: Path, coverage: pd.DataFrame, info_summary: pd.DataFrame, leakage_summary: pd.DataFrame, normal: pd.DataFrame, figures: list[str]) -> None:
    lines = [
        "# MLE Release Information Audit",
        "",
        "Step 4 asks what releasing only `mu_hat` preserves, distorts, and leaks after sampler correctness has been established.",
        "",
        "## Coverage",
        "",
        frame_md(coverage),
        "",
        "## Information Loss",
        "",
    ]
    if info_summary.empty:
        lines.append("Matched full-data vs MLE-only posterior pairs are not available yet. The script is ready for `results/release_information_runs/case_*` outputs.")
    else:
        lines.append(frame_md(info_summary.head(30)))
    lines.extend(["", "## Privacy Leakage", ""])
    if leakage_summary.empty:
        lines.append("Latent privacy diagnostics are not available yet or do not contain a max-|x_i-mu_star| column.")
    else:
        lines.append(frame_md(leakage_summary.head(30)))
    lines.extend(
        [
            "",
            "## Normal Sufficient Baseline",
            "",
            frame_md(normal),
            "",
            "## Interpretation Rules",
            "",
            "- Use only correctness-clean sampler rows for final scientific claims.",
            "- Average information-loss metrics across simulated datasets; single `mu_star=0` runs are illustrative only.",
            "- Treat privacy leakage as prior-to-posterior belief shift about latent extremes, not individual re-identification.",
            "- Student-t k=1,n=10 remains diagnostic-only unless the production correctness audit resolves it.",
            "",
            "## Expected Step 4 Runset Contract",
            "",
            "`results/release_information_runs/case_*` may contain `observed_data.csv`, `full_data_chain_samples.csv`, `full_data_posterior_summaries.csv`, `mle_only_chain_samples.csv`, `mle_only_posterior_summaries.csv`, and `latent_diagnostics.csv` or `geometry_diagnostics.csv`.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in figures)
    (out_dir / "release_information_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    release = load_release_runset(args.release_runset_dir)
    mle = load_mle_runset(args.mle_runset_dir)

    chains = pd.concat([release["chains"], mle["chains"]], ignore_index=True, sort=False)
    summaries = pd.concat([release["summaries"], mle["summaries"]], ignore_index=True, sort=False)
    latent = pd.concat([release["latent"], mle["latent"]], ignore_index=True, sort=False)
    observed = release["observed_data"]

    combined = combine_summaries(chains, summaries)
    info_rows = information_loss_table(combined, chains)
    info_summary = summarize_information_loss(info_rows)
    leakage_rows = privacy_leakage_table(latent, observed)
    leakage_summary = summarize_privacy(leakage_rows)
    normal = normal_sufficient_baseline(args.prior_sd, args.normal_sigma)
    coverage = diagnostic_coverage(combined, info_rows, latent, observed)
    figures = write_figures(args.out_dir, info_summary, leakage_summary, info_rows)

    outputs = {
        "posterior_summary_inputs.csv": combined,
        "information_loss_by_dataset.csv": info_rows,
        "information_loss_summary.csv": info_summary,
        "privacy_leakage_by_case.csv": leakage_rows,
        "privacy_leakage_summary.csv": leakage_summary,
        "sufficient_baseline_normal.csv": normal,
        "diagnostic_coverage.csv": coverage,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / name, index=False)
    write_report(args.out_dir, coverage, info_summary, leakage_summary, normal, figures)

    manifest = {
        "outputs": list(outputs) + ["release_information_report.md"],
        "figures": figures,
        "rows": {name.removesuffix(".csv"): int(len(frame)) for name, frame in outputs.items()},
        "inputs": {
            "mle_runset_dir": str(args.mle_runset_dir),
            "release_runset_dir": str(args.release_runset_dir),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
