"""Summarize all-model reference posterior audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_COLUMNS = ["model", "k", "n", "mu_star", "target_description", "estimator_type", "backend"]
METRICS = ["mean", "sd", "q025", "q50", "q975"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "k" in df.columns:
        df["k"] = pd.to_numeric(df["k"], errors="coerce")
    for column in ["n", "mu_star", "B", "seed", *METRICS, "var", "weighted_ess", "marginal_likelihood_estimate"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def reference_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = ["B", "weighted_ess", "marginal_likelihood_estimate", "mean", "var", "sd", "q025", "q50", "q975"]
    aggregations = {column: ["mean", "std", "min", "max"] for column in numeric if column in df.columns}
    out = df.groupby(GROUP_COLUMNS, dropna=False).agg(aggregations)
    out.columns = ["_".join(part for part in col if part) for col in out.columns.to_flat_index()]
    out = out.reset_index()
    out["source_file"] = df["source_file"].iloc[0] if "source_file" in df.columns and not df.empty else ""
    return out


def raw_key_columns(df: pd.DataFrame) -> list[str]:
    return ["model", "k", "n", "mu_star", "target_description", "B", "seed"]


def kde_differences_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    raw = df[df["estimator_type"].astype(str).eq("raw_weighted_mc")].copy()
    kde = df[df["estimator_type"].astype(str).eq("kde_grid")].copy()
    if raw.empty or kde.empty:
        return pd.DataFrame()
    keys = raw_key_columns(df)
    raw = raw[keys + METRICS].rename(columns={metric: f"raw_{metric}" for metric in METRICS})
    merged = kde.merge(raw, on=keys, how="left")
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"raw_{metric}"]
    merged["rel_sd_error"] = merged["delta_sd"] / merged["raw_sd"].replace(0, np.nan)
    keep = keys + ["backend", "estimator_type", *[f"delta_{metric}" for metric in METRICS], "rel_sd_error"]
    return merged[keep]


def seed_stability(df: pd.DataFrame) -> pd.DataFrame:
    if "seed" not in df.columns:
        return pd.DataFrame()
    aggregations = {metric: ["std", "min", "max"] for metric in METRICS if metric in df.columns}
    out = df.groupby(GROUP_COLUMNS, dropna=False).agg(aggregations)
    out.columns = ["seed_" + "_".join(part for part in col if part) for col in out.columns.to_flat_index()]
    out = out.reset_index()
    for metric in METRICS:
        min_col = f"seed_{metric}_min"
        max_col = f"seed_{metric}_max"
        if min_col in out.columns and max_col in out.columns:
            out[f"seed_{metric}_range"] = out[max_col] - out[min_col]
    return out


def backend_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    kde = df[df["estimator_type"].astype(str).eq("kde_grid")].copy()
    if kde.empty:
        return pd.DataFrame()
    keys = ["model", "k", "n", "mu_star", "target_description", "B", "seed"]
    left = kde[kde["backend"].astype(str).eq("scott")]
    right = kde[kde["backend"].astype(str).eq("SJ_transform")]
    if left.empty or right.empty:
        return pd.DataFrame()
    left = left[keys + METRICS].rename(columns={metric: f"scott_{metric}" for metric in METRICS})
    right = right[keys + METRICS].rename(columns={metric: f"SJ_transform_{metric}" for metric in METRICS})
    merged = left.merge(right, on=keys, how="inner")
    for metric in METRICS:
        merged[f"SJ_minus_scott_{metric}"] = merged[f"SJ_transform_{metric}"] - merged[f"scott_{metric}"]
    return merged


def laplace_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    laplace = df[df["model"].astype(str).eq("laplace")].copy()
    if laplace.empty:
        return pd.DataFrame()
    return reference_summary(laplace)


def warnings_markdown(df: pd.DataFrame) -> str:
    lines = ["# Reference Audit Warnings", ""]
    targets = set(df.get("target_description", pd.Series(dtype=str)).dropna().astype(str))
    if (
        {"deterministic_median_equals_mu_star", "median_interval_contains_mu_star"} <= targets
        or {"deterministic_np_median_equals_mu_star", "median_interval_contains_mu_star"} <= targets
    ):
        lines.append(
            "- Laplace has two valid targets depending on n. Odd-n Gibbs matches deterministic median references; "
            "even-n Gibbs should be compared to `median_interval_contains_mu_star`."
        )
    missing_sj = df[df["estimator_type"].astype(str).eq("kde_grid")].groupby(
        ["model", "k", "n", "target_description"], dropna=False
    )["backend"].apply(lambda x: "SJ_transform" not in set(x.astype(str)))
    for key, is_missing in missing_sj.items():
        if is_missing:
            lines.append(f"- Missing SJ_transform backend for {key}.")
    if len(lines) == 2:
        lines.append("- No target mismatches detected beyond declared model caveats.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = read_reference(args.in_csv)
    reference_summary(df).to_csv(args.out_dir / "reference_summary.csv", index=False)
    backend_sensitivity(df).to_csv(args.out_dir / "backend_sensitivity.csv", index=False)
    seed_stability(df).to_csv(args.out_dir / "seed_stability.csv", index=False)
    laplace_target_summary(df).to_csv(args.out_dir / "laplace_target_summary.csv", index=False)
    (args.out_dir / "reference_warnings.md").write_text(warnings_markdown(df), encoding="utf-8")
    print(f"wrote reference summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
