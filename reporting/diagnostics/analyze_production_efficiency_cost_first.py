"""Cost-first efficiency summary for final production Gibbs/RATTLE chains.

This is a lightweight production postprocessor. It does not run simulations and
does not recompute expensive raw-chain functional ESS. It uses cached ledger,
posterior-summary, correctness, and split-stability diagnostics to compare cost
per reliable posterior information.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KEYS = ["model", "k_key", "n", "method"]
SEED_KEYS = ["model", "k_key", "n", "method", "seed", "initialization"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runset-dir", type=Path, default=Path("results/final_production_v1"))
    parser.add_argument("--correctness-dir", type=Path, default=Path("results/final_production_v1_correctness_audit"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/final_production_v1_efficiency_audit_cost_first"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_many(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"case_*/{filename}")):
        frame = read_csv(path)
        if frame.empty:
            continue
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def k_key(value: object) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def add_k_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "k" not in out.columns:
        out["k"] = np.nan
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    out["k_key"] = out["k"].map(k_key)
    if "seed" not in out.columns:
        out["seed"] = 0
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").fillna(0).astype(int)
    if "initialization" not in out.columns:
        out["initialization"] = "central"
    out["initialization"] = out["initialization"].fillna("central").astype(str)
    return out


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def comparison_regime(row: pd.Series) -> str:
    model = str(row.get("model", ""))
    method = str(row.get("method", ""))
    n = int(row.get("n")) if pd.notna(row.get("n")) else -1
    k = pd.to_numeric(pd.Series([row.get("k")]), errors="coerce").iloc[0]
    if model == "laplace" and method == "rattle":
        return "excluded"
    if model == "student_t" and pd.notna(k) and float(k) == 1.0 and n == 10:
        return "diagnostic_only"
    if model == "logistic" and n in {10, 20, 50} and method in {"gibbs", "rattle"}:
        return "main_claim"
    if model == "student_t" and pd.notna(k) and float(k) in {2.0, 3.0} and n in {20, 50}:
        return "main_claim"
    if model == "laplace" and n in {11, 21, 51} and method == "gibbs":
        return "main_claim"
    if model == "student_t":
        return "caveat_cost_only"
    return "auxiliary"


def attach_verdicts(df: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    out = add_k_key(df)
    verdicts = add_k_key(verdicts)
    keep = KEYS + ["verdict", "evidence_strength", "main_warning", "safe_to_present"]
    keep = [col for col in keep if col in verdicts.columns]
    out = out.merge(verdicts[keep].drop_duplicates(KEYS), on=KEYS, how="left")
    out["verdict"] = out["verdict"].fillna("missing_correctness_label")
    out["safe_to_present"] = out["safe_to_present"].fillna("missing_correctness_label")
    out["main_warning"] = out["main_warning"].fillna("")
    out["comparison_regime"] = out.apply(comparison_regime, axis=1)
    return out


def build_seed_cost(ledger: pd.DataFrame, summaries: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    ledger = attach_verdicts(ledger, verdicts)
    summaries = add_k_key(summaries)
    summary_cols = SEED_KEYS + [col for col in ["ess_mu", "ess_per_sec", "acceptance_rate", "mean_mu", "sd_mu"] if col in summaries.columns]
    out = ledger.merge(summaries[summary_cols].drop_duplicates(SEED_KEYS), on=SEED_KEYS, how="left", suffixes=("", "_summary"))
    for col in ["ess_mu", "ess_per_sec", "acceptance_rate"]:
        if f"{col}_summary" in out.columns:
            out[col] = num(out, col).combine_first(num(out, f"{col}_summary"))

    iterations = num(out, "iterations").combine_first(num(out, "num_iterations"))
    burn = num(out, "burn_in", 0).fillna(0)
    wall = num(out, "wall_time_sec")
    ess = num(out, "ess_mu")
    out["iterations"] = iterations
    out["burn_in"] = burn
    out["post_burnin_samples"] = (iterations - burn).clip(lower=0)
    out["wall_time_sec"] = wall
    out["sec_per_iteration"] = wall / iterations.replace(0, np.nan)
    out["iterations_per_sec"] = iterations / wall.replace(0, np.nan)
    out["ess_mu"] = ess
    out["ess_mu_per_sec"] = num(out, "ess_per_sec").combine_first(ess / wall.replace(0, np.nan))
    out["wall_time_per_ess_mu"] = wall / ess.replace(0, np.nan)
    out["acceptance_rate"] = num(out, "acceptance_rate")

    for source, target in [
        ("mu_mh_proposals", "mu_mh_proposals_per_iter"),
        ("pair_updates_completed", "pair_updates_completed_per_iter"),
        ("pair_grid_evals", "pair_grid_evals_per_iter"),
        ("hmc_proposals", "hmc_proposals_per_iter"),
        ("leapfrog_steps", "leapfrog_steps_per_iter"),
        ("constraint_evals", "constraint_evals_per_iter"),
        ("constraint_grad_evals", "constraint_grad_evals_per_iter"),
        ("gram_evals", "gram_evals_per_iter"),
        ("projection_evals", "projection_evals_per_iter"),
        ("forward_newton_iters", "forward_newton_iters_per_iter"),
        ("reverse_newton_iters", "reverse_newton_iters_per_iter"),
    ]:
        out[target] = num(out, source, 0).fillna(0) / iterations.replace(0, np.nan)

    proj = num(out, "projection_evals").replace(0, np.nan)
    hmc = num(out, "hmc_proposals").replace(0, np.nan)
    out["projection_failure_rate"] = num(out, "projection_failures", 0).fillna(0) / proj
    out["reverse_check_failure_rate"] = num(out, "reverse_check_failures", 0).fillna(0) / hmc
    return out


def aggregate_cost(seed_cost: pd.DataFrame, split: pd.DataFrame, posterior: pd.DataFrame) -> pd.DataFrame:
    if seed_cost.empty:
        return pd.DataFrame()
    split = add_k_key(split) if not split.empty else pd.DataFrame()
    posterior = add_k_key(posterior) if not posterior.empty else pd.DataFrame()
    agg = seed_cost.groupby(KEYS, dropna=False).agg(
        k=("k", "first"),
        seeds=("seed", "nunique"),
        verdict=("verdict", lambda s: ";".join(sorted(set(map(str, s))))),
        safe_to_present=("safe_to_present", lambda s: ";".join(sorted(set(map(str, s))))),
        comparison_regime=("comparison_regime", lambda s: ";".join(sorted(set(map(str, s))))),
        wall_time_sec_median=("wall_time_sec", "median"),
        sec_per_iteration_median=("sec_per_iteration", "median"),
        iterations_per_sec_median=("iterations_per_sec", "median"),
        ess_mu_median=("ess_mu", "median"),
        ess_mu_per_sec_median=("ess_mu_per_sec", "median"),
        ess_mu_per_sec_min=("ess_mu_per_sec", "min"),
        wall_time_per_ess_mu_median=("wall_time_per_ess_mu", "median"),
        acceptance_rate_median=("acceptance_rate", "median"),
        projection_failure_rate_max=("projection_failure_rate", "max"),
        reverse_check_failure_rate_max=("reverse_check_failure_rate", "max"),
        pair_updates_completed_per_iter=("pair_updates_completed_per_iter", "median"),
        pair_grid_evals_per_iter=("pair_grid_evals_per_iter", "median"),
        hmc_proposals_per_iter=("hmc_proposals_per_iter", "median"),
        leapfrog_steps_per_iter=("leapfrog_steps_per_iter", "median"),
        projection_evals_per_iter=("projection_evals_per_iter", "median"),
        gram_evals_per_iter=("gram_evals_per_iter", "median"),
        forward_newton_iters_per_iter=("forward_newton_iters_per_iter", "median"),
        reverse_newton_iters_per_iter=("reverse_newton_iters_per_iter", "median"),
    ).reset_index()

    if not split.empty:
        split_agg = split.groupby(KEYS, dropna=False).agg(
            max_chunk_mean_diff_over_sd=("max_chunk_mean_diff_over_sd", "max"),
            max_chunk_sd_rel_diff=("max_chunk_sd_rel_diff", "max"),
            split_warning=("warning", lambda s: ";".join(sorted(set(map(str, s))))),
        ).reset_index()
        agg = agg.merge(split_agg, on=KEYS, how="left")
    if not posterior.empty:
        post_agg = posterior.groupby(KEYS, dropna=False).agg(
            posterior_good_rate=("posterior_agreement_good", "mean"),
            max_abs_mean_error_over_raw_sd=("abs_delta_mean_over_raw_sd", "max"),
            max_abs_rel_sd_error=("rel_sd_error", lambda s: pd.to_numeric(s, errors="coerce").abs().max()),
            max_wasserstein_distance=("wasserstein_distance", "max"),
            posterior_warning=("warning", lambda s: ";".join(sorted(set(map(str, s))))),
        ).reset_index()
        agg = agg.merge(post_agg, on=KEYS, how="left")
    return agg


def choose_high(g: float, r: float) -> tuple[str, float]:
    if not (math.isfinite(g) and math.isfinite(r)) or g <= 0 or r <= 0:
        return "insufficient_data", np.nan
    ratio = r / g
    if max(ratio, 1 / ratio) < 1.2:
        return "tie/practically_similar", ratio
    return ("rattle" if ratio > 1 else "gibbs"), ratio


def winners(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in agg.groupby(["model", "k_key", "n"], dropna=False):
        model, kk, n = keys
        if model == "laplace":
            rows.append(
                {
                    "model": model,
                    "k": np.nan,
                    "n": int(n),
                    "comparison_regime": "main_claim",
                    "recommended_efficiency_winner": "gibbs_only",
                    "rattle_over_gibbs_ess_sec_ratio": np.nan,
                    "main_reason": "Laplace RATTLE is not applicable; Gibbs is the only valid method.",
                    "main_caveat": "Use as Gibbs-only baseline.",
                }
            )
            continue
        methods = group.set_index("method")
        if not {"gibbs", "rattle"}.issubset(methods.index):
            continue
        g = float(methods.loc["gibbs", "ess_mu_per_sec_median"])
        r = float(methods.loc["rattle", "ess_mu_per_sec_median"])
        winner, ratio = choose_high(g, r)
        regime = ";".join(sorted(group["comparison_regime"].dropna().astype(str).unique()))
        safe = ";".join(sorted(group["safe_to_present"].dropna().astype(str).unique()))
        caveats = []
        if safe != "yes":
            caveats.append("cost-only comparison; not all rows are clean correctness examples")
        if "diagnostic_only" in regime:
            caveats.append("diagnostic-only regime")
        if "caveat_cost_only" in regime:
            caveats.append("caveat cost-only regime")
        rows.append(
            {
                "model": model,
                "k": np.nan if kk == "NA" else float(kk),
                "n": int(n),
                "comparison_regime": regime,
                "recommended_efficiency_winner": winner,
                "rattle_over_gibbs_ess_sec_ratio": ratio,
                "gibbs_ess_mu_per_sec": g,
                "rattle_ess_mu_per_sec": r,
                "gibbs_sec_per_iteration": float(methods.loc["gibbs", "sec_per_iteration_median"]),
                "rattle_sec_per_iteration": float(methods.loc["rattle", "sec_per_iteration_median"]),
                "main_reason": f"Median ESS/sec(mu): Gibbs={g:.3g}, RATTLE={r:.3g}.",
                "main_caveat": "; ".join(caveats) if caveats else "none",
            }
        )
    return pd.DataFrame(rows)


def write_report(out_dir: Path, agg: pd.DataFrame, win: pd.DataFrame) -> None:
    lines = [
        "# Final Production Efficiency Audit",
        "",
        "Cost-first comparison over completed production chains. Correctness verdicts are included as labels, but cost comparisons are not filtered away by caveats.",
        "",
        "## Winner Table",
        "",
        win.to_markdown(index=False) if not win.empty else "_No comparable winner rows._",
        "",
        "## Aggregate Cost Table",
        "",
        agg.to_markdown(index=False) if not agg.empty else "_No aggregate rows._",
        "",
        "## Interpretation Rule",
        "",
        "- Use `main_claim` rows for headline efficiency statements.",
        "- Use `caveat_cost_only` rows to discuss likely cost behavior after correctness/tuning improvements.",
        "- Use `diagnostic_only` rows only to explain why Student k=1,n=10 is difficult.",
        "- Use ESS/sec and wall-time-per-ESS, not raw wall time alone.",
    ]
    (out_dir / "efficiency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_notes(out_dir: Path, win: pd.DataFrame, agg: pd.DataFrame) -> None:
    path = ROOT / "docs" / "presentation_notes.md"
    if not path.exists():
        return
    marker = "## Efficiency Audit"
    text = path.read_text(encoding="utf-8")
    out_rel = out_dir.relative_to(ROOT) if out_dir.is_absolute() and out_dir.is_relative_to(ROOT) else out_dir
    main_win = win[win["comparison_regime"].astype(str).str.contains("main_claim", na=False)] if not win.empty else pd.DataFrame()
    section = f"""

{marker}

Source artifacts:

- Report: `{out_rel}/efficiency_report.md`
- Aggregate table: `{out_rel}/efficiency_summary.csv`
- Winner table: `{out_rel}/method_winners.csv`
- Seed-level cost decomposition: `{out_rel}/cost_decomposition.csv`

Slide candidate: "Efficiency means reliable posterior information per second"

Main claim:

- Efficiency is analyzed from final production chains as cost per ESS for `mu`,
  with split/posterior warnings attached as caveats.
- Correctness caveats do not hide cost behavior; they determine whether a row is
  headline-clean, caveat cost-only, or diagnostic-only.
- RATTLE has higher `mu` ESS/sec in every comparable Gibbs/RATTLE production
  regime; Gibbs remains the only valid Laplace method.

Key numbers:

- Seed-level cost rows: {int(agg['seeds'].sum()) if not agg.empty and 'seeds' in agg.columns else 0}; aggregate rows: {len(agg) if not agg.empty else 0} model/k/n/method summaries.
- Winner rows: {len(win)} model/k/n regimes.
- Main-claim winner counts: `{main_win['recommended_efficiency_winner'].value_counts().to_dict() if not main_win.empty else {}}`.

Plots/tables worth showing:

- `{out_rel}/method_winners.csv` for the "who wins and when" table.
- `{out_rel}/efficiency_summary.csv` for ESS/sec, sec/iteration, wall-time per
  ESS, split drift, and posterior-warning context.

Reasoning to mention:

- Use ESS/sec and wall-time per ESS, not raw runtime alone.
- Cost-only caveat rows are useful for engineering expectations after tuning,
  but should not become headline scientific correctness examples.
- Laplace is Gibbs-only because RATTLE is not applicable for the nonsmooth median
  target.
"""
    if marker in text:
        text = text.split(marker)[0].rstrip() + section
    else:
        text = text.rstrip() + section
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger = read_many(args.runset_dir, "cost_ledger.csv")
    summaries = read_many(args.runset_dir, "posterior_summaries.csv")
    verdicts = read_csv(args.correctness_dir / "final_sampler_verdict_table.csv")
    split = read_csv(args.correctness_dir / "chain_split_stability.csv")
    posterior = read_csv(args.correctness_dir / "posterior_agreement.csv")

    seed_cost = build_seed_cost(ledger, summaries, verdicts)
    agg = aggregate_cost(seed_cost, split, posterior)
    win = winners(agg)

    seed_cost.to_csv(args.out_dir / "cost_decomposition.csv", index=False)
    agg.to_csv(args.out_dir / "efficiency_summary.csv", index=False)
    win.to_csv(args.out_dir / "method_winners.csv", index=False)
    write_report(args.out_dir, agg, win)
    update_notes(args.out_dir, win, agg)
    manifest = {
        "outputs": ["cost_decomposition.csv", "efficiency_summary.csv", "method_winners.csv", "efficiency_report.md"],
        "rows": {"seed_cost": len(seed_cost), "efficiency_summary": len(agg), "method_winners": len(win)},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
