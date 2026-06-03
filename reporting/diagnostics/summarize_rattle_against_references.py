"""Summarize RATTLE tuning rows against reference-candidate moments.

The primary comparator is the raw weighted-MC moment reference from
``audit_kde_reference.py``. KDE rows are reported as smoothed density estimates
for sensitivity checks, not as a definitive target.

Smoke run:
    python -m reporting.diagnostics.summarize_rattle_against_references \
      --rattle-csv reporting/diagnostic_outputs/rattle_tuning_grid/student2_n20_gram_pair.csv \
      --kde-reference-audit-csv reporting/diagnostic_outputs/kde_reference_audit/student2_n20_reference_audit.csv \
      --raw-mc-B 20000 \
      --kde-backend SJ_transform
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReferenceRow:
    key: tuple[str, str, str]
    mu_star: float
    posterior_mean: float
    posterior_var: float
    source: str
    details: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rattle-csv", type=Path, required=True, help="RATTLE tuning-grid CSV.")
    parser.add_argument(
        "--kde-reference-audit-csv",
        type=Path,
        required=True,
        help="CSV produced by audit_kde_reference.py.",
    )
    parser.add_argument(
        "--gibbs-reference-csv",
        type=Path,
        default=None,
        help="Optional CSV with k,n,seed,mu_star and gibbs_variance columns.",
    )
    parser.add_argument(
        "--kde-backend",
        default="SJ_transform",
        choices=["SJ_transform", "t_abram", "scott", "silverman"],
        help="KDE smoothed density estimate backend to use for tertiary comparison.",
    )
    parser.add_argument(
        "--raw-mc-B",
        type=int,
        default=20000,
        help="B value to use for the raw weighted-MC moment reference.",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of rows/configs to print.")
    parser.add_argument(
        "--mu-star-tol",
        type=float,
        default=1e-8,
        help="Tolerance for warning about mu_star differences after k,n,seed matching.",
    )
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _s(row: dict[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return default if value is None else str(value)


def _b(row: dict[str, Any], key: str) -> bool:
    return _s(row, key).strip().lower() in {"true", "1", "yes", "y"}


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (_num_key(row, "k"), _num_key(row, "n"), _num_key(row, "seed"))


def _num_key(row: dict[str, Any], key: str) -> str:
    value = _f(row, key)
    if np.isfinite(value) and abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    if np.isfinite(value):
        return f"{value:.12g}"
    return _s(row, key)


def _fmt(value: float, digits: int = 6) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}g}"


def _rel_gap(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0:
        return np.nan
    return abs(value - reference) / reference


def _signed_gap(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0:
        return np.nan
    return (value - reference) / reference


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def _nanmedian(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan


def _nanmax(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan


def load_raw_mc_references(rows: list[dict[str, Any]], raw_mc_B: int) -> dict[tuple[str, str, str], ReferenceRow]:
    refs: dict[tuple[str, str, str], ReferenceRow] = {}
    for row in rows:
        if _s(row, "estimator_type") != "raw_weighted_mc":
            continue
        if int(round(_f(row, "B"))) != raw_mc_B:
            continue
        key = _key(row)
        refs[key] = ReferenceRow(
            key=key,
            mu_star=_f(row, "mu_star"),
            posterior_mean=_f(row, "posterior_mean"),
            posterior_var=_f(row, "posterior_var"),
            source=f"raw_weighted_mc_B{raw_mc_B}",
            details=f"weighted_ess={_fmt(_f(row, 'weighted_ess'), 4)}",
        )
    return refs


def load_kde_references(
    rows: list[dict[str, Any]],
    raw_mc_B: int,
    backend: str,
) -> dict[tuple[str, str, str], ReferenceRow]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(round(_f(row, "B"))) != raw_mc_B:
            continue
        if _s(row, "backend") != backend:
            continue
        if _s(row, "estimator_type") not in {"kde_quad", "kde_grid"}:
            continue
        by_key[_key(row)].append(row)

    refs: dict[tuple[str, str, str], ReferenceRow] = {}
    for key, vals in by_key.items():
        quad = [row for row in vals if _s(row, "estimator_type") == "kde_quad"]
        if quad:
            chosen = quad[0]
            details = "kde_quad"
        else:
            chosen = max(vals, key=lambda row: (_f(row, "bound_multiplier"), _f(row, "n_grid")))
            details = f"kde_grid n_grid={_fmt(_f(chosen, 'n_grid'), 0)} bound={_fmt(_f(chosen, 'bound_multiplier'), 0)}"
        refs[key] = ReferenceRow(
            key=key,
            mu_star=_f(chosen, "mu_star"),
            posterior_mean=_f(chosen, "posterior_mean"),
            posterior_var=_f(chosen, "posterior_var"),
            source=f"kde_{backend}",
            details=details,
        )
    return refs


def load_gibbs_references(path: Path | None) -> dict[tuple[str, str, str], ReferenceRow]:
    if path is None:
        return {}
    rows = _read(path)
    refs: dict[tuple[str, str, str], ReferenceRow] = {}
    for row in rows:
        if "gibbs_variance" not in row:
            continue
        refs[_key(row)] = ReferenceRow(
            key=_key(row),
            mu_star=_f(row, "mu_star"),
            posterior_mean=_f(row, "gibbs_mean"),
            posterior_var=_f(row, "gibbs_variance"),
            source="gibbs",
            details=f"iterations={_s(row, 'ref_gibbs_iterations', 'NA')}",
        )
    return refs


def attach_reference_gaps(
    rattle_rows: list[dict[str, Any]],
    raw_refs: dict[tuple[str, str, str], ReferenceRow],
    gibbs_refs: dict[tuple[str, str, str], ReferenceRow],
    kde_refs: dict[tuple[str, str, str], ReferenceRow],
    mu_star_tol: float,
) -> list[str]:
    warnings: list[str] = []
    seen_warnings: set[tuple[str, str, str, str]] = set()
    ref_sets = [
        ("raw_mc", raw_refs),
        ("gibbs", gibbs_refs),
        ("kde_backend", kde_refs),
    ]
    for row in rattle_rows:
        key = _key(row)
        rvar = _f(row, "posterior_var")
        row["merge_key"] = ",".join(key)
        for label, refs in ref_sets:
            ref = refs.get(key)
            if ref is None:
                row[f"{label}_var"] = np.nan
                row[f"rel_gap_{label}"] = np.nan
                if label == "raw_mc":
                    row[f"signed_gap_{label}"] = np.nan
                continue
            row[f"{label}_mean"] = ref.posterior_mean
            row[f"{label}_var"] = ref.posterior_var
            row[f"rel_gap_{label}"] = _rel_gap(rvar, ref.posterior_var)
            row[f"signed_gap_{label}"] = _signed_gap(rvar, ref.posterior_var)
            if label == "raw_mc":
                ci_low = _f(row, "posterior_var_ci_low")
                ci_high = _f(row, "posterior_var_ci_high")
                row["raw_mc_var_inside_rattle_ci"] = (
                    float(ci_low <= ref.posterior_var <= ci_high)
                    if np.isfinite(ci_low) and np.isfinite(ci_high)
                    else np.nan
                )

            r_mu = _f(row, "mu_star")
            if np.isfinite(r_mu) and np.isfinite(ref.mu_star):
                delta = abs(r_mu - ref.mu_star)
                warn_key = (*key, label)
                if delta > mu_star_tol and warn_key not in seen_warnings:
                    warnings.append(
                        f"mu_star differs for {label} match k,n,seed={key}: "
                        f"rattle={r_mu:.12g}, reference={ref.mu_star:.12g}, abs_diff={delta:.3g}"
                    )
                    seen_warnings.add(warn_key)
    return warnings


def config_key(row: dict[str, Any]) -> tuple[bool, float, int]:
    return (_b(row, "include_gram_correction"), _f(row, "eps"), int(round(_f(row, "L"))))


def summarize_configs(rows: list[dict[str, Any]], gap_label: str) -> list[dict[str, Any]]:
    groups: dict[tuple[bool, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[config_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for (include_gram, eps, L), vals in groups.items():
        rel_key = f"rel_gap_{gap_label}"
        signed_key = f"signed_gap_{gap_label}"
        rels = [_f(row, rel_key) for row in vals]
        signed = [_f(row, signed_key) for row in vals]
        med_rel = _nanmedian(rels)
        worst_rel = _nanmax(rels)
        out.append(
            {
                "include_gram_correction": include_gram,
                "eps": eps,
                "L": L,
                "n_seeds": len({_s(row, "seed") for row in vals}),
                f"median_rel_gap_{gap_label}": med_rel,
                f"mean_rel_gap_{gap_label}": _nanmean(rels),
                f"worst_seed_rel_gap_{gap_label}": worst_rel,
                f"median_signed_gap_{gap_label}": _nanmedian(signed),
                f"mean_signed_gap_{gap_label}": _nanmean(signed),
                "mean_posterior_var": _nanmean([_f(row, "posterior_var") for row in vals]),
                f"mean_{gap_label}_var": _nanmean([_f(row, f"{gap_label}_var") for row in vals]),
                "mean_acceptance_rate": _nanmean([_f(row, "acceptance_rate") for row in vals]),
                "mean_reverse_check_failure_rate": _nanmean([_f(row, "reverse_check_failure_rate") for row in vals]),
                "median_reverse_check_failure_rate": _nanmedian([_f(row, "reverse_check_failure_rate") for row in vals]),
                "mean_projection_failure_rate": _nanmean([_f(row, "projection_failure_rate") for row in vals]),
                "mean_ess_per_sec": _nanmean([_f(row, "ess_per_sec") for row in vals]),
                "median_ess_per_sec": _nanmedian([_f(row, "ess_per_sec") for row in vals]),
                "median_posterior_var_se": _nanmedian([_f(row, "posterior_var_se") for row in vals]),
                "raw_mc_var_ci_coverage_rate": _nanmean([_f(row, "raw_mc_var_inside_rattle_ci") for row in vals]),
                f"robust_score_{gap_label}": 0.5 * med_rel + 0.5 * worst_rel,
            }
        )
    score_key = f"robust_score_{gap_label}"
    return sorted(out, key=lambda row: (not np.isfinite(_f(row, score_key)), _f(row, score_key)))


def print_reference_inventory(
    raw_refs: dict[tuple[str, str, str], ReferenceRow],
    gibbs_refs: dict[tuple[str, str, str], ReferenceRow],
    kde_refs: dict[tuple[str, str, str], ReferenceRow],
    kde_backend: str,
) -> None:
    print("\nReference inventory")
    print(f"  raw weighted-MC moment reference rows: {len(raw_refs)}")
    print(f"  Gibbs reference rows: {len(gibbs_refs)}")
    print(f"  KDE smoothed density estimate rows ({kde_backend}): {len(kde_refs)}")
    if raw_refs:
        example = next(iter(raw_refs.values()))
        print(f"  primary source: {example.source} ({example.details})")
    if kde_refs:
        example = next(iter(kde_refs.values()))
        print(f"  tertiary KDE source: {example.source} ({example.details})")


def print_top_rows(rows: list[dict[str, Any]], top: int) -> None:
    valid = [row for row in rows if np.isfinite(_f(row, "rel_gap_raw_mc"))]
    valid.sort(key=lambda row: _f(row, "rel_gap_raw_mc"))
    print(f"\nTop {min(top, len(valid))} individual RATTLE rows versus raw weighted-MC moment reference")
    for row in valid[:top]:
        ci_low = _f(row, "posterior_var_ci_low")
        ci_high = _f(row, "posterior_var_ci_high")
        ci_text = (
            f" var_se={_fmt(_f(row, 'posterior_var_se'), 3)} "
            f"var_ci=[{_fmt(ci_low, 5)},{_fmt(ci_high, 5)}] "
            f"raw_in_ci={_fmt(_f(row, 'raw_mc_var_inside_rattle_ci'), 1)}"
            if np.isfinite(ci_low) and np.isfinite(ci_high)
            else ""
        )
        print(
            "  "
            f"gap={_fmt(_f(row, 'rel_gap_raw_mc'), 4)} "
            f"signed={_fmt(_f(row, 'signed_gap_raw_mc'), 4)} "
            f"gram={_b(row, 'include_gram_correction')} eps={_fmt(_f(row, 'eps'), 3)} "
            f"L={int(round(_f(row, 'L')))} seed={int(round(_f(row, 'seed')))} "
            f"rattle_var={_fmt(_f(row, 'posterior_var'))} raw_var={_fmt(_f(row, 'raw_mc_var'))}{ci_text} "
            f"accept={_fmt(_f(row, 'acceptance_rate'), 3)} "
            f"rev_fail={_fmt(_f(row, 'reverse_check_failure_rate'), 3)} "
            f"proj_fail={_fmt(_f(row, 'projection_failure_rate'), 3)} "
            f"ess/sec={_fmt(_f(row, 'ess_per_sec'), 4)}"
        )


def print_top_configs(configs: list[dict[str, Any]], top: int, gap_label: str, title: str) -> None:
    print(f"\nTop {min(top, len(configs))} configs {title}")
    for row in configs[:top]:
        print(
            "  "
            f"score={_fmt(_f(row, f'robust_score_{gap_label}'), 4)} "
            f"median_gap={_fmt(_f(row, f'median_rel_gap_{gap_label}'), 4)} "
            f"worst_gap={_fmt(_f(row, f'worst_seed_rel_gap_{gap_label}'), 4)} "
            f"mean_signed={_fmt(_f(row, f'mean_signed_gap_{gap_label}'), 4)} "
            f"gram={row['include_gram_correction']} eps={_fmt(row['eps'], 3)} L={row['L']} "
            f"n_seeds={row['n_seeds']} "
            f"mean_var={_fmt(row['mean_posterior_var'])} ref_var={_fmt(row[f'mean_{gap_label}_var'])} "
            f"median_var_se={_fmt(row['median_posterior_var_se'], 3)} "
            f"raw_ci_cover={_fmt(row['raw_mc_var_ci_coverage_rate'], 3)} "
            f"accept={_fmt(row['mean_acceptance_rate'], 3)} "
            f"rev_fail_mean={_fmt(row['mean_reverse_check_failure_rate'], 3)} "
            f"rev_fail_median={_fmt(row['median_reverse_check_failure_rate'], 3)} "
            f"proj_fail={_fmt(row['mean_projection_failure_rate'], 3)} "
            f"ess/sec_mean={_fmt(row['mean_ess_per_sec'], 4)} "
            f"ess/sec_median={_fmt(row['median_ess_per_sec'], 4)}"
        )


def print_gram_comparison(configs: list[dict[str, Any]]) -> None:
    by_gram: dict[bool, list[dict[str, Any]]] = defaultdict(list)
    for row in configs:
        by_gram[bool(row["include_gram_correction"])].append(row)

    print("\nGram versus no-Gram summary relative to raw weighted-MC moment reference")
    for include_gram in [False, True]:
        vals = by_gram.get(include_gram, [])
        if not vals:
            continue
        print(
            "  "
            f"gram={include_gram}: configs={len(vals)} "
            f"median_config_gap={_fmt(_nanmedian([_f(row, 'median_rel_gap_raw_mc') for row in vals]), 4)} "
            f"mean_config_gap={_fmt(_nanmean([_f(row, 'mean_rel_gap_raw_mc') for row in vals]), 4)} "
            f"median_signed={_fmt(_nanmedian([_f(row, 'median_signed_gap_raw_mc') for row in vals]), 4)} "
            f"best_score={_fmt(_f(vals[0], 'robust_score_raw_mc'), 4)}"
        )


def print_dispersion_direction(configs: list[dict[str, Any]], top: int) -> None:
    best = configs[: min(top, len(configs))]
    signed = np.asarray([_f(row, "mean_signed_gap_raw_mc") for row in best], dtype=float)
    signed = signed[np.isfinite(signed)]
    print("\nDispersion direction among best raw-MC-ranked configs")
    if signed.size == 0:
        print("  insufficient raw weighted-MC gaps to assess direction.")
        return
    under = int(np.sum(signed < 0))
    over = int(np.sum(signed > 0))
    if under == signed.size:
        note = "consistently under-dispersed relative to the raw weighted-MC moment reference"
    elif over == signed.size:
        note = "consistently over-dispersed relative to the raw weighted-MC moment reference"
    else:
        note = "mixed in direction relative to the raw weighted-MC moment reference"
    print(f"  {note}: under={under}, over={over}, median_signed_gap={_fmt(float(np.median(signed)), 4)}")


def print_kde_ranking_change(raw_configs: list[dict[str, Any]], kde_configs: list[dict[str, Any]], top: int, backend: str) -> None:
    valid_kde = [row for row in kde_configs if np.isfinite(_f(row, "robust_score_kde_backend"))]
    if not valid_kde:
        print(f"\nNo KDE {backend} ranking available from the supplied audit CSV.")
        return

    print_top_configs(valid_kde, top, "kde_backend", f"versus KDE smoothed density estimate ({backend})")

    kde_rank = {
        (row["include_gram_correction"], row["eps"], row["L"]): idx + 1
        for idx, row in enumerate(valid_kde)
    }
    print(f"\nRanking change when using KDE {backend} instead of raw weighted-MC")
    for idx, row in enumerate(raw_configs[: min(top, len(raw_configs))], start=1):
        key = (row["include_gram_correction"], row["eps"], row["L"])
        print(
            "  "
            f"raw_rank={idx} kde_rank={kde_rank.get(key, 'NA')} "
            f"gram={row['include_gram_correction']} eps={_fmt(row['eps'], 3)} L={row['L']} "
            f"raw_score={_fmt(_f(row, 'robust_score_raw_mc'), 4)} "
            f"kde_score={_fmt(_f(next((x for x in valid_kde if (x['include_gram_correction'], x['eps'], x['L']) == key), {}), 'robust_score_kde_backend'), 4)}"
        )


def main() -> None:
    args = parse_args()
    rattle_rows = _read(args.rattle_csv)
    audit_rows = _read(args.kde_reference_audit_csv)
    raw_refs = load_raw_mc_references(audit_rows, args.raw_mc_B)
    kde_refs = load_kde_references(audit_rows, args.raw_mc_B, args.kde_backend)
    gibbs_refs = load_gibbs_references(args.gibbs_reference_csv)

    print(f"Loaded {len(rattle_rows)} RATTLE rows from {args.rattle_csv}")
    print(f"Loaded {len(audit_rows)} KDE/reference-audit rows from {args.kde_reference_audit_csv}")
    print_reference_inventory(raw_refs, gibbs_refs, kde_refs, args.kde_backend)

    warnings = attach_reference_gaps(rattle_rows, raw_refs, gibbs_refs, kde_refs, args.mu_star_tol)
    if warnings:
        print("\nMerge warnings")
        for warning in warnings[:20]:
            print(f"  {warning}")
        if len(warnings) > 20:
            print(f"  ... {len(warnings) - 20} additional warnings omitted")

    missing_raw = sum(not np.isfinite(_f(row, "raw_mc_var")) for row in rattle_rows)
    if missing_raw:
        print(f"\nWarning: {missing_raw} RATTLE rows did not receive a raw weighted-MC reference.")

    print_top_rows(rattle_rows, args.top)

    raw_configs = summarize_configs(rattle_rows, "raw_mc")
    print_top_configs(raw_configs, args.top, "raw_mc", "versus raw weighted-MC moment reference")
    print_gram_comparison(raw_configs)
    print_dispersion_direction(raw_configs, args.top)

    if gibbs_refs:
        gibbs_configs = summarize_configs(rattle_rows, "gibbs")
        print_top_configs(gibbs_configs, args.top, "gibbs", "versus Gibbs reference candidate")

    kde_configs = summarize_configs(rattle_rows, "kde_backend")
    print_kde_ranking_change(raw_configs, kde_configs, args.top, args.kde_backend)


if __name__ == "__main__":
    main()
