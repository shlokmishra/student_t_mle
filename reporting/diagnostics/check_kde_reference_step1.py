"""Focused Step 1 sanity checks for KDE/reference posterior machinery.

This script does not replace ``audit_kde_reference.py``. It runs small,
non-flaky checks around the existing posterior, moment, and KDE backend code.

Example:
    python -m reporting.diagnostics.check_kde_reference_step1 --B 1000 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kde_ref.moments import raw_weighted_posterior_moments
from kde_ref.posterior import get_normalized_posterior_pdf, validate_posterior_1d
from kde_ref.reference_adapter import (
    DEFAULT_AUDIT_DIR,
    build_reference_summaries_from_samples,
    load_or_simulate_mle_errors,
)


BACKENDS = ("scott", "SJ_transform", "t_abram")


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=2.0, help="Student-t degrees of freedom.")
    parser.add_argument("--n-values", type=_ints, default=[10, 20, 50], help="Comma-separated sample sizes.")
    parser.add_argument("--B", type=int, default=1000, help="Number of simulated centered MLE errors.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic MLE-simulation seed.")
    parser.add_argument("--mu-star", type=float, default=0.0, help="Observed MLE value for comparisons.")
    parser.add_argument("--positive-mu-star", type=float, default=1.0, help="Positive observed MLE for sign check.")
    parser.add_argument("--prior-mean", type=float, default=0.0, help="Normal prior mean.")
    parser.add_argument("--prior-std", type=float, default=10.0, help="Normal prior standard deviation.")
    parser.add_argument("--backends", type=_strings, default=list(BACKENDS), help="Comma-separated KDE backends.")
    parser.add_argument("--grid-size", type=int, default=1000, help="Grid size for KDE summaries and normalization checks.")
    parser.add_argument("--bound-multiplier", type=float, default=5.0, help="Grid bound multiplier.")
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR, help="MLE-error cache location.")
    parser.add_argument("--include-quad", action="store_true", help="Also run KDE-quad summaries.")
    parser.add_argument("--mean-tol", type=float, default=0.15, help="Loose absolute tolerance for sign-check mean near zero.")
    parser.add_argument("--norm-tol", type=float, default=0.03, help="Loose absolute normalization tolerance.")
    parser.add_argument("--relative-warning-tol", type=float, default=0.35, help="Warn when KDE-grid differs this much from raw weighted-MC.")
    return parser.parse_args()


def _status(ok: bool) -> str:
    return "OK" if ok else "WARN"


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(float(b)), 1e-12)
    return abs(float(a) - float(b)) / denom


def sign_check(args: argparse.Namespace, z_samples: np.ndarray, backend: str) -> list[tuple[str, bool]]:
    params = {
        "k": args.k,
        "n": args.n_values[0],
        "prior_mean": args.prior_mean,
        "prior_std": args.prior_std,
        "kde_bw_method": backend,
    }
    pdf0 = get_normalized_posterior_pdf(0.0, params, z_samples, n_grid=args.grid_size)
    grid = np.linspace(-5.0, 5.0, args.grid_size)
    vals0 = np.maximum(pdf0(grid), 0.0)
    mean0 = float(np.trapezoid(grid * vals0, grid) / max(np.trapezoid(vals0, grid), 1e-300))

    pdf_pos = get_normalized_posterior_pdf(args.positive_mu_star, params, z_samples, n_grid=args.grid_size)
    vals_pos = np.maximum(pdf_pos(grid + args.positive_mu_star), 0.0)
    grid_pos = grid + args.positive_mu_star
    mean_pos = float(np.trapezoid(grid_pos * vals_pos, grid_pos) / max(np.trapezoid(vals_pos, grid_pos), 1e-300))

    near_zero = abs(mean0) <= args.mean_tol
    shifted_positive = mean_pos > 0.25 * args.positive_mu_star
    print("\nSign check: likelihood should use f_Z(mu_star - mu)")
    print(f"  backend={backend} mu_star=0 posterior mean={mean0:.6g} {_status(near_zero)}")
    print(f"  backend={backend} mu_star={args.positive_mu_star:g} posterior mean={mean_pos:.6g} {_status(shifted_positive)}")
    return [("sign_mu_star_zero", near_zero), ("sign_positive_mu_star", shifted_positive)]


def backend_and_normalization_checks(args: argparse.Namespace, z_samples: np.ndarray, n: int) -> list[tuple[str, bool]]:
    checks = []
    print(f"\nBackend availability and normalization, n={n}")
    for backend in args.backends:
        try:
            params = {
                "k": args.k,
                "n": n,
                "prior_mean": args.prior_mean,
                "prior_std": args.prior_std,
                "kde_bw_method": backend,
            }
            pdf, info = get_normalized_posterior_pdf(
                args.mu_star,
                params,
                z_samples,
                n_grid=args.grid_size,
                return_info=True,
            )
            lo = min(args.prior_mean - args.bound_multiplier * args.prior_std, args.mu_star - 5.0)
            hi = max(args.prior_mean + args.bound_multiplier * args.prior_std, args.mu_star + 5.0)
            integral = validate_posterior_1d(pdf, lo=lo, hi=hi, n_grid=args.grid_size)
            ok = abs(integral - 1.0) <= args.norm_tol
            print(
                f"  {backend}: ran, normalization={integral:.6g}, "
                f"normalization_constant={info['normalization_constant']:.6g} {_status(ok)}"
            )
            checks.append((f"backend_{backend}", True))
            checks.append((f"normalization_{backend}", ok))
        except Exception as exc:
            print(f"  {backend}: failed cleanly with {type(exc).__name__}: {exc}")
            checks.append((f"backend_{backend}", False))
    return checks


def raw_vs_kde_checks(args: argparse.Namespace, z_samples: np.ndarray, n: int) -> list[tuple[str, bool]]:
    print(f"\nRaw weighted-MC reference vs KDE summaries, n={n}")
    summaries = build_reference_summaries_from_samples(
        z_samples=z_samples,
        k=args.k,
        n=n,
        mu_star=args.mu_star,
        prior_mean=args.prior_mean,
        prior_std=args.prior_std,
        B=args.B,
        seed=args.seed,
        backends=args.backends,
        include_raw=True,
        include_kde_grid=True,
        include_kde_quad=args.include_quad,
        grid_size=args.grid_size,
        bound_multiplier=args.bound_multiplier,
    )
    raw = summaries[summaries["estimator_type"].eq("raw_weighted_mc")].iloc[0]
    print(
        "  raw weighted-MC reference: "
        f"mean={raw['mean']:.6g} sd={raw['sd']:.6g} "
        f"q025={raw['q025']:.6g} q50={raw['q50']:.6g} q975={raw['q975']:.6g}"
    )
    checks = []
    compare_cols = ("mean", "sd", "q025", "q50", "q975")
    for _, row in summaries[~summaries["estimator_type"].eq("raw_weighted_mc")].iterrows():
        diffs = {col: _rel_diff(row[col], raw[col]) for col in compare_cols}
        too_large = max(diffs.values()) > args.relative_warning_tol
        print(
            f"  {row['estimator_type']} {row['backend']}: "
            f"mean_diff={row['mean'] - raw['mean']:+.6g} "
            f"sd_diff={row['sd'] - raw['sd']:+.6g} "
            f"max_rel_diff={max(diffs.values()):.3g} {_status(not too_large)}"
        )
        # These are sensitivity warnings, not hard failures. The raw
        # weighted-MC row remains the benchmark for moments and quantiles.
        checks.append((f"raw_vs_{row['estimator_type']}_{row['backend']}_n{n}", True))
    return checks


def main() -> None:
    args = parse_args()
    all_checks: list[tuple[str, bool]] = []
    samples_by_n = {
        n: load_or_simulate_mle_errors(k=args.k, n=n, B=args.B, seed=args.seed, audit_dir=args.audit_dir)
        for n in args.n_values
    }

    first_n = args.n_values[0]
    raw0 = raw_weighted_posterior_moments(samples_by_n[first_n], 0.0, args.prior_mean, args.prior_std)
    print("Step 1 KDE/reference sanity check")
    print(
        f"  k={args.k:g} n_values={args.n_values} B={args.B} seed={args.seed} "
        f"raw weighted-MC reference mean at mu_star=0 for n={first_n}: {raw0['posterior_mean']:.6g}"
    )

    all_checks.extend(sign_check(args, samples_by_n[first_n], args.backends[0]))
    for n, z_samples in samples_by_n.items():
        all_checks.extend(backend_and_normalization_checks(args, z_samples, n))
        all_checks.extend(raw_vs_kde_checks(args, z_samples, n))

    failed = [name for name, ok in all_checks if not ok]
    print("\nSummary")
    print(f"  checks={len(all_checks)} hard_warnings={len(failed)}")
    if failed:
        print("  warnings:")
        for name in failed:
            print(f"    {name}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
