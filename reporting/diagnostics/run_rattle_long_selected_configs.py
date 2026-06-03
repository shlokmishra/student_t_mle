"""Run longer RATTLE checks for selected Student-t tuning configurations.

The default configuration list is intentionally compact and centered on the
Student-2 n=20 sensitivity audit. Defaults are smoke-sized; pass larger
``--iterations`` and ``--burnin`` for confirmation runs.

Smoke run:
    python -m reporting.diagnostics.run_rattle_long_selected_configs \
      --iterations 100 --burnin 20 --seeds 0 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import jax.random as random
import numpy as np

from analysis import ess_per_second
from models import loc_student, loc_student_rattle
from reporting.diagnostics.student_t_geometry import posterior_summary

OUT_CSV = Path("reporting/diagnostic_outputs/rattle_long_runs/student2_n20_selected_configs.csv")

DEFAULT_CONFIGS = [
    (False, 0.05, 5),
    (True, 0.03, 5),
    (True, 0.05, 3),
    (True, 0.02, 5),
]


def _bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = text.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text}")


def _ints(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif ":" in token:
            lo, hi = token.split(":", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token))
    return out


def _parse_configs(text: str) -> list[tuple[bool, float, int]]:
    configs: list[tuple[bool, float, int]] = []
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        parts = [part.strip() for part in token.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "Configs must be comma-separated include_gram:eps:L triples, e.g. false:0.05:5,true:0.03:5"
            )
        configs.append((_bool(parts[0]), float(parts[1]), int(parts[2])))
    return configs


def parse_args() -> argparse.Namespace:
    default_configs = ",".join(f"{str(g).lower()}:{eps}:{L}" for g, eps, L in DEFAULT_CONFIGS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=2.0, help="Student-t degrees of freedom.")
    parser.add_argument("--n", type=int, default=20, help="Sample size.")
    parser.add_argument("--seeds", type=_ints, default=[0], help="Comma-separated seeds or ranges, e.g. 0-9.")
    parser.add_argument("--configs", type=_parse_configs, default=DEFAULT_CONFIGS, help=f"Selected include_gram:eps:L triples. Default: {default_configs}")
    parser.add_argument("--iterations", type=int, default=100, help="RATTLE iterations; smoke-sized by default.")
    parser.add_argument("--burnin", type=int, default=20, help="Burn-in iterations.")
    parser.add_argument("--mu-true", type=float, default=2.0, help="Data-generating location.")
    parser.add_argument(
        "--mu-star-reference-csv",
        type=Path,
        default=None,
        help="Optional CSV with k,n,seed,mu_star. When present, use the stored observed MLE instead of resampling data.",
    )
    parser.add_argument("--prior-mean", type=float, default=0.0, help="Normal prior mean.")
    parser.add_argument("--prior-std", type=float, default=10.0, help="Normal prior standard deviation.")
    parser.add_argument("--proposal-std-mu", type=float, default=0.9, help="RW-MH proposal sd for mu.")
    parser.add_argument("--projection-tol", type=float, default=1e-10, help="RATTLE projection tolerance.")
    parser.add_argument("--reverse-position-tol", type=float, default=2e-2, help="Reverse position check tolerance.")
    parser.add_argument("--reverse-momentum-tol", type=float, default=2e-2, help="Reverse momentum check tolerance.")
    parser.add_argument("--num-batches", type=int, default=30, help="Number of contiguous batches for MC SE estimates.")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV, help="Output CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run completed configs.")
    return parser.parse_args()


KEY_FIELDS = [
    "model",
    "k",
    "n",
    "seed",
    "eps",
    "L",
    "projection_tol",
    "reverse_position_tol",
    "reverse_momentum_tol",
    "include_gram_correction",
    "iterations",
    "burnin",
]


def _key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def _completed(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {_key(row) for row in csv.DictReader(handle)}


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing = next(csv.reader(handle), None)
        if existing != fieldnames:
            raise ValueError(
                f"Existing CSV schema differs from current output schema: {path}. "
                "Use --overwrite or choose a new --out-csv."
            )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _case_key(k: float, n: int, seed: int) -> tuple[str, str, str]:
    return (f"{float(k):.12g}", str(int(n)), str(int(seed)))


def load_mu_star_refs(path: Path | None) -> dict[tuple[str, str, str], float]:
    if path is None:
        return {}
    refs: dict[tuple[str, str, str], float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if "mu_star" not in row:
                continue
            try:
                key = _case_key(float(row["k"]), int(float(row["n"])), int(float(row["seed"])))
                refs.setdefault(key, float(row["mu_star"]))
            except (TypeError, ValueError):
                continue
    return refs


def batch_means_uncertainty(samples: np.ndarray, num_batches: int) -> dict[str, float]:
    samples = np.asarray(samples, dtype=float)
    if samples.size < 4:
        return {
            "posterior_mean_se": np.nan,
            "posterior_var_se": np.nan,
            "posterior_var_ci_low": np.nan,
            "posterior_var_ci_high": np.nan,
            "uncertainty_num_batches": 0,
            "uncertainty_batch_size": 0,
        }
    batches = max(2, min(num_batches, samples.size // 2))
    batch_size = samples.size // batches
    trimmed = samples[: batches * batch_size]
    blocks = trimmed.reshape(batches, batch_size)
    batch_means = np.mean(blocks, axis=1)
    batch_vars = np.var(blocks, axis=1, ddof=1) if batch_size > 1 else np.var(blocks, axis=1)
    mean_se = float(np.std(batch_means, ddof=1) / np.sqrt(batches))
    var_se = float(np.std(batch_vars, ddof=1) / np.sqrt(batches))
    var_hat = float(np.var(samples, ddof=1)) if samples.size > 1 else 0.0
    return {
        "posterior_mean_se": mean_se,
        "posterior_var_se": var_se,
        "posterior_var_ci_low": max(0.0, var_hat - 1.96 * var_se),
        "posterior_var_ci_high": var_hat + 1.96 * var_se,
        "uncertainty_num_batches": batches,
        "uncertainty_batch_size": batch_size,
    }


def run_one(
    args: argparse.Namespace,
    seed: int,
    include_gram: bool,
    eps: float,
    L: int,
    mu_star_refs: dict[tuple[str, str, str], float],
) -> dict[str, Any]:
    params = {
        "n": args.n,
        "k": args.k,
        "num_iterations_T": args.iterations,
        "prior_mean": args.prior_mean,
        "prior_std": args.prior_std,
        "proposal_std_mu": args.proposal_std_mu,
        "rattle_step_size": eps,
        "rattle_num_steps": L,
        "rattle_proj_tol": args.projection_tol,
        "rattle_reverse_position_tol": args.reverse_position_tol,
        "rattle_reverse_momentum_tol": args.reverse_momentum_tol,
        "rattle_include_gram_correction": include_gram,
        "rattle_proj_damping": 1.0,
        "rattle_proj_line_search": True,
        "rattle_proj_init_strategy": "trial",
        "rattle_relaxed_position_tol": 5e-2,
        "rattle_relaxed_momentum_tol": 5e-2,
    }
    key = random.PRNGKey(seed)
    _, key_data, key_rattle = random.split(key, 3)
    case_key = _case_key(args.k, args.n, seed)
    if case_key in mu_star_refs:
        mu_star = float(mu_star_refs[case_key])
        mu_star_source = "reference_csv"
    else:
        data = np.asarray(loc_student.sample_data(key_data, params, loc=args.mu_true), dtype=float)
        mu_star = float(loc_student.get_mle(data, params))
        mu_star_source = "simulated_data"

    start = time.time()
    result = loc_student_rattle.run_rattle(key_rattle, mu_star, params, verbose=False)
    runtime = time.time() - start
    chain = np.asarray(result["mu_chain"], dtype=float)
    post = chain[min(args.burnin, chain.size - 1) :]
    summ = posterior_summary(post, prefix="posterior_")
    uncertainty = batch_means_uncertainty(post, args.num_batches)
    ess, ess_sec = ess_per_second(post, runtime)
    diag = dict(result["projection_diagnostics"])
    proposals = max(int(diag.get("proposals", args.iterations)), 1)

    return {
        "model": "loc_student",
        "k": args.k,
        "n": args.n,
        "seed": seed,
        "eps": eps,
        "L": L,
        "projection_tol": args.projection_tol,
        "reverse_position_tol": args.reverse_position_tol,
        "reverse_momentum_tol": args.reverse_momentum_tol,
        "include_gram_correction": include_gram,
        "iterations": args.iterations,
        "burnin": args.burnin,
        "mu_star": mu_star,
        "mu_star_source": mu_star_source,
        **summ,
        **uncertainty,
        "ess": ess,
        "ess_per_sec": ess_sec,
        "acceptance_rate": float(result["x_acceptance_rate"]),
        "mu_acceptance_rate": float(result["mu_acceptance_rate"]),
        "projection_failure_rate": float(diag.get("projection_failure_count", 0)) / proposals,
        "reverse_check_failure_rate": float(diag.get("reverse_check_failure_count", 0)) / proposals,
        "mean_hamiltonian_error": float(diag.get("delta_h_mean_abs", np.nan)),
        "max_hamiltonian_error": float(diag.get("delta_h_max_abs", np.nan)),
        "mean_projection_residual": float(diag.get("mean_projection_residual", np.nan)),
        "max_projection_residual": float(diag.get("max_projection_residual", np.nan)),
        "runtime_seconds": runtime,
    }


def main() -> None:
    args = parse_args()
    mu_star_refs = load_mu_star_refs(args.mu_star_reference_csv)
    done = set() if args.overwrite else _completed(args.out_csv)
    for seed in args.seeds:
        for include_gram, eps, L in args.configs:
            row_key = _key(
                {
                    "model": "loc_student",
                    "k": args.k,
                    "n": args.n,
                    "seed": seed,
                    "eps": eps,
                    "L": L,
                    "projection_tol": args.projection_tol,
                    "reverse_position_tol": args.reverse_position_tol,
                    "reverse_momentum_tol": args.reverse_momentum_tol,
                    "include_gram_correction": include_gram,
                    "iterations": args.iterations,
                    "burnin": args.burnin,
                }
            )
            if row_key in done:
                continue
            row = run_one(args, seed, include_gram, eps, L, mu_star_refs)
            _append_row(args.out_csv, row)
            print(f"wrote seed={seed} gram={include_gram} eps={eps} L={L}")


if __name__ == "__main__":
    main()
