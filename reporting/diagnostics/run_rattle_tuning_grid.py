"""Run a resume-safe Student-t RATTLE tuning grid.

Smoke run:
    python -m reporting.diagnostics.run_rattle_tuning_grid --iterations 30 --burnin 5 --k-values 2 --eps-values 0.01 --L-values 1 --seeds 0
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import Any

import jax.random as random
import numpy as np

from analysis import ess_per_second
from models import loc_student, loc_student_rattle
from reporting.diagnostics.student_t_geometry import posterior_summary

OUT_DIR = Path("reporting/diagnostic_outputs/rattle_tuning_grid")
OUT_CSV = OUT_DIR / "rattle_tuning_grid.csv"


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=_floats, default=[3.0, 2.0], help="Comma-separated k values.")
    parser.add_argument("--n-values", type=_ints, default=[20], help="Comma-separated sample sizes.")
    parser.add_argument("--eps-values", type=_floats, default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08], help="Comma-separated step sizes.")
    parser.add_argument("--L-values", type=_ints, default=[1, 2, 3, 5, 10], help="Comma-separated leapfrog step counts.")
    parser.add_argument("--seeds", type=_ints, default=[0], help="Comma-separated seeds.")
    parser.add_argument("--include-gram-values", default="false,true", help="Comma-separated booleans for Gram correction.")
    parser.add_argument("--iterations", type=int, default=100, help="RATTLE iterations per config; light by default.")
    parser.add_argument("--burnin", type=int, default=20, help="Burn-in iterations for summaries.")
    parser.add_argument("--mu-true", type=float, default=2.0, help="Data-generating location.")
    parser.add_argument(
        "--mu-star-reference-csv",
        type=Path,
        default=None,
        help="Optional CSV with k,n,seed,mu_star. When present, use the stored observed MLE instead of resampling data.",
    )
    parser.add_argument("--prior-mean", type=float, default=0.0, help="Normal prior mean.")
    parser.add_argument("--prior-std", type=float, default=10.0, help="Normal prior sd.")
    parser.add_argument("--proposal-std-mu", type=float, default=0.9, help="RW-MH proposal sd for mu.")
    parser.add_argument("--proj-tols", type=_floats, default=[1e-10], help="Comma-separated projection tolerances.")
    parser.add_argument("--reverse-position-tols", type=_floats, default=[5e-3], help="Comma-separated reverse position tolerances.")
    parser.add_argument("--reverse-momentum-tols", type=_floats, default=[5e-3], help="Comma-separated reverse momentum tolerances.")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV, help="Output CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run completed configs.")
    return parser.parse_args()


def _bools(text: str) -> list[bool]:
    out = []
    for part in text.split(","):
        val = part.strip().lower()
        if not val:
            continue
        if val in {"1", "true", "yes", "y"}:
            out.append(True)
        elif val in {"0", "false", "no", "n"}:
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(f"Invalid bool: {part}")
    return out


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


def run_one(args: argparse.Namespace, cfg: dict[str, Any], mu_star_refs: dict[tuple[str, str, str], float]) -> dict[str, Any]:
    params = {
        "n": cfg["n"],
        "k": cfg["k"],
        "num_iterations_T": args.iterations,
        "prior_mean": args.prior_mean,
        "prior_std": args.prior_std,
        "proposal_std_mu": args.proposal_std_mu,
        "rattle_step_size": cfg["eps"],
        "rattle_num_steps": cfg["L"],
        "rattle_proj_tol": cfg["projection_tol"],
        "rattle_reverse_position_tol": cfg["reverse_position_tol"],
        "rattle_reverse_momentum_tol": cfg["reverse_momentum_tol"],
        "rattle_include_gram_correction": cfg["include_gram_correction"],
        "rattle_proj_damping": 1.0,
        "rattle_proj_line_search": True,
        "rattle_proj_init_strategy": "trial",
        "rattle_relaxed_position_tol": 5e-2,
        "rattle_relaxed_momentum_tol": 5e-2,
    }
    key = random.PRNGKey(cfg["seed"])
    key, key_data, key_rattle = random.split(key, 3)
    case_key = _case_key(cfg["k"], cfg["n"], cfg["seed"])
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
    ess, ess_sec = ess_per_second(post, runtime)
    diag = dict(result["projection_diagnostics"])
    proposals = max(int(diag.get("proposals", args.iterations)), 1)
    return {
        "model": "loc_student",
        "k": cfg["k"],
        "n": cfg["n"],
        "seed": cfg["seed"],
        "eps": cfg["eps"],
        "L": cfg["L"],
        "projection_tol": cfg["projection_tol"],
        "reverse_position_tol": cfg["reverse_position_tol"],
        "reverse_momentum_tol": cfg["reverse_momentum_tol"],
        "include_gram_correction": cfg["include_gram_correction"],
        "iterations": args.iterations,
        "burnin": args.burnin,
        "mu_star": mu_star,
        "mu_star_source": mu_star_source,
        **summ,
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
    include_grams = _bools(args.include_gram_values)
    mu_star_refs = load_mu_star_refs(args.mu_star_reference_csv)
    done = set() if args.overwrite else _completed(args.out_csv)
    configs = itertools.product(
        args.k_values,
        args.n_values,
        args.seeds,
        args.eps_values,
        args.L_values,
        args.proj_tols,
        args.reverse_position_tols,
        args.reverse_momentum_tols,
        include_grams,
    )
    for k, n, seed, eps, L, proj_tol, rev_pos, rev_mom, include_gram in configs:
        cfg = {
            "k": k,
            "n": n,
            "seed": seed,
            "eps": eps,
            "L": L,
            "projection_tol": proj_tol,
            "reverse_position_tol": rev_pos,
            "reverse_momentum_tol": rev_mom,
            "include_gram_correction": include_gram,
        }
        row_key = _key({"model": "loc_student", **cfg, "iterations": args.iterations, "burnin": args.burnin})
        if row_key in done:
            continue
        row = run_one(args, cfg, mu_star_refs)
        _append_row(args.out_csv, row)
        print(f"wrote k={k} n={n} seed={seed} eps={eps} L={L} gram={include_gram}")


if __name__ == "__main__":
    main()
