"""Compare latent manifold geometry from Gibbs and RATTLE chains.

Smoke run:
    python -m reporting.diagnostics.compare_latent_geometry --iterations 50 --burnin 10 --k 2 --n 10 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import jax.random as random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models import loc_student, loc_student_rattle
from reporting.diagnostics.student_t_geometry import geometry_summary

OUT_DIR = Path("reporting/diagnostic_outputs/latent_geometry")


def _bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = text.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=2.0, help="Student-t degrees of freedom.")
    parser.add_argument("--n", type=int, default=20, help="Sample size.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    parser.add_argument("--mu-star", type=float, default=None, help="Optional observed MLE override.")
    parser.add_argument(
        "--mu-star-reference-csv",
        type=Path,
        default=None,
        help="Optional CSV with k,n,seed,mu_star. Used when --mu-star is absent.",
    )
    parser.add_argument("--iterations", type=int, default=100, help="MCMC iterations; smoke by default.")
    parser.add_argument("--burnin", type=int, default=20, help="Burn-in.")
    parser.add_argument("--thin", type=int, default=1, help="Retain every thin-th state after burn-in.")
    parser.add_argument("--eps", type=float, default=None, help="RATTLE step size. Default uses a conservative k-dependent value.")
    parser.add_argument("--L", type=int, default=None, help="RATTLE leapfrog steps. Default uses a conservative k-dependent value.")
    parser.add_argument(
        "--include-gram-correction",
        type=_bool,
        nargs="?",
        const=True,
        default=False,
        help="Use Gram-corrected RATTLE target; accepts true/false.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory.")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _case_key(k: float, n: int, seed: int) -> tuple[str, str, str]:
    return (f"{float(k):.12g}", str(int(n)), str(int(seed)))


def _load_mu_star_refs(path: Path | None) -> dict[tuple[str, str, str], float]:
    if path is None:
        return {}
    refs = {}
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


def _summary_rows(rows: list[dict]) -> list[dict]:
    metrics = [
        "gram",
        "log_gram",
        "max_abs_y",
        "count_abs_y_gt_sqrt_k",
        "count_abs_y_gt_5",
        "min_abs_psi_prime",
        "constraint_residual",
    ]
    out = []
    for method in sorted(set(row["method"] for row in rows)):
        method_rows = [row for row in rows if row["method"] == method]
        for metric in metrics:
            vals = np.asarray([float(row[metric]) for row in method_rows], dtype=float)
            out.append({
                "method": method,
                "metric": metric,
                "n_states": vals.size,
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "q025": float(np.quantile(vals, 0.025)),
                "q50": float(np.quantile(vals, 0.5)),
                "q975": float(np.quantile(vals, 0.975)),
            })
    return out


def _plot(rows: list[dict], out_dir: Path) -> None:
    metrics = ["gram", "log_gram", "max_abs_y", "count_abs_y_gt_sqrt_k", "count_abs_y_gt_5", "min_abs_psi_prime"]
    methods = sorted(set(row["method"] for row in rows))
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in methods:
            vals = [float(row[metric]) for row in rows if row["method"] == method]
            ax.hist(vals, bins=30, alpha=0.45, density=True, label=method)
        ax.set_xlabel(metric)
        ax.set_ylabel("density")
        ax.set_title(f"{metric} by latent sampler")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_distribution.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    params = {
        "n": args.n,
        "k": args.k,
        "num_iterations_T": args.iterations,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "proposal_std_mu": 0.9,
        "proposal_std_z": 0.03,
        "rattle_step_size": args.eps if args.eps is not None else (0.04 if args.k <= 2.0 else 0.05),
        "rattle_num_steps": args.L if args.L is not None else (2 if args.k <= 2.0 else 1),
        "rattle_reverse_position_tol": 2e-2 if args.k <= 2.0 else 1e-2,
        "rattle_reverse_momentum_tol": 2e-2 if args.k <= 2.0 else 1e-2,
        "rattle_proj_damping": 1.0,
        "rattle_proj_line_search": True,
        "rattle_proj_init_strategy": "trial",
        "rattle_relaxed_position_tol": 5e-2,
        "rattle_relaxed_momentum_tol": 5e-2,
        "rattle_include_gram_correction": args.include_gram_correction,
    }
    key = random.PRNGKey(args.seed)
    key, key_data, key_gibbs, key_rattle = random.split(key, 4)
    mu_star_refs = _load_mu_star_refs(args.mu_star_reference_csv)
    case_key = _case_key(args.k, args.n, args.seed)
    if args.mu_star is not None:
        mu_star = float(args.mu_star)
        mu_star_source = "cli"
    elif case_key in mu_star_refs:
        mu_star = float(mu_star_refs[case_key])
        mu_star_source = "reference_csv"
    else:
        data = np.asarray(loc_student.sample_data(key_data, params, loc=2.0), dtype=float)
        mu_star = float(loc_student.get_mle(data, params))
        mu_star_source = "simulated_data"
    gibbs = loc_student.run_gibbs(key_gibbs, mu_star, params.copy(), verbose=False)
    rattle = loc_student_rattle.run_rattle(key_rattle, mu_star, params.copy(), verbose=False)

    rows = []
    for method, chain in [("gibbs", np.asarray(gibbs["x_chain"])), ("rattle", np.asarray(rattle["x_chain"]))]:
        for idx in range(min(args.burnin, chain.shape[0] - 1), chain.shape[0], args.thin):
            rows.append({
                "method": method,
                "iteration": idx,
                "k": args.k,
                "n": args.n,
                "seed": args.seed,
                "mu_star": mu_star,
                "mu_star_source": mu_star_source,
                "include_gram_correction": args.include_gram_correction,
                "eps": params["rattle_step_size"],
                "L": params["rattle_num_steps"],
                **geometry_summary(chain[idx], mu_star, args.k),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "latent_geometry.csv"
    summary_path = args.out_dir / "latent_geometry_summary.csv"
    _write_csv(csv_path, rows)
    _write_csv(summary_path, _summary_rows(rows))
    _plot(rows, args.out_dir)
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
