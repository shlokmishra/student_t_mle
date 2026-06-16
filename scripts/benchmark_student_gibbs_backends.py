"""Benchmark Student-t Gibbs backends on small Grace-relevant grids."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import jax.random as random
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import loc_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["jax_loop", "jax_scan", "numba"], choices=["jax_loop", "jax_scan", "numba"])
    parser.add_argument("--k-values", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--n-values", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def base_params(k: float, n: int, iterations: int, backend: str) -> dict:
    return {
        "k": float(k),
        "n": int(n),
        "num_iterations_T": int(iterations),
        "proposal_std_mu": 0.3,
        "proposal_std_z": 0.02,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "gibbs_backend": backend,
    }


def force_materialized(chain: dict) -> None:
    mu_chain = chain["mu_chain"]
    x_chain = chain["x_chain"]
    if hasattr(mu_chain, "block_until_ready"):
        mu_chain.block_until_ready()
    if hasattr(x_chain, "block_until_ready"):
        x_chain.block_until_ready()
    np.asarray(mu_chain)
    np.asarray(x_chain)


def run_case(backend: str, k: float, n: int, iterations: int, warmup_iterations: int, seed: int) -> dict:
    print(
        f"START backend={backend} k={k:g} n={n} seed={seed} "
        f"iterations={iterations} warmup_iterations={warmup_iterations}",
        flush=True,
    )
    warmup_params = base_params(k, n, warmup_iterations, backend)
    force_materialized(loc_student.run_gibbs(random.PRNGKey(seed + 100_000), 0.0, warmup_params, verbose=False))

    params = base_params(k, n, iterations, backend)
    started = time.perf_counter()
    chain = loc_student.run_gibbs(random.PRNGKey(seed), 0.0, params, verbose=False)
    force_materialized(chain)
    elapsed = time.perf_counter() - started
    mus = np.asarray(chain["mu_chain"], dtype=float)
    post = mus[min(iterations // 5, iterations):]
    return {
        "backend": backend,
        "k": float(k),
        "n": int(n),
        "iterations": int(iterations),
        "seed": int(seed),
        "elapsed_sec": elapsed,
        "iterations_per_sec": iterations / elapsed if elapsed > 0 else np.nan,
        "pair_updates_per_sec": (iterations * (n // 2)) / elapsed if elapsed > 0 else np.nan,
        "mu_acceptance_rate": float(chain["mu_acceptance_rate"]),
        "pair_acceptance_rate": float(chain["pair_acceptance_rate"]),
        "z_acceptance_rate": float(chain["z_acceptance_rate"]),
        "posterior_mu_mean": float(np.mean(post)),
        "posterior_mu_sd": float(np.std(post)),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for backend in args.backends:
        for k in args.k_values:
            for n in args.n_values:
                for seed in args.seeds:
                    row = run_case(backend, k, n, args.iterations, args.warmup_iterations, seed)
                    rows.append(row)
                    print(
                        f"{backend} k={k:g} n={n} seed={seed}: "
                        f"{row['elapsed_sec']:.3f}s, {row['iterations_per_sec']:.1f} iter/s, "
                        f"mu_mean={row['posterior_mu_mean']:.4g}",
                        flush=True,
                    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
