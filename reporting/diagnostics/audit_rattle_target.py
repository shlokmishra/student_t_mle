"""Audit Student-t RATTLE target conventions and coarea/Gram quantities.

Smoke run:
    python -m reporting.diagnostics.audit_rattle_target --n 6 --k 2 --seed 0
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from models import loc_student_rattle
from reporting.diagnostics.student_t_geometry import (
    constraint_value,
    grad_constraint,
    gram,
    potential_with_gram,
    potential_without_gram,
    psi,
    psi_prime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=2.0, help="Student-t degrees of freedom.")
    parser.add_argument("--n", type=int, default=10, help="Number of latent observations.")
    parser.add_argument("--mu-star", type=float, default=2.0, help="Observed MLE value.")
    parser.add_argument("--mu", type=float, default=2.0, help="Current location parameter for x | mu.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for the audit state.")
    parser.add_argument("--jitter", type=float, default=0.1, help="Symmetric jitter scale around mu_star.")
    return parser.parse_args()


def make_state(seed: int, n: int, mu_star: float, k: float, jitter: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n // 2
    vals = jitter * rng.standard_t(df=k, size=half)
    y = np.concatenate([vals, -vals])
    if n % 2:
        y = np.concatenate([y, np.array([0.0])])
    return mu_star + y


def main() -> None:
    args = parse_args()
    x = make_state(args.seed, args.n, args.mu_star, args.k, args.jitter)
    y = x - args.mu_star
    grad = grad_constraint(x, args.mu_star, args.k)
    G = gram(x, args.mu_star, args.k)
    current_U = loc_student_rattle._potential_energy(x, args.mu, args.k)
    current_U_flag_false = loc_student_rattle._potential_energy(
        x, args.mu, args.k, mu_star=args.mu_star, include_gram_correction=False
    )
    current_U_flag_true = loc_student_rattle._potential_energy(
        x, args.mu, args.k, mu_star=args.mu_star, include_gram_correction=True
    )
    payload = {
        "model": "loc_student",
        "k": args.k,
        "n": args.n,
        "mu_star": args.mu_star,
        "mu": args.mu,
        "constraint_value": constraint_value(x, args.mu_star, args.k),
        "grad_constraint": grad.tolist(),
        "gram": G,
        "log_gram": float(np.log(max(G, 1e-300))),
        "psi": np.asarray(psi(y, args.k), dtype=float).tolist(),
        "psi_prime": np.asarray(psi_prime(y, args.k), dtype=float).tolist(),
        "current_rattle_potential": float(current_U),
        "current_rattle_potential_flag_false": float(current_U_flag_false),
        "potential_with_gram_correction": potential_with_gram(x, args.mu, args.mu_star, args.k),
        "potential_without_gram_correction": potential_without_gram(x, args.mu, args.k),
        "rattle_include_gram_correction_param": "rattle_include_gram_correction",
        "default_include_gram_correction": False,
        "flag_true_matches_formula": bool(np.allclose(current_U_flag_true, potential_with_gram(x, args.mu, args.mu_star, args.k))),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
