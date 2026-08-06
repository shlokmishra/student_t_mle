"""Profile Student-t Gibbs and RATTLE cost, ESS/sec, and ESJD locally.

The primary Gibbs timing path is the production `numba_full` backend.  The
script also includes a granular NumPy pair-sweep profiler so each conceptual
Gibbs substep can be timed for optimization analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import jax.random as random
import matplotlib
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp, ndtr, ndtri

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.cost_ledger import CostLedger
from models.loc_student import _initial_x, run_gibbs
from models.loc_student_rattle import run_rattle

EPS_Z = 1e-12
EPS_U = 1e-12
EPS_DIV = 1e-12


@dataclass
class Timer:
    totals: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    stack: list[dict[str, float]] = field(default_factory=list)

    @contextmanager
    def section(self, name: str):
        started = time.perf_counter()
        frame = {"child_time": 0.0}
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.stack.pop()
            exclusive = max(elapsed - frame["child_time"], 0.0)
            self.totals[name] = self.totals.get(name, 0.0) + exclusive
            self.calls[name] = self.calls.get(name, 0) + 1
            if self.stack:
                self.stack[-1]["child_time"] += elapsed


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def effective_sample_size(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n <= 1:
        return float(n)
    centered = values - np.mean(values)
    var = float(np.dot(centered, centered) / n)
    if var <= 0.0 or not np.isfinite(var):
        return float(n)
    autocorr_sum = 0.0
    for lag in range(1, n):
        acov = float(np.dot(centered[:-lag], centered[lag:]) / n)
        rho = acov / var
        if rho <= 0.0:
            break
        autocorr_sum += rho
    return float(max(n / (1.0 + 2.0 * autocorr_sum), 1.0))


def esjd(values: np.ndarray, burn_in: int) -> float:
    values = np.asarray(values, dtype=float)
    start = min(max(int(burn_in), 0), max(values.size - 1, 0))
    post = values[start:]
    if post.size <= 1:
        return np.nan
    deltas = np.diff(post)
    return float(np.mean(deltas * deltas))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=_floats, default=[2.0])
    parser.add_argument("--n-values", type=_ints, default=[10, 20, 50, 100, 200, 500, 1000, 2000])
    parser.add_argument("--seeds", type=_ints, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("results/student_sampler_profile_v1"))
    parser.add_argument("--proposal-std-mu", type=float, default=0.3)
    parser.add_argument("--proposal-std-z", type=float, default=0.02)
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-std", type=float, default=10.0)
    parser.add_argument("--rattle-step-size", type=float, default=0.05)
    parser.add_argument("--rattle-num-steps", type=int, default=2)
    parser.add_argument("--rattle-proj-tol", type=float, default=1e-10)
    parser.add_argument("--rattle-proj-max-iters", type=int, default=25)
    parser.add_argument("--rattle-reverse-position-tol", type=float, default=5e-3)
    parser.add_argument("--rattle-reverse-momentum-tol", type=float, default=5e-3)
    parser.add_argument("--rattle-include-gram-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reverse-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initialization", choices=["central", "tail_heavy", "random"], default="central")
    parser.add_argument("--gibbs-pairing-schedule", choices=["random_permutation", "random_parity"], default="random_permutation")
    parser.add_argument("--gibbs-pair-parallel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def z_support(k: float) -> tuple[float, float]:
    bound = 1.0 / (2.0 * math.sqrt(float(k)))
    return -bound + EPS_Z, bound - EPS_Z


def psi(y: np.ndarray, k: float) -> np.ndarray:
    return y / (k + y * y)


def psi_inverse(z: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    z_min, z_max = z_support(k)
    zc = np.clip(np.asarray(z, dtype=float), z_min, z_max)
    tval = 2.0 * math.sqrt(k) * zc
    discr = np.clip(1.0 - tval * tval, 0.0, None)
    root = np.sqrt(discr)
    denom = 2.0 * zc
    denom_safe = np.where(np.abs(denom) < EPS_DIV, np.sign(denom) * EPS_DIV + EPS_DIV, denom)
    y_plus = (1.0 + root) / denom_safe
    y_minus = (1.0 - root) / denom_safe
    y_plus = np.where(np.abs(zc) < EPS_DIV, 0.0, y_plus)
    y_minus = np.where(np.abs(zc) < EPS_DIV, 0.0, y_minus)
    return np.minimum(y_minus, y_plus), np.maximum(y_minus, y_plus)


def log_psi_prime_abs(y: np.ndarray, k: float) -> np.ndarray:
    return np.log(np.abs(k - y * y) + 1e-30) - 2.0 * np.log(k + y * y)


def student_logpdf_no_const(y: np.ndarray, loc: float, k: float) -> np.ndarray:
    return -0.5 * (k + 1.0) * np.log1p(((y - loc) * (y - loc)) / k)


def student_logpdf(y: np.ndarray, loc: float, k: float) -> np.ndarray:
    const = gammaln((k + 1.0) / 2.0) - gammaln(k / 2.0) - 0.5 * math.log(k * math.pi)
    return const + student_logpdf_no_const(y, loc, k)


def norm_logpdf(x: float, loc: float, scale: float) -> float:
    z = (x - loc) / scale
    return float(-0.5 * z * z - math.log(scale) - 0.5 * math.log(2.0 * math.pi))


def mu_logpdf(mu: float, x: np.ndarray, prior_loc: float, prior_scale: float, k: float) -> float:
    return float(np.sum(student_logpdf(x, mu, k)) + norm_logpdf(mu, prior_loc, prior_scale))


def q_logpdf_no_const(z: np.ndarray, mu_current: float, mu_star: float, k: float, timer: Timer | None = None) -> np.ndarray:
    z_min, z_max = z_support(k)
    in_supp = (z > z_min) & (z < z_max)
    if timer is None:
        y_lo, y_hi = psi_inverse(z, k)
    else:
        with timer.section("gibbs.z_mh.inverse_branches_for_q"):
            y_lo, y_hi = psi_inverse(z, k)
    loc = mu_current - mu_star
    vals = np.stack(
        [
            student_logpdf_no_const(y_lo, loc, k) - log_psi_prime_abs(y_lo, k),
            student_logpdf_no_const(y_hi, loc, k) - log_psi_prime_abs(y_hi, k),
        ],
        axis=0,
    )
    out = logsumexp(vals, axis=0)
    return np.where(in_supp, out, -np.inf)


def gibbs_base_params(args: argparse.Namespace, k: float, n: int, seed: int) -> dict[str, Any]:
    return {
        "n": int(n),
        "k": float(k),
        "num_iterations_T": int(args.iterations),
        "proposal_std_mu": float(args.proposal_std_mu),
        "proposal_std_z": float(args.proposal_std_z),
        "prior_mean": float(args.prior_mean),
        "prior_std": float(args.prior_std),
        "initialization": str(args.initialization),
        "initialization_seed": int(seed),
        "store_x_chain": False,
    }


def rattle_base_params(args: argparse.Namespace, k: float, n: int, seed: int) -> dict[str, Any]:
    params = gibbs_base_params(args, k, n, seed)
    params.update(
        {
            "rattle_step_size": float(args.rattle_step_size),
            "rattle_num_steps": int(args.rattle_num_steps),
            "rattle_proj_tol": float(args.rattle_proj_tol),
            "rattle_proj_max_iters": int(args.rattle_proj_max_iters),
            "rattle_reverse_position_tol": float(args.rattle_reverse_position_tol),
            "rattle_reverse_momentum_tol": float(args.rattle_reverse_momentum_tol),
            "rattle_projection_mode": "paper_fixed_direction",
            "rattle_include_gram_correction": bool(args.rattle_include_gram_correction),
            "reverse_check": bool(args.reverse_check),
        }
    )
    return params


def run_profiled_gibbs(args: argparse.Namespace, k: float, n: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    timer = Timer()
    params = gibbs_base_params(args, k, n, seed)
    T = int(args.iterations)
    mu_star = float(args.mu_star)
    x_cur = np.asarray(_initial_x(mu_star, n, k, params), dtype=float).copy()
    mus = np.zeros(T + 1, dtype=float)
    mus[0] = mu_star
    mu_acc = 0
    pair_completed = 0
    z_acc = 0
    z_min, z_max = z_support(k)
    started = time.perf_counter()

    for t_idx in range(1, T + 1):
        with timer.section("gibbs.mu_mh"):
            mu_cur = mus[t_idx - 1]
            mu_cand = mu_cur + float(args.proposal_std_mu) * rng.normal()
            log_cur = mu_logpdf(mu_cur, x_cur, float(args.prior_mean), float(args.prior_std), k)
            log_cand = mu_logpdf(mu_cand, x_cur, float(args.prior_mean), float(args.prior_std), k)
            log_alpha = log_cand - log_cur if np.isfinite(log_cand - log_cur) else -np.inf
            if math.log(max(rng.random(), EPS_U)) < log_alpha:
                mu_new = mu_cand
                mu_acc += 1
            else:
                mu_new = mu_cur
            mus[t_idx] = mu_new

        with timer.section("gibbs.pairing_permutation"):
            perm = rng.permutation(n)
            even = 2 * (n // 2)
            idx_i = perm[:even:2]
            idx_j = perm[1:even:2]
            xi = x_cur[idx_i]
            xj = x_cur[idx_j]

        with timer.section("gibbs.psi_delta"):
            yi = xi - mu_star
            yj = xj - mu_star
            zi = psi(yi, k)
            zj = psi(yj, k)
            delta = zi + zj

        with timer.section("gibbs.z_mh.sample_z_i"):
            low_int = np.maximum(z_min, delta - z_max)
            high_int = np.minimum(z_max, delta - z_min)
            valid = low_int < high_int
            sigma_z = float(args.proposal_std_z)
            cdf_low = ndtr((low_int - zi) / sigma_z)
            cdf_high = ndtr((high_int - zi) / sigma_z)
            mass = cdf_high - cdf_low
            u = cdf_low + rng.random(zi.shape[0]) * np.maximum(mass, 0.0)
            z_prop = zi + sigma_z * ndtri(np.clip(u, EPS_U, 1.0 - EPS_U))
            z_prop = np.clip(z_prop, low_int, high_int)

        with timer.section("gibbs.z_mh.transition_density"):
            log_norm_cur = np.log(np.maximum(mass, 1e-300))
            cdf_low_back = ndtr((low_int - z_prop) / sigma_z)
            cdf_high_back = ndtr((high_int - z_prop) / sigma_z)
            log_norm_prop = np.log(np.maximum(cdf_high_back - cdf_low_back, 1e-300))

        with timer.section("gibbs.z_mh.q_tilde_logpdf"):
            log_cur = q_logpdf_no_const(zi, mu_new, mu_star, k, timer) + q_logpdf_no_const(delta - zi, mu_new, mu_star, k, timer)
            log_prop = q_logpdf_no_const(z_prop, mu_new, mu_star, k, timer) + q_logpdf_no_const(delta - z_prop, mu_new, mu_star, k, timer)

        with timer.section("gibbs.z_mh.accept_z"):
            log_alpha = log_prop - log_cur + log_norm_cur - log_norm_prop
            z_accept = valid & np.isfinite(log_alpha) & (np.log(np.maximum(rng.random(zi.shape[0]), EPS_U)) < log_alpha)
            zi_tilde = np.where(z_accept, z_prop, zi)
            zj_tilde = delta - zi_tilde
            z_acc += int(np.sum(z_accept))

        with timer.section("gibbs.inverse_branches_final"):
            in_supp = (zi_tilde > z_min) & (zi_tilde < z_max) & (zj_tilde > z_min) & (zj_tilde < z_max)
            yi_lo, yi_hi = psi_inverse(zi_tilde, k)
            yj_lo, yj_hi = psi_inverse(zj_tilde, k)

        with timer.section("gibbs.branch_weights"):
            loc = mu_new - mu_star
            wi_lo = student_logpdf_no_const(yi_lo, loc, k) - log_psi_prime_abs(yi_lo, k)
            wi_hi = student_logpdf_no_const(yi_hi, loc, k) - log_psi_prime_abs(yi_hi, k)
            wj_lo = student_logpdf_no_const(yj_lo, loc, k) - log_psi_prime_abs(yj_lo, k)
            wj_hi = student_logpdf_no_const(yj_hi, loc, k) - log_psi_prime_abs(yj_hi, k)
            pi_hi = 1.0 / (1.0 + np.exp(np.clip(wi_lo - wi_hi, -700.0, 700.0)))
            pj_hi = 1.0 / (1.0 + np.exp(np.clip(wj_lo - wj_hi, -700.0, 700.0)))

        with timer.section("gibbs.sample_branch_pair"):
            yi_new = np.where(rng.random(yi_lo.shape[0]) < pi_hi, yi_hi, yi_lo)
            yj_new = np.where(rng.random(yj_lo.shape[0]) < pj_hi, yj_hi, yj_lo)
            yi_new = np.where(in_supp, yi_new, yi)
            yj_new = np.where(in_supp, yj_new, yj)
            pair_completed += int(np.sum(in_supp))

        with timer.section("gibbs.assign_x_pair"):
            x_cur[idx_i] = yi_new + mu_star
            x_cur[idx_j] = yj_new + mu_star

    elapsed = time.perf_counter() - started
    burn_in = min(max(int(args.burn_in), 0), max(mus.size - 1, 0))
    post = mus[burn_in:]
    ess_mu = effective_sample_size(post)
    esjd_mu = esjd(mus, burn_in)
    attempted = T * (n // 2)
    row = {
        "model": "student_t",
        "method": "gibbs_profile_numpy",
        "k": float(k),
        "n": int(n),
        "seed": int(seed),
        "iterations": int(T),
        "burn_in": int(args.burn_in),
        "elapsed_sec": float(elapsed),
        "sec_per_iteration": float(elapsed) / max(T, 1),
        "ess_mu": float(ess_mu),
        "ess_mu_per_sec": float(ess_mu) / max(elapsed, 1e-12),
        "esjd_mu": float(esjd_mu),
        "esjd_mu_per_sec": float(esjd_mu) / max(float(elapsed) / max(T, 1), 1e-12),
        "mu_acceptance_rate": float(mu_acc) / max(T, 1),
        "pair_acceptance_rate": float(pair_completed) / max(attempted, 1),
        "z_acceptance_rate": float(z_acc) / max(attempted, 1),
        "proposal_std_mu": float(args.proposal_std_mu),
        "proposal_std_z": float(args.proposal_std_z),
    }
    block_rows = []
    measured = float(sum(timer.totals.values()))
    if elapsed > measured:
        timer.totals["gibbs.other_loop_overhead"] = elapsed - measured
        timer.calls["gibbs.other_loop_overhead"] = 1
    for block, seconds in sorted(timer.totals.items()):
        block_rows.append(
            {
                "model": "student_t",
                "method": "gibbs_profile_numpy",
                "k": float(k),
                "n": int(n),
                "seed": int(seed),
                "iterations": int(T),
                "block": block,
                "calls": int(timer.calls.get(block, 0)),
                "elapsed_sec": float(seconds),
                "sec_per_iteration": float(seconds) / max(T, 1),
                "share_of_total": float(seconds) / max(elapsed, 1e-12),
            }
        )
    return row, block_rows


def run_profiled_gibbs_numba_full(args: argparse.Namespace, k: float, n: int, seed: int) -> dict[str, Any]:
    params = gibbs_base_params(args, k, n, seed)
    params.update(
        {
            "gibbs_backend": "numba_full",
            "gibbs_pairing_schedule": str(args.gibbs_pairing_schedule),
            "gibbs_pair_parallel": bool(args.gibbs_pair_parallel),
            "store_x_chain": False,
        }
    )
    ledger = CostLedger(
        "gibbs",
        n=int(n),
        k=float(k),
        mu_star=float(args.mu_star),
        seed=int(seed),
        model="student_t",
        iterations=int(args.iterations),
    )
    ledger.start()
    chain = run_gibbs(random.PRNGKey(int(seed)), float(args.mu_star), params, verbose=False, cost_ledger=ledger)
    ledger.stop()
    mus = np.asarray(chain["mu_chain"], dtype=float)
    burn_in = min(max(int(args.burn_in), 0), max(mus.size - 1, 0))
    post = mus[burn_in:]
    elapsed = float(ledger.counters.get("wall_time_sec", np.nan))
    ess_mu = effective_sample_size(post)
    esjd_mu = esjd(mus, burn_in)
    return {
        "model": "student_t",
        "method": "gibbs_numba_full",
        "k": float(k),
        "n": int(n),
        "seed": int(seed),
        "iterations": int(args.iterations),
        "burn_in": int(args.burn_in),
        "elapsed_sec": elapsed,
        "sec_per_iteration": elapsed / max(int(args.iterations), 1),
        "ess_mu": float(ess_mu),
        "ess_mu_per_sec": float(ess_mu) / max(elapsed, 1e-12),
        "esjd_mu": float(esjd_mu),
        "esjd_mu_per_sec": float(esjd_mu) / max(elapsed / max(int(args.iterations), 1), 1e-12),
        "mu_acceptance_rate": float(chain.get("mu_acceptance_rate", np.nan)),
        "pair_acceptance_rate": float(chain.get("pair_acceptance_rate", np.nan)),
        "z_acceptance_rate": float(chain.get("z_acceptance_rate", np.nan)),
        "gibbs_backend": "numba_full",
        "gibbs_pairing_schedule": str(args.gibbs_pairing_schedule),
        "gibbs_pair_parallel": bool(args.gibbs_pair_parallel),
    }


def run_profiled_rattle(args: argparse.Namespace, k: float, n: int, seed: int) -> dict[str, Any]:
    params = rattle_base_params(args, k, n, seed)
    ledger = CostLedger("rattle", n=int(n), k=float(k), mu_star=float(args.mu_star), seed=int(seed), model="student_t", iterations=int(args.iterations))
    ledger.start()
    chain = run_rattle(random.PRNGKey(int(seed)), float(args.mu_star), params, verbose=False, cost_ledger=ledger)
    ledger.stop()
    mus = np.asarray(chain["mu_chain"], dtype=float)
    burn_in = min(max(int(args.burn_in), 0), max(mus.size - 1, 0))
    post = mus[burn_in:]
    elapsed = float(ledger.counters.get("wall_time_sec", np.nan))
    ess_mu = effective_sample_size(post)
    esjd_mu = esjd(mus, burn_in)
    diag = dict(chain.get("projection_diagnostics", {}))
    return {
        "model": "student_t",
        "method": "rattle",
        "k": float(k),
        "n": int(n),
        "seed": int(seed),
        "iterations": int(args.iterations),
        "burn_in": int(args.burn_in),
        "elapsed_sec": elapsed,
        "sec_per_iteration": elapsed / max(int(args.iterations), 1),
        "ess_mu": float(ess_mu),
        "ess_mu_per_sec": float(ess_mu) / max(elapsed, 1e-12),
        "esjd_mu": float(esjd_mu),
        "esjd_mu_per_sec": float(esjd_mu) / max(elapsed / max(int(args.iterations), 1), 1e-12),
        "mu_acceptance_rate": float(chain.get("mu_acceptance_rate", np.nan)),
        "x_acceptance_rate": float(chain.get("x_acceptance_rate", np.nan)),
        "projection_failure_rate": float(diag.get("projection_failure_count", 0)) / max(float(ledger.counters.get("hmc_proposals", args.iterations)), 1.0),
        "reverse_check_failure_rate": float(diag.get("reverse_check_failure_count", 0)) / max(float(ledger.counters.get("hmc_proposals", args.iterations)), 1.0),
        "rattle_step_size": float(args.rattle_step_size),
        "rattle_num_steps": int(args.rattle_num_steps),
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "elapsed_sec",
        "sec_per_iteration",
        "ess_mu",
        "ess_mu_per_sec",
        "esjd_mu",
        "esjd_mu_per_sec",
        "mu_acceptance_rate",
        "pair_acceptance_rate",
        "z_acceptance_rate",
        "x_acceptance_rate",
        "projection_failure_rate",
        "reverse_check_failure_rate",
    ]
    present = [col for col in metrics if col in rows.columns]
    out = rows.groupby(["method", "k", "n"], as_index=False)[present].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(part) for part in col if part).rstrip("_") if isinstance(col, tuple) else str(col) for col in out.columns]
    out["num_seeds"] = rows.groupby(["method", "k", "n"]).size().to_numpy()
    return out


def summarize_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    return (
        blocks.groupby(["method", "k", "n", "block"], as_index=False)
        .agg(
            sec_per_iteration_mean=("sec_per_iteration", "mean"),
            sec_per_iteration_std=("sec_per_iteration", "std"),
            share_of_total_mean=("share_of_total", "mean"),
            calls_mean=("calls", "mean"),
        )
        .sort_values(["method", "k", "n", "block"])
    )


def write_figures(out_dir: Path, summary: pd.DataFrame, block_summary: pd.DataFrame) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    primary = summary[summary["method"].isin(["gibbs_numba_full", "rattle"])].copy()

    for metric, ylabel, filename in [
        ("ess_mu_per_sec_mean", "ESS(mu) per second", "student_gibbs_rattle_ess_per_sec.png"),
        ("esjd_mu_per_sec_mean", "ESJD(mu) per second", "student_gibbs_rattle_esjd_per_sec.png"),
        ("sec_per_iteration_mean", "seconds per iteration", "student_gibbs_rattle_sec_per_iter.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for method, part in primary.groupby("method"):
            part = part.sort_values("n")
            ax.plot(part["n"], part[metric], marker="o", label=str(method))
        ax.set_xscale("log")
        if metric != "sec_per_iteration_mean":
            ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs n")
        ax.legend()
        plt.tight_layout()
        path = fig_dir / filename
        plt.savefig(path, dpi=160)
        plt.close(fig)
        figures.append(str(path))

    gibbs_blocks = block_summary[block_summary["method"].eq("gibbs_profile_numpy")]
    for k, part in gibbs_blocks.groupby("k"):
        for n in sorted(part["n"].unique()):
            sub = part[part["n"].eq(n)].sort_values("sec_per_iteration_mean", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5.2))
            labels = [str(x).replace("gibbs.", "") for x in sub["block"]]
            ax.bar(labels, sub["sec_per_iteration_mean"])
            ax.set_title(f"Gibbs granular cost breakup, k={float(k):g}, n={int(n)}")
            ax.set_ylabel("seconds per iteration")
            ax.tick_params(axis="x", rotation=55)
            plt.tight_layout()
            path = fig_dir / f"gibbs_granular_cost_k{float(k):g}_n{int(n)}.png"
            plt.savefig(path, dpi=160)
            plt.close(fig)
            figures.append(str(path))

    (fig_dir / "figure_index.json").write_text(json.dumps({"figures": figures}, indent=2), encoding="utf-8")
    return figures


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    return df[[col for col in cols if col in df.columns]].to_markdown(index=False, floatfmt=".6g")


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, block_summary: pd.DataFrame, figures: list[str]) -> None:
    report = out_dir / "student_sampler_profile_report.md"
    cols = [
        "method",
        "n",
        "sec_per_iteration_mean",
        "ess_mu_mean",
        "ess_mu_per_sec_mean",
        "esjd_mu_mean",
        "esjd_mu_per_sec_mean",
        "mu_acceptance_rate_mean",
        "pair_acceptance_rate_mean",
        "z_acceptance_rate_mean",
        "x_acceptance_rate_mean",
    ]
    top_blocks = block_summary[block_summary["method"].eq("gibbs_profile_numpy")].sort_values(
        ["n", "sec_per_iteration_mean"], ascending=[True, False]
    )
    primary = summary[summary["method"].isin(["gibbs_numba_full", "rattle"])].sort_values(["n", "method"])
    block_cols = ["n", "block", "sec_per_iteration_mean", "share_of_total_mean", "calls_mean"]
    lines = [
        "# Student Sampler Cost and Efficiency Profile",
        "",
        "## Settings",
        "",
        f"- k values: {','.join(str(k) for k in args.k_values)}",
        f"- n values: {','.join(str(n) for n in args.n_values)}",
        f"- seeds: {','.join(str(seed) for seed in args.seeds)}",
        f"- iterations: {int(args.iterations)}",
        f"- burn-in for ESS/ESJD: {int(args.burn_in)}",
        f"- Primary Gibbs timing: production numba_full backend, pairing_schedule={args.gibbs_pairing_schedule}, pair_parallel={bool(args.gibbs_pair_parallel)}",
        f"- Granular Gibbs timing: NumPy diagnostic pair sweep with exclusive timers and proposal_std_z={float(args.proposal_std_z)}",
        f"- RATTLE: step_size={float(args.rattle_step_size)}, num_steps={int(args.rattle_num_steps)}, M=I",
        "",
        "## Interpretation",
        "",
        "The primary Gibbs-vs-RATTLE efficiency rows use the production `numba_full` Gibbs backend.  The separate `gibbs_profile_numpy` rows time the listed pair-update steps directly: z proposal, inverse branches, pair weights, branch sampling, and assignment.  Use those granular rows to identify cost structure, not as the fastest Gibbs benchmark.",
        "",
        "ESS/sec and ESJD/sec are short-run local diagnostics.  They combine movement and runtime, but production claims still need long-chain correctness and mixing checks.",
        "",
        "## Primary Efficiency Summary",
        "",
        markdown_table(primary, cols),
        "",
        "## All Timing Rows",
        "",
        markdown_table(summary, cols),
        "",
        "## Gibbs Granular Cost Breakup",
        "",
        markdown_table(top_blocks, block_cols),
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{path}`" for path in figures)
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.warmup:
        print("[warmup] compiling/running tiny numba_full Gibbs and RATTLE warmup", flush=True)
        for k in args.k_values:
            for n in args.n_values:
                gibbs_params = gibbs_base_params(args, float(k), int(n), int(args.seeds[0]))
                gibbs_params.update(
                    {
                        "num_iterations_T": 2,
                        "gibbs_backend": "numba_full",
                        "gibbs_pairing_schedule": str(args.gibbs_pairing_schedule),
                        "gibbs_pair_parallel": bool(args.gibbs_pair_parallel),
                        "store_x_chain": False,
                    }
                )
                run_gibbs(random.PRNGKey(int(args.seeds[0])), args.mu_star, gibbs_params, verbose=False)
                params = rattle_base_params(args, float(k), int(n), int(args.seeds[0]))
                params["num_iterations_T"] = 2
                run_rattle(random.PRNGKey(int(args.seeds[0])), args.mu_star, params, verbose=False)
        print("[warmup] done", flush=True)

    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    total = len(args.k_values) * len(args.n_values) * len(args.seeds) * 3
    case_idx = 0
    for k in args.k_values:
        for n in args.n_values:
            for seed in args.seeds:
                case_idx += 1
                print(f"[{case_idx}/{total}] Gibbs profile k={float(k):g} n={int(n)} seed={int(seed)}", flush=True)
                row, blocks = run_profiled_gibbs(args, float(k), int(n), int(seed))
                rows.append(row)
                block_rows.extend(blocks)
                print(f"  sec/iter={row['sec_per_iteration']:.6g} ESS/sec={row['ess_mu_per_sec']:.6g} ESJD/sec={row['esjd_mu_per_sec']:.6g}", flush=True)
                case_idx += 1
                print(f"[{case_idx}/{total}] Gibbs numba_full k={float(k):g} n={int(n)} seed={int(seed)}", flush=True)
                numba_row = run_profiled_gibbs_numba_full(args, float(k), int(n), int(seed))
                rows.append(numba_row)
                print(f"  sec/iter={numba_row['sec_per_iteration']:.6g} ESS/sec={numba_row['ess_mu_per_sec']:.6g} ESJD/sec={numba_row['esjd_mu_per_sec']:.6g}", flush=True)
                case_idx += 1
                print(f"[{case_idx}/{total}] RATTLE profile k={float(k):g} n={int(n)} seed={int(seed)}", flush=True)
                rattle_row = run_profiled_rattle(args, float(k), int(n), int(seed))
                rows.append(rattle_row)
                print(f"  sec/iter={rattle_row['sec_per_iteration']:.6g} ESS/sec={rattle_row['ess_mu_per_sec']:.6g} ESJD/sec={rattle_row['esjd_mu_per_sec']:.6g}", flush=True)

    rows_df = pd.DataFrame(rows)
    blocks_df = pd.DataFrame(block_rows)
    summary = summarize(rows_df)
    block_summary = summarize_blocks(blocks_df)
    rows_df.to_csv(args.out / "student_sampler_profile_rows.csv", index=False)
    blocks_df.to_csv(args.out / "student_gibbs_granular_blocks.csv", index=False)
    summary.to_csv(args.out / "student_sampler_profile_summary.csv", index=False)
    block_summary.to_csv(args.out / "student_gibbs_granular_block_summary.csv", index=False)
    figures = write_figures(args.out, summary, block_summary)
    write_report(args.out, args, summary, block_summary, figures)
    print(f"wrote comparison profile to {args.out}", flush=True)


if __name__ == "__main__":
    main()
