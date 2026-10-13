"""Tiny deterministic checks accompanying selected_mle_target_audit.md.

Run from the repository root with .venv/bin/python. Prints JSON; does not
launch chains or overwrite research artifacts. Checks identities, not mixing.
"""

import json
import os
from pathlib import Path
import runpy
import sys

if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from models import loc_affine, loc_student
from models.model_registry import model_validity_rows
from models.student_location_mle import global_mle, loglik


def main():
    c = (-17.0 + np.sqrt(33.0)) / 16.0
    x = np.array([4.0, 4.0, c])
    score = float(np.sum(x / (1.0 + x * x)))
    curvature = float(np.sum((1.0 - x * x) / (1.0 + x * x) ** 2))
    gap = float(loglik(x, 4.0, 1.0) - loglik(x, 0.0, 1.0))
    assert abs(score) < 1e-14 and curvature > 0.0 and gap > 0.0

    tied = np.array([-2.0, 2.0])
    selected = global_mle(tied, 1.0)
    expected = -np.sqrt(3.0)
    np.testing.assert_allclose(selected, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(loglik(tied, expected, 1.0), loglik(tied, -expected, 1.0))

    z = 0.01
    branches = loc_student.psi_inverse(z, 1.0)
    log_w = np.array([loc_student.inverse_branch_log_weight(y, 0.0, 1.0) for y in branches])
    corrected = np.exp(log_w - np.max(log_w))
    corrected /= corrected.sum()
    density = np.array([1.0 / (1.0 + y * y) for y in branches])
    density /= density.sum()
    outer = int(np.argmax(np.abs(branches)))
    np.testing.assert_allclose(corrected[outer], 0.5, atol=1e-14)
    assert density[outer] < 0.001

    logistic_rows = []
    for a in (0.0, 2.0):
        state = np.array([a, -a])
        deriv = 0.5 / np.cosh(state / 2.0) ** 2
        jt = np.linalg.norm(deriv) / np.sum(deriv)
        np.testing.assert_allclose(jt, 1.0 / np.sqrt(2.0), atol=1e-14)
        logistic_rows.append({"a": a, "curvature": float(deriv.sum()), "estimator_jacobian": float(jt)})
    assert logistic_rows[0]["curvature"] != logistic_rows[1]["curvature"]

    u = np.array([-4.0, -0.5, 0.2, 1.0, 6.0])
    current = loc_affine.normalized_affine_proposal(u, 0.7, "student", k=2.0)
    proposal = loc_affine.normalized_affine_proposal(current + np.array([0.1, -0.3, 0.2, 0.4, -0.2]), 0.7, "student", k=2.0)
    displacement = proposal - current
    np.testing.assert_allclose(np.sum((displacement - displacement.mean()) ** 2), loc_affine.centered_sum_squares(-displacement))
    affine_residual = abs(global_mle(proposal, 2.0) - 0.7)
    assert affine_residual < 1e-10

    # Invoke a few existing assertions directly, without pytest collection.
    root = Path(__file__).resolve().parents[2]
    existing_assertions = []
    for filename, names in (
        ("test_student_rattle_diagnostics.py", (
            "test_constraint_gram_matches_rattle_helpers",
            "test_production_potential_uses_the_gram_correction",
        )),
        ("test_loc_affine.py", (
            "test_student_global_mle_is_translation_equivariant",
            "test_local_gaussian_orbit_factor_is_symmetric",
            "test_normal_affine_proposal_is_centered_at_current_state",
        )),
    ):
        namespace = runpy.run_path(str(root / "tests" / filename))
        for name in names:
            namespace[name]()
            existing_assertions.append(filename + "::" + name)

    metadata = [r for r in model_validity_rows() if r["model"] in ("student_t", "logistic") and r["method"] in ("gibbs", "rattle")]
    result = {
        "strict_local_but_not_global": {"x": x.tolist(), "score_at_zero": score, "reduced_curvature_at_zero": curvature, "loglik_at_four_minus_zero": gap, "selected_mle": global_mle(x, 1.0)},
        "two_point_global_tie": {"x": tied.tolist(), "selected": selected, "analytic_lower_maximum": expected},
        "cauchy_branch": {"z": z, "branches": list(branches), "correct_outer_probability": float(corrected[outer]), "density_only_outer_probability": float(density[outer])},
        "logistic_same_support_different_curvature": logistic_rows,
        "affine_proposal_mle_residual": affine_residual,
        "existing_assertion_functions_passed_without_pytest": existing_assertions,
        "current_misleading_registry_rows": metadata,
        "assertions_passed": True,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
