import numpy as np

from models.loc.student import score_rattle as loc_student_rattle
from reporting.diagnostics.study_student_n3_branch_components import (
    branch_code,
    inverse_score,
    score_coordinates,
    x_from_z_branch,
)


def test_inverse_score_recovers_both_branches():
    k = 2.0
    z = np.array([-0.2, 0.1, 0.1])
    for bits in ((0, 0, 0), (1, 0, 1), (1, 1, 1)):
        x = x_from_z_branch(z, bits, k, 0.0)
        np.testing.assert_allclose(score_coordinates(x, 0.0, k), z, atol=1e-14)
        assert branch_code(x, 0.0, k) == "".join(str(bit) for bit in bits)


def test_rattle_accepts_exact_explicit_initial_state():
    k = 1.0
    initial_x = x_from_z_branch(np.array([-0.2, 0.1, 0.1]), (1, 0, 1), k, 0.0)
    params = {
        "k": k,
        "n": 3,
        "num_iterations_T": 1,
        "proposal_std_mu": 0.1,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "rattle_step_size": 0.005,
        "rattle_num_steps": 1,
        "initial_x": initial_x,
        "store_x_chain": True,
    }
    output = loc_student_rattle.run_rattle(
        np.array([4, 2], dtype=np.uint32), 0.0, params, verbose=False
    )
    np.testing.assert_allclose(output["x_chain"][0], initial_x)


def test_rattle_rejects_off_manifold_explicit_initial_state():
    params = {
        "k": 1.0,
        "n": 3,
        "num_iterations_T": 1,
        "proposal_std_mu": 0.1,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "initial_x": np.array([0.1, 0.2, 0.3]),
    }
    with np.testing.assert_raises_regex(ValueError, "not on the score manifold"):
        loc_student_rattle.run_rattle(
            np.array([4, 3], dtype=np.uint32), 0.0, params, verbose=False
        )
