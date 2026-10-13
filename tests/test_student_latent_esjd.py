import numpy as np

from scripts.study_student_latent_esjd import _movement


def test_movement_uses_whole_vector_per_coordinate_esjd():
    x_chain = np.array([[0.0, 0.0], [1.0, -1.0], [1.0, -1.0]])
    out = _movement(x_chain, burn_in=0, mu_star=0.0, k=2.0)
    assert out["x_esjd_per_coordinate"] == 0.5
    assert np.isclose(out["score_esjd_per_coordinate"], 1.0 / 18.0)
    assert out["x_move_rate"] == 0.5
    assert out["x_max_squared_jump"] == 1.0


def test_score_esjd_is_bounded_for_extreme_latent_coordinates():
    x_chain = np.array([[0.0, 0.0], [1e100, -1e100]])
    out = _movement(x_chain, burn_in=0, mu_star=0.0, k=2.0)
    assert np.isfinite(out["score_esjd_per_coordinate"])
    assert out["score_esjd_per_coordinate"] < 1e-100
