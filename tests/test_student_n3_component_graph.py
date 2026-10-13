from reporting.diagnostics.verify_student_n3_components import (
    chamber_graph,
    chambers,
    connected_components,
    predicted_mle,
)
import numpy as np

from reporting.diagnostics.study_student_n3_branch_components import x_from_z_branch


def test_n3_finite_chamber_graph_has_one_central_and_six_tail_components():
    graph = chamber_graph()
    components = connected_components(graph)
    assert len(chambers()) == 25
    assert sum(map(len, graph.values())) // 2 == 24
    assert sorted(map(len, components), reverse=True) == [13, 2, 2, 2, 2, 2, 2]
    assert sum(all(predicted_mle(node) for node in component) for component in components) == 1
    assert sum(not any(predicted_mle(node) for node in component) for component in components) == 6


def test_tail_components_pair_same_sign_two_outer_with_all_outer():
    components = connected_components(chamber_graph())
    tails = [component for component in components if len(component) == 2]
    for component in tails:
        outer_counts = sorted(sum(bits) for bits, _ in component)
        assert outer_counts == [2, 3]
        two_outer = next(node for node in component if sum(node[0]) == 2)
        signs = [sign for bit, sign in zip(*two_outer) if bit]
        assert signs[0] == signs[1]


def test_three_point_likelihood_difference_quartic_identity():
    y = x_from_z_branch(np.array([-0.2, 0.08, 0.12]), (0, 1, 1), 1.0, 0.0)
    e1 = float(np.sum(y))
    e2 = float(y[0] * y[1] + y[0] * y[2] + y[1] * y[2])
    e3 = float(np.prod(y))
    assert abs(e1 * (1.0 + e2) + e3 * (e2 - 3.0)) < 1e-10
    for location in (-2.0, -0.3, 0.4, 3.0):
        quartic = (
            location**4
            - 2.0 * e1 * location**3
            + (3.0 + e1**2 + 2.0 * e2) * location**2
            - 2.0 * (e1 * (2.0 + e2) + e3) * location
            + 3.0 + 2.0 * e1**2 + e2**2 + 2.0 * e1 * e3
        )
        product_at_location = float(np.prod(1.0 + (y - location) ** 2))
        product_at_zero = float(np.prod(1.0 + y**2))
        np.testing.assert_allclose(
            product_at_location - product_at_zero,
            location**2 * quartic,
            rtol=1e-11,
            atol=1e-11,
        )
