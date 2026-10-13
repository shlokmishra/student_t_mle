"""Independent identities, root counts, exact/rounded ties, and near-boundary MLEs."""
import numpy as np
import pytest
import sympy as sp

from reporting.diagnostics import student_n5_boundary as b
from models.student_location_mle import global_mle, stationary_points, StudentMLEError
from scripts.selected_mle_reference_oracle import selected_mle_oracle


def test_likelihood_factor_and_derivative_against_original_product():
    a = sp.Symbol("a")
    original = sp.prod(1+(x-b.s)**2 for x in [-a, -a, 0, a, a])
    reduced = b.D.subs({b.v: b.s**2, b.A: a*a})
    assert sp.expand(original-reduced) == 0
    assert sp.expand(sp.diff(reduced, b.s)-2*b.s*b.P.subs({b.v: b.s**2, b.A: a*a})*b.R.subs({b.v: b.s**2, b.A: a*a})) == 0
    j = 1+4*(1-b.A)/(1+b.A)**2
    assert sp.factor(j-sp.Rational(1, 2)) == (b.A-3)**2/(2*(b.A+1)**2)


def test_resultants_and_stationary_gap_reduction():
    assert sp.expand(sp.resultant(b.R, b.H, b.v)+(b.A**2-2*b.A+5)*b.K) == 0
    assert sp.expand(sp.resultant(b.R, b.H, b.A)+b.v**2*b.L) == 0
    assert sp.factor(sp.discriminant(b.R, b.v)) == 16*b.A*(b.A-5)
    assert sp.expand(sp.rem(b.H, b.R, b.v)-(b.N-b.B*b.v)/125) == 0


def test_exact_boundary_certificate_and_all_real_root_counts():
    c = b.algebraic_certificate()
    assert c["Sturm_counts"] == {"K_negative": 2, "K_positive": 1, "K_10_11": 1,
                                  "L_negative": 2, "L_positive": 1}
    assert c["exact_algebraic_oracle"]["submitted_to_binary_float_selector"] is False
    assert len(c["exact_algebraic_oracle"]["global_locations"]) == 3
    for name, poly in (("A_c", b.K), ("a_c", b.K.subs(b.A, b.s*b.s)),
                       ("v_c", b.L), ("s_c", b.L.subs(b.v, b.s*b.s))):
        lo, hi = map(sp.Rational, c["enclosures"][name]["rational"])
        assert 0 < hi-lo <= sp.Rational(2, 10**32)
        assert sp.Poly(poly).count_roots(lo, hi) == 1


def test_monotone_relative_value_derivative_identity():
    # Partial A derivative is the total derivative at a stationary v.
    ratio_derivative = 2*sp.diff(b.P, b.A)/b.P-4/(1+b.A)
    assert sp.factor(ratio_derivative-4*b.v*(b.A-3-b.v)/((1+b.A)*b.P)) == 0
    # v_plus>A-3 for every A>5, since sqrt(A(A-5)) > A-5.
    assert sp.expand(b.A*(b.A-5)-(b.A-5)**2) == 5*(b.A-5)
    assert b.D.subs({b.A: 5, b.v: 2})/(1+5)**4 == sp.Rational(4, 3)


def test_rounded_parameter_is_not_the_exact_algebraic_tie():
    a = float(sp.sqrt(b.critical_root()).evalf(45))
    for x in (np.nextafter(a, -np.inf), a, np.nextafter(a, np.inf)):
        assert b.K.subs(b.A, sp.Rational(float(x))**2) != 0
    assert b.symmetric_classification(sp.Rational(float(np.nextafter(a, -np.inf))))["status"] == "unique_zero"
    assert b.symmetric_classification(sp.Rational(float(np.nextafter(a, np.inf))))["status"] == "symmetric_pair_lower"


@pytest.mark.parametrize("a", [3., 4., 3.249505202119775-1e-12, 3.249505202119775+1e-12])
def test_selector_and_complete_rational_oracle_near_boundary(a):
    x = np.array([-a, -a, 0., a, a])
    o = selected_mle_oracle(x, 1., fast=True)
    m, info = global_mle(x, 1., return_diagnostics=True)
    expected = b.symmetric_classification(sp.Rational(a))
    assert abs(m-float(sp.N(expected["selected"], 60))) <= 2e-13
    assert abs(m-o["selected"]) <= 2e-13
    np.testing.assert_allclose(stationary_points(x, 1.), o["roots"], rtol=0, atol=2e-13)
    if expected["status"] == "symmetric_pair_lower":
        assert info["exact_tie"]  # two tied outer modes, not a triple tie


@pytest.mark.parametrize("delta", [-2.**-30, 2.**-30])
def test_asymmetry_splits_outer_ties_and_matches_oracle(delta):
    a = 3.2495052021197754
    x = np.array([-a, -a, delta, a, a])
    o = selected_mle_oracle(x, 1., fast=True)
    m, info = global_mle(x, 1., return_diagnostics=True)
    assert abs(m-o["selected"]) <= 2e-13
    assert not info["exact_tie"]
    assert np.sign(m) == np.sign(delta)


def test_score_tangent_asymmetry_has_opposite_mode_value_derivatives():
    # At the symmetric exact tie, an exact tangent h=(1,1,-4p,1,1)
    # is distinct from the symmetric a-direction; envelope derivatives of
    # the two outer log-likelihood gains are opposite and nonzero.
    p = (1-b.A)/(1+b.A)**2
    h = [1, 1, -4*p, 1, 1]
    gradients = [p, p, 1, p, p]
    assert sp.expand(sum(g*t for g, t in zip(gradients, h))) == 0
    j = 1+4*p
    # Using sum psi(x_i-s)=0, the weighted sum is (h_center-1)*psi(-s).
    weighted = (-4*p-1)*(-b.s/(1+b.s*b.s))
    assert sp.simplify(weighted-j*b.s/(1+b.s*b.s)) == 0


def test_requested_sub_float_precision_fails_explicitly():
    with pytest.raises(StudentMLEError, match="output_precision"):
        global_mle([-4., -4., 0., 4., 4.], 1., location_tol=1e-30)
