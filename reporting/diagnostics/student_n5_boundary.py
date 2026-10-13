"""Exact algebra for (-a,-a,0,a,a), normalized Student location; research only.

No production selector is imported. All polynomial inputs below are exact.
The complete classification proof is in student_n5_selection_boundary.md.
"""
from __future__ import annotations

import sympy as sp

A, v, s = sp.symbols("A v s")
P = v**2 + 2*(1-A)*v + (1+A)**2
R = 5*v**2 + (10-6*A)*v + (A-1)**2 + 4
D = (1+v)*P**2
H = sp.cancel((D-(1+A)**4)/v).expand()
K = 256*A**5-1973*A**4-7292*A**3-4310*A**2-1100*A-125
L = 256*v**5-1009*v**4-7744*v**3-14240*v**2-10240*v-2560
N = 64*A**4+82*A**3-50*A**2+950*A+250
B = 64*A**3+335*A**2-550*A-125


def critical_root():
    """The third real root of K, the only positive root; no float conversion."""
    return sp.CRootOf(K, 2)


def symmetric_classification(a):
    """Exact global minimizers for exact rational a>1.

    Rational a cannot be the algebraic three-way tie (K is irreducible).
    This independent family-specific oracle is used alongside the existing
    complete rational-data oracle, not in any sampler.
    """
    a = sp.Rational(a)
    if a <= 1:
        raise ValueError("this theorem requires normalized a>1")
    aa = a*a
    if aa <= 5:
        return {"global_locations": [sp.S.Zero], "selected": sp.S.Zero,
                "status": "unique_zero", "K_sign": sp.sign(K.subs(A, aa))}
    plus = (3*aa-5+2*sp.sqrt(aa*(aa-5)))/5
    if K.subs(A, aa) < 0:
        return {"global_locations": [sp.S.Zero], "selected": sp.S.Zero,
                "status": "unique_zero", "K_sign": -1}
    if K.subs(A, aa) > 0:
        root = sp.sqrt(plus)
        return {"global_locations": [-root, root], "selected": -root,
                "status": "symmetric_pair_lower", "K_sign": 1}
    raise ArithmeticError("unexpected rational root of irreducible K")


def decimal_enclosure(poly, digits=32, positive=True):
    """Return outward rational decimal endpoints, verified by Sturm counting."""
    poly = sp.Poly(poly)
    intervals = poly.intervals(eps=sp.Rational(1, 10**(digits+5)))
    candidates = [(lo, hi) for ((lo, hi), _) in intervals if lo > 0]
    if not positive or len(candidates) != 1:
        raise ValueError("exactly one positive root required")
    lo, hi = candidates[0]
    scale = 10**digits
    lower = sp.Rational(sp.floor(lo*scale), scale)
    upper = sp.Rational(sp.floor(hi*scale)+1, scale)
    assert poly.count_roots(lower, upper) == 1
    assert poly.eval(lower)*poly.eval(upper) < 0
    return lower, upper


def algebraic_certificate():
    """Exact elimination, root isolation, and the algebraic triple-tie oracle.

    The two substitution numerators vanish modulo K, proving stationarity
    and equal objective without feeding rounded algebraic data to a selector.
    Interval signs identify that stationary root as v_plus, not v_minus.
    """
    result_v = sp.factor(sp.resultant(R, H, v))
    result_A = sp.factor(sp.resultant(R, H, A))
    assert sp.expand(result_v+(A*A-2*A+5)*K) == 0
    assert sp.expand(result_A+v*v*L) == 0
    r_num = sp.together(R.subs(v, N/B)).as_numer_denom()[0]
    h_num = sp.together(H.subs(v, N/B)).as_numer_denom()[0]
    assert sp.rem(r_num, K, A) == 0
    assert sp.rem(h_num, K, A) == 0
    aa = sp.Poly(K, A)
    vv = sp.Poly(L, v)
    assert aa.is_irreducible and vv.is_irreducible
    assert aa.count_roots(0, sp.oo) == vv.count_roots(0, sp.oo) == 1
    bounds = {"A_c": decimal_enclosure(aa),
              "a_c": decimal_enclosure(sp.Poly(K.subs(A, s*s), s)),
              "v_c": decimal_enclosure(vv),
              "s_c": decimal_enclosure(sp.Poly(L.subs(v, s*s), s))}
    # Independent rational interval proof that N(A_c)/B(A_c) is the
    # larger R root. Polynomial signs over [10,11] suffice for this check:
    # N/B - (3A-5)/5 > 0 and B>0 on this whole interval.
    numerator = sp.expand(5*N-(3*A-5)*B)
    assert sp.Poly(B.subs(A, s+10), s).all_coeffs()[0] > 0
    assert all(c > 0 for c in sp.Poly(B.subs(A, s+10), s).all_coeffs())
    assert all(c > 0 for c in sp.Poly(numerator.subs(A, s+10), s).all_coeffs())
    ac = critical_root()
    vc = N.subs(A, ac)/B.subs(A, ac)
    # Link the unique positive L root to the exact K-field value explicitly.
    l_num = sp.together(L.subs(v, N/B)).as_numer_denom()[0]
    assert sp.rem(l_num, K, A) == 0
    return {
        "polynomials": {name: str(expr) for name, expr in
            (("P", P), ("R", R), ("H", H), ("K", K), ("L", L), ("N", N), ("B", B))},
        "resultant_v": str(result_v), "resultant_A": str(result_A),
        "R_discriminant": str(sp.factor(sp.discriminant(R, v))),
        "K_irreducible": True, "L_irreducible": True,
        "K_all_real_intervals": [[str(lo), str(hi), int(mult)] for ((lo, hi), mult)
                                 in aa.intervals(eps=sp.Rational(1, 10**32))],
        "L_all_real_intervals": [[str(lo), str(hi), int(mult)] for ((lo, hi), mult)
                                 in vv.intervals(eps=sp.Rational(1, 10**32))],
        "Sturm_counts": {"K_negative": int(aa.count_roots(-sp.oo, 0)),
                         "K_positive": int(aa.count_roots(0, sp.oo)),
                         "K_10_11": int(aa.count_roots(10, 11)),
                         "L_negative": int(vv.count_roots(-sp.oo, 0)),
                         "L_positive": int(vv.count_roots(0, sp.oo))},
        "enclosures": {key: {"rational": list(map(str, pair)),
                              "decimal": [str(sp.N(z, 40)) for z in pair],
                              "sturm_root_count": 1} for key, pair in bounds.items()},
        "decimal_approximations": {"A_c": str(ac.evalf(45)),
                                   "a_c": str(sp.sqrt(ac).evalf(45)),
                                   "v_c": str(vc.evalf(45)),
                                   "s_c": str(sp.sqrt(vc).evalf(45))},
        "exact_algebraic_oracle": {
            "A": "unique positive real root of K; CRootOf(K,2)",
            "v": "N(A)/B(A), the larger positive root of R",
            "global_locations": ["-sqrt(N(A)/B(A))", "0", "sqrt(N(A)/B(A))"],
            "selected": "-sqrt(N(A)/B(A))",
            "stationarity_remainder_mod_K": "0", "equal_objective_remainder_mod_K": "0",
            "submitted_to_binary_float_selector": False,
        },
    }
