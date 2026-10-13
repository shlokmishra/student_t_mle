"""Independent small-sample Student location oracle for validation only.

Requires SymPy 1.14.0. Input float values are treated as exact binary rationals.
All real stationary roots are isolated from the derivative of the likelihood
denominator polynomial. Rational objective bounds separate the global winner;
only an exactly symmetric pair is allowed to be reported as a proved tie.
This intentionally does not import the production MLE implementation.
"""

from __future__ import annotations

import math
import numpy as np
import sympy as sp


def _product_bounds(values, k, left, right):
    lower = upper = sp.S.One
    for value in values:
        nearest = 0 if left <= value <= right else min((value-left)**2, (value-right)**2)
        farthest = max((value-left)**2, (value-right)**2)
        lower *= k + nearest
        upper *= k + farthest
    return lower, upper


def selected_mle_oracle(data, k, *, digits=16, fast=False):
    """Return a bounded global selection or raise; never discard failed draws."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all() or not math.isfinite(k) or k <= 0:
        raise ValueError("finite nonempty data and positive finite k required")
    values = sorted(sp.Rational(float(v)) for v in x)
    kk = sp.Rational(float(k))
    m = sp.Symbol("m")
    polynomial = sp.Poly(1, m, domain=sp.QQ)
    for value in values:
        polynomial *= sp.Poly([1, -2*value, value*value+kk], m, domain=sp.QQ)
    derivative = polynomial.diff()
    center = (values[0] + values[-1])/2
    symmetric = all(values[i] + values[-1-i] == 2*center for i in range(len(values)))
    for precision in (digits, max(32, digits*2), max(64, digits*4)):
        intervals = derivative.intervals(eps=sp.Rational(1, 10**precision), fast=fast)
        bounds = [_product_bounds(values, kk, a, b) for ((a,b), _) in intervals]
        best_upper = min(upper for _, upper in bounds)
        possible = [i for i, (lower, _) in enumerate(bounds) if lower <= best_upper]
        if len(possible) == 1:
            winner = possible[0]
            status = "unique_global_bound"
        elif symmetric and len(possible) == 2 and possible[0]+possible[1] == len(intervals)-1:
            winner = possible[0]
            status = "exact_symmetric_global_tie_lower"
        else:
            continue
        roots = [float((a+b)/2) for ((a,b), _) in intervals]
        root = roots[winner]
        mid = sum(intervals[winner][0])/2
        objective = sum(sp.log(1+(v-mid)**2/kk).evalf(60) for v in values)
        return {
            "selected": root,
            "loglik": float(-(kk+1)/2*objective),
            "roots": roots,
            "multiplicities": [int(mult) for _, mult in intervals],
            "winner_index": winner,
            "selection_status": status,
            "root_digits": precision,
            "selected_interval": [str(v) for v in intervals[winner][0]],
        }
    raise ArithmeticError("unresolved global objective comparison; no particle accepted")
