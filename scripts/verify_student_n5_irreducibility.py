"""Small independent finite-field certificate for the two critical quintics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sympy as sp
from reporting.diagnostics import student_n5_boundary as b


def certificates():
    results = {}
    for name, expression, variable in (("K", b.K, b.A), ("L", b.L, b.v)):
        poly = sp.Poly(expression, variable, modulus=7).monic()
        x = sp.Poly(variable, variable, modulus=7)
        residue = x
        frobenius = []
        for _ in range(5):
            residue = (residue**7).rem(poly)
            frobenius.append(str(residue.as_expr()))
        gcd = poly.gcd(x**7-x)
        assert residue == x and gcd.degree() == 0
        # For prime degree five these are the complete Rabin conditions:
        # x^(7^5)=x mod f and gcd(f,x^7-x)=1. Gauss gives Q irreducibility.
        results[name] = dict(prime=7, monic_polynomial=str(poly.as_expr()),
                             degree=poly.degree(), frobenius_residues=frobenius,
                             first_frobenius_gcd=str(gcd.as_expr()),
                             final_residue_equals_variable=True,
                             irreducible_over_Q=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        stream.write(json.dumps(certificates(), indent=2)+"\n")
