"""Research-only n=3 classification and exact score/likelihood certificates.

No production selector or sampler imports this module. Numerical chamber labels
are diagnostics, not a substitute for checking the score constraint.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from numba import njit

from models.loc.student import selected_block as block


def chamber(x, k=1., release=0.):
    y = (np.asarray(x, dtype=float)-release)/math.sqrt(k)
    if y.shape != (3,) or not np.isfinite(y).all():
        raise ValueError('three finite observations required')
    bits = tuple(int(abs(v)>1.) for v in y)
    signs = tuple(int(np.sign(v)) if b else 0 for v,b in zip(y,bits))
    return bits, signs


def component(node):
    bits, signs = node
    for sign in (-1,1):
        same = [i for i in range(3) if bits[i] and signs[i]==sign]
        if len(same)==2:
            other = next(i for i in range(3) if i not in same)
            return f'T{other+1}{"+" if sign>0 else "-"}'
        if len(same)>2:
            raise ValueError('infeasible all-same outer signs on score zero')
    return 'C0'


def node_name(node):
    b,s=node
    return ''.join(map(str,b))+':'+''.join('+' if v>0 else '-' if v<0 else '.' for v in s)


def rational_witness(node):
    """Exact QQ points for every chamber, using a rational score triangle."""
    import sympy as sp
    # psi(-13/16)+psi(27/11)+psi(7)=0; reciprocal values give the other inverses.
    inners = (-sp.Rational(13,16),sp.Rational(11,27),sp.Rational(1,7))
    bits,signs=node
    for flip in (-1,1):
        for small in itertools.permutations(inners):
            y=tuple(flip*(1/v if b else v) for v,b in zip(small,bits))
            if all(not b or sp.sign(v)==s for v,b,s in zip(y,bits,signs)):
                return y
    raise ValueError('no witness for infeasible chamber')


def exact_quartic(y):
    """Demand exact inputs and stationarity; never drop a floating linear term."""
    import sympy as sp
    y=tuple(sp.sympify(v) for v in y)
    if len(y)!=3 or any(v.has(sp.Float) for v in y):
        raise ValueError('three exact symbolic/rational inputs required')
    s=sp.Symbol('s',real=True)
    d=sp.Poly(sp.prod(1+(v-s)**2 for v in y),s)
    linear=sp.simplify(d.nth(1))
    if linear!=0:
        raise ValueError('exact score zero required; linear term is '+str(linear))
    return d,sp.Poly(sum(d.nth(j)*s**(j-2) for j in range(2,7)),s)


def discriminant_certificate():
    """Symbolically verify the quartic/discriminant identities used in the proof."""
    import sympy as sp
    s,u,v,a=sp.symbols('s u v a',real=True)
    e3=-u*(1+v)/(v-3)
    q=s**4-2*u*s**3+(3+u*u+2*v)*s*s-2*(u*(2+v)+e3)*s+3+2*u*u+v*v+2*u*e3
    h=u**4*(v-3)**2+u*u*(-8*v**3+46*v*v-108*v+110)+16*v**4-104*v**3+193*v*v-78*v+9
    cert=16*(4*u*u+(v-3)**2)*(2*u*u*(3-v)+9*v*v+27)*h/(v-3)**4
    assert sp.factor(sp.discriminant(q,s)-cert)==0
    positive=sp.Poly(sp.expand(h.subs(v,-a)),u,a)
    assert all(c>0 for c in positive.coeffs())
    assert positive.nth(0,0)==9
    assert sp.factor(q.subs(s,0)-(3+v*v+8*u*u/(3-v)))==0
    return {'identity_verified':True,'Q':str(q),'discriminant':str(cert),
        'H_at_v_minus_a':str(positive.as_expr()),'positive_coefficients':True,
        'qualification':'v<=0; positive discriminant plus connectedness and the origin, not discriminant alone, gives no real roots'}


@njit(cache=True)
def traced_score_block(x,mu,t,k,sigma,length):
    """Diagnostic replay of the unchanged compiled block, retaining inner states."""
    proposed=x.copy();states=np.empty((length+1,len(x)));states[0]=proposed
    accept=draws=changed=0;error=0.
    pairs=np.empty((length,2),dtype=np.int64)
    for step in range(length):
        i=np.random.randint(len(x));j=np.random.randint(len(x)-1)
        if j>=i:j+=1
        pairs[step,0]=i;pairs[step,1]=j
        aa,dd,bb,ee=block.score_pair_kernel(proposed,i,j,mu,t,k,sigma)
        accept+=aa;draws+=dd;changed+=bb;error=max(error,ee);states[step+1]=proposed
    return proposed,accept,draws,changed,error,states,pairs
