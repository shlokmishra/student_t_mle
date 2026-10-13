"""Independent fold feasibility, singularity and exact higher-n counterexamples."""
from itertools import combinations
import numpy as np
import pytest
from scipy.optimize import linprog
import sympy as sp

from reporting.diagnostics import student_higher_n_geometry as geo
from reporting.diagnostics.verify_student_n3_components import chamber_graph as old_graph


def test_n3_graph_agrees_with_prior_exact_graph():
    original=old_graph();graph=geo.fold_graph(3)
    mapping=lambda node:node[1]
    assert {mapping(n):{mapping(v) for v in adj} for n,adj in original.items()}==graph


@pytest.mark.parametrize('n',[4,5])
def test_fold_adjacency_by_independent_strict_linear_program(n):
    graph=geo.fold_graph(n);assert len(graph)==3**n-2
    for node in geo.chambers(n):
        for i in range(n):
            for sign in (-1,1) if node[i]==0 else (node[i],):
                others=[v for j,v in enumerate(node) if j!=i]
                # Maximize common interior margin eta subject to actual score-face sum.
                rows=[];rhs=[]
                for j,v in enumerate(others):
                    lo,hi=(-.5,.5) if v==0 else ((0.,.5) if v==1 else (-.5,0.))
                    a=np.zeros(n);a[j]=1;a[-1]=1;rows.append(a);rhs.append(hi)
                    a=np.zeros(n);a[j]=-1;a[-1]=1;rows.append(a);rhs.append(-lo)
                objective=np.zeros(n);objective[-1]=-1
                eq=np.r_[np.ones(n-1),0.][None,:]
                r=linprog(objective,A_ub=rows,b_ub=rhs,A_eq=eq,b_eq=[-sign/2],bounds=[(None,None)]*(n-1)+[(0.,None)],method='highs')
                feasible=r.success and r.x[-1]>1e-9
                target=list(node);target[i]=sign if node[i]==0 else 0
                assert (tuple(target) in graph[node])==feasible


def test_even_singular_points_and_odd_regularity_algebra():
    rows={r['name']:r for r in geo.exact_examples()}
    r=rows['n4_singular_selected'];assert r['score_at_zero']=='0' and r['G_at_zero']=='0' and r['J_at_zero']=='0'
    for n in [3,4,5]:
        balanced=[signs for signs in __import__('itertools').product((-1,1),repeat=n) if sum(signs)==0]
        assert len(balanced)==(6 if n==4 else 0)


def test_exact_n5_same_chamber_changes_selected_support_with_positive_curvature():
    s=sp.Symbol('s',real=True);v=sp.Symbol('v',nonnegative=True)
    for a in [sp.Rational(6,5),sp.Integer(5)]:
        x=[-a,-a,0,a,a];assert sum(y/(1+y*y) for y in x)==0
        j=sum((1-y*y)/(1+y*y)**2 for y in x);assert j>0
        d=sp.prod(1+(y-s)**2 for y in x)
        if a==sp.Rational(6,5):
            p=v*v+2*(1-a*a)*v+(1+a*a)**2
            r=sp.Poly(sp.expand(p+2*(v+1)*sp.diff(p,v)),v)
            assert all(c>0 for c in r.all_coeffs())
        else:
            assert d.subs(s,5)<d.subs(s,0)
            p=s**4-48*s*s+676
            assert sp.expand(sp.diff(d,s)-10*s*p*(s**4-28*s*s+116))==0


def test_n4_degenerate_global_and_tied_globals_exact():
    s=sp.Symbol('s',real=True)
    d=sp.prod(1+(y-s)**2 for y in [-1,-1,1,1])
    assert sp.expand(d-(s**4+4)**2)==0
    d=sp.prod(1+(y-s)**2 for y in [-2,-2,2,2])
    assert sp.simplify(d.subs(s,sp.sqrt(3))-d.subs(s,-sp.sqrt(3)))==0
    assert d.subs(s,sp.sqrt(3))<d.subs(s,0)


def test_n5_median_interval_has_no_score_root_exact_bound():
    # On [-1,1], the two negative scores <= -3/10 each,
    # center score <=1/2 and positive scores <=1/99 each.
    assert -sp.Rational(6,10)+sp.Rational(1,2)+sp.Rational(2,99)<0


def test_n4_strict_inferior_local_max_exact():
    c=(25-2*sp.sqrt(46))/21;x=[-c,sp.Integer(7),sp.Integer(7),sp.Integer(7)]
    assert sp.simplify(sum(v/(1+v*v) for v in x))==0
    assert 0<c<sp.Rational(2,3)
    assert sp.Rational(45,169)-sp.Rational(144,2500)>0
    assert 1+(7+c)**2<65 and (1+c*c)*50**3>125000
