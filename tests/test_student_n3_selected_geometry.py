import numpy as np
import pytest
import sympy as sp

from models.loc.student import selected_block as block
from reporting.diagnostics import student_n3_selected_geometry as geo
from reporting.diagnostics.verify_student_n3_components import chambers, chamber_graph, connected_components


def test_exact_discriminant_certificate_and_origin():
    assert geo.discriminant_certificate()['positive_coefficients']
    _,q=geo.exact_quartic([0,0,0])
    assert q.as_expr()==q.gen**4+3*q.gen**2+3
    assert q.count_roots(-sp.oo,sp.oo)==0


def test_general_sextic_identity_retains_nonstationary_linear_term():
    a,b,c,s=sp.symbols('a b c s',real=True)
    e1=a+b+c;e2=a*b+a*c+b*c;e3=a*b*c
    n=e1*(1+e2)+e3*(e2-3)
    q=s**4-2*e1*s**3+(3+e1**2+2*e2)*s*s-2*(e1*(2+e2)+e3)*s+3+2*e1**2+e2**2+2*e1*e3
    d=sp.prod(1+(v-s)**2 for v in [a,b,c])
    assert sp.expand(d-d.subs(s,0)+2*n*s-s*s*q)==0


@pytest.mark.parametrize('node',chambers())
def test_exact_rational_chamber_witness_and_sturm_comparison(node):
    y=geo.rational_witness(node);d,q=geo.exact_quartic(y)
    assert geo.chamber(np.array(y,dtype=float))==node
    assert sum(v/(1+v*v) for v in y)==0
    e2=sum(y[i]*y[j] for i in range(3) for j in range(i+1,3))
    if geo.component(node)=='C0':
        assert e2<=0 and q.count_roots(-sp.oo,sp.oo)==0 and q.eval(0)>0
    else:
        assert e2>3 and q.count_roots(-sp.oo,sp.oo)>=2
        # Exact rational point between isolated simple roots verifies Q<0.
        intervals=q.intervals(eps=sp.Rational(1,10**8))
        mids=[(intervals[j][0][1]+intervals[j+1][0][0])/2 for j in range(len(intervals)-1)]
        assert any(q.eval(s)<0 for s in mids)
    assert sp.expand(d.as_expr()-d.eval(0)-q.gen**2*q.as_expr())==0


def test_independent_fold_feasibility_recovers_graph():
    graph=chamber_graph();edges=[]
    for node in chambers():
        bits,signs=node
        for i in range(3):
            for sign in (-1,1):
                if bits[i] and signs[i]!=sign:continue
                # Interior of the fold edge has both other scores opposite sign.
                if any(bits[j] and signs[j]!=-sign for j in range(3) if j!=i):continue
                bb=list(bits);ss=list(signs);bb[i]=1-bb[i];ss[i]=sign if bb[i] else 0
                target=(tuple(bb),tuple(ss))
                assert target in graph[node]
                edges.append(frozenset([node,target]))
    assert len(set(edges))==24
    assert sorted(map(len,connected_components(graph)))==[2,2,2,2,2,2,13]


def test_rounded_or_nonstationary_inputs_are_not_exact_factorizations():
    with pytest.raises(ValueError,match='exact symbolic'):geo.exact_quartic([0.,0.,0.])
    with pytest.raises(ValueError,match='linear term'):geo.exact_quartic([0,0,sp.Rational(1,10**20)])


@pytest.mark.parametrize('k',[1,2,3])
def test_likelihood_geometry_scaling_is_exact(k):
    y=geo.rational_witness(((1,0,1),(-1,0,1)))
    s=sp.Symbol('s',real=True);r=sp.sqrt(k)
    physical=sp.prod(k+(r*v-r*s)**2 for v in y)
    assert sp.expand(physical-k**3*sp.prod(1+(v-s)**2 for v in y))==0


@pytest.mark.parametrize('k,m',[(1.,4),(2.,16),(3.,16)])
def test_inner_trace_is_rng_identical_to_unchanged_block(k,m):
    x=np.sqrt(k)*np.array(geo.rational_witness(((0,0,0),(0,0,0))),dtype=float)
    block.seed_score_rng(753);a=block.score_block_kernel(x,.35,0.,k,.1,m)
    block.seed_score_rng(753);b=geo.traced_score_block(x,.35,0.,k,.1,m)
    for left,right in zip(a,b[:5]):np.testing.assert_array_equal(left,right)
    assert b[5].shape==(m+1,3)
    assert np.max(np.abs(np.sum(b[5]/(k+b[5]**2),axis=1)))<1e-13


def test_discrete_rattle_projection_can_cross_components_and_selected_step_rejects():
    from models.loc.student import selected_rattle as rattle
    x=np.zeros(3);p=np.array([-4.,-2.,6.])
    opts=rattle.options({'rattle_step_size':1.,'rattle_num_steps':1})
    y,q,ok,d=rattle.phase_proposal(x,p,0.,1.,opts)
    assert ok and geo.component(geo.chamber(y))=='T3-'
    assert d['reverse_position_error']<1e-12 and d['reverse_momentum_error']<1e-12
    class FixedMomentum:
        def standard_normal(self,n):return p.copy()
        def random(self):return 0.
    retained,state,diag=rattle.rattle_step(x,0.,0.,1.,FixedMomentum(),opts)
    assert not diag['accepted'] and diag['rejection']=='nonpositive_curvature'
    np.testing.assert_array_equal(retained,x)
