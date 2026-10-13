"""Preliminary finite-fold graph and exact small counterexamples, research only.

Graph components are not asserted to classify the full higher-n topology.
"""
import itertools


def chambers(n):
    if n<3:raise ValueError('n>=3 required')
    result=[]
    for signs in itertools.product((-1,0,1),repeat=n):
        if all(v==1 for v in signs) or all(v==-1 for v in signs):continue
        result.append(signs)  # zero is an inner branch, not a zero observation
    return result


def fold_graph(n):
    nodes=chambers(n);graph={node:set() for node in nodes}
    for node in nodes:
        for i in range(n):
            for sign in (-1,1) if node[i]==0 else (node[i],):
                same=sum(v==sign for j,v in enumerate(node) if j!=i)
                if same>n-3:continue
                target=list(node);target[i]=sign if node[i]==0 else 0;target=tuple(target)
                if target not in graph:raise ArithmeticError('infeasible fold edge')
                graph[node].add(target)
    return graph


def graph_summary(n):
    graph=fold_graph(n);unseen=set(graph);components=[]
    while unseen:
        start=unseen.pop();part={start};front=[start]
        while front:
            for other in graph[front.pop()]:
                if other in unseen:unseen.remove(other);part.add(other);front.append(other)
        components.append(sorted(part))
    return dict(n=n,chambers=len(graph),edges=sum(map(len,graph.values()))//2,
                component_sizes=sorted(map(len,components),reverse=True),components=components,
                qualification='relative-open single-fold adjacency only; not a full manifold/topology classification')


def exact_examples():
    import sympy as sp
    s=sp.Symbol('s',real=True)
    examples=[('n4_singular_selected',[-1,-1,1,1]),('n4_two_global_ties',[-2,-2,2,2]),
              ('n4_inferior_local_max',[-(25-2*sp.sqrt(46))/21,7,7,7]),
              ('n5_valid_same_chamber',[-sp.Rational(6,5),-sp.Rational(6,5),0,sp.Rational(6,5),sp.Rational(6,5)]),
              ('n5_inferior_local_and_two_global_ties',[-5,-5,0,5,5]),
              ('n5_no_median_interval_root',[-2,-2,0,100,100])]
    rows=[]
    for name,x in examples:
        x=list(map(sp.sympify,x))
        d=sp.Poly(sp.prod(1+(v-s)**2 for v in x),s)
        score=sp.simplify(sum(v/(1+v*v) for v in x))
        j=sp.simplify(sum((1-v*v)/(1+v*v)**2 for v in x))
        g=sp.simplify(sum(((1-v*v)/(1+v*v)**2)**2 for v in x))
        rows.append(dict(name=name,exact_data=list(map(str,x)),D=str(sp.factor(d.as_expr())),
                         D_prime=str(sp.factor(d.diff().as_expr())),score_at_zero=str(score),J_at_zero=str(j),G_at_zero=str(g)))
    return rows
