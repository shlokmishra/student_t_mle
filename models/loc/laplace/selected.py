"""Selected midpoint-median geometry and conditionals, in ordered coordinates."""
import math
import numpy as np
from numba import njit
from models.shared.selected_numerics import mh_accept


@njit(cache=True)
def open_uniform():
    u = np.random.random()
    while u == 0.:u = np.random.random()
    return u


@njit(cache=True)
def sample_left(mu, upper, b):
    d=(upper-mu)/b
    if not math.isfinite(d):raise FloatingPointError('Laplace truncation distance unresolved')
    if d<=0:
        value=upper+b*math.log(open_uniform())
    else:
        mass=1.-.5*math.exp(-d)
        if np.random.random()<.5/mass:
            value=mu+b*math.log(open_uniform())
        else:
            value=mu-b*math.log1p(-open_uniform()*(-math.expm1(-d)))
    if not math.isfinite(value) or not value<upper:raise FloatingPointError('Laplace truncated draw cannot be represented')
    return value


@njit(cache=True)
def sample_right(mu,lower,b):
    value=mu-sample_left(0.,mu-lower,b)
    if not math.isfinite(value) or not value>lower:raise FloatingPointError('Laplace right draw cannot be represented')
    return value


@njit(cache=True)
def draw_middle_gap(mu,t,b,upper):
    if upper<=0:raise FloatingPointError('empty central-gap support')
    a=abs(mu-t)
    if upper<=a:
        d=upper*open_uniform()
    else:
        mass=-math.expm1(-2*(upper-a)/b)
        tail=b*.5*mass
        if np.random.random()<a/(a+tail):d=a*open_uniform()
        else:d=a-b*.5*math.log1p(-open_uniform()*mass)
    if not 0<d<upper or not math.isfinite(d):raise FloatingPointError('central gap draw unresolved')
    return d


@njit(cache=True)
def gibbs_latent(q,mu,t,b):
    n=len(q);h=n//2;out=q.copy()
    if n%2:
        lower=upper=t;left_count=h;right_start=h+1;out[h]=t
    else:
        bound=np.inf
        if h>1:bound=min(t-q[h-2],q[h+1]-t)
        d=draw_middle_gap(mu,t,b,bound)
        lower,upper=t-d,t+d
        if not lower<t<upper:raise FloatingPointError('central pair reconstruction unresolved')
        out[h-1],out[h]=lower,upper;left_count=h-1;right_start=h+1
    for i in range(left_count):out[i]=sample_left(mu,lower,b)
    for i in range(right_start,n):out[i]=sample_right(mu,upper,b)
    return np.sort(out)


@njit(cache=True)
def tangent(v):
    p=v.copy();h=len(v)//2
    if len(v)%2:p[h]=0.
    else:
        a=(p[h-1]+p[h])*.5;p[h-1]-=a;p[h]-=a
    return p


@njit(cache=True)
def potential(q,mu,b):
    return np.sum(np.abs(q-mu))/b


@njit(cache=True)
def mu_step(mu,q,sd,pm,ps,b,noise,u):
    candidate=mu+sd*noise
    ratio=potential(q,mu,b)-potential(q,candidate,b)-.5*((candidate-pm)/ps)**2+.5*((mu-pm)/ps)**2
    accept=mh_accept(ratio,u)
    return (candidate if accept else mu),accept


@njit(cache=True)
def reflected_drift(q,p,duration,max_events=1000):
    """Specular drift in the ordered polyhedron, within the median tangent plane."""
    q=q.copy();p=p.copy();remaining=duration;events=0
    while remaining>0:
        collision=np.inf;wall=-1;second=np.inf
        for j in range(len(q)-1):
            gap=q[j+1]-q[j];velocity=p[j+1]-p[j]
            if gap<0:raise FloatingPointError('ordered boundary roundoff unresolved')
            if velocity<0:
                when=gap/(-velocity)
                if when<collision:second=collision;collision=when;wall=j
                else:second=min(second,when)
        if collision>remaining:
            q+=remaining*p;remaining=0.;break
        if abs(second-collision)<=32*np.finfo(np.float64).eps*max(1.,abs(collision)):
            raise FloatingPointError('simultaneous Laplace wall collision unresolved')
        q+=collision*p;remaining-=collision
        normal=np.zeros(len(q));normal[wall]=1.;normal[wall+1]=-1.;normal=tangent(normal)
        norm=np.dot(normal,normal)
        if norm<=0:raise FloatingPointError('zero reflected wall normal')
        # The exact drift hits this wall. Round its computed contact to the
        # intersection of the wall and median plane, only within arithmetic
        # error. Do not clip a materially out-of-order state into the domain.
        contact=q.copy();h=len(q)//2
        residual=q[wall]-q[wall+1]
        q-=normal*residual/norm
        if len(q)%2 and wall==h-1:q[h-1]=q[h]
        elif len(q)%2 and wall==h:q[h+1]=q[h]
        elif len(q)%2==0 and wall in (h-2,h-1,h):
            middle=contact[h-1]*.5+contact[h]*.5
            if wall==h-1:q[h-1]=middle;q[h]=middle
            elif wall==h-2:
                q[h-2]=q[h-1];q[h]=middle+(middle-q[h-1])
            else:
                q[h+1]=q[h];q[h-1]=middle+(middle-q[h])
        else:
            value=q[wall]*.5+q[wall+1]*.5;q[wall]=value;q[wall+1]=value
        if np.max(np.abs(q-contact))>64*np.finfo(np.float64).eps*max(1.,np.max(np.abs(contact))):
            raise FloatingPointError('Laplace wall contact error exceeds rounding bound')
        p-=2*np.dot(normal,p)/norm*normal
        events+=1
        if events>max_events:raise FloatingPointError('Laplace collision budget exhausted')
    return q,p,events


@njit(cache=True)
def trajectory(q,p,mu,b,step_size,steps):
    q=q.copy();p=p.copy();events=0
    for _ in range(steps):
        p-=.5*step_size*tangent(np.sign(q-mu)/b)
        q,p,e=reflected_drift(q,p,step_size);events+=e
        p-=.5*step_size*tangent(np.sign(q-mu)/b)
    return q,p,events


def check_state(x,t,tolerance=1e-10):
    q=np.asarray(x,dtype=float)
    if not np.isfinite(q).all() or np.any(np.diff(q)<0):raise FloatingPointError('finite ordered median state required')
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('positive finite median tolerance required')
    # Long-double midpoint avoids needless overflow in numpy's averaging formula.
    h=len(q)//2
    med=np.longdouble(q[h]) if len(q)%2 else (np.longdouble(q[h-1])+np.longdouble(q[h]))/2
    error=float(abs(med-np.longdouble(t)))
    if error>tolerance:raise FloatingPointError('selected midpoint-median constraint violated')
    return error
