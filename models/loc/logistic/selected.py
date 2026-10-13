"""Selected logistic location target and regular numerical machinery (unit scale)."""
import math
import numpy as np
from numba import njit
from models.shared.selected_numerics import mh_accept, truncnorm_log_mass
from models.loc.student.selected_block import truncated_score_draw


@njit(cache=True)
def geometry(x, t):
    z = np.tanh((x-t)*.5)
    e = np.exp(-np.abs(x-t))
    a = 2*e/(1+e)**2
    c, j, gram = np.sum(z), np.sum(a), np.dot(a,a)
    if not np.isfinite(x-t).all() or not math.isfinite(gram) or gram <= 0 or j <= 0:
        raise FloatingPointError('logistic geometry unresolved')
    return c, j, gram, z, a


@njit(cache=True)
def loglik(x, mu):
    value = 0.
    for v in x:
        d = abs(v-mu)
        value += -d-2*math.log1p(math.exp(-d))
    return value


@njit(cache=True)
def mu_step(mu, x, sd, pm, ps, noise, u):
    candidate = mu+sd*noise
    ratio = loglik(x,candidate)-loglik(x,mu)-.5*((candidate-pm)/ps)**2+.5*((mu-pm)/ps)**2
    accepted = mh_accept(ratio,u)
    return (candidate if accepted else mu), accepted


@njit(cache=True)
def q_logpdf(z, loc):
    if not -1 < z < 1:
        raise FloatingPointError('logistic inverse chart unresolved')
    y = 2*math.atanh(z); d = abs(y-loc)
    return -d-2*math.log1p(math.exp(-d))+math.log(2.)-math.log1p(-z*z)


@njit(cache=True)
def pair_log_ratio(z, zp, delta, loc, sigma):
    lo, hi = max(-1.,delta-1.), min(1.,delta+1.)
    return (q_logpdf(zp,loc)+q_logpdf(delta-zp,loc)-q_logpdf(z,loc)-q_logpdf(delta-z,loc)
            +truncnorm_log_mass(z,sigma,lo,hi)-truncnorm_log_mass(zp,sigma,lo,hi))


@njit(cache=True)
def score_block(x, mu, t, sigma, length):
    out = x.copy(); accepted = 0
    for _ in range(length):
        i = np.random.randint(len(x)); j = np.random.randint(len(x)-1)
        if j >= i: j += 1
        z = math.tanh((out[i]-t)*.5); zj = math.tanh((out[j]-t)*.5); delta = z+zj
        lo,hi = max(-1.,delta-1.),min(1.,delta+1.)
        zp,_ = truncated_score_draw(z,sigma,lo,hi)
        if mh_accept(pair_log_ratio(z,zp,delta,mu-t,sigma),np.random.random()):
            a,b = t+2*math.atanh(zp),t+2*math.atanh(delta-zp)
            if max(abs(math.tanh((a-t)*.5)-zp),abs(math.tanh((b-t)*.5)-(delta-zp))) > 64*np.finfo(np.float64).eps:
                raise FloatingPointError('logistic inverse reconstruction unresolved')
            out[i],out[j] = a,b;accepted += 1
    return out,accepted


def check_release(x,t,tolerance=1e-10,score_tolerance=1e-10):
    if not np.isfinite([t,tolerance,score_tolerance]).all() or min(tolerance,score_tolerance)<=0:
        raise ValueError('positive finite tolerances and finite release required')
    c,j,g,_,_ = geometry(np.asarray(x),t)
    if abs(c)>score_tolerance:
        raise FloatingPointError('logistic score constraint not preserved')
    lo,hi = np.nextafter(t-tolerance,t),np.nextafter(t+tolerance,t)
    if not lo<t<hi:
        raise FloatingPointError('logistic release bracket unrepresentable')
    if max(np.nextafter(t-lo,np.inf),np.nextafter(hi-t,np.inf))>tolerance:
        raise FloatingPointError('logistic release enclosure unresolved')
    def sign_at(v):
        terms=np.tanh((np.asarray(x,dtype=np.longdouble)-np.longdouble(v))/2)
        value=terms.sum(); error=64*np.finfo(np.longdouble).eps*np.abs(terms).sum()
        if abs(value)>error:return int(np.sign(value))
        # Selective high precision for cancellation; never substitute a residual/curvature estimate.
        import mpmath as mp
        with mp.workdps(80):
            val=mp.fsum(mp.tanh((mp.mpf(float(a))-mp.mpf(float(v)))/2) for a in x)
            if abs(val)<mp.mpf('1e-60'):raise FloatingPointError('logistic score sign unresolved')
            return int(mp.sign(val))
    if sign_at(lo)>0 and sign_at(hi)<0:return j,g
    # No silent rejection based on an inward bracket or ill-resolved root.
    raise FloatingPointError('logistic released root not enclosed within tolerance')


@njit(cache=True)
def potential_force(x,mu,t):
    _,_,g,z,a = geometry(x,t)
    return -loglik(x,mu)+.5*math.log(g), np.tanh((x-mu)*.5)-z*a*a/g


@njit(cache=True)
def trajectory(x,p,mu,t,h,steps,proj_tol=1e-13):
    q=x.copy();p=p.copy()
    for _ in range(steps):
        _,force=potential_force(q,mu,t); _,_,g,_,normal=geometry(q,t)
        free=q+h*(p-.5*h*force); lam=0.;new=free.copy();ok=False
        for _ in range(50):
            c,_,gn,_,nn=geometry(new,t)
            transverse=np.dot(nn,normal)
            if abs(transverse)<=1e-12*math.sqrt(g*gn):return q,p,False
            if abs(c)<=proj_tol:ok=True;break
            lam += c/(h*transverse);new=free-h*lam*normal
        if not ok:return q,p,False
        ph=(new-q)/h
        _,force=potential_force(new,mu,t)
        p=ph-.5*h*force
        _,_,gn,_,nn=geometry(new,t)
        p-=nn*(np.dot(nn,p)/gn)
        q=new
    return q,p,True
