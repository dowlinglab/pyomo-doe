"""Reduced quadratic heat-loss diagnostic, nested in the transport model.

theta*z'=u(t-d)-z; tau*v'=-v-beta*v²+K*z; T=Tref+delta+v.
Initial z=0,v=-delta. beta>=0 is phenomenological increased heat loss, not an
independently identified radiation coefficient. theta remains an input lag;
moving a sensor lag past the nonlinear balance would be a different model.
"""
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp


def stable_gradient(function,point,bounds,step=2e-6):
    """Keep full one-sided steps near active bounds; avoid tiny central steps."""
    result=[];f0=None
    for j,(lo,hi) in enumerate(bounds):
        h=step
        if lo==0 and hi==1:h=min(h,max(1e-8,1e-4*min(point[j],1-point[j])))
        unit=np.zeros(len(point));unit[j]=1.
        if point[j]-lo>=h and hi-point[j]>=h:
            value=(function(point+h*unit)-function(point-h*unit))/(2*h)
        else:
            if f0 is None:f0=function(point)
            if hi-point[j]<h:
                h=min(h,(point[j]-lo)/2)
                value=(3*f0-4*function(point-h*unit)+function(point-2*h*unit))/(2*h)
            else:
                h=min(h,(hi-point[j])/2)
                value=(-3*f0+4*function(point+h*unit)-function(point+2*h*unit))/(2*h)
        result.append(value)
    return np.asarray(result)


@njit(cache=True)
def advance(z,v,held,duration,K,tau,theta,beta,max_step):
    if beta==0:
        h=duration;a=h*(1/tau-1/theta);et=np.exp(-h/tau);eh=np.exp(-h/theta)
        if abs(a)<1e-5:cross=et*h/tau*(1+a/2+a*a/6+a*a*a/24+a**4/120)
        else:cross=theta/(theta-tau)*(eh-et)
        return eh*z+held*(-np.expm1(-h/theta)),et*v+K*held*(-np.expm1(-h/tau))+K*(z-held)*cross
    left=duration
    while left>1e-12:
        h=min(left,max_step,tau/50)
        if abs(z-held)>1e-6*max(1.,abs(held)):h=min(h,theta/4)
        # Strang split: nonlinear heat loss exactly, then exact linear forced
        # two-state response, then nonlinear loss. beta=0 is exact for any h.
        v=v/(1+beta*v*h/(2*tau))
        a=h*(1/tau-1/theta);et=np.exp(-h/tau);eh=np.exp(-h/theta)
        if abs(a)<1e-5:cross=et*h/tau*(1+a/2+a*a/6+a*a*a/24+a**4/120)
        else:cross=theta/(theta-tau)*(eh-et)
        v=et*v+K*held*(-np.expm1(-h/tau))+K*(z-held)*cross
        z=eh*z+held*(-np.expm1(-h/theta))
        v=v/(1+beta*v*h/(2*tau));left-=h
    return z,v


@njit(cache=True)
def predict_rise(time,command,K,tau,theta,delay,beta,ambient,max_step=.1):
    output=np.zeros(len(time));z=0.;v=-ambient;held=0.;event=0;now=time[0]
    for k in range(1,len(time)):
        target=time[k]
        while event<len(time) and time[event]+delay<=target:
            endpoint=time[event]+delay
            z,v=advance(z,v,held,max(0.,endpoint-now),K,tau,theta,beta,max_step)
            now=endpoint;held=command[event];event+=1
        z,v=advance(z,v,held,max(0.,target-now),K,tau,theta,beta,max_step)
        now=target;output[k]=ambient+v
    return output


def reference(time,command,K,tau,theta,delay,beta,ambient):
    shifted=time+delay
    grid=np.unique(np.r_[time,shifted[(shifted>=time[0])&(shifted<=time[-1])]])
    state=np.array([0.,-ambient]);values=[0.]
    for a,b in zip(grid[:-1],grid[1:]):
        index=np.searchsorted(time,(a+b)/2-delay,side='right')-1
        held=command[index] if index>=0 else 0.
        f=lambda s,x:np.array([(held-x[0])/theta,(-x[1]-beta*x[1]**2+K*x[0])/tau])
        sol=solve_ivp(f,(0.,b-a),state,rtol=2e-11,atol=1e-12,method='DOP853')
        if not sol.success:raise RuntimeError(sol.message)
        state=sol.y[:,-1];values.append(ambient+state[1])
    return np.asarray(values)[np.searchsorted(grid,time)]


def audit(max_step=.1):
    from transport import delayed_unit
    rng=np.random.RandomState(1232)
    time=np.r_[0.,np.cumsum(rng.uniform(.4,3.,60))];u=rng.uniform(0,100,len(time))
    rows=[]
    bounds=[(-2.,2.),(0.,1.)]
    for x in (np.array([-2.+1e-12,.02]),np.array([2.-1e-12,.98])):
        actual=stable_gradient(lambda q:np.sum((q-.3)**2),x,bounds)
        if np.max(abs(actual-2*(x-.3)))>1e-7:raise AssertionError('Boundary gradient')
    for K,tau,theta,delay,beta,ambient in [(.71,140.,16.,2.5,0.,-1.3),(.8,140.,16.,2.5,.002,1.7),(.8,140.,16.,0.,.006,-1.5),(.7,80.,80.,5.,.003,.5)]:
        p=predict_rise(time,u,K,tau,theta,delay,beta,ambient,max_step)
        q=reference(time,u,K,tau,theta,delay,beta,ambient)
        error=float(np.max(abs(p-q)))
        if error>3e-5:raise AssertionError(('Splitting error',error,max_step))
        if beta==0:
            exact=K*delayed_unit(time,u,tau,theta,delay)+ambient*(-np.expm1(-(time-time[0])/tau))
            if np.max(abs(p-exact))>2e-8:raise AssertionError('Nested linear response')
        rows.append({'beta':beta,'tau':tau,'max_abs_C':error})
    return {'status':'passed','max_step_s':max_step,'cases':rows,
            'scope':'Independent high-accuracy ODE and exact beta0 nesting at declared diagnostic values; selected fits require their own convergence/ODE audit'}
