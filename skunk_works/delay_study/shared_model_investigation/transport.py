"""Positive two-lag model with an additional explicit input transport delay."""
import numpy as np
from numba import njit
from scipy.linalg import expm
from pooled_models import PooledModel


@njit(cache=True)
def delayed_unit(time,command,tau,theta,delay):
    result=np.zeros(len(time))
    lag=0.;temperature=0.;held=0.;event=0;now=time[0]
    for k in range(1,len(time)):
        target=time[k]
        while event<len(time) and time[event]+delay<=target:
            endpoint=time[event]+delay
            h=max(0.,endpoint-now)
            a=h*(1/tau-1/theta);et=np.exp(-h/tau);eh=np.exp(-h/theta)
            if abs(a)<1e-5:
                cross=et*h/tau*(1+a/2+a*a/6+a*a*a/24+a**4/120)
            else:cross=theta/(theta-tau)*(eh-et)
            temperature=et*temperature+held*(-np.expm1(-h/tau))+(lag-held)*cross
            lag=eh*lag+held*(-np.expm1(-h/theta))
            now=endpoint;held=command[event];event+=1
        h=max(0.,target-now)
        a=h*(1/tau-1/theta);et=np.exp(-h/tau);eh=np.exp(-h/theta)
        if abs(a)<1e-5:
            cross=et*h/tau*(1+a/2+a*a/6+a*a*a/24+a**4/120)
        else:cross=theta/(theta-tau)*(eh-et)
        temperature=et*temperature+held*(-np.expm1(-h/tau))+(lag-held)*cross
        lag=eh*lag+held*(-np.expm1(-h/theta))
        now=target;result[k]=temperature
    return result


def independent_reference(time,command,tau,theta,delay):
    shifted=time+delay
    grid=np.unique(np.r_[time,shifted[(shifted>=time[0])&(shifted<=time[-1])]])
    A=np.array([[-1/theta,0.,1/theta],[1/tau,-1/tau,0.],[0.,0.,0.]])
    state=np.zeros(3);values=[0.];cache={}
    for a,b in zip(grid[:-1],grid[1:]):
        index=np.searchsorted(time,(a+b)/2-delay,side='right')-1
        state[2]=command[index] if index>=0 else 0.
        h=float(b-a)
        if h not in cache:cache[h]=expm(A*h)
        state=cache[h]@state;values.append(state[1])
    return np.asarray(values)[np.searchsorted(grid,time)]


class TransportModel(PooledModel):
    def __init__(self,records):
        super().__init__(records,'chain',1)
        self.kind='transport';self.nt=3

    def time_constants(self,coordinates):return np.exp(np.asarray(coordinates)[:2])

    def _design(self,coordinates):
        tau,theta=np.exp(coordinates[:2]);delay=coordinates[2]
        if delay<0:raise ValueError('Nonnegative delay required')
        X=np.zeros((self.count,1+self.ambient_width*len(self.records)));cursor=0
        for j,d in enumerate(self.records):
            count=len(d.time)-1;section=slice(cursor,cursor+count)
            X[section,0]=delayed_unit(d.time,d.input,tau,theta,delay)[1:]
            elapsed=d.time[1:]-d.time[0];a=-np.expm1(-elapsed/tau)
            if self.ambient_width==1:X[section,1+j]=a
            else:
                b=(elapsed-tau*a)/(d.time[-1]-d.time[0])
                X[section,1+2*j:1+2*j+2]=np.column_stack((a-b,b))
            cursor+=count
        return X

    def bounds(self):return [(np.log(5.),np.log(600.)),(np.log(.01),np.log(200.)),(0.,15.),(np.log(.05),np.log(3000.)),(0.,1.)]

    def frequency_response(self,point,coefficients,omega):
        tau,theta=self.time_constants(point)
        return coefficients[0]*np.exp(-1j*omega*point[2])/((1+1j*omega*tau)*(1+1j*omega*theta))


class PowerCurveModel(TransportModel):
    """Shared monotone command map followed by the positive transport model.

q_eff=q+c*q*(1-q/100), c in[-.8,.8]. The endpoints remain0and100 and the
map is monotone on the recorded command range. This changes the response
family; it is not a second constant alpha parameter. Reported frequency
response is the linear dynamic block, not a global nonlinear transfer function.
"""
    def __init__(self,records):
        super().__init__(records);self.kind='power_curve';self.nt=4
        if any(np.any((d.input<0)|(d.input>100)) for d in records):raise ValueError('Command outside0..100')

    def bounds(self):return [(np.log(5.),np.log(600.)),(np.log(.01),np.log(200.)),(0.,15.),(-.8,.8),(np.log(.05),np.log(3000.)),(0.,1.)]

    def _design(self,coordinates):
        tau,theta=np.exp(coordinates[:2]);delay,curvature=coordinates[2:4]
        X=np.zeros((self.count,1+self.ambient_width*len(self.records)));cursor=0
        for j,d in enumerate(self.records):
            count=len(d.time)-1;section=slice(cursor,cursor+count)
            command=d.input+curvature*d.input*(1-d.input/100)
            X[section,0]=delayed_unit(d.time,command,tau,theta,delay)[1:]
            elapsed=d.time[1:]-d.time[0];a=-np.expm1(-elapsed/tau)
            if self.ambient_width==1:X[section,1+j]=a
            else:
                b=(elapsed-tau*a)/(d.time[-1]-d.time[0]);X[section,1+2*j:1+2*j+2]=np.column_stack((a-b,b))
            cursor+=count
        return X


def audit():
    from data import Experiment
    from mean import unit_response
    rng=np.random.RandomState(452)
    t=np.r_[0.,np.cumsum(rng.uniform(.1,3,60))]
    u=rng.uniform(0,100,len(t));rows=[]
    for tau,theta in ((137.,18.),(18.,137.),(80.,80.),(80.,80*(1+1e-9)),(5.,.01)):
        for delay in (0.,2.35,9.001,15.):
            a=delayed_unit(t,u,tau,theta,delay)
            b=independent_reference(t,u,tau,theta,delay)
            error=float(np.max(abs(a-b)))
            if error>2e-8:raise AssertionError((tau,theta,delay,error))
            if delay==0:
                d=Experiment('synthetic',t,np.full(len(t),22.),u,22.)
                if np.max(abs(a-unit_response(d,tau,theta)))>2e-9:raise AssertionError('Zero delay mismatch')
            rows.append({'tau':tau,'theta':theta,'delay_s':delay,'independent_max_abs':error})
    from scipy.integrate import solve_ivp
    model=TransportModel([Experiment('synthetic',t,np.full(len(t),22.),u,22.)])
    model.ambient_mode='linear';model.ambient_width=2
    point=np.r_[np.log([137.,18.]),2.35,np.log(200.),.04]
    predicted=model.trajectories(point,np.array([.71,-1.3,1.7]))[0]
    slope=3/(t[-1]-t[0])
    sol=solve_ivp(lambda s,x:(-1.3+slope*(s-t[0])-x)/137.,(t[0],t[-1]),[0.],t_eval=t,rtol=2e-11,atol=1e-12)
    expected=22+.71*independent_reference(t,u,137.,18.,2.35)+sol.y[0]
    error=float(np.max(abs(predicted-expected)))
    if error>2e-8:raise AssertionError(('Transport linear ambient',error))
    rows.append({'ambient_mode':'linear','independent_max_abs':error})
    for curvature in (-.8,0.,.12,.8):
        mapped=u+curvature*u*(1-u/100)
        power=PowerCurveModel([Experiment('synthetic',t,np.full(len(t),22.),u,22.)])
        point=np.r_[np.log([137.,18.]),2.35,curvature,np.log(200.),.04]
        predicted=power.trajectories(point,np.array([.71,-1.3]))[0]
        expected=22+.71*independent_reference(t,mapped,137.,18.,2.35)-1.3*(-np.expm1(-t/137))
        error=float(np.max(abs(predicted-expected)))
        if error>2e-8:raise AssertionError(('Power curve',curvature,error))
        rows.append({'power_curvature':curvature,'independent_max_abs':error})
    return {'status':'passed','cases':rows,'scope':'Exact delayed left-hold events vs independent merged-grid matrix exponential; zero delay legacy, repeated poles, cold prehistory'}
