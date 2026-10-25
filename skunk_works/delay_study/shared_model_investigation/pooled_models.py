"""Matched held-input thermal chains and flexible stable real-pole LTI models.

The flexible model is tau*T'=-T+delta+sum_j c_j*z_j, with z_0=u and
each subsequent z_j another first-order input filter. Sum(c)=K. Ambient
enters the temperature equation, not an arbitrary output intercept. Its input
numerator is flexible, but its poles are real; it is not unrestricted LTI.
"""
from functools import lru_cache
import numpy as np
from numba import njit
from scipy.linalg import expm
from scipy.optimize import lsq_linear


@njit(cache=True)
def propagate(command, indices, transitions, forcing):
    m = transitions.shape[1]
    state = np.zeros(m)
    values = np.zeros((len(command), m))
    for k in range(len(command)-1):
        state = transitions[indices[k]]@state+forcing[indices[k]]*command[k]
        values[k+1] = state
    return values


class PooledModel:
    def __init__(self, records, kind, order):
        if kind not in ('chain','flexible_real','unequal_chain') or order not in (1,2,3,4):
            raise ValueError((kind,order))
        if kind == 'chain' and order > 3:
            raise ValueError('Chain order is1,2,3')
        if kind in ('flexible_real','unequal_chain') and order < 2:
            raise ValueError('Flexible state count is2,3,4')
        self.records, self.kind, self.order = records, kind, order
        intervals = np.concatenate([np.diff(d.time) for d in records])
        self.intervals = np.unique(intervals)
        self.indices = [np.searchsorted(self.intervals,np.diff(d.time)) for d in records]
        self.count = sum(len(d.time)-1 for d in records)
        self.m = order+1 if kind == 'chain' else order
        self.ninput = 1 if kind in ('chain','unequal_chain') else order
        self.nt = 2 if kind == 'chain' else self.m
        self.numerator_bound = 2.
        self.ambient_mode='constant'
        self.ambient_width=1
        self.y = np.concatenate([d.temperature[1:]-d.temperature[0] for d in records])
        self.design = lru_cache(maxsize=8)(self._design)

    def time_constants(self, logarithms):
        value = np.exp(np.asarray(logarithms))
        return np.r_[value[0],np.full(self.order,value[1]/self.order)] if self.kind == 'chain' else value

    def _design(self, logarithms):
        times = self.time_constants(logarithms)
        m = len(times)
        A = np.diag(-1/times)
        for j in range(1,m):
            A[j,j-1] = 1/times[j]
        B = np.zeros(m); B[0] = 1/times[0]
        augmented = np.zeros((m+1,m+1))
        augmented[:m,:m] = A
        augmented[:m,m] = B
        plans = np.array([expm(augmented*h)[:m] for h in self.intervals])
        transitions = np.ascontiguousarray(plans[:,:,:m])
        forcing = np.ascontiguousarray(plans[:,:,m])
        X = np.zeros((self.count,self.ninput+self.ambient_width*len(self.records)))
        cursor = 0
        for j,d in enumerate(self.records):
            h = propagate(d.input,self.indices[j],transitions,forcing)[1:]
            count = len(h); section = slice(cursor,cursor+count)
            X[section,0] = h[:,-1]
            if self.kind == 'flexible_real':
                X[section,1:self.ninput] = h[:,:-1]-h[:,[-1]]
            elapsed=d.time[1:]-d.time[0]
            a=-np.expm1(-elapsed/times[0])
            if self.ambient_width==1:X[section,self.ninput+j]=a
            else:
                b=(elapsed-times[0]*a)/(d.time[-1]-d.time[0])
                X[section,self.ninput+2*j:self.ninput+2*j+2]=np.column_stack((a-b,b))
            cursor += count
        return X

    def bounds(self):
        times = [(np.log(5.),np.log(600.))]
        times += [(np.log(.01),np.log(200.))]*(1 if self.kind=='chain' else self.m-1)
        return times+[(np.log(.05),np.log(3000.)),(0.,1.)]

    def coefficient_bounds(self):
        lo = [.2]+[-self.numerator_bound]*(self.ninput-1)+[-2.]*(self.ambient_width*len(self.records))
        hi = [1.5]+[self.numerator_bound]*(self.ninput-1)+[2.]*(self.ambient_width*len(self.records))
        return np.array(lo),np.array(hi)

    def projected(self, point, kernel='ou_nugget', fixed_gain=None):
        from covariance import Whitening
        nt = self.nt
        X = self.design(tuple(point[:nt]))
        W = Whitening([d.time[1:] for d in self.records],kernel,
                      [np.exp(point[nt]),point[nt+1]] if kernel!='iid' else [])
        joint = W(np.column_stack((X,self.y)))
        Xw,yw = joint[:,:-1],joint[:,-1]
        lower,upper=self.coefficient_bounds()
        if fixed_gain is None:
            solve = lsq_linear(Xw,yw,bounds=(lower,upper),tol=1e-11,lsmr_tol=1e-12)
            coefficients=solve.x
        else:
            if not lower[0]<=fixed_gain<=upper[0]:raise ValueError('Fixed gain outside declared bounds')
            solve=lsq_linear(Xw[:,1:],yw-Xw[:,0]*fixed_gain,bounds=(lower[1:],upper[1:]),tol=1e-11,lsmr_tol=1e-12)
            coefficients=np.r_[fixed_gain,solve.x]
        if not solve.success:
            raise RuntimeError('Bounded coefficient projection failed')
        r = Xw@coefficients-yw
        scale = np.sqrt(r@r/self.count)
        nll = .5*(self.count*(np.log(2*np.pi)+1+2*np.log(scale))+W.logdet)
        return float(nll),coefficients,float(scale),W

    def trajectories(self, point, coefficients):
        nt = self.nt
        X = self.design(tuple(point[:nt]))
        rise = X@coefficients
        predictions = []; cursor=0
        for d in self.records:
            count=len(d.time)-1
            predictions.append(np.r_[d.temperature[0],d.temperature[0]+rise[cursor:cursor+count]])
            cursor+=count
        return predictions

    def frequency_response(self, point, coefficients, omega):
        nt=self.nt
        times=self.time_constants(point[:nt])
        h=1/(1+1j*omega*times[0])
        if self.kind=='chain':
            return coefficients[0]*h/(1+1j*omega*times[1])**self.order
        if self.kind=='unequal_chain':return coefficients[0]/np.prod(1+1j*omega*times)
        physical=np.r_[coefficients[1:self.ninput],coefficients[0]-sum(coefficients[1:self.ninput])]
        result=physical[0]*h
        for k in range(1,self.m):
            h=h/(1+1j*omega*times[k])
            result+=physical[k]*h
        return result


def make_model(records,kind,order,numerator_bound=2.,ambient_mode='constant'):
    if kind=='transport':
        from transport import TransportModel
        if order!=1:raise ValueError('Transport extension applies to the n=1 baseline')
        model=TransportModel(records)
    else:model=PooledModel(records,kind,order)
    if numerator_bound<=0:raise ValueError('Positive numerator bound required')
    model.numerator_bound=float(numerator_bound)
    if ambient_mode not in ('constant','linear'):raise ValueError(ambient_mode)
    model.ambient_mode=ambient_mode;model.ambient_width=1 if ambient_mode=='constant' else 2
    return model


def audit():
    from scipy.integrate import solve_ivp
    from data import Experiment
    from mean import unit_response
    rng=np.random.RandomState(9132)
    t=np.r_[0.,np.cumsum(rng.uniform(.1,3,45))]
    u=rng.uniform(0,100,len(t))
    records=[Experiment('synthetic',t,np.full(len(t),22.),u,22.)]
    rows=[]
    for kind,order in [('chain',1),('chain',2),('chain',3),('flexible_real',2),('flexible_real',3),('flexible_real',4),('unequal_chain',3)]:
        model=PooledModel(records,kind,order)
        times=np.r_[137.,np.full(model.m-1,9.)]
        logarithms=np.log([137.,18.]) if kind=='chain' else np.log(times)
        actualtimes=model.time_constants(logarithms)
        c=np.r_[.71, np.linspace(-.08,.13,model.ninput-1), -1.3]
        point=np.r_[logarithms,np.log(200.),.04]
        pred=model.trajectories(point,c)[0]
        physical=np.r_[c[1:model.ninput],c[0]-sum(c[1:model.ninput])]
        if kind in ('chain','unequal_chain'):
            physical=np.r_[np.zeros(model.m-1),c[0]]
        state=np.zeros(model.m)
        expected=[22.]
        for k,dt in enumerate(np.diff(t)):
            def f(s,x):
                z=np.r_[u[k],x[:-1]]
                dx=np.empty(model.m)
                for j in range(model.m-1):
                    dx[j]=(z[j]-x[j])/actualtimes[j+1]
                dx[-1]=(-x[-1]+c[-1]+physical@z)/actualtimes[0]
                return dx
            sol=solve_ivp(f,(0.,dt),state,method='DOP853',rtol=2e-11,atol=1e-12)
            if not sol.success: raise AssertionError(sol.message)
            state=sol.y[:,-1]; expected.append(22+state[-1])
        error=float(np.max(np.abs(pred-expected)))
        if error>2e-8: raise AssertionError((kind,order,error))
        if kind=='chain' and order==1:
            legacy=22+.71*unit_response(records[0],137,18)-1.3*(-np.expm1(-t/137))
            if np.max(np.abs(legacy-pred))>2e-9: raise AssertionError('Legacy mismatch')
        rows.append({'kind':kind,'order':order,'ODE_max_abs_C':error})
    # Linear endpoint basis is independently checked as an ambient-only ODE,
    # rather than comparing two copies of the closed-form basis.
    for tau in (5.,137.,600.):
        model=make_model(records,'chain',1,ambient_mode='linear')
        point=np.r_[np.log([tau,18.]),np.log(200.),.04]
        c=np.array([.71,-1.3,1.7])
        prediction=model.trajectories(point,c)[0]
        slope=3./(t[-1]-t[0])
        solution=solve_ivp(lambda s,x:(-1.3+slope*(s-t[0])-x)/tau,
                           (t[0],t[-1]),[0.],t_eval=t,rtol=2e-11,atol=1e-12,method='DOP853')
        expected=22+.71*unit_response(records[0],tau,18.)+solution.y[0]
        error=float(np.max(abs(prediction-expected)))
        if error>2e-8:raise AssertionError(('Linear ambient',tau,error))
        rows.append({'kind':'chain','order':1,'ambient_mode':'linear','tau':tau,'ODE_max_abs_C':error})
    return {'status':'passed','cases':rows,'scope':'Exact irregular intervals, physical ambient forcing, free numerator and orders1/2/3; independent physical-state ODE'}
