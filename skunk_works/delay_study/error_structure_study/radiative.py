"""Coupled positive energy balances with normalized ambient radiation.

Gamma is radiative differential conductance at 50 C divided by nominal
convective Ua. Absolute beta/Ua calibration remains absorbed in gain.
"""
from functools import lru_cache
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from numba import njit
from covariance import Whitening
from nonlinear import stable_gradient
TREF=323.15


@njit(cache=True)
def rhs(x,u,ambient,g,ta,rs,r,q,lag,K,gamma):
    h1,s1,h2,s2,power=x
    radiation1=gamma*((h1+273.15)**4-(ambient+273.15)**4)/(4*TREF**3)
    radiation2=gamma*((h2+273.15)**4-(ambient+273.15)**4)/(4*TREF**3)
    v=np.empty(5,dtype=np.complex128)
    v[0]=(-g*(h1-ambient)+r*(s1-h1)+q*(h2-h1)+K*power-radiation1)/ta
    v[1]=(h1-s1)/rs
    v[2]=(-g*(h2-ambient)+r*(s2-h2)+q*(h1-h2)-radiation2)/ta
    v[3]=(h2-s2)/rs
    v[4]=(u-power)/lag
    return v


@njit(cache=True)
def forward(point,time,command,initial,ends,max_step):
    ta,rs,r,q,lag=np.exp(point[:5]);curve=point[5];K=np.exp(point[6]);gamma=point[7]
    nr=len(initial);N=len(time)-nr;answer=np.empty(2*N,dtype=np.complex128)
    origin=0;cursor=0
    # Conservative step bound uses the physical operating envelope. For complex
    # perturbations its real part is fixed; convergence is audited separately.
    maximumT=np.max(initial)+2+K.real*100/np.min(np.exp(point[8:8+nr].real))+273.15
    rate=max((np.max(np.exp(point[8:8+nr].real))+2*r.real+2*q.real+gamma.real*(maximumT/TREF)**3)/ta.real,1/rs.real,1/lag.real)
    limit=min(max_step,.2/rate)
    for j in range(nr):
        end=ends[j];g=np.exp(point[8+j]);ambient=initial[j,0]+point[8+nr+j]
        state=np.empty(5,dtype=np.complex128)
        state[0]=initial[j,0];state[1]=initial[j,0];state[2]=initial[j,1];state[3]=initial[j,1];state[4]=0.
        for k in range(origin,end-1):
            dt=time[k+1]-time[k];nsteps=int(np.ceil(dt/limit));h=dt/nsteps
            u=command[k]+curve*command[k]*(1-command[k]/100)
            for l in range(nsteps):
                k1=rhs(state,u,ambient,g,ta,rs,r,q,lag,K,gamma)
                k2=rhs(state+h*k1/2,u,ambient,g,ta,rs,r,q,lag,K,gamma)
                k3=rhs(state+h*k2/2,u,ambient,g,ta,rs,r,q,lag,K,gamma)
                k4=rhs(state+h*k3,u,ambient,g,ta,rs,r,q,lag,K,gamma)
                state=state+h*(k1+2*k2+2*k3+k4)/6
            answer[cursor]=state[1];answer[N+cursor]=state[3];cursor+=1
        origin=end
    return answer


class Model:
    def __init__(self,records,band=.1,max_step=.25,gamma_fixed=None,curve_fixed=None):
        self.records=records;self.band=band;self.max_step=max_step;self.gamma_fixed=gamma_fixed;self.curve_fixed=curve_fixed
        self.nr=len(records);self.nmean=8+2*self.nr;self.ns=self.nmean
        self.N=sum(len(d['t'])-1 for d in records)
        self.time=np.concatenate([d['t'] for d in records]);self.u=np.concatenate([d['u'] for d in records])
        self.initial=np.array([d['y'][0] for d in records]);self.ends=np.cumsum([len(d['t']) for d in records])
        self.times=[d['t'][1:] for d in records]
        self.y=np.concatenate([np.concatenate([d['y'][1:,k] for d in records]) for k in (0,1)])
        self.evaluate=lru_cache(maxsize=1)(self._evaluate)

    def bounds(self):
        return [(np.log(5),np.log(1000)),(np.log(.05),np.log(200)),(np.log(.005),np.log(30)),
                (np.log(.001),np.log(10)),(np.log(.01),np.log(30)),
                (-.8,.8) if self.curve_fixed is None else (self.curve_fixed,self.curve_fixed),
                (np.log(.2),np.log(3)),(0.,4.) if self.gamma_fixed is None else (self.gamma_fixed,self.gamma_fixed)]+\
                [(np.log(1-self.band),np.log(1+self.band))]*self.nr+[(-2.,2.)]*self.nr+\
                [(np.log(.05),np.log(3000)),(0.,1.)]*2+[(np.log(.05),np.log(20))]

    def centered_constraint(self,p):return float(sum(p[8:8+self.nr]))
    def centered_jacobian(self,p):
        a=np.zeros(len(p));a[8:8+self.nr]=1.
        return a

    def prediction(self,point,max_step=None):
        return forward(np.asarray(point,dtype=complex),self.time,self.u,self.initial,self.ends,self.max_step if max_step is None else max_step).real

    def covariance(self,p):
        n=self.nmean;rho=np.exp(p[-1])
        W=[Whitening(self.times,'ou_nugget',[np.exp(p[n+2*k]),p[n+2*k+1]]) for k in (0,1)]
        return W,rho

    def profile_noise(self,p,residual):
        W,rho=self.covariance(p)
        z=np.r_[W[0](residual[:self.N]),W[1](residual[self.N:])/rho]
        sigma2=float(z@z/(2*self.N))
        if sigma2<=0 or not np.isfinite(sigma2):raise ValueError('Invalid likelihood scale')
        nll=.5*(2*self.N*(np.log(2*np.pi)+1+np.log(sigma2))+W[0].logdet+W[1].logdet+2*self.N*np.log(rho))
        return float(nll),np.sqrt(sigma2),W,rho,z

    def _evaluate(self,key):
        p=np.asarray(key);mean=self.prediction(p);residual=self.y-mean
        nll,sigma,W,rho,z=self.profile_noise(p,residual)
        J=np.zeros((2*self.N,self.nmean));eps=1e-25
        bounds=self.bounds()
        for j in range(self.nmean):
            if bounds[j][0]==bounds[j][1]:continue
            c=p.astype(complex);c[j]+=eps*1j
            J[:,j]=forward(c,self.time,self.u,self.initial,self.ends,self.max_step).imag/eps
        Jw=np.vstack([W[0](J[:self.N]),W[1](J[self.N:])/rho])
        gradient=np.zeros(len(p));gradient[:self.nmean]=-z@Jw/(sigma*sigma)
        def noise(v):return self.profile_noise(np.r_[p[:self.nmean],v],residual)[0]
        gradient[self.nmean:]=stable_gradient(noise,p[self.nmean:],bounds[self.nmean:],step=2e-6)
        return nll,gradient

    def objective(self,p):return self.evaluate(tuple(p))[0]
    def gradient(self,p):return self.evaluate(tuple(p))[1]
