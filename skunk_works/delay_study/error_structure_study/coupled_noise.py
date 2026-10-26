"""Two-channel continuous-Gaussian OU disturbance comparison.

The actual observations are discrete. This module deliberately supplies a
matched continuous-observation sensitivity, not a bivariate bin likelihood.
Correlated OU driving noise and correlated measurement noise are distinct.
"""
from functools import lru_cache
import numpy as np
from numba import njit
from joint_quantized import Model as PhysicalModel


def covariance(noise,mode):
    a,l1,r1,b,l2,r2=np.exp(noise[:6]);S=np.diag([a*a,b*b]);R=np.diag([r1*r1,r2*r2])
    index=6
    if mode in ['process','both']:
        # A diagonal OU drift driven by correlated Brownian increments has
        # stationary cross covariance Qc12/(1/l1+1/l2), not rho*a*b unless
        # the two relaxation lengths agree.
        S[0,1]=S[1,0]=noise[index]*2*a*b*np.sqrt(l1*l2)/(l1+l2);index+=1
    if mode in ['measurement','both']:R[0,1]=R[1,0]=noise[index]*r1*r2
    return S,R,np.array([l1,l2])


@njit(cache=True)
def filter_gaussian(time,residual,S,R,ell):
    n=len(time);P=S.copy();state=np.zeros(2);Ksaved=np.empty((n,2,2));alphas=np.ones((n,2))
    inverse_innovation=np.empty((n,2));white=np.empty((n,2));predicted=np.empty((n,2));nll=0.
    for i in range(n):
        if i:
            alpha=np.exp(-(time[i]-time[i-1])/ell);alphas[i]=alpha
            state=alpha*state
            for a in range(2):
                for b in range(2):P[a,b]=alpha[a]*alpha[b]*P[a,b]+S[a,b]*(1-alpha[a]*alpha[b])
        predicted[i]=state;v=residual[i]-state;F=P+R
        det=F[0,0]*F[1,1]-F[0,1]*F[1,0]
        if det<=0:raise ValueError('Nonpositive innovation covariance')
        inv=np.array([[F[1,1],-F[0,1]],[-F[1,0],F[0,0]]])/det
        z=inv@v;inverse_innovation[i]=z;nll+=.5*(np.log(det)+v@z+2*np.log(2*np.pi))
        root=np.sqrt(F[0,0]);off=F[1,0]/root
        white[i,0]=v[0]/root;white[i,1]=(v[1]-off*white[i,0])/np.sqrt(F[1,1]-off*off)
        K=P@inv;Ksaved[i]=K;state=state+K@v;P=P-K@P;P=(P+P.T)/2
    # Reverse the data-dependent filter mean recursion. Covariances depend on
    # noise parameters only, so this yields Sigma^-1 residual without forming
    # a dense covariance or using future observations in a forward forecast.
    adjoint=np.zeros(2);gradient=np.empty((n,2))
    for i in range(n-1,-1,-1):
        z=inverse_innovation[i];K=Ksaved[i]
        gradient[i]=z+K.T@adjoint
        adjoint=alphas[i]*(-z+(np.eye(2)-K).T@adjoint)
    return nll,gradient,white,predicted


class Model:
    def __init__(self,records,band=.1,mode='process'):
        if mode not in ['independent','process','measurement','both']:raise ValueError('Unknown disturbance comparison')
        self.mode=mode;self.records=records;self.band=band;self.physical=PhysicalModel(records,band)
        self.nmean=self.physical.nmean;self.nshape=self.physical.nshape;self.N=self.physical.N
        self.nnoise=6+int(mode in ['process','both'])+int(mode in ['measurement','both'])
        self.evaluate=lru_cache(maxsize=1)(self._evaluate)

    def bounds(self):return self.physical.bounds()[:self.nmean+6]+[(-.98,.98)]*(self.nnoise-6)
    def coordinate_names(self):return self.physical.coordinate_names()+(['rho_OU_driving'] if self.mode in ['process','both'] else [])+(['rho_measurement'] if self.mode in ['measurement','both'] else [])
    def centered_constraint(self,p):return self.physical.centered_constraint(p)
    def centered_jacobian(self,p):
        return np.r_[self.physical.centered_jacobian(p[:self.nmean+6]),np.zeros(self.nnoise-6)]

    def _evaluate(self,key):
        p=np.asarray(key);mean,J=self.physical.mean(p[:self.nmean+6]);residual=self.physical.y-mean
        noise=p[self.nmean:];S,R,ell=covariance(noise,self.mode);score=np.zeros(2*self.N);nll=0.;cursor=0
        for d in self.records:
            n=len(d['t'])-1;r=np.column_stack([residual[c*self.N+cursor:c*self.N+cursor+n] for c in [0,1]])
            value,g,_,_=filter_gaussian(d['t'][1:],r,S,R,ell);nll+=value
            for c in [0,1]:score[c*self.N+cursor:c*self.N+cursor+n]=g[:,c]
            cursor+=n
        def fixed_mean(q):
            S,R,ell=covariance(q,self.mode);value=0.;cursor=0
            for d in self.records:
                n=len(d['t'])-1;r=np.column_stack([residual[c*self.N+cursor:c*self.N+cursor+n] for c in [0,1]])
                value+=filter_gaussian(d['t'][1:],r,S,R,ell)[0];cursor+=n
            return value
        gradient=np.zeros(len(p));gradient[:self.nmean]=-J.T@score;h=2e-5
        for j in range(self.nnoise):
            v=np.zeros(self.nnoise);v[j]=h
            gradient[self.nmean+j]=(fixed_mean(noise+v)-fixed_mean(noise-v))/(2*h)
        return float(nll),gradient

    def objective(self,p):return self.evaluate(tuple(p))[0]
    def gradient(self,p):return self.evaluate(tuple(p))[1]
