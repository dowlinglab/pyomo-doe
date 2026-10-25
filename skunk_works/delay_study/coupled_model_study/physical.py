"""Two-channel energy balances in identifiable scale-free coordinates."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import lsq_linear
from numba import njit
from covariance import Whitening


def load_records():
    config=json.loads((HERE.parent/'residual_study/config.json').read_text())
    records=[];hashes={}
    for name in config['records']:
        path=HERE.parents[2]/'data'/name;f=pd.read_csv(path)
        a=f[['Time','T1','T2','Q1','Q2']].values.astype(float)
        assert np.isfinite(a).all() and np.all(np.diff(a[:,0])>0) and not np.any(a[:,4])
        signature=hashlib.sha256(a.astype('<f8').tobytes()).hexdigest()
        assert signature not in hashes
        hashes[signature]=name
        records.append({'name':name,'t':a[:,0]-a[0,0],'y':a[:,1:3], 'u':a[:,3],
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return records


def matrices(shape,lag=False):
    ta,rs,r,q=np.exp(shape[:4]);d=1/rs
    A=np.array([[-(1+r+q)/ta,r/ta,q/ta,0], [d,-d,0,0],
                [q/ta,0,-(1+r+q)/ta,r/ta], [0,0,d,-d]])
    Bu=np.array([1/ta,0,0,0]);Ba=np.array([1/ta,0,1/ta,0])
    if lag:
        B=np.zeros(5);B[4]=1/np.exp(shape[4])
        full=np.zeros((5,5));full[:4,:4]=A;full[:4,4]=Bu;full[4,4]=-B[4]
        return full,B,np.r_[Ba,0]
    return A,Bu,Ba


@njit(cache=True)
def propagate(u,indices,transitions,forcing):
    x=np.zeros(transitions.shape[1]);out=np.zeros((len(u),2))
    for j in range(len(u)-1):
        x=transitions[indices[j]]@x+forcing[indices[j]]*u[j]
        out[j+1,0]=x[1];out[j+1,1]=x[3]
    return out


def transfers(shape,K,omega,lag=False):
    ta,rs,r,q=np.exp(shape[:4]);s=1j*omega
    plus=K/(ta*rs*s*s+(ta+rs*(1+r))*s+1)
    minus=K/(ta*rs*s*s+(ta+rs*(1+r+2*q))*s+1+2*q)
    F=1/(1+s*np.exp(shape[4])) if lag else 1
    return np.array([(plus+minus)*F/2,(plus-minus)*F/2])


class Model:
    def __init__(self,records,lag=False,curve=False,ambient='constant',kernel='ou_nugget'):
        self.records=records;self.lag=lag;self.curve=curve;self.width=1 if ambient=='constant' else 2
        self.kernel=kernel;self.ns=4+lag+curve
        self.N=sum(len(d['t'])-1 for d in records)
        self.times=[d['t'][1:] for d in records]
        self.intervals=np.unique(np.concatenate([np.diff(d['t']) for d in records]))
        self.indices=[np.searchsorted(self.intervals,np.diff(d['t'])) for d in records]
        self.design=lru_cache(maxsize=6)(self._design)
        self.y=np.concatenate([np.concatenate([d['y'][1:,k]-d['y'][0,0] for d in records]) for k in range(2)])

    def shape_bounds(self):
        b=[(np.log(5),np.log(1000)),(np.log(.05),np.log(200)),
           (np.log(.005),np.log(30)),(np.log(.001),np.log(10))]
        if self.lag:b.append((np.log(.01),np.log(30)))
        if self.curve:b.append((-.8,.8))
        return b

    def bounds(self):
        noise=[(np.log(.05),np.log(3000)),(0.,1.)]*2 if self.kernel!='iid' else []
        return self.shape_bounds()+noise+[(np.log(.05),np.log(20))]

    def coefficient_bounds(self):
        return np.r_[.2,np.full(self.width*len(self.records),-2.)],np.r_[3.,np.full(self.width*len(self.records),2.)]

    def _design(self,shape):
        shape=np.asarray(shape);A,B,Ba=matrices(shape,self.lag);m=len(A)
        aug=np.zeros((m+1,m+1));aug[:m,:m]=A;aug[:m,m]=B
        plans=np.array([expm(aug*h)[:m] for h in self.intervals])
        transitions=np.ascontiguousarray(plans[:,:,:m]);forcing=np.ascontiguousarray(plans[:,:,m])
        # Passive thermal matrix is similar to a symmetric matrix. Its orthogonal
        # eigensystem remains well behaved at repeated poles; the optional
        # actuator state is absent from ambient and homogeneous responses.
        ta,rs,r,q=np.exp(shape[:4]);mass=np.sqrt([ta,r*rs,ta,r*rs])
        vals,vectors=np.linalg.eigh(mass[:,None]*A[:4,:4]/mass[None,:])
        left=vectors.T*mass[None,:];right=vectors/mass[:,None]
        X=np.zeros((2*self.N,1+self.width*len(self.records)));initial=np.zeros(2*self.N);cursor=0
        for j,d in enumerate(self.records):
            n=len(d['t'])-1;ts=d['t'];u=d['u']
            if self.curve:u=u+shape[-1]*u*(1-u/100)
            response=propagate(u,self.indices[j],transitions,forcing)[1:]
            # Ambient is a common physical forcing, not independent output intercepts.
            # Its step and ramp solutions use a direct augmented exponential.
            T=ts[-1];init=np.zeros(4);init[2:4]=d['y'][0,1]-d['y'][0,0]
            E=np.exp(ts[1:,None]*vals[None,:])
            homogeneous=(E*(left@init))@right.T
            step=(np.expm1(ts[1:,None]*vals)/vals*(left@Ba[:4]))@right.T
            ramp=((np.expm1(ts[1:,None]*vals)-ts[1:,None]*vals)/vals**2*(left@Ba[:4]))@right.T/T
            for channel,state in [(0,1),(1,3)]:
                section=slice(channel*self.N+cursor,channel*self.N+cursor+n)
                X[section,0]=response[:,channel];initial[section]=homogeneous[:,state]
                if self.width==1:X[section,1+j]=step[:,state]
                else:X[section,1+2*j:3+2*j]=np.column_stack((step[:,state]-ramp[:,state],ramp[:,state]))
            cursor+=n
        return X,self.y-initial,initial

    def projected(self,point):
        X,y,initial=self.design(tuple(point[:self.ns]));nt=self.ns
        rho=np.exp(point[-1]);W=[]
        for channel in range(2):
            noise=[] if self.kernel=='iid' else [np.exp(point[nt+2*channel]),point[nt+2*channel+1]]
            W.append(Whitening(self.times,self.kernel,noise))
        joint=np.vstack([W[k](np.column_stack((X[k*self.N:(k+1)*self.N],y[k*self.N:(k+1)*self.N])))/(rho if k else 1) for k in range(2)])
        solve=lsq_linear(joint[:,:-1],joint[:,-1],bounds=self.coefficient_bounds(),tol=1e-10,lsmr_tol=1e-12)
        if not solve.success:raise RuntimeError('Linear nuisance projection did not converge')
        white=joint[:,:-1]@solve.x-joint[:,-1];sigma=np.sqrt(white@white/(2*self.N))
        nll=.5*(2*self.N*(np.log(2*np.pi)+1+2*np.log(sigma))+W[0].logdet+W[1].logdet+2*self.N*np.log(rho))
        return float(nll),solve.x,float(sigma),W,initial,X
