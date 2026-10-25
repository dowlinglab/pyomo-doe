"""One varying conductance, centered around a shared scale-free nominal model."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import Model as SharedModel


class Model(SharedModel):
    def __init__(self, records, band, lag=True, curve=True, ambient='constant', kernel='ou_nugget'):
        if not 0 <= band < 1:
            raise ValueError('Conductance relative band must be in [0,1)')
        super().__init__(records,lag,curve,ambient,kernel)
        self.band=band
        self.base_ns=self.ns
        self.nv=len(records) if band else 0
        self.ns+=self.nv
        self.local=[SharedModel([d],lag,curve,ambient,kernel) for d in records]

    def shape_bounds(self):
        return super().shape_bounds()+[(np.log(1-self.band),np.log(1+self.band))]*self.nv

    def deltas(self, point):
        return np.asarray(point[self.base_ns:self.ns]) if self.nv else np.zeros(len(self.records))

    def centered_constraint(self, point):
        return float(self.deltas(point).sum())

    def centered_jacobian(self, point):
        result=np.zeros(len(point));result[self.base_ns:self.ns]=1.
        return result

    def _design(self, shape):
        p=np.asarray(shape)
        deltas=self.deltas(p)
        X=np.zeros((2*self.N,1+self.width*len(self.records)))
        initial=np.zeros(2*self.N);cursor=0
        for j,(d,m,delta) in enumerate(zip(self.records,self.local,deltas)):
            local=p[:self.base_ns].copy()
            local[[0,2,3]]-=delta
            x,_,i=m.design(tuple(local))
            n=len(d['t'])-1
            for k in range(2):
                target=slice(k*self.N+cursor,k*self.N+cursor+n)
                source=slice(k*n,(k+1)*n)
                X[target,0]=x[source,0]*np.exp(-delta)
                start=1+j*self.width
                X[target,start:start+self.width]=x[source,1:]
                initial[target]=i[source]
            cursor+=n
        return X,self.y-initial,initial
