"""Bounded hidden-heater startup departures as linear nuisance coefficients."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import matrices
from bounded import Model as ConductanceModel


class Model(ConductanceModel):
    def __init__(self,records,band,startup_band,lag=True,curve=True,ambient='constant',kernel='ou_nugget'):
        super().__init__(records,band,lag,curve,ambient,kernel)
        self.startup_band=startup_band
        self.base_coefficients=1+self.width*len(records)

    def coefficient_bounds(self):
        lower,upper=super().coefficient_bounds()
        return np.r_[lower,np.full(2*len(self.records),-self.startup_band)],np.r_[upper,np.full(2*len(self.records),self.startup_band)]

    def _design(self,shape):
        X,y,initial=super()._design(shape)
        extra=np.zeros((2*self.N,2*len(self.records)));cursor=0
        for j,(d,delta) in enumerate(zip(self.records,self.deltas(shape))):
            local=np.asarray(shape[:self.base_ns]).copy();local[[0,2,3]]-=delta
            A,_,_=matrices(local,self.lag)
            ta,rs,r,_=np.exp(local[:4]);mass=np.sqrt([ta,r*rs,ta,r*rs])
            values,vectors=np.linalg.eigh(mass[:,None]*A[:4,:4]/mass[None,:])
            right=vectors/mass[:,None];left=vectors.T*mass[None,:]
            E=np.exp(d['t'][1:,None]*values[None,:]);n=len(d['t'])-1
            for h,state in enumerate([0,2]):
                response=(E*left[:,state])@right.T
                for k,output in enumerate([1,3]):
                    extra[k*self.N+cursor:k*self.N+cursor+n,2*j+h]=response[:,output]
            cursor+=n
        return np.column_stack([X,extra]),y,initial
