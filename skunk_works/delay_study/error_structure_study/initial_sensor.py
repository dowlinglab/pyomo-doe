"""Cold heater/sensor initial states allowed inside reported first-code bins.

This is a bounded sensitivity, not integration over initial measurement noise.
An optional additive buffer widens the bins; it is not a confidence interval.
"""
import numpy as np
from bounded import Model as BoundedModel
from physical import matrices
from quantized import bins


class Model(BoundedModel):
    def __init__(self,records,band,phase=-.5,noise_buffer=0.):
        if noise_buffer<0:raise ValueError('Nonnegative initial-noise buffer required')
        self.phase=phase;self.noise_buffer=noise_buffer
        super().__init__(records,band)

    def coefficient_bounds(self):
        lo,hi=super().coefficient_bounds();first=np.array([d['y'][0] for d in self.records])
        a,b=bins(first,self.phase)
        return np.r_[lo,(a-first-self.noise_buffer).ravel()],np.r_[hi,(b-first+self.noise_buffer).ravel()]

    def _design(self,shape):
        X,y,initial=super()._design(shape);extra=np.zeros((2*self.N,2*len(self.records)));cursor=0
        for j,(d,delta) in enumerate(zip(self.records,self.deltas(shape))):
            local=np.asarray(shape[:self.base_ns]).copy();local[[0,2,3]]-=delta
            A,_,_=matrices(local,True);ta,rs,r,_=np.exp(local[:4]);mass=np.sqrt([ta,r*rs,ta,r*rs])
            values,vectors=np.linalg.eigh(mass[:,None]*A[:4,:4]/mass[None,:])
            left=vectors.T*mass[None,:];right=vectors/mass[:,None];E=np.exp(d['t'][1:,None]*values)
            n=len(d['t'])-1
            for channel in [0,1]:
                cold=np.zeros(4);cold[2*channel:2*channel+2]=1.
                response=(E*(left@cold))@right.T
                for k,state in [(0,1),(1,3)]:
                    extra[k*self.N+cursor:k*self.N+cursor+n,2*j+channel]=response[:,state]
            cursor+=n
        return np.column_stack((X,extra)),y,initial
