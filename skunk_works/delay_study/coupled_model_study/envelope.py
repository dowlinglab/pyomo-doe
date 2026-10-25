"""Envelope-theorem derivatives avoid re-solving ambient regression per step."""
from functools import lru_cache
import numpy as np
from covariance import Whitening
from nonlinear import stable_gradient


class Objective:
    def __init__(self,model):
        self.model=model
        self.evaluate=lru_cache(maxsize=1)(self._evaluate)

    def __call__(self,point):return self.evaluate(tuple(point))[0]
    def gradient(self,point):return self.evaluate(tuple(point))[1]

    def _evaluate(self,point):
        p=np.array(point);m=self.model
        nll,c,sigma,W,initial,X=m.projected(p)
        residual=m.y-initial-X@c;rho=np.exp(p[-1])
        white=np.r_[W[0](residual[:m.N]),W[1](residual[m.N:])/rho]
        bounds=m.bounds();gradient=np.zeros(len(p));base_mean=initial+X@c
        # Conditional linear K and ambient coefficients solve a bounded convex
        # problem. The envelope derivative holds these at their current optimum.
        for j in range(m.ns):
            h=2e-5;unit=np.zeros(m.ns);unit[j]=h;lo,hi=bounds[j]
            def mean(q):
                design,_,init=m.design(tuple(q));return init+design@c
            if p[j]-lo>=h and hi-p[j]>=h:
                derivative=(mean(p[:m.ns]+unit)-mean(p[:m.ns]-unit))/(2*h)
            elif hi-p[j]<h:
                derivative=(3*base_mean-4*mean(p[:m.ns]-unit)+mean(p[:m.ns]-2*unit))/(2*h)
            else:
                derivative=(-3*base_mean+4*mean(p[:m.ns]+unit)-mean(p[:m.ns]+2*unit))/(2*h)
            wd=np.r_[W[0](derivative[:m.N]),W[1](derivative[m.N:])/rho]
            gradient[j]=-white@wd/sigma**2
        def noise_function(noise):
            pnoise=np.r_[p[:m.ns],noise];ratio=np.exp(pnoise[-1]);ws=[]
            for k in range(2):
                parameters=[] if m.kernel=='iid' else [np.exp(pnoise[m.ns+2*k]),pnoise[m.ns+2*k+1]]
                ws.append(Whitening(m.times,m.kernel,parameters))
            z=np.r_[ws[0](residual[:m.N]),ws[1](residual[m.N:])/ratio]
            return .5*(2*m.N*(np.log(2*np.pi)+1+np.log(z@z/(2*m.N)))+ws[0].logdet+ws[1].logdet+2*m.N*np.log(ratio))
        gradient[m.ns:]=stable_gradient(noise_function,p[m.ns:],bounds[m.ns:],step=2e-6)
        return nll,gradient
