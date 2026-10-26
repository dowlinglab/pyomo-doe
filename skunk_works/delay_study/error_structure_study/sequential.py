"""Block likelihood calculations for sequential feasible generalized LS.

Mean updates profile bounded linear gain/ambient coefficients. Covariance updates
hold the residual vector fixed and retain the determinant and profiled scale.
This is alternating Gaussian maximum likelihood, not inverse-variance weighting
of individual observations; quantized likelihood comparisons are separate.
"""
import numpy as np
from covariance import Whitening
from nonlinear import stable_gradient


def covariance_objective(noise, residual, times, kernel='ou_nugget'):
    N=len(residual)//2
    if 2*N!=len(residual):raise ValueError('Two equally sized channels required')
    ratio=np.exp(noise[-1]);plans=[]
    for k in [0,1]:
        parameters=[] if kernel=='iid' else [np.exp(noise[2*k]),noise[2*k+1]]
        plans.append(Whitening(times,kernel,parameters))
    z=np.r_[plans[0](residual[:N]),plans[1](residual[N:])/ratio]
    variance=float(z@z/(2*N))
    if variance<=0:raise FloatingPointError('Nonpositive profiled variance')
    return float(.5*(2*N*(np.log(2*np.pi)+1+np.log(variance))+
                       plans[0].logdet+plans[1].logdet+2*N*np.log(ratio)))


def covariance_gradient(noise,residual,times,bounds,kernel='ou_nugget'):
    return stable_gradient(lambda q:covariance_objective(q,residual,times,kernel),
                           noise,bounds,step=2e-6)


def residual_at(model,point):
    _,coef,_,_,initial,X=model.projected(point)
    return model.y-initial-X@coef,coef
