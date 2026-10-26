"""Unknown-run mean scenarios and exact marginal reported-code predictions."""
import numpy as np
from scipy.special import ndtri,ndtr
from physical import Model as MeanModel
from quantized import STEP,OFFSET,report,log_probability


def reported_mean(mean,sd,phase):
    mean=np.asarray(mean,dtype=float);lower=int(np.floor((mean.min()-11*sd-OFFSET)/STEP))-2
    upper=int(np.ceil((mean.max()+11*sd-OFFSET)/STEP))+2;codes=np.arange(lower,upper+1)
    lo=STEP*(codes+phase)+OFFSET;hi=lo+STEP;levels=np.round(STEP*codes+OFFSET,2)
    mass=np.exp(log_probability((lo[None,:]-mean[:,None])/sd,(hi[None,:]-mean[:,None])/sd))
    if np.max(abs(mass.sum(axis=1)-1))>1e-10:raise FloatingPointError('Reported-code expectation misses probability mass')
    return mass@levels


def reported_quantile(mean,sd,probability,phase):
    return report(np.asarray(mean)+sd*ndtri(probability),phase)


def mixture_quantile(means,sds,probability):
    means=np.asarray(means);sds=np.asarray(sds)[:,None]
    lo=np.min(means-11*sds,axis=0);hi=np.max(means+11*sds,axis=0)
    for _ in range(60):
        middle=(lo+hi)/2;cdf=np.mean(ndtr((middle[None,:]-means)/sds),axis=0)
        lo=np.where(cdf<probability,middle,lo);hi=np.where(cdf>=probability,middle,hi)
    return (lo+hi)/2


def mean_scenarios(record,shape,K,band,points):
    m=MeanModel([record],True,True);n=len(record['t'])-1
    logs=np.unique(np.r_[np.linspace(np.log(1-band),np.log(1+band),points),0.]) if band else np.array([0.])
    lower=np.full((n,2),np.inf);upper=np.full((n,2),-np.inf);nominal=None
    for delta in logs:
        local=np.asarray(shape).copy();local[[0,2,3]]-=delta
        X,_,initial=m.design(tuple(local));base=record['y'][0,0]+initial+X[:,0]*K*np.exp(-delta)
        low=base-2*abs(X[:,1]);high=base+2*abs(X[:,1])
        lo=np.column_stack((low[:n],low[n:]));hi=np.column_stack((high[:n],high[n:]))
        lower=np.minimum(lower,lo);upper=np.maximum(upper,hi)
        if delta==0.:nominal=np.column_stack((base[:n],base[n:]))
    return {'nominal':nominal,'lower':lower,'upper':upper,'conductance_grid_points':len(logs)}


def checked_scenarios(record,shape,K,band,config):
    a=mean_scenarios(record,shape,K,band,config['coarse_points'])
    b=mean_scenarios(record,shape,K,band,config['fine_points'])
    error=max(float(np.max(abs(a[k]-b[k]))) for k in ['lower','upper'])
    refinements=[{'points':config['fine_points'],'envelope_change_C':error}]
    if error>config['maximum_envelope_change_C']:
        c=mean_scenarios(record,shape,K,band,config['refinement_points'])
        error=max(float(np.max(abs(c[k]-b[k]))) for k in ['lower','upper']);b=c
        refinements.append({'points':config['refinement_points'],'envelope_change_C':error})
    if error>config['maximum_envelope_change_C']:raise FloatingPointError('Unknown-run scenario envelope needs a finer conductance grid')
    b['grid_checks']=refinements
    return b
