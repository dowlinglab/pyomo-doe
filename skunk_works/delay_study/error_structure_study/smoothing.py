"""Scalar quadrature smoothing and Fisher-identity likelihood scores."""
import numpy as np
from scipy.special import logsumexp
from quantized import filter_ou,LOG2PI


def smooth_ou(time,mean,observed,latent_sd,ell,measurement_sd,phase=-.5,nodes=128,tail_sd=12.):
    if min(latent_sd,ell,measurement_sd)<=0:raise ValueError('Positive latent and measurement variances required for smooth scores')
    S=latent_sd*latent_sd;R=measurement_sd*measurement_sd
    f=filter_ou(time,mean,observed,np.sqrt(S+R),ell,R/(S+R),phase,nodes=nodes,tail_sd=tail_sd,keep_history=True)
    h=f.pop('history');n=len(h);weights=[None]*n;weights[-1]=h[-1]['weight'].copy();cross=np.empty(n-1);closure=0.
    for k in range(n-2,-1,-1):
        a=np.exp(-(time[k+1]-time[k])/ell);q=S*(-np.expm1(-2*(time[k+1]-time[k])/ell))
        old=h[k];new=h[k+1]
        transition=-.5*((new['grid'][:,None]-a*old['grid'][None,:])**2/q+np.log(q)+LOG2PI)
        with np.errstate(divide='ignore'):
            pair=np.log(weights[k+1])[:,None]+np.log(old['weight'])[None,:]+transition-new['prior_log_density'][:,None]
        normalizer=float(logsumexp(pair));joint=np.exp(pair-normalizer)
        closure=max(closure,abs(np.expm1(normalizer)),float(np.max(abs(joint.sum(axis=1)-weights[k+1]))))
        weights[k]=joint.sum(axis=0);cross[k]=float(new['grid']@joint@old['grid'])
    ex=np.array([w@d['grid'] for w,d in zip(weights,h)])
    ex2=np.array([w@(d['grid']**2) for w,d in zip(weights,h)])
    ev=np.array([w@d['measurement_mean'] for w,d in zip(weights,h)])
    ev2=np.array([w@d['measurement_second'] for w,d in zip(weights,h)])
    dt=np.diff(time);a=np.exp(-dt/ell);q=S*(-np.expm1(-2*dt/ell))
    einnovation=ex2[1:]-2*a*cross+a*a*ex2[:-1]
    if np.any(einnovation<-1e-10):raise FloatingPointError('Negative smoothed process second moment')
    process_score=.5*(ex2[0]/S-1+np.sum(einnovation/q-1))
    da=a*dt/ell;dlogq=-2*a*da/(1-a*a)
    length_score=np.sum(da/q*(cross-a*ex2[:-1])+.5*dlogq*(einnovation/q-1))
    measurement_score=.5*(np.sum(ev2)/R-n)
    f.update({'smoothed_latent_mean':ex,'smoothed_latent_second':ex2,'smoothed_lag1_cross':cross,
              'smoothed_measurement_mean':ev,'smoothed_measurement_second':ev2,
              'mean_nll_gradient':-ev/R,
              'noise_nll_gradient_logsd_logell_logsd':-np.array([2*process_score,length_score,2*measurement_score]),
              'smoothing_closure_error':closure})
    return f
