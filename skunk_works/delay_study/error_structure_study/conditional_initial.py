"""Condition the residual OU state on the first reported temperature code.

The conditional log likelihood is log p(y0,...,yn) - log p(y0). First readings
remain omitted from the observation count, but now inform the residual state.
"""
import numpy as np
from smoothing import smooth_ou
from quantized import filter_ou


def smooth_record(record,mean,channel,latent,ell,measurement,phase,nodes,condition=False):
    if not condition:
        f=smooth_ou(record['t'][1:],mean[1:],record['y'][1:,channel],latent,ell,measurement,phase,nodes)
        f['initial_mean_nll_gradient']=0.
        return f
    f=smooth_ou(record['t'],mean,record['y'][:,channel],latent,ell,measurement,phase,nodes)
    first=smooth_ou(record['t'][:1],mean[:1],record['y'][:1,channel],latent,ell,measurement,phase,nodes)
    f['nll']-=first['nll']
    f['initial_mean_nll_gradient']=float(f['mean_nll_gradient'][0]-first['mean_nll_gradient'][0])
    f['mean_nll_gradient']=f['mean_nll_gradient'][1:]
    f['noise_nll_gradient_logsd_logell_logsd']-=first['noise_nll_gradient_logsd_logell_logsd']
    return f


def filter_record(record,mean,channel,latent,ell,measurement,phase,nodes=128,tail_sd=12.,condition=False,keep_history=False):
    sigma=np.hypot(latent,measurement);eta=measurement**2/sigma**2
    if not condition:
        return filter_ou(record['t'][1:],mean[1:],record['y'][1:,channel],sigma,ell,eta,phase,nodes,tail_sd,keep_history)
    f=filter_ou(record['t'],mean,record['y'][:,channel],sigma,ell,eta,phase,nodes,tail_sd,keep_history)
    f['nll']+=float(f['log_bin_probability'][0]);f['initial_log_bin_probability']=float(f['log_bin_probability'][0])
    if keep_history:
        f['initial_history']=f['history'][0];f['initial_time']=float(record['t'][0]);f['history']=f['history'][1:]
    for name,value in list(f.items()):
        if isinstance(value,np.ndarray) and len(value)==len(record['t']):f[name]=value[1:]
    return f
