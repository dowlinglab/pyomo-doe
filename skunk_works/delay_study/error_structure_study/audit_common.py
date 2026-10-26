"""Dimensional mean reconstruction shared by independent result audits."""
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from audit_bounded import independent


def dimensional_fit(point,coefficients,records,band):
    p=np.asarray(point);nr=len(records);ds=p[6:6+nr] if band else np.zeros(nr)
    return {'lag':True,'curve':True,'ambient':'constant','band':band,
        'record_names':[d['name'] for d in records],'coefficients':list(coefficients),
        'parameters':{'K_common_nominal':float(coefficients[0]),'tA_nominal_s':float(np.exp(p[0])),
            'rS_s':float(np.exp(p[1])),'Us_over_Ua_nominal':float(np.exp(p[2])),
            'Ux_over_Ua_nominal':float(np.exp(p[3])),'input_lag_s':float(np.exp(p[4])),
            'command_curvature':float(p[5]),'Ua_run_over_nominal':np.exp(ds).tolist(),
            'log_deviations':ds.tolist()}}


def dimensional_mean(point,coefficients,records,band):
    fit=dimensional_fit(point,coefficients,records,band)
    predictions=[independent(d,fit) for d in records]
    flat=np.concatenate([np.concatenate([p[1:,k] for p in predictions]) for k in [0,1]])
    return flat


def dense_gaussian_profile(records,residual,noise,kernel='ou_nugget'):
    """Profile common scale using explicit per-run correlation Cholesky factors."""
    N=sum(len(d['t'])-1 for d in records);ratio=np.exp(noise[-1]);sse=0.;logdet=0.
    for k in [0,1]:
        cursor=0
        for d in records:
            t=d['t'][1:];n=len(t);section=slice(k*N+cursor,k*N+cursor+n)
            if kernel=='iid':C=np.eye(n)
            else:
                ell=np.exp(noise[2*k]);eta=noise[2*k+1]
                C=(1-eta)*np.exp(-abs(t[:,None]-t[None,:])/ell)+eta*np.eye(n)
            L=cholesky(C,lower=True);z=solve_triangular(L,residual[section],lower=True)/(ratio if k else 1.)
            sse+=float(z@z);logdet+=float(2*np.log(np.diag(L)).sum())+2*n*(np.log(ratio) if k else 0.)
            cursor+=n
    sigma=np.sqrt(sse/(2*N));nll=.5*(2*N*(np.log(2*np.pi)+1+2*np.log(sigma))+logdet)
    return float(nll),float(sigma),float(sigma*ratio)
