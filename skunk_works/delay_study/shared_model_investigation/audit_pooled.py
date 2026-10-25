"""Independent physical-state propagation and dense likelihood for pooled fits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from scipy.linalg import expm,cholesky,solve_triangular
from data import load
from run_mean import write_json


def physical_reference(t,u,times,physical,ambient):
    m=len(times)
    endpoints=np.repeat(float(ambient),2) if np.ndim(ambient)==0 else np.asarray(ambient)
    A=np.zeros((m+3,m+3))
    for j in range(m-1):
        A[j,j]=-1/times[j+1]
        A[j,j-1 if j else m]=1/times[j+1]
    A[m-1,m-1]=-1/times[0]
    A[m-1,m]=physical[0]/times[0]
    A[m-1,m+1]=1/times[0]
    A[m-1,:m-1]=physical[1:]/times[0]
    A[m+1,m+2]=(endpoints[1]-endpoints[0])/(t[-1]-t[0])
    # Last two components are held command and ambient, not physical states.
    state=np.zeros(m+3);state[m+1]=endpoints[0];state[m+2]=1.
    result=[0.];cache={}
    for k,h in enumerate(np.diff(t)):
        if float(h) not in cache:cache[float(h)]=expm(A*h)
        state[m]=u[k]
        state=cache[float(h)]@state
        result.append(state[m-1])
    return np.asarray(result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    fit=json.loads((args.input/'fit.json').read_text())
    source=json.loads((args.input/'source_snapshot.json').read_text())
    arrays=np.load(args.input/'trajectories.npz')
    best=fit['best'];c=np.array(best['coefficients']);x=np.array(best['point'])
    times=np.array(fit['time_constants_s']);m=len(times)
    nt=len(x)-2
    physical=np.r_[np.zeros(m-1),c[0]] if fit['kind'] in ('chain','transport','unequal_chain') else np.r_[c[1:m],c[0]-sum(c[1:m])]
    ninput=fit.get('input_coefficient_count',1 if fit['kind']=='chain' else m)
    width=fit.get('ambient_width',1)
    ell,eta=np.exp(x[nt]),x[nt+1]
    quadratic=0.;logdet=0.;n=0;rows=[]
    for j,name in enumerate(fit['record_names']):
        d=load(name)
        raw=HERE.parents[2]/d.source
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=source['sha256'][d.source]:raise AssertionError('Raw binding')
        if fit['kind']=='transport':
            from transport import independent_reference
            predicted=d.temperature[0]+c[0]*independent_reference(d.time,d.input,*times,x[2])
            elapsed=d.time-d.time[0];a=-np.expm1(-elapsed/times[0])
            ambient=c[ninput+j] if width==1 else c[ninput+2*j:ninput+2*j+2]
            if width==1:predicted+=ambient*a
            else:
                b=(elapsed-times[0]*a)/(d.time[-1]-d.time[0]);predicted+=ambient[0]*(a-b)+ambient[1]*b
        else:
            ambient=c[ninput+j] if width==1 else c[ninput+2*j:ninput+2*j+2]
            predicted=d.temperature[0]+physical_reference(d.time,d.input,times,physical,ambient)
        error=float(np.max(np.abs(predicted-arrays['predicted_'+str(j)])))
        residual=d.temperature[1:]-predicted[1:]
        t=d.time[1:]
        C=np.eye(len(t)) if fit.get('kernel')=='iid' else (1-eta)*np.exp(-abs(t[:,None]-t[None,:])/ell)+eta*np.eye(len(t))
        L=cholesky(C,lower=True)
        z=solve_triangular(L,residual,lower=True)
        ie=float(np.max(np.abs(z/best['sigma_C']-arrays['standardized_innovation_'+str(j)])))
        if error>2e-7 or ie>2e-5:raise AssertionError((name,error,ie))
        quadratic+=z@z;logdet+=2*np.log(np.diag(L)).sum();n+=len(t)
        rows.append({'name':name,'trajectory_max_abs_C':error,'standardized_innovation_max_abs':ie})
    nll=.5*(n*(np.log(2*np.pi)+1+np.log(quadratic/n))+logdet)
    nllerror=abs(nll-best['nll'])
    if nllerror>3e-5:raise AssertionError(nllerror)
    p=fit['parameter_count']
    aicc=2*nll+2*p+2*p*(p+1)/(n-p-1)
    if abs(aicc-fit['nominal_AICc'])>6e-5:raise AssertionError('AICc')
    result={'status':'passed','kind':fit['kind'],'order':fit['order'],'producer_commit':source['commit'],
            'fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
            'negative_log_likelihood':float(nll),'nll_abs_error':float(nllerror),'per_run':rows,
            'scope':'Independent physical-state matrix propagation, raw hashes, dense covariance/innovations/likelihood and AICc arithmetic; no adequacy or global minimum claim'}
    write_json(args.output/'audit.json',result)
    print(json.dumps({k:result[k] for k in ('status','kind','order','nll_abs_error')}),flush=True)


if __name__=='__main__':main()
