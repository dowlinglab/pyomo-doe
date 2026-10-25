"""Independent dimensional radiative ODE and dense Gaussian likelihood audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import cholesky,solve_triangular
from physical import load_records
from radiative import Model


def direct(d,fit,j):
    p=fit['parameters'];ua0=.04;ua=ua0*p['Ua_run_over_nominal'][j]
    us=ua0*p['Us_over_Ua_nominal'];ux=ua0*p['Ux_over_Ua_nominal'];CH=ua0*p['tA_nominal_s'];CS=us*p['rS_s']
    beta=ua0*p['K_common_nominal'];rad=ua0*p['gamma_differential_50C_over_Ua']/(4*323.15**3)
    ambient=d['y'][0,0]+fit['coefficients'][1+j];L=p['input_lag_s'];c=p['command_curvature']
    state=np.r_[np.repeat(d['y'][0],2),0.];out=[d['y'][0]]
    for k,dt in enumerate(np.diff(d['t'])):
        u=d['u'][k]+c*d['u'][k]*(1-d['u'][k]/100)
        def balance(t,z):
            h1,s1,h2,s2,power=z
            return [(-ua*(h1-ambient)+us*(s1-h1)+ux*(h2-h1)+beta*power-rad*((h1+273.15)**4-(ambient+273.15)**4))/CH,
                    us*(h1-s1)/CS,(-ua*(h2-ambient)+us*(s2-h2)+ux*(h1-h2)-rad*((h2+273.15)**4-(ambient+273.15)**4))/CH,
                    us*(h2-s2)/CS,(u-power)/L]
        f=solve_ivp(balance,[0,dt],state,rtol=2e-11,atol=2e-12);assert f.success
        state=f.y[:,-1];out.append(state[[1,3]])
    return np.array(out)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    f=json.loads((args.input/'fit.json').read_text());a=np.load(args.input/'trajectories.npz');records=load_records()
    p=np.array(f['best']['point']);n=f['shape_count'];deltas=np.array(f['parameters']['log_deviations']);band=f['band']
    assert abs(sum(deltas))<1e-7 and np.all(np.exp(deltas)>=1-band-1e-7) and np.all(np.exp(deltas)<=1+band+1e-7)
    assert not f['stationarity_warning'],'Selected fit fails raw stationarity threshold'
    rows=[];nll=0.;nobs=0
    for j,d in enumerate(records):
        assert d['sha256']==f['raw_sha256'][d['name']]
        pred=direct(d,f,j);error=float(np.max(abs(pred-a['predicted_'+str(j)])));assert error<3e-6,(d['name'],error)
        innovation_error=[]
        for k in [0,1]:
            t=d['t'][1:];residual=d['y'][1:,k]-pred[1:,k];sigma=f['sigma1_C'] if k==0 else f['sigma2_C']
            ell,eta=np.exp(p[n+2*k]),p[n+2*k+1]
            C=(1-eta)*np.exp(-abs(t[:,None]-t[None,:])/ell)+eta*np.eye(len(t));L=cholesky(C,lower=True)
            z=solve_triangular(L,residual,lower=True)/sigma
            delta=float(np.max(abs(z-a['innovation_'+str(j)][:,k])));assert delta<5e-5,delta
            innovation_error.append(delta);nll+=.5*(z@z+2*np.log(np.diag(L)).sum()+2*len(t)*np.log(sigma)+len(t)*np.log(2*np.pi));nobs+=len(t)
        rows.append({'name':d['name'],'direct_ODE_max_error_C':error,'innovation_max_error':innovation_error})
    assert nobs==f['nobs'];nll_error=abs(nll-f['best']['nll']);assert nll_error<.001,nll_error
    m=Model(records,band,max_step=.125);fine=m.prediction(p)
    saved=np.concatenate([np.concatenate([a['predicted_'+str(j)][1:,k] for j in range(len(records))]) for k in [0,1]])
    step_error=float(np.max(abs(fine-saved)));assert step_error<3e-6,step_error
    result={'status':'passed','fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
            'nll_abs_error':nll_error,'full_record_step_halving_error_C':step_error,'per_run':rows,
            'scope':'Independent dimensional radiative balances, adaptive integration, full-record step halving, raw hashes, dense innovation and Gaussian likelihood arithmetic, selected stationarity. No global optimum or physical adequacy proof.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','nll_error':nll_error,'step_error':step_error}),flush=True)


if __name__=='__main__':main()
