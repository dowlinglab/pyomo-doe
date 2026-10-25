"""Audit selected quadratic-loss trajectories with independent accurate ODEs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from data import load
from nonlinear import reference,predict_rise
from run_mean import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    fit=json.loads((args.input/'fit.json').read_text());source=json.loads((args.input/'source_snapshot.json').read_text())
    arrays=np.load(args.input/'trajectories.npz');p=fit['parameters']
    rows=[];quadratic=0.;logdet=0.;N=0;SSE=0.
    for j,name in enumerate(fit['record_names']):
        d=load(name);raw=HERE.parents[2]/d.source
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=source['sha256'][d.source]:raise AssertionError('Raw binding')
        parameters=[p[k] for k in ('K','tau_s','theta_s','delay_s','beta_per_C')]+[p['ambient_offsets_C'][j]]
        rise=reference(d.time,d.input,*parameters)
        pred=d.temperature[0]+rise
        half=d.temperature[0]+predict_rise(d.time,d.input,*parameters,fit['max_step_s']/2)
        error=float(np.max(abs(pred-arrays['predicted_'+str(j)])))
        half_error=float(np.max(abs(half-pred)))
        r=d.temperature[1:]-pred[1:];t=d.time[1:]
        C=(1-p['eta'])*np.exp(-abs(t[:,None]-t[None,:])/p['ell_s'])+p['eta']*np.eye(len(t))
        L=cholesky(C,lower=True);z=solve_triangular(L,r,lower=True)
        quadratic+=z@z;logdet+=2*np.log(np.diag(L)).sum();N+=len(t);SSE+=r@r
        if error>3e-5 or half_error>3e-5:raise AssertionError((name,error,half_error))
        rows.append({'name':name,'selected_max_abs_C':error,'half_step_max_abs_C':half_error})
    nll=.5*(N*(np.log(2*np.pi)+1+np.log(quadratic/N))+logdet)
    nll_error=abs(nll-fit['best']['nll']);rmse=float(np.sqrt(SSE/N))
    if nll_error>.005:raise AssertionError(('Integration affects likelihood beyond declared tolerance',nll_error))
    if abs(rmse-fit['RMSE_C'])>3e-5:raise AssertionError('RMSE')
    result={'status':'passed','input_fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
            'independent_nll':float(nll),'nll_abs_error':float(nll_error),'RMSE_C':rmse,'per_run':rows,
            'max_temperature_tolerance_C':3e-5,'max_nll_tolerance':.005,
            'scope':'Every selected trajectory, raw hashes, accurate independent ODE, half integration step and dense Gaussian likelihood; does not certify stationary/global optimum or adequacy'}
    write_json(args.output/'audit.json',result)
    print(json.dumps({k:result[k] for k in ('status','nll_abs_error','RMSE_C')}),flush=True)


if __name__=='__main__':main()
