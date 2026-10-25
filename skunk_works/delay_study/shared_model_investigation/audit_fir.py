"""Audit impulse-bin predictions via input steps and dense covariance solves."""
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
from run_mean import write_json


def step_convolution(time,command,weights,width):
    age=time[:,None]-time[None,:]
    bins=np.floor(age/width).astype(int)
    index=np.clip(bins,0,len(weights)-1)
    cumulative=np.r_[0.,np.cumsum(weights)]
    S=cumulative[index]+weights[index]*np.clip((age-index*width)/width,0.,1.)
    S=np.where(age<0,0.,S)
    S=np.where(age>=len(weights)*width,cumulative[-1],S)
    return S@np.r_[command[0],np.diff(command)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=json.loads((args.input/'fir_results.json').read_text())
    source=json.loads((args.input/'source_snapshot.json').read_text())
    config=json.loads((HERE.parent/'residual_study/config.json').read_text())
    records=[load(n) for n in config['records']]
    tau,ell,eta,sigma=[result['noise_reference'][k] for k in ('tau_ambient_s','ell_s','eta','sigma_C')]
    plans=[]
    for d in records:
        path=HERE.parents[2]/d.source
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256'][d.source]:raise AssertionError('Raw binding')
        t=d.time[1:]
        C=(1-eta)*np.exp(-abs(t[:,None]-t[None,:])/ell)+eta*np.eye(len(t))
        plans.append(cholesky(C,lower=True))
    rows=[];N=sum(len(d.time)-1 for d in records)
    for cell in result['cells']:
        w=np.array(cell['weights']);width=cell['bin_width_s'];sse=0.;quadratic=0.
        if abs(w.sum()-cell['K'])>1e-9 or not .2-1e-9<=cell['K']<=1.5+1e-9:raise AssertionError('Gain')
        for j,d in enumerate(records):
            rise=step_convolution(d.time,d.input,w,width)
            rise+=cell['ambient_offsets'][j]*(-np.expm1(-(d.time-d.time[0])/tau))
            r=d.temperature[1:]-d.temperature[0]-rise[1:]
            z=solve_triangular(plans[j],r,lower=True)
            sse+=r@r;quadratic+=z@z
        rmse=float(np.sqrt(sse/N));data=float(quadratic/(sigma*sigma*N))
        penalty=float(cell['lambda']*np.sum((np.diff(w,n=2)/(.71/len(w)))**2))
        errors={'RMSE_C':abs(rmse-cell['RMSE_C']),'weighted_mean_square':abs(data-cell['weighted_mean_square']),
                'penalty':abs(penalty-cell['penalty']),'objective':abs(data+penalty-cell['objective'])}
        if errors['RMSE_C']>1e-8 or max(errors.values())>2e-5:raise AssertionError((cell['id'],errors))
        frequency=[]
        for period in (30.,120.,300.):
            omega=2*np.pi/period
            H=np.sum(w*np.exp(-1j*omega*(np.arange(len(w))+.5)*width))*np.sinc(omega*width/(2*np.pi))
            frequency.append({'period_s':period,'gain':float(abs(H)),'principal_phase_deg':float(np.angle(H,deg=True))})
        rows.append({'id':cell['id'],'metric_abs_errors':errors,'frequency_response':frequency})
        print(json.dumps({'id':cell['id'],'max_metric_error':max(errors.values())}),flush=True)
    if len(rows)!=16:raise AssertionError('Expected16FIRcells')
    write_json(args.output/'audit.json',{'status':'passed','cells':rows,
        'input_sha256':hashlib.sha256((args.input/'fir_results.json').read_bytes()).hexdigest(),
        'scope':'All16kernels, raw bindings, independent command-step convolution, dense covariance weighted error and penalty/objective arithmetic. Bin/regularization sensitivity is not calibrated uncertainty.'})


if __name__=='__main__':main()
