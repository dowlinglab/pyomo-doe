"""Independent scalar convergence study after a fold's curvature audit fails."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from audit_common import dimensional_mean
from quantized import filter_ou
from run_bounded import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--information',type=Path,required=True)
    args=p.parse_args();fit=json.loads((args.fit/'fit.json').read_text())
    info=json.loads((args.information/'information.json').read_text())
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    assert info['fit_sha256']==sha(args.fit/'fit.json')
    assert json.loads((args.information/'exit.json').read_text())['exit_code']==0
    lookup={d['name']:d for d in load_records()};records=[lookup[n] for n in fit['record_names']]
    assert all(d['sha256']==fit['raw_sha256'][d['name']] for d in records)
    point=np.array(fit['best']['point']);nshape=6+(len(records) if fit['band'] else 0)
    N=sum(len(d['t'])-1 for d in records);cache={};evaluations=[]
    steps=[.02,.01,.005,.002,.001]
    write(args.output/'design.json',{'steps':steps,'stencils':['forward_second_order','central_fourth_order'],
        'absolute_agreement_gate':.001,'fit_sha256':sha(args.fit/'fit.json'),
        'information_sha256':sha(args.information/'information.json'),
        'notice':'Convergence study only. No accepted covariance, eigenvalue clipping, fitted-point change or audit-tolerance relaxation.'})
    def objective(v):
        v=round(float(v),12)
        if v in cache:return cache[v]
        q=point.copy();q[2]+=v
        assert fit['bounds'][2][0]<=q[2]<=fit['bounds'][2][1]
        coef=np.r_[np.exp(q[nshape]),q[nshape+1:-6]]
        mean=dimensional_mean(q,coef,records,fit['band']);value=0.
        for k,noise in enumerate(fit['noise_parameters']):
            S=noise['latent_SD_C']**2;R=noise['measurement_SD_C']**2;cursor=0
            for d in records:
                n=len(d['t'])-1;section=slice(k*N+cursor,k*N+cursor+n)
                result=filter_ou(d['t'][1:],mean[section],d['y'][1:,k],np.sqrt(S+R),noise['ell_s'],R/(S+R),fit['phase'],256,12.)
                assert np.max(abs(result['posterior_normalization_ratio']-1))<1e-6
                value+=result['nll'];cursor+=n
        cache[v]=float(value);evaluations.append({'loading_step':v,'dimensional_forward_nll256':float(value)})
        write(args.output/'evaluations.json',evaluations);return value
    H=float(np.load(args.information/'information_matrices.npz')['observed_Hessian'][2,2]);rows=[]
    f0=objective(0.)
    for h in steps:
        forward=(2*f0-5*objective(h)+4*objective(2*h)-objective(3*h))/h**2
        row={'step':h,'forward_second_order':forward,'stored_H22':H,'forward_absolute_error':abs(forward-H)}
        if point[2]-2*h>=fit['bounds'][2][0] and point[2]+2*h<=fit['bounds'][2][1]:
            central=(-objective(2*h)+16*objective(h)-30*f0+16*objective(-h)-objective(-2*h))/(12*h*h)
            row.update({'central_fourth_order':central,'central_absolute_error':abs(central-H)})
        rows.append(row);write(args.output/'checks.json',rows)
    write(args.output/'diagnosis.json',{'status':'complete','checks':rows,
        'fit_sha256':sha(args.fit/'fit.json'),'information_sha256':sha(args.information/'information.json'),
        'evaluations_sha256':sha(args.output/'evaluations.json'),
        'scope':'Independent dimensional forward-likelihood finite differences at the saved point. Study of cancellation/truncation, not uncertainty acceptance or a new fitted model.'})


if __name__=='__main__':main()
