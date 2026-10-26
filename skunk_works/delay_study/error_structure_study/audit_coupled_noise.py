"""Full-data dimensional trajectory and dense joint Gaussian audits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from physical import load_records
from audit_bounded import independent
from coupled_noise import Model
from run_bounded import stationary,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args();fit=json.loads((args.input/'fit.json').read_text());a=np.load(args.input/'trajectories.npz')
    records=load_records();assert [d['name'] for d in records]==fit['record_names'];p=np.array(fit['best']['point'])
    m=Model(records,fit['band'],fit['mode']);bounds=np.array(m.bounds());assert np.all(p>=bounds[:,0]-1e-7) and np.all(p<=bounds[:,1]+1e-7)
    assert abs(m.centered_constraint(p))<1e-7
    legacy=dict(fit,lag=True,curve=True,ambient='constant');S=np.array(fit['latent_stationary_covariance_C2']);R=np.array(fit['measurement_covariance_C2'])
    assert np.linalg.eigvalsh(S).min()>0 and np.linalg.eigvalsh(R).min()>0
    ell=np.array([v['ell_s'] for v in fit['noise_parameters']]);total=0.;rows=[]
    for j,d in enumerate(records):
        assert d['sha256']==fit['raw_sha256'][d['name']]
        assert np.array_equal(a['observed_'+str(j)],d['y']) and np.array_equal(a['input_'+str(j)],d['u']) and np.array_equal(a['time_'+str(j)],d['t'])
        predicted=independent(d,legacy);error=float(np.max(abs(predicted-a['predicted_'+str(j)])));assert error<1e-8
        t=d['t'][1:];dt=t[:,None]-t[None,:];n=len(t);C=np.empty((2*n,2*n))
        for k in [0,1]:C[k::2,k::2]=S[k,k]*np.exp(-abs(dt)/ell[k])+R[k,k]*np.eye(n)
        C[0::2,1::2]=S[0,1]*np.exp(-np.maximum(dt,0)/ell[0]-np.maximum(-dt,0)/ell[1])+R[0,1]*np.eye(n)
        C[1::2,0::2]=C[0::2,1::2].T
        L=cholesky(C,lower=True);z=solve_triangular(L,(d['y'][1:]-predicted[1:]).ravel(),lower=True)
        whiten_error=float(np.max(abs(z-a['innovation_'+str(j)].ravel())));assert whiten_error<2e-6
        value=.5*(z@z+2*np.log(np.diag(L)).sum()+len(z)*np.log(2*np.pi));total+=value
        rows.append({'name':d['name'],'physical_mean_error_C':error,'dense_innovation_error':whiten_error,
            'dense_nll':float(value),'saved_run_nll_difference':float(value-fit['per_run'][j]['nll'])})
    current=m.objective(p);stat=stationary(m,p,m.gradient(p));assert abs(total-current)<1e-5 and abs(current-fit['best']['nll'])<1e-6
    assert stat['lagrangian_projected_gradient_max_abs']<.05
    baseline=json.loads(args.baseline.read_text());difference=float(total-baseline['best']['nll'])
    if fit['mode']=='independent':assert abs(difference)<1e-4,('Independent joint objective did not reproduce the baseline',difference)
    write(args.output/'audit.json',{'status':'passed','fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
        'baseline_sha256':hashlib.sha256(args.baseline.read_bytes()).hexdigest(),'mode':fit['mode'],
        'dense_total_nll':float(total),'dense_minus_current_nll':float(total-current),
        'nll_minus_prior_independent_Gaussian':difference,'stationarity':stat,'per_run':rows,
        'scope':'Raw bindings, physical signed-flux trajectories, explicit full two-channel covariance and dense Cholesky innovations/likelihood, feasibility and stationarity. Continuous-Gaussian matched comparison on discrete temperatures, not a joint bin-likelihood or scientific whitening/coverage proof.'})
    print(json.dumps({'status':'passed','mode':fit['mode'],'nll_minus_prior_independent':difference}),flush=True)


if __name__=='__main__':main()
