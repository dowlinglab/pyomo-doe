"""Fit centered hard-band conductance deviations with joint colored-error ML."""
import argparse
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from scipy.optimize import minimize
from physical import load_records
from bounded import Model
from envelope import Objective
from covariance_fit import projected_gradient


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def stationary(model,point,gradient):
    g=gradient.copy();a=model.centered_jacobian(point)
    bounds=np.asarray(model.bounds())
    free=(point>bounds[:,0]+1e-6)&(point<bounds[:,1]-1e-6)&(a!=0)
    multiplier=-float(g[free].mean()) if np.any(free) else 0.
    pg=projected_gradient(point,g+multiplier*a,model.bounds())
    return {'equality_residual':model.centered_constraint(point),
            'lagrangian_projected_gradient_max_abs':float(np.max(abs(pg))),
            'equality_multiplier_free_least_squares':multiplier,
            'note':'First-order diagnostic; inspect active-bound KKT multiplier if no free deviation remains.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);parser.add_argument('--band',type=float,required=True)
    parser.add_argument('--ambient',choices=['constant','linear'],default='constant')
    parser.add_argument('--starts',type=int,default=4);parser.add_argument('--maxiter',type=int,default=600)
    args=parser.parse_args();begun=time.monotonic()
    warm=json.loads(args.warm.read_text());records=load_records()
    assert warm['lag'] and warm['curve'] and warm['kernel']=='ou_nugget'
    model=Model(records,args.band,ambient=args.ambient)
    p=np.array(warm['best']['point']);point=np.r_[p[:6],np.zeros(model.nv),p[6:]]
    rng=np.random.RandomState(38091);objective=Objective(model);bounds=model.bounds()
    constraints=[{'type':'eq','fun':model.centered_constraint,'jac':model.centered_jacobian}] if model.nv else []
    starts=[]
    for j in range(args.starts):
        seed=point.copy()
        if j:
            seed[:5]+=rng.normal(0,.15,5);seed[5]+=rng.normal(0,.02)
            if model.nv:
                ds=rng.normal(0,1,model.nv);ds-=ds.mean()
                ds*=min(abs(np.log(1-args.band)),np.log(1+args.band))*.6/max(abs(ds))
                seed[6:model.ns]=ds
            seed=np.clip(seed,np.array(bounds)[:,0],np.array(bounds)[:,1])
        write(args.output/'progress.json',{'current_start':j,'completed_starts':len(starts),'elapsed_s':time.monotonic()-begun})
        try:
            f=minimize(objective,seed,jac=objective.gradient,bounds=bounds,constraints=constraints,
                       method='SLSQP',options={'maxiter':args.maxiter,'ftol':1e-9,'disp':False})
            stat=stationary(model,f.x,objective.gradient(f.x))
            row={'start':j,'seed':seed.tolist(),'point':f.x.tolist(),'nll':float(f.fun),
                 'solver_success':bool(f.success),'message':str(f.message),'nfev':int(f.nfev),
                 'stationarity':stat,'success':bool(f.success and abs(stat['equality_residual'])<1e-7)}
        except Exception as exc:
            row={'start':j,'seed':seed.tolist(),'success':False,'exception':repr(exc)}
        starts.append(row);write(args.output/'starts.json',starts)
        print(json.dumps({k:v for k,v in row.items() if k not in ['seed','point']}),flush=True)
    valid=[r for r in starts if r['success']]
    if not valid:raise RuntimeError('No converged feasible multistart; preserve starts and diagnose')
    best=min(valid,key=lambda r:r['nll']);p=np.array(best['point'])
    nll,c,sigma,W,initial,X=model.projected(p);prediction=initial+X@c;error=model.y-prediction
    rho=np.exp(p[-1]);z=np.r_[W[0](error[:model.N])/sigma,W[1](error[model.N:])/(sigma*rho)]
    cursor=0;arrays={};per_run=[]
    for j,d in enumerate(records):
        n=len(d['t'])-1;pred=d['y'].copy();innov=np.empty((n,2))
        for k in (0,1):
            section=slice(k*model.N+cursor,k*model.N+cursor+n)
            pred[1:,k]=d['y'][0,0]+prediction[section];innov[:,k]=z[section]
        arrays.update({'time_'+str(j):d['t'],'input_'+str(j):d['u'],'observed_'+str(j):d['y'],
                       'predicted_'+str(j):pred,'innovation_'+str(j):innov})
        per_run.append({'name':d['name'],'RMSE_by_channel_C':np.sqrt(np.mean((d['y'][1:]-pred[1:])**2,axis=0)).tolist()})
        cursor+=n
    delta=model.deltas(p);g=np.exp(delta)
    parameter_count=len(p)+len(c)+1-(1 if model.nv else 0)
    fit={'model':'bounded_Ua','band':args.band,'lag':True,'curve':True,'ambient':args.ambient,'kernel':'ou_nugget',
         'base_shape_count':model.base_ns,'shape_count':model.ns,'best':best,'starts':starts,
         'coefficients':c.tolist(),'sigma1_C':sigma,'sigma2_C':sigma*rho,'record_names':[d['name'] for d in records],
         'raw_sha256':{d['name']:d['sha256'] for d in records},'bounds':bounds,'nobs':2*model.N,
         'parameter_count':parameter_count,'nominal_AICc':2*nll+2*parameter_count+2*parameter_count*(parameter_count+1)/(2*model.N-parameter_count-1),
         'parameters':{'K_common_nominal':float(c[0]),'tA_nominal_s':float(np.exp(p[0])),
             'rS_s':float(np.exp(p[1])),'Us_over_Ua_nominal':float(np.exp(p[2])),
             'Ux_over_Ua_nominal':float(np.exp(p[3])),'input_lag_s':float(np.exp(p[4])),
             'command_curvature':float(p[5]),'Ua_run_over_nominal':g.tolist(),'log_deviations':delta.tolist()},
         'RMSE_by_channel_C':[float(np.sqrt(np.mean(error[k*model.N:(k+1)*model.N]**2))) for k in (0,1)],
         'per_run':per_run,'stationarity_warning':best['stationarity']['lagrangian_projected_gradient_max_abs']>.05,
         'elapsed_s':time.monotonic()-begun,
         'notice':'One relative conductance varies per run; centered hard bands, all other physical coefficients shared. Absolute Ua is uncalibrated; all data reused. Independent-channel OU+nugget errors and cold-start states are assumptions. Nominal AICc at active bounds is descriptive.'}
    write(args.output/'fit.json',fit);np.savez_compressed(args.output/'trajectories.npz',**arrays)
    print(json.dumps({'status':'complete','nll':nll,'RMSE':fit['RMSE_by_channel_C'],'Ua_ratios':g.tolist(),'stationarity_warning':fit['stationarity_warning']}),flush=True)


if __name__=='__main__':main()
