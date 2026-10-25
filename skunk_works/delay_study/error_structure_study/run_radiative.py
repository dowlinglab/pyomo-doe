"""Matched two-output convection/radiation fit with centered run conductances."""
import argparse
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.optimize import minimize
from physical import load_records
from radiative import Model,TREF
from run_bounded import stationary,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);parser.add_argument('--band',type=float,default=.1)
    parser.add_argument('--gamma-fixed',type=float);parser.add_argument('--curve-fixed',type=float)
    parser.add_argument('--starts',type=int,default=4);parser.add_argument('--maxiter',type=int,default=600)
    parser.add_argument('--max-step',type=float,default=.25);args=parser.parse_args();begun=time.monotonic()
    warm=json.loads(args.warm.read_text());records=load_records();nr=len(records)
    assert warm['ambient']=='constant' and warm['lag'] and warm['curve']
    model=Model(records,args.band,args.max_step,args.gamma_fixed,args.curve_fixed)
    base=np.array(warm['best']['point'][:6]);K=warm['coefficients'][0]
    delta=np.array(warm['parameters'].get('log_deviations',np.zeros(nr)))
    ambient=np.array(warm['coefficients'][1:1+nr]);noise=warm['best']['point'][warm['shape_count']:]
    warm_gamma=warm['parameters'].get('gamma_differential_50C_over_Ua',0.)
    secant_factor=(TREF+296.15)*(TREF*TREF+296.15**2)/(4*TREF**3)
    seeds=[args.gamma_fixed]*args.starts if args.gamma_fixed is not None else [0.,.25,.75,1.5][:args.starts]
    if args.starts>4:raise ValueError('Declare additional radiation starts before expanding this driver')
    starts=[];bounds=model.bounds()
    for j,gamma in enumerate(seeds):
        shape=base.copy();factor=(1+gamma*secant_factor)/(1+warm_gamma*secant_factor)
        shape[[0,2,3]]+=np.log(factor)
        if args.curve_fixed is not None:shape[5]=args.curve_fixed
        seed=np.r_[shape,np.log(K*factor),gamma,delta,ambient,noise]
        seed=np.clip(seed,np.array(bounds)[:,0],np.array(bounds)[:,1])
        write(args.output/'progress.json',{'current_start':j,'completed_starts':len(starts),'elapsed_s':time.monotonic()-begun})
        try:
            scale=2*model.N
            trace=[]
            def callback(x):
                value=model.objective(x)
                trace.append({'iteration':len(trace),'nll':value,'gamma':float(x[7]),'elapsed_s':time.monotonic()-begun})
                write(args.output/('trace_start_'+str(j)+'.json'),trace)
            f=minimize(lambda p:model.objective(p)/scale,seed,jac=lambda p:model.gradient(p)/scale,method='SLSQP',bounds=bounds,callback=callback,
                constraints=[{'type':'eq','fun':model.centered_constraint,'jac':model.centered_jacobian}],
                options={'maxiter':args.maxiter,'ftol':1e-12})
            stat=stationary(model,f.x,model.gradient(f.x))
            row={'start':j,'gamma_seed':gamma,'seed':seed.tolist(),'point':f.x.tolist(),'nll':float(f.fun*scale),'optimizer_scale':scale,
                 'solver_success':bool(f.success),'message':str(f.message),'nfev':int(f.nfev),'stationarity':stat,
                 'success':bool(f.success and abs(stat['equality_residual'])<1e-7)}
        except Exception as exc:row={'start':j,'gamma_seed':gamma,'seed':seed.tolist(),'success':False,'exception':repr(exc)}
        starts.append(row);write(args.output/'starts.json',starts)
        print(json.dumps({k:v for k,v in row.items() if k not in ['seed','point']}),flush=True)
    valid=[r for r in starts if r['success']]
    if not valid:raise RuntimeError('No feasible converged radiation start')
    best=min(valid,key=lambda r:r['nll']);p=np.array(best['point']);prediction=model.prediction(p)
    residual=model.y-prediction;nll,sigma,W,rho,z=model.profile_noise(p,residual)
    innovations=z/sigma;arrays={};cursor=0;per_run=[]
    for j,d in enumerate(records):
        n=len(d['t'])-1;pred=d['y'].copy();innovation=np.empty((n,2))
        for k in [0,1]:
            section=slice(k*model.N+cursor,k*model.N+cursor+n);pred[1:,k]=prediction[section];innovation[:,k]=innovations[section]
        per_run.append({'name':d['name'],'RMSE_by_channel_C':np.sqrt(np.mean((d['y'][1:]-pred[1:])**2,axis=0)).tolist()})
        arrays.update({'time_'+str(j):d['t'],'input_'+str(j):d['u'],'observed_'+str(j):d['y'],'predicted_'+str(j):pred,'innovation_'+str(j):innovation});cursor+=n
    parameter_count=len(p)-(args.gamma_fixed is not None)-(args.curve_fixed is not None)
    fit={'model':'convection_ambient_radiation','band':args.band,'gamma_fixed':args.gamma_fixed,'curve_fixed':args.curve_fixed,
         'lag':True,'curve':True,'ambient':'constant','kernel':'ou_nugget','shape_count':model.nmean,
         'best':best,'starts':starts,'bounds':bounds,'coefficients':np.r_[np.exp(p[6]),p[8+nr:model.nmean]].tolist(),
         'sigma1_C':sigma,'sigma2_C':sigma*rho,'record_names':[d['name'] for d in records],
         'raw_sha256':{d['name']:d['sha256'] for d in records},'nobs':2*model.N,'parameter_count':parameter_count,
         'parameters':{'K_common_nominal':float(np.exp(p[6])),'tA_nominal_s':float(np.exp(p[0])),
             'rS_s':float(np.exp(p[1])),'Us_over_Ua_nominal':float(np.exp(p[2])),'Ux_over_Ua_nominal':float(np.exp(p[3])),
             'input_lag_s':float(np.exp(p[4])),'command_curvature':float(p[5]),'gamma_differential_50C_over_Ua':float(p[7]),
             'Ua_run_over_nominal':np.exp(p[8:8+nr]).tolist(),'log_deviations':p[8:8+nr].tolist()},
         'RMSE_by_channel_C':[float(np.sqrt(np.mean(residual[k*model.N:(k+1)*model.N]**2))) for k in [0,1]],
         'nominal_AICc':2*nll+2*parameter_count+2*parameter_count*(parameter_count+1)/(2*model.N-parameter_count-1),
         'per_run':per_run,'max_RK4_step_s':args.max_step,
         'stationarity_warning':best['stationarity']['lagrangian_projected_gradient_max_abs']>.05,
         'elapsed_s':time.monotonic()-begun,
         'notice':'Air and large-enclosure radiative surroundings assumed equal within run. Ua means convection; all loss/gain terms refit. Gamma is normalized conductance, not a measured emissivity. No interheater radiation, absolute power calibration, new data or quantized likelihood in this fit. Noise and bounded-parameter assumptions remain conditional.'}
    write(args.output/'fit.json',fit);np.savez_compressed(args.output/'trajectories.npz',**arrays)
    print(json.dumps({'status':'complete','nll':nll,'RMSE':fit['RMSE_by_channel_C'],'gamma':p[7],'stationarity_warning':fit['stationarity_warning']}),flush=True)


if __name__=='__main__':main()
