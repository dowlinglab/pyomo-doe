"""Matched Gaussian joint refits with parsimonious cross-channel disturbances."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.optimize import minimize
from physical import load_records
from coupled_noise import Model,covariance,filter_gaussian
from run_bounded import stationary,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gaussian',type=Path,required=True);parser.add_argument('--quantized',type=Path,required=True)
    parser.add_argument('--mode',choices=['independent','process','measurement','both'],required=True)
    parser.add_argument('--maxiter',type=int,default=300);args=parser.parse_args();begun=time.monotonic()
    gaussian=json.loads(args.gaussian.read_text());quantized=json.loads(args.quantized.read_text());records=load_records()
    assert gaussian['band']==quantized['band'] and gaussian['record_names']==quantized['record_names']==[d['name'] for d in records]
    assert not gaussian.get('startup_offsets_C') and gaussian['ambient']=='constant'
    assert not quantized.get('condition_initial_code') and not quantized.get('initial_sensor_bins')
    m=Model(records,gaussian['band'],args.mode);nt=m.nshape;shape=np.array(gaussian['best']['point'][:nt])
    coefficients=np.array(gaussian['coefficients']);noise=[]
    for k in [0,1]:
        ell=np.exp(gaussian['best']['point'][nt+2*k]);eta=gaussian['best']['point'][nt+2*k+1]
        sigma=gaussian['sigma1_C'] if k==0 else gaussian['sigma2_C']
        noise.extend(np.log([max(.002,sigma*np.sqrt(1-eta)),ell,max(.0001,sigma*np.sqrt(eta))]))
    base=np.r_[shape,np.log(coefficients[0]),coefficients[1:],noise,np.zeros(m.nnoise-6)]
    swapped=base.copy();swapped[[1,4]]=swapped[[4,1]];swapped[m.nmean+6:]=.3
    binseed=np.r_[quantized['best']['point'],np.full(m.nnoise-6,-.3)]
    seeds=[base,swapped,binseed];bounds=m.bounds();starts=[]
    for index,seed in enumerate(seeds):
        seed=np.clip(seed,np.array(bounds)[:,0],np.array(bounds)[:,1]);trace=[];failures=[];evaluations=[0]
        def evaluate(p):
            evaluations[0]+=1
            try:
                value,gradient=m.evaluate(tuple(p))
                if not np.isfinite(value) or not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite Gaussian objective/score')
                write(args.output/'progress.json',{'start':index,'evaluations':evaluations[0],'nll':value,'elapsed_s':time.monotonic()-begun})
                return value,gradient
            except Exception as exc:
                failures.append({'point':p.tolist(),'exception':repr(exc)});write(args.output/('failures_'+str(index)+'.json'),failures)
                return 1e20,np.zeros(len(p))
        def callback(p):
            value,score=m.evaluate(tuple(p));trace.append({'point':p.tolist(),'nll':value})
            write(args.output/('trace_'+str(index)+'.json'),trace)
        result=minimize(evaluate,seed,jac=True,method='SLSQP',bounds=bounds,
            constraints=[{'type':'eq','fun':m.centered_constraint,'jac':m.centered_jacobian}],callback=callback,
            options={'maxiter':args.maxiter,'ftol':1e-9})
        value,score=m.evaluate(tuple(result.x));stat=stationary(m,result.x,score)
        accepted=bool(result.success and stat['lagrangian_projected_gradient_max_abs']<.05 and abs(m.centered_constraint(result.x))<1e-7)
        row={'start':index,'seed':seed.tolist(),'point':result.x.tolist(),'nll':value,'solver_success':bool(result.success),
            'message':str(result.message),'stationarity':stat,'success':accepted,'numerical_failures':len(failures)}
        starts.append(row);write(args.output/'starts.json',starts);print(json.dumps({k:row[k] for k in ['start','nll','success','message']}),flush=True)
    valid=[r for r in starts if r['success']]
    if not valid:raise RuntimeError('No accepted joint Gaussian cross-channel fit; retain starts and diagnose')
    best=min(valid,key=lambda r:r['nll']);p=np.array(best['point']);mean=m.physical.mean(p[:m.nmean+6],False)
    S,R,ell=covariance(p[m.nmean:],args.mode);arrays={};cursor=0;per=[]
    for j,d in enumerate(records):
        n=len(d['t'])-1;pred=d['y'].copy()
        for k in [0,1]:pred[1:,k]=mean[k*m.N+cursor:k*m.N+cursor+n]
        value,g,z,conditional=filter_gaussian(d['t'][1:],d['y'][1:]-pred[1:],S,R,ell)
        arrays.update({'time_'+str(j):d['t'],'observed_'+str(j):d['y'],'predicted_'+str(j):pred,'input_'+str(j):d['u'],
            'innovation_'+str(j):z,'conditional_analogue_mean_'+str(j):pred[1:]+conditional})
        per.append({'name':d['name'],'nll':value,'RMSE_by_channel_C':np.sqrt(np.mean((d['y'][1:]-pred[1:])**2,axis=0)).tolist()});cursor+=n
    coeff=np.r_[np.exp(p[nt]),p[nt+1:m.nmean]];ds=p[6:nt] if m.band else np.zeros(len(records))
    output={'model':'continuous_Gaussian_cross_channel_comparison','family':'conductance','mode':args.mode,'band':m.band,
        'kernel':'bivariate_ou_nugget','best':best,'starts':starts,'nmean':m.nmean,'coordinate_names':m.coordinate_names(),
        'record_names':[d['name'] for d in records],'raw_sha256':{d['name']:d['sha256'] for d in records},
        'coefficients':coeff.tolist(),'noise_parameters':[{'latent_SD_C':float(np.sqrt(S[k,k])),'ell_s':float(ell[k]),'measurement_SD_C':float(np.sqrt(R[k,k]))} for k in [0,1]],
        'latent_stationary_covariance_C2':S.tolist(),'measurement_covariance_C2':R.tolist(),
        'parameters':{'K_common_nominal':float(coeff[0]),'tA_nominal_s':float(np.exp(p[0])),'rS_s':float(np.exp(p[1])),
            'Us_over_Ua_nominal':float(np.exp(p[2])),'Ux_over_Ua_nominal':float(np.exp(p[3])),'input_lag_s':float(np.exp(p[4])),
            'command_curvature':float(p[5]),'Ua_run_over_nominal':np.exp(ds).tolist(),'log_deviations':ds.tolist()},
        'per_run':per,'source_inputs_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.gaussian,args.quantized]},
        'elapsed_s':time.monotonic()-begun,
        'notice':'Continuous-Gaussian matched covariance sensitivity on quantized hardware temperatures. Independent mode nests the prior Gaussian objective; process/measurement correlations are working disturbance terms. This is not the quantized joint likelihood and its innovations do not establish calibrated bin probabilities. Independent full-data trajectory/dense likelihood audits and comparison with mean-linked nonrandomized diagnostics remain required.'}
    write(args.output/'fit.json',output);np.savez_compressed(args.output/'trajectories.npz',**arrays)


if __name__=='__main__':main()
