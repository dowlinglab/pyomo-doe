"""Joint physical/noise maximum likelihood for quantized two-output records."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.optimize import minimize,brentq
from physical import load_records
from joint_quantized import Model
from run_bounded import stationary,write
from quantized import filter_ou,randomized_residual


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);parser.add_argument('--noise1',type=Path,required=True);parser.add_argument('--noise2',type=Path,required=True)
    parser.add_argument('--family',choices=['conductance','radiation'],default='conductance');parser.add_argument('--band',type=float,default=.1)
    parser.add_argument('--phase',type=float,default=-.5);parser.add_argument('--nodes',type=int,default=128)
    parser.add_argument('--startup-band',type=float,default=0.);parser.add_argument('--starts',type=int,default=2)
    parser.add_argument('--maxiter',type=int,default=300);args=parser.parse_args();begun=time.monotonic()
    if args.starts>2:raise ValueError('Declare additional starts before extending the two branch comparison')
    records=load_records();nr=len(records);warm=json.loads(args.warm.read_text());noise_fits=[json.loads(p.read_text()) for p in [args.noise1,args.noise2]]
    model=Model(records,args.band,args.phase,args.nodes,args.family,args.startup_band);bounds=model.bounds();scale=2*model.N
    shape=np.asarray(warm['best']['point'][:6]);delta=np.array(warm['parameters'].get('log_deviations',np.zeros(nr)));coef=np.array(warm['coefficients'])
    if args.family=='conductance':
        extra=np.array(warm.get('startup_offsets_C') or np.zeros((nr,2))).ravel() if args.startup_band else np.zeros(0)
        initial_mean=np.r_[shape,delta if args.band else np.zeros(0),np.log(coef[0]),coef[1:1+nr],extra]
    else:
        if warm.get('model')=='convection_ambient_radiation':initial_mean=np.array(warm['best']['point'][:model.nmean])
        else:initial_mean=np.r_[shape,np.log(coef[0]),0.,delta,coef[1:1+nr]]
    noise=[]
    for f in noise_fits:
        p=f['parameters'];sigma=p['sigma_total_analogue_C'];eta=p['independent_variance_fraction']
        noise.extend(np.log([sigma*np.sqrt(1-eta),p['ell_s'],max(sigma*np.sqrt(eta),.0001)]))
    initial=np.r_[initial_mean,noise];starts=[];evaluation_count=0;numerical_failures=[]
    for j in range(args.starts):
        seed=initial.copy()
        if j:seed[1],seed[4]=seed[4],seed[1]
        seed=np.clip(seed,np.array(bounds)[:,0],np.array(bounds)[:,1])
        if args.band:
            start=6 if args.family=='conductance' else 8;ds=seed[start:start+nr].copy();lo,hi=np.log(1-args.band),np.log(1+args.band)
            shift=brentq(lambda v:float(np.clip(ds-v,lo,hi).sum()),-1.,1.);seed[start:start+nr]=np.clip(ds-shift,lo,hi)

        trace=[]
        # Failed quadrature points are retained as numerical failures and given
        # an optimization barrier. Accepted results must evaluate the actual
        # likelihood and scores successfully, then pass a finer-grid audit.
        def evaluate(p):
            nonlocal evaluation_count
            evaluation_count+=1
            try:
                value=model.objective(p);gradient=model.gradient(p)
                if not np.isfinite(value) or not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite joint likelihood or score')
            except Exception as exc:
                row={'point':np.asarray(p).tolist(),'exception':repr(exc),'elapsed_s':time.monotonic()-begun};numerical_failures.append(row)
                write(args.output/'numerical_failures.json',numerical_failures)
                value=1e8+1e5*np.sum((p-seed)**2);gradient=2e5*(p-seed)
            write(args.output/'progress.json',{'current_start':j,'evaluations':evaluation_count,'latest_nll':float(value),'elapsed_s':time.monotonic()-begun})
            return float(value/scale),gradient/scale
        def callback(p):
            try:value=model.objective(p)
            except Exception:value=None
            trace.append({'iteration':len(trace),'nll':value,'elapsed_s':time.monotonic()-begun})
            write(args.output/('trace_start_'+str(j)+'.json'),trace)
        try:
            fit=minimize(evaluate,seed,jac=True,method='SLSQP',bounds=bounds,callback=callback,
                constraints=[{'type':'eq','fun':model.centered_constraint,'jac':model.centered_jacobian}] if args.band else [],
                options={'maxiter':args.maxiter,'ftol':1e-12})
            nll=model.objective(fit.x);stat=stationary(model,fit.x,model.gradient(fit.x))
            row={'start':j,'branch_seed':'sensor/input lag swapped' if j else 'saved branch','seed':seed.tolist(),'point':fit.x.tolist(),
                 'nll':float(nll),'solver_success':bool(fit.success),'message':str(fit.message),'stationarity':stat,
                 'success':bool(fit.success and abs(stat['equality_residual'])<1e-7 and stat['lagrangian_projected_gradient_max_abs']<.05)}
        except Exception as exc:row={'start':j,'seed':seed.tolist(),'success':False,'exception':repr(exc)}
        starts.append(row);write(args.output/'starts.json',starts);print(json.dumps({k:v for k,v in row.items() if k not in ['seed','point']}),flush=True)
    valid=[r for r in starts if r['success']]
    if not valid:raise RuntimeError('No feasible stationary joint quantized fit; preserve trials and diagnose')
    best=min(valid,key=lambda r:r['nll']);p=np.array(best['point']);mean=model.mean(p,False);residual=model.y-mean
    if args.family=='conductance':
        shared=p[:6];ds=p[6:6+nr] if args.band else np.zeros(nr);coefficients=np.r_[np.exp(p[model.nshape]),p[model.nshape+1:model.nmean]];gamma=None
    else:
        shared=p[:6];ds=p[8:8+nr];coefficients=np.r_[np.exp(p[6]),p[8+nr:model.nmean]];gamma=float(p[7])
    arrays={};per_run=[];noise_parameters=[];diagnostics=[];cursor=0
    for channel in [0,1]:
        latent,ell,measurement=np.exp(p[model.nmean+3*channel:model.nmean+3*channel+3])
        noise_parameters.append({'latent_SD_C':float(latent),'ell_s':float(ell),'measurement_SD_C':float(measurement)})
    for j,d in enumerate(records):
        n=len(d['t'])-1;pred=d['y'].copy()
        for channel in [0,1]:
            section=slice(channel*model.N+cursor,channel*model.N+cursor+n);pred[1:,channel]=mean[section]
            cov=noise_parameters[channel];latent=cov['latent_SD_C'];ell=cov['ell_s'];measurement=cov['measurement_SD_C']
            f=filter_ou(d['t'][1:],mean[section],d['y'][1:,channel],np.hypot(latent,measurement),ell,measurement**2/(latent**2+measurement**2),args.phase,nodes=256)
            stream=10371+100000*channel+1000*j;arrays['PIT_'+str(j)+'_'+str(channel)]=randomized_residual(f,stream)
            diagnostics.append({'name':d['name'],'channel':channel+1,'seed':stream,'bin_nll_256':f['nll'],
                                'normalization_error':float(np.max(abs(f['posterior_normalization_ratio']-1)))})
        arrays.update({'time_'+str(j):d['t'],'observed_'+str(j):d['y'],'predicted_'+str(j):pred,'input_'+str(j):d['u']})
        per_run.append({'name':d['name'],'RMSE_by_channel_C':np.sqrt(np.mean((d['y'][1:]-pred[1:])**2,axis=0)).tolist()});cursor+=n
    parameter_count=sum(lo<hi for lo,hi in bounds)-int(args.band>0)
    result={'model':'joint_quantized_'+args.family,'family':args.family,'kernel':'quantized_ou_nugget','band':args.band,'phase':args.phase,
        'startup_band_C':args.startup_band,'best':best,'starts':starts,'bounds':bounds,'nmean':model.nmean,'shape_count':model.nmean,
        'coefficients':coefficients.tolist(),'noise_parameters':noise_parameters,'record_names':[d['name'] for d in records],
        'raw_sha256':{d['name']:d['sha256'] for d in records},'nobs':2*model.N,'parameter_count':parameter_count,
        'parameters':{'K_common_nominal':float(coefficients[0]),'tA_nominal_s':float(np.exp(shared[0])),'rS_s':float(np.exp(shared[1])),
            'Us_over_Ua_nominal':float(np.exp(shared[2])),'Ux_over_Ua_nominal':float(np.exp(shared[3])),
            'input_lag_s':float(np.exp(shared[4])),'command_curvature':float(shared[5]),
            'gamma_differential_50C_over_Ua':gamma,'Ua_run_over_nominal':np.exp(ds).tolist(),'log_deviations':ds.tolist()},
        'RMSE_by_channel_C':[float(np.sqrt(np.mean(residual[k*model.N:(k+1)*model.N]**2))) for k in [0,1]],
        'nominal_AICc':2*best['nll']+2*parameter_count+2*parameter_count*(parameter_count+1)/(2*model.N-parameter_count-1),
        'per_run':per_run,'selected_quadrature':model.details(p),'diagnostic_streams':diagnostics,
        'source_inputs_sha256':{str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in [args.warm,args.noise1,args.noise2]},
        'elapsed_s':time.monotonic()-begun,'numerical_failures':len(numerical_failures),
        'notice':'Mean and both independent-channel OU/analogue-noise models re-estimated under conditional bin probabilities. Discrete likelihood and assumed bin phase; no cross-channel noise component, uncertainty calibration or external validation claimed. All twelve records reused. Observed information and boundary profiles remain necessary.'}
    write(args.output/'fit.json',result);np.savez_compressed(args.output/'trajectories.npz',**arrays)
    print(json.dumps({'status':'complete','family':args.family,'bin_nll':best['nll'],'RMSE':result['RMSE_by_channel_C'],'parameters':result['parameters'],'noise':noise_parameters}),flush=True)


if __name__=='__main__':main()
