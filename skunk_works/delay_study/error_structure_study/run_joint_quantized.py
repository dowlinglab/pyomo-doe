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
from conditional_initial import filter_record


def center_seed(seed,bounds,start,count):
    point=np.clip(np.asarray(seed).copy(),np.array(bounds)[:,0],np.array(bounds)[:,1])
    ds=point[start:start+count].copy();lo=np.array(bounds)[start:start+count,0];hi=np.array(bounds)[start:start+count,1]
    shift=brentq(lambda v:float(np.clip(ds-v,lo,hi).sum()),-1.,1.)
    point[start:start+count]=np.clip(ds-shift,lo,hi)
    return point


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);parser.add_argument('--noise1',type=Path,required=True);parser.add_argument('--noise2',type=Path,required=True)
    parser.add_argument('--family',choices=['conductance','radiation'],default='conductance');parser.add_argument('--band',type=float,default=.1)
    parser.add_argument('--records',type=int,nargs='+');parser.add_argument('--dataset',type=Path);parser.add_argument('--inflate-initial-noise',action='store_true');parser.add_argument('--phase',type=float,default=-.5);parser.add_argument('--nodes',type=int,default=128)
    parser.add_argument('--startup-band',type=float,default=0.);parser.add_argument('--starts',type=int,default=3)
    parser.add_argument('--fix-coordinate',nargs=2,action='append',default=[],metavar=('NAME','VALUE'))
    parser.add_argument('--initial-sensor-bins',action='store_true');parser.add_argument('--initial-noise-buffer',type=float,default=0.)
    parser.add_argument('--minimum-sensor-loading',type=float,default=.005)
    parser.add_argument('--resume-starts',type=Path,help='Re-evaluate saved terminal start points and export; does not repeat optimization')
    parser.add_argument('--condition-initial-code',action='store_true')
    parser.add_argument('--score-backend',choices=['reference','compiled'],default='reference')
    parser.add_argument('--maxiter',type=int,default=300);args=parser.parse_args();begun=time.monotonic()
    if args.starts>3:raise ValueError('Declare additional starts before extending the physical/noise branch comparison')
    records=load_records()
    dataset_receipt=None
    if args.dataset:
        data=np.load(args.dataset,allow_pickle=False)
        names=data['record_names'].tolist()
        assert names==[d['name'] for d in records]
        replaced=[]
        for j,d in enumerate(records):
            assert np.array_equal(data['time_'+str(j)],d['t']) and np.array_equal(data['input_'+str(j)],d['u'])
            v=dict(d);v['y']=data['observed_'+str(j)].copy()
            assert v['y'].shape==d['y'].shape and np.array_equal(v['y'][0],d['y'][0]) and np.isfinite(v['y']).all()
            replaced.append(v)
        records=replaced
        dataset_receipt={'path':str(args.dataset),'sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),'type':'synthetic conditional observations with unchanged commands/times/first readings'}
    if args.records is not None:
        assert len(set(args.records))==len(args.records) and min(args.records)>=0 and max(args.records)<len(records)
        records=[records[j] for j in args.records]
    nr=len(records);warm=json.loads(args.warm.read_text());noise_fits=[json.loads(p.read_text()) for p in [args.noise1,args.noise2]]
    fixed={name:float(value) for name,value in args.fix_coordinate}
    if len(fixed)!=len(args.fix_coordinate):raise ValueError('Repeated fixed coordinate')
    model=Model(records,args.band,args.phase,args.nodes,args.family,args.startup_band,fixed=fixed,
                initial_sensor_bins=args.initial_sensor_bins,initial_noise_buffer=args.initial_noise_buffer,
                minimum_sensor_loading=args.minimum_sensor_loading,condition_initial_code=args.condition_initial_code,score_backend=args.score_backend);bounds=model.bounds();scale=2*model.N
    indices=[warm['record_names'].index(d['name']) for d in records]
    shape=np.asarray(warm['best']['point'][:6]).copy()
    original_delta=np.array(warm['parameters'].get('log_deviations',np.zeros(len(warm['record_names']))))
    delta=original_delta[indices].copy();center=float(delta.mean()) if args.band else 0.
    delta-=center
    shape[[0,2,3]]-=center
    oldcoef=np.array(warm['coefficients']);coef=np.r_[oldcoef[0]*np.exp(-center),oldcoef[1:1+len(warm['record_names'])][indices]]
    if args.family=='conductance':
        extra=np.array(warm.get('startup_offsets_C') or np.zeros((len(warm['record_names']),2)))[indices].ravel() if args.startup_band else np.zeros(0)
        if args.initial_sensor_bins:extra=np.array(warm.get('initial_sensor_offsets_C') or np.zeros((len(warm['record_names']),2)))[indices].ravel()
        initial_mean=np.r_[shape,delta if args.band else np.zeros(0),np.log(coef[0]),coef[1:],extra]
    else:
        gamma=warm['parameters'].get('gamma_differential_50C_over_Ua') or 0.
        initial_mean=np.r_[shape,np.log(coef[0]),gamma*np.exp(-center),delta,coef[1:]]
    noise=[]
    for channel,f in enumerate(noise_fits):
        if warm.get('kernel')=='quantized_ou_nugget':
            p=warm['noise_parameters'][channel]
            noise.extend(np.log([p['latent_SD_C'],p['ell_s'],p['measurement_SD_C']]))
        else:
            p=f['parameters'];sigma=p['sigma_total_analogue_C'];eta=p['independent_variance_fraction']
            noise.extend(np.log([sigma*np.sqrt(1-eta),p['ell_s'],max(sigma*np.sqrt(eta),.0001)]))
    initial=np.r_[initial_mean,noise]
    if args.inflate_initial_noise:
        residual=model.y-model.mean(initial,False)
        for channel in [0,1]:
            rms=float(np.sqrt(np.mean(residual[channel*model.N:(channel+1)*model.N]**2)))
            measurement=np.exp(initial[model.nmean+3*channel+2])
            latent=np.sqrt(max(rms*rms-.3223**2/12-measurement*measurement,.002**2))
            initial[model.nmean+3*channel]=max(initial[model.nmean+3*channel],np.log(latent))
    starts=[];evaluation_count=0;numerical_failures=[]
    for j in range(0 if args.resume_starts else args.starts):
        seed=initial.copy()
        if j==1:seed[1],seed[4]=seed[4],seed[1]
        if j==2:
            seed[model.nmean+2]=np.log(.005);seed[model.nmean+5]=np.log(.005);seed[model.nmean+4]=np.log(10.)
        seed=np.clip(seed,np.array(bounds)[:,0],np.array(bounds)[:,1])
        if args.band:
            seed=center_seed(seed,bounds,6 if args.family=='conductance' else 8,nr)

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
            # A compact atomic point checkpoint supports future interruption
            # without expanding every historical trace row by all coordinates.
            checkpoint=args.output/('checkpoint_start_'+str(j)+'.json')
            temporary=checkpoint.with_suffix('.tmp')
            write(temporary,{'start':j,'completed_callback_iterations':len(trace),
                'point':np.asarray(p).tolist(),'nll':value,'score_backend':args.score_backend,
                'elapsed_s':time.monotonic()-begun})
            temporary.replace(checkpoint)
        try:
            fit=minimize(evaluate,seed,jac=True,method='SLSQP',bounds=bounds,callback=callback,
                constraints=[{'type':'eq','fun':model.centered_constraint,'jac':model.centered_jacobian}] if args.band else [],
                options={'maxiter':args.maxiter,'ftol':1e-12})
            nll=model.objective(fit.x);stat=stationary(model,fit.x,model.gradient(fit.x))
            row={'start':j,'branch_seed':['saved branch','sensor/input lag swapped','low analogue noise'][j],'seed':seed.tolist(),'point':fit.x.tolist(),
                 'nll':float(nll),'solver_success':bool(fit.success),'message':str(fit.message),'stationarity':stat,
                 'success':bool(fit.success and abs(stat['equality_residual'])<1e-7 and stat['lagrangian_projected_gradient_max_abs']<.05)}
        except Exception as exc:row={'start':j,'seed':seed.tolist(),'success':False,'exception':repr(exc)}
        starts.append(row);write(args.output/'starts.json',starts);print(json.dumps({k:v for k,v in row.items() if k not in ['seed','point']}),flush=True)
    if args.resume_starts:
        saved=json.loads(args.resume_starts.read_text())
        for old in saved:
            row=dict(old);row['saved_nll']=old.get('nll');row['saved_stationarity']=old.get('stationarity')
            if old.get('success'):
                try:
                    p=np.array(old['point']);bb=np.array(bounds)
                    if len(p)!=len(bounds) or np.any(p<bb[:,0]-1e-7) or np.any(p>bb[:,1]+1e-7):raise ValueError('Saved point chart/bounds differ')
                    value=model.objective(p);stat=stationary(model,p,model.gradient(p))
                    row.update({'nll':float(value),'stationarity':stat,'saved_point_reevaluated':True,
                        'success':bool(old.get('solver_success') and abs(stat['equality_residual'])<1e-7 and stat['lagrangian_projected_gradient_max_abs']<.05)})
                except Exception as exc:row.update({'success':False,'saved_point_reevaluation_exception':repr(exc)})
            starts.append(row)
        write(args.output/'starts.json',starts)
        write(args.output/'resume_receipt.json',{'path':str(args.resume_starts),
            'starts_sha256':hashlib.sha256(args.resume_starts.read_bytes()).hexdigest(),
            'source_receipt_sha256':hashlib.sha256((args.resume_starts.parent/'source_snapshot.json').read_bytes()).hexdigest() if (args.resume_starts.parent/'source_snapshot.json').exists() else None,
            'notice':'Saved terminal points re-evaluated under current committed likelihood and stationarity checks. No optimization rerun or promotion of originally failed starts.'})
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
        if args.initial_sensor_bins:pred[0]+=coefficients[1+nr:].reshape(nr,2)[j]
        for channel in [0,1]:
            section=slice(channel*model.N+cursor,channel*model.N+cursor+n);pred[1:,channel]=mean[section]
            cov=noise_parameters[channel];latent=cov['latent_SD_C'];ell=cov['ell_s'];measurement=cov['measurement_SD_C']
            first,_=model.initial_mean_and_index(p,j,channel)
            f=filter_record(d,np.r_[first,mean[section]],channel,latent,ell,measurement,args.phase,nodes=256,tail_sd=12.,condition=args.condition_initial_code)
            stream=10371+100000*channel+1000*j;arrays['PIT_'+str(j)+'_'+str(channel)]=randomized_residual(f,stream)
            diagnostics.append({'name':d['name'],'channel':channel+1,'seed':stream,'bin_nll_256':f['nll'],
                                'normalization_error':float(np.max(abs(f['posterior_normalization_ratio']-1)))})
        arrays.update({'time_'+str(j):d['t'],'observed_'+str(j):d['y'],'predicted_'+str(j):pred,'input_'+str(j):d['u']})
        per_run.append({'name':d['name'],'RMSE_by_channel_C':np.sqrt(np.mean((d['y'][1:]-pred[1:])**2,axis=0)).tolist()});cursor+=n
    parameter_count=int(sum(bool(lo<hi) for lo,hi in bounds)-int(args.band>0))
    result={'model':'joint_quantized_'+args.family,'family':args.family,'kernel':'quantized_ou_nugget','band':args.band,'phase':args.phase,
        'startup_band_C':args.startup_band,'initial_noise_inflation':args.inflate_initial_noise,'dataset':dataset_receipt,'selected_input_indices':args.records,'best':best,'starts':starts,'bounds':bounds,'nmean':model.nmean,'shape_count':model.nmean,
        'fixed_coordinates':fixed,'coordinate_names':model.coordinate_names(),
        'initial_sensor_bins':args.initial_sensor_bins,'initial_noise_buffer_C':args.initial_noise_buffer,
        'minimum_sensor_loading':args.minimum_sensor_loading,
        'resumed_starts':str(args.resume_starts) if args.resume_starts else None,
        'condition_initial_code':args.condition_initial_code,'score_backend':args.score_backend,
        'initial_sensor_offsets_C':coefficients[1+nr:].reshape(nr,2).tolist() if args.initial_sensor_bins else None,
        'startup_offsets_C':coefficients[1+nr:].reshape(nr,2).tolist() if args.startup_band else None,
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
        'notice':'Mean and both independent-channel OU/analogue-noise models re-estimated under conditional bin probabilities. raw_sha256 binds original experimental design files; dataset identifies any synthetic observation override. Discrete likelihood and assumed bin phase; no cross-channel noise component, uncertainty calibration or external validation claimed. All twelve records informed structure selection; record_names gives the actual fit subset. Observed information and boundary profiles remain necessary.'}
    write(args.output/'fit.json',result);np.savez_compressed(args.output/'trajectories.npz',**arrays)
    print(json.dumps({'status':'complete','family':args.family,'bin_nll':best['nll'],'RMSE':result['RMSE_by_channel_C'],'parameters':result['parameters'],'noise':noise_parameters}),flush=True)


if __name__=='__main__':main()
