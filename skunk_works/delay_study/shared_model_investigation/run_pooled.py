"""Multistart pooled OU-plus-measurement fits with bounded nuisance projection."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from scipy.optimize import minimize
from data import load
from covariance_fit import numerical_gradient,projected_gradient
from diagnostics import input_correlation
from run_mean import write_json
from pooled_models import PooledModel,audit


def seeds(model,count):
    rng=np.random.RandomState(20261025+model.order)
    nt=2 if model.kind=='chain' else model.m
    if model.kind=='chain':
        initial=[[137.48,18.37],[18.37,137.48],[150.,12.],[130.,30.],[50.,70.],[200.,5.]]
    else:
        initial=[[137.48]+[18.37/(model.m-1)]*(model.m-1),
                 [18.37]+[137.48]+[8.]*(model.m-2),
                 [150.]+[3.,20.,80.][:model.m-1],
                 [120.]+[7.,40.,100.][:model.m-1]]
    bounds=model.bounds()
    while len(initial)<count:
        initial.append([np.exp(rng.uniform(lo,hi)) for lo,hi in bounds[:nt]])
    noise=[[213.87,.0358],[50.,.1],[300.,.02],[30.,.3]]
    return [np.r_[np.log(t),np.log(noise[j%4][0]),noise[j%4][1]] for j,t in enumerate(initial[:count])]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kind',choices=['chain','flexible_real'],required=True)
    parser.add_argument('--order',type=int,required=True)
    parser.add_argument('--starts',type=int,default=24)
    parser.add_argument('--maxiter',type=int,default=600)
    args=parser.parse_args()
    write_json(args.output/'forward_audit.json',audit())
    config=json.loads((HERE.parent/'residual_study'/'config.json').read_text())
    records=[load(n) for n in config['records']]
    model=PooledModel(records,args.kind,args.order)
    bounds=model.bounds()
    objective=lambda x:model.projected(x)[0]
    derivative=lambda x:numerical_gradient(objective,x,bounds,step=2e-6)
    starts=[]; best=None; begun=time.monotonic()
    for j,seed in enumerate(seeds(model,args.starts)):
        write_json(args.output/'progress.json',{'current_start':j,'completed_starts':len(starts),'elapsed_s':time.monotonic()-begun})
        try:
            fit=minimize(objective,seed,jac=derivative,method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':args.maxiter,'maxfun':10000,'ftol':1e-12,'gtol':1e-5,'maxls':35})
            def rowof(f,solver):
                pg=float(np.max(np.abs(projected_gradient(f.x,derivative(f.x),bounds))))
                return {'solver':solver,'success':bool(f.success),'message':str(f.message),
                        'point':f.x.tolist(),'nll':float(f.fun),'projected_gradient_max_abs':pg,
                        'nfev':int(f.nfev),'nit':int(f.nit)}
            attempts=[rowof(fit,'L-BFGS-B')]
            if not fit.success or attempts[0]['projected_gradient_max_abs']>.05:
                recovered=minimize(objective,fit.x,jac=derivative,method='SLSQP',bounds=bounds,
                                   options={'maxiter':args.maxiter,'ftol':1e-10})
                attempts.append(rowof(recovered,'SLSQP refinement'))
                if recovered.success and (not fit.success or recovered.fun<=fit.fun+1e-8):
                    fit=recovered
            row=rowof(fit,'selected'); row.update({'start':j,'seed':seed.tolist(),'attempts':attempts})
            nll,c,sigma,W=model.projected(fit.x)
            row.update({'nll':nll,'coefficients':c.tolist(),'sigma_C':sigma})
            if row['success'] and (best is None or nll<best['nll']): best=row
        except Exception as exc:
            row={'start':j,'seed':seed.tolist(),'success':False,'exception':repr(exc)}
        starts.append(row)
        write_json(args.output/'starts.json',starts)
        print(json.dumps(row),flush=True)
    if best is None: raise RuntimeError('No converged starts')
    point=np.array(best['point']); c=np.array(best['coefficients'])
    nll,c,sigma,W=model.projected(point)
    predictions=model.trajectories(point,c)
    residual=np.concatenate([d.temperature[1:]-p[1:] for d,p in zip(records,predictions)])
    innovations=W(residual)/sigma
    cursor=0; arrays={}; metrics=[]
    for j,(d,p) in enumerate(zip(records,predictions)):
        count=len(d.time)-1; z=innovations[cursor:cursor+count]; cursor+=count
        cc=input_correlation(z,d.input[1:],30)
        values=[abs(r['correlation']) for r in cc if r['correlation'] is not None]
        metrics.append({'name':d.name,'RMSE_C':float(np.sqrt(np.mean((d.temperature[1:]-p[1:])**2))),
                        'mean_residual_C':float(np.mean(d.temperature[1:]-p[1:])),
                        'max_abs_input_innovation_correlation_lags30':max(values) if values else None})
        arrays.update({'time_'+str(j):d.time,'input_'+str(j):d.input,'observed_'+str(j):d.temperature,
                       'predicted_'+str(j):p,'standardized_innovation_'+str(j):z})
    nt=2 if args.kind=='chain' else model.m
    pcount=nt+len(c)+3  # ell, eta, profiled marginal variance
    result={'kind':args.kind,'order':args.order,'state_count':model.m,'record_names':[d.name for d in records],
            'best':best,'starts':starts,'nobs':model.count,'parameter_count':pcount,
            'nominal_AICc':2*nll+2*pcount+2*pcount*(pcount+1)/(model.count-pcount-1),
            'RMSE_C':float(np.sqrt(np.mean(residual**2))),'per_run':metrics,'bounds':bounds,
            'coefficient_bounds':[v.tolist() for v in model.coefficient_bounds()],
            'time_constants_s':model.time_constants(point[:nt]).tolist(),
            'elapsed_s':time.monotonic()-begun,
            'stationarity_warning':best['projected_gradient_max_abs']>.05,
            'all_starts_successful':all(s['success'] for s in starts),
            'comparison_notice':'Same twelve records, exact left-held input, first T fixed, per-run±2C physical ambient forcing, shared OU+nugget reestimated. Flexible model permits free numerator and unequal real poles; it is not unrestricted LTI. Nominal likelihood criteria do not certify residual adequacy.'}
    write_json(args.output/'fit.json',result)
    np.savez_compressed(args.output/'trajectories.npz',**arrays)
    print(json.dumps({k:result[k] for k in ('kind','order','RMSE_C','nominal_AICc','elapsed_s','stationarity_warning')}),flush=True)


if __name__=='__main__': main()
