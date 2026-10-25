"""Joint two-output fit with bounded common ambient and independent colored errors."""
import argparse
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from scipy.optimize import minimize
from covariance_fit import numerical_gradient,projected_gradient
from run_mean import write_json
from physical import Model,load_records,matrices


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lag',action='store_true');parser.add_argument('--curve',action='store_true')
    parser.add_argument('--ambient',choices=['constant','linear'],default='constant')
    parser.add_argument('--kernel',choices=['iid','ou_nugget'],default='ou_nugget')
    parser.add_argument('--starts',type=int,default=12);parser.add_argument('--maxiter',type=int,default=500)
    parser.add_argument('--warm',type=Path);args=parser.parse_args()
    records=load_records();model=Model(records,args.lag,args.curve,args.ambient,args.kernel)
    objective=lambda p:model.projected(p)[0];bounds=model.bounds()
    gradient=lambda p:numerical_gradient(objective,p,bounds,step=2e-6)
    base=[[150,12,.4,.35],[200,20,.2,.4],[100,3,1.,.3],[220,40,1,.5],[80,10,3,.25],[300,2,.1,.6]]
    rng=np.random.RandomState(12713);initial=[]
    for j in range(args.starts):
        shape=np.log(base[j]) if j<len(base) else np.array([rng.uniform(*b) for b in model.shape_bounds()[:4]])
        if args.lag:shape=np.r_[shape,np.log([2.5,5.,1.,10.][j%4])]
        if args.curve:shape=np.r_[shape,[.1,0.,.25,-.1][j%4]]
        noise=[] if args.kernel=='iid' else [np.log([200.,30.,500.][j%3]),.06,np.log([100.,10.,300.][j%3]),.2]
        initial.append(np.r_[shape,noise,np.log(.7)])
    if args.warm:
        warm=json.loads(args.warm.read_text());shape=np.asarray(warm['best']['point'][:warm['shape_count']])
        core=list(shape[:4])
        if args.lag:core.append(shape[4] if warm['lag'] else np.log(2.5))
        if args.curve:core.append(shape[-1] if warm['curve'] else .1)
        noise=[] if args.kernel=='iid' else (warm['best']['point'][warm['shape_count']:] if warm['kernel']!='iid' else [np.log(200),.06,np.log(100),.2,np.log(.7)])
        initial[0]=np.r_[core,noise] if args.kernel!='iid' else np.r_[core,warm['best']['point'][-1]]
    starts=[];begun=time.monotonic()
    for j,seed in enumerate(initial):
        write_json(args.output/'progress.json',{'current_start':j,'completed_starts':len(starts),'elapsed_s':time.monotonic()-begun})
        attempts=[]
        try:
            def entry(f,solver):
                return {'solver':solver,'point':f.x.tolist(),'nll':float(f.fun),'success':bool(f.success),'message':str(f.message),
                        'projected_gradient':float(np.max(abs(projected_gradient(f.x,gradient(f.x),bounds)))),'nfev':int(f.nfev)}
            result=minimize(objective,seed,jac=gradient,method='L-BFGS-B',bounds=bounds,
                options={'maxiter':args.maxiter,'maxfun':14000,'ftol':1e-12,'gtol':1e-5,'maxls':30})
            attempts.append(entry(result,'L-BFGS-B'))
            if not result.success or attempts[-1]['projected_gradient']>.05:
                other=minimize(objective,result.x,jac=gradient,method='SLSQP',bounds=bounds,options={'maxiter':args.maxiter,'ftol':1e-9})
                attempts.append(entry(other,'SLSQP'))
            valid=[r for r in attempts if r['success']]
            if not valid:raise RuntimeError('No converged attempt')
            row=dict(min(valid,key=lambda r:r['nll']));row.update({'seed':seed.tolist(),'start':j,'attempts':attempts})
        except Exception as exc:row={'start':j,'seed':seed.tolist(),'success':False,'exception':repr(exc),'attempts':attempts}
        starts.append(row);write_json(args.output/'starts.json',starts)
        print(json.dumps({k:v for k,v in row.items() if k not in ['attempts','point','seed']}),flush=True)
    valid=[r for r in starts if r['success']]
    if not valid:raise RuntimeError('No successful starts')
    best=min(valid,key=lambda r:r['nll']);point=np.array(best['point'])
    nll,c,sigma,W,initial,X=model.projected(point);prediction=initial+X@c
    residual=model.y-prediction;rho=np.exp(point[-1]);innov=np.r_[W[0](residual[:model.N])/sigma,W[1](residual[model.N:])/(sigma*rho)]
    arrays={};per_run=[];cursor=0
    for j,d in enumerate(records):
        n=len(d['t'])-1;pred=np.empty_like(d['y']);pred[0]=d['y'][0]
        z=np.empty((n,2))
        for k in range(2):
            section=slice(k*model.N+cursor,k*model.N+cursor+n);pred[1:,k]=d['y'][0,0]+prediction[section];z[:,k]=innov[section]
        per_run.append({'name':d['name'],'RMSE_by_channel_C':np.sqrt(np.mean((d['y'][1:]-pred[1:])**2,axis=0)).tolist()})
        arrays.update({'time_'+str(j):d['t'],'input_'+str(j):d['u'],'observed_'+str(j):d['y'],'predicted_'+str(j):pred,'innovation_'+str(j):z});cursor+=n
    ta,rs,r,q=np.exp(point[:4]);A,_,_=matrices(point,args.lag);pcount=len(point)+len(c)+1;nobs=2*model.N
    fit={'lag':args.lag,'curve':args.curve,'ambient':args.ambient,'kernel':args.kernel,'shape_count':model.ns,
         'coordinate_names':['log_tA','log_rS','log_Us_over_Ua','log_Ux_over_Ua']+(['log_input_lag'] if args.lag else [])+(['command_curvature'] if args.curve else [])+([] if args.kernel=='iid' else ['log_ell1','eta1','log_ell2','eta2'])+['log_sigma2_over_sigma1'],
         'best':best,'starts':starts,'coefficients':c.tolist(),'sigma1_C':sigma,'sigma2_C':sigma*rho,
         'parameters':{'K_common':float(c[0]),'tA_s':ta,'rS_s':rs,'Us_over_Ua':r,'Ux_over_Ua':q,'input_lag_s':float(np.exp(point[4])) if args.lag else None,'command_curvature':float(point[model.ns-1]) if args.curve else 0.},
         'thermal_mode_times_s':(-1/np.linalg.eigvals(A[:4,:4])).real.tolist(),'record_names':[d['name'] for d in records],
         'raw_sha256':{d['name']:d['sha256'] for d in records},'per_run':per_run,'N_per_channel':model.N,'nobs':nobs,'parameter_count':pcount,
         'RMSE_by_channel_C':[float(np.sqrt(np.mean(residual[k*model.N:(k+1)*model.N]**2))) for k in range(2)],
         'nominal_AICc':2*nll+2*pcount+2*pcount*(pcount+1)/(nobs-pcount-1),'bounds':bounds,
         'stationarity_warning':best['projected_gradient']>.05,'elapsed_s':time.monotonic()-begun,
         'notice':'Two-output retrospective pooled fit; independent channel colored errors and symmetric thermal properties are assumptions. Initial heater temperatures equal first sensor readings; alpha scale absorbed. No comparison of two-output likelihood with a T1-only likelihood.'}
    write_json(args.output/'fit.json',fit);np.savez_compressed(args.output/'trajectories.npz',**arrays)
    print(json.dumps({'status':'complete','RMSE':fit['RMSE_by_channel_C'],'nll':nll,'parameters':fit['parameters'],'stationarity_warning':fit['stationarity_warning']}),flush=True)


if __name__=='__main__':main()
