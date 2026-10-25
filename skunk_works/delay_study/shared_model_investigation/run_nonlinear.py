"""Bounded shared quadratic-loss fit with OU+nugget and per-run ambient offsets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from scipy.optimize import minimize
from covariance import Whitening
from covariance_fit import numerical_gradient,projected_gradient
from data import load
from diagnostics import input_correlation
from run_mean import write_json
from nonlinear import predict_rise,audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--starts',type=int,default=6)
    parser.add_argument('--max-step',type=float,default=.1)
    parser.add_argument('--maxiter',type=int,default=500)
    parser.add_argument('--warm-start',type=Path)
    parser.add_argument('--fixed-beta',type=float)
    args=parser.parse_args()
    write_json(args.output/'forward_audit.json',audit(args.max_step))
    base=json.loads(args.reference.read_text());best=base['best']
    records=[load(n) for n in base['record_names']];N=sum(len(d.time)-1 for d in records)
    if base.get('ambient_width',1)!=1:raise ValueError('Constant-ambient reference required')
    point=np.r_[best['point'][:3],np.log(best['coefficients'][0]),0.,best['point'][-2:],best['coefficients'][1:]]
    bounds=[(np.log(5.),np.log(600.)),(np.log(.01),np.log(200.)),(0.,15.),(np.log(.2),np.log(1.5)),(0.,2.),(np.log(.05),np.log(3000.)),(0.,1.)]+[(-2.,2.)]*len(records)
    def state(x):
        tau,theta=np.exp(x[:2]);K=np.exp(x[3]);beta=.01*x[4]
        predictions=[d.temperature[0]+predict_rise(d.time,d.input,K,tau,theta,x[2],beta,x[7+j],args.max_step) for j,d in enumerate(records)]
        raw=np.concatenate([d.temperature[1:]-p[1:] for d,p in zip(records,predictions)])
        W=Whitening([d.time[1:] for d in records],'ou_nugget',[np.exp(x[5]),x[6]])
        z=W(raw);sigma=float(np.sqrt(z@z/N));nll=float(.5*(N*(np.log(2*np.pi)+1+2*np.log(sigma))+W.logdet))
        return nll,predictions,raw,z/sigma,sigma
    nested=state(point)[0]
    if abs(nested-best['nll'])>1e-5:raise AssertionError(('Reference beta0 likelihood',nested,best['nll']))
    write_json(args.output/'nested_reference_audit.json',{'status':'passed','nll_abs_error':abs(nested-best['nll'])})
    if args.fixed_beta is not None and not 0<=args.fixed_beta<=.02:raise ValueError('Fixed beta outside declared bounds')
    decode=lambda x:np.insert(x,4,args.fixed_beta/.01) if args.fixed_beta is not None else x
    optbounds=[b for j,b in enumerate(bounds) if j!=4] if args.fixed_beta is not None else bounds
    function=lambda x:state(decode(x))[0]
    gradient=lambda x:numerical_gradient(function,x,optbounds,step=2e-6)
    starts=[];chosen=None;begun=time.monotonic()
    for j in range(args.starts):
        seed=point.copy();seed[4]=[.1,0.,.3,.6,.03,1.][j%6]
        if j==0 and args.warm_start:seed=np.array(json.loads(args.warm_start.read_text())['best']['point'])
        if j==3 or (args.fixed_beta is not None and j==2):seed[:2]=point[1::-1]
        if j>=6:
            rng=np.random.RandomState(805+j);seed[:2]+=rng.uniform(-.15,.15,2);seed[2]=np.clip(seed[2]+rng.uniform(-1,1),0,15)
        write_json(args.output/'progress.json',{'current_start':j,'completed':len(starts),'elapsed_s':time.monotonic()-begun})
        try:
            optseed=np.delete(seed,4) if args.fixed_beta is not None else seed
            fit=minimize(function,optseed,jac=gradient,method='L-BFGS-B',bounds=optbounds,
                         options={'maxiter':args.maxiter,'maxfun':12000,'ftol':1e-12,'gtol':1e-5,'maxls':35})
            pg=float(np.max(abs(projected_gradient(fit.x,gradient(fit.x),optbounds))))
            attempts=[{'solver':'L-BFGS-B','nll':float(fit.fun),'success':bool(fit.success),'projected_gradient_max_abs':pg}]
            if not fit.success or pg>.05:
                second=minimize(function,fit.x,jac=gradient,method='SLSQP',bounds=optbounds,options={'maxiter':args.maxiter,'ftol':1e-10})
                pg2=float(np.max(abs(projected_gradient(second.x,gradient(second.x),optbounds))))
                attempts.append({'solver':'SLSQP refinement','nll':float(second.fun),'success':bool(second.success),'projected_gradient_max_abs':pg2})
                if second.success and (not fit.success or second.fun<=fit.fun+1e-8):fit=second;pg=pg2
            row={'start':j,'seed':seed.tolist(),'point':decode(fit.x).tolist(),'nll':float(fit.fun),'success':bool(fit.success),'message':str(fit.message),'projected_gradient_max_abs':pg,'nfev':int(fit.nfev),'nit':int(fit.nit),'attempts':attempts}
            if row['success'] and (chosen is None or row['nll']<chosen['nll']):chosen=row
        except Exception as exc:row={'start':j,'seed':seed.tolist(),'success':False,'exception':repr(exc)}
        starts.append(row);write_json(args.output/'starts.json',starts);print(json.dumps(row),flush=True)
    if chosen is None:raise RuntimeError('No converged nonlinear starts')
    x=np.array(chosen['point']);nll,predictions,raw,z,sigma=state(x);cursor=0;metrics=[];arrays={}
    for j,(d,p) in enumerate(zip(records,predictions)):
        n=len(d.time)-1;local=z[cursor:cursor+n];cursor+=n
        cc=input_correlation(local,d.input[1:],30);valid=[abs(v['correlation']) for v in cc if v['correlation'] is not None]
        metrics.append({'name':d.name,'RMSE_C':float(np.sqrt(np.mean((d.temperature[1:]-p[1:])**2))),'max_abs_input_innovation_correlation_lags30':max(valid) if valid else None})
        arrays.update({'time_'+str(j):d.time,'observed_'+str(j):d.temperature,'input_'+str(j):d.input,'predicted_'+str(j):p,'standardized_innovation_'+str(j):local})
    parameters={'K':float(np.exp(x[3])),'tau_s':float(np.exp(x[0])),'theta_s':float(np.exp(x[1])),'delay_s':float(x[2]),'beta_per_C':float(.01*x[4]),'ell_s':float(np.exp(x[5])),'eta':float(x[6]),'ambient_offsets_C':x[7:].tolist()}
    count=len(x)+1-(args.fixed_beta is not None)
    result={'kind':'quadratic_loss','best':chosen,'starts':starts,'record_names':base['record_names'],'parameters':parameters,'sigma_C':sigma,'nobs':N,'RMSE_C':float(np.sqrt(np.mean(raw**2))),
            'nominal_AICc':2*nll+2*count+2*count*(count+1)/(N-count-1),'parameter_count':count,'bounds':bounds,
            'per_run':metrics,'max_step_s':args.max_step,'stationarity_warning':chosen['projected_gradient_max_abs']>.05,
            'reference_fit_sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            'warm_start_fit_sha256':hashlib.sha256(args.warm_start.read_bytes()).hexdigest() if args.warm_start else None,
            'fixed_beta_per_C':args.fixed_beta,
            'notice':'One phenomenological quadratic heat-loss term; beta0 nests positive transport model. Input lag assignment retained across nonlinear balance. No full two-temperature model or absolute physical calibration. Splitting solver requires selected-fit independent ODE/convergence check before inference.'}
    write_json(args.output/'fit.json',result);np.savez_compressed(args.output/'trajectories.npz',**arrays)
    print(json.dumps({'status':'complete','parameters':parameters,'RMSE_C':result['RMSE_C'],'nll':nll,'stationarity_warning':result['stationarity_warning']}),flush=True)


if __name__=='__main__':main()
