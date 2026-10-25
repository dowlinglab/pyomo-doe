"""Refit scalar OU/analogue-noise covariance under the conditional bin likelihood.

Mean trajectories remain fixed. Each channel is fit jointly over twelve runs;
this is a covariance step, not a joint physical-model refit. Quadrature resolution
is raised on failed posterior-normalization checks and checked again at the end.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from scipy.optimize import minimize
from nonlinear import stable_gradient
from covariance_fit import projected_gradient
from quantized import filter_ou,randomized_residual
from decompose import diagnostic


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fit',type=Path,required=True);parser.add_argument('--channel',type=int,choices=[1,2],required=True)
    parser.add_argument('--phase',type=float,default=-.5);parser.add_argument('--starts',type=int,default=2)
    parser.add_argument('--maxiter',type=int,default=80);parser.add_argument('--nodes',type=int,default=128)
    args=parser.parse_args();begun=time.monotonic();f=json.loads((args.fit/'fit.json').read_text());a=np.load(args.fit/'trajectories.npz');k=args.channel-1
    data=[(a['time_'+str(j)][1:],a['predicted_'+str(j)][1:,k],a['observed_'+str(j)][1:,k]) for j in range(len(f['record_names']))]
    N=sum(len(d[0]) for d in data);bounds=[(np.log(.02),np.log(3.)),(np.log(.05),np.log(3000.)),(0.,1.)]
    ns=f['shape_count'];sigma=f['sigma1_C'] if k==0 else f['sigma2_C'];seed=np.array([np.log(sigma),f['best']['point'][ns+2*k],f['best']['point'][ns+2*k+1]])
    evaluations=[]
    @lru_cache(maxsize=12)
    def evaluate(key):
        p=np.array(key);sigma,ell=np.exp(p[:2]);eta=p[2];nll=0.;max_nodes=0;normalization=0.;failure=None
        try:
            for t,mean,y in data:
                nodes=args.nodes
                while True:
                    value=filter_ou(t,mean,y,sigma,ell,eta,args.phase,nodes=nodes)
                    err=float(np.max(abs(value['posterior_normalization_ratio']-1)))
                    if err<=1e-6:break
                    if nodes>=512:raise FloatingPointError('Posterior normalization did not converge through 512 nodes')
                    nodes*=2
                nll+=value['nll'];max_nodes=max(max_nodes,nodes);normalization=max(normalization,err)
        except Exception as exc:
            failure=repr(exc);nll=1e10+float(np.sum((p-seed)**2))*1e4
        row={'point':p.tolist(),'bin_nll':float(nll),'maximum_nodes':max_nodes,'maximum_normalization_error':normalization,
             'failure':failure,'elapsed_s':time.monotonic()-begun}
        evaluations.append(row)
        with (args.output/'evaluations.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        write(args.output/'progress.json',{'evaluations':len(evaluations),'latest':row})
        return float(nll/N)
    objective=lambda p:evaluate(tuple(p))
    gradient=lambda p:stable_gradient(objective,p,bounds,step=2e-4)
    starts=[]
    for j in range(args.starts):
        p=seed.copy()
        if j:p[2]=0.;p[0]+=np.log(.9)
        try:
            fit=minimize(objective,p,jac=gradient,method='L-BFGS-B',bounds=bounds,
                options={'maxiter':args.maxiter,'maxfun':240,'gtol':2e-7,'ftol':1e-11,'maxls':25})
            pg=float(np.max(abs(projected_gradient(fit.x,gradient(fit.x),bounds))))*N
            row={'start':j,'seed':p.tolist(),'point':fit.x.tolist(),'bin_nll':float(fit.fun*N),
                 'success':bool(fit.success and fit.fun<1e8/N),'message':str(fit.message),'projected_gradient_raw':pg}
        except Exception as exc:row={'start':j,'seed':p.tolist(),'success':False,'exception':repr(exc)}
        starts.append(row);write(args.output/'starts.json',starts);print(json.dumps(row),flush=True)
    valid=[r for r in starts if r['success']]
    if not valid:raise RuntimeError('No successful observation-noise fit; preserve trials and diagnose')
    best=min(valid,key=lambda r:r['bin_nll']);p=np.array(best['point']);sigma,ell=np.exp(p[:2]);eta=p[2]
    diagnostics=[];arrays={};full_nll=[];converged=True
    for j,(t,mean,y) in enumerate(data):
        estimates={}
        for nodes in [128,256,512]:
            v=filter_ou(t,mean,y,sigma,ell,eta,args.phase,nodes=nodes);estimates[nodes]=v
            if nodes>=256:
                norm=float(np.max(abs(v['posterior_normalization_ratio']-1)))
                difference=abs(v['nll']-estimates[nodes//2]['nll'])
                if difference<1e-4 and norm<1e-6:break
        okay=difference<1e-4 and norm<1e-6;converged=converged and okay;full_nll.append(v['nll'])
        checks=[]
        for s in [10371,10372,10373,10374,10375]:
            z=randomized_residual(v,s);checks.append({'seed':s,**diagnostic(z,[1,5,10,20,60])})
            arrays['PIT_'+str(j)+'_'+str(s)]=z
        for key in ['log_bin_probability','cdf_lower','cdf_upper','posterior_mean','posterior_SD']:arrays[key+'_'+str(j)]=v[key]
        diagnostics.append({'name':f['record_names'][j],'nodes':nodes,'converged':okay,'node_nll_difference':difference,
                            'maximum_normalization_error':norm,'bin_nll':v['nll'],'PIT':checks})
    result={'channel':args.channel,'phase':args.phase,'mean_fit':str(args.fit),
        'mean_fit_sha256':hashlib.sha256((args.fit/'fit.json').read_bytes()).hexdigest(),
        'mean_arrays_sha256':hashlib.sha256((args.fit/'trajectories.npz').read_bytes()).hexdigest(),
        'parameters':{'sigma_total_analogue_C':float(sigma),'ell_s':float(ell),'independent_variance_fraction':float(eta),
                      'sigma_analogue_measurement_C':float(sigma*np.sqrt(eta))},
        'best':best,'starts':starts,'bounds':bounds,'nobs':N,'validated_bin_nll':float(sum(full_nll)),
        'all_selected_quadrature_converged':converged,'per_run':diagnostics,'elapsed_s':time.monotonic()-begun,
        'notice':'Conditional bin-probability ML with saved physical mean fixed. No mean refit, joint cross-channel covariance, estimated-bin-phase law or interval coverage claim. Numerical posterior-grid approximation is validated by normalization and node-count checks; failed points retained.'}
    write(args.output/'observation_fit.json',result);np.savez_compressed(args.output/'predictive_arrays.npz',**arrays)
    print(json.dumps({'status':'complete','channel':args.channel,'parameters':result['parameters'],'converged':converged}),flush=True)


if __name__=='__main__':main()
