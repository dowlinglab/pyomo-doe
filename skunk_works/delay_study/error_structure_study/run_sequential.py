"""Record every covariance/mean update and compare alternating with joint ML."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from scipy.optimize import minimize
from bounded import Model
from physical import load_records
from envelope import Objective
from run_bounded import stationary,write
from sequential import covariance_objective,covariance_gradient,residual_at


def mean_update(model,point,maxiter):
    objective=Objective(model);noise=point[model.ns:].copy();nobs=2*model.N
    def full(shape):return np.r_[shape,noise]
    constraints=[{'type':'eq','fun':model.centered_constraint,
                  'jac':lambda x:model.centered_jacobian(x)}] if model.nv else []
    result=minimize(lambda s:objective(full(s))/nobs,point[:model.ns],
        jac=lambda s:objective.gradient(full(s))[:model.ns]/nobs,
        method='SLSQP',bounds=model.shape_bounds(),constraints=constraints,
        options={'ftol':1e-12,'maxiter':maxiter})
    updated=full(result.x)
    return updated,{'solver_success':bool(result.success),'message':str(result.message),
                    'nfev':int(result.nfev),'nll':float(objective(updated)),
                    'stationarity':stationary(model,updated,objective.gradient(updated))}


def serialize(model,point):
    nll,coef,sigma,_,_,_=model.projected(point)
    return {'point':point.tolist(),'nll':float(nll),'coefficients':coef.tolist(),
            'sigma1_C':sigma,'sigma2_C':float(sigma*np.exp(point[-1]))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True)
    parser.add_argument('--initial',choices=['iid','joint'],default='iid')
    parser.add_argument('--cycles',type=int,default=30);parser.add_argument('--maxiter',type=int,default=400)
    args=parser.parse_args();begun=time.monotonic();warm=json.loads(args.warm.read_text())
    records=load_records();model=Model(records,warm['band']);point=np.asarray(warm['best']['point'])
    if len(point)!=len(model.bounds()):raise ValueError('Warm mean/noise chart differs')
    trace=[];all_blocks_success=True
    if args.initial=='iid':
        iid=Model(records,warm['band'],kernel='iid')
        seed=np.r_[point[:model.ns],point[-1]]
        ip,block=mean_update(iid,seed,args.maxiter);all_blocks_success&=block['solver_success']
        # At eta=1 the OU length is unidentified. An interior covariance seed
        # avoids using the derivative in that unidentified direction as a stop.
        point=np.r_[ip[:iid.ns],np.log(30.),.5,np.log(30.),.5,ip[-1]]
        trace.append({'stage':'iid_mean','fit':serialize(iid,ip),'block':block,
            'covariance_seed_note':'Interior eta=.5, ell=30 s after IID fit; full likelihood may increase at this explicit initialization only.'})
    objective=Objective(model);converged=False
    for iteration in range(args.cycles):
        before=serialize(model,point);residual,coef=residual_at(model,point)
        fixed_shape=point[:model.ns].copy();noise=point[model.ns:].copy();noise_bounds=model.bounds()[model.ns:]
        nobs=2*model.N
        result=minimize(lambda q:covariance_objective(q,residual,model.times)/nobs,noise,
            jac=lambda q:covariance_gradient(q,residual,model.times,noise_bounds)/nobs,
            method='L-BFGS-B',bounds=noise_bounds,
            options={'ftol':2.22e-15,'gtol':1e-8,'maxiter':args.maxiter,'maxls':40})
        after_cov=np.r_[fixed_shape,result.x]
        frozen_nll=covariance_objective(result.x,residual,model.times)
        profiled_cov=serialize(model,after_cov)
        point,mean_block=mean_update(model,after_cov,args.maxiter)
        after=serialize(model,point);stat=stationary(model,point,objective.gradient(point))
        # Each optimizer must reduce its own normalized Gaussian objective.
        # Linear coefficient reprojection is a separate nonincreasing update.
        decreases=[before['nll']-frozen_nll,frozen_nll-profiled_cov['nll'],
                   profiled_cov['nll']-after['nll']]
        if min(decreases)<-2e-5:raise RuntimeError('An alternating block increased likelihood: '+str(decreases))
        change=float(np.max(abs(point-np.asarray(before['point']))));improvement=before['nll']-after['nll']
        blocks_ok=bool(result.success and mean_block['solver_success']);all_blocks_success&=blocks_ok
        converged=bool(blocks_ok and improvement<1e-5 and change<2e-4 and
                       stat['lagrangian_projected_gradient_max_abs']<.05 and
                       abs(stat['equality_residual'])<1e-7)
        row={'cycle':iteration,'before':before,'frozen_residual_covariance_nll':frozen_nll,
             'after_linear_reprojection':profiled_cov,'after_mean':after,
             'nll_decreases':decreases,'point_change_max_abs':change,
             'covariance_block':{'solver_success':bool(result.success),'message':str(result.message),
                                 'nfev':int(result.nfev),'fixed_coefficients':coef.tolist()},
             'mean_block':mean_block,'stationarity':stat,'converged':converged}
        trace.append(row);write(args.output/'iterations.json',trace)
        write(args.output/'progress.json',{'cycle':iteration,'nll':after['nll'],
               'improvement':improvement,'point_change':change,'stationarity':stat,
               'elapsed_s':time.monotonic()-begun})
        print(json.dumps({'cycle':iteration,'nll':after['nll'],'improvement':improvement,
                           'point_change':change,'converged':converged}),flush=True)
        if converged:break
    final=serialize(model,point)
    write(args.output/'sequential_fit.json',{'status':'converged' if converged else 'iteration_limit',
        'initial':args.initial,'band':model.band,'kernel':'ou_nugget','ambient':'constant',
        'lag':True,'curve':True,'shape_count':model.ns,'nobs':2*model.N,'final':final,
        'stationarity':stat,'all_blocks_solver_success':all_blocks_success,
        'joint_reference_nll':warm['best']['nll'],'nll_minus_joint_reference':final['nll']-warm['best']['nll'],
        'warm_sha256':hashlib.sha256(args.warm.read_bytes()).hexdigest(),
        'record_names':[d['name'] for d in records],'raw_sha256':{d['name']:d['sha256'] for d in records},
        'elapsed_s':time.monotonic()-begun,
        'notice':'Alternating covariance ML and constrained GLS with covariance normalization and profiled variance. Continuous Gaussian approximation to quantized observations. Iteration caps and block failures retained, not declared convergence.'})


if __name__=='__main__':main()
