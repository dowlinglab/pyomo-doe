"""Scoped pooled gain/total-delay profiles, reestimating all other parameters."""
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
from pooled_models import make_model
from run_mean import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--parameters',nargs='+',choices=['K','theta','delay','power'],default=['K','theta'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest={'status':'running','curves':[]}
    write_json(args.output/'matrix_manifest.json',manifest)
    for folder in args.inputs:
        fit=json.loads((folder/'fit.json').read_text())
        if fit['kind'] not in ('chain','transport','power_curve','mapped_chain'):raise ValueError('Profiles support chains and input-map/transport extensions')
        records=[load(n) for n in fit['record_names']]
        model=make_model(records,fit['kind'],fit['order'],fit.get('numerator_bound',2.),fit.get('ambient_mode','constant'))
        reference=np.array(fit['best']['point']);K=fit['best']['coefficients'][0]
        reference_nll=fit['best']['nll'];bounds=model.bounds()
        fixed_nll=model.projected(reference,fixed_gain=K)[0]
        if abs(fixed_nll-reference_nll)>3e-5:raise AssertionError('Fixed-gain center does not reproduce projected reference')
        for parameter in args.parameters:
            if parameter=='delay' and fit['kind'] not in ('transport','power_curve'):raise ValueError('Additional delay applies only to transport extension')
            if parameter=='power' and fit['kind'] not in ('power_curve','mapped_chain'):raise ValueError('Input-map curvature not present')
            power_index=model.base_nt if fit['kind']=='mapped_chain' else 3
            center=K if parameter=='K' else (reference[power_index] if parameter=='power' else (reference[2] if parameter=='delay' else np.exp(reference[1])))
            points=[];warm=reference.copy();begun=time.monotonic()
            factors=([0.]+[sign*d for d in (.02,.05,.1,.2,.5,1.) for sign in (-1,1)]+[-center]
                     if parameter in ('delay','power') else [0.]+[sign*d for d in (.002,.005,.01,.02,.05,.1) for sign in (-1,1)])
            for displacement in factors:
                value=center+displacement if parameter in ('delay','power') else center*np.exp(displacement)
                if parameter=='K' and not .2<=value<=1.5:continue
                if parameter=='theta' and not .01<=value<=200:continue
                if parameter=='delay' and not 0<=value<=15:continue
                if parameter=='power' and not -.8<=value<=.8:continue
                if parameter=='K':
                    freebounds=bounds
                    decode=lambda q:q
                    objective=lambda q:model.projected(q,fixed_gain=value)[0]
                    seeds=[warm.copy(),reference.copy()]
                else:
                    index=power_index if parameter=='power' else (2 if parameter=='delay' else 1)
                    freebounds=[b for j,b in enumerate(bounds) if j!=index]
                    def decode(q):return np.insert(q,index,value if parameter in ('delay','power') else np.log(value))
                    objective=lambda q:model.projected(decode(q))[0]
                    seeds=[np.delete(warm,index),np.delete(reference,index)]
                # Explore the other thermal-role basin as a numerical check.
                opposite=reference.copy();opposite[:2]=reference[1::-1]
                for j in range(2):opposite[j]=np.clip(opposite[j],*bounds[j])
                seeds.append(opposite if parameter=='K' else np.delete(opposite,index))
                unique=[]
                for seed in seeds:
                    if not any(np.max(abs(seed-prev))<1e-10 for prev in unique):unique.append(seed)
                gradient=lambda q:numerical_gradient(objective,q,freebounds,step=2e-6)
                attempts=[]
                for seed in unique:
                    try:
                        result=minimize(objective,seed,jac=gradient,method='L-BFGS-B',bounds=freebounds,
                                        options={'maxiter':500,'maxfun':6000,'ftol':1e-12,'gtol':1e-5,'maxls':35})
                        def record(r,solver):
                            pg=float(np.max(abs(projected_gradient(r.x,gradient(r.x),freebounds))))
                            return {'solver':solver,'success':bool(r.success),'point':decode(r.x).tolist(),
                                    'nll':float(r.fun),'projected_gradient_max_abs':pg,'message':str(r.message)}
                        attempt=record(result,'L-BFGS-B');attempts.append(attempt)
                        if not attempt['success'] or attempt['projected_gradient_max_abs']>.05:
                            refined=minimize(objective,result.x,jac=gradient,method='SLSQP',bounds=freebounds,
                                             options={'maxiter':500,'ftol':1e-10})
                            attempts.append(record(refined,'SLSQP refinement'))
                    except Exception as exc:attempts.append({'success':False,'exception':repr(exc)})
                valid=[r for r in attempts if r['success']]
                if not valid:raise RuntimeError('No converged profile attempts')
                best=min(valid,key=lambda r:r['nll']);warm=np.array(best['point'])
                _,coefficients,sigma,_=model.projected(warm,fixed_gain=value if parameter=='K' else None)
                point={'value':float(value),'displacement':displacement,'displacement_scale':'absolute' if parameter in ('delay','power') else 'log_ratio','best':best,'attempts':attempts,
                       'coefficients':coefficients.tolist(),'sigma_C':sigma,
                       'twice_delta_nll':2*(best['nll']-reference_nll),
                       'stationarity_warning':best['projected_gradient_max_abs']>.05,
                       'lower_than_reference':best['nll']<reference_nll-1e-6}
                points.append(point)
                name=folder.name+'_'+parameter
                write_json(args.output/(name+'.json'),{'id':name,'parameter':parameter,'model':str(folder),
                           'reference_nll':reference_nll,'reference_value':center,'reference_point':reference.tolist(),
                           'points':points,'status':'running','elapsed_s':time.monotonic()-begun})
                print(json.dumps({'curve':name,'value':value,'twice_delta_nll':point['twice_delta_nll'],'warning':point['stationarity_warning']}),flush=True)
            path=args.output/(name+'.json');curve=json.loads(path.read_text())
            curve['points']=sorted(points,key=lambda r:r['value'])
            curve['status']='complete_with_issues' if any(r['stationarity_warning'] or r['lower_than_reference'] for r in points) else 'complete'
            curve['nominal_95_threshold']=3.841458820694124
            curve['notice']='Nominal profile support under bounded misspecified working likelihood; grid and multistarts do not certify global optimum or95%coverage. Total theta is chain mean delay.'
            write_json(path,curve);manifest['curves'].append({'id':name,'status':curve['status']})
            write_json(args.output/'matrix_manifest.json',manifest)
    manifest['status']='complete_with_issues' if any(r['status']!='complete' for r in manifest['curves']) else 'complete'
    write_json(args.output/'matrix_manifest.json',manifest)


if __name__=='__main__':main()
