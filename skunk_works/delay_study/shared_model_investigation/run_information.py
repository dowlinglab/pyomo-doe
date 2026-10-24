"""Scaled conditional mean information and observed projected curvature checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from scipy.special import expit,logit
from data import load
from covariance import Whitening
from pooled_models import PooledModel
from run_mean import write_json


def hessian(function,point,step):
    n=len(point);H=np.empty((n,n));f0=function(point)
    for j in range(n):
        ej=np.zeros(n);ej[j]=step
        H[j,j]=(function(point+ej)-2*f0+function(point-ej))/step**2
        for k in range(j):
            ek=np.zeros(n);ek[k]=step
            H[j,k]=H[k,j]=(function(point+ej+ek)-function(point+ej-ek)-function(point-ej+ek)+function(point-ej-ek))/(4*step**2)
    return H


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=[]
    for folder in args.inputs:
        fit=json.loads((folder/'fit.json').read_text());best=fit['best']
        records=[load(n) for n in fit['record_names']]
        model=PooledModel(records,fit['kind'],fit['order'])
        x=np.array(best['point']);c=np.array(best['coefficients']);sigma=best['sigma_C']
        nt=2 if fit['kind']=='chain' else model.m
        kernel=fit.get('kernel','ou_nugget')
        W=Whitening([d.time[1:] for d in records],kernel,[np.exp(x[-2]),x[-1]] if kernel!='iid' else [])
        X=model.design(tuple(x[:nt]));J=[]
        for k in range(nt):
            xp=x[:nt].copy();xm=x[:nt].copy();xp[k]+=2e-5;xm[k]-=2e-5
            J.append((model.design(tuple(xp))@c-model.design(tuple(xm))@c)/(4e-5))
        J.extend([X[:,k]*c[0] for k in range(model.ninput)])
        J=W(np.column_stack(J))/sigma
        B=W(X[:,model.ninput:])/sigma
        J=J-B@np.linalg.lstsq(B,J,rcond=None)[0]
        U,s,Vh=np.linalg.svd(J,full_matrices=False)
        names=(['log_tau','log_theta'] if fit['kind']=='chain' else ['log_tau']+['log_lag'+str(k) for k in range(1,model.m)])
        names+=['log_K']+['numerator_coefficient'+str(k)+'_scaled_by_K' for k in range(1,model.ninput)]
        info=J.T@J
        eigen=np.linalg.eigvalsh(info)
        ranks={str(threshold):int(np.sum(s>s[0]*threshold)) for threshold in (1e-6,1e-8,1e-10,1e-12)}
        fullrank=bool(ranks['1e-08']==len(names))
        conditional_sd=None
        if fullrank:
            covariance=(Vh.T*(1/s**2))@Vh
            conditional_sd=np.sqrt(np.maximum(np.diag(covariance),0)).tolist()
        curves=[]
        if fit.get('objective_policy','joint_ou')=='joint_ou' and 0<x[-1]<1:
            z=np.r_[x[:-1],logit(x[-1])]
            f=lambda q:model.projected(np.r_[q[:-1],expit(q[-1])])[0]
            matrices=[]
            for step in (2e-4,1e-4):
                H=hessian(f,z,step);matrices.append(H)
                curves.append({'step':step,'eigenvalues':np.linalg.eigvalsh(H).tolist(),'matrix':H.tolist()})
            relative_difference=float(np.linalg.norm(matrices[0]-matrices[1])/max(np.linalg.norm(matrices[1]),1.))
        else:relative_difference=None
        row={'id':folder.name,'kind':fit['kind'],'order':fit['order'],
             'fit_sha256':hashlib.sha256((folder/'fit.json').read_bytes()).hexdigest(),
             'mean_coordinate_order':names,'singular_values':s.tolist(),'information_eigenvalues':eigen.tolist(),
             'relative_singular_value_ranks':ranks,'condition_J':float(s[0]/s[-1]),
             'weakest_normalized_mean_direction':Vh[-1].tolist(),
             'conditional_coordinate_SD':conditional_sd,
             'ambient_bound_count':int(np.sum(np.abs(c[model.ninput:])>2-1e-7)),
             'projected_observed_curvature':curves,'curvature_step_relative_difference':relative_difference,
             'notice':'Conditional mean information fixes covariance and allows all ambient nuisance directions even at active bounds. SDs are local working approximations, not calibrated intervals. Observed curvature uses logit eta and profiles bounded linear coefficients. Pole permutations, coalescence, bounds and misspecification can invalidate regular interpretation.'}
        rows.append(row)
        write_json(args.output/(folder.name+'_information.json'),row)
        print(json.dumps({'id':row['id'],'rank':ranks,'condition_J':row['condition_J'],'curvature_relative_difference':relative_difference}),flush=True)
    write_json(args.output/'summary.json',{'status':'complete','models':rows,
        'scope':'Scaled sensitivity SVD/eigendecomposition and two-step projected curvature; profile checks remain where curvature is indefinite/weak or bounds matter'})


if __name__=='__main__':main()
