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
from pooled_models import make_model
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
    parser.add_argument('--skip-curvature',action='store_true')
    args=parser.parse_args();rows=[]
    for folder in args.inputs:
        fit=json.loads((folder/'fit.json').read_text());best=fit['best']
        records=[load(n) for n in fit['record_names']]
        model=make_model(records,fit['kind'],fit['order'],fit.get('numerator_bound',2.),fit.get('ambient_mode','constant'))
        x=np.array(best['point']);c=np.array(best['coefficients']);sigma=best['sigma_C']
        nt=model.nt
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
        names=(['log_tau','log_theta','additional_delay_seconds','power_curvature'] if fit['kind']=='power_curve' else
               (['log_tau','log_theta','additional_delay_seconds'] if fit['kind']=='transport' else
               (['log_tau','log_theta'] if fit['kind']=='chain' else ['log_tau']+['log_lag'+str(k) for k in range(1,model.m)]))
               )
        names+=['log_K']+['numerator_coefficient'+str(k)+'_scaled_by_K' for k in range(1,model.ninput)]
        info=J.T@J
        eigen=np.linalg.eigvalsh(info)
        ranks={str(threshold):int(np.sum(s>s[0]*threshold)) for threshold in (1e-6,1e-8,1e-10,1e-12)}
        fullrank=bool(ranks['1e-08']==len(names))
        conditional_sd=None
        if fullrank:
            covariance=(Vh.T*(1/s**2))@Vh
            conditional_sd=np.sqrt(np.maximum(np.diag(covariance),0)).tolist()
        response_uncertainty=[]
        if fullrank:
            for period in (5.,30.,120.,300.):
                omega=2*np.pi/period;H=model.frequency_response(x,c,omega);D=[]
                for k in range(nt):
                    plus=x.copy();minus=x.copy();plus[k]+=2e-5;minus[k]-=2e-5
                    D.append((model.frequency_response(plus,c,omega)-model.frequency_response(minus,c,omega))/(4e-5))
                for k in range(model.ninput):
                    plus=c.copy();minus=c.copy();plus[k]+=c[0]*2e-5;minus[k]-=c[0]*2e-5
                    D.append((model.frequency_response(x,plus,omega)-model.frequency_response(x,minus,omega))/(4e-5))
                D=np.array(D);gA=(H.conjugate()*D).real/abs(H);gP=(D/H).imag
                response_uncertainty.append({'period_s':period,'gain':float(abs(H)),'phase_deg':float(np.angle(H,deg=True)),
                    'gain_SD':float(np.sqrt(max(0.,gA@covariance@gA))),
                    'phase_SD_deg':float(np.sqrt(max(0.,gP@covariance@gP))*180/np.pi),
                    'reference':'Linear block / local input-map linearization at command50;5s is outside primary excitation' if fit['kind']=='power_curve' else 'Conditional mean transfer;5s is outside primary excitation'})
        permutation=None
        if fit['kind']=='unequal_chain' and model.m==3:
            swapped=x.copy();swapped[1:3]=x[2:0:-1]
            prediction_error=float(np.max(abs(model.design(tuple(x[:nt]))@c-model.design(tuple(swapped[:nt]))@c)))
            nll_error=abs(model.projected(x)[0]-model.projected(swapped)[0])
            if prediction_error>1e-7 or nll_error>2e-5:raise AssertionError('Input-lag permutation invariance')
            permutation={'lag_swap_trajectory_max_abs_C':prediction_error,'lag_swap_nll_abs_error':nll_error,
                         'interpretation':'Exact global label ambiguity of the two input lags; local rank does not remove it'}
        curves=[]
        if not args.skip_curvature and fit.get('objective_policy','joint_ou')=='joint_ou' and 0<x[-1]<1:
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
             'conditional_coordinate_covariance':covariance.tolist() if fullrank else None,
             'conditional_response_uncertainty':response_uncertainty,'global_permutation_check':permutation,
             'ambient_bound_count':int(np.sum(np.abs(c[model.ninput:])>2-1e-7)),
             'projected_observed_curvature':curves,'curvature_step_relative_difference':relative_difference,
             'notice':'Conditional mean information fixes covariance and allows all ambient nuisance directions even at active bounds. SDs are local working approximations, not calibrated intervals. Observed curvature uses logit eta and profiles bounded linear coefficients. Pole permutations, coalescence, bounds and misspecification can invalidate regular interpretation.'}
        rows.append(row)
        write_json(args.output/(folder.name+'_information.json'),row)
        print(json.dumps({'id':row['id'],'rank':ranks,'condition_J':row['condition_J'],'curvature_relative_difference':relative_difference}),flush=True)
    write_json(args.output/'summary.json',{'status':'complete','models':rows,
        'scope':'Scaled sensitivity SVD/eigendecomposition and two-step projected curvature; profile checks remain where curvature is indefinite/weak or bounds matter'})


if __name__=='__main__':main()
