"""Observed bin-likelihood curvature and conditional analogue-information bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.linalg import null_space
from physical import load_records
from covariance import Whitening
from joint_quantized import Model
from run_bounded import write


def eigensystem(matrix,basis,names):
    I=basis.T@matrix@basis;I=(I+I.T)/2;values,vectors=np.linalg.eigh(I);directions=basis@vectors
    rows=[]
    for j in list(range(min(5,len(values))))+list(range(max(5,len(values)-2),len(values))):
        indices=np.argsort(abs(directions[:,j]))[-6:][::-1]
        rows.append({'eigenvalue':float(values[j]),'loadings':{names[i]:float(directions[i,j]) for i in indices}})
    return {'dimension':len(values),'eigenvalues':values.tolist(),'directions':rows,
            'positive_definite':bool(np.min(values)>0),'condition_number_if_positive':float(values[-1]/values[0]) if values[0]>0 else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--step',type=float,default=2e-4)
    parser.add_argument('--columns',type=int,nargs='+');parser.add_argument('--nodes',type=int,default=128)
    args=parser.parse_args();begun=time.monotonic();fit=json.loads((args.input/'fit.json').read_text())
    if fit.get('dataset') or fit.get('fixed_coordinates'):raise ValueError('Use the unconstrained hardware reference')
    lookup={d['name']:d for d in load_records()};records=[lookup[n] for n in fit['record_names']]
    m=Model(records,fit['band'],fit['phase'],args.nodes,fit['family'],fit.get('startup_band_C',0.),
            initial_sensor_bins=fit.get('initial_sensor_bins',False),initial_noise_buffer=fit.get('initial_noise_buffer_C',0.),
            minimum_sensor_loading=fit.get('minimum_sensor_loading',.005),condition_initial_code=fit.get('condition_initial_code',False))
    p=np.array(fit['best']['point']);bounds=np.array(m.bounds());names=m.coordinate_names();g0=m.gradient(p);columns={};stencils={}
    requested=list(range(len(p))) if args.columns is None else args.columns
    for j in requested:
        if j<0 or j>=len(p):raise ValueError('Column out of range')
        h=args.step;u=np.zeros(len(p));u[j]=h
        if bounds[j,0]==bounds[j,1]:column=np.zeros(len(p));stencil='fixed'
        elif p[j]-bounds[j,0]>=h and bounds[j,1]-p[j]>=h:
            column=(m.gradient(p+u)-m.gradient(p-u))/(2*h);stencil='central'
        elif bounds[j,1]-p[j]<h:
            column=(3*g0-4*m.gradient(p-u)+m.gradient(p-2*u))/(2*h);stencil='backward second order'
        else:column=(-3*g0+4*m.gradient(p+u)-m.gradient(p+2*u))/(2*h);stencil='forward second order'
        columns[j]=column;stencils[str(j)]=stencil
        np.savez_compressed(args.output/'columns.npz',**{str(k):v for k,v in columns.items()})
        write(args.output/'progress.json',{'completed_columns':list(columns),'requested_columns':requested,'elapsed_s':time.monotonic()-begun})
        print(json.dumps({'column':j,'name':names[j],'stencil':stencil}),flush=True)
    result={'status':'complete','fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
        'coordinate_names':names,'point':p.tolist(),'gradient':g0.tolist(),'step':args.step,'nodes':args.nodes,
        'columns':requested,'stencils':stencils,'elapsed_s':time.monotonic()-begun,
        'notice':'Observed Hessian in explicit log/linear optimization coordinates. Off-face finite differences are likelihood derivatives, not feasible confidence directions. Active constraints and weak/disconnected branches require profiles; this is not an automatic Wald covariance.'}
    if len(columns)==len(p):
        raw=np.column_stack([columns[j] for j in range(len(p))]);H=(raw+raw.T)/2
        symmetry=float(np.linalg.norm(raw-raw.T)/max(np.linalg.norm(H),1e-30));a=m.centered_jacobian(p)[None,:]
        Q=null_space(a) if np.any(a) else np.eye(len(p));active=(p-bounds[:,0]<1e-5)|(bounds[:,1]-p<1e-5)
        constraints=np.vstack([a,np.eye(len(p))[active]]) if np.any(a) else np.eye(len(p))[active]
        free=null_space(constraints) if len(constraints) else np.eye(len(p))
        result.update({'Hessian_relative_asymmetry':symmetry,'centered_tangent':eigensystem(H,Q,names),
            'conditional_active_face':eigensystem(H,free,names),'active_coordinates':[names[j] for j in np.flatnonzero(active)]})
        face=(free.T@H@free);ev,vec=np.linalg.eigh((face+face.T)/2);direction_checks=[]
        for j in range(min(3,len(ev))):
            direction=free@vec[:,j];checks=[]
            for h in [args.step,3*args.step]:
                value=float(direction@(m.gradient(p+h*direction)-m.gradient(p-h*direction))/(2*h))
                checks.append({'step':h,'directional_curvature':value,'difference_from_matrix':value-float(ev[j])})
            direction_checks.append({'active_face_eigenvalue':float(ev[j]),'checks':checks,
                'numerically_separated_from_step_change':bool(abs(ev[j])>abs(checks[1]['directional_curvature']-checks[0]['directional_curvature']))})
        result['weak_active_face_direction_step_checks']=direction_checks
        # Unquantized analogue observations contain at least as much expected
        # mean information as their codes when covariance is held known. These
        # matrices are conditional comparisons, not a substitute bin Hessian.
        mean,J=m.mean(p)
        matrices={'observed_Hessian':H,'raw_Hessian':raw,'centered_basis':Q,'active_face_basis':free,'mean_Jacobian':J}
        if not m.condition_initial_code:
            weighted=[];iid=[]
            for k,noise in enumerate(fit['noise_parameters']):
                S=noise['latent_SD_C']**2;R=noise['measurement_SD_C']**2
                W=Whitening([d['t'][1:] for d in records],'ou_nugget',[noise['ell_s'],R/(S+R)])
                block=J[k*m.N:(k+1)*m.N];weighted.append(W(block)/np.sqrt(S+R));iid.append(block/np.sqrt(S+R))
            jw=np.vstack(weighted);ji=np.vstack(iid)
            matrices.update({'analogue_mean_known_noise':jw.T@jw,'iid_mean_same_marginal_variance':ji.T@ji})
            result['conditional_analogue_mean_information_notice']='Known covariance, unreported analogue temperatures: an expected-information upper bound for quantized mean information. IID comparison uses the same marginal variance. Neither includes covariance estimation, threshold uncertainty or model selection.'
        else:
            result['conditional_analogue_mean_information_notice']='Not supplied: conditioning on an initial code gives a non-Gaussian initial latent distribution, so an unconditional Gaussian covariance matrix is not this conditional model information. The observed bin Hessian remains available.'
        np.savez_compressed(args.output/'information_matrices.npz',**matrices)
    write(args.output/'information.json',result)


if __name__=='__main__':main()
