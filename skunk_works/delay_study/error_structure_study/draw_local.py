"""Audited active-face local-normal parameter-mixing sensitivity for transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from physical import load_records
from joint_quantized import Model
from run_bounded import write


def conditional_draws(point,bounds,H,basis,count,seed,maximum_proposals=1000000):
    """Normal on the selected active face, truncated by the remaining hard box."""
    precision=basis.T@H@basis;precision=(precision+precision.T)/2
    L=cholesky(precision,lower=True);rng=np.random.RandomState(seed);accepted=[];attempted=0
    while len(accepted)<count and attempted<maximum_proposals:
        z=rng.normal(size=(basis.shape[1],512))
        candidates=point[:,None]+basis@solve_triangular(L.T,z,lower=False)
        for candidate in candidates.T:
            attempted+=1
            if np.all(candidate>=bounds[:,0]-1e-10) and np.all(candidate<=bounds[:,1]+1e-10):accepted.append(candidate.copy())
            if len(accepted)>=count or attempted>=maximum_proposals:break
    if len(accepted)!=count:raise RuntimeError('Local truncated-normal rejection rate too high; no completed mixture sensitivity')
    return np.array(accepted),attempted


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--information',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--draws',type=int,default=256)
    parser.add_argument('--seed',type=int,default=931771);args=parser.parse_args()
    fitpath=args.input/'fit.json';fit=json.loads(fitpath.read_text());fitsha=hashlib.sha256(fitpath.read_bytes()).hexdigest()
    audit=json.loads((args.audit/'audit.json').read_text());assert audit['status']=='passed' and audit['fit_sha256']==fitsha
    assert json.loads((args.audit/'exit.json').read_text())['exit_code']==0
    information=json.loads((args.information/'information.json').read_text());assert information['fit_sha256']==fitsha
    matrix_path=args.information/'information_matrices.npz';matrixsha=hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    assert audit['matrix_sha256']==matrixsha
    if fit.get('dataset') or fit.get('fixed_coordinates') or fit.get('condition_initial_code'):raise ValueError('Use a supported unconditioned hardware fit')
    records=[d for d in load_records() if d['name'] in fit['record_names']]
    m=Model(records,fit['band'],fit['phase'],128,fit['family'],minimum_sensor_loading=fit.get('minimum_sensor_loading',.005))
    assert not fit.get('startup_band_C') and not fit.get('initial_sensor_bins')
    assert m.coordinate_names()==information['coordinate_names']
    point=np.array(fit['best']['point']);bounds=np.array(m.bounds());arrays=np.load(matrix_path)
    points,proposals=conditional_draws(point,bounds,arrays['observed_Hessian'],arrays['active_face_basis'],args.draws,args.seed)
    assert all(abs(m.centered_constraint(p))<1e-7 for p in points)
    notice=('Local-normal mixing on the selected active-bound face, truncated by inactive box limits, in explicit optimization coordinates. Conditions on the selected likelihood basin and fixes active loading/deviation bounds; excludes released-bound/profile branches, model selection and initial/bin convention uncertainty. Not calibrated simultaneous confidence or a posterior distribution. Combine with hard unknown-run nuisance ranges and separately reported model/initial/profile sensitivities; empirical gain-LR recovery tests do not validate this whole mixture.')
    method='observed-bin-Hessian active-face local normal, box truncated'
    for count in sorted(set([min(64,args.draws),args.draws])):
        np.savez_compressed(args.output/('parameter_draws_'+str(count)+'.npz'),points=points[:count],fit_sha256=np.array(fitsha),method=np.array(method),calibration_notice=np.array(notice))
    write(args.output/'draw_receipt.json',{'status':'complete','fit_sha256':fitsha,'information_matrix_sha256':matrixsha,
        'information_audit_sha256':hashlib.sha256((args.audit/'audit.json').read_bytes()).hexdigest(),
        'seed':args.seed,'draws':args.draws,'proposals':proposals,'acceptance_fraction':args.draws/proposals,
        'fixed_active_coordinates':information['active_coordinates'],'method':method,'calibration_notice':notice})
    print(json.dumps({'status':'complete','draws':args.draws,'proposals':proposals}),flush=True)


if __name__=='__main__':main()
