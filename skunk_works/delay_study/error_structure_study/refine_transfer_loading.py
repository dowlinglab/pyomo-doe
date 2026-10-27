"""Investigate nonpositive fold curvature with declared additional loading starts."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from continuation import parent_receipt
from fit_task import run
from run_bounded import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loading_seed(fit,value,source_sha):
    """A labeled optimization seed with no copied objective or fit-success claim."""
    assert fit['family']=='conductance' and fit['kernel']=='quantized_ou_nugget'
    assert not any(fit.get(k) for k in ['dataset','fixed_coordinates','startup_band_C','initial_sensor_bins','condition_initial_code'])
    index=fit['coordinate_names'].index('log_Us_over_Ua');assert index==2
    point=deepcopy(fit['best']['point']);point[index]=float(np.log(value))
    lo,hi=fit['bounds'][index];assert lo-1e-12<=point[index]<=hi+1e-12
    return {'status':'optimization_seed_only','kernel':fit['kernel'],
        'record_names':deepcopy(fit['record_names']),'best':{'point':point},
        'parameters':{'log_deviations':deepcopy(fit['parameters']['log_deviations'])},
        'coefficients':deepcopy(fit['coefficients']),'noise_parameters':deepcopy(fit['noise_parameters']),
        'seed_provenance':{'source_fit_sha256':source_sha,'changed_coordinate':'log_Us_over_Ua',
            'loading_seed_value':float(value),'notice':'Starting vector only; no measured nll or optimizer outcome is asserted.'}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--fit-audit',type=Path,required=True)
    p.add_argument('--fold',required=True);p.add_argument('--information',type=Path,required=True)
    p.add_argument('--information-audit',type=Path,required=True)
    p.add_argument('--noise1',type=Path,required=True);p.add_argument('--noise2',type=Path,required=True)
    args=p.parse_args();begun=time.monotonic();config=json.loads((HERE/'transfer_config.json').read_text())
    policy=config['weak_curvature_refinement'];collection=json.loads((args.input/'transfer_fits.json').read_text())
    assert json.loads((args.input/'exit.json').read_text())['exit_code']==0
    prior_audit=json.loads((args.fit_audit/'audit.json').read_text())
    assert json.loads((args.fit_audit/'exit.json').read_text())['exit_code']==0
    assert prior_audit['status']=='passed' and prior_audit['kind']=='transfer' and prior_audit['source_sha256']==sha(args.input/'transfer_fits.json')
    selected=[(j,r) for j,r in enumerate(collection['folds']) if r['id']==args.fold];assert len(selected)==1
    index,fold=selected[0];assert fold['success'];old=Path(fold['selected_output']);fit=json.loads((old/'fit.json').read_text())
    check=json.loads((args.fit_audit/('point_'+str(index))/'audit.json').read_text())
    assert check['status']=='passed' and check['fit_sha256']==sha(old/'fit.json')
    parent_receipt(old,require_terminal=True)
    information=json.loads((args.information/'information.json').read_text())
    failed=json.loads((args.information_audit/'audit.json').read_text())
    assert json.loads((args.information/'exit.json').read_text())['exit_code']==0
    assert json.loads((args.information_audit/'exit.json').read_text())['exit_code']!=0
    assert failed['status']=='failed' and failed['failure_kind']=='nonpositive_observed_curvature'
    assert failed['fit_sha256']==information['fit_sha256']==sha(old/'fit.json')
    assert failed['information_sha256']==sha(args.information/'information.json')
    assert failed['matrix_sha256']==sha(args.information/'information_matrices.npz')
    direction=information['conditional_active_face']['directions'][0]
    assert direction['eigenvalue']<=0 and abs(direction['loadings'].get('log_Us_over_Ua',0.))>.9
    records=load_records();assert fit['selected_input_indices']==fold['train_indices']
    assert fit['record_names']==fold['train_names']==[records[j]['name'] for j in fold['train_indices']]
    assert not set(fold['train_indices'])&set(fold['test_indices'])
    for j in fold['train_indices']:assert fit['raw_sha256'][records[j]['name']]==records[j]['sha256']
    write(args.output/'design.json',{'policy':policy,'fold':fold,'prior_fit_sha256':sha(old/'fit.json'),
        'fit_collection_sha256':sha(args.input/'transfer_fits.json'),
        'prior_fit_audit_sha256':sha(args.fit_audit/'audit.json'),
        'failed_information_audit_sha256':sha(args.information_audit/'audit.json'),
        'information_matrix_sha256':sha(args.information/'information_matrices.npz'),
        'scope':'Additional starts for the same training data, model, hard bounds and fitting criterion; no new parameter, test nuisance fit or accepted covariance.'})
    attempts=[{'success':True,'output':str(old),'nll':fit['best']['nll'],
        'role':'Original independently audited fit; retained as a candidate'}]
    for j,value in enumerate(policy['loading_seed_values']):
        warm=args.output/('loading_seed_'+str(j)+'.json')
        write(warm,loading_seed(fit,value,sha(old/'fit.json')))
        result=run(args.output/('seed_'+str(j)),warm,noise1=args.noise1,noise2=args.noise2,
            family=fit['family'],band=fit['band'],phase=fit['phase'],records=fold['train_indices'],
            minimum_sensor_loading=fit.get('minimum_sensor_loading',.005),
            starts=policy['starts_per_seed'],maxiter=policy['maxiter'],score_backend='compiled')
        attempts.append(dict(result,loading_seed_value=value));write(args.output/'attempts.json',attempts)
        write(args.output/'progress.json',{'additional_seed_tasks_attempted':j+1,
            'declared_seed_tasks':len(policy['loading_seed_values']),
            'successful_additional_tasks':sum(r['success'] for r in attempts[1:]),'elapsed_s':time.monotonic()-begun})
    best=min([r for r in attempts if r['success']],key=lambda r:r['nll'])
    updated=dict(fold,success=True,selected_output=best['output'],attempts=attempts,
        curvature_refinement={'original_output':str(old),'original_nll':fit['best']['nll'],
            'selected_nll':best['nll'],'nll_improvement':fit['best']['nll']-best['nll'],
            'notice':'Selection is by successful objective value only, never by covariance definiteness. Selected-point and new information audits remain required.'})
    write(args.output/'transfer_fits.json',{'status':'complete','candidate':collection['candidate'],
        'reference_fit_sha256':collection['reference_fit_sha256'],'config':config,'folds':[updated],
        'elapsed_s':time.monotonic()-begun,
        'notice':'Single-fold additional-start investigation after nonpositive curvature. Original collection/fit/forecasts are unchanged. This selected point is not an accepted uncertainty result until its independent selected-point and information audits pass.'})


if __name__=='__main__':main()
