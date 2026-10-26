"""Verify centered-band algebra and unchanged canonical forecast arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from transfer import reference_transport_bounds,checked_scenarios
from run_transfer_fit import folds
from run_bounded import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--reference-audit',type=Path,required=True)
    parser.add_argument('--training-fit',type=Path,required=True);parser.add_argument('--saved-predictions',type=Path,required=True)
    args=parser.parse_args();algebra=[]
    for retained in [9,11]:
        total=12;omitted=total-retained;b=.1;lo,hi=np.log([1-b,1+b]);bounds=reference_transport_bounds(b,total,retained)
        attained=[]
        for extreme in [lo,hi]:
            delta=np.r_[np.full(retained,-omitted*extreme/retained),np.full(omitted,extreme)]
            assert abs(delta.sum())<1e-14 and np.all(delta>=lo-1e-14) and np.all(delta<=hi+1e-14)
            value=np.exp(delta[retained:]-delta[:retained].mean());assert np.ptp(value)==0
            attained.append(float(value[0]))
        assert np.max(abs(bounds-attained))<1e-14
        algebra.append({'retained':retained,'omitted':omitted,'transported_ratio_bounds':bounds.tolist(),
            'attained_extreme_ratios':attained,'global_centering_and_original_bounds_verified':True})
    config=json.loads((HERE/'transfer_config.json').read_text());reference=json.loads((args.reference/'fit.json').read_text())
    audit=json.loads(args.reference_audit.read_text());assert audit['status']=='passed' and audit['fit_sha256']==sha(args.reference/'fit.json')
    records=load_records();assert reference['record_names']==[d['name'] for d in records] and reference['band']==.1
    delta=np.array(reference['parameters']['log_deviations']);assert abs(delta.sum())<1e-7
    diagnostic=[]
    for fold in folds(config,'primary'):
        if fold['type']!='leave_protocol_out':continue
        shift=float(delta[fold['train_indices']].mean());ratios=np.exp(delta[fold['test_indices']]-shift)
        diagnostic.append({'fold_id':fold['id'],'training_log_reference_shift':shift,
            'all_data_fit_recentered_test_ratios':ratios.tolist(),
            'test_names':[records[j]['name'] for j in fold['test_indices']],
            'outside_canonical_training_band':((ratios<.9-1e-7)|(ratios>1.1+1e-7)).tolist(),
            'notice':'All-data fitted reference diagnostic, not observed conductance or an input to the blind forecast bounds.'})
    fit=json.loads((args.training_fit/'fit.json').read_text())
    prediction=json.loads((args.saved_predictions/'transfer_predictions.json').read_text())
    assert prediction['fit_sha256']==sha(args.training_fit/'fit.json') and not prediction['parameter_draws']
    arrays=np.load(args.saved_predictions/'transfer_predictions.npz',allow_pickle=False);lookup={d['name']:d for d in records};changes=[]
    for j,name in enumerate(prediction['test_names']):
        scenario=checked_scenarios(lookup[name],np.array(fit['best']['point'][:6]),fit['parameters']['K_common_nominal'],fit['band'],config['nuisance_scenario_grid'])
        errors={key:float(np.max(abs(scenario[key]-arrays[saved+'_'+str(j)]))) for key,saved in
            [('nominal','latent_nominal'),('lower','nuisance_mean_low'),('upper','nuisance_mean_high')]}
        assert max(errors.values())<1e-10
        changes.append({'name':name,'canonical_arithmetic_errors_C':errors})
    write(args.output/'check.json',{'status':'passed','reference_fit_sha256':sha(args.reference/'fit.json'),
        'reference_fit_audit_sha256':sha(args.reference_audit),'training_fit_sha256':sha(args.training_fit/'fit.json'),
        'saved_prediction_sha256':sha(args.saved_predictions/'transfer_predictions.json'),
        'algebra':algebra,'retrospective_reference_diagnostic':diagnostic,'unchanged_canonical_forecasts':changes,
        'notice':'Bounds use the declared original cohort size, training size and band, never fitted test outputs. This stress check leaves training constraints and canonical forecasts unchanged; it is not a population law or calibrated predictive interval.'})


if __name__=='__main__':main()
