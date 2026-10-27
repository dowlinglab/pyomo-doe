"""Verify seed provenance and exact retention of all nonloading inputs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from refine_transfer_loading import loading_seed
from run_bounded import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    args=p.parse_args();path=args.input/'fit.json';fit=json.loads(path.read_text());before=deepcopy(fit)
    digest=hashlib.sha256(path.read_bytes()).hexdigest();values=[.03,.3,3.,30.];cases=[]
    config=json.loads((Path(__file__).resolve().parent/'transfer_config.json').read_text())
    assert config['weak_curvature_refinement']['loading_seed_values']==values
    for value in values:
        warm=loading_seed(fit,value,digest)
        original=np.array(fit['best']['point']);seed=np.array(warm['best']['point'])
        assert np.array_equal(np.delete(seed,2),np.delete(original,2)) and seed[2]==np.log(value)
        assert warm['record_names']==fit['record_names'] and warm['noise_parameters']==fit['noise_parameters']
        assert warm['coefficients']==fit['coefficients'] and warm['parameters']['log_deviations']==fit['parameters']['log_deviations']
        assert warm['status']=='optimization_seed_only' and set(warm['best'])=={'point'}
        warm['coefficients'][0]+=1;warm['best']['point'][0]+=1
        assert fit==before,'Seed construction mutated the original input'
        cases.append({'loading':value,'all_other_point_coordinates_unchanged':True,'mean_noise_record_inputs_unchanged':True,'no_objective_or_success_claim':True})
    rejected=[]
    for value in [.0001,100.]:
        try:loading_seed(fit,value,digest)
        except AssertionError:rejected.append(value)
        else:raise AssertionError('Out-of-bound loading seed accepted')
    write(args.output/'check.json',{'status':'passed','input_fit_sha256':digest,'cases':cases,
        'out_of_bounds_rejected':rejected,'scope':'Seed-only provenance/mutation/bounds control; not an objective, optimizer or covariance test.'})


if __name__=='__main__':main()
