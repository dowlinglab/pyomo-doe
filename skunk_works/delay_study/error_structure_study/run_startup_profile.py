"""Representative initial-heater profile after a consequential lag-label change."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from joint_quantized import Model
from continuation import parent_receipt
from fit_task import run
from run_bounded import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection(fit,policy):
    names=fit['coordinate_names'];point=np.array(fit['best']['point'])
    indices=[j for j,name in enumerate(names) if name.startswith('initial_H')]
    assert len(indices)==2*len(fit['record_names']) and fit['startup_band_C']==2.
    index=max(indices,key=lambda j:abs(point[j]))
    values=list(policy['values_C'])+([float(point[index])] if policy['include_selected_value'] else [])
    values=sorted(set(values),key=lambda v:(abs(v-point[index]),v))
    return index,names[index],values


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--fit-audit',type=Path,required=True)
    p.add_argument('--noise1',type=Path,required=True);p.add_argument('--noise2',type=Path,required=True)
    args=p.parse_args();begun=time.monotonic();fit=json.loads((args.input/'fit.json').read_text())
    assert fit['family']=='conductance' and fit['kernel']=='quantized_ou_nugget' and fit['startup_band_C']==2.
    assert not any(fit.get(k) for k in ['dataset','fixed_coordinates','initial_sensor_bins','condition_initial_code'])
    assert parent_receipt(args.input,require_terminal=True)['exit']['exit_code']==0
    assert parent_receipt(args.fit_audit,require_terminal=True)['exit']['exit_code']==0
    audit=json.loads((args.fit_audit/'audit.json').read_text())
    assert audit['status']=='passed' and audit['fit_sha256']==sha(args.input/'fit.json')
    records=load_records();assert fit['record_names']==[d['name'] for d in records]
    assert all(fit['raw_sha256'][d['name']]==d['sha256'] for d in records)
    policy=json.loads((HERE/'uncertainty_config.json').read_text())['consequential_startup_profile']
    index,name,ordered=selection(fit,policy)
    model=Model(records,fit['band'],fit['phase'],policy['nodes'],'conductance',startup_band=2.,
        minimum_sensor_loading=fit.get('minimum_sensor_loading',.005),score_backend='compiled')
    assert model.coordinate_names()==fit['coordinate_names'];lo,hi=model.bounds()[index]
    assert all(lo<=v<=hi for v in ordered)
    write(args.output/'design.json',{'policy':policy,'coordinate':name,'coordinate_index':index,
        'selected_value_C':fit['best']['point'][index],'ordered_values_C':ordered,
        'fit_sha256':sha(args.input/'fit.json'),'fit_audit_sha256':sha(args.fit_audit/'audit.json'),
        'score_backend':'compiled','scope':policy['scope']})
    rows=[];successful=[]
    for j,value in enumerate(ordered):
        warm=args.input/'fit.json'
        if successful:warm=Path(min(successful,key=lambda r:abs(r['coordinate_value']-value))['output'])/'fit.json'
        row=run(args.output/('point_'+str(j)),warm,noise1=args.noise1,noise2=args.noise2,
            family='conductance',band=fit['band'],phase=fit['phase'],nodes=policy['nodes'],
            starts=policy['starts'],maxiter=policy['maxiter'],fixed={name:value},startup_band=2.,
            minimum_sensor_loading=fit.get('minimum_sensor_loading',.005),score_backend='compiled')
        row.update({'coordinate_value':value,'physical_value':value})
        if row['success']:
            row['twice_nll_minus_reference']=2*(row['nll']-fit['best']['nll']);successful.append(row)
        rows.append(row);write(args.output/'points.json',rows)
        write(args.output/'progress.json',{'points_attempted':len(rows),'points_successful':len(successful),
            'declared':len(ordered),'elapsed_s':time.monotonic()-begun})
    write(args.output/'profile.json',{'status':'complete' if all(r['success'] for r in rows) else 'completed_with_failures',
        'coordinate':name,'optimization_coordinate_is_log':False,'reference_nll':fit['best']['nll'],
        'reference_coordinate':fit['best']['point'][index],'profile_bounds':[lo,hi],
        'fit_sha256':sha(args.input/'fit.json'),'points':sorted(rows,key=lambda r:r['coordinate_value']),
        'lower_nll_than_reference':any(r['success'] and r['nll']<fit['best']['nll']-1e-3 for r in rows),
        'elapsed_s':time.monotonic()-begun,'notice':policy['scope']+
            ' All selected points require independent audits. A lower point requires an unconstrained startup-baseline refinement before interpreting deviances. Two starts do not prove global profiles.'})
    if not successful:raise RuntimeError('No startup profile point converged; retain failures')


if __name__=='__main__':main()
