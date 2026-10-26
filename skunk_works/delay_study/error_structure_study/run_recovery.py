"""Refitted quantized-model recovery and gain LR coverage, with failure counts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.stats import beta
from physical import load_records
from joint_quantized import Model
from bin_diagnostics import sample_observations,filter_records
from nonrandom import summaries as deterministic_summaries
from fit_task import run
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--noise1',type=Path,required=True);parser.add_argument('--noise2',type=Path,required=True)
    parser.add_argument('--first',type=int,default=0);parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--purpose',choices=['coverage','pilot'],default='coverage');parser.add_argument('--limit',type=int)
    parser.add_argument('--score-backend',choices=['reference','compiled'],default='reference')
    args=parser.parse_args();begun=time.monotonic();config=json.loads((HERE/'uncertainty_config.json').read_text())['recovery']
    if args.stride<1 or not 0<=args.first<config['draws']:raise ValueError('Invalid shard')
    if args.limit is not None and args.purpose!='pilot':raise ValueError('A coverage shard must finish its declared draw indices')
    fit=json.loads((args.input/'fit.json').read_text());assert fit['family']=='conductance' and fit['kernel']=='quantized_ou_nugget'
    assert not fit.get('fixed_coordinates') and not fit.get('dataset') and not fit.get('startup_band_C') and not fit.get('initial_sensor_bins')
    if fit.get('condition_initial_code'):raise ValueError('Conditional-first-code recovery requires its corresponding conditional latent-state sampler')
    lookup={d['name']:d for d in load_records()};records=[lookup[n] for n in fit['record_names']]
    if len(records)!=len(lookup):raise ValueError('Declare a separate reduced-design recovery study')
    arrays=np.load(args.input/'trajectories.npz');means=[arrays['predicted_'+str(j)] for j in range(len(records))]
    m=Model(records,fit['band'],fit['phase'],128,fit['family'],minimum_sensor_loading=fit.get('minimum_sensor_loading',.005));truth=np.array(fit['best']['point']);gain_index=m.coordinate_names().index('log_K')
    for j,d in enumerate(records):assert d['sha256']==fit['raw_sha256'][d['name']] and np.array_equal(d['y'],arrays['observed_'+str(j)])
    indices=list(range(args.first,config['draws'],args.stride))
    if args.limit is not None:indices=indices[:args.limit]
    write(args.output/'design.json',{'purpose':args.purpose,'indices':indices,'total_declared_draws':config['draws'],
        'fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
        'arrays_sha256':hashlib.sha256((args.input/'trajectories.npz').read_bytes()).hexdigest(),
        'truth_point':truth.tolist(),'coordinate_names':m.coordinate_names(),'config':config})
    rows=[]
    for index in indices:
        directory=args.output/('draw_'+str(index));directory.mkdir();seed=config['seed_base']+config['seed_stride']*index
        row={'draw':index,'seed':seed,'attempts':[],'free_fit_success':False,'coverage_complete':False}
        try:
            synthetic=sample_observations(records,means,fit['noise_parameters'],fit['phase'],seed);data={}
            for j,d in enumerate(synthetic):data.update({'time_'+str(j):d['t'],'input_'+str(j):d['u'],'observed_'+str(j):d['y']})
            dataset=directory/'observations.npz';np.savez_compressed(dataset,record_names=np.array([d['name'] for d in records]),**data)
            row['dataset_sha256']=hashlib.sha256(dataset.read_bytes()).hexdigest()
            common={'noise1':args.noise1,'noise2':args.noise2,'family':fit['family'],'band':fit['band'],
                    'phase':fit['phase'],'dataset':dataset,'maxiter':config['maxiter'],
                    'minimum_sensor_loading':fit.get('minimum_sensor_loading',.005),'score_backend':args.score_backend}
            free=run(directory/'free',args.input/'fit.json',starts=config['free_fit_starts'],**common);row['attempts'].append(free)
            if not free['success']:
                free=run(directory/'free_retry',args.input/'fit.json',starts=config['free_fit_failed_retry_starts'],**common);row['attempts'].append(free)
            if not free['success']:raise RuntimeError('No accepted free fit after declared retry')
            ff=json.loads((Path(free['output'])/'fit.json').read_text());row['free_fit_success']=True
            null=run(directory/'true_gain',Path(free['output'])/'fit.json',starts=config['fixed_true_gain_fit_starts'],
                     fixed={'log_K':float(truth[gain_index])},**common);row['attempts'].append(null)
            if not null['success']:
                null=run(directory/'true_gain_retry',Path(free['output'])/'fit.json',starts=config['fixed_true_gain_failed_retry_starts'],
                         fixed={'log_K':float(truth[gain_index])},**common);row['attempts'].append(null)
            if not null['success']:raise RuntimeError('No accepted true-gain constrained fit after declared retry')
            if null['nll']<free['nll']-1e-3:
                correction=run(directory/'free_basin_correction',Path(null['output'])/'fit.json',starts=2,**common);row['attempts'].append(correction)
                if correction['success'] and correction['nll']<free['nll']:
                    free=correction;ff=json.loads((Path(free['output'])/'fit.json').read_text())
            raw_lr=float(2*(null['nll']-free['nll']))
            if raw_lr < -2e-3:raise RuntimeError('Constrained fit beats the free fit after basin correction')
            row.update({'coverage_complete':True,'raw_gain_LR':raw_lr,'gain_LR':max(raw_lr,0.),
                'gain_LR_region_covers_truth':bool(max(raw_lr,0.)<=config['nominal_gain_LR_cutoff']),
                'selected_free_output':free['output'],'selected_null_output':null['output']})
        except Exception as exc:row['exception']=repr(exc)
        if row['free_fit_success']:
            row['recovered_point']=ff['best']['point'];row['recovered_parameters']=ff['parameters'];row['recovered_noise']=ff['noise_parameters']
            # A diagnostic failure must not erase a completed coverage draw.
            # Use the selected free-fit mean/noise on exactly this generated
            # record collection; these references account for refitting within
            # the fixed chosen family, but do not repeat structure selection.
            try:
                selected=Path(row.get('selected_free_output',free['output']))
                fa=np.load(selected/'trajectories.npz')
                fitted_means=[fa['predicted_'+str(j)] for j in range(len(synthetic))]
                filtered=filter_records(synthetic,fitted_means,ff['noise_parameters'],ff['phase'])
                diagnostic=deterministic_summaries(synthetic,filtered);del filtered
                write(directory/'refitted_nonrandom_diagnostics.json',diagnostic)
                row['refitted_diagnostic_metrics']=diagnostic['metrics'];row['refitted_diagnostic_success']=True
            except Exception as exc:
                row['refitted_diagnostic_success']=False;row['refitted_diagnostic_exception']=repr(exc)
        rows.append(row);write(directory/'recovery.json',row);write(args.output/'draws.json',rows)
        write(args.output/'progress.json',{'attempted':len(rows),'declared_in_shard':len(indices),
            'accepted_free_fits':sum(r['free_fit_success'] for r in rows),
            'coverage_complete':sum(r['coverage_complete'] for r in rows),'elapsed_s':time.monotonic()-begun})
        print(json.dumps({key:row.get(key) for key in ['draw','free_fit_success','coverage_complete','gain_LR_region_covers_truth','exception']}),flush=True)
    complete=[r for r in rows if r['coverage_complete']];n=len(complete);k=sum(r['gain_LR_region_covers_truth'] for r in complete);failed=len(rows)-n
    interval=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.] if n else None
    write(args.output/'recovery_summary.json',{'status':'complete' if not failed else 'completed_with_failures','purpose':args.purpose,
        'indices':indices,'attempts':len(rows),'complete_gain_coverage_draws':n,'failed_coverage_draws':failed,
        'covered_complete_draws':k,'coverage_among_complete_draws':float(k/n) if n else None,
        'binomial_95_Monte_Carlo_interval_among_complete':interval,
        'failure_worst_case_coverage_bounds':[k/len(rows),(k+failed)/len(rows)] if rows else None,
        'elapsed_s':time.monotonic()-begun,'notice':'Refitted parametric simulation conditional on the fitted generating model, bins, recorded commands/times, initial readings and hard bands. Gain coverage uses the nominal scalar LR cutoff and refits all permitted nuisance coordinates. Counts include every failed draw; conditional success-only coverage may be biased. No coverage claim for model misspecification, new device populations, unmeasured threshold phase or all individual parameters.'})


if __name__=='__main__':main()
