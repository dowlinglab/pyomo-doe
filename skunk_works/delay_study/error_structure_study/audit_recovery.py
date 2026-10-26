"""Audit disjoint refitted recovery shards and retain all failure denominators.

An explicit partial mode inspects an immutable completed prefix; it cannot
produce the final fifty-draw status. Optional prior audits are reusable only
with identical numerical auditor source and matching fit/array/data hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.stats import beta
from physical import load_records
from bin_diagnostics import sample_observations
from audit_joint_quantized import audit
from run_bounded import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(k,n):
    if not n:return None
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shards',type=Path,nargs='+',required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--reuse-audits',type=Path,nargs='*',default=[])
    args=parser.parse_args();begun=time.monotonic()
    config=json.loads((HERE/'uncertainty_config.json').read_text())['recovery']
    fit=json.loads((args.input/'fit.json').read_text());lookup={d['name']:d for d in load_records()}
    records=[lookup[n] for n in fit['record_names']]
    assert not fit.get('condition_initial_code') and not fit.get('dataset')
    arrays=np.load(args.input/'trajectories.npz');means=[arrays['predicted_'+str(j)] for j in range(len(records))]
    truth=np.array(fit['best']['point']);fit_hash=sha(args.input/'fit.json')
    rows=[];bindings=[];declared=[]
    for directory in args.shards:
        assert (directory/'exit.json').exists(),'Audit an immutable terminal shard only'
        design=json.loads((directory/'design.json').read_text())
        assert design['config']==config and design['fit_sha256']==fit_hash and np.array_equal(design['truth_point'],truth)
        declared+=design['indices']
        items=json.loads((directory/'draws.json').read_text())
        assert len(set(r['draw'] for r in items))==len(items)
        assert set(r['draw'] for r in items)<=set(design['indices'])
        rows+=items
        bindings.append({'path':str(directory),'draws_sha256':sha(directory/'draws.json'),
            'design_sha256':sha(directory/'design.json'),'source_sha256':sha(directory/'source_snapshot.json'),
            'exit':json.loads((directory/'exit.json').read_text())})
    assert len(set(declared))==len(declared),'Overlapping declared shard indices'
    assert sorted(declared)==list(range(config['draws'])),'Shards must declare the complete design'
    indices=[r['draw'] for r in rows];assert len(set(indices))==len(indices)
    if not args.allow_partial:assert sorted(indices)==list(range(config['draws'])),'Incomplete fifty-draw recovery'
    auditor_sha=sha(HERE/'audit_joint_quantized.py');cached={}
    for directory in args.reuse_audits:
        previous=json.loads((directory/'audit.json').read_text())
        assert previous['numerical_auditor_sha256']==auditor_sha
        for row in previous['rows']:
            for task in row.get('numerical_audits',[]):
                if task.get('status')=='passed':cached[task['fit_sha256']]=(directory,task)
    audited=[]
    for row in sorted(rows,key=lambda r:r['draw']):
        result={'draw':row['draw'],'seed':row['seed'],'coverage_complete':row['coverage_complete'],
            'free_fit_success':row['free_fit_success'],'numerical_audits':[],'failures':[]}
        assert row['seed']==config['seed_base']+config['seed_stride']*row['draw']
        # Locate the original generated data even for copied terminal draws.
        origin=Path(row['reused_completed_draw']['path']).parent if row.get('reused_completed_draw') else None
        if origin is None:
            candidates=[p/('draw_'+str(row['draw'])) for p in args.shards if (p/('draw_'+str(row['draw']))/'observations.npz').exists()]
            assert len(candidates)==1;origin=candidates[0]
        dataset=origin/'observations.npz'
        try:
            assert sha(dataset)==row['dataset_sha256']
            saved=np.load(dataset,allow_pickle=False)
            assert saved['record_names'].tolist()==fit['record_names']
            generated=sample_observations(records,means,fit['noise_parameters'],fit['phase'],row['seed'])
            for j,d in enumerate(generated):
                assert np.array_equal(saved['time_'+str(j)],d['t']) and np.array_equal(saved['input_'+str(j)],d['u'])
                assert np.array_equal(saved['observed_'+str(j)],d['y']),'Seed/data regeneration mismatch'
            result['dataset_sha256']=sha(dataset)
            labels=['selected_free_output','selected_null_output'] if row['coverage_complete'] else []
            selected={}
            for label in labels:
                directory=Path(row[label]);selected[label]=json.loads((directory/'fit.json').read_text())
                assert selected[label]['dataset']['sha256']==sha(dataset)
                digest=sha(directory/'fit.json');reuse=cached.get(digest)
                if reuse:
                    parent,task=reuse
                    assert task['arrays_sha256']==sha(directory/'trajectories.npz') and task['dataset_binding']['sha256']==sha(dataset)
                    numerical=dict(task,reused_audit={'path':str(parent),'audit_sha256':sha(parent/'audit.json')})
                else:
                    numerical=audit(SimpleNamespace(input=directory))
                    write(args.output/('draw_'+str(row['draw']))/(label+'.json'),numerical)
                result['numerical_audits'].append(dict(numerical,role=label))
                if numerical['status']!='passed':result['failures'].append([label,numerical['failures']])
            if row['coverage_complete']:
                free=selected['selected_free_output'];null=selected['selected_null_output']
                names=json.loads((origin.parent/'design.json').read_text())['coordinate_names']
                index=names.index('log_K')
                assert abs(null['fixed_coordinates']['log_K']-truth[index])<1e-14
                assert not free.get('fixed_coordinates')
                lr=2*(null['best']['nll']-free['best']['nll'])
                assert abs(lr-row['raw_gain_LR'])<1e-8 and lr>=-2e-3
                assert bool(max(lr,0.)<=config['nominal_gain_LR_cutoff'])==row['gain_LR_region_covers_truth']
                result.update({'gain_LR':max(lr,0.),'covered':row['gain_LR_region_covers_truth'],
                    'recovered_point':free['best']['point'],'recovered_parameters':free['parameters'],
                    'recovered_noise':free['noise_parameters']})
            result['status']='passed' if not result['failures'] else 'failed'
        except Exception as exc:result.update({'status':'failed','exception':repr(exc)})
        audited.append(result);write(args.output/'points.json',audited)
        write(args.output/'progress.json',{'attempted':len(audited),'available':len(rows),'declared':config['draws'],
            'passed':sum(r['status']=='passed' for r in audited),'elapsed_s':time.monotonic()-begun})
    complete=[r for r in audited if r['status']=='passed' and r['coverage_complete']]
    n=len(complete);k=sum(r['covered'] for r in complete);failed=len(rows)-n
    final={'status':'partial' if len(rows)<config['draws'] else ('passed' if all(r['status']=='passed' for r in audited) else 'failed'),
        'fit_sha256':fit_hash,'numerical_auditor_sha256':auditor_sha,'shards':bindings,'config':config,
        'available_draws':len(rows),'declared_draws':config['draws'],'missing_draws':sorted(set(range(config['draws']))-set(indices)),
        'rows':audited,'complete_audited_coverage_draws':n,'failed_or_unaudited_available_draws':failed,
        'covered_audited_draws':k,'coverage_among_audited_complete':k/n if n else None,
        'binomial_95_Monte_Carlo_interval':interval(k,n),
        'failure_worst_case_bounds_among_available':[k/len(rows),(k+failed)/len(rows)] if rows else None,
        'elapsed_s':time.monotonic()-begun,
        'notice':'Conditional parametric recovery with all failures retained; not a real-data model-adequacy test. Partial output is not a fifty-draw coverage result. Each accepted LR pair is independently audited and observations are regenerated from the declared seed.'}
    write(args.output/'audit.json',final)
    if any(r['status']=='failed' for r in audited):raise RuntimeError('Recovery numerical/data audit failed; see retained rows')


if __name__=='__main__':main()
