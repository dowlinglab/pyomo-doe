"""Refit declared retrospective folds without estimating test-run nuisance values."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
from physical import load_records
from fit_task import run
from run_bounded import write


def folds(config,candidate):
    result=[];kinds=config['candidates'][candidate]['fold_types']
    if 'leave_run_out' in kinds:
        result += [{'id':'run_'+str(j+1),'test_indices':[j],'type':'leave_run_out'} for j in config['leave_run_out']['run_indices']]
    if 'leave_protocol_out' in kinds:
        result += [dict(r,type='leave_protocol_out') for r in config['leave_protocol_out']]
    for r in result:r['train_indices']=[j for j in range(12) if j not in r['test_indices']]
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--noise1',type=Path,required=True);parser.add_argument('--noise2',type=Path,required=True)
    parser.add_argument('--candidate',choices=['primary','shared'],required=True);parser.add_argument('--folds',nargs='+')
    parser.add_argument('--score-backend',choices=['reference','compiled'],default='reference')
    args=parser.parse_args();begun=time.monotonic();config=json.loads((HERE/'transfer_config.json').read_text())
    fit=json.loads((args.input/'fit.json').read_text());candidate=config['candidates'][args.candidate]
    assert fit['kernel']=='quantized_ou_nugget' and fit['family']==candidate['family'] and fit['band']==candidate['band']
    assert not fit.get('fixed_coordinates') and not fit.get('dataset') and not fit.get('startup_band_C') and not fit.get('initial_sensor_bins')
    if fit.get('condition_initial_code'):raise ValueError('Declare and propagate conditional-first-code transfer conventions before this sweep')
    records=load_records();assert fit['record_names']==[d['name'] for d in records]
    chosen=folds(config,args.candidate)
    if args.folds:
        assert len(set(args.folds))==len(args.folds) and set(args.folds)<=set(r['id'] for r in chosen)
        chosen=[r for r in chosen if r['id'] in args.folds]
    rows=[]
    for fold in chosen:
        common={'noise1':args.noise1,'noise2':args.noise2,'family':fit['family'],'band':fit['band'],
                'phase':fit['phase'],'maxiter':config['maxiter'],'records':fold['train_indices'],
                'minimum_sensor_loading':fit.get('minimum_sensor_loading',.005),'score_backend':args.score_backend}
        attempt=run(args.output/fold['id'],args.input/'fit.json',starts=config['starts'],**common);attempts=[attempt]
        if not attempt['success']:
            attempt=run(args.output/(fold['id']+'_retry'),args.input/'fit.json',starts=config['failed_retry_starts'],**common);attempts.append(attempt)
        row=dict(fold,success=attempt['success'],attempts=attempts,selected_output=attempt['output'] if attempt['success'] else None,
            train_names=[records[j]['name'] for j in fold['train_indices']],test_names=[records[j]['name'] for j in fold['test_indices']])
        rows.append(row);write(args.output/'folds.json',rows)
        write(args.output/'progress.json',{'attempted':len(rows),'declared':len(chosen),'successful':sum(r['success'] for r in rows),'elapsed_s':time.monotonic()-begun})
    write(args.output/'transfer_fits.json',{'status':'complete' if all(r['success'] for r in rows) else 'completed_with_failures',
        'candidate':args.candidate,'reference_fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
        'config':config,'folds':rows,'elapsed_s':time.monotonic()-begun,
        'notice':'Internal leave-run/protocol-out refits; all records informed structure selection. Full-data parameters are warm starts only. Test future outputs do not enter the training objective, covariance, nuisance fitting or centering. Numerical fit audits and parameter-inclusive prediction analysis remain separate.'})


if __name__=='__main__':main()
