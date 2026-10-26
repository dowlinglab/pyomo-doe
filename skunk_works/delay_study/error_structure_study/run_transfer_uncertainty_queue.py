"""Audit local fold uncertainty and propagate the declared 64/256-draw checks.

Every stage executes committed frozen code in a real sequential child process.
An information/approximation failure stops the queue for scientific review; it
does not turn into a narrower covariance or an omitted uncertainty component.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from stage_execution import run_stage
from run_bounded import write

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--fit-audit',type=Path,required=True)
    parser.add_argument('--folds',nargs='+',required=True)
    args=parser.parse_args();begun=time.monotonic()
    collection=json.loads((args.input/'transfer_fits.json').read_text())
    audit=json.loads((args.fit_audit/'audit.json').read_text())
    assert json.loads((args.input/'exit.json').read_text())['exit_code']==0
    assert json.loads((args.fit_audit/'exit.json').read_text())['exit_code']==0
    assert audit['status']=='passed' and audit['kind']=='transfer'
    assert audit['source_sha256']==sha(args.input/'transfer_fits.json')
    config=json.loads((HERE/'transfer_config.json').read_text());mix=config['parameter_mixture_sensitivity']
    assert mix['draws']==256 and mix['coarse_draws']==64
    folds={r['id']:(j,r) for j,r in enumerate(collection['folds'])}
    assert len(set(args.folds))==len(args.folds) and set(args.folds)<=set(folds)
    design={'candidate':collection['candidate'],'fold_ids':args.folds,'mixture':mix,
        'fit_collection_sha256':sha(args.input/'transfer_fits.json'),
        'fit_collection_audit_sha256':sha(args.fit_audit/'audit.json'),
        'failure_policy':'Stop at the first numerical or approximation failure; preserve all prior stages and failed logs.'}
    write(args.output/'design.json',design);rows=[]
    for name in args.folds:
        index,fold=folds[name];assert fold['success']
        fit=Path(fold['selected_output']);check=args.fit_audit/('point_'+str(index))/'audit.json'
        result=json.loads(check.read_text());assert result['status']=='passed' and result['fit_sha256']==sha(fit/'fit.json')
        root=args.output/name;row={'fold_id':name,'candidate':collection['candidate'],'fit':str(fit),'fit_audit':str(check)}
        def stage(driver,label,arguments):
            write(args.output/'current_stage.json',{'fold_id':name,'stage':label,'started_elapsed_s':time.monotonic()-begun})
            return run_stage(args.output,driver,root/label,arguments)
        try:
            if collection['candidate']=='primary' and fold['type']=='leave_protocol_out' and config['reference_transport_sensitivity']['enabled']:
                prediction=stage('run_transfer_predict.py','reference_transport',['--input',fit,'--reference-transport'])
                checked=stage('audit_transfer_prediction.py','reference_transport_audit',['--input',prediction,'--fit',fit,'--fit-audit',check])
                row['reference_transport']={'directory':str(prediction),'audit':str(checked/'audit.json')}
            information=stage('run_information.py','information',['--input',fit])
            checked=stage('audit_information.py','information_audit',['--input',information,'--fit',fit])
            draws=stage('draw_local.py','draws',['--input',fit,'--information',information,'--audit',checked,
                '--draws',mix['draws'],'--seed',mix['seed']])
            row.update({'information':str(information),'information_audit':str(checked),'draws':str(draws)})
            for count in [mix['coarse_draws'],mix['draws']]:
                label='parameter_'+str(count)
                prediction=stage('run_transfer_predict.py',label,['--input',fit,'--parameter-draws',draws/('parameter_draws_'+str(count)+'.npz')])
                checked=stage('audit_transfer_prediction.py',label+'_audit',['--input',prediction,'--fit',fit,'--fit-audit',check])
                row[label]={'directory':str(prediction),'audit':str(checked/'audit.json')}
            row['status']='passed'
        except Exception as exc:
            row.update({'status':'failed','exception':repr(exc)})
            rows.append(row);write(args.output/'folds.json',rows)
            raise
        rows.append(row);write(args.output/'folds.json',rows)
        write(args.output/'progress.json',{'completed_folds':len(rows),'declared':len(args.folds),'elapsed_s':time.monotonic()-begun})
    write(args.output/'summary.json',{'status':'complete','folds':rows,'elapsed_s':time.monotonic()-begun,
        'notice':'Conditional local sensitivities, not calibrated total predictive intervals or a validation of the fitted model family. Every component retains its own numerical audit and real process exit.'})


if __name__=='__main__':main()
