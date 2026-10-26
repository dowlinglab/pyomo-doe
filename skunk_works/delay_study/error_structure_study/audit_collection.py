"""Apply full selected-point audits to every accepted profile or fold fit.

Failed optimization tasks remain in the denominator; an audit never retries
optimization or promotes a failed task. Each numerical audit is persisted before
the next task so interruption leaves usable, source-bound work.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from audit_joint_quantized import audit
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--kind',choices=['profile','transfer'],required=True)
    args=parser.parse_args();begun=time.monotonic()
    source=args.input/('profile.json' if args.kind=='profile' else 'transfer_fits.json')
    content=json.loads(source.read_text());items=content['points' if args.kind=='profile' else 'folds']
    rows=[]
    for j,item in enumerate(items):
        row={'index':j,'optimization_success':item['success']}
        if args.kind=='profile':row.update({k:item[k] for k in ['coordinate_value','physical_value']})
        else:row['fold_id']=item['id']
        if item['success']:
            directory=Path(item['output' if args.kind=='profile' else 'selected_output'])
            try:
                result=audit(SimpleNamespace(input=directory))
                write(args.output/('point_'+str(j))/'audit.json',result)
                row.update({'audit_status':result['status'],'fit':str(directory),
                    'fit_sha256':result['fit_sha256'],'failures':result['failures']})
            except Exception as exc:row.update({'audit_status':'failed','exception':repr(exc),'fit':str(directory)})
        else:row['audit_status']='not_audited_failed_optimization'
        rows.append(row);write(args.output/'points.json',rows)
        write(args.output/'progress.json',{'attempted':len(rows),'declared':len(items),
            'passed':sum(r['audit_status']=='passed' for r in rows),'elapsed_s':time.monotonic()-begun})
    result={'status':'passed' if all(r['audit_status']=='passed' for r in rows) else 'completed_with_failures',
        'kind':args.kind,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'rows':rows,'elapsed_s':time.monotonic()-begun,
        'scope':'Every successful point receives independent dimensional mean, rectangle probabilities, 128/256/512-node and tail/stationarity audits. Solver failures are retained, not retried.'}
    write(args.output/'audit.json',result)
    if any(r['audit_status']=='failed' for r in rows):raise RuntimeError('Selected-point audit failed; see saved rows')


if __name__=='__main__':main()
