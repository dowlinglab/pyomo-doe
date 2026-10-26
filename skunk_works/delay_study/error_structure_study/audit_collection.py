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
    parser.add_argument('--reuse-collection',type=Path)
    args=parser.parse_args();begun=time.monotonic()
    source=args.input/('profile.json' if args.kind=='profile' else 'transfer_fits.json')
    content=json.loads(source.read_text());items=content['points' if args.kind=='profile' else 'folds']
    reused={};reuse_receipt=None
    if args.reuse_collection:
        old=args.reuse_collection
        assert (old/'exit.json').exists(),'Previous audit collection must be terminal'
        before=json.loads((old/'source_snapshot.json').read_text())['sha256']
        after=json.loads((args.output/'source_snapshot.json').read_text())['sha256']
        # Only controller/output-writing modules may differ. Every numerical
        # module, data file and configuration must match byte for byte.
        ignored={'audit_collection.py','audit_recovery.py','run_profile_queue.py'}
        a={k:v for k,v in before.items() if Path(k).name not in ignored}
        b={k:v for k,v in after.items() if Path(k).name not in ignored}
        assert a==b,'Numerical auditor dependencies, configuration or raw data changed'
        previous=json.loads((old/'audit.json').read_text())
        assert previous['kind']==args.kind and previous['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        for row in previous['rows']:
            if row.get('audit_status')=='passed':
                path=old/('point_'+str(row['index']))/'audit.json'
                result=json.loads(path.read_text());assert result['status']=='passed'
                reused[row['index']]=(path,result)
        reuse_receipt={'path':str(old),'audit_sha256':hashlib.sha256((old/'audit.json').read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256((old/'source_snapshot.json').read_bytes()).hexdigest(),
            'notice':'Only terminal passed numerical point audits with identical numerical source, raw data/configuration and fit/trajectory hashes are reused. Failed or unsaved points are recomputed.'}
    rows=[]
    for j,item in enumerate(items):
        row={'index':j,'optimization_success':item['success']}
        if args.kind=='profile':row.update({k:item[k] for k in ['coordinate_value','physical_value']})
        else:row['fold_id']=item['id']
        if item['success']:
            directory=Path(item['output' if args.kind=='profile' else 'selected_output'])
            try:
                destination=args.output/('point_'+str(j));destination.mkdir(parents=True,exist_ok=False)
                if j in reused:
                    path,result=reused[j]
                    assert result['fit_sha256']==hashlib.sha256((directory/'fit.json').read_bytes()).hexdigest()
                    assert result['arrays_sha256']==hashlib.sha256((directory/'trajectories.npz').read_bytes()).hexdigest()
                    row['reused_numerical_audit']={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                else:result=audit(SimpleNamespace(input=directory))
                write(destination/'audit.json',result)
                row.update({'audit_status':result['status'],'fit':str(directory),
                    'fit_sha256':result['fit_sha256'],'failures':result['failures']})
            except Exception as exc:row.update({'audit_status':'failed','exception':repr(exc),'fit':str(directory)})
        else:row['audit_status']='not_audited_failed_optimization'
        rows.append(row);write(args.output/'points.json',rows)
        write(args.output/'progress.json',{'attempted':len(rows),'declared':len(items),
            'passed':sum(r['audit_status']=='passed' for r in rows),'elapsed_s':time.monotonic()-begun})
    result={'status':'passed' if all(r['audit_status']=='passed' for r in rows) else 'completed_with_failures',
        'kind':args.kind,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'rows':rows,'elapsed_s':time.monotonic()-begun,'reuse_collection':reuse_receipt,
        'scope':'Every successful point receives independent dimensional mean, rectangle probabilities, 128/256/512-node and tail/stationarity audits. Solver failures are retained, not retried.'}
    write(args.output/'audit.json',result)
    if any(r['audit_status']=='failed' for r in rows):raise RuntimeError('Selected-point audit failed; see saved rows')


if __name__=='__main__':main()
