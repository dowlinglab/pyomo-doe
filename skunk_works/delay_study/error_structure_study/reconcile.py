"""Reconcile saved jobs against a separately obtained direct-tool process list.

The macOS sandbox can block process enumeration in Python subprocesses. This
controller never enumerates processes itself and never launches/restarts jobs.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--process-snapshot',type=Path,required=True);args=parser.parse_args()
    snapshot=json.loads(args.process_snapshot.read_text());assert snapshot['exit_code'] in [0,1]
    checked=datetime.strptime(snapshot['checked_utc'],'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    now=datetime.now(timezone.utc);age=(now-checked).total_seconds()
    if not -5<=age<=180:raise ValueError('Obtain a fresh direct-tool process snapshot first: age='+str(age))
    manifest=json.loads(args.manifest.read_text());old={r['id']:r for r in manifest['jobs']}
    processes={r['pid']:r['command'] for r in snapshot['processes']};jobs=[];needs=[];live=[];bound_pids=[]
    for directory in sorted((args.manifest.parent/'results').iterdir()):
        if not (directory/'process.json').exists():continue
        process=json.loads((directory/'process.json').read_text());source=json.loads((directory/'source_snapshot.json').read_text())
        row=dict(old.get(directory.name,{}));row.update({'id':directory.name,'process':process,
            'output':str(directory.relative_to(args.manifest.parents[3])),
            'source_commit':source['commit'],
            'source_receipt_sha256':hashlib.sha256((directory/'source_snapshot.json').read_bytes()).hexdigest()})
        if (directory/'exit.json').exists():
            exit=json.loads((directory/'exit.json').read_text());row['status']='terminal';row['exit_code']=exit['exit_code']
            row['finished_utc']=exit['finished_utc']
        elif processes.get(process['pid'])==' '.join(process['command']):
            row['status']='running_verified';live.append(directory.name);bound_pids.append(process['pid'])
            row['last_verified_utc']=snapshot['checked_utc']
        else:
            row['status']='needs_reconciliation';needs.append(directory.name)
        if (directory/'starts.json').exists():row['starts_recorded']=len(json.loads((directory/'starts.json').read_text()))
        jobs.append(row)
    manifest.update({'jobs':jobs,'live_jobs':live,'updated_utc':now.isoformat(),
        'process_check':{'path':str(args.process_snapshot.relative_to(args.manifest.parent)),
            'checked_utc':snapshot['checked_utc'],'sha256':hashlib.sha256(args.process_snapshot.read_bytes()).hexdigest(),
            'method':snapshot['method'],'snapshot_age_s':age},
        'reconciliation_warnings':{'missing_exit_or_matching_process':needs,
            'unbound_process_pids':[p for p in processes if p not in bound_pids]}})
    args.manifest.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'live_jobs':live,'terminal_jobs':sum(r['status']=='terminal' for r in jobs),
        'warnings':manifest['reconciliation_warnings']}))


if __name__=='__main__':main()
