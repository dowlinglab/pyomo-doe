"""Run a declared subset of remaining profiles and audit each before advancing.

This is one numerical process, using the parent frozen source for all fits and
audits. A better-than-reference point stops the queue for baseline review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import run_profiles
import audit_collection
from run_bounded import write
HERE=Path(__file__).resolve().parent


def invoke(main,argv):
    previous=sys.argv
    try:sys.argv=argv;main()
    finally:sys.argv=previous


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--radiation-input',type=Path,required=True)
    parser.add_argument('--noise1',type=Path,required=True);parser.add_argument('--noise2',type=Path,required=True)
    parser.add_argument('--coordinates',nargs='+',required=True)
    args=parser.parse_args();begun=time.monotonic()
    config=json.loads((HERE/'uncertainty_config.json').read_text())
    lookup={r['coordinate']:r for r in config['profiles']}
    assert len(set(args.coordinates))==len(args.coordinates) and set(args.coordinates)<=set(lookup)
    assert 'log_K' not in args.coordinates,'Reuse the original completed gain profile'
    schedule=[lookup[n] for n in args.coordinates]
    write(args.output/'design.json',{'profiles':schedule,'starts':config['profile_starts'],
        'maxiter':config['profile_maxiter'],'score_backend':'compiled',
        'reference_fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
        'radiation_fit_sha256':hashlib.sha256((args.radiation_input/'fit.json').read_bytes()).hexdigest()})
    rows=[]
    for entry in schedule:
        name=entry['coordinate'];directory=args.output/name;directory.mkdir()
        selected=args.radiation_input if entry.get('family')=='radiation' else args.input
        argv=['run_profiles.py','--output',str(directory),'--input',str(selected),
            '--noise1',str(args.noise1),'--noise2',str(args.noise2),'--coordinate',name,
            '--values']+[str(v) for v in entry['values']]
        argv+=['--starts',str(config['profile_starts']),'--maxiter',str(config['profile_maxiter']),'--score-backend','compiled']
        if entry.get('physical_values'):argv+=['--physical-values']
        row={'coordinate':name,'output':str(directory)}
        try:
            invoke(run_profiles.main,argv)
            profile=json.loads((directory/'profile.json').read_text())
            row['profile_status']=profile['status']
            if profile['lower_nll_than_reference']:raise RuntimeError('Profile found a better baseline; refine before continuing')
            audit=args.output/(name+'_audit');audit.mkdir()
            invoke(audit_collection.main,['audit_collection.py','--output',str(audit),'--input',str(directory),'--kind','profile'])
            row['audit_status']=json.loads((audit/'audit.json').read_text())['status']
            row['status']='passed' if row['audit_status']=='passed' and profile['status']=='complete' else 'completed_with_failures'
        except Exception as exc:
            row.update({'status':'failed','exception':repr(exc),'traceback':traceback.format_exc()})
            rows.append(row);write(args.output/'profiles.json',rows)
            raise
        rows.append(row);write(args.output/'profiles.json',rows)
        write(args.output/'progress.json',{'attempted':len(rows),'declared':len(schedule),'elapsed_s':time.monotonic()-begun})
    write(args.output/'summary.json',{'status':'complete' if all(r['status']=='passed' for r in rows) else 'completed_with_failures',
        'profiles':rows,'elapsed_s':time.monotonic()-begun,
        'notice':'Prespecified grids and nuisance refits; no routine chi-square confidence claims at boundaries or disconnected branches. Independent point audits precede each next profile.'})


if __name__=='__main__':main()
