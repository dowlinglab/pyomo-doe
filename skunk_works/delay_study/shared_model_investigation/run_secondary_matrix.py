"""Sequential fixed-weight/IID chain sensitivity and independent fit audits.

The outer freeze receipt binds every child to the same committed source. The
matrix occupies one numerical slot, and never retries over an existing child.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--starts',type=int,default=8)
    args=parser.parse_args()
    source=json.loads((args.output/'source_snapshot.json').read_text())
    tasks=[]
    for policy in ('fixed_ou','iid'):
        for order in (1,2,3):
            tasks.append({'id':'chain'+str(order)+'_'+policy,'driver':'run_pooled.py',
                          'args':['--kind','chain','--order',str(order),'--starts',str(args.starts),'--objective',policy]})
    for task in list(tasks):
        tasks.append({'id':task['id']+'_audit','driver':'audit_pooled.py',
                      'args':['--input',str(args.output/task['id'])]})
    state={'status':'running','tasks':[],'max_child_numerical_processes':1}
    write(args.output/'matrix_manifest.json',state)
    for task in tasks:
        folder=args.output/task['id'];folder.mkdir(exist_ok=False)
        command=[sys.executable,str(HERE/task['driver']),'--output',str(folder)]+task['args']
        binding=dict(source);binding['command']=command
        write(folder/'source_snapshot.json',binding)
        row={'id':task['id'],'status':'running','command':command,'started_unix':time.time()}
        state['tasks'].append(row)
        with (folder/'run.log').open('w') as log:
            child=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=child.pid
            write(folder/'process.json',{'pid':child.pid,'parent_pid':os.getpid(),'command':command})
            write(args.output/'matrix_manifest.json',state)
            code=child.wait()
        write(folder/'exit.json',{'exit_code':code})
        row.update({'status':'complete' if code==0 else 'failed','exit_code':code})
        write(args.output/'matrix_manifest.json',state)
        if code:
            state['status']='failed';write(args.output/'matrix_manifest.json',state)
            print((folder/'run.log').read_text()[-4000:],flush=True)
            raise SystemExit(code)
    state['status']='complete';write(args.output/'matrix_manifest.json',state)
    print(json.dumps({'status':'complete','tasks':len(tasks)}),flush=True)


if __name__=='__main__':main()
