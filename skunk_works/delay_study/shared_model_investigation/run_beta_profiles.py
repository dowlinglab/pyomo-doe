"""Sequential fixed-beta fits and selected-fit independent ODE audits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
from run_mean import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    fit=json.loads((args.input/'fit.json').read_text());binding=json.loads((args.output/'source_snapshot.json').read_text())
    state={'status':'running','tasks':[]};write_json(args.output/'matrix_manifest.json',state)
    points=[]
    for value in (0.,.001,.0015,.002,.003,.0035,.004):
        name='beta_'+str(value).replace('.','p')
        for audit in (False,True):
            folder=args.output/(name+'_audit' if audit else name);folder.mkdir(exist_ok=False)
            if audit:
                command=[sys.executable,str(HERE/'audit_nonlinear.py'),'--input',str(args.output/name),'--output',str(folder)]
            else:
                command=[sys.executable,str(HERE/'run_nonlinear.py'),'--reference',str(args.reference),'--warm-start',str(args.input/'fit.json'),'--starts','3','--fixed-beta',str(value),'--max-step',str(fit['max_step_s']),'--output',str(folder)]
            source=dict(binding);source['command']=command;write_json(folder/'source_snapshot.json',source)
            row={'id':folder.name,'status':'running','command':command};state['tasks'].append(row)
            with (folder/'run.log').open('w') as log:
                child=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
                row['pid']=child.pid;write_json(folder/'process.json',{'pid':child.pid,'parent_pid':os.getpid(),'command':command});write_json(args.output/'matrix_manifest.json',state)
                code=child.wait()
            write_json(folder/'exit.json',{'exit_code':code});row.update({'status':'complete' if code==0 else 'failed','exit_code':code});write_json(args.output/'matrix_manifest.json',state)
            if code:
                state['status']='failed';write_json(args.output/'matrix_manifest.json',state)
                print((folder/'run.log').read_text()[-4000:],flush=True);raise SystemExit(code)
        selected=json.loads((args.output/name/'fit.json').read_text());checked=json.loads((args.output/(name+'_audit')/'audit.json').read_text())
        points.append({'beta_per_C':value,'nll':selected['best']['nll'],'twice_delta_nll':2*(selected['best']['nll']-fit['best']['nll']),
                       'stationarity_warning':checked['stationarity_warning'],'audited_gradient':checked['stable_boundary_projected_gradient_max_abs'],
                       'independent_nll_error':checked['nll_abs_error']})
        write_json(args.output/'profile.json',{'status':'running','reference_beta_per_C':fit['parameters']['beta_per_C'],'reference_nll':fit['best']['nll'],'points':points})
    state['status']='complete';write_json(args.output/'matrix_manifest.json',state)
    write_json(args.output/'profile.json',{'status':'complete','reference_beta_per_C':fit['parameters']['beta_per_C'],'reference_nll':fit['best']['nll'],'points':points,
                'notice':'Scoped grid profile with three declared starts including swapped thermal role; all nuisance/noise parameters refitted. Nominal likelihood support remains conditional on the working mean/covariance and bounds.'})


if __name__=='__main__':main()
