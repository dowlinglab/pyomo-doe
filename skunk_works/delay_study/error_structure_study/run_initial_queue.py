"""Declared bin-phase, buffered cold-start and hidden-heater sensitivities.

Run each fit and its independent dimensional/bin audit sequentially in the
single frozen parent process. This changes initialization conventions only;
the primary reported fit and the fifty-draw recovery design remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import run_joint_quantized
from audit_joint_quantized import audit
from run_bounded import write

CASES={
    'fixed_floor':{'phase':0.},
    'bins_floor':{'phase':0.,'initial_sensor_bins':True},
    'bins_buffer_nearest':{'phase':-.5,'initial_sensor_bins':True,'initial_noise_buffer_C':.1},
    'bins_buffer_floor':{'phase':0.,'initial_sensor_bins':True,'initial_noise_buffer_C':.1},
    'hidden_heater_2C':{'phase':-.5,'startup_band_C':2.},
    'hidden_heater_5C':{'phase':-.5,'startup_band_C':5.}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--noise1',type=Path,required=True);parser.add_argument('--noise2',type=Path,required=True)
    parser.add_argument('--cases',choices=list(CASES),nargs='+',required=True)
    args=parser.parse_args();begun=time.monotonic();fit=json.loads((args.input/'fit.json').read_text())
    assert fit['family']=='conductance' and fit['band']==.1 and fit['phase']==-.5
    assert not any(fit.get(k) for k in ['dataset','fixed_coordinates','startup_band_C','initial_sensor_bins','condition_initial_code'])
    assert len(set(args.cases))==len(args.cases)
    inputs=[args.input/'fit.json',args.noise1,args.noise2]
    binding={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    write(args.output/'design.json',{'cases':{k:CASES[k] for k in args.cases},'starts':3,'maxiter':300,
        'score_backend':'compiled','input_sha256':binding,
        'notice':'Nearest-bin fit is already terminal and audited; reuse it. Initial buffers are bounded sensitivities, not noise-posterior integration. Hidden heater shifts do not move initial sensors. All records informed structure selection.'})
    rows=[]
    for name in args.cases:
        case=CASES[name];directory=args.output/name;directory.mkdir()
        argv=['run_joint_quantized.py','--output',str(directory),'--warm',str(args.input/'fit.json'),
            '--noise1',str(args.noise1),'--noise2',str(args.noise2),'--band','.1','--family','conductance',
            '--phase',str(case['phase']),'--starts','3','--maxiter','300','--score-backend','compiled']
        if case.get('initial_sensor_bins'):argv+=['--initial-sensor-bins']
        if case.get('initial_noise_buffer_C'):argv+=['--initial-noise-buffer',str(case['initial_noise_buffer_C'])]
        if case.get('startup_band_C'):argv+=['--startup-band',str(case['startup_band_C'])]
        write(directory/'task_request.json',{'argv':argv,'input_sha256':binding,
            'source':'Shares frozen parent process/source; no independent child process'})
        previous=sys.argv;row={'case':name,'output':str(directory)}
        try:
            sys.argv=argv;run_joint_quantized.main()
            selected=json.loads((directory/'fit.json').read_text())
            write(directory/'task_exit.json',{'success':True,'output':str(directory),'nll':selected['best']['nll']})
            destination=args.output/(name+'_audit');destination.mkdir()
            checked=audit(SimpleNamespace(input=directory));write(destination/'audit.json',checked)
            row.update({'audit_status':checked['status'],'nll':selected['best']['nll'],
                'RMSE_by_channel_C':selected['RMSE_by_channel_C'],'parameters':selected['parameters'],
                'noise_parameters':selected['noise_parameters']})
            if checked['status']!='passed':raise RuntimeError('Initialization sensitivity failed independent numerical audit')
        except Exception as exc:
            row.update({'status':'failed','exception':repr(exc),'traceback':traceback.format_exc()})
            rows.append(row);write(args.output/'cases.json',rows)
            raise
        finally:sys.argv=previous
        row['status']='passed';rows.append(row);write(args.output/'cases.json',rows)
        write(args.output/'progress.json',{'attempted':len(rows),'declared':len(args.cases),'elapsed_s':time.monotonic()-begun})
    write(args.output/'summary.json',{'status':'complete','cases':rows,'elapsed_s':time.monotonic()-begun,
        'notice':'Sensitivity comparisons under specified initial states and quantizer phase. Improved fitted likelihood with extra initial-state coordinates is not validation or evidence of a known startup history. Preserve the primary uncertainty/transfer design.'})


if __name__=='__main__':main()
