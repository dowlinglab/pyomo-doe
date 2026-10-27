"""Invoke the frozen joint fitter inside a measured parent experiment job."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from run_bounded import write


def command(output,warm,noise1,noise2,family,band,phase,nodes=128,starts=2,maxiter=300,
        fixed=None,dataset=None,records=None,minimum_sensor_loading=.005,condition_initial_code=False,score_backend='reference',continue_task=None,startup_band=0.):
    argv=['run_joint_quantized.py','--output',str(output),'--warm',str(warm),
        '--noise1',str(noise1),'--noise2',str(noise2),'--family',family,'--band',str(band),
        '--phase',str(phase),'--nodes',str(nodes),'--starts',str(starts),'--maxiter',str(maxiter)]
    argv+=['--minimum-sensor-loading',str(minimum_sensor_loading)]
    argv+=['--score-backend',score_backend]
    if startup_band:
        if startup_band<0 or continue_task is not None:raise ValueError('Startup profiling supports fresh tasks with positive bounds only')
        argv+=['--startup-band',str(startup_band)]
    if condition_initial_code:argv+=['--condition-initial-code']
    for name,value in (fixed or {}).items():argv+=['--fix-coordinate',name,str(value)]
    if dataset is not None:argv+=['--dataset',str(dataset)]
    if records is not None:argv+=['--records']+[str(j) for j in records]
    if continue_task is not None:argv+=['--continue-task',str(continue_task)]
    inputs=[warm,noise1,noise2]+([dataset] if dataset is not None else [])
    return argv,inputs


def run(output,warm,*args,**kwargs):
    import run_joint_quantized
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    argv,inputs=command(output,warm,*args,**kwargs)
    write(output/'task_request.json',{'argv':argv,'started_utc':datetime.now(timezone.utc).isoformat(),
        'input_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs},
        'source':'Shares the parent frozen source_snapshot.json, runtime and process; no independent child process is launched.'})
    previous=sys.argv;begun=time.monotonic()
    try:
        sys.argv=argv;run_joint_quantized.main();fit=json.loads((output/'fit.json').read_text())
        result={'success':True,'output':str(output),'nll':fit['best']['nll'],
                'elapsed_s':time.monotonic()-begun,'finished_utc':datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        result={'success':False,'output':str(output),'exception':repr(exc),
                'elapsed_s':time.monotonic()-begun,'finished_utc':datetime.now(timezone.utc).isoformat()}
    finally:sys.argv=previous
    write(output/'task_exit.json',result)
    return result
