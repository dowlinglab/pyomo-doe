"""Exercise continuation with small generated, all-fixed physical-model tasks.

These are control-flow tests, not parameter-estimation experiments or coverage
draws. Fixture markers explicitly disclose their simulated interruption.
"""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
import physical
import run_joint_quantized
from joint_quantized import Model
from fit_task import run,command
from continuation import validate_request,run_task
from quantized import report
from bin_diagnostics import sample_observations
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    records=[]
    for j in range(2):
        t=np.arange(10,dtype=float);u=50+20*np.sin(t/2+j)
        records.append({'name':'generated_control_'+str(j),'t':t,'u':u,
            'y':report(np.full((len(t),2),22.)),'sha256':'generated-control-fixture-'+str(j)})
    m=Model(records);p=np.r_[np.log([175.,15.,.03,.4,2.7]),.12,[-.02,.02],np.log(.87),[.2,-.3],np.log([.3,30.,.04,.1,10.,.03])]
    flat=m.mean(p,False);means=[];cursor=0
    for d in records:
        n=len(d['t'])-1;v=d['y'].copy()
        for k in [0,1]:v[1:,k]=flat[k*m.N+cursor:k*m.N+cursor+n]
        means.append(v);cursor+=n
    noise=[{'latent_SD_C':.3,'ell_s':30.,'measurement_SD_C':.04},
        {'latent_SD_C':.1,'ell_s':10.,'measurement_SD_C':.03}]
    records=sample_observations(records,means,noise,-.5,71271)
    warm=args.output/'warm.json';noise1=args.output/'noise1.json';noise2=args.output/'noise2.json'
    write(warm,{'record_names':[d['name'] for d in records],'kernel':'quantized_ou_nugget','band':.1,
        'best':{'point':p.tolist()},'coefficients':[.87,.2,-.3],
        'parameters':{'log_deviations':[-.02,.02]},'noise_parameters':noise})
    write(noise1,{});write(noise2,{})
    fixed=dict(zip(m.coordinate_names(),p.tolist()))
    common={'noise1':noise1,'noise2':noise2,'family':'conductance','band':.1,'phase':-.5,
        'starts':2,'maxiter':2,'fixed':fixed,'score_backend':'compiled'}
    original_physical=physical.load_records;original_joint=run_joint_quantized.load_records
    physical.load_records=lambda:records;run_joint_quantized.load_records=lambda:records
    try:
        original=args.output/'original';result=run(original,warm,**common);assert result['success'],result
        starts=json.loads((original/'starts.json').read_text());assert len(starts)==2 and all(r['success'] for r in starts)
        fixture=args.output/'interrupted_fixture';fixture.mkdir()
        (fixture/'task_request.json').write_bytes((original/'task_request.json').read_bytes())
        write(fixture/'starts.json',starts[:1]);write(fixture/'test_fixture.json',{'purpose':'Simulated interruption after first genuinely evaluated all-fixed start; no inference or real process claim'})
        argv,_=command(args.output/'continued',warm,continue_task=fixture,**common)
        validate_request(fixture,argv)
        rejected=[]
        for flag,value in [('--band','.2'),('--maxiter','3'),('--phase','0.')]:
            altered=list(argv);altered[altered.index(flag)+1]=value
            try:validate_request(fixture,altered)
            except (ValueError,AssertionError):rejected.append(flag)
            else:raise AssertionError('Changed protocol accepted '+flag)
        continued=run(args.output/'continued',warm,continue_task=fixture,**common);assert continued['success'],continued
        rows=json.loads((args.output/'continued/starts.json').read_text())
        assert rows[0]['reused_terminal_start'] and not rows[1].get('reused_terminal_start') and len(rows)==2
        # A saved iterate cannot claim success when the original iteration
        # budget is exhausted, even though this all-fixed point is stationary.
        exhausted=args.output/'exhausted_fixture';exhausted.mkdir()
        for filename in ['task_request.json','starts.json','test_fixture.json']:(exhausted/filename).write_bytes((fixture/filename).read_bytes())
        write(exhausted/'checkpoint_start_1.json',{'start':1,'completed_callback_iterations':2,'point':p.tolist(),'nll':starts[0]['nll']})
        result=run(args.output/'exhausted',warm,continue_task=exhausted,**common);assert result['success']
        erows=json.loads((args.output/'exhausted/starts.json').read_text());assert erows[0]['success'] and not erows[1]['success'] and 'budget exhausted' in erows[1]['message']
        failed=args.output/'failed_fixture';failed.mkdir()
        for filename in ['task_request.json','test_fixture.json']:(failed/filename).write_bytes((fixture/filename).read_bytes())
        row=dict(starts[0],success=False,solver_success=False,message='Explicit failed-start fixture')
        write(failed/'starts.json',[row]);result=run(args.output/'failed_preserved',warm,continue_task=failed,**common);assert result['success']
        frows=json.loads((args.output/'failed_preserved/starts.json').read_text());assert not frows[0]['success'] and frows[1]['success']
        reused=run_task(args.output/'reused_terminal',warm,previous=original,**common)
        assert reused['success'] and reused['reused_terminal_task'] and reused['output']==str(original)
        assert not (args.output/'reused_terminal/fit.json').exists(),'Terminal reuse must not repeat fitting/export'
    finally:
        physical.load_records=original_physical;run_joint_quantized.load_records=original_joint
    write(args.output/'check.json',{'status':'passed','protocol_changes_rejected':rejected,
        'completed_prefix_retained':True,'unfinished_branch_solved':True,'exhausted_checkpoint_not_promoted':True,
        'failed_start_not_promoted':True,'terminal_task_reused_without_optimization':True,
        'scope':'Generated ten-time-point, two-channel all-fixed physical-model tasks, explicit interruption fixtures and input/protocol guards. No scientific estimation, solver convergence-rate or uncertainty calibration claim.'})
    print(json.dumps({'status':'passed','protocol_changes_rejected':rejected}),flush=True)


if __name__=='__main__':main()
