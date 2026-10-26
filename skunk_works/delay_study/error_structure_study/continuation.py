"""Strict input/protocol binding for reusing completed frozen fit attempts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from run_bounded import stationary


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments(argv):
    result={};flag=None
    for token in argv[1:]:
        if token.startswith('--'):
            flag=token
            if flag in result and flag!='--fix-coordinate':raise ValueError('Repeated CLI flag in continuation request')
            if flag not in result:result[flag]=[]
        elif flag is None:raise ValueError('Unbound CLI continuation argument')
        else:result[flag].append(token)
    return result


def validate_request(previous,current_argv):
    previous=Path(previous);request=previous/'task_request.json';old=json.loads(request.read_text())
    before=arguments(old['argv']);after=arguments(current_argv);inputs={}
    for flag in ['--warm','--noise1','--noise2','--dataset']:
        if (flag in before)!=(flag in after):raise ValueError('Continuation input changed: '+flag)
        if flag not in before:continue
        assert len(before[flag])==len(after[flag])==1
        a=before.pop(flag)[0];b=after.pop(flag)[0]
        expected=old['input_sha256'][a];assert sha(a)==expected and sha(b)==expected,('Changed input',flag)
        inputs[flag]={'old_path':a,'current_path':b,'sha256':expected}
    for flags in [before,after]:
        for flag in ['--output','--score-backend','--continue-task']:flags.pop(flag,None)
    if before!=after:raise ValueError('Scientific solver protocol changed: '+repr((before,after)))
    return {'previous_task':str(previous),'request_sha256':sha(request),'inputs':inputs,
        'protocol_arguments':before,'allowed_differences':['output directory','reference/compiled mathematically checked score backend','continuation provenance flag']}


def saved_rows(previous,requested_starts):
    path=Path(previous)/'starts.json';rows=json.loads(path.read_text()) if path.exists() else []
    if [r['start'] for r in rows]!=list(range(len(rows))) or len(rows)>requested_starts:
        raise ValueError('Completed starts must be the full preserved prefix, including failures')
    return rows


def parent_receipt(previous,require_terminal=False):
    if (Path(previous)/'test_fixture.json').exists():
        return {'test_fixture_sha256':sha(Path(previous)/'test_fixture.json'),
            'notice':'Explicit control-flow test fixture, not a measured previous process'}
    for folder in [Path(previous)]+list(Path(previous).parents):
        if (folder/'source_snapshot.json').exists():
            if require_terminal and not (folder/'exit.json').exists():raise ValueError('Previous frozen parent is not terminal')
            return {'path':str(folder),'source_sha256':sha(folder/'source_snapshot.json'),
                'exit':json.loads((folder/'exit.json').read_text()) if (folder/'exit.json').exists() else None}
    raise ValueError('Previous task has no frozen parent or explicit test fixture')


def recheck_row(model,row):
    answer=dict(row);answer['reused_terminal_start']=True
    if not row.get('success'):return answer
    point=np.array(row['point']);bounds=np.array(model.bounds())
    if len(point)!=len(bounds) or np.any(point<bounds[:,0]-1e-7) or np.any(point>bounds[:,1]+1e-7):raise ValueError('Saved point chart/bounds changed')
    value=model.objective(point);stat=stationary(model,point,model.gradient(point))
    if abs(value-row['nll'])>1e-3 or abs(stat['equality_residual'])>1e-7 or stat['lagrangian_projected_gradient_max_abs']>=.05:
        raise ValueError('Completed start failed current likelihood/stationarity gate')
    answer.update({'saved_nll':row['nll'],'saved_stationarity':row.get('stationarity'),'nll':float(value),'stationarity':stat})
    return answer


def run_task(output,warm,previous=None,**kwargs):
    """Reuse a terminal subtask or continue its completed starts in a new folder.

    Entire terminal successful/failed attempts keep their original outcome.
    A previously interrupted attempt never becomes successful by declaration;
    its completed starts are rechecked and any unfinished branch is solved.
    """
    from fit_task import run
    from run_bounded import write
    output=Path(output)
    if previous is None or not (Path(previous)/'task_request.json').exists():return run(output,warm,**kwargs)
    previous=Path(previous)
    if not (previous/'task_exit.json').exists():return run(output,warm,continue_task=previous,**kwargs)
    # Build the exact request through the same argument builder used by run().
    from fit_task import command
    argv,inputs=command(output,warm,**kwargs);receipt=validate_request(previous,argv)
    receipt['frozen_parent']=parent_receipt(previous)
    result=json.loads((previous/'task_exit.json').read_text())
    if result['success']:
        from physical import load_records
        from joint_quantized import Model
        fit=json.loads((previous/'fit.json').read_text());records=load_records()
        if kwargs.get('dataset'):
            data=np.load(kwargs['dataset'],allow_pickle=False)
            assert data['record_names'].tolist()==[d['name'] for d in records]
            records=[dict(d,y=data['observed_'+str(j)]) for j,d in enumerate(records)]
        if kwargs.get('records') is not None:records=[records[j] for j in kwargs['records']]
        assert fit['record_names']==[d['name'] for d in records]
        m=Model(records,kwargs['band'],kwargs['phase'],kwargs.get('nodes',128),kwargs['family'],
            fixed=kwargs.get('fixed'),minimum_sensor_loading=kwargs.get('minimum_sensor_loading',.005),
            condition_initial_code=kwargs.get('condition_initial_code',False),score_backend=kwargs.get('score_backend','reference'))
        check=recheck_row(m,fit['best']);receipt['selected_point_recheck']={'nll_difference':check['nll']-fit['best']['nll'],'stationarity':check['stationarity']}
    output.mkdir(parents=True,exist_ok=False)
    receipt.update({'prior_exit_sha256':sha(previous/'task_exit.json'),
        'notice':'Original completed attempt reused with its outcome intact. Selected successful point is rechecked; no optimization repeated and no prior failure promoted.'})
    write(output/'reuse_receipt.json',receipt)
    answer=dict(result,reused_terminal_task=True,reuse_receipt=str(output/'reuse_receipt.json'))
    write(output/'task_exit.json',answer);return answer
