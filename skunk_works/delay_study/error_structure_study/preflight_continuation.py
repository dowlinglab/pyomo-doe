"""Read-only selected-start rechecks before migrating owned numerical jobs."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from joint_quantized import Model
from continuation import arguments,validate_request,saved_rows,recheck_row,sha,parent_receipt
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tasks',type=Path,nargs='+',required=True);args=parser.parse_args();rows=[]
    for j,path in enumerate(args.tasks):
        request=json.loads((path/'task_request.json').read_text());argv=list(request['argv']);options=arguments(argv)
        argv[argv.index('--output')+1]=str(args.output/('not_dispatched_'+str(j)))
        if '--score-backend' in argv:argv[argv.index('--score-backend')+1]='compiled'
        else:argv+=['--score-backend','compiled']
        receipt=validate_request(path,argv);receipt['parent']=parent_receipt(path)
        records=load_records()
        if '--dataset' in options:
            data=np.load(options['--dataset'][0],allow_pickle=False)
            assert data['record_names'].tolist()==[d['name'] for d in records]
            for i,d in enumerate(records):
                assert np.array_equal(data['time_'+str(i)],d['t']) and np.array_equal(data['input_'+str(i)],d['u'])
                assert np.array_equal(data['observed_'+str(i)][0],d['y'][0])
            records=[dict(d,y=data['observed_'+str(i)]) for i,d in enumerate(records)]
        if '--records' in options:records=[records[int(i)] for i in options['--records']]
        fixed_values=options.get('--fix-coordinate',[]);assert len(fixed_values)%2==0
        fixed={fixed_values[i]:float(fixed_values[i+1]) for i in range(0,len(fixed_values),2)}
        def value(flag,default=None):return options.get(flag,[default])[0]
        m=Model(records,float(value('--band',.1)),float(value('--phase',-.5)),int(value('--nodes',128)),value('--family','conductance'),
            fixed=fixed,minimum_sensor_loading=float(value('--minimum-sensor-loading',.005)),
            condition_initial_code='--condition-initial-code' in options,score_backend='compiled')
        starts=saved_rows(path,int(value('--starts',2)));checks=[]
        for old in starts:
            new=recheck_row(m,old);checks.append({'start':old['start'],'prior_success':old['success'],
                'nll_difference':new['nll']-old['nll'] if old.get('success') else None,
                'current_stationarity':new.get('stationarity') if old.get('success') else None})
        receipt['completed_start_checks']=checks;receipt['completed_starts_sha256']=sha(path/'starts.json') if starts else None
        rows.append(receipt);write(args.output/'preflight.json',{'status':'in_progress','tasks':rows})
    write(args.output/'preflight.json',{'status':'passed','tasks':rows,
        'notice':'Read-only checks of immutable input bindings and currently completed starts. No process stopped, fit dispatched, unfinished point inferred or earlier failure promoted. A later continuation must recheck its final saved prefix after the old parent is terminal.'})
    print(json.dumps({'status':'passed','tasks':len(rows),'completed_starts':sum(len(r['completed_start_checks']) for r in rows)}),flush=True)


if __name__=='__main__':main()
