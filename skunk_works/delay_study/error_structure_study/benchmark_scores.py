"""Full-data reference/compiled likelihood and score agreement, plus node pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from joint_quantized import Model
from fast_smoothing import smooth_ou
from quantized import report
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--datasets',type=Path,nargs='*',default=[]);args=parser.parse_args()
    begun=time.monotonic();smooth_ou(np.array([0.,1.]),np.array([25.,25.]),report([25.,25.1]),.3,10.,.03)
    compile_s=time.monotonic()-begun;lookup={d['name']:d for d in load_records()};cases=[];rows=[]
    for path in args.inputs:
        fit=json.loads((path/'fit.json').read_text());assert not fit.get('dataset')
        records=[lookup[n] for n in fit['record_names']]
        assert all(d['sha256']==fit['raw_sha256'][d['name']] for d in records)
        cases.append((path.name,fit,records,{'fit_sha256':hashlib.sha256((path/'fit.json').read_bytes()).hexdigest()}))
    for dataset in args.datasets:
        fit=cases[0][1];data=np.load(dataset,allow_pickle=False);records=[]
        assert data['record_names'].tolist()==fit['record_names']
        for j,name in enumerate(fit['record_names']):
            d=lookup[name];assert np.array_equal(data['time_'+str(j)],d['t']) and np.array_equal(data['input_'+str(j)],d['u'])
            assert np.array_equal(data['observed_'+str(j)][0],d['y'][0]);records.append(dict(d,y=data['observed_'+str(j)]))
        cases.append((str(dataset),fit,records,{'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'physical_truth_fit_sha256':cases[0][3]['fit_sha256']}))
    for label,fit,records,binding in cases:
        kwargs={'band':fit['band'],'phase':fit['phase'],'family':fit['family'],'startup_band':fit.get('startup_band_C',0.),
            'initial_sensor_bins':fit.get('initial_sensor_bins',False),'initial_noise_buffer':fit.get('initial_noise_buffer_C',0.),
            'minimum_sensor_loading':fit.get('minimum_sensor_loading',.005),'condition_initial_code':fit.get('condition_initial_code',False)}
        p=np.array(fit['best']['point']);reference=Model(records,**kwargs);new=Model(records,score_backend='compiled',**kwargs)
        start=time.monotonic();a=reference.evaluate(tuple(p));oldtime=time.monotonic()-start
        start=time.monotonic();b=new.evaluate(tuple(p));newtime=time.monotonic()-start
        nllerr=abs(a[0]-b[0]);gerr=float(np.max(abs(a[1]-b[1])));scaled=float(np.max(abs(a[1]-b[1])/(1+abs(a[1]))))
        assert nllerr<2e-7 and gerr<1e-3,(label,nllerr,gerr)
        row={'case':label,'source_bindings':binding,'nll_abs_error':nllerr,'score_max_abs_error':gerr,
            'score_max_scaled_error':scaled,'reference_s':oldtime,'compiled_s':newtime,'warm_speed_ratio':oldtime/newtime,
            'reference_nodes':[d['nodes'] for d in a[2]],'compiled_nodes':[d['nodes'] for d in b[2]],'node_pilot':[]}
        # This tests the existing adaptive gates, not permission to reduce the
        # measured order. A lower order is eligible only after full score/nll
        # agreement here and later selected-point high-order audits.
        if fit['family']=='conductance':
            for count in [64,96]:
                m=Model(records,nodes=count,score_backend='compiled',**kwargs);start=time.monotonic();value,g,details=m.evaluate(tuple(p));elapsed=time.monotonic()-start
                error=abs(value-a[0]);gradient=float(np.max(abs(g-a[1])))
                row['node_pilot'].append({'requested_nodes':count,'actual_nodes':[d['nodes'] for d in details],
                    'nll_abs_error':error,'score_max_abs_error':gradient,'elapsed_s':elapsed,
                    'passes_strict_score_agreement':bool(error<2e-7 and gradient<1e-3)})
        rows.append(row);write(args.output/'benchmark.json',{'status':'in_progress','JIT_warmup_s':compile_s,'rows':rows})
        print(json.dumps({'case':label,'nll_error':nllerr,'score_error':gerr,'speed_ratio':oldtime/newtime,'node_pilot':row['node_pilot']}),flush=True)
    write(args.output/'benchmark.json',{'status':'passed','JIT_warmup_s':compile_s,'rows':rows,
        'notice':'Same source-pinned quadrature law and adaptive gates. Timings are single warm evaluations under concurrent study jobs. Lower-order pilot results do not alter live jobs or authorize scientific claims from unaudited fits. Synthetic cases use fixed generated observations at the generating parameter, not accepted recovered estimates.'})


if __name__=='__main__':main()
