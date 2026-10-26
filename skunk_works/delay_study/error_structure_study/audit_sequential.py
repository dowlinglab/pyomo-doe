"""Check full sequential fits against dimensional means and dense Gaussian ML."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from physical import load_records
from bounded import Model
from envelope import Objective
from run_bounded import stationary,write
from audit_common import dimensional_mean,dense_gaussian_profile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();f=json.loads((args.input/'sequential_fit.json').read_text());trace=json.loads((args.input/'iterations.json').read_text())
    assert json.loads((args.input/'exit.json').read_text())['exit_code']==0
    assert hashlib.sha256(args.reference.read_bytes()).hexdigest()==f['warm_sha256']
    reference=json.loads(args.reference.read_text());records=load_records();assert f['record_names']==[d['name'] for d in records]
    assert all(f['raw_sha256'][d['name']]==d['sha256'] for d in records)
    m=Model(records,f['band']);ns=m.ns;observed=np.concatenate([np.concatenate([d['y'][1:,k] for d in records]) for k in [0,1]])
    rows=[];cycles=[r for r in trace if 'cycle' in r];assert cycles
    for j,r in enumerate(cycles):
        assert r['cycle']==j and r['covariance_block']['solver_success'] and r['mean_block']['solver_success']
        before=r['before'];linear=r['after_linear_reprojection'];after=r['after_mean']
        a=np.array(before['point']);b=np.array(linear['point']);c=np.array(after['point'])
        assert np.max(abs(a[:ns]-b[:ns]))<1e-14 and np.max(abs(b[ns:]-c[ns:]))<1e-14
        assert np.max(abs(np.array(before['coefficients'])-r['covariance_block']['fixed_coefficients']))<1e-13
        decreases=[before['nll']-r['frozen_residual_covariance_nll'],r['frozen_residual_covariance_nll']-linear['nll'],linear['nll']-after['nll']]
        assert min(decreases)>-2e-5 and np.max(abs(np.array(decreases)-r['nll_decreases']))<1e-9
        if j:assert abs(before['nll']-cycles[j-1]['after_mean']['nll'])<1e-8
        for point in [a,b,c]:
            bb=np.array(m.bounds());assert np.all(point>=bb[:,0]-1e-7) and np.all(point<=bb[:,1]+1e-7)
            assert abs(m.centered_constraint(point))<1e-7
        rows.append({'cycle':j,'block_nll_decreases':decreases})
    final=f['final'];point=np.array(final['point']);pred=dimensional_mean(point,final['coefficients'],records,f['band'])
    X,_,initial=m.design(tuple(point[:ns]));ref=np.concatenate([np.concatenate([np.repeat(d['y'][0,0],len(d['t'])-1) for d in records]) for k in [0,1]])
    internal=ref+initial+X@np.array(final['coefficients']);mean_error=float(np.max(abs(pred-internal)));assert mean_error<2e-7
    nll,sigma1,sigma2=dense_gaussian_profile(records,observed-pred,point[ns:])
    assert abs(nll-final['nll'])<3e-5 and abs(sigma1-final['sigma1_C'])<1e-8 and abs(sigma2-final['sigma2_C'])<1e-8
    dense_blocks=[]
    for r in [cycles[0],cycles[-1]] if len(cycles)>1 else cycles:
        before=r['before'];linear=r['after_linear_reprojection'];a=np.array(before['point']);b=np.array(linear['point'])
        residual=observed-dimensional_mean(a,before['coefficients'],records,f['band'])
        value,_,_=dense_gaussian_profile(records,residual,b[ns:]);error=abs(value-r['frozen_residual_covariance_nll']);assert error<3e-5
        residual=observed-dimensional_mean(b,linear['coefficients'],records,f['band'])
        value,_,_=dense_gaussian_profile(records,residual,b[ns:]);linear_error=abs(value-linear['nll']);assert linear_error<3e-5
        dense_blocks.append({'cycle':r['cycle'],'frozen_mean_covariance_nll_error':error,'linear_reprojection_nll_error':linear_error})
    objective=Objective(m);stat=stationary(m,point,objective.gradient(point));assert stat['lagrangian_projected_gradient_max_abs']<.05
    iid_audit=None
    if trace[0].get('stage')=='iid_mean':
        iid=Model(records,f['band'],kernel='iid');q=np.array(trace[0]['fit']['point']);g=Objective(iid).gradient(q)
        fixed_ratio_gradient=float(g[-1]);g[-1]=0.;mean_stat=stationary(iid,q,g)
        iid_audit={'channel_SD_ratio_fixed_at_initial_mean_stage':float(np.exp(q[-1])),
            'whole_objective_ratio_score_not_optimized_in_this_block':fixed_ratio_gradient,
            'conditional_mean_stationarity':mean_stat,
            'notice':'IID temporal errors with a fixed initial relative channel weight, not a joint IID variance-component MLE. Subsequent covariance blocks estimate that ratio.'}
    assert f['status']=='converged' and f['all_blocks_solver_success'] and cycles[-1]['converged']
    result={'status':'passed','input_sha256':hashlib.sha256((args.input/'sequential_fit.json').read_bytes()).hexdigest(),
        'trace_sha256':hashlib.sha256((args.input/'iterations.json').read_bytes()).hexdigest(),
        'final_dimensional_mean_max_error_C':mean_error,'final_dense_nll_abs_error':abs(nll-final['nll']),
        'final_nll_minus_joint_reference':float(nll-reference['best']['nll']),'cycles':rows,'dense_block_checks':dense_blocks,
        'final_stationarity':stat,'initial_iid_stage':iid_audit,
        'scope':'Full twelve-run dimensional mean/dense likelihood, fixed-mean covariance and linear-reprojection arithmetic for first/last cycles, all-cycle coordinate/centering/monotonicity, and final convergence. No global-optimum or scientific residual adequacy claim.'}
    write(args.output/'audit.json',result);print(json.dumps({k:result[k] for k in ['status','final_dense_nll_abs_error','final_nll_minus_joint_reference']}),flush=True)


if __name__=='__main__':main()
