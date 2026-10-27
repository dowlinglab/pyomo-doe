"""Independent dimensional/score and fixed-state controls before startup profiles."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from audit_bounded import independent
from joint_quantized import Model
from quantized import report
from run_bounded import stationary,write
from fit_task import command
from run_startup_profile import selection


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warm',type=Path,required=True);args=p.parse_args()
    warm=json.loads(args.warm.read_text());original=load_records();records=[]
    for d in original[:2]:records.append(dict(d,t=d['t'][:61].copy(),u=d['u'][:61].copy(),y=d['y'][:61].copy()))
    shape=np.r_[warm['best']['point'][:6],-.03,.03];shape[2]=np.log(.03)
    coef=np.array([.88,.4,-.3,.5,-.4,-.2,.3]);offsets=coef[3:].reshape(2,2)
    point=np.r_[shape,np.log(coef[0]),coef[1:],np.log([.3,30.,.05,.1,10.,.03])]
    model=Model(records,.1,-.5,128,'conductance',startup_band=2.)
    assert len(point)==len(model.coordinate_names())==len(model.bounds())
    mean,J=model.mean(point);parameters=dict(warm['parameters'])
    parameters.update({'K_common_nominal':coef[0],'Us_over_Ua_nominal':float(np.exp(shape[2])),
        'Ua_run_over_nominal':np.exp(shape[6:]).tolist()})
    physical={'parameters':parameters,'record_names':[d['name'] for d in records],
        'lag':True,'ambient':'constant','coefficients':coef.tolist(),'startup_offsets_C':offsets.tolist()}
    error=0.;cursor=0;generated=[];rng=np.random.RandomState(201731)
    for j,d in enumerate(records):
        expected=independent(d,physical);n=len(d['t'])-1;y=d['y'].copy()
        for k in [0,1]:
            values=mean[k*model.N+cursor:k*model.N+cursor+n]
            error=max(error,float(np.max(abs(values-expected[1:,k]))))
            y[1:,k]=report(values+rng.normal(0,.05,n),-.5)
        generated.append(dict(d,y=y));cursor+=n
    assert error<1e-8
    model=Model(generated,.1,-.5,128,'conductance',startup_band=2.,score_backend='compiled')
    reference=Model(generated,.1,-.5,128,'conductance',startup_band=2.)
    score=model.gradient(point);assert abs(model.objective(point)-reference.objective(point))<1e-8
    assert np.max(abs(score-reference.gradient(point))/(1+abs(score)))<1e-6
    names=model.coordinate_names();indices=[j for j,n in enumerate(names) if n.startswith('initial_H')];checks=[]
    for j in indices:
        delta=np.zeros(len(point));delta[j]=2e-5
        fd=(model.objective(point+delta)-model.objective(point-delta))/(4e-5)
        scaled=abs(fd-score[j])/(1+abs(fd));assert scaled<1e-4;checks.append({'coordinate':names[j],'scaled_score_error':float(scaled)})
    name=names[indices[0]];fixed=Model(generated,.1,-.5,128,'conductance',startup_band=2.,fixed={name:0.})
    q=point.copy();q[indices[0]]=0.;assert fixed.bounds()[indices[0]]==(0.,0.)
    assert abs(fixed.objective(q)-reference.objective(q))<1e-9
    injected=np.zeros(len(q));injected[indices[0]]=1e6
    assert stationary(fixed,q,injected)['lagrangian_projected_gradient_max_abs']==0.
    policy=json.loads((HERE/'uncertainty_config.json').read_text())['consequential_startup_profile']
    index,selected,values=selection(warm,policy)
    assert selected==warm['coordinate_names'][index] and 0. in values and -2. in values and 2. in values
    default,_=command(args.output/'unused',args.warm,args.warm,args.warm,'conductance',.1,-.5)
    startup,_=command(args.output/'unused',args.warm,args.warm,args.warm,'conductance',.1,-.5,startup_band=2.)
    assert '--startup-band' not in default and startup[startup.index('--startup-band')+1]=='2.0'
    write(args.output/'check.json',{'status':'passed','dimensional_mean_max_error_C':error,
        'startup_coordinate_score_checks':checks,'fixed_state_objective_and_stationarity':'passed',
        'selected_real_coordinate':selected,'selected_grid_C':values,
        'scope':'Two short actual-command synthetic records, independent dimensional hidden-state propagation, all four initial-heater bin scores, reference/compiled score agreement, fixed-coordinate bounds/objective/KKT controls and CLI flag routing. Not a hardware profile or a global optimum claim.'})


if __name__=='__main__':main()
