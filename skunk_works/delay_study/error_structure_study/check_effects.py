"""Independent hidden-state propagation and startup-profile derivative check."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from effects import Model
from physical import load_records
from audit_bounded import independent
from envelope import Objective
from nonlinear import stable_gradient


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);args=parser.parse_args()
    warm=json.loads(args.warm.read_text());base=np.array(warm['best']['point'][:6]);base[2]=np.log(.03)
    records=load_records();d=dict(records[0])
    for key in ['t','u','y']:d[key]=d[key][:181].copy()
    model=Model([d],.1,2.);p=np.r_[base,.03];X,_,initial=model.design(tuple(p))
    coef=np.array([.8,.5,1.8,-.6]);pred=(initial+X@coef).reshape(2,-1).T+d['y'][0,0]
    fit={'parameters':{'Us_over_Ua_nominal':float(np.exp(base[2])),'Ux_over_Ua_nominal':float(np.exp(base[3])),
        'tA_nominal_s':float(np.exp(base[0])),'rS_s':float(np.exp(base[1])),'K_common_nominal':.8,
        'Ua_run_over_nominal':[float(np.exp(.03))],'input_lag_s':float(np.exp(base[4])),'command_curvature':float(base[5])},
        'record_names':[d['name']],'lag':True,'ambient':'constant','coefficients':coef.tolist(),'startup_offsets_C':[[1.8,-.6]]}
    independent_prediction=independent(d,fit)[1:]
    error=float(np.max(abs(pred-independent_prediction)));assert error<1e-8,error
    short=[]
    for d in records[:3]:
        v=dict(d)
        for key in ['t','u','y']:v[key]=d[key][:181].copy()
        short.append(v)
    model=Model(short,.1,2.);point=np.r_[base,[-.02,0.,.02],warm['best']['point'][warm['shape_count']:]]
    objective=Objective(model)
    direct=stable_gradient(lambda p:model.projected(p)[0],point,model.bounds(),step=3e-6)
    envelope=objective.gradient(point)
    gradient_error=float(np.max(abs(direct-envelope)/(1+abs(direct))))
    assert gradient_error<3e-3,gradient_error
    result={'status':'passed','direct_dimensional_matrix_max_error_C':error,'profiled_gradient_max_scaled_error':gradient_error,
        'scope':'Nonzero hidden initial-state columns against independent dimensional augmented exponential, plus full/profiled gradient including bounded startup nuisance. Not evidence that hardware starts were unequilibrated.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
