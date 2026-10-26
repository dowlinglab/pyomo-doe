"""Dimensional cold-state propagation and initial-bin score verification."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from initial_sensor import Model
from joint_quantized import Model as JointModel
from audit_bounded import independent
from quantized import report
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);args=parser.parse_args()
    warm=json.loads(args.warm.read_text());original=load_records();records=[]
    for d in original[:2]:records.append(dict(d,t=d['t'][:81].copy(),y=d['y'][:81].copy(),u=d['u'][:81].copy()))
    shape=np.r_[warm['best']['point'][:6],-.03,.03];shape[2]=np.log(.03);rows=[];rng=np.random.RandomState(81331)
    for phase,buffer in [(-.5,0.),(0.,0.),(-.5,.1),(0.,.1)]:
        m=Model(records,.1,phase,buffer);lo,hi=m.coefficient_bounds();coef=(lo+hi)/2;coef[:3]=[.88,.4,-.3]
        coef[3:]+=.2*(hi[3:]-lo[3:])*np.array([-1.,1.,1.,-1.])
        X,_,initial=m.design(tuple(shape));mean=initial+X@coef
        p=dict(warm['parameters']);p.update({'K_common_nominal':coef[0],'Us_over_Ua_nominal':float(np.exp(shape[2])),
             'Ua_run_over_nominal':np.exp(shape[6:]).tolist()})
        fit={'parameters':p,'record_names':[d['name'] for d in records],'lag':True,'curve':True,
             'ambient':'constant','coefficients':coef.tolist(),'initial_sensor_offsets_C':coef[3:].reshape(2,2).tolist()}
        cursor=0;error=0.;generated=[]
        for j,d in enumerate(records):
            pred=independent(d,fit);n=len(d['t'])-1;y=d['y'].copy()
            for k in [0,1]:
                expected=d['y'][0,0]+mean[k*m.N+cursor:k*m.N+cursor+n]
                error=max(error,float(np.max(abs(pred[1:,k]-expected))))
                y[1:,k]=report(expected+rng.normal(0,.05,n),phase)
            generated.append(dict(d,y=y));cursor+=n
        assert error<1e-8,(phase,buffer,error)
        joint=JointModel(generated,.1,phase,128,'conductance',initial_sensor_bins=True,initial_noise_buffer=buffer)
        point=np.r_[shape,np.log(coef[0]),coef[1:],np.log([.3,30.,.05,.1,10.,.03])]
        assert len(point)==len(joint.coordinate_names()) and len(point)==len(joint.bounds())
        score=joint.gradient(point);checks=[]
        for j in range(joint.nshape+1+len(records),joint.nmean):
            u=np.zeros(len(point));u[j]=2e-5
            fd=(joint.objective(point+u)-joint.objective(point-u))/(4e-5)
            checks.append(abs(fd-score[j])/(1+abs(fd)))
        assert max(checks)<1e-4,(phase,buffer,checks)
        rows.append({'phase':phase,'buffer_C':buffer,'dimensional_mean_max_error_C':error,
            'initial_sensor_coordinate_score_scaled_max_error':max(checks),
            'initial_offset_bounds_C':list(zip(lo[3:].tolist(),hi[3:].tolist()))})
    write(args.output/'audit.json',{'status':'passed','cases':rows,
        'scope':'Independent dimensional cold heater/sensor initial-state propagation, two phases/two buffers, bin-likelihood initial-state scores on synthetic outputs with recorded times/commands. Bounds are scenarios, not a probability law for initial sensor errors.'})
    print(json.dumps({'status':'passed','max_mean_error':max(r['dimensional_mean_max_error_C'] for r in rows),
                      'max_score_error':max(r['initial_sensor_coordinate_score_scaled_max_error'] for r in rows)}),flush=True)


if __name__=='__main__':main()
