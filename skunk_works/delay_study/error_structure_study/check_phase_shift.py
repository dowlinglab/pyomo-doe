"""Test the absolute-temperature/threshold-phase symmetry of the linear mean."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from joint_quantized import Model
from quantized import STEP,report,bins
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);args=parser.parse_args()
    warm=json.loads(args.warm.read_text());records=[dict(d,t=d['t'][:81],u=d['u'][:81],y=d['y'][:81].copy()) for d in load_records()[:2]]
    rng=np.random.RandomState(81193);C=STEP/2;rows=[]
    for band in [0.,.1]:
        nearest=Model(records,band,-.5,128,'conductance',initial_sensor_bins=True)
        shape=np.r_[warm['best']['point'][:6],[-.03,.03] if band else []];shape[2]=np.log(.03)
        lo,hi=nearest.mean_model.coefficient_bounds();coef=(lo+hi)/2;coef[:3]=[.88,.2,-.3]
        p=np.r_[shape,np.log(coef[0]),coef[1:],np.log([.3,30.,.05,.1,10.,.03])]
        mu=nearest.mean(p,False);cursor=0;generated=[]
        for j,d in enumerate(records):
            n=len(d['t'])-1;y=d['y'].copy()
            for k in [0,1]:
                analog=mu[k*nearest.N+cursor:k*nearest.N+cursor+n]+rng.normal(0,.05,n)
                y[1:,k]=report(analog,-.5)
                assert np.array_equal(report(analog+C,0.),y[1:,k])
            generated.append(dict(d,y=y));cursor+=n
        nearest=Model(generated,band,-.5,128,'conductance',initial_sensor_bins=True)
        floor=Model(generated,band,0.,128,'conductance',initial_sensor_bins=True)
        shifted=p.copy();shifted[nearest.nshape+1:nearest.nmean]+=C
        bounds=np.array(floor.bounds());assert np.all(shifted>=bounds[:,0]) and np.all(shifted<=bounds[:,1])
        difference=floor.mean(shifted,False)-nearest.mean(p,False)
        mean_error=float(np.max(abs(difference-C)));assert mean_error<1e-10
        likelihood_error=abs(nearest.objective(p)-floor.objective(shifted));assert likelihood_error<1e-7
        gradient_error=float(np.max(abs(nearest.gradient(p)-floor.gradient(shifted)))/(1+np.max(abs(nearest.gradient(p)))))
        assert gradient_error<1e-6
        code_error=0.
        for d in generated:
            a,b=bins(d['y'][1:],-.5);aa,bb=bins(d['y'][1:],0.)
            code_error=max(code_error,float(np.max(abs(aa-a-C))),float(np.max(abs(bb-b-C))))
        assert code_error<1e-12
        rows.append({'band':band,'absolute_shift_C':C,'mean_translation_max_error_C':mean_error,
            'bin_translation_max_error_C':code_error,'bin_nll_invariance_error':likelihood_error,
            'score_invariance_scaled_error':gradient_error})
    write(args.output/'audit.json',{'status':'passed','cases':rows,
        'scope':'Linear coupled mean with cold initial sensor-bin coordinates, common absolute shift of both sensor/heater initial states and ambient, slack ambient bounds. Nearest/floor thresholds shift by half a code. Not a symmetry of fixed initial readings, active ambient limits, nonlinear absolute-temperature radiation or independently calibrated thermometry.'})
    print(json.dumps({'status':'passed','max_nll_error':max(r['bin_nll_invariance_error'] for r in rows),
                      'max_mean_error_C':max(r['mean_translation_max_error_C'] for r in rows)}),flush=True)


if __name__=='__main__':main()
