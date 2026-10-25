"""Check envelope derivatives against fully reprofiled likelihood differences."""
import argparse
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'));sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from physical import Model,load_records
from envelope import Objective
from nonlinear import stable_gradient
from run_mean import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rng=np.random.RandomState(2918);data=load_records()[0];data={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in data.items()}
    for k in ['t','u','y']:data[k]=data[k][:180]
    data['y'][:]=[22,22.32];rows=[]
    for lag,curve,ambient,kernel in [(False,False,'constant','iid'),(True,False,'constant','ou_nugget'),(True,True,'constant','ou_nugget'),(True,True,'linear','ou_nugget')]:
        d={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in data.items()};m=Model([d],lag,curve,ambient,kernel)
        shape=np.r_[np.log([150.,12.,.4,.35]),[np.log(3.)] if lag else [],[.1] if curve else []]
        X,_,initial=m.design(tuple(shape));coeff=np.r_[.9,[-.3] if ambient=='constant' else [-.3,.4]]
        y=(initial+X@coeff).reshape(2,-1).T+22+rng.normal(0,.08,(len(d['t'])-1,2));d['y'][1:]=y
        m=Model([d],lag,curve,ambient,kernel);point=np.r_[shape,[] if kernel=='iid' else [np.log(20),.2,np.log(30),.3],np.log(.7)]
        objective=Objective(m)
        for at_bound in [False,True]:
            p=point.copy()
            if at_bound:p[2]=m.bounds()[2][0]+1e-12
            actual=objective.gradient(p);expected=stable_gradient(lambda x:m.projected(x)[0],p,m.bounds(),step=2e-5)
            normalized=float(np.max(abs(actual-expected)/(1+abs(expected))))
            if normalized>3e-3:raise AssertionError(('Envelope derivative',normalized,actual,expected))
            rows.append({'lag':lag,'curve':curve,'ambient':ambient,'kernel':kernel,'near_shape_bound':at_bound,
                'max_normalized_derivative_error':normalized,'envelope':actual.tolist(),'full_reprofiled':expected.tolist()})
    write_json(args.output/'audit.json',{'status':'passed','cases':rows,'scope':'Envelope theorem derivatives versus independently re-solving the bounded linear regression at every difference step, at interior and near-boundary synthetic cases. Does not prove global optimization.'})
    print('Envelope derivative checks passed',flush=True)


if __name__=='__main__':main()
