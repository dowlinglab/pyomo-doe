"""Audit zero-band nesting and varying-Ua balances by direct ODE integration."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from scipy.integrate import solve_ivp
from physical import Model as SharedModel,load_records
from bounded import Model
from envelope import Objective
from nonlinear import stable_gradient


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);args=parser.parse_args()
    records=load_records();fit=json.loads(args.warm.read_text());p=np.array(fit['best']['point'])
    shared=SharedModel(records,True,True,fit['ambient'],fit['kernel'])
    fixed=Model(records,0,ambient=fit['ambient']);vary=Model(records,.1,ambient=fit['ambient'])
    nominal=np.r_[p[:6],np.zeros(12),p[6:]]
    n0,c0,_,_,i0,X0=shared.projected(p)
    nf,cf,_,_,ifixed,Xfixed=fixed.projected(p)
    nv,cv,_,_,ivary,Xvary=vary.projected(nominal)
    matrix_error=max(np.max(abs(X0-Xfixed)),np.max(abs(X0-Xvary)),np.max(abs(i0-ivary)))
    nll_error=max(abs(n0-nf),abs(n0-nv))
    assert matrix_error<1e-10 and nll_error<1e-6,(matrix_error,nll_error)
    # Synthetic cold starts with nonzero channel difference, nonzero ambient,
    # and a nonconstant held input exercise all coupled coordinates.
    direct_errors=[]
    for delta in [np.log(.9),0.,np.log(1.1)]:
        d={k:(v.copy() if hasattr(v,'copy') else v) for k,v in records[0].items()}
        d['t']=d['t'][:121];d['u']=d['u'][:121];d['y']=d['y'][:121]
        m=Model([d],.2);shape=np.r_[p[:6],delta]
        x,_,init=m.design(tuple(shape));K=.8;ambient=.7
        predicted=(init+x@np.array([K,ambient])).reshape(2,-1).T+d['y'][0,0]
        ta,rs,r,q=np.exp(p[:4]);g=np.exp(delta);L=np.exp(p[4]);curve=p[5]
        reference=d['y'][0,0]
        state=np.array([reference,reference,d['y'][0,1],d['y'][0,1],0.])
        out=[]
        for j,dt in enumerate(np.diff(d['t'])):
            u=d['u'][j];mapped=u+curve*u*(1-u/100)
            def balance(t,z):
                h1,s1,h2,s2,power=z
                A=reference+ambient
                return [(-g*(h1-A)+r*(s1-h1)+q*(h2-h1)+K*power)/ta,
                        (h1-s1)/rs,(-g*(h2-A)+r*(s2-h2)+q*(h1-h2))/ta,
                        (h2-s2)/rs,(mapped-power)/L]
            result=solve_ivp(balance,[0,dt],state,rtol=2e-11,atol=2e-12)
            assert result.success
            state=result.y[:,-1];out.append(state[[1,3]])
        error=float(np.max(abs(np.array(out)-predicted)));direct_errors.append(error)
        assert error<2e-7,error
    # Three short real-input records keep the full profiled gradient check
    # affordable while covering centered deviation and covariance coordinates.
    short=[]
    for d in records[:3]:
        v=dict(d)
        for key in ['t','u','y']:v[key]=d[key][:181].copy()
        short.append(v)
    m=Model(short,.1);point=np.r_[p[:6],[-.02,0.,.02],p[6:]]
    # Keep this numerical check away from shape/noise boundaries.
    point[2]=np.log(.03)
    o=Objective(m);envelope=o.gradient(point)
    numerical=stable_gradient(lambda v:m.projected(v)[0],point,m.bounds(),step=3e-6)
    gradient_error=float(np.max(abs(envelope-numerical)/(1+abs(numerical))))
    assert gradient_error<3e-3,gradient_error
    assert abs(m.centered_constraint(point))<1e-14
    result={'status':'passed','zero_band_design_max_abs_C_per_unit':float(matrix_error),
            'zero_band_nll_max_abs':float(nll_error),'direct_ODE_max_abs_C':direct_errors,
            'profiled_gradient_max_scaled_error':gradient_error,
            'notice':'Verifies reparameterization, zero-band nesting, held-input mean propagation and profiled derivatives; not synthetic statistical recovery or a global optimum proof.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
