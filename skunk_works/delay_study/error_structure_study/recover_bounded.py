"""Noiseless synthetic recovery of centered deviations under actual ENV2 inputs."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.optimize import minimize, lsq_linear
from physical import load_records
from bounded import Model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);args=parser.parse_args()
    fit=json.loads(args.warm.read_text());base=np.array(fit['best']['point'][:6]);base[2]=np.log(.03)
    # One record from each sine period. Actual sampling and commands retained.
    records=[load_records()[j] for j in (0,3,6)]
    known=np.array([-.04,.01,.03]);coeff=np.array([.8,.5,-.7,.9])
    m=Model(records,.1);X,_,initial=m.design(tuple(np.r_[base,known]));truth=initial+X@coeff
    synthetic=[];cursor=0
    for d in records:
        v=dict(d);y=d['y'].copy();n=len(d['t'])-1
        for k in (0,1):y[1:,k]=d['y'][0,0]+truth[k*m.N+cursor:k*m.N+cursor+n]
        v['y']=y;synthetic.append(v);cursor+=n
    recovered=Model(synthetic,.1);largest=0.;rows=[]
    def projected(delta,return_coeff=False):
        X,y,_=recovered.design(tuple(np.r_[base,delta]))
        f=lsq_linear(X,y,bounds=recovered.coefficient_bounds(),tol=1e-12,lsmr_tol=1e-13)
        assert f.success
        residual=X@f.x-y
        return (float(residual@residual),f.x) if return_coeff else float(residual@residual)
    for seed in [np.zeros(3),np.array([.02,-.04,.02])]:
        f=minimize(projected,seed,method='SLSQP',bounds=[(np.log(.9),np.log(1.1))]*3,
                   constraints=[{'type':'eq','fun':lambda d:float(sum(d)),'jac':lambda d:np.ones(3)}],
                   options={'maxiter':300,'ftol':1e-13})
        sse,c=projected(f.x,True);error=float(np.max(abs(f.x-known)))
        assert f.success and error<5e-6 and np.max(abs(c-coeff))<2e-5,(f.message,error,c)
        largest=max(largest,error)
        rows.append({'seed':seed.tolist(),'recovered_deviations':f.x.tolist(),'coefficients':c.tolist(),'SSE_C2':sse,'solver_success':bool(f.success)})
    result={'status':'passed','known_log_deviations':known.tolist(),'known_coefficients':coeff.tolist(),
        'max_deviation_error':largest,'attempts':rows,
        'scope':'Noiseless centered conductance/gain/ambient recovery with shared dynamic parameters fixed, actual three-period ENV2 inputs and times. Does not establish all-parameter statistical identifiability or interval coverage.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
