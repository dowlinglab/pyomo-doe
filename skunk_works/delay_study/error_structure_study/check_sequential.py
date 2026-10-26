"""Independent dense covariance checks before full sequential fitting."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from bounded import Model
from physical import load_records
from sequential import covariance_objective,covariance_gradient,residual_at
from run_bounded import write
from run_sequential import mean_update


def dense(noise,residual,times):
    blocks=[];det=0.;N=sum(len(t) for t in times);ratio=np.exp(noise[-1])
    for k in [0,1]:
        cursor=0;ell=np.exp(noise[2*k]);eta=noise[2*k+1]
        for t in times:
            C=(1-eta)*np.exp(-abs(t[:,None]-t[None,:])/ell)+eta*np.eye(len(t))
            L=cholesky(C,lower=True);n=len(t)
            z=solve_triangular(L,residual[k*N+cursor:k*N+cursor+n],lower=True)/(ratio if k else 1)
            blocks.append(z);det+=2*np.log(np.diag(L)).sum()+2*n*(np.log(ratio) if k else 0)
            cursor+=n
    z=np.concatenate(blocks)
    return float(.5*(2*N*(np.log(2*np.pi)+1+np.log(z@z/(2*N)))+det))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);args=parser.parse_args()
    warm=json.loads(args.warm.read_text());original=load_records()
    records=[dict(d,t=d['t'][:71],y=d['y'][:71].copy(),u=d['u'][:71]) for d in original[:2]]
    model=Model(records,.1);shape=np.r_[warm['best']['point'][:6],0.,0.]
    X,_,initial=model.design(tuple(shape));coef=np.array([.88,.2,-.3]);pred=initial+X@coef
    rng=np.random.RandomState(17091);noise=np.array([np.log(12),.15,np.log(40),.4,np.log(.3)])
    # Add reproducible nonzero errors to test scores away from an optimum.
    errors=rng.normal(0,.08,len(pred));cursor=0
    for j,d in enumerate(records):
        n=len(d['t'])-1
        for k in [0,1]:
            s=slice(k*model.N+cursor,k*model.N+cursor+n)
            d['y'][1:,k]=d['y'][0,0]+pred[s]+errors[s]
        cursor+=n
    model=Model(records,.1);point=np.r_[shape,noise];residual,c=residual_at(model,point)
    value=covariance_objective(noise,residual,model.times)
    expected=dense(noise,residual,model.times);error=abs(value-expected)
    assert error<1e-9
    projected=model.projected(point)[0];assert abs(value-projected)<1e-9
    grad=covariance_gradient(noise,residual,model.times,model.bounds()[model.ns:])
    h=1e-5;reference=[]
    for j in range(len(noise)):
        u=np.zeros(len(noise));u[j]=h
        reference.append((dense(noise+u,residual,model.times)-dense(noise-u,residual,model.times))/(2*h))
    score_error=float(np.max(abs(grad-reference)/(1+abs(np.asarray(reference)))))
    assert score_error<2e-6
    updated,block=mean_update(model,point,300)
    assert block['solver_success'] and block['nll']<=projected+1e-7
    assert abs(model.centered_constraint(updated))<1e-7
    write(args.output/'audit.json',{'status':'passed','dense_nll_abs_error':error,
        'projected_nll_abs_error':abs(value-projected),'noise_gradient_scaled_max_abs_error':score_error,
        'mean_update_nll_decrease':projected-block['nll'],'mean_block':block,
        'scope':'Two short actual input/time designs with synthetic outputs; independent dense determinant/quadratic form, covariance finite differences, constrained mean update. Not statistical recovery or full-record convergence.'})
    print(json.dumps({'status':'passed','dense_error':error,'score_error':score_error}),flush=True)


if __name__=='__main__':main()
