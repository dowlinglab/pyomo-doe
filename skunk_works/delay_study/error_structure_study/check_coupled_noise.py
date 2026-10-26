"""Independent dense Gaussian covariance, innovation and mean-score checks."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from coupled_noise import covariance,filter_gaussian,Model
from physical import load_records
from run_bounded import write


def dense_covariance(time,noise,mode):
    a,l1,r1,b,l2,r2=np.exp(noise[:6]);C=np.zeros((2*len(time),2*len(time)))
    cross=0.;whitecross=0.;i=6
    if mode in ['process','both']:cross=noise[i]*2*a*b*np.sqrt(l1*l2)/(l1+l2);i+=1
    if mode in ['measurement','both']:whitecross=noise[i]*r1*r2
    sd=[a,b];ell=[l1,l2];measurement=[r1,r2]
    for i,ti in enumerate(time):
        for j,tj in enumerate(time):
            for k in [0,1]:
                for h in [0,1]:
                    if k==h:value=sd[k]**2*np.exp(-abs(ti-tj)/ell[k])
                    else:value=cross*np.exp(-(ti-tj)/ell[k] if ti>=tj else -(tj-ti)/ell[h])
                    if i==j:value+=measurement[k]**2 if k==h else whitecross
                    C[2*i+k,2*j+h]=value
    return C


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);args=parser.parse_args();fit=json.loads(args.warm.read_text())
    rng=np.random.RandomState(771981);time=np.cumsum(rng.uniform(.7,1.3,40));residual=rng.normal(size=(40,2))*.1;rows=[]
    for mode in ['independent','process','measurement','both']:
        noise=np.r_[np.log([.4,40.,.1,.15,7.,.06]),[.4]*(int(mode in ['process','both'])+int(mode in ['measurement','both']))]
        S,R,ell=covariance(noise,mode);value,g,z,pred=filter_gaussian(time,residual,S,R,ell)
        C=dense_covariance(time,noise,mode);L=cholesky(C,lower=True);zz=solve_triangular(L,residual.ravel(),lower=True)
        dense=.5*(zz@zz+2*np.log(np.diag(L)).sum()+len(zz)*np.log(2*np.pi))
        gradient=np.linalg.solve(C,residual.ravel()).reshape(-1,2)
        error=max(abs(value-dense),np.max(abs(g-gradient)),np.max(abs(z.ravel()-zz)));assert error<1e-9,(mode,error)
        records=[]
        for d in load_records()[:2]:
            v=dict(d);v.update({key:d[key][:31].copy() for key in ['t','u','y']});records.append(v)
        m=Model(records,.1,mode);p=np.r_[fit['best']['point'][:6],[-.02,.02],np.log(.85),[.2,-.3],noise]
        analytic=m.gradient(p);errors=[];h=2e-5
        for j in range(len(p)):
            v=np.zeros(len(p));v[j]=h;fd=(m.objective(p+v)-m.objective(p-v))/(2*h)
            errors.append(abs(analytic[j]-fd)/(1+abs(fd)))
        assert max(errors)<1e-4,(mode,max(errors))
        rows.append({'mode':mode,'dense_value_score_innovation_max_error':float(error),'joint_coordinate_score_max_scaled_error':float(max(errors))})
    write(args.output/'check.json',{'status':'passed','checks':rows,
        'scope':'Explicit unequal-time full observation covariance and dense Cholesky, reverse filter mean score and all joint coordinates. Continuous-Gaussian approximation only; does not validate quantized multichannel filtering.'})
    print(json.dumps({'status':'passed','checks':rows}),flush=True)


if __name__=='__main__':main()
