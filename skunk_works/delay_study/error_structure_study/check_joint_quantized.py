"""Full physical-plus-observation score checks on held-input synthetic records."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from quantized import report
from joint_quantized import Model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);args=parser.parse_args()
    fit=json.loads(args.warm.read_text());rng=np.random.RandomState(813331);rows=[]
    for family,band in [('conductance',.1),('radiation',.1),('conductance',0.),('radiation',0.)]:
        records=[]
        for d in load_records()[:3]:
            v=dict(d)
            for name in ['t','u','y']:v[name]=d[name][:61].copy()
            records.append(v)
        m=Model(records,band=band,nodes=128,family=family);shape=np.asarray(fit['best']['point'][:6]).copy();shape[2]=np.log(.03)
        if family=='conductance':mean=np.r_[shape,[-.02,0.,.02] if band else np.zeros(0),np.log(.85),[.3,-.5,.7]]
        else:mean=np.r_[shape,np.log(.85),.5,[-.02,0.,.02] if band else np.zeros(3),[.3,-.5,.7]]
        noise=np.log([.3,30.,.05,.1,10.,.03]);point=np.r_[mean,noise];truth=m.mean(point,False);cursor=0;generated=[]
        for j,d in enumerate(records):
            v=dict(d);y=d['y'].copy();n=len(d['t'])-1
            for k in [0,1]:
                latent,ell,measurement=np.exp(noise[3*k:3*k+3]);x=rng.normal(0,latent)
                for i in range(n):
                    if i:
                        a=np.exp(-(d['t'][i+1]-d['t'][i])/ell);x=a*x+rng.normal(0,latent*np.sqrt(1-a*a))
                    y[i+1,k]=report(truth[k*m.N+cursor+i]+x+rng.normal(0,measurement))
            v['y']=y;generated.append(v);cursor+=n
        model=Model(generated,band=band,nodes=128,family=family);gradient=model.gradient(point);checks=[]
        # Independent finite differences of the marginal bin likelihood,
        # including shape, per-run deviation, ambient/gain and covariance.
        from quantized import filter_ou
        def nll(p):
            mu=model.mean(p,False);value=0.;cursor=0
            for c in [0,1]:
                latent,ell,measurement=np.exp(p[model.nmean+3*c:model.nmean+3*c+3]);cursor=0
                for d in generated:
                    n=len(d['t'])-1;section=slice(c*model.N+cursor,c*model.N+cursor+n)
                    value+=filter_ou(d['t'][1:],mu[section],d['y'][1:,c],np.sqrt(latent*latent+measurement*measurement),ell,measurement*measurement/(latent*latent+measurement*measurement),nodes=128)['nll'];cursor+=n
            return value
        assert abs(nll(point)-model.objective(point))<1e-9
        h=2e-5
        for j in range(len(point)):
            d=np.zeros(len(point));d[j]=h
            fd=(nll(point+d)-nll(point-d))/(2*h)
            error=abs(fd-gradient[j])/(1+abs(fd));checks.append(error)
        assert max(checks)<2e-3,(family,max(checks))
        rows.append({'family':family,'band':band,'coordinates_checked':len(point),'maximum_scaled_score_error':max(checks),
            'normalization_error':max(r['normalization_error'] for r in model.details(point)),
            'smoothing_closure_error':max(r['smoothing_closure_error'] for r in model.details(point))})
    result={'status':'passed','rows':rows,'scope':'All mean and covariance coordinate scores against finite differences of the marginal bin likelihood on three 60-observation synthetic held-input records, for linear-conductance and radiative families. Does not establish full hardware optimum or practical identifiability.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
