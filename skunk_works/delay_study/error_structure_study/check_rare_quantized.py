"""Posterior-domain checks for rare bins and the preserved zero-band seed."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.integrate import nquad
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from physical import load_records
from joint_quantized import Model
from quantized import report,bins,filter_ou


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--failed',type=Path,required=True);args=parser.parse_args()
    t=np.array([0.,1.]);mu=np.array([25.,25.]);y=report([25.,25.75]);latent=.2;ell=700.;measurement=.025
    S=latent*latent;R=measurement*measurement;C=S*np.exp(-abs(t[:,None]-t[None,:])/ell)+R*np.eye(2)
    density=multivariate_normal(cov=C);lo,hi=bins(y);lo-=mu;hi-=mu;precision=np.linalg.inv(C)
    mode=minimize(lambda z:(float(z@precision@z)/2,precision@z),np.clip(np.zeros(2),lo,hi),jac=True,
        method='L-BFGS-B',bounds=list(zip(lo,hi)),options={'gtol':1e-12,'ftol':1e-15}).x
    anchor=density.logpdf(mode)
    mass,error=nquad(lambda x,z:np.exp(density.logpdf([x,z])-anchor),list(zip(lo,hi)),opts={'epsabs':1e-12,'epsrel':1e-10})
    reference=-anchor-np.log(mass);rare=[]
    for tail in [9.,12.,16.]:
        f=filter_ou(t,mu,y,np.sqrt(S+R),ell,R/(S+R),nodes=512,tail_sd=tail)
        rare.append({'tail_SD':tail,'bin_nll':f['nll'],'direct_rectangle_nll_error':abs(f['nll']-reference),
                     'normalization_error':float(np.max(abs(f['posterior_normalization_ratio']-1)))})
    assert rare[1]['direct_rectangle_nll_error']<1e-5 and rare[2]['direct_rectangle_nll_error']<1e-5,rare
    starts=json.loads((args.failed/'starts.json').read_text());point=np.array(starts[0]['seed']);records=load_records();model=Model(records,0.)
    mean=model.mean(point,False);rows=[]
    for j in [2,6,9,10]:
        c=1;d=records[j];start=sum(len(x['t'])-1 for x in records[:j]);n=len(d['t'])-1
        section=slice(model.N+start,model.N+start+n);latent,ell,measurement=np.exp(point[model.nmean+3:model.nmean+6])
        values={}
        for tail,nodes in [(12.,256),(16.,512)]:
            values[tail]=filter_ou(d['t'][1:],mean[section],d['y'][1:,c],np.hypot(latent,measurement),ell,measurement**2/(latent**2+measurement**2),nodes=nodes,tail_sd=tail)
        difference=abs(values[12.]['nll']-values[16.]['nll']);norm=max(float(np.max(abs(v['posterior_normalization_ratio']-1))) for v in values.values())
        assert difference<1e-4 and norm<1e-6,(d['name'],difference,norm)
        rows.append({'name':d['name'],'12_vs_16_SD_nll_difference':difference,'normalization_error':norm})
    result={'status':'passed','rare_two_point_checks':rare,'preserved_zero_band_seed':rows,
        'scope':'Rare bivariate observation rectangle against scaled independent integration, posterior normalization and 12/16-SD support convergence at four problematic zero-band record seeds. Does not validate arbitrary extreme trial parameters or establish hardware adequacy.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
