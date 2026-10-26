"""Independent bivariate observation-bin forecasts and no-future-code test."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm
from quantized import bins,report,STEP,OFFSET
from bin_diagnostics import filter_records
from updated_diagnostics import forecast_codes
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    phase=-.5;means=np.array([[22.1,22.3],[22.2,22.4],[22.7,22.6]])
    d={'name':'synthetic','t':np.array([0.,1.1,2.7]),'u':np.array([50.,60.,60.]),
        'y':report(means+np.array([[0.,0.],[.13,-.05],[.11,.01]]),phase)}
    noise=[{'latent_SD_C':.3,'ell_s':10.,'measurement_SD_C':.04},
           {'latent_SD_C':.08,'ell_s':3.,'measurement_SD_C':.03}]
    f=filter_records([d],[means],noise,phase,256)[0];rows=forecast_codes(d,means,noise,f,phase,horizons=(0.,1.))
    checks=[]
    for k,p in enumerate(noise):
        variance=p['latent_SD_C']**2+p['measurement_SD_C']**2;sd=np.sqrt(variance)
        covariance=p['latent_SD_C']**2*np.exp(-(d['t'][2]-d['t'][1])/p['ell_s'])
        gain=covariance/variance;conditional_sd=np.sqrt(variance-covariance*gain)
        low,high=bins(d['y'][1:2,k],phase);lo=low[0]-means[1,k];hi=high[0]-means[1,k]
        den=ndtr(hi/sd)-ndtr(lo/sd);n=np.arange(180,280);levels=np.round(STEP*n+OFFSET,2)
        lower=STEP*(n+phase)+OFFSET;upper=lower+STEP
        probability=np.array([quad(lambda x:norm.pdf(x/sd)/sd*(ndtr((b-means[2,k]-gain*x)/conditional_sd)-ndtr((a-means[2,k]-gain*x)/conditional_sd)),lo,hi,epsabs=1e-12,epsrel=1e-11)[0]/den for a,b in zip(lower,upper)])
        assert abs(probability.sum()-1)<1e-10
        exact=float(levels@probability);short=next(r for r in rows if r['channel']==k+1 and r['cutoff_s']==0 and r['horizon_s']==1.)
        got=float(d['y'][2,k]-short['mean_error_C']);error=abs(exact-got);assert error<2e-9,(k,error)
        marginal=np.array([ndtr((b-means[1,k])/sd)-ndtr((a-means[1,k])/sd) for a,b in zip(lower,upper)])
        first=float(levels@marginal);sse=(d['y'][1,k]-first)**2+(d['y'][2,k]-exact)**2
        one=next(r for r in rows if r['channel']==k+1 and r['cutoff_s']==0 and r['horizon_s']==0.)
        assert abs(one['SSE_C2']-sse)<1e-9
        checks.append({'channel':k+1,'bivariate_bin_forecast_error_C':error,'one_step_SSE_error':abs(one['SSE_C2']-sse)})
    changed=dict(d,y=d['y'].copy());changed['y'][2,:]=report(changed['y'][2,:]+3*STEP,phase)
    newf=filter_records([changed],[means],noise,phase,256)[0];newrows=forecast_codes(changed,means,noise,newf,phase,horizons=(1.,))
    for k in [0,1]:
        old=next(r for r in rows if r['channel']==k+1 and r['cutoff_s']==0 and r['horizon_s']==1.)
        new=next(r for r in newrows if r['channel']==k+1 and r['cutoff_s']==0)
        assert abs((d['y'][2,k]-old['mean_error_C'])-(changed['y'][2,k]-new['mean_error_C']))<1e-12
    write(args.output/'check.json',{'status':'passed','checks':checks,'future_code_leakage_error_C':0.,
        'scope':'Two scored observations, exact bivariate Gaussian observation-bin integration, marginal code expectation, aggregate one-step SSE and future-output perturbation. Does not validate a fitted physical model.'})
    print(json.dumps({'status':'passed','checks':checks}),flush=True)


if __name__=='__main__':main()
