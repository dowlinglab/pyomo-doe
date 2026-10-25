"""Independent rectangle-integrated Gaussian posterior moments and score checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import nquad
from scipy.stats import multivariate_normal
from quantized import bins,report,filter_ou
from smoothing import smooth_ou


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    t=np.array([0.,2.3]);mean=np.array([25.,25.2]);y=report([25.05,25.28]);latent=.3;measurement=.05;ell=7.
    f=smooth_ou(t,mean,y,latent,ell,measurement,nodes=128)
    S=latent*latent*np.exp(-abs(t[:,None]-t[None,:])/ell);C=S+measurement**2*np.eye(2)
    B=S@np.linalg.inv(C);conditional=S-B@S;density=multivariate_normal(cov=C)
    lo,hi=bins(y);lo-=mean;hi-=mean
    def integrate(fun):return nquad(lambda a,b:density.pdf([a,b])*fun(np.array([a,b])),list(zip(lo,hi)),opts={'epsabs':1e-12,'epsrel':1e-10})[0]
    mass=integrate(lambda z:1.);errors=[]
    for k in [0,1]:
        expected=integrate(lambda z:(B@z)[k])/mass
        errors.append(abs(expected-f['smoothed_latent_mean'][k]))
        second=integrate(lambda z:conditional[k,k]+(B@z)[k]**2)/mass
        errors.append(abs(second-f['smoothed_latent_second'][k]))
        ev=integrate(lambda z:(z-B@z)[k])/mass
        errors.append(abs(ev-f['smoothed_measurement_mean'][k]))
        ev2=integrate(lambda z:conditional[k,k]+(z-B@z)[k]**2)/mass
        errors.append(abs(ev2-f['smoothed_measurement_second'][k]))
    cross=integrate(lambda z:conditional[0,1]+(B@z)[0]*(B@z)[1])/mass
    errors.append(abs(cross-f['smoothed_lag1_cross'][0]))
    assert max(errors)<1e-9,max(errors)
    h=2e-5;noise=np.log([latent,ell,measurement]);gradient=[]
    def nll(noise,mu=mean):
        s,l,v=np.exp(noise)
        return filter_ou(t,mu,y,np.sqrt(s*s+v*v),l,v*v/(s*s+v*v),nodes=128)['nll']
    for k in range(3):
        d=np.zeros(3);d[k]=h;gradient.append((nll(noise+d)-nll(noise-d))/(2*h))
    noise_error=float(np.max(abs(np.array(gradient)-f['noise_nll_gradient_logsd_logell_logsd'])))
    means=[]
    for k in [0,1]:
        d=np.zeros(2);d[k]=h;means.append((nll(noise,mean+d)-nll(noise,mean-d))/(2*h))
    mean_error=float(np.max(abs(np.array(means)-f['mean_nll_gradient'])))
    assert noise_error<2e-6 and mean_error<2e-6,(noise_error,mean_error)
    # Longer dependent sequence checks the full forward/backward recurrence.
    rng=np.random.RandomState(81331);t=np.cumsum(rng.uniform(.8,1.2,120));mu=25+.4*np.sin(t/8)
    x=np.empty(len(t));x[0]=rng.normal(0,latent)
    for k in range(1,len(t)):
        a=np.exp(-(t[k]-t[k-1])/ell);x[k]=a*x[k-1]+rng.normal(0,latent*np.sqrt(1-a*a))
    observed=report(mu+x+rng.normal(0,measurement,len(t)))
    g=smooth_ou(t,mu,observed,latent,ell,measurement,nodes=128)
    assert g['smoothing_closure_error']<1e-9
    assert np.all(g['smoothed_latent_second']>=g['smoothed_latent_mean']**2-1e-10)
    # A directional perturbation avoids mirroring a single-coordinate check.
    d=np.array([.3,-.4,.2]);vals=[]
    for sign in [-1,1]:
        s,l,v=np.exp(noise+sign*h*d)
        vals.append(filter_ou(t,mu,observed,np.sqrt(s*s+v*v),l,v*v/(s*s+v*v),nodes=128)['nll'])
    directional=abs((vals[1]-vals[0])/(2*h)-d@g['noise_nll_gradient_logsd_logell_logsd'])
    assert directional<2e-4,directional
    result={'status':'passed','independent_posterior_moment_max_error':max(errors),
        'mean_score_FD_error':mean_error,'noise_score_FD_error':noise_error,
        'long_sequence_directional_score_error':directional,'smoothing_closure_error':g['smoothing_closure_error'],
        'scope':'Independent bivariate Gaussian posterior moments integrated over quantization bins, mean/covariance score derivatives and a 120-point dependent sequence. No claim of hardware adequacy or all-parameter recovery.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
