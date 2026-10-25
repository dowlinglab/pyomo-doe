"""Independent Gaussian rectangle integrals and conditional-PIT recovery tests."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad, nquad
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from quantized import STEP,OFFSET,bins,report,log_probability,filter_ou,randomized_residual


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cases=[]
    for a,b in [(-50.,-49.9),(-10.,-9.99),(-1.,1.),(8.,8.0001),(0.1,.100000001),(40.,40.2)]:
        mode=np.clip(0.,a,b)
        integral=quad(lambda v:np.exp(-.5*(v*v-mode*mode)),a,b,epsabs=1e-13,epsrel=1e-12)[0]
        reference=-.5*(mode*mode+np.log(2*np.pi))+np.log(integral)
        error=abs(float(log_probability(a,b))-reference)
        assert error<5e-6,(a,b,error)
        cases.append({'a':a,'b':b,'log_probability_abs_error':error})
    for x in [-3.,0.,2.]:
        h=.0001
        error=abs(float(log_probability(x-h/2,x+h/2))-np.log(h)+.5*(x*x+np.log(2*np.pi)))
        assert error<1e-7,error
    analogue=np.linspace(20.,70.,1001)
    for phase in [-.5,0.]:
        lo,hi=bins(report(analogue,phase),phase)
        assert np.all(analogue>=lo-1e-12) and np.all(analogue<=hi+1e-12)
    codes=np.arange(-1000,1500);lo=STEP*codes+OFFSET;hi=lo+STEP
    for sigma in [.05,.4,2.]:
        total=np.exp(logsumexp(log_probability((lo-25)/sigma,(hi-25)/sigma)))
        assert abs(total-1)<1e-12,total
    # Integrate the bivariate *observation* density directly, not the filter's
    # latent-state recurrence. This covariance includes the independent nugget.
    rectangles=[]
    for eta in [0.,.1,.8,.99999,1.]:
        for phase in [-.5,0.]:
            t=np.array([0.,1.7]);mean=np.array([25.,25.2]);y=report([25.1,25.32],phase)
            low,high=bins(y,phase);low-=mean;high-=mean
            sigma=.4;ell=20.;rho=(1-eta)*np.exp(-1.7/ell);conditional=sigma*np.sqrt(1-rho*rho)
            ref=quad(lambda v:np.exp(-.5*(v/sigma)**2)/np.sqrt(2*np.pi)/sigma*
                np.exp(log_probability((low[1]-rho*v)/conditional,(high[1]-rho*v)/conditional)),
                low[0],high[0],epsabs=1e-13,epsrel=1e-11)[0]
            f=filter_ou(t,mean,y,sigma,ell,eta,phase,nodes=128)
            error=abs(f['nll']+np.log(ref));norm=float(np.max(abs(f['posterior_normalization_ratio']-1)))
            assert error<1e-7 and norm<1e-7,(eta,phase,error,norm)
            rectangles.append({'eta':eta,'phase':phase,'nll_abs_error':error,'normalization_error':norm})
    t=np.array([0.,2.,4.5]);mean=np.array([25.,25.1,25.3]);y=report([25.1,25.3,25.2])
    lo,hi=bins(y);lo-=mean;hi-=mean;sigma=.3;eta=.15;ell=10.
    C=sigma*sigma*((1-eta)*np.exp(-abs(t[:,None]-t[None,:])/ell)+eta*np.eye(3))
    density=multivariate_normal(cov=C)
    reference=nquad(lambda a,b,c:density.pdf([a,b,c]),list(zip(lo,hi)),opts={'epsabs':1e-11,'epsrel':1e-9})[0]
    f=filter_ou(t,mean,y,sigma,ell,eta,nodes=128)
    error3=abs(f['nll']+np.log(reference));assert error3<1e-7,error3
    # A fixed, known generating model tests the observation filter and PIT.
    # Parameters are not estimated here; this is not an inferential coverage test.
    rng=np.random.RandomState(718303);n=1024;t=np.cumsum(rng.uniform(.7,1.3,n))
    mean=25+.6*np.sin(2*np.pi*t/30);sigma=.4;eta=.1;ell=20.
    latent=np.empty(n);latent[0]=rng.normal(0,sigma*np.sqrt(1-eta))
    for j in range(1,n):
        a=np.exp(-(t[j]-t[j-1])/ell)
        latent[j]=a*latent[j-1]+rng.normal(0,sigma*np.sqrt((1-eta)*(1-a*a)))
    y=report(mean+latent+rng.normal(0,sigma*np.sqrt(eta),n))
    filters={k:filter_ou(t,mean,y,sigma,ell,eta,nodes=k) for k in [64,128,256]}
    convergence=abs(filters[128]['nll']-filters[256]['nll'])
    assert convergence<1e-6,convergence
    normalization=float(np.max(abs(filters[256]['posterior_normalization_ratio']-1)))
    assert normalization<1e-7,normalization
    z=randomized_residual(filters[256],10371)
    acf=[float(np.corrcoef(z[k:],z[:-k])[0,1]) for k in range(1,21)]
    assert abs(z.mean())<.15 and .8<z.std()<1.2 and max(abs(np.array(acf)))<.18,(z.mean(),z.std(),max(abs(np.array(acf))))
    result={'status':'passed','normal_mass_cases':cases,'two_point_rectangles':rectangles,
        'three_point_integral_nll_abs_error':error3,'synthetic_observations':n,
        'quadrature_nll':{str(k):f['nll'] for k,f in filters.items()},
        '128_vs_256_nll_abs_difference':convergence,'normalization_max_abs_error':normalization,
        'known_truth_PIT':{'mean':float(z.mean()),'SD':float(z.std()),'ACF_lags1to20':acf,'seed':10371},
        'scope':'Independent one-/two-/three-dimensional Gaussian integrals, bin arithmetic, extreme tails, IID endpoint, synthetic conditional-PIT calibration with known parameters. No hardware-noise adequacy or fitted-parameter coverage claim.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
