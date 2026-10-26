"""Independent conditional Gaussian rectangles check down/same/up forecasts."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from quantized import bins,filter_ou,report
from bin_diagnostics import transition_probabilities


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();rows=[]
    for phase in [-.5,0.]:
        t=np.array([1.,2.3]);mean=np.array([20.05,20.2]);y=report(np.array([20.1,20.5]),phase)
        latent=.14;measurement=.04;ell=4.3;sigma=np.hypot(latent,measurement);eta=measurement**2/sigma**2
        f=filter_ou(t,mean,y,sigma,ell,eta,phase,256,12.,True)
        predicted=transition_probabilities(t,mean,np.r_[y[0],y[:-1]],latent,ell,measurement,f,phase)[1]
        lo,hi=bins(y[:1],phase);cov=latent**2*np.exp(-(t[1]-t[0])/ell)
        sd=np.sqrt(sigma**2-cov**2/sigma**2);mass=norm.cdf((hi[0]-mean[0])/sigma)-norm.cdf((lo[0]-mean[0])/sigma)
        def joint(boundary):
            return quad(lambda z:norm.pdf(z,mean[0],sigma)*norm.cdf((boundary-mean[1]-cov/sigma**2*(z-mean[0]))/sd),
                        lo[0],hi[0],epsabs=1e-13,epsrel=1e-12)[0]/mass
        a=joint(lo[0]);b=joint(hi[0]);reference=np.array([a,b-a,1-b]);error=float(np.max(abs(reference-predicted)))
        assert error<1e-10,(phase,error)
        rows.append({'phase':phase,'conditional_probability_max_error':error,'probabilities':predicted.tolist()})
    result={'status':'passed','cases':rows,'scope':'Independent bivariate Gaussian observation integral conditional on the first code. Checks previous filtering state and bin boundaries; no whitening adequacy or simulation coverage claim.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
