"""Tail and two-implementation quadrature/score checks before accelerated use."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import log_ndtr
from quantized import report,log_probability
from smoothing import smooth_ou as reference
from fast_smoothing import smooth_ou as compiled,normal_logcdf,interval_logmass
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    values=np.r_[np.linspace(-100,100,2001),[-1e6,-1000.,-40.,-30.00001,-30.,0.,30.,40.,1000.]]
    got=np.array([normal_logcdf(x) for x in values]);target=log_ndtr(values)
    scaled=float(np.max(abs(got-target)/(1+abs(target))));assert scaled<2e-14,scaled
    intervals=[(-40.,-39.999),(40.,40.001),(-1000.,-999.9999),(1000.,1000.0001),(-1e6,-1e6+.3),(-.0001,.0001),(-3.,4.)]
    interval_checks=[]
    for a,b in intervals:
        q=interval_logmass(a,b);r=float(log_probability(a,b));error=abs(q-r)/(1+abs(r));assert error<5e-10,(a,b,error)
        interval_checks.append({'a':a,'b':b,'compiled':q,'reference':r,'scaled_error':error})
    rng=np.random.RandomState(482133);timepoints=np.cumsum(rng.uniform(.8,1.2,150));mu=25+.4*np.sin(timepoints/8);rows=[]
    for phase in [-.5,0.]:
        for latent,ell,measurement in [(.3,7.,.05),(.536,658.,.025),(.068,62.,.032),(.3,10.,.005)]:
            state=rng.normal(0,latent);values=[]
            for i,t in enumerate(timepoints):
                if i:
                    a=np.exp(-(t-timepoints[i-1])/ell);state=a*state+rng.normal(0,latent*np.sqrt(1-a*a))
                values.append(mu[i]+state+rng.normal(0,measurement))
            y=report(values,phase)
            start=time.monotonic();a=reference(timepoints,mu,y,latent,ell,measurement,phase,128);oldtime=time.monotonic()-start
            start=time.monotonic();b=compiled(timepoints,mu,y,latent,ell,measurement,phase,128);newtime=time.monotonic()-start
            errors={key:float(np.max(abs(np.asarray(a[key])-np.asarray(b[key])))) for key in [
                'log_bin_probability','mean_nll_gradient','noise_nll_gradient_logsd_logell_logsd','posterior_normalization_ratio',
                'smoothed_latent_mean','smoothed_latent_second','smoothed_lag1_cross','smoothed_measurement_mean','smoothed_measurement_second']}
            assert abs(a['nll']-b['nll'])<1e-7 and max(errors.values())<2e-5,(latent,ell,measurement,errors)
            rows.append({'phase':phase,'noise':[latent,ell,measurement],'nll_abs_error':abs(a['nll']-b['nll']),
                'max_abs_errors':errors,'reference_s':oldtime,'compiled_s_including_first_JIT':newtime,
                'reference_normalization_error':float(np.max(abs(a['posterior_normalization_ratio']-1))),
                'compiled_normalization_error':float(np.max(abs(b['posterior_normalization_ratio']-1))),
                'compiled_smoothing_closure':b['smoothing_closure_error']})
    t=np.array([0.,1.]);mean=np.array([25.,25.]);y=report([25.,25.75]);a=reference(t,mean,y,.2,700.,.025,nodes=512);b=compiled(t,mean,y,.2,700.,.025,nodes=512)
    rare={'nll_abs_error':abs(a['nll']-b['nll']),
        'mean_score_max_error':float(np.max(abs(a['mean_nll_gradient']-b['mean_nll_gradient']))),
        'noise_score_max_error':float(np.max(abs(a['noise_nll_gradient_logsd_logell_logsd']-b['noise_nll_gradient_logsd_logell_logsd'])))}
    assert max(rare.values())<2e-5,rare
    write(args.output/'check.json',{'status':'passed','normal_logcdf_max_scaled_error':scaled,'intervals':interval_checks,
        'sequences':rows,'rare_two_point_case':rare,'scope':'Independent SciPy tails, two implementations of all retained quadrature moments/Fisher scores over eight dependent sequences and a rare two-code jump. Separate direct Gaussian moment/score and full-data joint benchmarks remain required.'})
    print(json.dumps({'status':'passed','maximum_sequence_score_error':max(max(r['max_abs_errors'].values()) for r in rows),'rare':rare}),flush=True)


if __name__=='__main__':main()
