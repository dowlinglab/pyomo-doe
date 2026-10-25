"""Gaussian bin probabilities and deterministic OU measurement filtering.

Hardware conversion uses a shifted reported-code grid. Bin phase remains an
assumption: -0.5 is nearest-code and 0 is a floor-code convention. The filter
integrates latent OU states with Gauss-Legendre quadrature; every update reports
its posterior normalization error against the direct predictive-bin mixture.
"""
import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr
from numpy.polynomial.legendre import leggauss

STEP=0.3223
OFFSET=-50.
LOG2PI=np.log(2*np.pi)


def log_probability(a,b):
    """Log standard-normal mass in [a,b], including positive-tail stability."""
    a,b=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    if np.any(b<=a):raise ValueError('Bin upper bound must exceed lower bound')
    right=a>=0
    upper=np.where(right,log_ndtr(-a),log_ndtr(b))
    lower=np.where(right,log_ndtr(-b),log_ndtr(a))
    return upper+np.log(-np.expm1(lower-upper))


def bins(reported,phase=-.5):
    y=np.asarray(reported,dtype=float)
    codes=np.rint((y-OFFSET)/STEP)
    if not np.all(abs(np.round(STEP*codes+OFFSET,2)-y)<1e-9):
        raise ValueError('Observation does not lie on the recorded hardware grid')
    lo=STEP*(codes+phase)+OFFSET
    return lo,lo+STEP


def report(analogue,phase=-.5):
    n=np.floor((np.asarray(analogue)-OFFSET)/STEP-phase)
    return np.round(STEP*n+OFFSET,2)


def filter_ou(time,mean,observed,sigma,ell,eta,phase=-.5,nodes=128,tail_sd=9.):
    """Return predictive bin probabilities and an approximate latent posterior.

    Latent X has variance sigma^2*(1-eta) and OU length ell; independent analogue
    noise has variance sigma^2*eta. Predictive observation mixtures integrate
    transition plus measurement noise analytically. Only the latent posterior
    update uses quadrature. No process clipping or simulator noise is assumed.
    """
    time=np.asarray(time);mean=np.asarray(mean);observed=np.asarray(observed)
    if not len(time)==len(mean)==len(observed):raise ValueError('Unequal lengths')
    if sigma<=0 or ell<=0 or not 0<=eta<=1:raise ValueError('Invalid covariance')
    if np.any(np.diff(time)<=0):raise ValueError('Time must strictly increase')
    lo,hi=bins(observed,phase);low=lo-mean;high=hi-mean
    if eta==1:
        lp=log_probability(low/sigma,high/sigma)
        return {'nll':float(-lp.sum()),'log_bin_probability':lp,'cdf_lower':ndtr(low/sigma),
                'cdf_upper':ndtr(high/sigma),'log_CDF_lower':log_ndtr(low/sigma),'log_SF_upper':log_ndtr(-high/sigma),'posterior_normalization_ratio':np.ones(len(time)),
                'posterior_mean':np.zeros(len(time)),'posterior_SD':np.zeros(len(time))}
    points,weights=leggauss(nodes)
    slow=sigma*sigma*(1-eta);measurement=sigma*np.sqrt(eta)
    previous=None;mass=None;logp=[];cdflo=[];cdfhi=[];normalization=[];pm=[];ps=[];logcdf=[];logsf=[]
    for k in range(len(time)):
        if k==0:
            lp=float(log_probability(low[k]/sigma,high[k]/sigma))
            cdflo.append(float(ndtr(low[k]/sigma)));cdfhi.append(float(ndtr(high[k]/sigma)))
            logcdf.append(float(log_ndtr(low[k]/sigma)));logsf.append(float(log_ndtr(-high[k]/sigma)))
            centers=np.array([0.]);mix=np.array([1.]);variance=slow
        else:
            alpha=np.exp(-(time[k]-time[k-1])/ell)
            variance=slow*(-np.expm1(-2*(time[k]-time[k-1])/ell))
            centers=alpha*previous;mix=mass
            total_sd=np.sqrt(variance+measurement*measurement)
            probabilities=log_probability((low[k]-centers)/total_sd,(high[k]-centers)/total_sd)
            lp=float(logsumexp(np.log(mix)+probabilities))
            cdflo.append(float(mix@ndtr((low[k]-centers)/total_sd)))
            cdfhi.append(float(mix@ndtr((high[k]-centers)/total_sd)))
            logcdf.append(float(logsumexp(np.log(mix)+log_ndtr((low[k]-centers)/total_sd))))
            logsf.append(float(logsumexp(np.log(mix)+log_ndtr((centers-high[k])/total_sd))))
        # At eta=0 the latent posterior is confined exactly to the observation
        # interval; eta>0 is truncated in measurement-noise SD units, then
        # checked by integrating back against the predictive-bin probability.
        lower=low[k]-tail_sd*measurement;upper=high[k]+tail_sd*measurement
        center=float(mix@centers)
        spread=np.sqrt(variance+mix@((centers-center)**2))
        plower=center-tail_sd*spread;pupper=center+tail_sd*spread
        if max(lower,plower)<min(upper,pupper):
            lower=max(lower,plower);upper=min(upper,pupper)
        elif measurement:
            lower=min(lower,plower);upper=max(upper,pupper)
        x=(upper+lower)/2+(upper-lower)/2*points
        qw=weights*(upper-lower)/2
        lpredict=logsumexp(np.log(mix)[None,:]-.5*((x[:,None]-centers[None,:])**2/variance+np.log(variance)+LOG2PI),axis=1)
        if measurement:
            lmeasure=log_probability((low[k]-x)/measurement,(high[k]-x)/measurement)
        else:lmeasure=np.zeros(nodes)
        unnormalized=lpredict+lmeasure+np.log(qw)
        integral=float(logsumexp(unnormalized))
        normalization.append(float(np.exp(integral-lp)))
        mass=np.exp(unnormalized-integral)
        # Keep strictly positive mixture weights without moving the normalized
        # distribution at floating-point precision; discard underflowed atoms.
        take=mass>0;previous=x[take];mass=mass[take];mass/=mass.sum()
        m=float(mass@previous);pm.append(m);ps.append(float(np.sqrt(mass@((previous-m)**2))))
        logp.append(lp)
    return {'nll':float(-np.sum(logp)),'log_bin_probability':np.asarray(logp),
            'cdf_lower':np.asarray(cdflo),'cdf_upper':np.asarray(cdfhi),
            'log_CDF_lower':np.asarray(logcdf),'log_SF_upper':np.asarray(logsf),
            'posterior_normalization_ratio':np.asarray(normalization),
            'posterior_mean':np.asarray(pm),'posterior_SD':np.asarray(ps)}


def randomized_residual(result,seed):
    """Randomized PIT transformed to Gaussian; seed and repetitions are reported."""
    from scipy.special import ndtri
    from scipy.optimize import brentq
    rng=np.random.RandomState(seed)
    draw=rng.uniform(size=len(result['log_bin_probability']))
    left=np.logaddexp(result['log_CDF_lower'],np.log(draw)+result['log_bin_probability'])
    right=np.logaddexp(result['log_SF_upper'],np.log1p(-draw)+result['log_bin_probability'])
    def inverse(logp):
        if logp>-700:return float(ndtri(np.exp(logp)))
        return float(brentq(lambda x:float(log_ndtr(x))-logp,-max(50.,2*np.sqrt(-2*logp)),0.))
    z=np.array([inverse(l) if l<np.log(.5) else -inverse(r) for l,r in zip(left,right)])
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite PIT; inspect predictive tails')
    return z
