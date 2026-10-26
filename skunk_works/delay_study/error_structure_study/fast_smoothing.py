"""Compiled implementation of the reference scalar quadrature Fisher score.

No quadrature order, posterior support, tail cutoff, latent law or likelihood
definition is changed. The NumPy/SciPy implementation remains the independent
reference. This kernel retains zero-weight grid locations instead of dropping
them; zero weights contribute neither forward mixtures nor backward pairs.
"""
from functools import lru_cache
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from numba import njit
from quantized import bins

LOG2PI=math.log(2*math.pi)
SQRT2=math.sqrt(2.)


@njit(cache=True)
def normal_logcdf(x):
    if x>0:return math.log1p(-.5*math.erfc(x/SQRT2))
    if x>=-30:return math.log(.5*math.erfc(-x/SQRT2))
    z=1/(x*x);term=1.;series=1.
    for k in range(1,13):
        term*=-(2*k-1)*z;series+=term
    return -.5*x*x-.5*LOG2PI-math.log(-x)+math.log(series)


@njit(cache=True)
def interval_logmass(a,b):
    if b<=a:raise ValueError('Invalid normal interval')
    if a>=0:upper=normal_logcdf(-a);lower=normal_logcdf(-b)
    else:upper=normal_logcdf(b);lower=normal_logcdf(a)
    return upper+math.log(-math.expm1(lower-upper))


@njit(cache=True)
def logsum(values):
    peak=np.max(values);total=0.
    for value in values:total+=math.exp(value-peak)
    return peak+math.log(total)


@njit(cache=True)
def kernel(time,mean,lo,hi,latent_sd,ell,measurement_sd,points,legendre_weights,tail_sd):
    n=len(time);nodes=len(points);S=latent_sd*latent_sd;R=measurement_sd*measurement_sd
    grids=np.empty((n,nodes));weights=np.empty((n,nodes));prior_log=np.empty((n,nodes))
    measurement_mean=np.zeros((n,nodes));measurement_second=np.zeros((n,nodes))
    normalization=np.empty(n);logp=np.empty(n);post_mean=np.empty(n);post_sd=np.empty(n)
    centers=np.zeros(nodes);logmix=np.full(nodes,-np.inf);logmix[0]=0.;mix=np.zeros(nodes);mix[0]=1.
    for k in range(n):
        low=lo[k]-mean[k];high=hi[k]-mean[k]
        if k==0:
            count=1;variance=S;lp=interval_logmass(low/math.sqrt(S+R),high/math.sqrt(S+R));component=np.zeros(1)
        else:
            count=nodes;a=math.exp(-(time[k]-time[k-1])/ell);variance=S*(-math.expm1(-2*(time[k]-time[k-1])/ell))
            total_sd=math.sqrt(variance+R);component=np.empty(nodes)
            for j in range(nodes):
                centers[j]=a*grids[k-1,j];mix[j]=weights[k-1,j]
                logmix[j]=math.log(mix[j]) if mix[j]>0 else -np.inf
                component[j]=logmix[j]+interval_logmass((low-centers[j])/total_sd,(high-centers[j])/total_sd)
            lp=logsum(component);component-=lp
        gain=variance/(variance+R);conditional_sd=math.sqrt(variance*R/(variance+R))
        lower=np.inf;upper=-np.inf
        for j in range(count):
            if component[j]>-40.:
                lower=min(lower,centers[j]+gain*(low-centers[j]));upper=max(upper,centers[j]+gain*(high-centers[j]))
        lower-=tail_sd*conditional_sd;upper+=tail_sd*conditional_sd
        logpost=np.empty(nodes);half=(upper-lower)/2;mid=(upper+lower)/2
        for i in range(nodes):
            x=mid+half*points[i];grids[k,i]=x;terms=np.empty(count)
            for j in range(count):terms[j]=logmix[j]-.5*((x-centers[j])**2/variance+math.log(variance)+LOG2PI)
            prior_log[k,i]=logsum(terms)
            lm=interval_logmass((low-x)/measurement_sd,(high-x)/measurement_sd)
            logpost[i]=prior_log[k,i]+lm+math.log(legendre_weights[i]*half)
        integral=logsum(logpost);normalization[k]=math.exp(integral-lp);logp[k]=lp
        mass=np.exp(logpost-integral);mass/=mass.sum();weights[k]=mass
        m=0.;second=0.
        for i in range(nodes):
            x=grids[k,i];m+=mass[i]*x;second+=mass[i]*x*x
            if mass[i]>0:
                a=(low-x)/measurement_sd;b=(high-x)/measurement_sd;lb=interval_logmass(a,b)
                ra=math.exp(-.5*a*a-.5*LOG2PI-lb);rb=math.exp(-.5*b*b-.5*LOG2PI-lb)
                measurement_mean[k,i]=measurement_sd*(ra-rb)
                v2=R*(1+a*ra-b*rb)
                if v2<-1e-10:raise FloatingPointError('Negative truncated Gaussian second moment')
                measurement_second[k,i]=max(v2,0.)
        post_mean[k]=m;post_sd[k]=math.sqrt(max(second-m*m,0.))
    smooth=np.empty((n,nodes));smooth[-1]=weights[-1];cross=np.empty(max(n-1,0));closure=0.
    for k in range(n-2,-1,-1):
        a=math.exp(-(time[k+1]-time[k])/ell);q=S*(-math.expm1(-2*(time[k+1]-time[k])/ell))
        pair=np.full((nodes,nodes),-np.inf);peak=-np.inf
        for i in range(nodes):
            if smooth[k+1,i]<=0:continue
            for j in range(nodes):
                if weights[k,j]<=0:continue
                value=math.log(smooth[k+1,i])+math.log(weights[k,j])-.5*((grids[k+1,i]-a*grids[k,j])**2/q+math.log(q)+LOG2PI)-prior_log[k+1,i]
                pair[i,j]=value;peak=max(peak,value)
        total=0.
        for i in range(nodes):
            for j in range(nodes):total+=math.exp(pair[i,j]-peak)
        normalizer=peak+math.log(total);closure=max(closure,abs(math.expm1(normalizer)))
        old=np.zeros(nodes);moment=0.
        for i in range(nodes):
            row=0.
            for j in range(nodes):
                value=math.exp(pair[i,j]-normalizer);old[j]+=value;row+=value
                moment+=grids[k+1,i]*value*grids[k,j]
            closure=max(closure,abs(row-smooth[k+1,i]))
        smooth[k]=old;cross[k]=moment
    ex=np.empty(n);ex2=np.empty(n);ev=np.empty(n);ev2=np.empty(n)
    for k in range(n):
        ex[k]=smooth[k]@grids[k];ex2[k]=smooth[k]@(grids[k]**2)
        ev[k]=smooth[k]@measurement_mean[k];ev2[k]=smooth[k]@measurement_second[k]
    process=.5*(ex2[0]/S-1);length=0.
    for k in range(n-1):
        dt=time[k+1]-time[k];a=math.exp(-dt/ell);q=S*(-math.expm1(-2*dt/ell))
        moment=ex2[k+1]-2*a*cross[k]+a*a*ex2[k]
        if moment<-1e-10:raise FloatingPointError('Negative smoothed process second moment')
        process+=.5*(moment/q-1);da=a*dt/ell;dlogq=-2*a*da/(1-a*a)
        length+=da/q*(cross[k]-a*ex2[k])+.5*dlogq*(moment/q-1)
    measurement=.5*(ev2.sum()/R-n);noise=-np.array([2*process,length,2*measurement])
    return -logp.sum(),-ev/R,noise,normalization,closure,logp,ex,ex2,cross,ev,ev2,post_mean,post_sd


@lru_cache(maxsize=8)
def rule(nodes):return leggauss(nodes)


def smooth_ou(time,mean,observed,latent_sd,ell,measurement_sd,phase=-.5,nodes=128,tail_sd=12.):
    if min(latent_sd,ell,measurement_sd)<=0:raise ValueError('Positive variances required')
    time=np.asarray(time,dtype=float);mean=np.asarray(mean,dtype=float);observed=np.asarray(observed,dtype=float)
    if not len(time)==len(mean)==len(observed) or not len(time) or np.any(np.diff(time)<=0):raise ValueError('Invalid time/observation arrays')
    lo,hi=bins(observed,phase);points,weights=rule(nodes)
    values=kernel(time,mean,lo,hi,float(latent_sd),float(ell),float(measurement_sd),points,weights,float(tail_sd))
    names=['nll','mean_nll_gradient','noise_nll_gradient_logsd_logell_logsd','posterior_normalization_ratio',
        'smoothing_closure_error','log_bin_probability','smoothed_latent_mean','smoothed_latent_second',
        'smoothed_lag1_cross','smoothed_measurement_mean','smoothed_measurement_second','posterior_mean','posterior_SD']
    return dict(zip(names,values))


def smooth_record(record,mean,channel,latent,ell,measurement,phase,nodes,condition=False):
    if not condition:
        f=smooth_ou(record['t'][1:],mean[1:],record['y'][1:,channel],latent,ell,measurement,phase,nodes)
        f['initial_mean_nll_gradient']=0.;return f
    f=smooth_ou(record['t'],mean,record['y'][:,channel],latent,ell,measurement,phase,nodes)
    first=smooth_ou(record['t'][:1],mean[:1],record['y'][:1,channel],latent,ell,measurement,phase,nodes)
    f['nll']-=first['nll'];f['initial_mean_nll_gradient']=float(f['mean_nll_gradient'][0]-first['mean_nll_gradient'][0])
    f['mean_nll_gradient']=f['mean_nll_gradient'][1:]
    f['noise_nll_gradient_logsd_logell_logsd']-=first['noise_nll_gradient_logsd_logell_logsd']
    return f
