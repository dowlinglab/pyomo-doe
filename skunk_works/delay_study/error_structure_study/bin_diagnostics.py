"""Conditional bin-transition predictions and run-aware innovation summaries."""
import numpy as np
from scipy.special import ndtr
from scipy.stats import kstest,skew,kurtosis
from scipy.linalg import solve_toeplitz
from quantized import bins,report,filter_ou,randomized_residual


def transition_probabilities(time,mean,previous_reported,latent_sd,ell,measurement_sd,filtered,phase):
    """Down/same/up probabilities relative to the preceding reported code.

    Use the *previous* filtering distribution. A smoothed state or the current
    observation's posterior would leak the observation being predicted.
    """
    lo,hi=bins(previous_reported,phase);S=latent_sd**2;R=measurement_sd**2;answer=[]
    for k in range(len(time)):
        if k==0:centers=np.array([0.]);weights=np.array([1.]);variance=S+R
        else:
            a=np.exp(-(time[k]-time[k-1])/ell);old=filtered['history'][k-1]
            centers=a*old['grid'];weights=old['weight'];variance=S*(-np.expm1(-2*(time[k]-time[k-1])/ell))+R
        lower=float(weights@ndtr((lo[k]-mean[k]-centers)/np.sqrt(variance)))
        upper=float(weights@ndtr((hi[k]-mean[k]-centers)/np.sqrt(variance)))
        answer.append([lower,upper-lower,1-upper])
    result=np.maximum(np.array(answer),0.);result/=result.sum(axis=1)[:,None]
    return result


def sample_observations(records,means,noise,phase,seed):
    rng=np.random.RandomState(seed);answer=[]
    for j,d in enumerate(records):
        y=d['y'].copy();t=d['t'][1:]
        for k in [0,1]:
            p=noise[k];S=p['latent_SD_C']**2;ell=p['ell_s'];state=rng.normal(0,np.sqrt(S));latent=[]
            for i in range(len(t)):
                if i:
                    a=np.exp(-(t[i]-t[i-1])/ell)
                    state=a*state+rng.normal(0,np.sqrt(S*(-np.expm1(-2*(t[i]-t[i-1])/ell))))
                latent.append(state)
            analogue=means[j][1:,k]+np.array(latent)+rng.normal(0,p['measurement_SD_C'],len(t))
            y[1:,k]=report(analogue,phase)
        answer.append(dict(d,y=y))
    return answer


def filter_records(records,means,noise,phase,nodes=128):
    result=[]
    for j,d in enumerate(records):
        channels=[]
        for k in [0,1]:
            p=noise[k];S=p['latent_SD_C']**2;R=p['measurement_SD_C']**2;n=nodes
            while True:
                f=filter_ou(d['t'][1:],means[j][1:,k],d['y'][1:,k],np.sqrt(S+R),p['ell_s'],R/(S+R),phase,n,12.,True)
                error=float(np.max(abs(f['posterior_normalization_ratio']-1)))
                if error<1e-6:break
                if n>=512:raise FloatingPointError('Predictive diagnostic integration failed '+str((d['name'],k,error)))
                n*=2
            f['transition_probabilities']=transition_probabilities(d['t'][1:],means[j][1:,k],d['y'][:-1,k],
                p['latent_SD_C'],p['ell_s'],p['measurement_SD_C'],f,phase)
            f['nodes']=n;channels.append(f)
        result.append(channels)
    return result


def summaries(records,filtered,seed):
    z=[[randomized_residual(f,seed+100000*k+1000*j) for k,f in enumerate(pair)] for j,pair in enumerate(filtered)]
    rows=[];metrics={};per_channel=[]
    for k in [0,1]:
        values=[v[k] for v in z];pooled=np.concatenate(values);mu=float(pooled.mean());var=float(np.mean((pooled-mu)**2))
        acf=[]
        for lag in range(1,61):
            pairs=[(v[lag:]-mu,v[:-lag]-mu) for v in values if len(v)>lag]
            acf.append(float(sum(a@b for a,b in pairs)/sum(len(a) for a,b in pairs)/var))
        input_values=np.concatenate([d['u'][:-1] for d in records]);u=input_values-input_values.mean()
        corr=float((pooled-mu)@u/np.sqrt(np.sum((pooled-mu)**2)*np.sum(u*u)))
        probs=np.concatenate([f[k]['transition_probabilities'] for f in filtered]);observed=[]
        for d in records:observed.extend(np.sign(d['y'][1:,k]-d['y'][:-1,k]).astype(int)+1)
        counts=np.bincount(observed,minlength=3);expected=probs.sum(axis=0)
        variance=np.sum(probs*(1-probs),axis=0)
        transition_z=(counts-expected)/np.sqrt(np.maximum(variance,1e-12))
        metrics.update({'T'+str(k+1)+'_'+key:float(value) for key,value in {
            'mean_abs':abs(mu),'SD_error_abs':abs(np.sqrt(var)-1),
            'max_abs_acf_1_60':max(abs(np.asarray(acf))),'KS_distance':kstest(pooled,'norm').statistic,
            'input_correlation_abs':abs(corr),'skew_abs':abs(skew(pooled)),
            'excess_kurtosis_abs':abs(kurtosis(pooled)),
            'max_transition_standardized_discrepancy_abs':max(abs(transition_z)),
            'tail_1_96_fraction_error_abs':abs(np.mean(abs(pooled)>1.96)-.0499957903),
            'tail_3_fraction_error_abs':abs(np.mean(abs(pooled)>3)-.0026997961)}.items()})
        per_channel.append({'channel':k+1,'mean':mu,'SD':np.sqrt(var),'acf_lags_1_60':acf,
            'input_correlation':corr,'transition_counts_down_same_up':counts.tolist(),
            'transition_expected_counts':expected.tolist(),'transition_martingale_variance_sum':variance.tolist(),
            'transition_standardized_discrepancy':transition_z.tolist()})
        for j,d in enumerate(records):
            v=z[j][k];p=filtered[j][k]['transition_probabilities'];obs=np.sign(d['y'][1:,k]-d['y'][:-1,k]).astype(int)+1
            centered=v-v.mean();den=centered@centered;n=len(v)
            aa=np.array([centered[h:]@centered[:-h]/den for h in range(1,min(60,n-1)+1)])
            pacf=[float(solve_toeplitz(np.r_[1.,aa[:h-1]],aa[:h])[-1]) for h in range(1,min(20,n-1)+1)]
            ljung={str(h):float(n*(n+2)*np.sum(aa[:h]**2/(n-np.arange(1,h+1)))) for h in [1,5,10,20,60] if h<n}
            for h,q in ljung.items():
                metrics['T'+str(k+1)+'_maximum_run_Ljung_Box_Q_'+h]=max(metrics.get('T'+str(k+1)+'_maximum_run_Ljung_Box_Q_'+h,0.),q)
            metrics['T'+str(k+1)+'_maximum_run_abs_PACF_1_20']=max(metrics.get('T'+str(k+1)+'_maximum_run_abs_PACF_1_20',0.),max(abs(np.array(pacf))))
            period=next((period for stem,period in [('sin_50_50_run',30.),('sin_2_50_run',120.),('sin_5_50_run',300.)] if stem in d['name']),None)
            phase_projection=None
            if period:
                t=d['t'][1:];X=np.column_stack([np.ones(n)]+[f(2*np.pi*h*t/period) for h in [1,2,3] for f in [np.sin,np.cos]])
                base=float(v@v-n*v.mean()**2);res=v-X@np.linalg.lstsq(X,v,rcond=None)[0]
                phase_projection=float(max(0.,1-res@res/base)) if base else 0.
                key='T'+str(k+1)+'_maximum_run_phase_projection_fraction'
                metrics[key]=max(metrics.get(key,0.),phase_projection)
            # Fixed command/temperature bins expose variance changes without
            # choosing cut points from a favorable fitted residual pattern.
            for label,driver,edges in [('command',d['u'][:-1],[0.,25.,50.,75.,100.01]),
                                       ('temperature',d['y'][1:,k],[15.,30.,45.,60.,80.])]:
                for a,b in zip(edges[:-1],edges[1:]):
                    take=(driver>=a)&(driver<b)
                    if take.sum()<30:continue
                    key='T'+str(k+1)+'_maximum_run_'+label+'_bin_mean_abs'
                    metrics[key]=max(metrics.get(key,0.),float(abs(v[take].mean())))
                    key='T'+str(k+1)+'_maximum_run_'+label+'_bin_variance_error_abs'
                    metrics[key]=max(metrics.get(key,0.),float(abs(v[take].var()-1)))
            rows.append({'name':d['name'],'channel':k+1,'PIT_mean':float(v.mean()),'PIT_SD':float(v.std()),
                'bin_nll':filtered[j][k]['nll'],'nodes':filtered[j][k]['nodes'],
                'ACF_lags_1_60':aa.tolist(),'PACF_lags_1_20':pacf,'Ljung_Box_Q':ljung,
                'phase_fundamental_and_harmonics_projection_fraction':phase_projection,
                'transition_counts':np.bincount(obs,minlength=3).tolist(),'transition_expected':p.sum(axis=0).tolist()})
    cross=[]
    for lag in range(-10,11):
        pairs=[]
        for pair in z:
            a,b=pair
            if lag>0:a,b=a[lag:],b[:-lag]
            if lag<0:a,b=a[:lag],b[-lag:]
            pairs.append((a,b))
        a=np.concatenate([p[0] for p in pairs]);b=np.concatenate([p[1] for p in pairs])
        cross.append(float(np.corrcoef(a,b)[0,1]))
    metrics['max_abs_cross_channel_correlation_lags_minus10_10']=max(abs(np.array(cross)))
    return {'metrics':metrics,'channels':per_channel,'cross_channel_lags_minus10_10':cross,'per_run':rows},z
