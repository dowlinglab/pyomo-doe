"""Predictive checks that retain code information without adding PIT jitter."""
import numpy as np
from scipy.special import log_ndtr,ndtri
from scipy.optimize import brentq

LOG2PI=np.log(2*np.pi)


def inverse_logcdf(logp):
    if np.isneginf(logp):return -np.inf
    if logp>np.log(.5)+1e-12:raise ValueError('Use the smaller normal tail')
    if logp>-700:return float(ndtri(np.exp(logp)))
    return float(brentq(lambda x:float(log_ndtr(x))-logp,-max(50.,2*np.sqrt(-2*logp)),0.))


def expected_gaussian_pit(filtered):
    """E[Phi^-1(U) | reported code, past], no auxiliary randomization.

    The conditional score is generally neither Gaussian nor unit variance.
    Its serial/cross diagnostics therefore use the same simulated-code map.
    A midpoint quantile handles intervals thinner than 1e-8 of the local tail;
    this limit avoids subtracting numerically coincident tail quantiles.
    """
    answer=[];thin=0
    for ll,rs,lp in zip(filtered['log_CDF_lower'],filtered['log_SF_upper'],filtered['log_bin_probability']):
        if lp-min(ll,rs)<np.log(1e-8):
            left=np.logaddexp(ll,np.log(.5)+lp);right=np.logaddexp(rs,np.log(.5)+lp)
            z=inverse_logcdf(left) if left<np.log(.5) else -inverse_logcdf(right)
            answer.append(z);thin+=1;continue
        # Reconstruct the opposite endpoint tails from disjoint bin+tail
        # probabilities. Complementing a CDF rounded to 1 loses far tails.
        a=inverse_logcdf(ll) if ll<=np.log(.5) else -inverse_logcdf(np.logaddexp(rs,lp))
        b=-inverse_logcdf(rs) if rs<=np.log(.5) else inverse_logcdf(np.logaddexp(ll,lp))
        if a==-np.inf and b==np.inf:value=0.
        elif a==-np.inf:value=-np.exp(-.5*b*b-.5*LOG2PI-lp)
        elif b==np.inf:value=np.exp(-.5*a*a-.5*LOG2PI-lp)
        else:
            difference=.5*(b-a)*(b+a)
            if difference==0:value=0.
            elif difference>0:value=np.exp(-.5*a*a-.5*LOG2PI-lp)*(-np.expm1(-difference))
            else:value=-np.exp(-.5*b*b-.5*LOG2PI-lp)*(-np.expm1(difference))
        answer.append(value)
    z=np.array(answer)
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite nonrandomized PIT expectation')
    return z,thin


def code_events(d,channel):
    return np.eye(3)[np.sign(d['y'][1:,channel]-d['y'][:-1,channel]).astype(int)+1]


def joint_transition_checks(records,filtered):
    """Joint cell discrepancies and centered cross-channel event products.

    The product probabilities encode the fitted independent-channel null.
    Dependent cells and estimated parameters preclude an automatic chi-square
    reference. Simulation supplies the reference for both maximum and sum.
    """
    per_run=[];totals=[np.zeros((3,3)) for _ in range(4)]
    for d,f in zip(records,filtered):
        p,q=[v['transition_probabilities'] for v in f];a,b=[code_events(d,k) for k in [0,1]]
        pq=p[:,:,None]*q[:,None,:];ab=a[:,:,None]*b[:,None,:]
        error=np.sum(ab-pq,axis=0);variance=np.sum(pq*(1-pq),axis=0)
        centered=np.einsum('ti,tj->ij',a-p,b-q)
        crossvar=np.einsum('ti,tj->ij',p*(1-p),q*(1-q))
        for accumulator,value in zip(totals,[error,variance,centered,crossvar]):accumulator+=value
        per_run.append({'name':d['name'],'observed_joint_counts':ab.sum(axis=0).tolist(),
            'expected_joint_counts':pq.sum(axis=0).tolist(),
            'joint_discrepancy_standardized':(error/np.sqrt(np.maximum(variance,1.))).tolist(),
            'centered_event_cross_standardized':(centered/np.sqrt(np.maximum(crossvar,1.))).tolist()})
    joint=totals[0]/np.sqrt(np.maximum(totals[1],1.));cross=totals[2]/np.sqrt(np.maximum(totals[3],1.))
    metrics={'pooled_joint_transition_max_abs':float(np.max(abs(joint))),
        'pooled_centered_event_cross_max_abs':float(np.max(abs(cross))),
        'pooled_centered_event_cross_sum_squares':float(np.sum(cross**2)),
        'maximum_run_joint_transition_abs':float(max(np.max(abs(np.array(v['joint_discrepancy_standardized']))) for v in per_run)),
        'maximum_run_centered_event_cross_abs':float(max(np.max(abs(np.array(v['centered_event_cross_standardized']))) for v in per_run))}
    return {'metrics':metrics,'pooled_joint_transition_standardized':joint.tolist(),
        'pooled_centered_event_cross_standardized':cross.tolist(),'per_run':per_run,
        'notice':'Predictive martingale variance sums with floor 1 for rare cells; no independent-cell chi-square approximation. Marginal misspecification can also affect centered cross products.'}


def summaries(records,filtered):
    scores=[];thin=[]
    for pair in filtered:
        v=[expected_gaussian_pit(f) for f in pair];scores.append([x[0] for x in v]);thin.append([x[1] for x in v])
    joint=joint_transition_checks(records,filtered);metrics=dict(joint['metrics']);channels=[];per_run=[]
    for k in [0,1]:
        values=[v[k] for v in scores];pooled=np.concatenate(values);mu=pooled.mean();variance=pooled.var()
        acf=[]
        for lag in range(1,61):
            pairs=[(v[lag:]-mu,v[:-lag]-mu) for v in values if len(v)>lag]
            acf.append(float(sum(a@b for a,b in pairs)/sum(len(a) for a,b in pairs)/variance))
        u=np.concatenate([d['u'][:-1] for d in records]);correlation=float(np.corrcoef(pooled,u)[0,1])
        metrics.update({'T'+str(k+1)+'_score_mean_abs':float(abs(mu)),
            'T'+str(k+1)+'_score_max_abs_ACF_1_60':float(max(abs(np.array(acf)))),
            'T'+str(k+1)+'_score_input_correlation_abs':abs(correlation)})
        channels.append({'channel':k+1,'mean':float(mu),'SD':float(np.sqrt(variance)),
            'acf_lags_1_60':acf,'input_correlation':correlation})
        for j,d in enumerate(records):
            v=values[j];period=next((p for stem,p in [('sin_50_50_run',30.),('sin_2_50_run',120.),('sin_5_50_run',300.)] if stem in d['name']),None)
            projection=None
            if period:
                t=d['t'][1:];X=np.column_stack([np.ones(len(t))]+[f(2*np.pi*h*t/period) for h in [1,2,3] for f in [np.sin,np.cos]])
                e=v-X@np.linalg.lstsq(X,v,rcond=None)[0];den=np.sum((v-v.mean())**2)
                projection=float(max(0.,1-e@e/den)) if den else 0.
                key='T'+str(k+1)+'_score_maximum_phase_projection_fraction';metrics[key]=max(metrics.get(key,0.),projection)
            per_run.append({'name':d['name'],'channel':k+1,'mean':float(v.mean()),'SD':float(v.std()),
                'phase_projection_fraction':projection,'thin_interval_approximation_count':thin[j][k]})
    cross=[]
    for lag in range(-10,11):
        pairs=[]
        for a,b in scores:
            if lag>0:a,b=a[lag:],b[:-lag]
            elif lag<0:a,b=a[:lag],b[-lag:]
            pairs.append((a,b))
        cross.append(float(np.corrcoef(np.concatenate([v[0] for v in pairs]),np.concatenate([v[1] for v in pairs]))[0,1]))
    metrics['score_max_abs_cross_channel_correlation_minus10_10']=float(max(abs(np.array(cross))))
    return {'metrics':metrics,'score_channels':channels,'score_cross_channel_lags_minus10_10':cross,
        'score_per_run':per_run,'joint_transition':joint,
        'scope':'Deterministic conditional expected Gaussian-PIT scores and joint code-event checks. Scores are not Gaussian/unit-SD residuals. Compare against matched code-generating simulations.'}
