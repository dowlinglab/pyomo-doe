"""Matched final-mean decomposition, complex response and code forecasts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ndtr
from physical import load_records
from frequency import regression,held_fundamental,response_summary
from run_frequency import period_of
from decompose import partition
from bin_diagnostics import filter_records
from transfer import reported_mean
from run_bounded import write


def forecast_codes(d,mean,noise,filtered,phase,horizons=(0.,1.,10.,30.,60.)):
    """Reported-code mean forecasts using only the appropriate earlier history.

    Zero denotes the one-step prediction at every scored point. Positive h
    denotes a forecast from the current filtered state to the first recorded
    time at least h seconds later. Commands and fitted physical means are
    treated known; all fitted parameters came from these same records.
    """
    t=d['t'][1:];answers=[]
    for k,p in enumerate(noise):
        S=p['latent_SD_C']**2;R=p['measurement_SD_C']**2;ell=p['ell_s']
        f=filtered[k]
        for horizon in horizons:
            if horizon==0:
                target=np.arange(len(t));source=target-1
            else:
                source=np.arange(len(t));target=np.searchsorted(t,t+horizon);valid=target<len(t)
                source=source[valid];target=target[valid]
            predicted=[]
            for old,new in zip(source,target):
                if old<0:centers=np.array([mean[new+1,k]]);weights=np.array([1.]);sd=np.sqrt(S+R)
                else:
                    a=np.exp(-(t[new]-t[old])/ell);history=f['history'][old]
                    centers=mean[new+1,k]+a*history['grid'];weights=history['weight']
                    sd=np.sqrt(S*(-np.expm1(-2*(t[new]-t[old])/ell))+R)
                predicted.append(float(weights@reported_mean(centers,sd,phase)))
            predicted=np.array(predicted);e=d['y'][target+1,k]-predicted
            for cutoff in [0.,60.,180.,450.]:
                keep=t[target]>=cutoff
                if not np.any(keep):continue
                answers.append({'channel':k+1,'cutoff_s':cutoff,'horizon_s':horizon,
                    'definition':'one-step' if horizon==0 else 'at-least-h-after-filtering',
                    'nobs':int(keep.sum()),'SSE_C2':float(e[keep]@e[keep]),'RMSE_C':float(np.sqrt(np.mean(e[keep]**2))),
                    'mean_error_C':float(e[keep].mean()),
                    'actual_horizon_mean_s':float(np.mean(t[target[keep]]-t[source[keep]])) if horizon else None})
    return answers


def complex_region(summary,regression_fit,input_coeff):
    u=complex(input_coeff[1],input_coeff[2]);T=np.array([[u.real,u.imag],[-u.imag,u.real]])/abs(u)**2
    h=np.array([summary['H_real'],summary['H_imag']]);regions={}
    for lag,V in regression_fit['covariances'].items():
        C=T@V@T.T;distance=float(h@np.linalg.pinv(C)@h)
        regions[lag]={'complex_covariance':C.tolist(),'zero_Mahalanobis_distance_squared':distance,
            'working_95_ellipse_contains_zero':bool(distance<=5.991464547107979),
            'phase_interval_reportable':bool(distance>5.991464547107979),
            'notice':'Working HAC complex ellipse; phase delta method withheld when zero is inside. Not a fitted thermal-parameter confidence region.'}
    return regions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fits',type=Path,nargs='+',required=True);parser.add_argument('--audits',type=Path,nargs='+',required=True)
    parser.add_argument('--forecast-fit',type=str,required=True);args=parser.parse_args()
    if len(args.fits)!=len(args.audits):raise ValueError('Bind each fit to one independent audit')
    config=json.loads((HERE/'config.json').read_text());lookup={d['name']:d for d in load_records()}
    rows=[];freq=[];folds=[];operating=[];forecasts=[];bindings={};model_data={}
    for path,audit_path in zip(args.fits,args.audits):
        fit=json.loads((path/'fit.json').read_text());audit=json.loads((audit_path/'audit.json').read_text())
        fitsha=hashlib.sha256((path/'fit.json').read_bytes()).hexdigest();assert audit['status']=='passed' and audit['fit_sha256']==fitsha
        assert json.loads((audit_path/'exit.json').read_text())['exit_code']==0
        if fit.get('condition_initial_code'):raise ValueError('Conditional-first-code diagnostics require matching conditional forecast distributions')
        arrays=np.load(path/'trajectories.npz');records=[lookup[n] for n in fit['record_names']]
        means=[arrays['predicted_'+str(j)] for j in range(len(records))];label=path.name
        bindings[label]={'fit_sha256':fitsha,'arrays_sha256':hashlib.sha256((path/'trajectories.npz').read_bytes()).hexdigest(),
            'audit_sha256':hashlib.sha256((audit_path/'audit.json').read_bytes()).hexdigest()}
        expected=[]
        for j,d in enumerate(records):
            assert fit['raw_sha256'][d['name']]==d['sha256'] and np.array_equal(d['y'],arrays['observed_'+str(j)])
            t=d['t'];period=period_of(d['name']);code_mean=np.empty_like(means[j])
            for k,p in enumerate(fit['noise_parameters']):code_mean[:,k]=reported_mean(means[j][:,k],np.hypot(p['latent_SD_C'],p['measurement_SD_C']),fit['phase'])
            expected.append(code_mean)
            for kind,pred in [('analogue_physical_mean',means[j]),('marginal_reported_code_expectation',code_mean)]:
                for cutoff in config['diagnostic_startup_windows_s']:
                    take=t[1:]>=cutoff;ts=t[1:][take];error=(d['y']-pred)[1:][take]
                    for k in [0,1]:
                        e=error[:,k];parts=partition(ts,e,period);sse=float(e@e)
                        assert abs(sum(parts)-sse)<1e-7
                        rows.append({'model':label,'prediction_kind':kind,'name':d['name'],'channel':k+1,'cutoff_s':cutoff,
                            'nobs':len(e),'SSE_C2':sse,'RMSE_C':float(np.sqrt(np.mean(e**2))),'mean_error_C':float(e.mean()),
                            'centered_RMSE_C':float(e.std()),'offset_SSE_C2':parts[0],'additional_trend_SSE_C2':parts[1],
                            'additional_fundamental_SSE_C2':parts[2],'additional_harmonic2_SSE_C2':parts[3],
                            'additional_harmonic3_SSE_C2':parts[4],'remainder_SSE_C2':parts[5]})
            if period:
                for cutoff in config['diagnostic_startup_windows_s']:
                    take=t[1:]>=cutoff;ts=t[1:][take];inp=held_fundamental(t,d['u'],2*np.pi/period,max(cutoff,t[1]),t[-1])
                    for k in [0,1]:
                        observed=regression(ts,d['y'][1:,k][take],period);summary=response_summary(observed,inp)
                        region=complex_region(summary,observed,inp);matched={}
                        for kind,pred in [('analogue',means[j]),('reported_code',code_mean)]:
                            v=response_summary(regression(ts,pred[1:,k][take],period),inp)
                            matched[kind]=v
                        freq.append({'model':label,'name':d['name'],'channel':k+1,'cutoff_s':cutoff,'period_s':period,
                            'settled_cycles_not_assumed':float(np.ptp(ts)/period),'measured':summary,'measured_complex_regions':region,
                            'matched_sampled':matched,'phase_difference_from_reported_deg':float(np.angle(complex(summary['H_real'],summary['H_imag'])/complex(matched['reported_code']['H_real'],matched['reported_code']['H_imag']),deg=True))})
                        phase=np.mod(2*np.pi*ts/period+np.angle(complex(inp[1],inp[2])),2*np.pi)
                        which=np.minimum(11,(phase/(2*np.pi)*12).astype(int));e=(d['y']-code_mean)[1:,k][take]
                        r=regression(ts,e,period);v=e-r['nuisance_component']
                        folds.append({'model':label,'name':d['name'],'channel':k+1,'cutoff_s':cutoff,'period_s':period,
                            'phase_midpoint_rad':((np.arange(12)+.5)*2*np.pi/12).tolist(),'counts':[int(np.sum(which==i)) for i in range(12)],
                            'detrended_reported_error_C':[float(v[which==i].mean()) if np.any(which==i) else None for i in range(12)]})
            # Descriptive operating-point bins; command applies on the prior
            # interval, and measured heating/cooling groups do not imply cause.
            slope=(np.interp(t+10,t,d['y'][:,0])-np.interp(t-10,t,d['y'][:,0]))/20
            sign=np.where(slope>.002,'heating',np.where(slope<-.002,'cooling','flat'))
            for k in [0,1]:
                error=(d['y']-code_mean)[1:,k];u=d['u'][:-1]
                for low in np.arange(15.,80.,5.):
                    for qlow in np.arange(0.,100.,20.):
                        for direction in ['heating','cooling','flat']:
                            take=(t[1:]>=10)&(t[1:]<=t[-1]-10)&(d['y'][1:,0]>=low)&(d['y'][1:,0]<low+5)&(u>=qlow)&(u<=100 if qlow==80 else u<qlow+20)&(sign[1:]==direction)
                            if take.sum()<8:continue
                            operating.append({'model':label,'name':d['name'],'channel':k+1,'temperature_bin_low_C':low,
                                'command_bin_low_percent':qlow,'direction':direction,'nobs':int(take.sum()),
                                'mean_reported_error_C':float(error[take].mean()),'RMSE_reported_C':float(np.sqrt(np.mean(error[take]**2)))})
        if label==args.forecast_fit:
            filtered=filter_records(records,means,fit['noise_parameters'],fit['phase'])
            for j,d in enumerate(records):
                forecasts.extend([dict(r,model=label,name=d['name']) for r in forecast_codes(d,means[j],fit['noise_parameters'],filtered[j],fit['phase'])])
            del filtered
        model_data[label]=(records,expected)
        print(json.dumps({'model':label,'complete':True}),flush=True)
    if not forecasts:raise ValueError('The requested forecast fit is absent')
    pd.DataFrame(rows).to_csv(args.output/'decomposition.csv',index=False);pd.DataFrame(operating).to_csv(args.output/'operating_bins.csv',index=False)
    pd.DataFrame(forecasts).to_csv(args.output/'forecast_horizons.csv',index=False)
    write(args.output/'frequency.json',freq);write(args.output/'phase_folds.json',folds)
    aggregate=[]
    for model in model_data:
        for kind in ['analogue_physical_mean','marginal_reported_code_expectation']:
            for cutoff in config['diagnostic_startup_windows_s']:
                for k in [1,2]:
                    selected=[r for r in rows if r['model']==model and r['prediction_kind']==kind and r['cutoff_s']==cutoff and r['channel']==k]
                    n=sum(r['nobs'] for r in selected);sse=sum(r['SSE_C2'] for r in selected)
                    aggregate.append({'model':model,'prediction_kind':kind,'channel':k,'cutoff_s':cutoff,'nobs':n,
                        'RMSE_C':float(np.sqrt(sse/n)),**{key:sum(r[key] for r in selected)/sse for key in selected[0] if key.endswith('_SSE_C2')}})
    write(args.output/'aggregate.json',aggregate)
    fig,axes=plt.subplots(2,3,figsize=(11,6),sharex='col')
    primary=args.forecast_fit
    for c,period in enumerate([30.,120.,300.]):
        for row in [r for r in folds if r['model']==primary and r['period_s']==period and r['cutoff_s']==450.]:
            axes[row['channel']-1,c].plot(np.rad2deg(row['phase_midpoint_rad']),row['detrended_reported_error_C'],'o-',ms=3,label=row['name'].split('_run_')[-1].replace('.csv',''))
        for k in [0,1]:axes[k,c].axhline(0,color='gray',lw=.5);axes[k,c].set(title=str(int(period))+' s, T'+str(k+1),ylabel='Reported-code error, C');axes[k,c].grid(alpha=.2)
        axes[1,c].set_xlabel('Recorded input phase, degrees')
    axes[0,0].legend(title='Repeat');fig.tight_layout();fig.savefig(args.output/'phase_repeats.png',dpi=160);fig.savefig(args.output/'phase_repeats.pdf');plt.close(fig)
    write(args.output/'receipt.json',{'status':'complete','source_bindings':bindings,'forecast_fit':primary,
        'notice':'Matched sampled means and marginal code expectations; actual held input projection; descriptive HAC30/60/120 complex response regions; predeclared startup windows. Quantized noise forecasts use only earlier code histories, but mean/noise estimates use the whole data. No external validation or causal attribution from operating groups. Parameter and model-form uncertainty are excluded from conditional forecast RMSE diagnostics.'})


if __name__=='__main__':main()
