"""Per-run trajectories, harmonic diagnostics, and causal residual forecasts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
import pandas as pd
from frequency import held_fundamental
from run_frequency import period_of
from run_mean import write_json


def filtered_disturbance(t,residual,ell,eta,sigma):
    slow=(1-eta)*sigma**2; measurement=eta*sigma**2
    mean=0.; variance=slow; means=[]; variances=[]
    for j,r in enumerate(residual):
        if j:
            a=np.exp(-(t[j]-t[j-1])/ell)
            mean=a*mean; variance=a*a*variance+slow*(1-a*a)
        gain=variance/(variance+measurement)
        mean+=gain*(r-mean)
        variance=variance*measurement/(variance+measurement)
        means.append(mean);variances.append(variance)
    return np.array(means),np.array(variances)


def audit_forecast():
    rng=np.random.RandomState(985)
    t=np.r_[0.,np.cumsum(rng.uniform(.6,2.,34))]
    r=rng.normal(0,1,len(t));ell,eta,sigma=17.,.16,.8
    mean,var=filtered_disturbance(t,r,ell,eta,sigma)
    C=sigma**2*((1-eta)*np.exp(-abs(t[:,None]-t[None,:])/ell)+eta*np.eye(len(t)))
    errors=[]
    for origin,target in [(0,1),(5,7),(13,24),(25,33)]:
        cross=C[target,:origin+1]
        directmean=cross@np.linalg.solve(C[:origin+1,:origin+1],r[:origin+1])
        directvar=C[target,target]-cross@np.linalg.solve(C[:origin+1,:origin+1],cross)
        a=np.exp(-(t[target]-t[origin])/ell)
        predmean=a*mean[origin]
        predvar=a*a*var[origin]+(1-eta)*sigma**2*(1-a*a)+eta*sigma**2
        error=max(abs(directmean-predmean),abs(directvar-predvar));errors.append(error)
    if max(errors)>1e-10:raise AssertionError(errors)
    return {'status':'passed','dense_conditional_mean_variance_max_abs':float(max(errors)),
            'scope':'Forecasting uses observations through origin only; independent Gaussian conditional check'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    write_json(args.output/'forecast_audit.json',audit_forecast())
    fit=json.loads((args.input/'fit.json').read_text());best=fit['best']
    arrays=np.load(args.input/'trajectories.npz')
    if 'noise_parameters' in best:
        ell,eta=best['noise_parameters'];sigma=best['sigma_total_ML_C']
    else:
        x=best['point'];ell,eta=np.exp(x[-2]),x[-1];sigma=best['sigma_C']
    forecasts=[];harmonics=[];series=[]
    for j,name in enumerate(fit['record_names']):
        t=arrays['time_'+str(j)];t=t-t[0]
        y=arrays['observed_'+str(j)];pred=arrays['predicted_'+str(j)]
        u=arrays['input_'+str(j)];e=y[1:]-pred[1:];tr=t[1:]
        z=arrays['standardized_innovation_'+str(j)]
        means,variances=filtered_disturbance(tr,e,ell,eta,sigma)
        for horizon in (1.,10.,60.,300.):
            target=np.searchsorted(tr,tr+horizon)
            origins=np.flatnonzero(target<len(tr));target=target[origins]
            a=np.exp(-(tr[target]-tr[origins])/ell)
            error=e[target]-a*means[origins]
            conditional_variance=a*a*variances[origins]+(1-eta)*sigma**2*(1-a*a)+eta*sigma**2
            forecasts.append({'name':name,'horizon_s':horizon,'n':len(target),
                              'conditional_SSE_C2':float(error@error),'conditional_RMSE_C':float(np.sqrt(np.mean(error**2))),
                              'same_targets_free_SSE_C2':float(e[target]@e[target]),
                              'same_targets_free_RMSE_C':float(np.sqrt(np.mean(e[target]**2))),
                              'observation_fraction_inside_working_conditional95':float(np.mean(abs(error)<=1.96*np.sqrt(conditional_variance)))})
        period=period_of(name)
        if period:
            for cutoff in (300.,450.,600.):
                keep=t>=cutoff;ts=t[keep];w=2*np.pi/period
                X=np.column_stack([np.ones(len(ts)),(ts-ts.mean())/(ts[-1]-ts[0])]+
                                  [f(k*w*ts) for k in (1,2,3) for f in (np.sin,np.cos)])
                measured=np.linalg.lstsq(X,y[keep],rcond=None)[0]
                model=np.linalg.lstsq(X,pred[keep],rcond=None)[0]
                rounded=np.round(19.93908288+.322299839*np.rint((pred-19.93908288)/.322299839),2)
                rounded_coeff=np.linalg.lstsq(X,rounded[keep],rcond=None)[0]
                amplitude=lambda b:[float(np.hypot(*b[2*k:2*k+2])) for k in (1,2,3)]
                input_amp=[]
                for k in (1,2,3):
                    b=held_fundamental(t,u,k*w,cutoff,t[-1]);input_amp.append(float(np.hypot(*b[1:])))
                harmonics.append({'name':name,'period_s':period,'cutoff_s':cutoff,
                                  'cycles':float((t[-1]-cutoff)/period),'design_condition':float(np.linalg.cond(X)),
                                  'measured_temperature_amplitudes_C':amplitude(measured),
                                  'model_temperature_amplitudes_C':amplitude(model),
                                  'rounded_model_temperature_amplitudes_C':amplitude(rounded_coeff),
                                  'input_projected_amplitudes_percent':input_amp})
        Xtrend=np.column_stack((np.ones(len(tr)),(tr-tr.mean())/(tr[-1]-tr[0])))
        trend=Xtrend@np.linalg.lstsq(Xtrend,e,rcond=None)[0]
        series.append({'name':name,'time':t,'observed':y,'predicted':pred,'residual':e,'innovation':z,'trend':trend})
    pd.DataFrame(forecasts).to_csv(args.output/'forecast_errors.csv',index=False)
    summary=[]
    for h in (1.,10.,60.,300.):
        selected=[r for r in forecasts if r['horizon_s']==h];n=sum(r['n'] for r in selected)
        summary.append({'horizon_s':h,'n':n,'conditional_RMSE_C':float(np.sqrt(sum(r['conditional_SSE_C2'] for r in selected)/n)),
                        'same_targets_free_RMSE_C':float(np.sqrt(sum(r['same_targets_free_SSE_C2'] for r in selected)/n))})
    write_json(args.output/'diagnostics.json',{'forecast_summary':summary,'harmonic_cells':harmonics,
        'input_fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
        'notice':'Within-record retrospective fitted parameters/ambient; conditioning only on temperatures through each origin. Not held-out prediction. Forecast ranges condition on parameters and covariance. Harmonic regressions descriptive; finite-window projection leakage, trend, input harmonics and quantization prevent unique nonlinear attribution. Rounded-model comparison is deterministic sensitivity, not an independent calibration/noise law.'})
    plots(args.output,series,summary)
    print(json.dumps({'status':'complete','forecast_summary':summary,'harmonic_cells':len(harmonics)}),flush=True)


def plots(output,series,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'savefig.dpi':150})
    for stem,label in [('sin_50_50','30-second sine'),('sin_2_50','120-second sine'),('sin_5_50','300-second sine'),('step_50','Step')]:
        selected=[r for r in series if stem+'_run' in r['name']]
        fig,axes=plt.subplots(3,3,figsize=(12,7),sharex=True)
        for j,row in enumerate(selected):
            axes[j,0].plot(row['time'],row['observed'],color='black',linewidth=.65,label='Observed')
            axes[j,0].plot(row['time'],row['predicted'],color='#2166ac',linewidth=.8,label='Shared model')
            axes[j,1].plot(row['time'][1:],row['residual'],color='black',linewidth=.6,label='Residual')
            axes[j,1].plot(row['time'][1:],row['trend'],color='#b2182b',linewidth=1,label='Descriptive offset/trend')
            axes[j,2].plot(row['time'][1:],row['innovation'],color='#2166ac',linewidth=.55)
            for col in (1,2):axes[j,col].axhline(0,color='gray',linewidth=.5)
            for col in range(3):
                axes[j,col].axvline(300,color='gray',linestyle=':',linewidth=.7)
                axes[j,col].grid(alpha=.2)
            axes[j,0].set_ylabel('Run '+str(j+1)+'; T1, C')
        for col,text in enumerate(('Temperature','Raw residual, C','Standardized innovation')):axes[0,col].set_title(text)
        axes[0,0].legend();axes[0,1].legend()
        for ax in axes[-1]:ax.set_xlabel('Elapsed time, s')
        fig.suptitle(label+'; pooled reference; dotted line marks 300 s')
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(output/('runs_'+stem+'.png'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.semilogx([r['horizon_s'] for r in summary],[r['conditional_RMSE_C'] for r in summary],'o-',label='Condition on temperature history')
    ax.semilogx([r['horizon_s'] for r in summary],[r['same_targets_free_RMSE_C'] for r in summary],'o--',label='Free run; same targets')
    ax.set(xlabel='Forecast horizon, s',ylabel='Pooled RMSE, C',title='Within-record fitted model; retrospective parameters')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'prediction_vs_simulation.png');plt.close(fig)
    write_json(output/'figure_receipt.json',{'visual_review':'pending','figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.png')}})


if __name__=='__main__':main()
