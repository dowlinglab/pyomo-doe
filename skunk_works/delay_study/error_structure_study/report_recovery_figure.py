"""Five-inch report figure from audited recovery points and diagnostic summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from run_bounded import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--summary-audit',type=Path,required=True)
    p.add_argument('--recovery-audit',type=Path,required=True);p.add_argument('--fit',type=Path,required=True)
    args=p.parse_args();paths=[args.summary/'summary.json',args.summary_audit/'audit.json',args.recovery_audit/'audit.json',args.fit/'fit.json']
    summary,checked,recovery,fit=[json.loads(q.read_text()) for q in paths]
    for directory in [args.summary,args.summary_audit,args.recovery_audit,args.fit]:
        assert json.loads((directory/'exit.json').read_text())['exit_code']==0
    assert summary['status']=='complete' and checked['status']==recovery['status']=='passed'
    assert checked['summary_sha256']==sha(paths[0]) and summary['recovery_audit_sha256']==sha(paths[2])
    assert summary['fit_sha256']==recovery['fit_sha256']==sha(paths[3])
    assert checked['checked_draws']==50 and len(recovery['rows'])==50
    points=np.array([r['recovered_point'] for r in recovery['rows']]);names=fit['coordinate_names']
    gain=np.exp(points[:,names.index('log_K')]);gain0=fit['parameters']['K_common_nominal']
    sensor=np.exp(points[:,names.index('log_rS')]);lag=np.exp(points[:,names.index('log_input_lag')])
    reverse=sensor<lag;assert int(reverse.sum())==summary['lag_label_recovery']['sensor_time_smaller_than_input_time_count']
    errors=100*(gain/gain0-1);metrics=summary['refitted_diagnostic_comparisons']
    selected=['T1_score_input_correlation_abs','T1_score_maximum_phase_projection_fraction',
        'T2_score_max_abs_ACF_1_60','score_max_abs_cross_channel_correlation_minus10_10']
    data=np.array([metrics[n]['data'] for n in selected])
    quantiles=np.array([metrics[n]['refitted_reference_q025_q50_q975'] for n in selected])
    values={'draws':50,'gain_error_percent':errors.tolist(),'sensor_time_s':sensor.tolist(),
        'input_time_s':lag.tolist(),'lag_order_reversed':reverse.tolist(),
        'diagnostic_names':selected,'hardware_statistics':data.tolist(),
        'refitted_reference_q025_q50_q975':quantiles.tolist()}
    write(args.output/'figure_values.json',values)
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
        'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,1,figsize=(5.,7.4))
    ax=axes[0]
    ax.scatter(sensor[~reverse],lag[~reverse],s=16,alpha=.6,label='Generating label order',color='tab:blue')
    ax.scatter(sensor[reverse],lag[reverse],s=22,alpha=.7,marker='^',label='Reversed order',color='tab:orange')
    ax.plot(fit['parameters']['rS_s'],fit['parameters']['input_lag_s'],'k*',ms=9,label='Generating fit')
    upper=max(17.,1.05*max(float(sensor.max()),float(lag.max())))
    ticks=np.arange(0,upper,5.)
    ax.plot([0,upper],[0,upper],':',lw=.7,color='.6')
    ax.set(xlim=(0,upper),ylim=(0,upper),xticks=ticks,yticks=ticks,
        xlabel='Fitted sensor time (s)',ylabel='Fitted input time (s)',title='(a) Lag labels reverse in '+str(int(reverse.sum()))+'/50 refits')
    ax.legend(loc='upper right',framealpha=.9)
    ax=axes[1];counts,_,_=ax.hist(errors,bins=12,alpha=.7,color='tab:blue')
    ax.axvline(0,color='k',lw=.8);ax.set_ylim(0,max(counts)*1.3)
    ax.set(xlabel='Gain estimation error (%)',ylabel='Generated collections',title='(b) Gain recovery under the fitted model')
    ax.text(.02,.96,str(summary['coverage']['covered'])+'/50 nominal gain regions include the generating value',
        transform=ax.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':1.,'pad':2.})
    ax=axes[2];x=np.arange(4)
    ax.errorbar(x,quantiles[:,1],yerr=np.vstack([quantiles[:,1]-quantiles[:,0],quantiles[:,2]-quantiles[:,1]]),
        fmt='o',ms=3,color='tab:blue',lw=2,label='Refitted reference 2.5/50/97.5%')
    ax.plot(x,data,'x',color='tab:red',ms=6,label='Hardware')
    ax.set(xticks=x,xticklabels=['T1 input','T1 phase','T2 ACF','Cross'],ylabel='Diagnostic magnitude',
        ylim=(0,max(data.max(),quantiles.max())*1.3),title='(c) Residual structure remains after refitting')
    ax.legend(loc='upper left',fontsize=7.5,framealpha=.9)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('Conditional recovery: 50 independently audited refits',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.975))
    for suffix in ['png','pdf']:fig.savefig(args.output/('recovery_report.'+suffix),dpi=200)
    plt.close(fig)
    write(args.output/'figure_receipt.json',{'status':'complete','source_bindings_sha256':{str(q):sha(q) for q in paths},
        'figure_values_sha256':sha(args.output/'figure_values.json'),
        'outputs_sha256':{suffix:sha(args.output/('recovery_report.'+suffix)) for suffix in ['png','pdf']},
        'size_inches':[5.,7.4],
        'scope':'Plotting audited saved points/statistics only. Repeated-fit quantiles are not hardware confidence intervals; diagnostic ranges are descriptive and do not adjust for structure selection or multiple diagnostics.'})


if __name__=='__main__':main()
