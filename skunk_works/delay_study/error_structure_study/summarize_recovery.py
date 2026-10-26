"""Conditional parameter recovery, gain coverage and refitted code diagnostics.

No optimizer runs here. Require numerical audits and report every missing or
failed outcome. A labeled partial mode is for checking the summary controller,
never for replacing the prespecified fifty-draw study.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
from audit_recovery import interval
from audit_provenance import numerical_binding,diagnostic_binding
from run_bounded import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage(rows,declared):
    complete=[r for r in rows if r['status']=='passed' and r['coverage_complete']]
    n=len(complete);k=sum(r['covered'] for r in complete)
    return {'declared':declared,'audited_attempts':len(rows),'complete':n,'covered':k,
        'failed_or_incomplete_audited':len(rows)-n,'missing':declared-len(rows),
        'coverage_among_complete':k/n if n else None,'binomial_95_Monte_Carlo_interval':interval(k,n),
        'failure_worst_case_bounds_all_declared':[k/declared,(k+declared-n)/declared]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--data-diagnostics',type=Path,required=True);parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--information-audit',type=Path)
    args=parser.parse_args();begun=time.monotonic()
    source=json.loads((args.audit/'audit.json').read_text());fit=json.loads((args.input/'fit.json').read_text())
    assert json.loads((args.audit/'exit.json').read_text())['exit_code']==0
    assert source['fit_sha256']==sha(args.input/'fit.json')
    assert source['numerical_dependency_binding']==numerical_binding(args.output)
    assert source['diagnostic_implementation_binding']==diagnostic_binding(args.output)
    rows=source['rows'];declared=source['declared_draws'];indices=[r['draw'] for r in rows]
    assert len(set(indices))==len(rows) and set(indices)<=set(range(declared))
    if not args.allow_partial:
        assert source['status']=='passed' and sorted(indices)==list(range(declared))
        assert args.information_audit,'Final recovery summary must compare sampling spread with the proposed local approximation'
    else:assert source['status'] in ['passed','partial']
    count=coverage(rows,declared)
    assert count['complete']==source['complete_audited_coverage_draws'] and count['covered']==source['covered_audited_draws']
    parameter_rows=[r for r in rows if r['status']=='passed' and r['free_fit_success']]
    for row in parameter_rows:
        assert any(a['status']=='passed' and a['role']=='selected_free_output' for a in row['numerical_audits'])
    points=np.array([r['recovered_point'] for r in parameter_rows]);truth=np.array(fit['best']['point'])
    names=fit['coordinate_names'];bounds=np.array(fit['bounds']);table=[]
    if len(points):
        assert points.shape==(len(parameter_rows),len(truth)) and len(names)==len(truth)
        for j,name in enumerate(names):
            values=points[:,j];physical=np.exp(values) if name.startswith('log_') else values
            table.append({'coordinate':name,'generating_coordinate':float(truth[j]),
                'generating_physical_value':float(np.exp(truth[j])) if name.startswith('log_') else float(truth[j]),
                'mean_coordinate_bias':float(values.mean()-truth[j]),
                'coordinate_sampling_SD':float(values.std(ddof=1)) if len(values)>1 else None,
                'coordinate_q025_q50_q975':np.quantile(values,[.025,.5,.975]).tolist(),
                'physical_q025_q50_q975':np.quantile(physical,[.025,.5,.975]).tolist(),
                'lower_bound_fraction':float(np.mean(values-bounds[j,0]<1e-5)),
                'upper_bound_fraction':float(np.mean(bounds[j,1]-values<1e-5))})
    local_comparison=[];information_binding=None
    if args.information_audit:
        path=args.information_audit/'audit.json';information=json.loads(path.read_text())
        assert information['status']=='passed' and information['fit_sha256']==source['fit_sha256']
        assert json.loads((args.information_audit/'exit.json').read_text())['exit_code']==0
        information_binding={'path':str(path),'sha256':sha(path)}
        lookup={r['coordinate']:r for r in table}
        for row in information['local_comparison']:
            empirical=lookup.get(row['coordinate']);se=row['bin_observed_mean_and_noise_nuisance_SE']
            if empirical is None:continue
            spread=empirical['coordinate_sampling_SD']
            local_comparison.append({'coordinate':row['coordinate'],'generating_point_active_face_SE':se,
                'refitted_sampling_SD':spread,'sampling_SD_to_active_face_SE':spread/se if spread is not None else None,
                'mean_bias_to_active_face_SE':empirical['mean_coordinate_bias']/se,
                'notice':'Refits can change the active face or lag basin; this spread comparison is not coverage calibration for every local mixture or retrospective training design.'})
    diagnostic_source=args.data_diagnostics/'comparison.json';comparison=json.loads(diagnostic_source.read_text())
    assert comparison['status']=='complete' and comparison['fit_sha256']==source['fit_sha256']
    data_path=args.data_diagnostics/'data_diagnostics.json';data=json.loads(data_path.read_text())['metrics']
    diagnostic_rows=[];diagnostic_failures=[];diagnostic_bindings=[]
    for row in parameter_rows:
        if not row.get('refitted_diagnostic_success'):
            diagnostic_failures.append({'draw':row['draw'],'exception':row.get('refitted_diagnostic_exception')});continue
        receipt=row['refitted_diagnostic_binding'];path=Path(receipt['path'])
        assert sha(path)==receipt['sha256']
        content=json.loads(path.read_text());assert content['metrics']==row['refitted_diagnostic_metrics']
        assert set(content['metrics'])==set(data)
        assert np.isfinite(list(content['metrics'].values())).all()
        diagnostic_rows.append(row);diagnostic_bindings.append({'draw':row['draw'],**receipt})
    diagnostic_table={}
    for name,value in data.items():
        values=np.array([r['refitted_diagnostic_metrics'][name] for r in diagnostic_rows])
        diagnostic_table[name]={'data':value,'reference_successes':len(values),
            'refitted_reference_q025_q50_q975':np.quantile(values,[.025,.5,.975]).tolist() if len(values) else None,
            'descriptive_upper_tail_rank':float((1+np.sum(values>=value))/(len(values)+1)) if len(values) else None,
            'fixed_parameter_reference':comparison['comparisons'][name]}
    full=len(rows)==declared and source['status']=='passed' and information_binding is not None
    failures=count['failed_or_incomplete_audited'] or diagnostic_failures or len(diagnostic_rows)<declared
    status=('complete_with_failures' if failures else 'complete') if full else 'partial'
    result={'status':status,'fit_sha256':source['fit_sha256'],'recovery_audit_sha256':sha(args.audit/'audit.json'),
        'hardware_diagnostics_sha256':sha(data_path),'fixed_reference_sha256':sha(diagnostic_source),
        'coverage':count,'accepted_audited_free_fits':len(parameter_rows),'parameter_sampling_summary':table,
        'local_information_audit_binding':information_binding,'sampling_vs_local_active_face':local_comparison,
        'refitted_diagnostic_successes':len(diagnostic_rows),'refitted_diagnostic_failures':diagnostic_failures,
        'refitted_diagnostic_unavailable_all_declared':declared-len(diagnostic_rows),
        'diagnostic_source_bindings':diagnostic_bindings,'refitted_diagnostic_comparisons':diagnostic_table,
        'elapsed_s':time.monotonic()-begun,
        'notice':'Conditional repeated-data sampling under the selected fitted generating model, quantizer phase, hard nuisance bounds and design. Empirical estimator quantiles are not confidence intervals for hardware parameters. Refitted diagnostics account for parameter fitting within this family, not structure selection or an unknown device population. Tail ranks remain descriptive. Coverage retains all failed or missing draws; partial outputs are not fifty-draw calibration.'}
    if len(points):
        rs=np.exp(points[:,names.index('log_rS')]);lag=np.exp(points[:,names.index('log_input_lag')])
        result['lag_label_recovery']={'sensor_time_smaller_than_input_time_count':int(np.sum(rs<lag)),
            'accepted_draws':len(points),'unordered_fast_time_q025_q50_q975_s':np.quantile(np.minimum(rs,lag),[.025,.5,.975]).tolist(),
            'unordered_slow_time_q025_q50_q975_s':np.quantile(np.maximum(rs,lag),[.025,.5,.975]).tolist()}
    write(args.output/'summary.json',result)
    if not len(points):return
    fig,axes=plt.subplots(2,2,figsize=(8,6));gain=np.exp(points[:,names.index('log_K')]);gain0=np.exp(truth[names.index('log_K')])
    if len(gain)<3:axes[0,0].plot(100*(gain/gain0-1),np.ones(len(gain)),'o',color='tab:blue')
    else:axes[0,0].hist(100*(gain/gain0-1),bins=12,color='tab:blue',alpha=.7)
    axes[0,0].axvline(0,color='k',lw=1);axes[0,0].set(xlabel='Gain estimation error (%)',ylabel='Generated data sets')
    if len(rs)<3:
        axes[0,1].plot(np.minimum(rs,lag),np.ones(len(rs)),'o',label='Faster fitted time')
        axes[0,1].plot(np.maximum(rs,lag),np.ones(len(rs)),'o',label='Slower fitted time')
    else:
        axes[0,1].hist(np.minimum(rs,lag),bins=12,alpha=.6,label='Faster fitted time')
        axes[0,1].hist(np.maximum(rs,lag),bins=12,alpha=.6,label='Slower fitted time')
    axes[0,1].set(xlabel='Fitted time (s)',ylabel='Generated data sets');axes[0,1].legend(fontsize=7)
    ell=np.exp(points[:,names.index('log_ell_T1')]);sd=np.exp(points[:,names.index('log_latent_SD_T1')])
    ell0=fit['noise_parameters'][0]['ell_s'];sd0=fit['noise_parameters'][0]['latent_SD_C']
    axes[1,0].plot(ell,sd,'o',ms=3,alpha=.7,label='Refitted');axes[1,0].plot(ell0,sd0,'kx',ms=8,label='Generating fit')
    axes[1,0].set(xscale='log',yscale='log',xlabel='T1 fitted OU length (s)',ylabel='T1 fitted latent SD (C)')
    for axis,values,reference in [(axes[1,0].xaxis,ell,ell0),(axes[1,0].yaxis,sd,sd0)]:
        low=.95*min(float(values.min()),reference);high=1.05*max(float(values.max()),reference)
        axis.set_major_locator(FixedLocator(np.geomspace(low,high,4)))
        axis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:.3g}'))
        axis.set_minor_locator(NullLocator())
    axes[1,0].legend(fontsize=7)
    selected=['T1_score_input_correlation_abs','T1_score_maximum_phase_projection_fraction',
        'T2_score_max_abs_ACF_1_60','score_max_abs_cross_channel_correlation_minus10_10']
    labels=['T1 input','T1 phase','T2 ACF','Cross']
    for j,name in enumerate(selected):
        item=diagnostic_table[name];q=item['refitted_reference_q025_q50_q975']
        if q is not None:axes[1,1].plot([j,j],[q[0],q[2]],color='tab:blue',lw=2);axes[1,1].plot(j,q[1],'o',color='tab:blue',ms=3)
        axes[1,1].plot(j,item['data'],'x',color='tab:red',ms=7)
    axes[1,1].set(xticks=range(4),xticklabels=labels,ylabel='Code diagnostic magnitude')
    axes[1,1].text(.02,.98,'Blue: refitted reference 2.5/50/97.5%\nRed: hardware',ha='left',va='top',transform=axes[1,1].transAxes,fontsize=7)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle(('PARTIAL CONTROLLER CHECK: ' if status=='partial' else '')+str(len(rows))+'/'+str(declared)+' audited draws; conditional on the selected generating model',fontsize=10)
    fig.tight_layout();fig.savefig(args.output/'recovery_summary.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
