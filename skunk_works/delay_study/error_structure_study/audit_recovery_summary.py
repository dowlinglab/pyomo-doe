"""Independent arithmetic for the complete source-bound recovery summary.

Use scalar binomial-tail inversion rather than the producer's beta quantiles,
manual sorted interpolation rather than np.quantile, and saved selected-fit
objectives/points rather than trusting the summary's numbers.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from run_bounded import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual,expected):
    assert math.isfinite(float(actual)) and math.isfinite(float(expected))
    assert abs(actual-expected)<1e-10*(1+abs(expected)),(actual,expected)


def quantile(values,probability):
    ordered=sorted(values);at=(len(ordered)-1)*probability;low=int(math.floor(at));high=int(math.ceil(at))
    return ordered[low]+(at-low)*(ordered[high]-ordered[low])


def binomial_limits(k,n):
    def tail(p,lower):
        indices=range(k,n+1) if lower else range(k+1)
        return math.fsum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in indices)
    result=[]
    for lower in [True,False]:
        if lower and k==0:result.append(0.);continue
        if not lower and k==n:result.append(1.);continue
        left=0.;right=1.
        for _ in range(100):
            mid=(left+right)/2;value=tail(mid,lower)
            if (value<.025 if lower else value>.025):left=mid
            else:right=mid
        result.append((left+right)/2)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--data-diagnostics',type=Path,required=True)
    p.add_argument('--information-audit',type=Path,required=True);args=p.parse_args()
    summary=json.loads((args.input/'summary.json').read_text());audit=json.loads((args.audit/'audit.json').read_text())
    fit=json.loads((args.fit/'fit.json').read_text());bindings={}
    def bind(path):bindings[str(path)]=sha(path)
    for directory in [args.input,args.audit,args.fit,args.data_diagnostics,args.information_audit]:
        assert json.loads((directory/'exit.json').read_text())['exit_code']==0;bind(directory/'exit.json')
    assert summary['status']=='complete' and audit['status']=='passed'
    assert summary['recovery_audit_sha256']==sha(args.audit/'audit.json')
    assert summary['fit_sha256']==audit['fit_sha256']==sha(args.fit/'fit.json')
    for path in [args.input/'summary.json',args.audit/'audit.json',args.fit/'fit.json']:bind(path)
    rows=audit['rows'];assert [r['draw'] for r in rows]==list(range(50)) and audit['declared_draws']==50
    assert all(r['status']=='passed' and r['coverage_complete'] and r['free_fit_success'] for r in rows)
    original={}
    for item in audit['shards']:
        path=Path(item['path'])/'draws.json';assert sha(path)==item['draws_sha256'];bind(path)
        for row in json.loads(path.read_text()):assert row['draw'] not in original;original[row['draw']]=row
    assert set(original)==set(range(50));points=[];covered=0;likelihood_ratios=[]
    for row in rows:
        source=original[row['draw']]
        free=Path(row['selected_free_output'])/'fit.json';null=Path(source['selected_null_output'])/'fit.json'
        checks={r['role']:r for r in row['numerical_audits']}
        assert checks['selected_free_output']['status']==checks['selected_null_output']['status']=='passed'
        assert sha(free)==checks['selected_free_output']['fit_sha256'] and sha(null)==checks['selected_null_output']['fit_sha256']
        bind(free);bind(null);f=json.loads(free.read_text());z=json.loads(null.read_text())
        assert source['selected_free_output']==row['selected_free_output']
        assert f['best']['point']==row['recovered_point']
        close(z['fixed_coordinates']['log_K'],fit['best']['point'][fit['coordinate_names'].index('log_K')])
        lr=max(0.,2*(z['best']['nll']-f['best']['nll']));close(row['gain_LR'],lr)
        hit=lr<=audit['config']['nominal_gain_LR_cutoff'];assert hit==row['covered']
        covered+=hit;points.append(f['best']['point']);likelihood_ratios.append(lr)
    count=summary['coverage'];assert count['declared']==count['audited_attempts']==count['complete']==50
    assert count['covered']==covered and count['missing']==count['failed_or_incomplete_audited']==0
    close(count['coverage_among_complete'],covered/50)
    assert summary['accepted_audited_free_fits']==50
    for value in count['failure_worst_case_bounds_all_declared']:close(value,covered/50)
    interval=binomial_limits(covered,50)
    for actual,expected in zip(count['binomial_95_Monte_Carlo_interval'],interval):close(actual,expected)
    # Closed-form all-success control is independent of any beta-quantile code.
    endpoint=binomial_limits(50,50);close(endpoint[0],.025**(1/50));close(endpoint[1],1.)
    table={r['coordinate']:r for r in summary['parameter_sampling_summary']};names=fit['coordinate_names']
    assert set(table)==set(names);moments={}
    for j,name in enumerate(names):
        values=[row[j] for row in points];mean=math.fsum(values)/50
        sd=math.sqrt(math.fsum((value-mean)**2 for value in values)/49);truth=fit['best']['point'][j]
        row=table[name];close(row['mean_coordinate_bias'],mean-truth);close(row['coordinate_sampling_SD'],sd)
        close(row['generating_coordinate'],truth)
        close(row['generating_physical_value'],math.exp(truth) if name.startswith('log_') else truth)
        physical=[math.exp(v) for v in values] if name.startswith('log_') else values
        for key,source in [('coordinate_q025_q50_q975',values),('physical_q025_q50_q975',physical)]:
            for actual,q in zip(row[key],[.025,.5,.975]):close(actual,quantile(source,q))
        close(row['lower_bound_fraction'],sum(v-fit['bounds'][j][0]<1e-5 for v in values)/50)
        close(row['upper_bound_fraction'],sum(fit['bounds'][j][1]-v<1e-5 for v in values)/50)
        moments[name]={'bias':mean-truth,'SD':sd}
    information=json.loads((args.information_audit/'audit.json').read_text());bind(args.information_audit/'audit.json')
    assert information['status']=='passed' and information['fit_sha256']==summary['fit_sha256']
    assert summary['local_information_audit_binding']['sha256']==sha(args.information_audit/'audit.json')
    info_rows={r['coordinate']:r for r in information['local_comparison']}
    for row in summary['sampling_vs_local_active_face']:
        name=row['coordinate'];se=info_rows[name]['bin_observed_mean_and_noise_nuisance_SE']
        close(row['generating_point_active_face_SE'],se);close(row['refitted_sampling_SD'],moments[name]['SD'])
        close(row['sampling_SD_to_active_face_SE'],moments[name]['SD']/se);close(row['mean_bias_to_active_face_SE'],moments[name]['bias']/se)
    rs=[math.exp(row[names.index('log_rS')]) for row in points];lag=[math.exp(row[names.index('log_input_lag')]) for row in points]
    labels=summary['lag_label_recovery'];assert labels['accepted_draws']==50
    assert labels['sensor_time_smaller_than_input_time_count']==sum(a<b for a,b in zip(rs,lag))
    for key,values in [('unordered_fast_time_q025_q50_q975_s',[min(a,b) for a,b in zip(rs,lag)]),
                       ('unordered_slow_time_q025_q50_q975_s',[max(a,b) for a,b in zip(rs,lag)])]:
        for actual,q in zip(labels[key],[.025,.5,.975]):close(actual,quantile(values,q))
    data_path=args.data_diagnostics/'data_diagnostics.json';reference_path=args.data_diagnostics/'comparison.json'
    assert sha(data_path)==summary['hardware_diagnostics_sha256'] and sha(reference_path)==summary['fixed_reference_sha256']
    bind(data_path);bind(reference_path);data=json.loads(data_path.read_text())['metrics']
    fixed=json.loads(reference_path.read_text())['comparisons'];diagnostics=[]
    for row in rows:
        assert row['refitted_diagnostic_success'];item=row['refitted_diagnostic_binding'];path=Path(item['path'])
        assert sha(path)==item['sha256'];bind(path);metrics=json.loads(path.read_text())['metrics']
        assert metrics==row['refitted_diagnostic_metrics'] and set(metrics)==set(data);diagnostics.append(metrics)
    assert summary['refitted_diagnostic_successes']==50 and summary['refitted_diagnostic_unavailable_all_declared']==0 and not summary['refitted_diagnostic_failures']
    assert set(summary['refitted_diagnostic_comparisons'])==set(data)
    for name,value in data.items():
        row=summary['refitted_diagnostic_comparisons'][name];values=[d[name] for d in diagnostics]
        close(row['data'],value);assert row['reference_successes']==50 and row['fixed_parameter_reference']==fixed[name]
        for actual,q in zip(row['refitted_reference_q025_q50_q975'],[.025,.5,.975]):close(actual,quantile(values,q))
        close(row['descriptive_upper_tail_rank'],(1+sum(v>=value for v in values))/51)
    write(args.output/'audit.json',{'status':'passed','summary_sha256':sha(args.input/'summary.json'),
        'checked_draws':50,'checked_selected_fit_objectives':100,'checked_coordinates':len(names),
        'checked_refitted_diagnostic_statistics':len(data),'independent_binomial_95_MC_interval':interval,
        'covered':covered,'likelihood_ratio_min_max':[min(likelihood_ratios),max(likelihood_ratios)],
        'source_bindings_sha256':bindings,
        'scope':'Independent scalar binomial inversion, saved-fit LR/count arithmetic, sample moments, manual order-statistic interpolation, bound/label counts and all source-bound refitted diagnostic summary values. Does not refit, prove global minima, recheck physical trajectories/likelihood quadrature, validate model adequacy or calibrate every local transfer mixture.'})


if __name__=='__main__':main()
