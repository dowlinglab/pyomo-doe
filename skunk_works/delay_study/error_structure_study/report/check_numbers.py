"""Bind intermediate report values to terminal fits and independent audits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;DELAY=STUDY.parent
sys.path.insert(0,str(STUDY))
sys.path.insert(0,str(DELAY/'coupled_model_study'))
import numpy as np
from physical import load_records


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    text=(HERE/'main.tex').read_text();bindings={};checked=[];records=load_records()
    def binding(path):bindings[str(path.relative_to(DELAY))]=sha(path)
    # Historical T1-only benchmark: reuse its independently audited fit, bind
    # unchanged raw data, and recompute the scalar SSE without ranking unlike
    # single-/two-output likelihoods or ambient definitions.
    prior=DELAY/'shared_model_investigation/results'
    directory=prior/'mapped_unequal3';historical=json.loads((directory/'fit.json').read_text())
    audit_path=prior/'mapped_unequal_audit/mapped_unequal3_audit/audit.json'
    historical_audit=json.loads(audit_path.read_text())
    assert historical_audit['status']=='passed' and historical_audit['fit_sha256']==sha(directory/'fit.json')
    assert json.loads((directory/'exit.json').read_text())['exit_code']==0
    assert json.loads((prior/'mapped_unequal_audit/exit.json').read_text())['exit_code']==0
    assert historical['kind']=='mapped_unequal' and historical['state_count']==3
    assert historical['ambient_mode']=='linear' and historical['ambient_width']==2
    assert historical['record_names']==[d['name'] for d in records]
    source=json.loads((directory/'source_snapshot.json').read_text())
    arrays=np.load(directory/'trajectories.npz');sse=0.;nobs=0
    for j,d in enumerate(records):
        assert source['sha256']['data/'+d['name']]==d['sha256']
        assert np.array_equal(arrays['time_'+str(j)],d['t'])
        assert np.array_equal(arrays['input_'+str(j)],d['u'])
        assert np.array_equal(arrays['observed_'+str(j)],d['y'][:,0])
        error=d['y'][1:,0]-arrays['predicted_'+str(j)][1:];sse+=float(error@error);nobs+=len(error)
    benchmark_rmse=float(np.sqrt(sse/nobs));assert nobs==historical['nobs']==10795
    benchmark_nobs=nobs
    assert abs(benchmark_rmse-historical['RMSE_C'])<1e-12
    assert f'{benchmark_rmse:.3f}\\C' in text
    for p in [directory/'fit.json',directory/'trajectories.npz',directory/'source_snapshot.json',
        directory/'exit.json',audit_path,prior/'mapped_unequal_audit/exit.json']:binding(p)
    def audited(root,name,audit_name=None):
        directory=root/name;f=json.loads((directory/'fit.json').read_text());a=root/(audit_name or name+'_audit')
        assert json.loads((directory/'exit.json').read_text())['exit_code']==0
        assert json.loads((a/'exit.json').read_text())['exit_code']==0
        check=json.loads((a/'audit.json').read_text());assert check['status']=='passed' and check['fit_sha256']==sha(directory/'fit.json')
        for p in [directory/'fit.json',directory/'trajectories.npz',directory/'exit.json',a/'audit.json',a/'exit.json']:binding(p)
        arrays=np.load(directory/'trajectories.npz');sse=np.zeros(2);count=0
        for j,d in enumerate(records):
            assert f['raw_sha256'][d['name']]==d['sha256'] and np.array_equal(d['y'],arrays['observed_'+str(j)])
            residual=d['y'][1:]-arrays['predicted_'+str(j)][1:];sse+=np.sum(residual**2,axis=0);count+=len(residual)
        rmse=np.sqrt(sse/count);assert np.max(abs(rmse-f['RMSE_by_channel_C']))<1e-10
        return f,rmse
    root=STUDY/'results';old,rmse=audited(DELAY/'coupled_model_study/results','main_lag_curve_ou')
    table=[('Shared short lag and curved command',old,rmse)]
    labels=[('band_05_constant','Conductance band 5\\%'),('band_10_constant','Conductance band 10\\%'),
            ('band_20_constant','Conductance band 20\\%'),('band_10_affine','10\\% band, affine ambient'),
            ('band_10_startup2','10\\% band, hidden heater starts'),('radiation_free_scaled','10\\% convective band, free radiation')]
    fits={}
    for name,label in labels:
        f,r=audited(root,name);fits[name]=f;table.append((label,f,r))
    for label,f,r in table:
        expected=f'{label} & {r[0]:.3f} & {r[1]:.3f}'
        assert expected in text,expected;checked.append({'label':label,'RMSE_C':r.tolist()})
    startup5,_=audited(root,'band_10_startup5')
    assert max(abs(np.array(fits['band_10_startup2']['startup_offsets_C']).ravel()))<.35
    assert max(abs(np.array(startup5['startup_offsets_C']).ravel()))<.35
    assert np.max(abs(np.array(startup5['RMSE_by_channel_C'])-fits['band_10_startup2']['RMSE_by_channel_C']))<1e-6
    radiation_delta=fits['band_10_constant']['best']['nll']-fits['radiation_free_scaled']['best']['nll']
    assert f'{radiation_delta:.3f}' in text
    assert f"{fits['radiation_free_scaled']['parameters']['gamma_differential_50C_over_Ua']:.4f}" in text
    aggregate=root/'decomposition/aggregate.csv';binding(aggregate)
    row=next(r for r in csv.DictReader(aggregate.open()) if r['model']=='main_lag_curve_ou' and r['channel']=='1' and r['cutoff_s']=='0')
    for key,places in [('offset_fraction',1),('additional_trend_fraction',1),('additional_fundamental_fraction',2)]:
        assert f"{100*float(row[key]):.{places}f}\\%" in text
    count=sum(d['y'].size for d in records);assert f'{count:,}' in text
    for d in records:
        n=np.rint((d['y']+50)/.3223);assert np.max(abs(np.round(.3223*n-50,2)-d['y']))<1e-9
    scenarios=root/'radiation_scale/scenarios.csv';binding(scenarios)
    selected=[r for r in csv.DictReader(scenarios.open()) if float(r['surface_C'])==70.54]
    for key,places in [('heat_loss_W',3),('nonlinear_remainder_vs_50C_secant_W',4)]:
        values=[float(r[key]) for r in selected];assert f'{min(values):.{places}f}--{max(values):.{places}f}' in text
    limit=root/'loading_limit_check/audit.json';binding(limit);lf=json.loads(limit.read_text());assert lf['status']=='passed'
    r=next(r for r in lf['positive_loading_scenarios'] if r['Us_over_Ua']==.005);a,b=r['forced_waveform_swapped_lag_max_difference_by_channel_C']
    assert f'{a:.5f}/{b:.5f}' in text
    for k in [0,1]:
        path=root/('bin_noise_band10_T'+str(k+1)+'_v2/observation_fit.json');nf=json.loads(path.read_text());binding(path)
        p=nf['parameters'];measurement=p['sigma_total_analogue_C']*np.sqrt(p['independent_variance_fraction'])
        assert f'{measurement:.4f}' in text
    joint,jr=audited(root,'joint_conductance_b10_export_v2','joint_conductance_b10_audit')
    assert f'{jr[0]:.3f}/{jr[1]:.3f}' in text
    assert f"{joint['parameters']['K_common_nominal']:.4f}" in text and f"{joint['parameters']['tA_nominal_s']:.2f}" in text
    assert abs(joint['parameters']['K_common_nominal']/fits['band_10_constant']['parameters']['K_common_nominal']-1)<.001
    for key,places in [('latent_SD_C',4),('ell_s',2),('measurement_SD_C',4)]:
        a,b=[n[key] for n in joint['noise_parameters']];assert f'{a:.{places}f} & {b:.{places}f}' in text
    phase=root/'phase_reference_check/audit.json';binding(phase);assert json.loads(phase.read_text())['status']=='passed';assert '0.16115' in text
    path=root/'joint_b10_predictive_diagnostics/comparison.json';binding(path);diagnostic=json.loads(path.read_text())
    assert diagnostic['status']=='complete' and diagnostic['simulation_attempts']==50 and diagnostic['simulation_successes']==50 and diagnostic['simulation_failures']==0
    values=diagnostic['comparisons'];a=values['T1_max_abs_acf_1_60']['data_randomization_min_median_max'][1];b=values['T2_max_abs_acf_1_60']['data_randomization_min_median_max'][1]
    assert f'{a:.4f}/{b:.4f}' in text
    assert f"{values['max_abs_cross_channel_correlation_lags_minus10_10']['data_randomization_min_median_max'][1]:.4f}" in text
    assert f"{values['T1_max_transition_standardized_discrepancy_abs']['simulation_upper_tail_rank_for_data_median']:.4f}" in text
    for name,audit_name in [('joint_conductance_b05_export_v2','joint_conductance_b05_audit'),
                            ('joint_conductance_b20_export_v2','joint_conductance_b20_audit'),
                            ('joint_radiation_b10_export_v2','joint_radiation_b10_audit')]:
        f,r=audited(root,name,audit_name)
        assert f'{r[0]:.3f}/{r[1]:.3f}' in text and f"{f['best']['nll']:.4f}" in text
        if f['family']=='radiation':assert f"{joint['best']['nll']-f['best']['nll']:.3f}" in text
    assert f"{joint['best']['nll']:.4f}" in text
    lower,_=audited(root,'joint_b10_loading_lower')
    assert f"{joint['best']['nll']-lower['best']['nll']:.5f}" in text
    initial,ir=audited(root,'joint_b10_initial_sensor_bins')
    for value,places in [(initial['best']['nll'],4),(initial['parameters']['K_common_nominal'],4),
                          (initial['parameters']['input_lag_s'],5),(ir[0],3),(ir[1],3)]:
        assert f'{value:.{places}f}' in text
    condition,_=audited(root,'joint_b10_condition_first_code')
    for value,places in [(condition['noise_parameters'][0]['latent_SD_C'],4),
                          (condition['noise_parameters'][0]['ell_s'],1)]:assert f'{value:.{places}f}' in text
    information=root/'joint_b10_information_v2_audit/audit.json';binding(information)
    ia=json.loads(information.read_text());assert ia['status']=='passed' and ia['fit_sha256']==sha(root/'joint_conductance_b10_export_v2/fit.json')
    for name in ['log_K','log_tA','log_input_lag']:
        row=next(r for r in ia['local_comparison'] if r['coordinate']==name)
        for key in ['IID_known_analogue_variance_SE','correlated_known_analogue_covariance_SE','bin_observed_mean_and_noise_nuisance_SE']:
            assert f'{row[key]:.7f}' in text
    assert f"{ia['diffusion'][0]['conditional_log_SD_log_length_correlation']:.4f}" in text
    ratios={r['coordinate']:r['correlated_to_IID_analogue_SE_ratio'] for r in ia['local_comparison']}
    for name,places in [('log_K',2),('log_tA',2),('log_input_lag',3)]:assert f'{ratios[name]:.{places}f}' in text
    before=2*joint['noise_parameters'][0]['latent_SD_C']**2/joint['noise_parameters'][0]['ell_s']
    after=2*condition['noise_parameters'][0]['latent_SD_C']**2/condition['noise_parameters'][0]['ell_s']
    assert f'{100*abs(after/before-1):.2f}\\%' in text
    assert initial['nmean']-joint['nmean']==24 and 'adds 24' in text
    path=root/'joint_b10_nonrandom_diagnostics/comparison.json';binding(path);nr=json.loads(path.read_text())
    assert nr['status']=='complete' and nr['successes']==50 and nr['failures']==0
    for key in ['T1_score_input_correlation_abs','T1_score_maximum_phase_projection_fraction',
                'T2_score_max_abs_ACF_1_60','score_max_abs_cross_channel_correlation_minus10_10']:
        row=nr['comparisons'][key]
        assert f"{row['data']:.5f} & {row['reference_q025_q50_q975'][2]:.5f}" in text
        assert row['upper_tail_rank']==1/51
    path=root/'joint_b10_cross_localization_v2/comparison.json';binding(path);cross=json.loads(path.read_text())
    assert cross['status']=='complete' and cross['successes']==50 and cross['failures']==0
    row=cross['comparisons']['run_trend_and_phase_removed_pooled_cross_max_abs']
    for value in [row['data'],row['reference_q025_q50_q975'][2]]:assert f'{value:.5f}' in text
    forecasts=root/'updated_mean_diagnostics/forecast_horizons.csv';binding(forecasts)
    receipt=root/'updated_mean_diagnostics/receipt.json';binding(receipt)
    audit_path=root/'updated_mean_diagnostics_audit_v2/audit.json';binding(audit_path)
    check_path=root/'updated_diagnostics_check/check.json';binding(check_path)
    assert json.loads(audit_path.read_text())['status']=='passed' and json.loads(audit_path.read_text())['receipt_sha256']==sha(receipt)
    assert json.loads(check_path.read_text())['status']=='passed'
    forecast_rows=list(csv.DictReader(forecasts.open()))
    for horizon,expected_n in [(0.,10795),(60.,10072)]:
        values=[]
        for channel in [1,2]:
            chosen=[r for r in forecast_rows if r['model']=='joint_conductance_b10_export_v2' and float(r['cutoff_s'])==0. and float(r['horizon_s'])==horizon and int(r['channel'])==channel]
            assert len(chosen)==12 and len(set(r['name'] for r in chosen))==12
            nobs=sum(int(r['nobs']) for r in chosen);assert nobs==expected_n
            values.append(np.sqrt(sum(float(r['SSE_C2']) for r in chosen)/nobs))
        assert f'{values[0]:.3f}/{values[1]:.3f}\\C' in text
    for name,coordinates in [('gaussian_process_correlation',[('rho_OU_driving',4)]),
                             ('gaussian_measurement_correlation',[('rho_measurement',5)]),
                             ('gaussian_both_correlations',[('rho_OU_driving',5),('rho_measurement',5)])]:
        path=root/name/'fit.json';binding(path);f=json.loads(path.read_text())
        ap=root/(name+'_audit')/'audit.json';binding(ap);a=json.loads(ap.read_text())
        assert a['status']=='passed' and a['fit_sha256']==sha(path)
        for coordinate,places in coordinates:
            value=f['best']['point'][f['coordinate_names'].index(coordinate)]
            assert f'{value:.{places}f}' in text
        if name=='gaussian_both_correlations':assert f"{-a['nll_minus_prior_independent_Gaussian']:.5f}" in text
    # Independently pool all three canonical groups from the forecast arrays.
    # The groups reuse records; no pooled thirty-six-record inference is made.
    from continuation import parent_receipt
    summaryroot=root/'transfer_all_canonical_summary'
    summary=json.loads((summaryroot/'summary.json').read_text());binding(summaryroot/'summary.json')
    index=json.loads((summaryroot/'analysis_index_snapshot.json').read_text());binding(summaryroot/'analysis_index_snapshot.json')
    assert summary['status']=='partial' and json.loads((summaryroot/'exit.json').read_text())['exit_code']==0
    assert summary['all_declared_folds_accounted'] and not summary['all_required_uncertainty_components_accounted']
    assert len(index['transfer_analysis'])==20
    groups={};lookup={d['name']:d for d in records};first=None
    for entry in index['transfer_analysis']:
        kind='leave_run_out' if entry['fold_id'].startswith('run_') else 'leave_protocol_out'
        key=(entry['candidate'],kind)
        if key not in groups:groups[key]={'sse':np.zeros(2),'offsets':np.zeros(2),
            'covered':np.zeros((2,2)),'widths':np.zeros((2,2)),'nobs':0,'seen':[]}
        pooled=groups[key]
        fp=Path(entry['fit_directory'])/'fit.json';fa=Path(entry['fit_audit'])
        fcheck=json.loads(fa.read_text());assert fcheck['status']=='passed' and fcheck['fit_sha256']==sha(fp)
        assert parent_receipt(fa.parent,require_terminal=True)['exit']['exit_code']==0
        binding(fp);binding(fa)
        directory=Path(entry['predictions']['noise']['directory']);ap=Path(entry['predictions']['noise']['audit'])
        a=json.loads(ap.read_text());assert a['status']=='passed' and a['fit_sha256']==sha(fp)
        assert a['prediction_sha256']==sha(directory/'transfer_predictions.json') and a['arrays_sha256']==sha(directory/'transfer_predictions.npz')
        assert json.loads((directory/'exit.json').read_text())['exit_code']==0 and json.loads((ap.parent/'exit.json').read_text())['exit_code']==0
        for p in [directory/'transfer_predictions.json',directory/'transfer_predictions.npz',ap,directory/'exit.json',ap.parent/'exit.json']:binding(p)
        metadata=json.loads((directory/'transfer_predictions.json').read_text());arrays=np.load(directory/'transfer_predictions.npz')
        for j,row in enumerate(metadata['per_run']):
            record=lookup[row['name']];assert row['raw_sha256']==record['sha256']
            assert np.array_equal(arrays['observed_'+str(j)],record['y'])
            assert row['name'] not in metadata['train_names']
            pooled['seen'].append(row['name']);y=arrays['observed_'+str(j)][1:];error=y-arrays['reported_mean_'+str(j)]
            pooled['sse']+=np.sum(error**2,axis=0);pooled['offsets']+=len(y)*np.mean(error,axis=0)**2;pooled['nobs']+=len(y)
            for k,prefix in enumerate(['noise_only_reported','nuisance_noise_reported']):
                low=arrays[prefix+'_low_'+str(j)];high=arrays[prefix+'_high_'+str(j)]
                pooled['covered'][k]+=np.sum((y>=low)&(y<=high),axis=0);pooled['widths'][k]+=np.sum(high-low,axis=0)
        if entry['candidate']=='primary' and entry['fold_id']=='protocol_30s':first=entry
    definitions={('primary','leave_run_out'):('Band: omit one run','Band/run'),
        ('primary','leave_protocol_out'):('Band: omit protocol','Band/protocol'),
        ('shared','leave_protocol_out'):('Shared: omit protocol','Shared/protocol')}
    assert set(groups)==set(definitions) and '10,795' in text
    for key,(label,short) in definitions.items():
        pooled=groups[key];nobs=pooled['nobs'];sse=pooled['sse'];offsets=pooled['offsets']
        assert len(pooled['seen'])==len(set(pooled['seen']))==12 and nobs==10795
        result=next(r for r in summary['groups'] if (r['candidate'],r['fold_type'])==key)
        rmse=np.sqrt(sse/nobs);assert np.max(abs(rmse-result['reported_RMSE_by_channel_C']))<1e-10
        assert f'{label} & {rmse[0]:.3f} & {rmse[1]:.3f}' in text
        for k,kind in enumerate(['noise','conditions + noise']):
            cover=100*pooled['covered'][k]/nobs;width=pooled['widths'][k]/nobs
            prefix=['noise_only','hard_nuisance_noise'][k]
            assert np.max(abs(cover/100-result[prefix+'_marginal_coverage']))<1e-10
            assert np.max(abs(width-result[prefix+'_mean_width_C']))<1e-10
            expected=f'{short}, {kind} & {cover[0]:.2f}/{cover[1]:.2f} & {width[0]:.3f}/{width[1]:.3f}'
            assert expected in text,expected
        if key[0]=='primary':
            centered=np.sqrt((sse-offsets)/nobs);fraction=100*offsets/sse
            assert f'{fraction[0]:.1f}/{fraction[1]:.1f}\\%' in text
            assert f'{centered[0]:.3f}/{centered[1]:.3f}\\C' in text
    # Check the first local-mixture sensitivity directly, retaining its limits.
    mixtures={};total=0;first_width=np.zeros(2)
    for variant in ['noise','parameter_64','parameter_256']:
        entry=first['predictions'][variant];directory=Path(entry['directory']);ap=Path(entry['audit'])
        a=json.loads(ap.read_text());assert a['status']=='passed'
        assert a['prediction_sha256']==sha(directory/'transfer_predictions.json') and a['arrays_sha256']==sha(directory/'transfer_predictions.npz')
        assert json.loads((directory/'exit.json').read_text())['exit_code']==0 and parent_receipt(ap.parent,require_terminal=True)['exit']['exit_code']==0
        for path in [directory/'transfer_predictions.json',directory/'transfer_predictions.npz',ap]:binding(path)
        arrays=np.load(directory/'transfer_predictions.npz');mixtures[variant]=arrays
        width=np.zeros(2);nobs=0
        for j in range(3):
            prefix='nuisance_noise_reported' if variant=='noise' else 'parameter_mixture_reported'
            lo=arrays[prefix+'_low_'+str(j)];hi=arrays[prefix+'_high_'+str(j)]
            width+=np.sum(hi-lo,axis=0);nobs+=len(lo)
        if variant in ['noise','parameter_256']:
            assert f'{width[0]/nobs:.3f}/{width[1]/nobs:.3f}\\C' in text
    changes=[]
    for j in range(3):
        a=mixtures['parameter_64'];b=mixtures['parameter_256']
        changes.append(np.stack([abs(a['parameter_mixture_reported_'+edge+'_'+str(j)]-b['parameter_mixture_reported_'+edge+'_'+str(j)]) for edge in ['low','high']]))
    maximum=np.max([d.max(axis=(0,1)) for d in changes],axis=0)
    fraction=100*np.sum([np.sum(d>1e-9,axis=(0,1)) for d in changes],axis=0)/sum(d.shape[0]*d.shape[1] for d in changes)
    assert np.max(abs(maximum-.33))<1e-9 and '0.33 C' in text
    assert f'{fraction[0]:.2f}/{fraction[1]:.2f}\\%' in text
    ia_path=Path(first['information_audit'])/'audit.json';binding(ia_path);case_info=json.loads(ia_path.read_text())
    assert case_info['status']=='passed' and case_info['fit_sha256']==sha(Path(first['fit_directory'])/'fit.json')
    row=next(r for r in case_info['local_comparison'] if r['coordinate']=='log_input_lag')
    assert f"{row['bin_observed_mean_and_noise_nuisance_SE']:.4f}" in text
    # The coarse and exchanged-lag values retain their terminal point audits.
    profile=root/'profile_b10_short_lag_v2';pa=root/'profile_b10_short_lag_audit'
    content=json.loads((profile/'profile.json').read_text());check=json.loads((pa/'audit.json').read_text())
    assert check['status']=='passed' and check['source_sha256']==sha(profile/'profile.json') and len(check['rows'])==9
    for path in [profile/'profile.json',profile/'exit.json',pa/'audit.json',pa/'exit.json']:binding(path)
    for value in [7.,15.]:
        row=next(r for r in content['points'] if abs(r['physical_value']-value)<1e-9)
        assert f"{row['twice_nll_minus_reference']:.3f}" in text
    exchange,_=audited(root,'profile_b10_exchanged_lag_point')
    for value,places in [(exchange['parameters']['input_lag_s'],5),(exchange['parameters']['rS_s'],4),
        (2*(exchange['best']['nll']-joint['best']['nll']),5)]:assert f'{value:.{places}f}' in text
    path=root/'reference_transport_check/check.json';binding(path);assert json.loads(path.read_text())['status']=='passed'
    from transfer import reference_transport_bounds
    low,high=reference_transport_bounds(.1,12,9)
    assert f'[{low:.5f},{high:.5f}]' in text
    # Check the concise Markdown companion against the same independently
    # pooled arrays, not by comparing its text with the LaTeX transcription.
    markdown=(STUDY/'report.md').read_text()
    assert f'{benchmark_rmse:.3f} C' in markdown
    md_fits=[('Shared conductance','joint_conductance_b00_export_v3','joint_conductance_b00_audit'),
        ('5% run band','joint_conductance_b05_export_v2','joint_conductance_b05_audit'),
        ('10% run band','joint_conductance_b10_export_v2','joint_conductance_b10_audit'),
        ('20% run band','joint_conductance_b20_export_v2','joint_conductance_b20_audit'),
        ('Radiation, 10% convective band','joint_radiation_b10_export_v2','joint_radiation_b10_audit')]
    for label,name,audit_name in md_fits:
        selected,r=audited(root,name,audit_name)
        expected=f'| {label} | {r[0]:.3f} | {r[1]:.3f} | {selected["best"]["nll"]:.4f} |'
        assert expected in markdown,expected
        if selected['family']=='radiation':assert f'{joint["best"]["nll"]-selected["best"]["nll"]:.3f}' in markdown
    for key,places in [('K_common_nominal',4),('tA_nominal_s',2),('rS_s',2),('input_lag_s',2)]:
        assert f'{joint["parameters"][key]:.{places}f}' in markdown
    assert f'{count:,}' in markdown and 'round(0.3223*n - 50, 2)' in markdown
    old_partition=next(r for r in csv.DictReader(aggregate.open()) if r['model']=='main_lag_curve_ou' and r['channel']=='1' and r['cutoff_s']=='0')
    for key,places in [('offset_fraction',1),('additional_trend_fraction',1),('additional_fundamental_fraction',2)]:
        assert f'{100*float(old_partition[key]):.{places}f}%' in markdown
    for key,places in [('heat_loss_W',3),('nonlinear_remainder_vs_50C_secant_W',4)]:
        values=[float(r[key]) for r in csv.DictReader(scenarios.open()) if float(r['surface_C'])==70.54]
        assert f'{min(values):.{places}f}--{max(values):.{places}f}' in markdown
    for key in ['T1_score_input_correlation_abs','score_max_abs_cross_channel_correlation_minus10_10']:
        row=nr['comparisons'][key]
        for value in [row['data'],row['reference_q025_q50_q975'][2]]:assert f'{value:.5f}' in markdown
    row=cross['comparisons']['run_trend_and_phase_removed_pooled_cross_max_abs']
    for value in [row['data'],row['reference_q025_q50_q975'][2]]:assert f'{value:.5f}' in markdown
    for name,places in [('log_K',2),('log_tA',2),('log_input_lag',3)]:assert f'{ratios[name]:.{places}f}' in markdown
    for horizon in [0.,60.]:
        values=[]
        for channel in [1,2]:
            chosen=[r for r in forecast_rows if r['model']=='joint_conductance_b10_export_v2' and float(r['cutoff_s'])==0. and float(r['horizon_s'])==horizon and int(r['channel'])==channel]
            values.append(np.sqrt(sum(float(r['SSE_C2']) for r in chosen)/sum(int(r['nobs']) for r in chosen)))
        assert f'{values[0]:.3f}/{values[1]:.3f}' in markdown
    for value,places in [(exchange['parameters']['input_lag_s'],5),(exchange['parameters']['rS_s'],4),
        (2*(exchange['best']['nll']-joint['best']['nll']),5)]:assert f'{value:.{places}f}' in markdown
    md_labels={('primary','leave_run_out'):'10% band, omit one run',
        ('primary','leave_protocol_out'):'10% band, omit a protocol',
        ('shared','leave_protocol_out'):'Shared conductance, omit a protocol'}
    for key,label in md_labels.items():
        pooled=groups[key];rmse=np.sqrt(pooled['sse']/pooled['nobs'])
        assert f'| {label} | {rmse[0]:.3f} | {rmse[1]:.3f} |' in markdown
    pooled=groups[('primary','leave_protocol_out')];nobs=pooled['nobs']
    for k in [0,1]:
        cover=100*pooled['covered'][k]/nobs;assert f'{cover[0]:.2f}/{cover[1]:.2f}%' in markdown
    widths=pooled['widths'][1]/nobs;assert f'{widths[0]:.3f}/{widths[1]:.3f}' in markdown
    centered=np.sqrt((pooled['sse']-pooled['offsets'])/nobs);fraction=100*pooled['offsets']/pooled['sse']
    assert f'{fraction[0]:.1f}/{fraction[1]:.1f}%' in markdown and f'{centered[0]:.3f}/{centered[1]:.3f}' in markdown
    for variant in ['noise','parameter_256']:
        arrays=mixtures[variant];prefix='nuisance_noise_reported' if variant=='noise' else 'parameter_mixture_reported'
        width=sum(np.sum(arrays[prefix+'_high_'+str(j)]-arrays[prefix+'_low_'+str(j)],axis=0) for j in range(3))
        nobs=sum(len(arrays[prefix+'_high_'+str(j)]) for j in range(3));width=width/nobs
        assert f'{width[0]:.3f}/{width[1]:.3f}' in markdown
    from check_recovery_initial import check as recovery_initial_check
    recovery_initial_check(STUDY,text,markdown,binding)
    binding(HERE/'check_recovery_initial.py')
    args.output.write_text(json.dumps({'status':'passed','source_sha256':sha(HERE/'main.tex'),
        'markdown_source_sha256':sha(STUDY/'report.md'),
        'checker_source_sha256':sha(Path(__file__)),
        'python':platform.python_version(),'data_count_including_initial_readings':count,'Gaussian_table':checked,
        'Gaussian_radiation_nll_improvement':radiation_delta,'joint_bin_RMSE_C':jr.tolist(),
        'historical_T1_only_benchmark':{'RMSE_C':benchmark_rmse,'nobs':benchmark_nobs,
            'notice':'Different output set and affine ambient; no matched likelihood or AIC comparison.'},
        'source_bindings_sha256':bindings,
        'scope':'Intermediate numeric values, raw observation/fit/audit bindings, direct RMSE arithmetic and displayed rounding. Does not certify pending scientific gates, complete-goal status, writing interpretation or PDF layout.'},indent=2)+'\n')
    print(json.dumps({'status':'passed','table_rows':len(table),'bound_files':len(bindings),'raw_temperature_count':count}))


if __name__=='__main__':main()
