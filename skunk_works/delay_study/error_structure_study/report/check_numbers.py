"""Bind intermediate report values to terminal fits and independent audits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;DELAY=STUDY.parent
sys.path.insert(0,str(DELAY/'coupled_model_study'))
import numpy as np
from physical import load_records


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    text=(HERE/'main.tex').read_text();bindings={};checked=[];records=load_records()
    def binding(path):bindings[str(path.relative_to(DELAY))]=sha(path)
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
    args.output.write_text(json.dumps({'status':'passed','source_sha256':sha(HERE/'main.tex'),
        'checker_source_sha256':sha(Path(__file__)),
        'python':platform.python_version(),'data_count_including_initial_readings':count,'Gaussian_table':checked,
        'Gaussian_radiation_nll_improvement':radiation_delta,'joint_bin_RMSE_C':jr.tolist(),
        'source_bindings_sha256':bindings,
        'scope':'Intermediate numeric values, raw observation/fit/audit bindings, direct RMSE arithmetic and displayed rounding. Does not certify pending scientific gates, complete-goal status, writing interpretation or PDF layout.'},indent=2)+'\n')
    print(json.dumps({'status':'passed','table_rows':len(table),'bound_files':len(bindings),'raw_temperature_count':count}))


if __name__=='__main__':main()
