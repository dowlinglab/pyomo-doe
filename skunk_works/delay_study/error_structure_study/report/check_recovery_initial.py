"""Verify new report claims against fully audited recovery and code-space data."""
import hashlib
import json
from pathlib import Path
import numpy as np


def check(study,text,markdown,binding):
    root=study/'results';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=[root/'recovery_b10_full50_summary/summary.json',root/'recovery_b10_full50_summary_audit/audit.json',
        root/'recovery_b10_full50_audit/audit.json',root/'initial_reported_code_summary/summary.json']
    for p in paths:
        binding(p);binding(p.parent/'exit.json');assert json.loads((p.parent/'exit.json').read_text())['exit_code']==0
    summary,audit,recovery,initial=[json.loads(p.read_text()) for p in paths]
    assert summary['status']=='complete' and audit['status']==recovery['status']=='passed'
    assert audit['summary_sha256']==sha(paths[0]) and summary['recovery_audit_sha256']==sha(paths[2])
    assert len(recovery['rows'])==audit['checked_draws']==50
    fit=json.loads((root/'joint_conductance_b10_export_v2/fit.json').read_text())
    names=fit['coordinate_names'];points=np.array([r['recovered_point'] for r in recovery['rows']])
    sensor=np.exp(points[:,names.index('log_rS')]);lag=np.exp(points[:,names.index('log_input_lag')])
    assert int(np.sum(sensor<lag))==15
    for field,values in [('fast',np.minimum(sensor,lag)),('slow',np.maximum(sensor,lag))]:
        q=np.quantile(values,[.025,.975]);assert f'{q[0]:.3f}--{q[1]:.3f}' in text and f'{q[0]:.3f}--{q[1]:.3f}' in markdown
    assert summary['coverage']['covered']==49 and '49/50' in text and '49/50' in markdown
    # Audit uses independent binomial-tail inversion; the complete receipt is
    # bound above. The report renders its rounded limits checked explicitly.
    low,high=summary['coverage']['binomial_95_Monte_Carlo_interval']
    assert f'{100*low:.2f}--{100*high:.2f}' in text and f'{100*low:.2f}--{100*high:.2f}' in markdown
    # Independently compute the two sampling/local ratios from saved points.
    info=json.loads((root/'joint_b10_information_v2_audit/audit.json').read_text())
    for coordinate,places in [('log_K',3),('log_tA',3)]:
        local=next(v for v in info['local_comparison'] if v['coordinate']==coordinate)['bin_observed_mean_and_noise_nuisance_SE']
        ratio=float(np.std(points[:,names.index(coordinate)],ddof=1)/local)
        assert f'{ratio:.{places}f}' in text and f'{ratio:.{places}f}' in markdown
    for key in ['T1_score_input_correlation_abs','T1_score_maximum_phase_projection_fraction',
                'T2_score_max_abs_ACF_1_60','score_max_abs_cross_channel_correlation_minus10_10']:
        row=summary['refitted_diagnostic_comparisons'][key]
        assert f"{row['refitted_reference_q025_q50_q975'][2]:.5f}" in text
    assert initial['status']=='complete' and initial['independent_scalar_code_expectation_max_error_C']<1e-10
    for case,bindings in initial['source_bindings_sha256'].items():
        for name,digest in bindings.items():
            path=Path(name);assert sha(path)==digest;binding(path)
    assert all(max(r['actual_reported_code_mean_max_difference_C'])<.00030 for r in initial['phase_comparisons'])
    assert all(max(r['exact_translated_nearest_code_mean_max_difference_C'])<3.1e-12 for r in initial['phase_comparisons'])
    for name in ['fixed_nearest','bins_nearest','hidden_heater_2C']:
        row=next(v for v in initial['cases'] if v['case']==name)
        values=[v['reported_code_RMSE_C'] for v in row['metrics']]
        assert f'{values[0]:.3f}/{values[1]:.3f}' in text
    row=next(v for v in initial['cases'] if v['case']=='hidden_heater_2C')
    assert f"{row['sensor_time_s']:.3f}" in text and f"{row['input_time_s']:.3f}" in text
    hidden=json.loads((root/'initial_remaining_b10/hidden_heater_2C/fit.json').read_text())
    maximum=float(np.max(abs(np.array(hidden['startup_offsets_C']))));assert f'{maximum:.3f}' in text and f'{maximum:.3f}' in markdown
    fig=root/'recovery_report_figure_v2/figure_receipt.json';review=fig.parent/'visual_review.json'
    for p in [fig,review]:binding(p)
    assert json.loads(review.read_text())['status']=='reviewed'
    assert sha(study/'report/images/recovery_report.pdf')==json.loads(fig.read_text())['outputs_sha256']['pdf']
    binding(study/'report/images/recovery_report.pdf')
