"""Check final nuisance-profile, startup and Taylor claims from saved evidence.

Recomputes profile differences and convergence decisions. Numerical point
propagation/likelihood checks remain separately source-bound prerequisites.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def check(study,text,markdown,binding):
    root=study/'results';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def read(path):
        binding(path)
        return json.loads(path.read_text())
    def terminal(directory):
        result=read(directory/'exit.json');assert result['exit_code']==0
    profiles=root/'profile_remaining_b10';terminal(profiles)
    # The measured queue owns separate profile and audit directories. Bind
    # each actual terminal collection; reject a plotting-only value witness.
    figure=root/'remaining_profiles_figure_v3';terminal(figure)
    values=read(figure/'figure_values.json')
    receipt=read(figure/'figure_receipt.json');review=read(figure/'visual_review.json')
    assert receipt['points']==28 and review['status']=='reviewed'
    assert receipt['figure_values_sha256']==sha(figure/'figure_values.json')
    assert sha(study/'report/images/remaining_profiles.pdf')==receipt['outputs_sha256']['pdf']
    binding(study/'report/images/remaining_profiles.pdf')
    for name,digest in values['source_bindings_sha256'].items():
        path=Path(name);assert sha(path)==digest;binding(path)
    count=0
    for row in values['profiles']:
        # Figure metadata names the measured sources; independently locate
        # profiles by their coordinate rather than depending on subplot order.
        found=[]
        for path in profiles.rglob('profile.json'):
            content=json.loads(path.read_text())
            if content['coordinate']==row['coordinate']:found.append((path,content))
        assert len(found)==1,(row['coordinate'],len(found))
        source,content=found[0];binding(source)
        accepted=[r for r in content['points'] if r['success']]
        assert len(accepted)==len(row['x'])
        audit_paths=[]
        for path in profiles.rglob('audit.json'):
            a=json.loads(path.read_text())
            if a.get('source_sha256')==sha(source) and a.get('kind')=='profile':audit_paths.append((path,a))
        assert len(audit_paths)==1
        ap,a=audit_paths[0];binding(ap)
        assert a['status']=='passed' and len(a['rows'])==len(accepted)
        for point in accepted:
            fitpath=Path(point['output'])/'fit.json';fit=read(fitpath)
            delta=2*(fit['best']['nll']-content['reference_nll'])
            j=int(np.argmin(abs(np.array(row['x'])-point['physical_value'])))
            assert abs(row['x'][j]-point['physical_value'])<1e-9
            assert abs(row['twice_nll_increase'][j]-delta)<1e-8
            matching=[v for v in a['rows'] if abs(v['coordinate_value']-point['coordinate_value'])<1e-10]
            assert len(matching)==1 and matching[0]['audit_status']=='passed'
            assert abs(point['nll']-fit['best']['nll'])<1e-9
            count+=1
        for x,digits in {'log_Us_over_Ua':[(.1,3),(.3,3)],'log_Ua_ratio_9':[(.925,3)],
                         'radiation_gamma':[(0.,3)]}.get(row['coordinate'],[]):
            j=int(np.argmin(abs(np.array(row['x'])-x)));formatted=f"{row['twice_nll_increase'][j]:.{digits}f}"
            assert formatted in text and formatted in markdown,formatted
    assert count==28
    startup=root/'startup_profile_summary';terminal(startup)
    summary=read(startup/'summary.json')
    assert summary['status']=='complete_with_declared_solver_failures' and not summary['baseline_refinement_required']
    assert summary['accepted_points']==5 and summary['declared_points']==6
    for name,digest in summary['source_bindings_sha256'].items():
        path=Path(name);assert sha(path)==digest;binding(path)
    failed=[v for v in summary['rows'] if not v['optimization_success']]
    assert len(failed)==1 and failed[0]['initial_H1_offset_6_C']==-2.
    assert 'nll' not in failed[0] and 'twice_nll_minus_original_reference' not in failed[0]
    assert len(failed[0]['unaccepted_start_outcomes'])==2
    for r in failed[0]['unaccepted_start_outcomes']:
        assert not r['success'] and not r['solver_success'] and 'Iteration limit' in r['message']
    zero=next(v for v in summary['rows'] if v['initial_H1_offset_6_C']==0.)
    delta=2*(zero['nll']-summary['reference_nll'])
    assert abs(delta-zero['twice_nll_minus_original_reference'])<1e-10
    for v in [delta]:assert f'{v:.3f}' in text and f'{v:.3f}' in markdown
    assert f"{zero['sensor_time_s']:.3f}/{zero['input_time_s']:.3f}" in text
    fig=read(startup/'figure_receipt.json');review=read(startup/'visual_review.json')
    assert review['status']=='reviewed' and fig['summary_sha256']==sha(startup/'summary.json')
    assert sha(study/'report/images/startup_profile.pdf')==fig['outputs_sha256']['pdf']
    binding(study/'report/images/startup_profile.pdf')
    # Recompute the Taylor discrepancy and adjacent-order/tail gate from the
    # saved independent forward likelihoods, rather than trusting a pass label.
    local=root/'local_taylor_quality';follow=root/'local_taylor_quadrature_followup'
    terminal(local);terminal(follow)
    original=read(local/'summary.json');q=read(follow/'summary.json');design=read(follow/'design.json')
    assert len(original['cases'])==3 and all(r['evaluated']==r['declared']==16 for r in original['cases'])
    assert q['status']=='completed_with_unresolved_quadrature' and q['source_summary_sha256']==sha(local/'summary.json')
    for case in original['cases']:
        for name,digest in case['source_bindings_sha256'].items():
            path=Path(name);assert sha(path)==digest;binding(path)
    representative=next(v for v in q['results'] if v['role']=='median_error')
    extreme=next(v for v in q['results'] if v['role']=='worst')
    for r in q['results']:
        nll=np.array([v['nll'] for v in r['evaluations']]);dev=np.array([v['twice_nll_minus_reference'] for v in r['evaluations']])
        assert np.max(abs(np.diff(dev)-2*np.diff(nll)))<1e-9
        changes=abs(np.diff(dev));assert np.max(abs(changes-r['adjacent_deviance_changes']))<1e-9
        passed=bool(max(changes[-2:])<.001)
        assert passed==r['passed_1e_minus3_highest_order_and_tail_gates']
        if passed:
            error=abs(dev[-1]-r['taylor_prediction']);assert abs(error-r['confirmed_absolute_taylor_error'])<1e-9
        else:assert r['confirmed_absolute_taylor_error'] is None
    assert representative['passed_1e_minus3_highest_order_and_tail_gates'] and not extreme['passed_1e_minus3_highest_order_and_tail_gates']
    for value in [representative['evaluations'][-1]['twice_nll_minus_reference'],representative['taylor_prediction'],representative['confirmed_absolute_taylor_error']]:
        assert f'{value:.3f}' in text and f'{value:.3f}' in markdown
    assert 'withheld' in text and 'withheld' in markdown
    return {'profile_points':count,'startup_accepted':5,'startup_failed':1,'Taylor_representative_absolute_error':representative['confirmed_absolute_taylor_error'],'extreme_deviance_withheld':True}
