"""Bind audited numerical evidence to a concise physical-model follow-up report."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from run_mean import write_json


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--selected',nargs='+',required=True);args=parser.parse_args()
    analysis=json.loads((args.analysis/'analysis.json').read_text());r=args.results;models=analysis['models'];jobs=[];bindings={}
    for d in sorted(p for p in r.iterdir() if p.is_dir() and (p/'source_snapshot.json').exists()):
        if d==args.output:continue
        exit_path=d/'exit.json'
        if not exit_path.exists():raise AssertionError(('Nonterminal job',d.name))
        code=json.loads(exit_path.read_text())['exit_code'];snapshot=json.loads((d/'source_snapshot.json').read_text())
        jobs.append({'id':d.name,'exit_code':code,'source_commit':snapshot['commit'],'command':snapshot['command']})
        if code:raise AssertionError(('Failed job requires explicit disposition',d.name))
    fits={}
    for name in args.selected:
        p=r/name/'fit.json';f=json.loads(p.read_text());fits[name]=f;audit=r/(name+'_audit')/'audit.json';a=json.loads(audit.read_text())
        assert a['status']=='passed' and a['fit_sha256']==sha(p)==models[name]['fit_sha256']
        assert models[name]['arrays_sha256']==sha(r/name/'trajectories.npz')
        bindings[str(p.relative_to(r))]=sha(p);bindings[str(audit.relative_to(r))]=sha(audit)
        binding=r/name/'trajectories.npz';bindings[str(binding.relative_to(r))]=sha(binding)
    assert len({f['nobs'] for f in fits.values()})==1
    base=fits[args.selected[0]];best_name=min(args.selected,key=lambda n:fits[n]['nominal_AICc']);best=fits[best_name]
    text=['# Can coupled thermal balances explain the unequal lags?','',
        'The second temperature confirms substantial heat transfer to the unpowered heater. '
        'A symmetric four-state energy-balance model can represent that coupling, but it cannot by itself explain the fast T1 phase. '
        'Adding one shared first-order input lag tests an additional mechanism; fitting it does not identify its hardware origin.','',
        '## What was fitted','',
        'All twelve distinct ENV2 records, both T1 and T2, with Q2 identically zero. Commands use exact left holds on the recorded irregular timestamps. '
        'The initial hidden heater temperatures equal their respective first sensor readings; first observations are conditioned on. '
        'One common physical ambient offset per run is bounded by ±2 C relative to its first T1. '
        'The affine sensitivity bounds both ambient endpoints by the same ±2 C. '
        'Channel errors use separate OU-plus-measurement covariance parameters and scales; independence between channels is a working assumption. '
        'A shared constant alpha is absorbed into the gain.','',
        'The positive reciprocal balances are, for heater i and the other heater j,','',
        r'\[C_H\dot H_i=U_a(A-H_i)+U_s(S_i-H_i)+U_x(H_j-H_i)+\beta Q_i,\qquad C_S\dot S_i=U_s(H_i-S_i).\]','',
        'The estimated scale-free parameters are K=beta/Ua, tA=CH/Ua, rS=CS/Us, r=Us/Ua and q=Ux/Ua. '
        'Absolute capacities and conductances still need an independent power calibration. '
        'The optional command map is u+c*u*(1-u/100); the optional lag satisfies L*dz/dt=u-z after that map.','',
        '## Why four thermal states do not settle the phase question','',
        'For fixed ambient, common and differential sensor responses have transfers','',
        r'\[G_+=\frac{K}{t_A r_Ss^2+[t_A+r_S(1+r)]s+1},\quad G_-=\frac{K}{t_A r_Ss^2+[t_A+r_S(1+r+2q)]s+1+2q}.\]','',
        r'\[T_1/Q_1=(G_++G_-)/2,\qquad T_2/Q_1=(G_+-G_-)/2.\]','',
        'Each modal transfer has strictly negative imaginary part for positive frequency and positive parameters. Their sum does too. '
        'The base T1 phase therefore stays between -180 and 0 degrees. The fast sine measurements are about -193 degrees on the frequency-unwrapped branch. '
        'This is a restriction of the stated symmetric linear model, not a proof against every possible thermal network. '
        'Its transfer zeros also matter: four thermal poles do not make it equivalent to the earlier all-pole cascade.','',
        '## Matched two-output results','',
        '| Model | T1 RMSE, C | T2 RMSE, C | Nominal AICc | Projected gradient | Successful starts |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in args.selected:
        f=fits[n];text.append('| '+n+' | '+' | '.join([f"{f['RMSE_by_channel_C'][0]:.4f}",f"{f['RMSE_by_channel_C'][1]:.4f}",f"{f['nominal_AICc']:.2f}",f"{f['best']['projected_gradient']:.4g}",str(sum(s['success'] for s in f['starts']))+'/'+str(len(f['starts']))])+' |')
    iid_path=r/'pilot_lag_iid/fit.json';iid=json.loads(iid_path.read_text());bindings[str(iid_path.relative_to(r))]=sha(iid_path)
    text+=['',f"The IID pilot pushed the extra input lag to its {iid['parameters']['input_lag_s']:.2f} s lower bound, "
        f"with T1/T2 RMSE {iid['RMSE_by_channel_C'][0]:.3f}/{iid['RMSE_by_channel_C'][1]:.3f} C. "
        'The colored-error analysis supports a finite short lag while accepting more slow temperature mismatch. '
        'The error model changes the fitted dynamics as well as their uncertainty; a lag estimate should always name the fitting criterion.']
    text+=['','AICc compares the same two-output observations under the working Gaussian covariance. '
        'It is not compared with the earlier T1-only likelihood. A lower correlated-error objective need not lower raw temperature RMSE. '
        'These are retrospective fits; all records have informed model selection.','',
        f'The smallest nominal AICc in this bounded comparison is **{best_name}**. Its parameters are:','',
        '| Parameter | Estimate |','| --- | ---: |']
    for k,v in best['parameters'].items():
        if v is not None:text.append(f'| {k} | {v:.6g} |')
    q=best['parameters']['Ux_over_Ua'];K=best['parameters']['K_common']
    text+=['',f'K_common is the gain of T1+T2. Before the command map, the individual steady gains are '
        f'{K*(1+q)/(1+2*q):.5f} C/percent for T1 and {K*q/(1+2*q):.5f} C/percent for T2. '
        'It should not be read as the earlier single-channel gain. The four thermal modal time constants are '
        +', '.join(f'{x:.4g}' for x in sorted(best['thermal_mode_times_s']))+' s; the optional input lag is additional.']
    text+=['','## Residual structure and frequency response','',
        '| Model | Max absolute innovation ACF, lags 1–60 | Fast T1 input/innovation correlation range | Max absolute cross-channel innovation correlation, lags ±20 |',
        '| --- | ---: | ---: | ---: |']
    for n in args.selected:
        m=models[n];per=m['per_run'];acf=max(abs(x) for d in per for c in d['innovation_ACF_lags1to60'] for x in c)
        cross=max(abs(x) for d in per for x in d['innovation_cross_channel_lags_minus20to20'])
        fast=[max(abs(x) for x in d['innovation_input_lags_minus60to60'][0]) for d in per if 'sin_50_50' in d['name']]
        text.append(f'| {n} | {acf:.3f} | {min(fast):.3f}–{max(fast):.3f} | {cross:.3f} |')
    text+=['','These are descriptive maxima across many lags and runs, not single prespecified hypothesis tests. '
        'Residual cycles and cross-channel dependence must be assessed before treating the assumed covariance as an adequate stochastic law.','',
        '| Model | Fast T1 matched phase error range, degrees | Fast T1 matched/measured gain range |',
        '| --- | ---: | ---: |']
    for n in args.selected:
        f=[x for x in models[n]['frequency'] if x['channel']==1 and x['period_s']==30 and x['cutoff_s']==450]
        p=[x['matched_phase_error_deg'] for x in f];g=[x['matched_gain_ratio'] for x in f]
        text.append(f'| {n} | {min(p):.2f} to {max(p):.2f} | {min(g):.3f}–{max(g):.3f} |')
    f=[x for x in models[best_name]['frequency'] if x['channel']==2 and x['period_s']==30 and x['cutoff_s']==450]
    sd=[x['measured']['HAC']['60']['phase_SD_deg'] for x in f]
    weak_path=r/'weak_frequency_v2/weak_frequency.json';weak=json.loads(weak_path.read_text())
    w=[x for x in weak['rows'] if x['channel']==2 and x['period_s']==30 and x['cutoff_s']==450]
    count=sum(x['HAC']['60']['working95_ellipse_contains_zero'] for x in w)
    bindings[str(weak_path.relative_to(r))]=sha(weak_path)
    text+=['',f'The fast T2 phase has working HAC60 standard errors of {min(sd):.1f}–{max(sd):.1f} degrees across repeats. '
        f'The joint complex-gain 95% ellipses include zero in {count}/{len(w)} fast T2 repeats after 450 s. '
        'A phase confidence set from an ellipse containing zero is unbounded around the circle; the scalar phase bars are only delta-method diagnostics. '
        'Its small oscillation cannot reliably locate an additional fast mode. The slow rise still supplies coupling information. '
        'Frequency plots use identical finite windows for measured and simulated temperatures; the JSON retains 300, 450 and 600 s cutoffs.','',
        '![Joint trajectories](results/final_analysis/joint_trajectories.png)','',
        '![Joint frequency response](results/final_analysis/joint_frequency.png)','',
        '![Weak fast T2 response](results/weak_frequency_v2/fast_T2_complex.png)','',
        '## What can be interpreted physically','',
        'Unequal decay rates are physically plausible because different thermal capacities and conductances produce different eigenmodes. '
        'Their numerical values should not be assigned to particular hardware parts solely from a fitted cascade.','',
        '| Model | Mean-information condition number | Active shape bounds | Weakest local coordinate |',
        '| --- | ---: | --- | --- |']
    for n in args.selected:
        info=models[n]['information'];weak=max(info['weakest_unit_direction'],key=lambda k:abs(info['weakest_unit_direction'][k]))
        text.append(f"| {n} | {info['condition_number']:.3g} | {', '.join(info['active_shape_bounds']) or 'None within 1e-4'} | {weak} |")
    text+=['','These matrices fix the fitted covariance and project out the ambient nuisance tangent. '
        'They diagnose weak directions; they are not profile likelihood intervals and do not include noise re-estimation or model-selection uncertainty. '
        'Regular Wald intervals are inappropriate at active bounds.','',
        'In the limit of negligible sensor heat loading, the common sensor factor 1/(1+rS*s) and common command factor 1/(1+L*s) commute. '
        'Swapping their time constants leaves both command-to-temperature transfers unchanged. '
        'Ambient and startup assumptions can break this equivalence, so any apparent allocation can depend on those assumptions. '
        'An extra lag that improves the fit is therefore evidence for a missing dynamic contribution, not a calibrated actuator or sensor time constant.','',
        '## Open questions','',
        '- Does relaxing heater/sensor symmetry improve both outputs while preserving the fast phase evidence? The present comparison does not answer that.',
        '- How much of the remaining cross-channel dependence comes from common environmental disturbance versus mean-model error?',
        '- Can independently measured electrical power, sensor dynamics or ambient temperature distinguish the competing lag interpretations?',
        '- Would separately driving heater 2 identify modes that Q1-only excitation leaves weak? No new experiment was performed or designed here.','',
        '## Verification and scope','',
        'The frozen foundation checks compare exact holds with independent dimensional energy-balance integration, transfer inversion and the common/differential identities. '
        'Every selected fit has a separate dimensional-matrix forward audit and dense-Cholesky covariance/likelihood audit. '
        'The analysis checks its pole/residue reconstruction against the analytic transfer. '
        'Source commits, raw hashes, multistarts, solver outcomes and audit bindings are retained in `run_manifest.json` and `results/`. '
        'Numerical agreement verifies implementation, not the physical assumptions or scientific adequacy. The previous completed study and email delivery are unchanged.','',
        'The reciprocal balances follow the [Notre Dame four-state TC-Lab notes](https://ndcbe.github.io/controls/tclab/03.06-Four-State-Model.html). '
        'The [APMonitor state-space treatment](https://apmonitor.com/pdc/index.php/Main/TCLabStateSpace) also discusses thermal coupling and a sensor-lag approximation. '
        'The phase restriction and interpretation above are derived here. No unreturned literature-search result is used as evidence.','']
    report='\n'.join(text);(args.output/'report.md').write_text(report)
    bindings[str((args.analysis/'analysis.json').relative_to(r))]=sha(args.analysis/'analysis.json')
    write_json(args.output/'run_manifest.json',{'status':'complete_for_researcher_review','selected_models':args.selected,
        'jobs':jobs,'live_jobs':[],'bindings_sha256':bindings,'report_sha256':sha(args.output/'report.md'),
        'stationarity_warnings':[n for n in args.selected if fits[n]['stationarity_warning']],
        'visual_review':'See results/final_analysis/visual_review.json','prior_goal':'Complete; heartbeat remains paused',
        'scope':'Bounded physical two-output follow-up; no hardware, DoE, email, untouched validation or unique physical mechanism established.'})
    print(json.dumps({'status':'complete','models':args.selected,'report_sha256':sha(args.output/'report.md')}),flush=True)


if __name__=='__main__':main()
