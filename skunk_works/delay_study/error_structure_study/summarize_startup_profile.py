"""Interpret the audited startup grid without promoting failed optimizations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from continuation import parent_receipt
from run_bounded import write


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--reference-fit',type=Path,required=True);p.add_argument('--reference-audit',type=Path,required=True)
    args=p.parse_args();paths=[args.input/'profile.json',args.audit/'audit.json',args.reference_fit/'fit.json',args.reference_audit/'audit.json']
    for d in [args.input,args.audit,args.reference_fit,args.reference_audit]:assert parent_receipt(d,require_terminal=True)['exit']['exit_code']==0
    profile,checked,reference,refcheck=[json.loads(q.read_text()) for q in paths]
    assert profile['fit_sha256']==refcheck['fit_sha256']==sha(paths[2]) and refcheck['status']=='passed'
    assert checked['kind']=='profile' and checked['source_sha256']==sha(paths[0])
    assert checked['status'] in ['passed','completed_with_failures']
    assert len(profile['points'])==len(checked['rows'])==6
    assert profile['coordinate']=='initial_H1_offset_6' and reference['startup_band_C']==2.
    bindings={str(q):sha(q) for q in paths};rows=[]
    for j,(point,auditrow) in enumerate(zip(profile['points'],checked['rows'])):
        assert auditrow['optimization_success']==point['success']
        assert abs(auditrow['coordinate_value']-point['coordinate_value'])<1e-12
        directory=Path(point['output']);task=directory/'task_exit.json';result=json.loads(task.read_text())
        assert result['success']==point['success'];bindings[str(task)]=sha(task)
        row={'initial_H1_offset_6_C':point['coordinate_value'],'optimization_success':point['success']}
        if point['success']:
            fitpath=directory/'fit.json';fit=json.loads(fitpath.read_text());auditpath=args.audit/('point_'+str(j))/'audit.json';check=json.loads(auditpath.read_text())
            assert auditrow['audit_status']==check['status']=='passed'
            assert check['fit_sha256']==sha(fitpath) and check['arrays_sha256']==sha(directory/'trajectories.npz')
            assert fit['fixed_coordinates'][profile['coordinate']]==point['coordinate_value']
            difference=2*(fit['best']['nll']-reference['best']['nll']);assert abs(difference-point['twice_nll_minus_reference'])<1e-9
            row.update({'nll':fit['best']['nll'],'twice_nll_minus_original_reference':difference,
                'sensor_time_s':fit['parameters']['rS_s'],'input_time_s':fit['parameters']['input_lag_s'],
                'K':fit['parameters']['K_common_nominal'],'latent_reference_RMSE_C':fit['RMSE_by_channel_C']})
            for q in [fitpath,auditpath,directory/'trajectories.npz']:bindings[str(q)]=sha(q)
        else:
            assert auditrow['audit_status']=='not_audited_failed_optimization'
            starts=directory/'starts.json';attempts=json.loads(starts.read_text());bindings[str(starts)]=sha(starts)
            row.update({'exception':result.get('exception'),'unaccepted_start_outcomes':[
                {k:v.get(k) for k in ['start','solver_success','message','nll','stationarity','success']} for v in attempts],
                'notice':'No accepted profile value or likelihood deviance assigned to this failed task.'})
        rows.append(row)
    lower=any(r['optimization_success'] and r['nll']<reference['best']['nll']-1e-3 for r in rows)
    assert lower==profile['lower_nll_than_reference']
    result={'status':'baseline_refinement_required' if lower else 'complete_with_declared_solver_failures' if any(not r['optimization_success'] for r in rows) else 'complete',
        'reference_nll':reference['best']['nll'],'reference_coordinate_C':profile['reference_coordinate'],
        'accepted_points':sum(r['optimization_success'] for r in rows),'declared_points':6,'rows':rows,
        'baseline_refinement_required':lower,'source_bindings_sha256':bindings,
        'scope':'Coarse representative nuisance grid with all other startup/dynamic/ambient/noise terms refit. Failed tasks retained without a numerical profile value. No global minimum, confidence endpoint, chi-square exclusion or initial-noise posterior claim. A lower accepted point requires unconstrained reference refinement before LR interpretation.'}
    write(args.output/'summary.json',result)
    x=np.array([r['initial_H1_offset_6_C'] for r in rows]);success=np.array([r['optimization_success'] for r in rows])
    def values(key):return np.array([r[key] if r['optimization_success'] else np.nan for r in rows])
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(5.,2.7));axes[0].plot(x,values('twice_nll_minus_original_reference'),'o-',ms=3,lw=.8)
    axes[0].plot(profile['reference_coordinate'],0.,'k*',ms=6);axes[0].set(ylabel='Twice nll difference',title='(a) Startup profile')
    axes[1].plot(x,values('sensor_time_s'),'o-',ms=3,label='Sensor time');axes[1].plot(x,values('input_time_s'),'s-',ms=3,label='Input lag')
    axes[1].set(ylabel='Fitted time (s)',title='(b) Lag-label sensitivity');axes[1].legend(fontsize=7)
    for ax in axes:ax.set(xlabel='Initial heater offset (C)',xticks=[-2,-1,0,1,2],xlim=(-2.2,2.2));ax.grid(alpha=.15)
    if not np.all(success):axes[0].text(.03,.96,'Failed offsets: '+', '.join(str(v) for v in x[~success]),transform=axes[0].transAxes,va='top',fontsize=7,bbox={'facecolor':'white','edgecolor':'none'})
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(args.output/('startup_profile.'+suffix),dpi=240)
    plt.close(fig)
    write(args.output/'figure_receipt.json',{'summary_sha256':sha(args.output/'summary.json'),
        'outputs_sha256':{suffix:sha(args.output/('startup_profile.'+suffix)) for suffix in ['png','pdf']},
        'notice':'Failed grid points break the plotted lines. No rejected trial objective is displayed as a validated profile value.'})


if __name__=='__main__':main()
