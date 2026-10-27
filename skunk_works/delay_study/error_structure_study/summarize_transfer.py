"""Aggregate audited retrospective folds without double-counting their records.

The execution manifest owns the analysis index. This driver snapshots that
index, then verifies every consumed fit/prediction/audit hash. Separate groups
cover primary leave-run, primary leave-protocol and shared leave-protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
import matplotlib.pyplot as plt
from physical import load_records
from run_transfer_fit import folds
from continuation import parent_receipt
from run_bounded import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction(entry,fit_hash,expected_names):
    root=Path(entry['directory']);audit_path=Path(entry['audit']);audit=json.loads(audit_path.read_text())
    parent_receipt(audit_path.parent,require_terminal=True)
    assert json.loads((root/'exit.json').read_text())['exit_code']==0
    assert audit['status']=='passed' and audit['fit_sha256']==fit_hash
    assert audit['prediction_sha256']==sha(root/'transfer_predictions.json')
    assert audit['arrays_sha256']==sha(root/'transfer_predictions.npz')
    metadata=json.loads((root/'transfer_predictions.json').read_text())
    assert metadata['test_names']==expected_names and metadata['fit_sha256']==fit_hash
    if metadata['parameter_draws']:
        source=metadata['parameter_draws'];assert sha(Path(source['path']))==source['sha256']
    arrays=np.load(root/'transfer_predictions.npz',allow_pickle=False)
    return metadata,arrays,{'directory':str(root),'audit':str(audit_path),'audit_sha256':sha(audit_path),
        'prediction_sha256':audit['prediction_sha256'],'arrays_sha256':audit['arrays_sha256']}


def parameter_chain(entry,fit_hash,variants):
    """Bind every consumed mixture to its accepted curvature and draw receipt."""
    used=set(variants)&{'parameter_64','parameter_256'}
    if not used:return None
    information=Path(entry['information']);auditor=Path(entry['information_audit']);draws=Path(entry['draws'])
    paths=[information/'information.json',information/'information_matrices.npz',auditor/'audit.json',draws/'draw_receipt.json']
    info=json.loads(paths[0].read_text());audit=json.loads(paths[2].read_text());receipt=json.loads(paths[3].read_text())
    for directory in [information,auditor,draws]:
        assert parent_receipt(directory,require_terminal=True)['exit']['exit_code']==0
        paths += [directory/'source_snapshot.json',directory/'exit.json']
    assert info['fit_sha256']==audit['fit_sha256']==receipt['fit_sha256']==fit_hash
    assert audit['status']=='passed' and receipt['status']=='complete'
    assert audit['information_sha256']==sha(information/'information.json')
    assert audit['matrix_sha256']==receipt['information_matrix_sha256']==sha(information/'information_matrices.npz')
    assert receipt['information_audit_sha256']==sha(auditor/'audit.json')
    assert receipt['fixed_active_coordinates']==info['active_coordinates']
    matrices=np.load(information/'information_matrices.npz');basis=matrices['active_face_basis']
    small=basis.T@matrices['observed_Hessian']@basis
    minimum=float(np.linalg.eigvalsh((small+small.T)/2)[0]);assert minimum>0
    for variant in used:
        count=int(variant.rsplit('_',1)[1]);metadata=variants[variant][0]['parameter_draws']
        path=Path(metadata['path']);assert path.resolve()==(draws/('parameter_draws_'+str(count)+'.npz')).resolve()
        assert receipt['parameter_files_sha256'][path.name]==metadata['sha256']==sha(path)
        data=np.load(path,allow_pickle=False)
        assert str(data['fit_sha256'])==fit_hash and data['points'].shape==(count,len(info['point']))
        paths.append(path)
    return {'source_bindings_sha256':{str(p):sha(p) for p in paths},
        'minimum_recomputed_active_face_eigenvalue':minimum,
        'notice':'Covariance/draw provenance and positive curvature verified; the local selected-basin approximation still excludes released-bound, initialization and model-selection uncertainty.'}


def aggregate_groups(rows,expected,record_names,partial=False):
    group_rows=[]
    for candidate,kind in [('primary','leave_run_out'),('primary','leave_protocol_out'),('shared','leave_protocol_out')]:
        selected=[r for r in rows if r['candidate']==candidate and r['fold_type']==kind and r['status']=='audited']
        runs=[d for r in selected for d in r['per_run']];run_names=[d['name'] for d in runs]
        assert len(set(run_names))==len(run_names),'A record was double-counted within a transfer group'
        required=[f for f in expected.values() if f['candidate']==candidate and f['type']==kind]
        if not partial and len(selected)==len(required):assert set(run_names)==set(record_names)
        count=sum(r['nobs_per_channel'] for r in runs)
        group={'candidate':candidate,'fold_type':kind,'declared_folds':len(required),'audited_folds':len(selected),
            'unique_test_records':len(run_names),'nobs_per_channel':count}
        if count:
            for prefix in ['latent','reported']:
                group[prefix+'_RMSE_by_channel_C']=np.sqrt(np.sum([r[prefix+'_SSE_by_channel_C2'] for r in runs],axis=0)/count).tolist()
            group['reported_signed_mean_error_C']=(np.sum([r['reported_signed_error_sum_C'] for r in runs],axis=0)/count).tolist()
            offsets=np.sum([r['reported_run_offset_SSE_C2'] for r in runs],axis=0)
            total=np.sum([r['reported_SSE_by_channel_C2'] for r in runs],axis=0);centered=total-offsets
            assert np.all(centered>=-1e-7)
            group['reported_run_offset_SSE_fraction']=(offsets/total).tolist()
            group['reported_per_run_centered_RMSE_C']=np.sqrt(np.maximum(centered,0.)/count).tolist()
            for variant in ['noise_only','hard_nuisance_noise','parameter_64','parameter_256','reference_transport']:
                if all(variant+'_covered_counts' in r for r in runs):
                    group[variant+'_marginal_coverage']=(np.sum([r[variant+'_covered_counts'] for r in runs],axis=0)/count).tolist()
                    group[variant+'_mean_width_C']=(np.sum([r[variant+'_width_sum_C'] for r in runs],axis=0)/count).tolist()
            if all('parameter_64_to_256_max_bound_change_C' in r for r in runs):
                group['parameter_64_to_256_max_bound_change_C']=np.max([r['parameter_64_to_256_max_bound_change_C'] for r in runs],axis=0).tolist()
                group['parameter_64_to_256_changed_bound_fraction']=(np.sum([r['parameter_64_to_256_changed_bound_counts'] for r in runs],axis=0)/(2*count)).tolist()
        group_rows.append(group)
    return group_rows


def completion_state(rows,declared):
    indexed=len(rows)==declared and all(r['status'] in ['audited','failed'] for r in rows)
    components=indexed and all(r['status']=='failed' or r.get('required_uncertainty_complete',False) for r in rows)
    complete=components and all(r['status']=='audited' for r in rows)
    return {'status':'complete' if complete else ('complete_with_failures' if components else 'partial'),
        'all_declared_folds_accounted':indexed,'all_required_uncertainty_components_accounted':components}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();content=args.manifest.read_bytes();manifest=json.loads(content)
    registry=manifest['transfer_analysis'];write(args.output/'analysis_index_snapshot.json',{
        'manifest_path':str(args.manifest),'manifest_sha256':hashlib.sha256(content).hexdigest(),'transfer_analysis':registry})
    config=json.loads((HERE/'transfer_config.json').read_text());records=load_records()
    expected={(candidate,f['id']):dict(f,candidate=candidate) for candidate in ['primary','shared'] for f in folds(config,candidate)}
    keys=[(r['candidate'],r['fold_id']) for r in registry]
    assert len(set(keys))==len(keys) and set(keys)<=set(expected)
    if not args.allow_partial:assert set(keys)==set(expected),'Missing declared retrospective fold'
    rows=[];bindings=[];missing=[]
    for entry,key in zip(registry,keys):
        fold=expected[key];candidate=config['candidates'][entry['candidate']]
        identity={'candidate':entry['candidate'],'fold_id':entry['fold_id'],'fold_type':fold['type']}
        if entry.get('status')=='failed':
            receipt=Path(entry['failure_receipt']);failure=json.loads(receipt.read_text())
            parent_receipt(receipt.parent,require_terminal=True)
            selected=next(r for r in failure['folds'] if r['id']==entry['fold_id'])
            assert selected['success'] is False and len(selected['attempts'])==2
            assert all(r['success'] is False for r in selected['attempts'])
            rows.append(dict(identity,status='failed',failure_receipt=str(receipt),failure_sha256=sha(receipt)))
            continue
        try:
            fitroot=Path(entry['fit_directory']);fitpath=fitroot/'fit.json';fit=json.loads(fitpath.read_text())
            assert fit['family']==candidate['family'] and fit['band']==candidate['band']
            assert fit['record_names']==[records[j]['name'] for j in fold['train_indices']]
            names=[records[j]['name'] for j in fold['test_indices']]
            fit_audit=Path(entry['fit_audit']);fa=json.loads(fit_audit.read_text())
            assert fa['status']=='passed' and fa['fit_sha256']==sha(fitpath)
            variants={};sources=[]
            required={'noise','parameter_64','parameter_256'}
            if entry['candidate']=='primary' and fold['type']=='leave_protocol_out' and config['reference_transport_sensitivity']['enabled']:
                required.add('reference_transport')
            for name in ['noise','parameter_64','parameter_256','reference_transport']:
                if name not in entry['predictions']:continue
                metadata,arrays,source=prediction(entry['predictions'][name],sha(fitpath),names)
                assert bool(metadata.get('reference_transport_sensitivity',False))==(name=='reference_transport')
                variants[name]=(metadata,arrays);sources.append(dict(source,variant=name))
            chain=parameter_chain(entry,sha(fitpath),variants)
            assert 'noise' in variants,'Noise-only forecast not audited'
            if not args.allow_partial:
                assert required<=set(variants),'Declared parameter/reference sensitivities missing'
            run_rows=[]
            for j,dname in enumerate(names):
                metadata,arrays=variants['noise'];y=arrays['observed_'+str(j)][1:];latent=arrays['latent_nominal_'+str(j)]
                reported=arrays['reported_mean_'+str(j)];n=len(y)
                item={'name':dname,'nobs_per_channel':n,'latent_SSE_by_channel_C2':np.sum((y-latent)**2,axis=0).tolist(),
                    'reported_SSE_by_channel_C2':np.sum((y-reported)**2,axis=0).tolist(),
                    'reported_signed_error_sum_C':np.sum(y-reported,axis=0).tolist(),
                    'reported_run_offset_SSE_C2':(np.sum(y-reported,axis=0)**2/n).tolist()}
                for prefix,label in [('noise_only','noise_only'),('nuisance_noise','hard_nuisance_noise')]:
                    lo=arrays[prefix+'_reported_low_'+str(j)];hi=arrays[prefix+'_reported_high_'+str(j)]
                    item[label+'_covered_counts']=np.sum((y>=lo)&(y<=hi),axis=0).tolist()
                    item[label+'_width_sum_C']=np.sum(hi-lo,axis=0).tolist()
                mixture={}
                for variant in ['parameter_64','parameter_256']:
                    if variant not in variants:continue
                    md,a=variants[variant];assert np.array_equal(a['observed_'+str(j)][1:],y)
                    count=int(variant.rsplit('_',1)[1]);assert md['parameter_draws']['draws']==count
                    lo=a['parameter_mixture_reported_low_'+str(j)];hi=a['parameter_mixture_reported_high_'+str(j)]
                    assert lo.shape==hi.shape==y.shape
                    item[variant+'_covered_counts']=np.sum((y>=lo)&(y<=hi),axis=0).tolist()
                    item[variant+'_width_sum_C']=np.sum(hi-lo,axis=0).tolist();mixture[variant]=(lo,hi)
                if len(mixture)==2:
                    coarse=variants['parameter_64'][0]['parameter_draws'];fine=variants['parameter_256'][0]['parameter_draws']
                    cp=np.load(coarse['path'],allow_pickle=False);fp=np.load(fine['path'],allow_pickle=False)
                    assert np.array_equal(cp['points'],fp['points'][:64]),'Coarse/fine draws are not the declared shared prefix'
                    difference=np.stack([abs(a-b) for a,b in zip(mixture['parameter_64'],mixture['parameter_256'])])
                    item['parameter_64_to_256_max_bound_change_C']=np.max(difference,axis=(0,1)).tolist()
                    item['parameter_64_to_256_changed_bound_counts']=np.sum(difference>1e-9,axis=(0,1)).tolist()
                    item['parameter_64_to_256_compared_bound_counts_per_channel']=2*n
                if 'reference_transport' in variants:
                    md,a=variants['reference_transport'];assert md['parameter_draws'] is None
                    assert np.max(abs(a['reported_mean_'+str(j)]-reported))<1e-10
                    lo=a['nuisance_noise_reported_low_'+str(j)];hi=a['nuisance_noise_reported_high_'+str(j)]
                    item['reference_transport_covered_counts']=np.sum((y>=lo)&(y<=hi),axis=0).tolist()
                    item['reference_transport_width_sum_C']=np.sum(hi-lo,axis=0).tolist()
                    item['reference_transport_ratio_bounds']=md['nuisance_ratio_bounds']
                run_rows.append(item)
            row=dict(identity,status='audited',fit_directory=str(fitroot),fit_sha256=sha(fitpath),
                fit_audit_sha256=sha(fit_audit),per_run=run_rows,
                parameter_sensitivity_complete={'parameter_64','parameter_256'}<=set(variants),
                required_uncertainty_complete=required<=set(variants))
            parameters=fit['parameters'];ta=parameters['tA_nominal_s']
            row['reference_invariant_parameters']={
                'beta_over_CH_C_per_percent_per_s':parameters['K_common_nominal']/ta,
                'Us_over_CH_per_s':parameters['Us_over_Ua_nominal']/ta,
                'Ux_over_CH_per_s':parameters['Ux_over_Ua_nominal']/ta,
                'sensor_time_s':parameters['rS_s'],'input_lag_s':parameters['input_lag_s']}
            row['reference_notice']='Each training subset defines its own centered geometric Ua nominal. Compare invariant rates and predictions rather than raw nominal coordinates alone.'
            rows.append(row);bindings.append(dict(identity,predictions=sources,parameter_chain=chain))
        except Exception as exc:
            if not args.allow_partial:raise
            rows.append(dict(identity,status='not_ready',exception=repr(exc)));missing.append(key)
    group_rows=aggregate_groups(rows,expected,[d['name'] for d in records],args.allow_partial)
    state=completion_state(rows,len(expected));complete=state['status']=='complete'
    result={**state,'declared_folds':len(expected),'indexed_folds':len(registry),
        'missing_fold_ids':[list(k) for k in sorted(set(expected)-set(keys))],
        'rows':rows,'groups':group_rows,'source_bindings':bindings,
        'notice':'Each candidate/fold-type group contains each omitted record once; groups reuse the same hardware observations and are not independent replicates. Pointwise marginal coverage and unknown-run hard-scenario envelopes are descriptive, not simultaneous or automatically calibrated total-95% ranges. Local parameter mixtures condition on an audited active face and selected basin. All twelve records informed structure selection and warm starts; transfer is retrospective.'}
    if manifest.get('local_approximation_assessment'):
        declared=manifest['local_approximation_assessment'];initial=Path(declared['initial_report']);follow=Path(declared['followup_report'])
        for path in [initial,follow]:assert parent_receipt(path.parent,require_terminal=True)['exit']['exit_code']==0
        source=json.loads(initial.read_text());checked=json.loads(follow.read_text())
        assert checked['source_summary_sha256']==sha(initial)
        case=next(r for r in source['cases'] if r['id']=='largest_run_log_tA_SE')
        selected=next(e for e in registry if e['candidate']=='primary' and e['fold_id']==case['fold_id'])
        assert case['source_bindings_sha256'][str(Path(selected['fit_directory'])/'fit.json')]==sha(Path(selected['fit_directory'])/'fit.json')
        median=next(r for r in checked['results'] if r['role']=='median_error')
        assert median['passed_1e_minus3_highest_order_and_tail_gates']
        values=median['evaluations'];assert abs(values[-1]['twice_nll_minus_reference']-values[-2]['twice_nll_minus_reference'])<.001
        assert abs(values[-2]['twice_nll_minus_reference']-values[-3]['twice_nll_minus_reference'])<.001
        worst=next(r for r in checked['results'] if r['role']=='worst')
        result['local_approximation_assessment']={'source_reports_sha256':{str(p):sha(p) for p in [initial,follow]},
            'selected_fold':case['fold_id'],'representative_draw_index':median['draw_index'],
            'representative_actual_twice_nll_difference':values[-1]['twice_nll_minus_reference'],
            'representative_taylor_prediction':median['taylor_prediction'],
            'extreme_draw_quadrature_gate_passed':worst['passed_1e_minus3_highest_order_and_tail_gates'],
            'interpretation':declared['interpretation']}
    write(args.output/'summary.json',result)
    available=[g for g in group_rows if g['nobs_per_channel']]
    if not available:return
    fig,axes=plt.subplots(1,2,figsize=(8,3.2));x=np.arange(len(available));width=.35
    for k in [0,1]:
        axes[0].bar(x+(k-.5)*width,[g['reported_RMSE_by_channel_C'][k] for g in available],width,label='T'+str(k+1))
    labels=[g['candidate']+'\n'+g['fold_type'].replace('leave_','').replace('_out','').replace('_',' ') for g in available]
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Retrospective reported-mean RMSE (C)');axes[0].legend(fontsize=7)
    for variant,marker in [('noise_only','o'),('hard_nuisance_noise','s'),('parameter_256','^')]:
        for k in [0,1]:
            values=[g.get(variant+'_marginal_coverage',[np.nan,np.nan])[k] for g in available]
            if not np.isfinite(values).any():continue
            axes[1].plot(x+(k-.5)*.12,values,marker=marker,color=['tab:blue','tab:orange'][k],
                linestyle='none',label=variant.replace('_',' ')+' T'+str(k+1))
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Observed pointwise coverage',ylim=(0,1.02));axes[1].legend(fontsize=6)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    title='Retrospective transfer; local-normal ranges remain uncalibrated' if 'local_approximation_assessment' in result else 'Internal transfer; nuisance envelopes and local mixtures are conditional'
    fig.suptitle(('PARTIAL: ' if not complete else '')+title,fontsize=10)
    fig.tight_layout();fig.savefig(args.output/'transfer_summary.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
