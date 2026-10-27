"""Check local quadratic likelihood shape at declared saved parameter draws."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from joint_quantized import Model
from draw_local import conditional_draws
from continuation import parent_receipt
from audit_common import dimensional_mean
from run_bounded import write


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);args=p.parse_args()
    manifest=json.loads(args.manifest.read_text());root=args.manifest.parent/'results'
    policy=json.loads((HERE/'uncertainty_config.json').read_text())['local_taylor_check']
    candidates=[]
    for entry in manifest['transfer_analysis']:
        if entry['candidate']=='primary' and entry['fold_id'].startswith('run_'):
            auditpath=Path(entry['information_audit'])/'audit.json';audit=json.loads(auditpath.read_text());assert audit['status']=='passed'
            row=next(r for r in audit['local_comparison'] if r['coordinate']=='log_tA')
            candidates.append({'entry':entry,'log_tA_SE':row['bin_observed_mean_and_noise_nuisance_SE'],
                'audit_sha256':sha(auditpath)})
    assert len(candidates)==12;selected=sorted(candidates,key=lambda r:(-r['log_tA_SE'],r['entry']['fold_id']))[0]
    protocol=next(e for e in manifest['transfer_analysis'] if e['candidate']=='primary' and e['fold_id']=='protocol_30s')
    cases=[{'id':'full_data','fit_directory':str(root/'joint_conductance_b10_export_v2'),
        'information':str(root/'joint_b10_information_v2'),'information_audit':str(root/'joint_b10_information_v2_audit'),
        'draws':str(root/'joint_b10_local_parameter_draws')},
        dict(protocol,id='protocol_30s'),dict(selected['entry'],id='largest_run_log_tA_SE')]
    indices=np.linspace(0,255,policy['draws_checked_per_case']).astype(int).tolist()
    write(args.output/'design.json',{'policy':policy,'indices':indices,'cases':cases,
        'selection_candidates':[{k:r[k] for k in ['log_tA_SE','audit_sha256']}|{'fold_id':r['entry']['fold_id']} for r in candidates],
        'manifest_sha256':sha(args.manifest),'notice':'Case selection uses training information geometry, not test prediction errors. This is a descriptive approximation diagnostic.'})
    lookup={d['name']:d for d in load_records()};results=[]
    for case in cases:
        fitdir=Path(case['fit_directory']);information=Path(case['information']);auditor=Path(case['information_audit']);drawdir=Path(case['draws'])
        fit=json.loads((fitdir/'fit.json').read_text());info=json.loads((information/'information.json').read_text());audit=json.loads((auditor/'audit.json').read_text());receipt=json.loads((drawdir/'draw_receipt.json').read_text())
        for directory in [fitdir,information,auditor,drawdir]:
            assert parent_receipt(directory,require_terminal=True)['exit']['exit_code']==0
        assert audit['status']=='passed' and fit['family']=='conductance'
        assert not any(fit.get(k) for k in ['fixed_coordinates','startup_band_C','initial_sensor_bins','condition_initial_code','dataset'])
        fitsha=sha(fitdir/'fit.json');assert fitsha==info['fit_sha256']==audit['fit_sha256']==receipt['fit_sha256']
        assert audit['matrix_sha256']==receipt['information_matrix_sha256']==sha(information/'information_matrices.npz')
        assert receipt['information_audit_sha256']==sha(auditor/'audit.json')
        assert audit['information_sha256']==sha(information/'information.json') and receipt['draws']==256
        records=[lookup[n] for n in fit['record_names']];assert all(d['sha256']==fit['raw_sha256'][d['name']] for d in records)
        model=Model(records,fit['band'],fit['phase'],128,'conductance',minimum_sensor_loading=fit.get('minimum_sensor_loading',.005),score_backend='compiled')
        assert model.coordinate_names()==info['coordinate_names']==fit['coordinate_names']
        point=np.array(fit['best']['point']);matrices=np.load(information/'information_matrices.npz');H=matrices['observed_Hessian']
        path=drawdir/'parameter_draws_256.npz';points=np.load(path)['points'];assert points.shape==(256,len(point))
        regenerated,proposals=conditional_draws(point,np.array(model.bounds()),H,matrices['active_face_basis'],256,receipt['seed'])
        assert proposals==receipt['proposals'] and np.max(abs(regenerated-points))<1e-10
        f0=model.objective(point);g=model.gradient(point);assert abs(f0-fit['best']['nll'])<1e-3
        rows=[]
        for index in indices:
            q=points[index];delta=q-point;linear=float(2*g@delta);quadratic=float(delta@H@delta)
            row={'draw_index':index,'twice_linear_term':linear,'quadratic_term':quadratic,'taylor_prediction':linear+quadratic,
                'loading_ratio':float(np.exp(q[2]))}
            try:
                value=model.objective(q);assert np.isfinite(value);actual=float(2*(value-f0));row.update({'status':'evaluated','twice_nll_difference':actual,
                    'absolute_taylor_error':abs(actual-linear-quadratic),'relative_error_to_1_plus_abs_actual':abs(actual-linear-quadratic)/(1+abs(actual))})
            except Exception as exc:row.update({'status':'failed_evaluation','exception':repr(exc)})
            rows.append(row);write(args.output/(case['id']+'_draws.json'),rows)
        finite=[r for r in rows if r['status']=='evaluated'];witness=None
        if finite:
            worst=max(finite,key=lambda r:r['absolute_taylor_error']);other=Model(records,fit['band'],fit['phase'],256,'conductance',minimum_sensor_loading=fit.get('minimum_sensor_loading',.005),score_backend='compiled')
            high=other.objective(points[worst['draw_index']]);base=other.objective(point)
            difference=float(2*(high-base));error=abs(difference-worst['twice_nll_difference'])
            q=points[worst['draw_index']];coefficients=np.r_[np.exp(q[model.nshape]),q[model.nshape+1:model.nmean]]
            independent=dimensional_mean(q,coefficients,records,fit['band']);calculated=model.mean(q,False)
            mean_error=float(np.max(abs(independent-calculated)))
            witness={'draw_index':worst['draw_index'],'twice_nll_difference256':difference,
                'order128_to256_absolute_change':error,'passed_1e_minus3_gate':bool(error<1e-3),
                'independent_dimensional_mean_max_error_C':mean_error,'passed_1e_minus7_mean_gate':bool(mean_error<1e-7)}
        files=[fitdir/'fit.json',information/'information.json',information/'information_matrices.npz',auditor/'audit.json',drawdir/'draw_receipt.json',path]
        errors=[r['absolute_taylor_error'] for r in finite]
        results.append({'id':case['id'],'fold_id':case.get('fold_id'),'evaluated':len(finite),'declared':len(indices),
            'absolute_taylor_error_median_max':np.quantile(errors,[.5,1.]).tolist() if errors else None,
            'loading_ratio_all256_q025_q50_q975':np.quantile(np.exp(points[:,2]),[.025,.5,.975]).tolist(),
            'quadrature_witness':witness,'source_bindings_sha256':{str(p):sha(p) for p in files}})
        write(args.output/'results.json',results)
    write(args.output/'summary.json',{'status':'complete','cases':results,'policy':policy,
        'scope':'Actual training bin likelihood versus its observed-Hessian Taylor approximation at16 prespecified saved draws per case. Not refitting, coverage calibration or a posterior check. Numerical curvature audits certify the Hessian near the fitted point; this diagnostic assesses its use away from that point. Report failed evaluations and quadrature witnesses rather than treating every positive Hessian as a validated global uncertainty distribution.'})


if __name__=='__main__':main()
