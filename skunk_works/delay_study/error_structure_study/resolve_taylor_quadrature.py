"""Convergence follow-up for the local-shape check's failed extreme witness."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from audit_common import dimensional_mean
from quantized import filter_ou
from continuation import parent_receipt
from run_bounded import write


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);args=p.parse_args()
    assert parent_receipt(args.input,require_terminal=True)['exit']['exit_code']==0
    design=json.loads((args.input/'design.json').read_text());summary=json.loads((args.input/'summary.json').read_text())
    case=next(v for v in design['cases'] if v['id']=='largest_run_log_tA_SE')
    original=next(v for v in summary['cases'] if v['id']==case['id'])
    assert original['quadrature_witness']['passed_1e_minus3_gate'] is False
    for name,digest in original['source_bindings_sha256'].items():assert sha(Path(name))==digest
    rows=json.loads((args.input/(case['id']+'_draws.json')).read_text());finite=[r for r in rows if r['status']=='evaluated']
    assert len(finite)==16
    median=float(np.median([r['absolute_taylor_error'] for r in finite]))
    worst=max(finite,key=lambda r:r['absolute_taylor_error'])
    typical=min(finite,key=lambda r:(abs(r['absolute_taylor_error']-median),r['draw_index']))
    chosen=[('worst',worst),('median_error',typical)]
    fitdir=Path(case['fit_directory']);fit=json.loads((fitdir/'fit.json').read_text())
    fitcheck=Path(case['fit_audit']);audit=json.loads(fitcheck.read_text())
    assert parent_receipt(fitcheck.parent,require_terminal=True)['exit']['exit_code']==0
    assert audit['status']=='passed' and audit['fit_sha256']==sha(fitdir/'fit.json')
    points=np.load(Path(case['draws'])/'parameter_draws_256.npz')['points']
    lookup={d['name']:d for d in load_records()};records=[lookup[n] for n in fit['record_names']]
    assert all(d['sha256']==fit['raw_sha256'][d['name']] for d in records)
    nshape=fit['coordinate_names'].index('log_K');nmean=len(fit['best']['point'])-6
    assert fit['coordinate_names'][-6:]==['log_latent_SD_T1','log_ell_T1','log_measurement_SD_T1','log_latent_SD_T2','log_ell_T2','log_measurement_SD_T2']
    policy=json.loads((HERE/'uncertainty_config.json').read_text())['taylor_quadrature_followup']
    write(args.output/'design.json',{'policy':policy,'selected':[{k:v for k,v in row.items()}|{'role':label} for label,row in chosen],
        'original_summary_sha256':sha(args.input/'summary.json'),'original_evaluations_sha256':sha(args.input/(case['id']+'_draws.json')),
        'fit_audit_sha256':sha(fitcheck)})
    results=[];N=sum(len(d['t'])-1 for d in records)
    for label,row in chosen:
        q=points[row['draw_index']];coef=np.r_[np.exp(q[nshape]),q[nshape+1:nmean]]
        mean=dimensional_mean(q,coef,records,fit['band']);noise=np.exp(q[-6:]).reshape(2,3)
        values=[]
        for nodes,tail in [(n,12.) for n in policy['nodes']]+[(policy['nodes'][-1],16.)]:
            nll=0.;normalization=0.
            for k,(latent,ell,measurement) in enumerate(noise):
                variance=latent**2+measurement**2;cursor=0
                for d in records:
                    n=len(d['t'])-1;section=slice(k*N+cursor,k*N+cursor+n)
                    result=filter_ou(d['t'][1:],mean[section],d['y'][1:,k],np.sqrt(variance),ell,measurement**2/variance,fit['phase'],nodes,tail)
                    nll+=result['nll'];normalization=max(normalization,float(np.max(abs(result['posterior_normalization_ratio']-1))));cursor+=n
            assert np.isfinite(nll) and normalization<1e-6
            values.append({'nodes':nodes,'tail_SD':tail,'nll':float(nll),
                'twice_nll_minus_reference':float(2*(nll-fit['best']['nll'])),'maximum_normalization_error':normalization})
            write(args.output/(label+'_evaluations.json'),values)
        changes=[abs(b['twice_nll_minus_reference']-a['twice_nll_minus_reference']) for a,b in zip(values,values[1:])]
        confirmed=changes[-2]<.001 and changes[-1]<.001
        results.append({'role':label,'draw_index':row['draw_index'],'evaluations':values,'adjacent_deviance_changes':changes,
            'passed_1e_minus3_highest_order_and_tail_gates':confirmed,'taylor_prediction':row['taylor_prediction'],
            'confirmed_absolute_taylor_error':abs(values[-1]['twice_nll_minus_reference']-row['taylor_prediction']) if confirmed else None,
            'notice':'Precise discrepancy is withheld if the declared highest-order/tail gates fail; no tolerance relaxation.'})
        write(args.output/'results.json',results)
    write(args.output/'summary.json',{'status':'complete' if all(r['passed_1e_minus3_highest_order_and_tail_gates'] for r in results) else 'completed_with_unresolved_quadrature',
        'results':results,'source_summary_sha256':sha(args.input/'summary.json'),
        'scope':'Independent dimensional propagation with forward-only128/256/512/1024-node likelihoods and a1024-node tail12/16 comparison at worst and median-error saved draws. No refitting or confidence calibration.'})


if __name__=='__main__':main()
