"""Check released report scalars, saved predictions and reviewed figure bindings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from run_mean import write_json


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--study',type=Path,required=True);args=parser.parse_args();p=args.study;r=p/'results'
    manifest=json.loads((p/'run_manifest.json').read_text());report=(p/'report.md').read_text()
    assert manifest['status']=='complete_for_researcher_review' and not manifest['live_jobs'] and not manifest['stationarity_warnings']
    assert sha(p/'report.md')==manifest['report_sha256']
    checked=[]
    for name in manifest['selected_models']:
        f=json.loads((r/name/'fit.json').read_text());a=np.load(r/name/'trajectories.npz')
        residual=np.vstack([a['observed_'+str(j)][1:]-a['predicted_'+str(j)][1:] for j in range(len(f['record_names']))])
        rmse=np.sqrt(np.mean(residual**2,axis=0));assert np.max(abs(rmse-f['RMSE_by_channel_C']))<1e-10
        n=f['nobs'];k=f['parameter_count'];aicc=2*f['best']['nll']+2*k+2*k*(k+1)/(n-k-1)
        assert abs(aicc-f['nominal_AICc'])<1e-8
        row=next(line for line in report.splitlines() if line.startswith('| '+name+' |'))
        cells=[v.strip() for v in row.strip('|').split('|')]
        assert cells[1:4]==[f'{rmse[0]:.4f}',f'{rmse[1]:.4f}',f'{aicc:.2f}']
        audit=json.loads((r/(name+'_audit')/'audit.json').read_text());assert audit['status']=='passed' and audit['fit_sha256']==sha(r/name/'fit.json')
        checked.append({'model':name,'recomputed_RMSE_C':rmse.tolist(),'independent_audit_nll_abs_error':audit['nll_abs_error']})
    visual=json.loads((r/'final_analysis/visual_review.json').read_text());assert visual['status']=='passed' and visual['report_sha256']==sha(p/'report.md')
    linked=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report)
    assert len(linked)==3
    for link in linked:assert sha(p/link)==visual['figure_sha256'][link]
    for link,value in manifest['bindings_sha256'].items():assert sha(r/link)==value
    write_json(args.output/'audit.json',{'status':'passed','selected_models':checked,'report_sha256':sha(p/'report.md'),
        'figure_count':len(linked),'bindings_checked':len(manifest['bindings_sha256']),
        'scope':'Saved two-output RMSE recomputation, AICc arithmetic, printed table cells, selected forward/likelihood audit hashes, report and reviewed figures. No new physical validation or global-optimum proof.'})
    print('Final report and numerical bindings passed',flush=True)


if __name__=='__main__':main()
