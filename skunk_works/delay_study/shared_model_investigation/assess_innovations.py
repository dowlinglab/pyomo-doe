"""Descriptive innovation adequacy checks across all selected model families."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from diagnostics import input_correlation
from audit_saved_mean import acf
from run_mean import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();models=[]
    for folder in args.inputs:
        fit=json.loads((folder/'fit.json').read_text());arrays=np.load(folder/'trajectories.npz');rows=[];total=0.;n=0
        for j,name in enumerate(fit['record_names']):
            z=arrays['standardized_innovation_'+str(j)];u=arrays['input_'+str(j)][1:]
            correlation=acf(z,30);cross=input_correlation(z,u,30)
            values=[abs(r['correlation']) for r in cross if r['correlation'] is not None]
            blocks=np.array_split(z,6)
            rows.append({'name':name,'mean':float(z.mean()),'SD':float(z.std()),
                         'acf_lags1_to30':correlation,'max_abs_acf_lags1_to30':float(max(abs(v) for v in correlation)),
                         'max_abs_input_correlation_lags30':max(values) if values else None,
                         'block_means':[float(v.mean()) for v in blocks],
                         'block_SD':[float(v.std()) for v in blocks],
                         'fraction_abs_greater3':float(np.mean(abs(z)>3))})
            total+=z@z;n+=len(z)
        if abs(total/n-1)>1e-7:raise AssertionError(('Profile variance normalization',folder.name,total/n))
        models.append({'id':folder.name,'fit_sha256':hashlib.sha256((folder/'fit.json').read_bytes()).hexdigest(),
                       'pooled_mean_square':float(total/n),'per_run':rows,
                       'max_abs_acf':max(v['max_abs_acf_lags1_to30'] for v in rows),
                       'SD_range':[min(v['SD'] for v in rows),max(v['SD'] for v in rows)],
                       'fast_input_correlation_range':[min(v['max_abs_input_correlation_lags30'] for v in rows[:3]),max(v['max_abs_input_correlation_lags30'] for v in rows[:3])]})
    write_json(args.output/'assessment.json',{'status':'complete','models':models,
        'notice':'Descriptive same-record diagnostics, not a calibrated joint whiteness test. Pooled mean square1 follows fitted scale algebra and does not certify per-run covariance adequacy. Quantization and data-dependent model selection remain relevant.'})
    print(json.dumps({'status':'complete','models':[{'id':m['id'],'max_abs_acf':m['max_abs_acf'],'SD_range':m['SD_range'],'fast_cross':m['fast_input_correlation_range']} for m in models]}),flush=True)


if __name__=='__main__':main()
