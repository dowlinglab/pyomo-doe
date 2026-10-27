"""Generate report tables from the terminal complete preferred transfer summary."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def table(header,rows):
    return '\\begin{tabularx}{\\linewidth}{@{}Xrr@{}}\n\\toprule\n'+header+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+'\\\\' for row in rows)+'\n\\bottomrule\n\\end{tabularx}\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    source=args.input/'summary.json';data=json.loads(source.read_text());assert json.loads((args.input/'exit.json').read_text())['exit_code']==0
    assert data['status']=='complete' and data['indexed_folds']==20 and data['all_required_uncertainty_components_accounted']
    assert 'local_approximation_assessment' in data
    args.output.mkdir(parents=True,exist_ok=True)
    definitions=[('primary','leave_run_out','Band: omit one run','Band/run','10% band, omit one run'),
        ('primary','leave_protocol_out','Band: omit protocol','Band/protocol','10% band, omit a protocol'),
        ('shared','leave_protocol_out','Shared: omit protocol','Shared/protocol','Shared conductance, omit a protocol')]
    nominal=[];ranges=[];resolution=[];md_nominal=[];md_ranges=[];md_resolution=[]
    for candidate,kind,label,short,mdlabel in definitions:
        g=next(v for v in data['groups'] if (v['candidate'],v['fold_type'])==(candidate,kind));assert g['nobs_per_channel']==10795 and g['unique_test_records']==12
        rmse=g['reported_RMSE_by_channel_C'];nominal.append([label,f'{rmse[0]:.3f}',f'{rmse[1]:.3f}']);md_nominal.append(f'| {mdlabel} | {rmse[0]:.3f} | {rmse[1]:.3f} |')
        for name,field,mdname in [('noise','noise_only','noise'),('conditions + noise','hard_nuisance_noise','conditions + noise'),('parameters','parameter_256','parameters')]:
            coverage=[100*v for v in g[field+'_marginal_coverage']];width=g[field+'_mean_width_C']
            ranges.append([short+', '+name,f'{coverage[0]:.2f}/{coverage[1]:.2f}',f'{width[0]:.3f}/{width[1]:.3f}'])
            md_ranges.append(f'| {mdlabel}, {mdname} | {coverage[0]:.2f}/{coverage[1]:.2f}% | {width[0]:.3f}/{width[1]:.3f} |')
        if 'reference_transport_marginal_coverage' in g:
            coverage=[100*v for v in g['reference_transport_marginal_coverage']];width=g['reference_transport_mean_width_C']
            ranges.append([short+', reference stress',f'{coverage[0]:.2f}/{coverage[1]:.2f}',f'{width[0]:.3f}/{width[1]:.3f}'])
            md_ranges.append(f'| {mdlabel}, reference stress | {coverage[0]:.2f}/{coverage[1]:.2f}% | {width[0]:.3f}/{width[1]:.3f} |')
        maximum=g['parameter_64_to_256_max_bound_change_C'];fraction=[100*v for v in g['parameter_64_to_256_changed_bound_fraction']]
        resolution.append([short,f'{maximum[0]:.2f}/{maximum[1]:.2f}',f'{fraction[0]:.2f}/{fraction[1]:.2f}'])
        md_resolution.append(f'| {mdlabel} | {maximum[0]:.2f}/{maximum[1]:.2f} | {fraction[0]:.2f}/{fraction[1]:.2f}% |')
    files={
        'transfer_nominal.tex':table('Nominal reported-mean RMSE, $\\C$ & T1 & T2',nominal),
        'transfer_ranges.tex':table('Range and group & Coverage, \\% & Mean width, $\\C$\\\\\n & T1/T2 & T1/T2',ranges),
        'transfer_resolution.tex':table('64/256 comparison & Max change, $\\C$ & Endpoints changed, \\%\\\\\n & T1/T2 & T1/T2',resolution),
        'transfer_tables.md':'| Comparison group | T1 nominal RMSE, C | T2 nominal RMSE, C |\n| --- | ---: | ---: |\n'+'\n'.join(md_nominal)+'\n\n| Range and group | Coverage, T1/T2 | Mean width, T1/T2, C |\n| --- | ---: | ---: |\n'+'\n'.join(md_ranges)+'\n\n| 64/256 comparison | Maximum endpoint change, T1/T2, C | Endpoints changed, T1/T2 |\n| --- | ---: | ---: |\n'+'\n'.join(md_resolution)+'\n'}
    for name,content in files.items():(args.output/name).write_text(content)
    (args.output/'source.json').write_text(json.dumps({'summary_path':str(source.resolve()),'summary_sha256':sha(source),
        'generator_sha256':sha(Path(__file__)),'outputs_sha256':{name:sha(args.output/name) for name in files},
        'scope':'Table transcription only; final numeric checker independently pools forecast arrays. Parameters denotes an exploratory local-mixture sensitivity whose calibration failed a selected likelihood-shape check.'},indent=2)+'\n')


if __name__=='__main__':main()
