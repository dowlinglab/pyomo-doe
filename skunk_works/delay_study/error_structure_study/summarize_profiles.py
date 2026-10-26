"""Plot only terminal independently audited profile grids, without CI interpolation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from continuation import parent_receipt
from run_bounded import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(directory,audit_directory):
    parent_receipt(directory,require_terminal=True);parent_receipt(audit_directory,require_terminal=True)
    path=directory/'profile.json';profile=json.loads(path.read_text())
    audit=json.loads((audit_directory/'audit.json').read_text())
    assert audit['status']=='passed' and audit['kind']=='profile' and audit['source_sha256']==sha(path)
    rows=[];name=profile['coordinate']
    for item in profile['points']:
        row={k:item[k] for k in ['success','coordinate_value','physical_value']}
        if item['success']:
            fitpath=Path(item['output'])/'fit.json';fit=json.loads(fitpath.read_text())
            matched=[r for r in audit['rows'] if abs(r['coordinate_value']-item['coordinate_value'])<1e-12]
            assert len(matched)==1 and matched[0]['audit_status']=='passed' and matched[0]['fit_sha256']==sha(fitpath)
            assert abs(fit['best']['nll']-item['nll'])<1e-9
            index=fit['coordinate_names'].index(name);coordinate=fit['best']['point'][index]
            assert abs(coordinate-item['coordinate_value'])<1e-10 and fit['fixed_coordinates'][name]==item['coordinate_value']
            physical=float(np.exp(coordinate)) if profile['optimization_coordinate_is_log'] else coordinate
            assert abs(physical-item['physical_value'])<1e-10*(1+abs(physical))
            deviance=2*(fit['best']['nll']-profile['reference_nll'])
            assert abs(deviance-item['twice_nll_minus_reference'])<1e-9
            row.update({'nll':fit['best']['nll'],'twice_nll_minus_reference':deviance,
                'fit':str(fitpath),'fit_sha256':sha(fitpath),
                'fitted_sensor_time_s':fit['parameters']['rS_s'],'fitted_input_lag_s':fit['parameters']['input_lag_s']})
        rows.append(row)
    return {'coordinate':name,'profile_path':str(path),'profile_sha256':sha(path),
        'audit_path':str(audit_directory/'audit.json'),'audit_sha256':sha(audit_directory/'audit.json'),
        'reference_fit_sha256':profile['fit_sha256'],'reference_nll':profile['reference_nll'],
        'reference_physical_value':float(np.exp(profile['reference_coordinate'])) if profile['optimization_coordinate_is_log'] else profile['reference_coordinate'],
        'physical_is_positive_log_chart':profile['optimization_coordinate_is_log'],'points':rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',type=Path,nargs=2,action='append',required=True,metavar=('PROFILE','AUDIT'))
    args=parser.parse_args();profiles=[collect(*pair) for pair in args.profile]
    assert len(set(r['coordinate'] for r in profiles))==len(profiles)
    fig,axes=plt.subplots(2,len(profiles),figsize=(5*len(profiles),6),squeeze=False)
    labels={'log_K':'Nominal gain (C/%)','log_input_lag':'Input lag L (s)',
        'log_Us_over_Ua':'Sensor loading Us/Ua','log_Ua_ratio_9':'Run 10 conductance ratio',
        'log_ell_T1':'T1 correlation length (s)','radiation_gamma':'Normalized radiation coefficient'}
    for j,profile in enumerate(profiles):
        rows=[r for r in profile['points'] if r['success']]
        x=[r['physical_value'] for r in rows];y=[r['twice_nll_minus_reference'] for r in rows]
        for k in [0,1]:
            ax=axes[k,j];ax.plot(x,y,'o-' if k==0 else 'o',ms=4,lw=.8,label='Audited grid; lines guide the eye')
            ax.axvline(profile['reference_physical_value'],color='0.5',ls=':',lw=.8,label='Reference fitted coordinate')
            if profile['physical_is_positive_log_chart'] and max(x)/min(x)>4:ax.set_xscale('log')
            ax.set_xlabel(labels.get(profile['coordinate'],profile['coordinate']))
            ax.set_ylabel('2 x (nll - reference nll)');ax.grid(alpha=.2)
            if k==0:ax.set_title('Full grid');ax.legend(fontsize=7)
            else:ax.set_ylim(-.5,12);ax.set_title('Detail; no interval interpolation')
    fig.suptitle('Conditional profile grids; boundary and basin limits remain')
    fig.tight_layout();fig.savefig(args.output/'profile_grids.png',dpi=160);plt.close(fig)
    write(args.output/'summary.json',{'status':'complete_for_supplied_grids','profiles':profiles,
        'notice':'Every successful point is source-bound to its passed full numerical audit. A line between grid points is a visual guide, not an optimized profile or an interpolated confidence endpoint. No cutoff calibration, global optimization proof or completion of missing profile coordinates is claimed.'})


if __name__=='__main__':main()
