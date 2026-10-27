"""Plot the four terminal, independently audited nuisance-profile grids."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from run_bounded import write


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--radiation-fit',type=Path,required=True);args=p.parse_args()
    assert json.loads((args.input/'exit.json').read_text())['exit_code']==0
    columns=[('log_Us_over_Ua','Sensor loading $U_s/U_{a,0}$','log'),
        ('log_Ua_ratio_9','Step run 1 conductance ratio','linear'),
        ('log_ell_T1','T1 disturbance length (s)','log'),
        ('radiation_gamma','Radiation coefficient at 50 C / $U_{a,0}$','linear')]
    fit=json.loads((args.fit/'fit.json').read_text())
    assert 'step_50_run_1' in fit['record_names'][9]
    profiles=[];binding={}
    for name,label,scale in columns:
        source=args.input/name/'profile.json';audit=args.input/(name+'_audit')/'audit.json'
        profile=json.loads(source.read_text());check=json.loads(audit.read_text())
        reference=args.radiation_fit if name=='radiation_gamma' else args.fit
        assert profile['coordinate']==name and profile['status']=='complete'
        assert not profile['lower_nll_than_reference']
        assert profile['fit_sha256']==sha(reference/'fit.json')
        assert check['status']=='passed' and check['kind']=='profile' and check['source_sha256']==sha(source)
        assert len(check['rows'])==len(profile['points']) and all(r['success'] for r in profile['points'])
        x=[];y=[]
        for row in profile['points']:
            point=Path(row['output'])/'fit.json';saved=json.loads(point.read_text())
            value=2*(saved['best']['nll']-profile['reference_nll'])
            assert abs(value-row['twice_nll_minus_reference'])<1e-9
            index=saved['coordinate_names'].index(name)
            assert abs(saved['best']['point'][index]-row['coordinate_value'])<1e-9
            binding[str(point)]=sha(point);x.append(row['physical_value']);y.append(value)
        binding.update({str(q):sha(q) for q in [source,audit,reference/'fit.json']})
        profiles.append({'coordinate':name,'label':label,'xscale':scale,'x':x,'twice_nll_increase':y,
            'reference_x':float(np.exp(profile['reference_coordinate'])) if name.startswith('log_') else profile['reference_coordinate']})
    write(args.output/'figure_values.json',{'profiles':profiles,'source_bindings_sha256':binding,
        'notice':'All declared points shown. Lines connect coarse evaluated grids, not interpolated confidence endpoints or global profiles. Conditional on model, bounds, initialization and bin phase; no significance cutoff is drawn.'})
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(5.,4.7))
    for ax,row,letter in zip(axes.ravel(),profiles,'abcd'):
        ax.plot(row['x'],row['twice_nll_increase'],'o-',ms=3,lw=.8)
        ax.plot(row['reference_x'],0.,'k*',ms=6)
        ax.set(xscale=row['xscale'],xlabel=row['label'],ylabel='Twice nll increase',title='('+letter+') '+
            {'log_Us_over_Ua':'Weak loading information','log_Ua_ratio_9':'Active run bound',
             'log_ell_T1':'Disturbance length','radiation_gamma':'Nonnegative radiation'}[row['coordinate']])
        if row['coordinate']=='radiation_gamma':
            ax.set_yscale('symlog',linthresh=1.)
            ax.set_xscale('symlog',linthresh=.02)
            ax.set_xticks([0.,.02,.1,1.,4.]);ax.set_xticklabels(['0','0.02','0.1','1','4'])
        ax.grid(alpha=.15)
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(args.output/('remaining_profiles.'+suffix),dpi=240)
    plt.close(fig)
    write(args.output/'figure_receipt.json',{'status':'complete','figure_values_sha256':sha(args.output/'figure_values.json'),
        'outputs_sha256':{suffix:sha(args.output/('remaining_profiles.'+suffix)) for suffix in ['png','pdf']},
        'size_inches':[5.,4.7],'points':sum(len(r['x']) for r in profiles),
        'scope':'Source-bound plotting and profile arithmetic only. Separate numerical point audits are prerequisites; visual review and final scientific interpretation remain required.'})


if __name__=='__main__':main()
