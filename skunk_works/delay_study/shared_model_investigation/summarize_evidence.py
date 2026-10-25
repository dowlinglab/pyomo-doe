"""Consolidate audited artifacts; independently check saved prediction arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
import pandas as pd
from run_mean import write_json

MODELS=['chain1','chain2','chain3','flexible2','flexible3','flexible4','unequal3',
        'transport','transport_linear','power_curve','power_curve_linear','quadratic',
        'mapped_chain1','mapped_chain2','mapped_chain3','mapped_unequal3']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();hashes={}
    def read(relative):
        p=args.results/relative
        hashes[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    assessment=read('innovation_assessment/assessment.json')
    innovations={r['id']:r for r in assessment['models']}
    rows=[];arithmetic=[]
    names=None
    for name in MODELS:
        assert read(name+'/exit.json')['exit_code']==0
        fit=read(name+'/fit.json')
        path=args.results/name/'trajectories.npz'
        hashes[name+'/trajectories.npz']=hashlib.sha256(path.read_bytes()).hexdigest()
        a=np.load(path);e=[];z=[]
        if names is None:names=fit['record_names']
        assert fit['record_names']==names and len(names)==12
        for j in range(12):
            e.extend(a['observed_'+str(j)][1:]-a['predicted_'+str(j)][1:])
            z.extend(a['standardized_innovation_'+str(j)])
        rmse=float(np.sqrt(np.mean(np.square(e))))
        error=abs(rmse-fit['RMSE_C']);assert error<1e-10 and len(e)==10795
        assert abs(np.mean(np.square(z))-1)<1e-7
        best=fit['best'];k=fit['parameter_count'];n=len(e)
        aicc=2*best['nll']+2*k+2*k*(k+1)/(n-k-1)
        assert abs(aicc-fit['nominal_AICc'])<1e-8
        diag=innovations[name]
        assert diag['fit_sha256']==hashes[name+'/fit.json']
        row={'id':name,'RMSE_C':rmse,'nll':best['nll'],'nominal_AICc':aicc,
             'parameter_count':k,'successful_starts':sum(r['success'] for r in fit['starts']),
             'total_starts':len(fit['starts']),'max_abs_acf':diag['max_abs_acf'],
             'fast_cross_min':diag['fast_input_correlation_range'][0],
             'fast_cross_max':diag['fast_input_correlation_range'][1],
             'innovation_SD_min':diag['SD_range'][0],'innovation_SD_max':diag['SD_range'][1]}
        if name=='quadratic':
            row['K']=fit['parameters']['K']
            row['beta_per_C']=fit['parameters']['beta_per_C']
        else:
            row['K']=best['coefficients'][0]
            row['time_constants_s']=fit['time_constants_s']
            row['power_curvature']=fit.get('power_curvature')
        if name!='quadratic':
            source='information_mapped' if name.startswith('mapped_') else 'information_final'
            info=read(source+'/'+name+'_information.json')
            assert info['fit_sha256']==hashes[name+'/fit.json']
            row['condition_J']=info['condition_J']
            row['local_logK_SD']=info['conditional_coordinate_SD'][info['mean_coordinate_order'].index('log_K')]
            response=[r for r in info['conditional_response_uncertainty'] if r['period_s']==30.][0]
            row['local_block_phase30_SD_deg']=response['phase_SD_deg']
        rows.append(row);arithmetic.append({'id':name,'nobs':len(e),'RMSE_abs_error':error})
    curves=[]
    for folder in ['chain_profiles_audit','mapped_profiles1_audit','mapped_profiles23_audit','transport_profiles_audit','power_profiles_audit']:
        audit=read(folder+'/audit.json');assert audit['status']=='passed'
        for r in audit['curves']:
            assert r['numerical_warnings']==0 and r['lower_reference_points']==0
            curves.append({'id':r['id'],'reference_value':r['reference_value'],
                           'nominal_grid_support':r['nominal_support'],'audit':folder+'/audit.json'})
    beta=read('beta_profiles/profile.json')
    assert beta['status']=='complete' and len(beta['points'])==7
    assert not any(r['stationarity_warning'] for r in beta['points'])
    reported=[]
    for line in args.report.read_text().splitlines():
        if not line.startswith('|'):continue
        cells=[v.strip() for v in line.strip('|').split('|')]
        if len(cells)!=4:continue
        try:reported.append([float(v) for v in cells[1:]])
        except ValueError:pass
    expected=[[float(format(r['RMSE_C'],'.3f')),float(format(r['nll'],'.1f')),r['parameter_count']] for r in rows]
    if reported!=expected:raise AssertionError('Report model table differs from audited arrays/counts')
    pd.DataFrame(rows).to_csv(args.output/'model_table.csv',index=False)
    table=['| Model | Raw RMSE, C | Negative log likelihood | Parameters |',
           '|---|---:|---:|---:|']
    table += ['| {id} | {RMSE_C:.3f} | {nll:.1f} | {parameter_count} |'.format(**r) for r in rows]
    (args.output/'report_table.md').write_text('\n'.join(table)+'\n')
    plot(args,rows,read)
    write_json(args.output/'evidence.json',{'status':'passed','models':rows,'profiles':curves,'beta_profile':beta,
        'report_sha256':hashlib.sha256(args.report.read_bytes()).hexdigest(),'report_table_rows_checked':len(reported),
        'arithmetic_checks':arithmetic,'consumed_sha256':hashes,
        'scope':'Saved prediction arithmetic, counts, AICc, artifact hashes and audited support. Independent propagation/dense-likelihood audits remain separate. These checks do not establish scientific model adequacy.'})
    print(json.dumps({'status':'passed','models':len(rows),'audited_profile_curves':len(curves),
                      'max_RMSE_abs_error':max(r['RMSE_abs_error'] for r in arithmetic)}),flush=True)


def plot(args,rows,read):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'savefig.dpi':150})
    select=['chain1','chain2','flexible3','power_curve_linear','mapped_chain2','mapped_chain3','mapped_unequal3']
    values={r['id']:r for r in rows};pos=np.arange(len(select))
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
    axes[0].barh(pos,[values[n]['RMSE_C'] for n in select]);axes[0].set_xlabel('Raw RMSE, C')
    for j,n in enumerate(select):
        r=values[n]
        axes[1].plot([r['fast_cross_min'],r['fast_cross_max']],[j,j],'o-',color='#2166ac')
        axes[2].plot([r['innovation_SD_min'],r['innovation_SD_max']],[j,j],'o-',color='#2166ac')
    axes[1].set_xlabel('Fast runs: max |input correlation|')
    axes[2].set_xlabel('Innovation SD across twelve runs');axes[2].axvline(1,color='gray',linestyle=':')
    axes[0].set_yticks(pos);axes[0].set_yticklabels(select);axes[0].invert_yaxis()
    for ax in axes:ax.grid(axis='x',alpha=.2)
    fig.suptitle('Same-record diagnostics; ranges are across runs, not confidence intervals')
    fig.tight_layout();fig.savefig(args.output/'fit_and_innovations.png');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(10,9))
    for n in (1,2,3):
        for col,param in enumerate(['K','theta']):
            ax=axes[n-1,col]
            for folder,stem,label in [('chain_profiles','chain','Original mean'),
                ('mapped_profiles1' if n==1 else 'mapped_profiles23','mapped_chain','Command map + affine ambient')]:
                curve=read(folder+'/'+stem+str(n)+'_'+param+'.json')
                points=sorted(curve['points'],key=lambda r:r['value'])
                ax.plot([r['value'] for r in points],[r['twice_delta_nll'] for r in points],'o-',markersize=3,label=label)
            ax.axhline(3.841458820694124,color='gray',linestyle=':')
            ax.set(xlabel='K, C / heater percent' if param=='K' else 'Total chain theta, s',
                   ylabel='Twice profile NLL difference',title='n='+str(n),ylim=(-.2,12))
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Nominal bounded working profiles; model discrepancy limits coverage claims')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(args.output/'stage_profiles.png');plt.close(fig)
    write_json(args.output/'figure_receipt.json',{'visual_review':'pending',
        'figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.png')}})


if __name__=='__main__':main()
