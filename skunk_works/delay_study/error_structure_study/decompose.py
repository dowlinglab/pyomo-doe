"""Bind prior evidence and partition saved two-channel simulation errors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'coupled_model_study'))
sys.path.insert(0, str(HERE.parent/'shared_model_investigation'))
import numpy as np
import pandas as pd
from scipy import stats
from physical import load_records
from run_frequency import period_of


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def columns(t, period):
    scale = max(np.ptp(t), 1.)
    blocks = [np.ones((len(t), 1)), ((t-t.mean())/scale)[:, None]]
    if period:
        for k in (1, 2, 3):
            blocks.append(np.column_stack((np.sin(2*np.pi*k*t/period), np.cos(2*np.pi*k*t/period))))
    return blocks


def partition(t, e, period):
    blocks = columns(t, period)
    last = float(e@e)
    energies = []
    for j in range(len(blocks)):
        X = np.column_stack(blocks[:j+1])
        residual = e-X@np.linalg.lstsq(X, e, rcond=None)[0]
        current = float(residual@residual)
        energies.append(last-current)
        last = current
    if not period:
        energies.extend([0., 0., 0.])
    return energies+[last]


def diagnostic(z, lags):
    x = z-z.mean()
    den = float(x@x)
    acf = np.array([float(x[k:]@x[:-k]/den) if den else 0. for k in range(1, max(lags)+1)])
    n = len(z)
    return {'mean':float(z.mean()), 'SD':float(z.std()),
            'skewness':float(stats.skew(z)) if den else None,
            'excess_kurtosis':float(stats.kurtosis(z)) if den else None,
            'abs_exceedance_1_96_fraction':float(np.mean(abs(z)>1.96)),
            'ACF_lags_1_to_60':acf.tolist(),
            'Ljung_Box_Q':{str(h):float(n*(n+2)*np.sum(acf[:h]**2/(n-np.arange(1,h+1)))) for h in lags},
            'notice':'Descriptive diagnostics; no fitted-model or multiple-comparison p-value calibration yet.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((HERE/'config.json').read_text())
    records=load_records()
    prior_manifest=json.loads((args.prior/'run_manifest.json').read_text())
    assert not prior_manifest['live_jobs']
    bindings={}
    for name, expected in prior_manifest['bindings_sha256'].items():
        actual=digest(args.prior/'results'/name)
        assert actual==expected, ('Changed prior evidence',name)
        bindings[name]=actual
    prior_exits={}
    for job in prior_manifest['jobs']:
        path=args.prior/'results'/job['id']/'exit.json'
        value=json.loads(path.read_text())
        assert value['exit_code']==job['exit_code']==0, path
        prior_exits[job['id']]=value['exit_code']
    write(args.output/'reconciliation.json',{'prior_manifest_sha256':digest(args.prior/'run_manifest.json'),
        'prior_bindings_verified':bindings,'prior_terminal_exits':prior_exits,
        'raw_sha256':{d['name']:d['sha256'] for d in records},
        'config_sha256':digest(HERE/'config.json'), 'status':'passed',
        'controller_live_process_check':'pgrep delay_study|tclab-coupled|tclab-shared returned no matches before launch'})
    rows=[];innovations=[];aggregates=[];models={}
    components=['offset','additional_trend','additional_fundamental','additional_harmonic2','additional_harmonic3','remainder']
    for label in config['selected_prior_models']:
        path=args.prior/'results'/label
        fit=json.loads((path/'fit.json').read_text());arrays=np.load(path/'trajectories.npz')
        assert fit['record_names']==[d['name'] for d in records]
        for j,d in enumerate(records):
            t=arrays['time_'+str(j)];y=arrays['observed_'+str(j)];u=arrays['input_'+str(j)]
            p=arrays['predicted_'+str(j)];z=arrays['innovation_'+str(j)]
            assert fit['raw_sha256'][d['name']]==d['sha256']
            assert np.array_equal(t,d['t']) and np.array_equal(y,d['y']) and np.array_equal(u,d['u'])
            period=period_of(d['name'])
            for cutoff in config['diagnostic_startup_windows_s']:
                keep=t[1:]>=cutoff;ts=t[1:][keep];e=(y-p)[1:][keep]
                for c in range(2):
                    energy=partition(ts,e[:,c],period)
                    row={'model':label,'name':d['name'],'channel':c+1,'cutoff_s':cutoff,'period_s':period,
                         'nobs':len(ts),'SSE_C2':float(e[:,c]@e[:,c]),'mean_residual_C':float(e[:,c].mean()),
                         'RMSE_C':float(np.sqrt(np.mean(e[:,c]**2))),
                         'centered_RMSE_C':float(e[:,c].std()),
                         'cycles_in_window':float(np.ptp(ts)/period) if period else None}
                    row.update({key+'_SSE_C2':val for key,val in zip(components,energy)})
                    rows.append(row)
                    diag=diagnostic(z[keep,c],config['diagnostic_lags_samples'])
                    innovations.append({'model':label,'name':d['name'],'channel':c+1,'cutoff_s':cutoff,**diag})
        models[label]=(fit,arrays)
        for cutoff in config['diagnostic_startup_windows_s']:
            for c in (1,2):
                group=[r for r in rows if r['model']==label and r['channel']==c and r['cutoff_s']==cutoff]
                n=sum(r['nobs'] for r in group);sse=sum(r['SSE_C2'] for r in group)
                aggregates.append({'model':label,'channel':c,'cutoff_s':cutoff,'nobs':n,'SSE_C2':sse,'RMSE_C':np.sqrt(sse/n),
                    **{key+'_fraction':sum(r[key+'_SSE_C2'] for r in group)/sse for key in components}})
    pd.DataFrame(rows).to_csv(args.output/'decomposition.csv',index=False)
    pd.DataFrame(aggregates).to_csv(args.output/'aggregate.csv',index=False)
    write(args.output/'innovation_diagnostics.json',{'rows':innovations,'notice':'Raw saved OU+nugget whitened innovations, not quantization-aware or calibrated normality tests.'})
    previous=args.prior/'results/final_analysis/analysis.json'
    write(args.output/'frequency_evidence.json',{'source':str(previous),'sha256':digest(previous),
        'reuse':'Existing matched amplitudes/phases and HAC uncertainty are reused. No new frequency uncertainty estimates computed here.'})
    plots(args.output,aggregates,models,components)
    write(args.output/'figure_receipt.json',{'visual_review':'pending','sha256':{p.name:digest(p) for p in args.output.glob('*.png')}})
    print(json.dumps({'status':'complete','models':len(models),'decompositions':len(rows),'prior_exits':len(prior_exits)}),flush=True)


def plots(output, aggregates, models, components):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey='row')
    labels=['Base','Lag','Lag + curve','Affine ambient']
    for r,c in enumerate((1,2)):
        for col,cutoff in enumerate((0,450)):
            ax=axes[r,col];bottom=np.zeros(4)
            for component in components:
                values=np.array([a[component+'_fraction'] for a in aggregates if a['cutoff_s']==cutoff and a['channel']==c])
                ax.bar(np.arange(4),values,bottom=bottom,label=component.replace('additional_','').replace('_',' '));bottom+=values
            ax.set_xticks(np.arange(4));ax.set_xticklabels(labels,rotation=20)
            ax.set_title('T%d; after %d s'%(c,cutoff));ax.set_ylabel('Fraction of squared error')
            ax.set_ylim(0,1)
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Nested offset, trend and harmonic projections; remainder includes unmodeled effects')
    fig.tight_layout();fig.savefig(output/'error_partition.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(10,8),sharex='row')
    fit,a=models['main_lag_curve_ou']
    for row,j in enumerate((0,6,9)):
        t=a['time_'+str(j)];y=a['observed_'+str(j)];p=a['predicted_'+str(j)];e=y-p
        for c in range(2):
            axes[row,c].plot(t[1:],e[1:,c],lw=.7,label='Residual')
            axes[row,c].axhline(0,color='k',lw=.5)
            axes[row,c].set_title(fit['record_names'][j]+'; T'+str(c+1),fontsize=8)
            axes[row,c].set_ylabel('Error, C');axes[row,c].grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Time, s')
    fig.tight_layout();fig.savefig(output/'selected_residuals.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
