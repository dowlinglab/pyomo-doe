"""Decompose reference errors and estimate measured harmonic response per repeat."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'residual_study'))
import numpy as np
import pandas as pd
from frequency import audit, held_fundamental, regression, response_summary
from run_mean import write_json


def period_of(name):
    for stem, period in [('sin_50_50', 30.), ('sin_2_50', 120.), ('sin_5_50', 300.)]:
        if stem+'_run' in name:
            return period
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output/'method_audit.json', audit())
    fit_path = args.baseline/'fit.json'
    arrays_path = args.baseline/'trajectories.npz'
    fit = json.loads(fit_path.read_text())
    arrays = np.load(arrays_path)
    best = fit['best']
    rows, decomposition, folded = [], [], []
    for j, name in enumerate(fit['record_names']):
        t = arrays['time_'+str(j)]
        t = t-t[0]
        y = arrays['observed_'+str(j)]
        pred = arrays['predicted_'+str(j)]
        u = arrays['input_'+str(j)]
        e = y-pred
        period = period_of(name)
        tr, er = t[1:], e[1:]
        sse = float(er@er)
        offset = float(len(er)*er.mean()**2)
        X = np.column_stack((np.ones(len(tr)), (tr-tr.mean())/(tr[-1]-tr[0])))
        after_trend = er-X@np.linalg.lstsq(X, er, rcond=None)[0]
        trend_sse = float(np.sum((er-er.mean())**2)-after_trend@after_trend)
        harmonic_sse = 0.
        if period:
            Xh = np.column_stack((X, np.sin(2*np.pi*tr/period), np.cos(2*np.pi*tr/period)))
            rem = er-Xh@np.linalg.lstsq(Xh, er, rcond=None)[0]
            harmonic_sse = float(after_trend@after_trend-rem@rem)
        decomposition.append({'name': name, 'period_s': period, 'nobs': len(er), 'SSE_C2': sse,
                              'mean_residual_C': float(er.mean()), 'constant_SSE_C2': offset,
                              'additional_trend_SSE_C2': trend_sse, 'additional_fundamental_SSE_C2': harmonic_sse,
                              'remaining_SSE_C2': sse-offset-trend_sse-harmonic_sse,
                              'ambient_offset_C': best['coefficients'][1+j],
                              'ambient_at_bound': bool(abs(best['coefficients'][1+j]) > 2-1e-7)})
        if not period:
            continue
        for cutoff in (300., 450., 600.):
            keep = t >= cutoff
            ts = t[keep]
            inp = held_fundamental(t, u, 2*np.pi/period, cutoff, t[-1])
            for nuisance in ('trend', 'reference_exponentials'):
                kwargs = {'tau': best['tau'], 'theta': best['theta'], 'nuisance': nuisance}
                measured = regression(ts, y[keep], period, **kwargs)
                model = regression(ts, pred[keep], period, **kwargs)
                m = response_summary(measured, inp)
                p = response_summary(model, inp)
                theoretical = best['coefficients'][0]/((1+1j*2*np.pi/period*best['tau'])*(1+1j*2*np.pi/period*best['theta']))
                row = {'name': name, 'run': int(name.split('_run_')[1].split('.')[0]),
                       'period_s': period, 'cutoff_s': cutoff, 'nuisance': nuisance,
                       'observations': int(keep.sum()), 'cycles_in_window': float((t[-1]-cutoff)/period),
                       'design_condition': measured['condition'], 'design_rank': measured['rank'],
                       'measured': m, 'matched_sampled_n1': p,
                       'continuous_n1_gain': float(abs(theoretical)), 'continuous_n1_phase_deg': float(np.angle(theoretical, deg=True)),
                       'phase_difference_from_matched_n1_deg': float(np.angle(complex(m['H_real'],m['H_imag'])/complex(p['H_real'],p['H_imag']),deg=True)),
                       'gain_ratio_to_matched_n1': m['gain_C_per_percent']/p['gain_C_per_percent']}
                rows.append(row)
                if cutoff == 300 and nuisance == 'trend':
                    residual_fit = regression(ts, e[keep], period, nuisance='trend')
                    phase = np.mod(2*np.pi*ts/period+np.angle(complex(inp[1], inp[2])), 2*np.pi)
                    phase_bin = np.minimum(11, (phase/(2*np.pi)*12).astype(int))
                    raw = e[keep]-residual_fit['nuisance_component']
                    innovations = arrays['standardized_innovation_'+str(j)][keep[1:]]
                    innovation_fit = regression(ts, innovations, period, nuisance='trend')
                    centered_i = innovations-innovation_fit['nuisance_component']
                    folded.append({'name': name, 'period_s': period, 'run': row['run'],
                                   'phase_midpoint_rad': ((np.arange(12)+.5)*2*np.pi/12).tolist(),
                                   'bin_counts': [int(np.sum(phase_bin == k)) for k in range(12)],
                                   'residual_cycle_C': [float(np.mean(raw[phase_bin == k])) for k in range(12)],
                                   'innovation_cycle': [float(np.mean(centered_i[phase_bin == k])) for k in range(12)]})
    pd.DataFrame(decomposition).to_csv(args.output/'error_decomposition.csv', index=False)
    write_json(args.output/'frequency_estimates.json', {'rows': rows,
        'baseline_fit_sha256': hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        'baseline_arrays_sha256': hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        'windows_declared_before_measurement_s': [300,450,600], 'HAC_lags_samples': [30,60,120],
        'notice': 'Descriptive per-run estimates, input treated fixed. HAC asymptotic working uncertainty; sample lag approximates seconds. Reference-exponential sensitivity conditions on pooled rates. No independent coverage or pure-delay identification claim.'})
    write_json(args.output/'phase_folds.json', {'runs': folded,
        'notice': '12 phase bins after300s; joint fitted offset/trend removed; bin sample counts are not independent repetitions'})
    make_plots(args.output, rows, folded)
    print(json.dumps({'status':'complete','frequency_cells':len(rows),'folded_repeats':len(folded)}), flush=True)


def make_plots(output, rows, folded):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9, 'savefig.dpi':160})
    colors = ['#2166ac','#b2182b','#1b7837']
    fig, axes = plt.subplots(2,3,figsize=(12,6),sharex=True)
    for col, period in enumerate((30,120,300)):
        for run in (1,2,3):
            selected = [r for r in rows if r['period_s']==period and r['run']==run and r['nuisance']=='trend']
            for k, key in enumerate(('gain_C_per_percent','phase_deg')):
                ax = axes[k,col]
                sd = 'gain_SD' if k==0 else 'phase_SD_deg'
                ax.errorbar([r['cutoff_s']+(run-2)*6 for r in selected], [r['measured'][key] for r in selected],
                            yerr=[1.96*r['measured']['HAC']['60'][sd] for r in selected],
                            fmt='o-',color=colors[run-1],label='Run '+str(run),capsize=2)
                ax.plot([r['cutoff_s'] for r in selected], [r['matched_sampled_n1'][key] for r in selected],
                        color=colors[run-1],linestyle='--',alpha=.6)
                ax.grid(alpha=.2)
        axes[0,col].set_title(str(period)+'-second sine')
        axes[1,col].set_xlabel('Startup cutoff, s')
    axes[0,0].set_ylabel('Gain, C / heater percent')
    axes[1,0].set_ylabel('Output-input phase, degrees')
    axes[0,0].legend()
    fig.suptitle('Measured response (solid) and matched n=1 (dashed); working HAC60 ranges')
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(output/'frequency_windows.png')
    plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(12,6),sharex=True)
    for col, period in enumerate((30,120,300)):
        for row in [r for r in folded if r['period_s']==period]:
            for k,key in enumerate(('residual_cycle_C','innovation_cycle')):
                axes[k,col].plot(np.rad2deg(row['phase_midpoint_rad']),row[key],'-o',markersize=3,
                                 color=colors[row['run']-1],label='Run '+str(row['run']))
                axes[k,col].axhline(0,color='gray',linewidth=.5)
                axes[k,col].grid(alpha=.2)
        axes[0,col].set_title(str(period)+'-second sine')
        axes[1,col].set_xlabel('Actual input phase, degrees')
    axes[0,0].set_ylabel('Detrended residual, C')
    axes[1,0].set_ylabel('Detrended standardized innovation')
    axes[0,0].legend()
    fig.suptitle('Phase-aligned repeats after300s; descriptive bins, no independent-sample bands')
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(output/'phase_repeats.png')
    plt.close(fig)
    write_json(output/'figure_receipt.json', {'visual_review':'pending',
        'figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.png')}})


if __name__ == '__main__':
    main()
