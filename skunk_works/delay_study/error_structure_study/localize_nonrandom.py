"""Locate deterministic code-score dependence and separate phase projections."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
import matplotlib.pyplot as plt
from physical import load_records
from run_frequency import period_of
from bin_diagnostics import filter_records
from nonrandom import expected_gaussian_pit
from run_bounded import write


def cross_curves(scores):
    curve=[]
    for lag in range(-10,11):
        pairs=[]
        for pair in scores:
            a,b=pair
            if lag>0:a,b=a[lag:],b[:-lag]
            elif lag<0:a,b=a[:lag],b[-lag:]
            pairs.append((a,b))
        curve.append(float(np.corrcoef(np.concatenate([p[0] for p in pairs]),np.concatenate([p[1] for p in pairs]))[0,1]))
    return curve


def describe(records,scores):
    per=[];centered=[];phase_removed=[];trend_removed=[]
    for d,pair in zip(records,scores):
        t=d['t'][1:];trend=np.column_stack([np.ones(len(t)),(t-t.mean())/np.ptp(t)])
        period=period_of(d['name']);X=trend
        if period:X=np.column_stack([trend]+[f(2*np.pi*h*t/period) for h in [1,2,3] for f in [np.sin,np.cos]])
        cc=[];pp=[];tt=[];channel=[]
        for k,v in enumerate(pair):
            cc.append(v-v.mean());tt.append(v-trend@np.linalg.lstsq(trend,v,rcond=None)[0]);pp.append(v-X@np.linalg.lstsq(X,v,rcond=None)[0])
            channel.append({'channel':k+1,'input_correlation':float(np.corrcoef(v,d['u'][:-1])[0,1]) if np.ptp(d['u'][:-1]) else None,
                'trend_projection_fraction':float(1-tt[-1]@tt[-1]/np.sum((v-v.mean())**2)),
                'trend_and_phase_projection_fraction':float(1-pp[-1]@pp[-1]/np.sum((v-v.mean())**2))})
        centered.append(cc);trend_removed.append(tt);phase_removed.append(pp)
        raw=cross_curves([pair]);phase=cross_curves([pp])
        per.append({'name':d['name'],'period_s':period,'channels':channel,
            'raw_cross_lags_minus10_10':raw,'after_trend_and_phase_cross_lags_minus10_10':phase,
            'raw_cross_max_abs':float(max(abs(np.array(raw)))),'after_trend_and_phase_cross_max_abs':float(max(abs(np.array(phase))))})
    curves={'raw':cross_curves(scores),'run_mean_removed':cross_curves(centered),
        'run_trend_removed':cross_curves(trend_removed),'run_trend_and_phase_removed':cross_curves(phase_removed)}
    metrics={key+'_pooled_cross_max_abs':float(max(abs(np.array(value)))) for key,value in curves.items()}
    metrics.update({'maximum_run_cross_abs':max(r['raw_cross_max_abs'] for r in per),
        'maximum_run_after_trend_and_phase_cross_abs':max(r['after_trend_and_phase_cross_max_abs'] for r in per)})
    return {'metrics':metrics,'curves':curves,'per_run':per}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--references',type=Path,required=True)
    args=parser.parse_args();begun=time.monotonic();fit=json.loads((args.input/'fit.json').read_text())
    reference=json.loads((args.references/'comparison.json').read_text());fitsha=hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest()
    assert reference['fit_sha256']==fitsha and reference['simulation_failures']==0 and not fit.get('condition_initial_code')
    lookup={d['name']:d for d in load_records()};records=[lookup[n] for n in fit['record_names']]
    arrays=np.load(args.input/'trajectories.npz');means=[arrays['predicted_'+str(j)] for j in range(len(records))]
    saved={};synthetic=np.load(args.references/'simulated_observations.npz');refs=json.loads((args.references/'simulated_diagnostics.json').read_text())
    def process(data,label):
        filtered=filter_records(data,means,fit['noise_parameters'],fit['phase']);scores=[]
        for j,pair in enumerate(filtered):
            v=[expected_gaussian_pit(f)[0] for f in pair];scores.append(v)
            for k in [0,1]:saved[label+'_'+str(j)+'_'+str(k)]=v[k]
        del filtered;return describe(data,scores)
    actual=process(records,'data');write(args.output/'data_localization.json',actual);results=[];failures=[]
    for ref in refs:
        index=ref['simulation'];data=[dict(d,y=synthetic['observed_'+str(index)+'_'+str(j)]) for j,d in enumerate(records)]
        try:
            result=process(data,'simulation_'+str(index));result.update({'simulation':index,'observation_seed':ref['observation_seed']});results.append(result)
        except Exception as exc:failures.append({'simulation':index,'exception':repr(exc)})
        write(args.output/'simulated_localization.json',results);write(args.output/'failures.json',failures)
        write(args.output/'progress.json',{'attempted':len(results)+len(failures),'successful':len(results),'failed':len(failures),'elapsed_s':time.monotonic()-begun})
    if not results:raise RuntimeError('No successful localization reference')
    comparisons={}
    for key,value in actual['metrics'].items():
        v=np.array([r['metrics'][key] for r in results]);comparisons[key]={'data':value,'reference_q025_q50_q975':np.quantile(v,[.025,.5,.975]).tolist(),'upper_tail_rank':float((1+np.sum(v>=value))/(1+len(v)))}
    write(args.output/'comparison.json',{'status':'complete' if not failures else 'completed_with_failures',
        'fit_sha256':fitsha,'saved_observations_sha256':hashlib.sha256((args.references/'simulated_observations.npz').read_bytes()).hexdigest(),
        'attempts':len(refs),'successes':len(results),'failures':len(failures),'comparisons':comparisons,
        'scope':'Descriptive localization prompted by nonrandomized cross-dependence. Same projection applied to data and fixed-model simulations. Removing a phase projection is not a fitted physical correction or stochastic whitening; residual cross structure can still reflect mean misspecification. No causal environmental attribution, refitted p-value or model-selection adjustment.'})
    np.savez_compressed(args.output/'conditional_expected_scores.npz',record_names=np.array(fit['record_names']),**saved)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['raw','run_trend_and_phase_removed']):
        v=np.array([r['curves'][key] for r in results]);lo,hi=np.quantile(v,[.025,.975],axis=0)
        ax.fill_between(np.arange(-10,11),lo,hi,color='gray',alpha=.25);ax.plot(np.arange(-10,11),actual['curves'][key],'k-')
        ax.axhline(0,color='gray',lw=.5);ax.set(xlabel='Sample lag within runs',ylabel='Code-score cross-correlation',title=key.replace('_',' '));ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'cross_localization.png',dpi=160);fig.savefig(args.output/'cross_localization.pdf');plt.close(fig)


if __name__=='__main__':main()
