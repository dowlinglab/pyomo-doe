"""Independent-stream PIT and nonrandomized checks with fixed-model references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from physical import load_records
from bin_diagnostics import filter_records,summaries,sample_observations
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--replicates',type=int,default=20)
    parser.add_argument('--simulations',type=int,default=50);parser.add_argument('--nodes',type=int,default=128)
    args=parser.parse_args();begun=time.monotonic();fit=json.loads((args.input/'fit.json').read_text())
    assert fit['kernel']=='quantized_ou_nugget' and not fit.get('dataset')
    lookup={d['name']:d for d in load_records()};records=[lookup[n] for n in fit['record_names']]
    arrays=np.load(args.input/'trajectories.npz');means=[arrays['predicted_'+str(j)] for j in range(len(records))]
    for j,d in enumerate(records):
        assert d['sha256']==fit['raw_sha256'][d['name']] and np.array_equal(d['y'],arrays['observed_'+str(j)])
    noise=fit['noise_parameters'];phase=fit['phase'];filtered=filter_records(records,means,noise,phase,args.nodes)
    actual=[];first_z=None
    for j in range(args.replicates):
        row,z=summaries(records,filtered,13071+10000000*j);row['randomization_seed']=13071+10000000*j;actual.append(row)
        if j==0:first_z=z
    del filtered
    write(args.output/'data_diagnostics.json',actual)
    simulated=[];failed=[];saved={};names=[d['name'] for d in records]
    for j in range(args.simulations):
        seed=29071+1000000*j
        try:
            synthetic=sample_observations(records,means,noise,phase,seed)
            sf=filter_records(synthetic,means,noise,phase,args.nodes)
            row,_=summaries(synthetic,sf,73071+10000000*j);del sf
            row.update({'simulation':j,'observation_seed':seed,'randomization_seed':73071+10000000*j})
            simulated.append(row)
            for k,d in enumerate(synthetic):saved['observed_'+str(j)+'_'+str(k)]=d['y']
        except Exception as exc:failed.append({'simulation':j,'observation_seed':seed,'exception':repr(exc)})
        write(args.output/'simulated_diagnostics.json',simulated);write(args.output/'failures.json',failed)
        write(args.output/'progress.json',{'attempted':j+1,'successful':len(simulated),'failed':len(failed),'elapsed_s':time.monotonic()-begun})
        print(json.dumps({'simulation':j,'successful':len(simulated),'failed':len(failed)}),flush=True)
    if not simulated:raise RuntimeError('All fixed-model diagnostic simulations failed')
    comparisons={}
    for key in actual[0]['metrics']:
        a=np.array([r['metrics'][key] for r in actual]);s=np.array([r['metrics'][key] for r in simulated])
        comparisons[key]={'data_randomization_min_median_max':np.quantile(a,[0.,.5,1.]).tolist(),
            'fixed_model_reference_q025_q50_q975':np.quantile(s,[.025,.5,.975]).tolist(),
            'simulation_upper_tail_rank_for_data_median':float((1+np.sum(s>=np.median(a)))/(1+len(s)))}
    write(args.output/'comparison.json',{'status':'complete' if not failed else 'completed_with_failures',
        'fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
        'arrays_sha256':hashlib.sha256((args.input/'trajectories.npz').read_bytes()).hexdigest(),
        'randomizations':args.replicates,'simulation_attempts':args.simulations,'simulation_successes':len(simulated),
        'simulation_failures':len(failed),'phase':phase,'comparisons':comparisons,'elapsed_s':time.monotonic()-begun,
        'notice':'Conditional fixed-fitted-parameter simulation references, not refitted-bootstrap p-values or coverage calibration. All observed records informed selection. Separate independent random streams by run/channel/replicate. Transition counts are nonrandomized predictive checks. A Gaussian-looking randomized PIT alone does not establish adequate whitening.'})
    np.savez_compressed(args.output/'simulated_observations.npz',record_names=np.array(names),**saved)
    fig,axes=plt.subplots(2,2,figsize=(9,6.5))
    for k,color in [(0,'tab:blue'),(1,'tab:orange')]:
        pooled=np.sort(np.concatenate([r[k] for r in first_z]));theory=norm.ppf((np.arange(len(pooled))+.5)/len(pooled))
        axes[0,0].plot(theory[::20],pooled[::20],'.',ms=2,color=color,label='T'+str(k+1))
        acf=np.array([r['channels'][k]['acf_lags_1_60'] for r in simulated]);lo,hi=np.quantile(acf,[.025,.975],axis=0)
        axes[0,1].fill_between(np.arange(1,61),lo,hi,color=color,alpha=.15)
        axes[0,1].plot(np.arange(1,61),actual[0]['channels'][k]['acf_lags_1_60'],color=color,label='T'+str(k+1))
        q=np.array([r['channels'][k]['transition_standardized_discrepancy'] for r in simulated]);lo,hi=np.quantile(q,[.025,.975],axis=0)
        x=np.arange(3)+(.12 if k else -.12);v=np.array(actual[0]['channels'][k]['transition_standardized_discrepancy'])
        axes[1,0].vlines(x,lo,hi,color=color,lw=5,alpha=.35)
        axes[1,0].plot(x,v,'o',color=color,label='T'+str(k+1))
    axes[0,0].plot([-4,4],[-4,4],'k--',lw=.8);axes[0,0].set(xlabel='Normal quantile',ylabel='Randomized predictive residual',title='One recorded independent randomization')
    axes[0,1].set(xlabel='Sample lag (within runs)',ylabel='Pooled autocorrelation',title='PIT correlation and fixed-model references')
    axes[1,0].set(xticks=np.arange(3),xticklabels=['Down','Same code','Up'],ylabel='Standardized count discrepancy',title='Nonrandomized bin-transition checks')
    cross=np.array([r['cross_channel_lags_minus10_10'] for r in simulated]);lo,hi=np.quantile(cross,[.025,.975],axis=0)
    axes[1,1].fill_between(np.arange(-10,11),lo,hi,color='gray',alpha=.3)
    axes[1,1].plot(np.arange(-10,11),actual[0]['cross_channel_lags_minus10_10'],'k-')
    axes[1,1].set(xlabel='Sample lag (within runs)',ylabel='Cross-channel correlation',title='Independent-channel noise assumption')
    for ax in axes.flat:ax.axhline(0,color='gray',lw=.5);ax.grid(alpha=.15)
    axes[0,0].legend();axes[0,1].legend();axes[1,0].legend();fig.tight_layout()
    fig.savefig(args.output/'predictive_diagnostics.png',dpi=160);fig.savefig(args.output/'predictive_diagnostics.pdf');plt.close(fig)


if __name__=='__main__':main()
