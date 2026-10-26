"""Matched nonrandomized checks using saved independent code simulations."""
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
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm
from physical import load_records
from quantized import bins
from bin_diagnostics import filter_records
from nonrandom import summaries
from run_bounded import write


def prefix_cdf_audit(records,means,noise,filtered,phase):
    """Direct Gaussian prefix rectangles, independent of latent-state filtering."""
    rows=[]
    for j,d in enumerate(records):
        t=d['t'][1:4]
        for k,p in enumerate(noise):
            S=p['latent_SD_C']**2;R=p['measurement_SD_C']**2
            C=S*np.exp(-abs(t[:,None]-t[None,:])/p['ell_s'])+R*np.eye(3)
            lo,hi=bins(d['y'][1:4,k],phase);lo-=means[j][1:4,k];hi-=means[j][1:4,k]
            sd0=np.sqrt(C[0,0]);beta=C[1,0]/C[0,0];sd1=np.sqrt(C[1,1]-C[1,0]**2/C[0,0])
            cov2=C[:2,:2];b2=np.linalg.solve(cov2,C[:2,2]);sd2=np.sqrt(C[2,2]-C[2,:2]@b2)
            def density0(x):return norm.pdf(x/sd0)/sd0
            den0=ndtr(hi[0]/sd0)-ndtr(lo[0]/sd0)
            def density2(x,y):return density0(x)*norm.pdf((y-beta*x)/sd1)/sd1
            den2=quad(lambda x:density0(x)*(ndtr((hi[1]-beta*x)/sd1)-ndtr((lo[1]-beta*x)/sd1)),lo[0],hi[0],epsabs=1e-12,epsrel=1e-10)[0]
            assert den0>1e-12 and den2>1e-12,('Rare-prefix audit needs scaled integration',d['name'],k)
            for index in range(3):
                for side,cut in [('lower',lo[index]),('upper',hi[index])]:
                    if index==0:expected=ndtr(cut/sd0)
                    elif index==1:expected=quad(lambda x:density0(x)*ndtr((cut-beta*x)/sd1),lo[0],hi[0],epsabs=1e-12,epsrel=1e-10)[0]/den0
                    else:
                        expected=quad(lambda x:quad(lambda y:density2(x,y)*ndtr((cut-b2[0]*x-b2[1]*y)/sd2),lo[1],hi[1],epsabs=1e-12,epsrel=1e-10)[0],lo[0],hi[0],epsabs=1e-12,epsrel=1e-10)[0]/den2
                    got=float(filtered[j][k]['cdf_'+side][index]);error=abs(got-expected)
                    assert error<2e-7,(d['name'],k,index,side,got,expected,error)
                    rows.append({'name':d['name'],'channel':k+1,'prefix_index':index,'side':side,
                        'filter_CDF':got,'direct_rectangle_CDF':float(expected),'absolute_error':float(error)})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--nodes',type=int,default=128);parser.add_argument('--render-from',type=Path)
    args=parser.parse_args();begun=time.monotonic()
    if args.render_from:
        assert json.loads((args.render_from/'exit.json').read_text())['exit_code']==0
        actual=json.loads((args.render_from/'data_diagnostics.json').read_text())
        simulated=json.loads((args.render_from/'simulated_diagnostics.json').read_text())
        make_plots(args.output,actual,simulated)
        write(args.output/'render_receipt.json',{'status':'rendered','numeric_source':str(args.render_from),
            'source_sha256':{name:hashlib.sha256((args.render_from/name).read_bytes()).hexdigest() for name in ['data_diagnostics.json','simulated_diagnostics.json','comparison.json']},
            'notice':'Rendering only; no filtering, optimization or simulation repeated. Colorbar extensions disclose capped heatmap values.'})
        return
    fit=json.loads((args.input/'fit.json').read_text());reference=json.loads((args.references/'comparison.json').read_text())
    fitsha=hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest()
    assert reference['fit_sha256']==fitsha and reference['simulation_failures']==0
    assert json.loads((args.references/'exit.json').read_text())['exit_code']==0
    assert not fit.get('condition_initial_code') and not fit.get('dataset')
    lookup={d['name']:d for d in load_records()};records=[lookup[n] for n in fit['record_names']]
    arrays=np.load(args.input/'trajectories.npz');means=[arrays['predicted_'+str(j)] for j in range(len(records))]
    for j,d in enumerate(records):assert d['sha256']==fit['raw_sha256'][d['name']] and np.array_equal(d['y'],arrays['observed_'+str(j)])
    filtered=filter_records(records,means,fit['noise_parameters'],fit['phase'],args.nodes)
    prefixes=prefix_cdf_audit(records,means,fit['noise_parameters'],filtered,fit['phase'])
    write(args.output/'prefix_CDF_audit.json',{'status':'passed','checks':prefixes})
    actual=summaries(records,filtered);del filtered;write(args.output/'data_diagnostics.json',actual)
    synthetic=np.load(args.references/'simulated_observations.npz');assert list(synthetic['record_names'])==fit['record_names']
    references=json.loads((args.references/'simulated_diagnostics.json').read_text());simulated=[];failed=[]
    for row in references:
        index=row['simulation']
        try:
            data=[dict(d,y=synthetic['observed_'+str(index)+'_'+str(j)]) for j,d in enumerate(records)]
            f=filter_records(data,means,fit['noise_parameters'],fit['phase'],args.nodes)
            result=summaries(data,f);del f;result.update({'simulation':index,'observation_seed':row['observation_seed']});simulated.append(result)
        except Exception as exc:failed.append({'simulation':index,'exception':repr(exc)})
        write(args.output/'simulated_diagnostics.json',simulated);write(args.output/'failures.json',failed)
        write(args.output/'progress.json',{'attempted':len(simulated)+len(failed),'successful':len(simulated),'failed':len(failed),'elapsed_s':time.monotonic()-begun})
        print(json.dumps({'simulation':index,'successful':len(simulated),'failed':len(failed)}),flush=True)
    if not simulated:raise RuntimeError('All nonrandomized reference checks failed')
    comparisons={}
    for key,value in actual['metrics'].items():
        values=np.array([s['metrics'][key] for s in simulated])
        comparisons[key]={'data':value,'reference_q025_q50_q975':np.quantile(values,[.025,.5,.975]).tolist(),
            'upper_tail_rank':float((1+np.sum(values>=value))/(1+len(values)))}
    write(args.output/'comparison.json',{'status':'complete' if not failed else 'completed_with_failures',
        'fit_sha256':fitsha,'saved_reference_comparison_sha256':hashlib.sha256((args.references/'comparison.json').read_bytes()).hexdigest(),
        'saved_observations_sha256':hashlib.sha256((args.references/'simulated_observations.npz').read_bytes()).hexdigest(),
        'attempts':len(references),'successes':len(simulated),'failures':len(failed),'comparisons':comparisons,
        'elapsed_s':time.monotonic()-begun,'scope':'Matched deterministic statistics on data and fifty saved fixed-model simulations; no PIT jitter. Reference parameters are fitted, not refitted per simulation; ranks are descriptive and do not repeat model selection. Conditional score variance is below 1 under coarsening, so no Gaussian/unit-variance claim.'})
    make_plots(args.output,actual,simulated)


def make_plots(output,actual,simulated):
    fig,axes=plt.subplots(2,2,figsize=(9,6.5))
    for k,color in [(0,'tab:blue'),(1,'tab:orange')]:
        refs=np.array([s['score_channels'][k]['acf_lags_1_60'] for s in simulated]);lo,hi=np.quantile(refs,[.025,.975],axis=0)
        axes[0,0].fill_between(np.arange(1,61),lo,hi,color=color,alpha=.15)
        axes[0,0].plot(np.arange(1,61),actual['score_channels'][k]['acf_lags_1_60'],color=color,label='T'+str(k+1))
    refs=np.array([s['score_cross_channel_lags_minus10_10'] for s in simulated]);lo,hi=np.quantile(refs,[.025,.975],axis=0)
    axes[0,1].fill_between(np.arange(-10,11),lo,hi,color='gray',alpha=.25)
    axes[0,1].plot(np.arange(-10,11),actual['score_cross_channel_lags_minus10_10'],'k-')
    for ax,key,title in [(axes[1,0],'pooled_joint_transition_standardized','Joint down/same/up counts'),(axes[1,1],'pooled_centered_event_cross_standardized','Centered cross-channel events')]:
        im=ax.imshow(actual['joint_transition'][key],vmin=-3,vmax=3,cmap='RdBu_r');fig.colorbar(im,ax=ax,extend='both')
        ax.set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=['Down','Same','Up'],yticklabels=['Down','Same','Up'],xlabel='T2 event',ylabel='T1 event',title=title)
    axes[0,0].set(xlabel='Sample lag within runs',ylabel='Score autocorrelation',title='No randomized PIT jitter');axes[0,0].legend()
    axes[0,1].set(xlabel='Sample lag within runs',ylabel='Score cross-correlation',title='Independent-channel working model')
    for ax in axes[0]:ax.axhline(0,color='gray',lw=.5);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'nonrandomized_diagnostics.png',dpi=160);fig.savefig(output/'nonrandomized_diagnostics.pdf');plt.close(fig)


if __name__=='__main__':main()
