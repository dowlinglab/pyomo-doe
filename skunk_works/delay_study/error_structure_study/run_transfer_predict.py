"""Blind-nuisance transfer scenarios, quantized noise and parameter sensitivities."""
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
from physical import load_records
from joint_quantized import Model
from quantized import report,filter_ou
from transfer import checked_scenarios,reported_mean,reported_quantile,mixture_quantile
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--parameter-draws',type=Path)
    args=parser.parse_args();begun=time.monotonic();config=json.loads((HERE/'transfer_config.json').read_text())
    fit=json.loads((args.input/'fit.json').read_text());assert fit['family']=='conductance' and fit['kernel']=='quantized_ou_nugget'
    if fit.get('dataset') or fit.get('fixed_coordinates') or fit.get('startup_band_C') or fit.get('initial_sensor_bins'):raise ValueError('Declare separate transfer initial-state/model scenario')
    if fit.get('condition_initial_code'):raise ValueError('Conditional-first-code marginal transfer needs its non-Gaussian initial latent mixture')
    all_records=load_records();train=[d for d in all_records if d['name'] in fit['record_names']]
    test=[d for d in all_records if d['name'] not in fit['record_names']]
    assert test and [d['name'] for d in train]==fit['record_names']
    for d in train:assert d['sha256']==fit['raw_sha256'][d['name']]
    model=Model(train,fit['band'],fit['phase'],128,'conductance',minimum_sensor_loading=fit.get('minimum_sensor_loading',.005))
    points=None;draw_metadata=None
    if args.parameter_draws:
        a=np.load(args.parameter_draws,allow_pickle=False);assert a['fit_sha256'].item()==hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest()
        points=a['points'];assert points.ndim==2 and points.shape[1]==len(model.bounds()) and np.isfinite(points).all()
        bounds=np.array(model.bounds());assert np.all(points>=bounds[:,0]-1e-7) and np.all(points<=bounds[:,1]+1e-7)
        assert all(abs(model.centered_constraint(p))<1e-7 for p in points)
        draw_metadata={'path':str(args.parameter_draws),'sha256':hashlib.sha256(args.parameter_draws.read_bytes()).hexdigest(),
            'draws':len(points),'method':str(a['method'].item()),'calibration_notice':str(a['calibration_notice'].item())}
    result=[];arrays={}
    for j,d in enumerate(test):
        scenarios=checked_scenarios(d,np.asarray(fit['best']['point'][:6]),fit['parameters']['K_common_nominal'],fit['band'],config['nuisance_scenario_grid'])
        predicted=scenarios['nominal'];y=d['y'][1:];obsmean=np.empty_like(y);noise_low=np.empty_like(y);noise_high=np.empty_like(y)
        outer_low=np.empty_like(y);outer_high=np.empty_like(y);scores=[]
        for k,noise in enumerate(fit['noise_parameters']):
            sd=np.hypot(noise['latent_SD_C'],noise['measurement_SD_C']);obsmean[:,k]=reported_mean(predicted[:,k],sd,fit['phase'])
            noise_low[:,k]=reported_quantile(predicted[:,k],sd,.025,fit['phase']);noise_high[:,k]=reported_quantile(predicted[:,k],sd,.975,fit['phase'])
            outer_low[:,k]=reported_quantile(scenarios['lower'][:,k],sd,.025,fit['phase']);outer_high[:,k]=reported_quantile(scenarios['upper'][:,k],sd,.975,fit['phase'])
            nodes=128
            while True:
                f=filter_ou(d['t'][1:],predicted[:,k],y[:,k],sd,noise['ell_s'],noise['measurement_SD_C']**2/sd**2,fit['phase'],nodes,12.)
                if np.max(abs(f['posterior_normalization_ratio']-1))<1e-6:break
                if nodes>=512:raise FloatingPointError('Transfer nominal bin score requires a finer integration')
                nodes*=2
            scores.append({'channel':k+1,'conditional_one_step_bin_nll':f['nll'],'nodes':nodes})
        row={'name':d['name'],'raw_sha256':d['sha256'],'nobs_per_channel':len(y),
            'latent_free_simulation_RMSE_C':np.sqrt(np.mean((y-predicted)**2,axis=0)).tolist(),
            'reported_conditional_mean_RMSE_C':np.sqrt(np.mean((y-obsmean)**2,axis=0)).tolist(),
            'signed_reported_mean_error_C':np.mean(y-obsmean,axis=0).tolist(),
            'nominal_mean_noise_only_marginal_coverage':np.mean((y>=noise_low)&(y<=noise_high),axis=0).tolist(),
            'hard_nuisance_plus_noise_outer_coverage':np.mean((y>=outer_low)&(y<=outer_high),axis=0).tolist(),
            'scenario_grid_checks':scenarios['grid_checks'],'conditional_bin_scores':scores,
            'parameter_uncertainty_included':points is not None}
        if points is not None:
            low=[];high=[];sds=[]
            for point in points:
                s=checked_scenarios(d,point[:6],np.exp(point[model.nshape]),fit['band'],config['nuisance_scenario_grid'])
                low.append(s['lower']);high.append(s['upper'])
                sds.append([np.hypot(np.exp(point[model.nmean+3*k]),np.exp(point[model.nmean+3*k+2])) for k in [0,1]])
            low=np.array(low);high=np.array(high);sds=np.array(sds);pl=np.empty_like(y);pu=np.empty_like(y)
            for k in [0,1]:
                pl[:,k]=report(mixture_quantile(low[:,:,k],sds[:,k],.025),fit['phase'])
                pu[:,k]=report(mixture_quantile(high[:,:,k],sds[:,k],.975),fit['phase'])
            row['parameter_mixture_hard_nuisance_noise_outer_coverage']=np.mean((y>=pl)&(y<=pu),axis=0).tolist()
            arrays.update({'parameter_mixture_reported_low_'+str(j):pl,'parameter_mixture_reported_high_'+str(j):pu})
        arrays.update({'time_'+str(j):d['t'],'observed_'+str(j):d['y'],'input_'+str(j):d['u'],
            'latent_nominal_'+str(j):predicted,'reported_mean_'+str(j):obsmean,
            'nuisance_mean_low_'+str(j):scenarios['lower'],'nuisance_mean_high_'+str(j):scenarios['upper'],
            'noise_only_reported_low_'+str(j):noise_low,'noise_only_reported_high_'+str(j):noise_high,
            'nuisance_noise_reported_low_'+str(j):outer_low,'nuisance_noise_reported_high_'+str(j):outer_high})
        result.append(row);write(args.output/'progress.json',{'completed_runs':len(result),'declared_runs':len(test),'elapsed_s':time.monotonic()-begun})
        fig,axes=plt.subplots(2,1,figsize=(7,5),sharex=True)
        for k,ax in enumerate(axes):
            t=d['t'][1:];ax.fill_between(t,outer_low[:,k],outer_high[:,k],alpha=.18,color='tab:blue',label='Hard nuisance + marginal noise envelope')
            ax.plot(t,y[:,k],'k.',ms=1,label='Reported temperature');ax.plot(t,obsmean[:,k],color='tab:blue',lw=1,label='Nominal reported mean')
            if points is not None:ax.plot(t,pl[:,k],color='tab:orange',lw=.7);ax.plot(t,pu[:,k],color='tab:orange',lw=.7,label='Parameter-mixture sensitivity')
            ax.set_ylabel('T'+str(k+1)+' (C)');ax.grid(alpha=.15)
        axes[0].legend(fontsize=7);axes[-1].set_xlabel('Time (s)');fig.suptitle(d['name'].replace('validation_experiment_env_2_','').replace('.csv',''))
        fig.tight_layout();fig.savefig(args.output/('transfer_'+str(j)+'.png'),dpi=150);plt.close(fig)
    write(args.output/'transfer_predictions.json',{'status':'complete','fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
        'train_names':fit['record_names'],'test_names':[d['name'] for d in test],'parameter_draws':draw_metadata,
        'per_run':result,'elapsed_s':time.monotonic()-begun,
        'notice':'Retrospective transfer: only the first test temperature pair initializes the mean; no later test output fits conductance, ambient or physical parameters. Hard-band/ambient envelopes are numerically converged scenarios, not distributions over nuisance parameters. Noise ranges are marginal and not simultaneous. Parameter-mixture sensitivities inherit the supplied draw approximation/calibration limits; no automatic total-95% claim. Filtered bin scores use past test outputs for the residual state and are distinct from free simulation. All twelve records informed structure selection.'})
    np.savez_compressed(args.output/'transfer_predictions.npz',**arrays)


if __name__=='__main__':main()
