"""Audit retrospective forecasts, code probabilities and reported metrics.

Full nominal trajectories and all array-derived RMSE/coverage are checked.
Parameter-mixture CDF and component-physics checks use saved, declared sparse
witness times; the numerical kernel's separate limiting-case checks remain
necessary. Neither check establishes predictive calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.integrate import nquad
from scipy.special import ndtr
from scipy.stats import multivariate_normal
from physical import load_records
from audit_bounded import independent
from continuation import parent_receipt
from quantized import STEP,OFFSET,bins,filter_ou
from transfer import checked_scenarios
from run_bounded import write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direct(record,parameters,K,g=1.,ambient=0.):
    legacy={'parameters':dict(parameters,Ua_run_over_nominal=[g]),'record_names':[record['name']],
        'coefficients':[K,ambient],'lag':True,'curve':True,'ambient':'constant'}
    return independent(record,legacy)[1:]


def expected_reported(mean,sd,phase):
    # Independent CDF subtraction over a wide common code grid. The production
    # implementation uses stable log-bin probabilities instead.
    code=np.arange(np.floor((mean.min()-11*sd-OFFSET)/STEP)-2,
                   np.ceil((mean.max()+11*sd-OFFSET)/STEP)+3)
    lo=STEP*(code+phase)+OFFSET;hi=lo+STEP
    mass=ndtr((hi[None,:]-mean[:,None])/sd)-ndtr((lo[None,:]-mean[:,None])/sd)
    assert np.max(abs(mass.sum(axis=1)-1))<1e-10
    return mass@np.round(STEP*code+OFFSET,2)


def quantile_check(codes,means,sds,p,phase,mixture=False):
    assert np.isfinite(codes).all() and np.isfinite(means).all() and np.isfinite(sds).all()
    assert np.all(np.asarray(sds)>0) and 0<p<1
    lo,hi=bins(codes,phase)
    if mixture:
        lower=np.mean(ndtr((lo[None,:]-means)/sds[:,None]),axis=0)
        upper=np.mean(ndtr((hi[None,:]-means)/sds[:,None]),axis=0)
    else:lower=ndtr((lo-means)/sds);upper=ndtr((hi-means)/sds)
    assert np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(lower<=upper)
    error=float(max(0.,np.max(lower-p),np.max(p-upper)))
    assert error<1e-10,('Discrete quantile CDF bracket',error)
    return error


def audit(args):
    prediction=json.loads((args.input/'transfer_predictions.json').read_text())
    assert json.loads((args.input/'exit.json').read_text())['exit_code']==0
    fitpath=args.fit/'fit.json';fit=json.loads(fitpath.read_text())
    check=json.loads(args.fit_audit.read_text());assert check['status']=='passed' and check['fit_sha256']==sha(fitpath)
    parent_receipt(args.fit_audit.parent,require_terminal=True)
    assert prediction['fit_sha256']==sha(fitpath) and prediction['status']=='complete'
    assert fit['family']=='conductance' and not any(fit.get(k) for k in ['dataset','fixed_coordinates','startup_band_C','initial_sensor_bins','condition_initial_code'])
    records=load_records();train=[d for d in records if d['name'] in fit['record_names']]
    test=[d for d in records if d['name'] not in fit['record_names']]
    assert prediction['train_names']==[d['name'] for d in train]==fit['record_names']
    assert prediction['test_names']==[d['name'] for d in test] and test
    for d in train:assert fit['raw_sha256'][d['name']]==d['sha256']
    arrays=np.load(args.input/'transfer_predictions.npz',allow_pickle=False)
    config=json.loads((HERE/'transfer_config.json').read_text());K=fit['parameters']['K_common_nominal'];phase=fit['phase']
    draws=None;draw_receipt=None
    if prediction['parameter_draws']:
        metadata=prediction['parameter_draws'];path=Path(metadata['path']);assert sha(path)==metadata['sha256']
        receipt_path=Path(metadata['draw_receipt_path']);assert sha(receipt_path)==metadata['draw_receipt_sha256']
        draw_receipt=json.loads(receipt_path.read_text())
        assert draw_receipt['status']=='complete' and draw_receipt['fit_sha256']==sha(fitpath)
        assert draw_receipt['parameter_files_sha256'][path.name]==sha(path)
        parent=parent_receipt(path.parent,require_terminal=True);assert parent['exit']['exit_code']==0
        draws=np.load(path,allow_pickle=False)['points'];assert len(draws)==metadata['draws']
        assert draws.shape[1]==len(fit['coordinate_names']) and np.isfinite(draws).all()
    rows=[]
    assert len(prediction['per_run'])==len(test)
    for j,(record,saved) in enumerate(zip(test,prediction['per_run'])):
        assert saved['name']==record['name'] and saved['raw_sha256']==record['sha256']
        for key,value in [('time',record['t']),('input',record['u']),('observed',record['y'])]:
            assert np.array_equal(arrays[key+'_'+str(j)],value)
        observed=record['y'][1:];n=len(observed);nominal=direct(record,fit['parameters'],K)
        assert saved['nobs_per_channel']==n and saved['parameter_uncertainty_included']==(draws is not None)
        assert np.max(abs(nominal-arrays['latent_nominal_'+str(j)]))<2e-8
        scenario=checked_scenarios(record,np.array(fit['best']['point'][:6]),K,fit['band'],config['nuisance_scenario_grid'])
        for label in ['lower','upper']:
            assert np.max(abs(scenario[label]-arrays['nuisance_mean_'+('low' if label=='lower' else 'high')+'_'+str(j)]))<1e-8
        containment=0.
        for g in sorted(set([1-fit['band'],1.,1+fit['band']])):
            for ambient in [-2.,2.]:
                physical=direct(record,fit['parameters'],K,g,ambient)
                error=max(0.,float(np.max(scenario['lower']-physical)),float(np.max(physical-scenario['upper'])))
                containment=max(containment,error);assert error<2e-8
        metric={'latent_free_simulation_RMSE_C':np.sqrt(np.mean((observed-nominal)**2,axis=0)).tolist()}
        reported=arrays['reported_mean_'+str(j)]
        metric.update({'reported_conditional_mean_RMSE_C':np.sqrt(np.mean((observed-reported)**2,axis=0)).tolist(),
            'signed_reported_mean_error_C':np.mean(observed-reported,axis=0).tolist()})
        for prefix,key in [('noise_only','nominal_mean_noise_only_marginal_coverage'),
                           ('nuisance_noise','hard_nuisance_plus_noise_outer_coverage')]:
            low=arrays[prefix+'_reported_low_'+str(j)];high=arrays[prefix+'_reported_high_'+str(j)]
            assert low.shape==observed.shape and np.all(low<=high)
            metric[key]=np.mean((observed>=low)&(observed<=high),axis=0).tolist()
        if draws is not None:
            low=arrays['parameter_mixture_reported_low_'+str(j)];high=arrays['parameter_mixture_reported_high_'+str(j)]
            assert np.all(low<=high)
            metric['parameter_mixture_hard_nuisance_noise_outer_coverage']=np.mean((observed>=low)&(observed<=high),axis=0).tolist()
        for key,value in metric.items():assert np.max(abs(np.array(value)-saved[key]))<1e-10,(record['name'],key)
        channel=[]
        for k,noise in enumerate(fit['noise_parameters']):
            S=noise['latent_SD_C']**2;R=noise['measurement_SD_C']**2;sd=np.sqrt(S+R)
            expectation_error=float(np.max(abs(expected_reported(nominal[:,k],sd,phase)-reported[:,k])))
            assert expectation_error<2e-8
            brackets=[]
            for prefix,means in [('noise_only',nominal),('nuisance_noise',None)]:
                for suffix,p in [('low',.025),('high',.975)]:
                    center=means[:,k] if means is not None else scenario['lower' if suffix=='low' else 'upper'][:,k]
                    brackets.append(quantile_check(arrays[prefix+'_reported_'+suffix+'_'+str(j)][:,k],center,sd,p,phase))
            score=saved['conditional_bin_scores'][k];nodes=max(256,score['nodes'])
            f=filter_ou(record['t'][1:],nominal[:,k],observed[:,k],sd,noise['ell_s'],R/(S+R),phase,nodes,12.)
            assert np.max(abs(f['posterior_normalization_ratio']-1))<1e-6
            score_error=float(abs(f['nll']-score['conditional_one_step_bin_nll']));assert score_error<1e-3
            t=record['t'][1:4];mean=nominal[:3,k];y=observed[:3,k]
            C=S*np.exp(-abs(t[:,None]-t[None,:])/noise['ell_s'])+R*np.eye(3)
            density=multivariate_normal(mean=mean,cov=C);lo,hi=bins(y,phase)
            mass,integration_error=nquad(lambda a,b,c:density.pdf([a,b,c]),list(zip(lo,hi)),opts={'epsabs':1e-11,'epsrel':1e-8})
            prefix=filter_ou(t,mean,y,sd,noise['ell_s'],R/(S+R),phase,512,16.)['nll']
            rectangle_error=float(abs(-np.log(mass)-prefix));assert rectangle_error<3e-5
            item={'channel':k+1,'independent_code_expectation_max_error_C':expectation_error,
                'discrete_quantile_CDF_bracket_error':max(brackets),'nodes':nodes,
                'conditional_bin_nll_refinement_error':score_error,'independent_three_bin_nll_error':rectangle_error,
                'rectangle_integrator_error':integration_error}
            if draws is not None:
                witness=arrays['parameter_component_witness_indices_'+str(j)];assert np.issubdtype(witness.dtype,np.integer)
                expected=np.unique(np.r_[np.linspace(0,n-1,9).astype(int),
                    [int(np.argmin(abs(record['t'][1:]-t))) for t in [60.,180.,450.]]])
                assert np.array_equal(witness,expected),'Witness times must follow the declared input/time design'
                low=arrays['parameter_component_witness_low_'+str(j)];high=arrays['parameter_component_witness_high_'+str(j)]
                sds=arrays['parameter_component_SD_'+str(j)];assert low.shape==high.shape==(len(draws),len(witness),2)
                names=fit['coordinate_names'];expected_sd=np.hypot(np.exp(draws[:,names.index('log_latent_SD_T'+str(k+1))]),np.exp(draws[:,names.index('log_measurement_SD_T'+str(k+1))]))
                assert np.max(abs(sds[:,k]-expected_sd))<1e-12 and np.all(low<=high)
                errors=[]
                for suffix,p,means in [('low',.025,low),('high',.975,high)]:
                    errors.append(quantile_check(arrays['parameter_mixture_reported_'+suffix+'_'+str(j)][witness,k],means[:,:,k],sds[:,k],p,phase,True))
                item.update({'parameter_mixture_witness_times':len(witness),'parameter_mixture_CDF_bracket_error':max(errors)})
            channel.append(item)
        # Independently check physical component containment at every witness
        # for the first, middle and last parameter draw, not merely the mean fit.
        component_checks=[]
        if draws is not None:
            for q in sorted(set([0,len(draws)//2,len(draws)-1])):
                point=draws[q];names=fit['coordinate_names'];kk=float(np.exp(point[names.index('log_K')]))
                parameters={'K_common_nominal':kk,'tA_nominal_s':np.exp(point[0]),'rS_s':np.exp(point[1]),
                    'Us_over_Ua_nominal':np.exp(point[2]),'Ux_over_Ua_nominal':np.exp(point[3]),
                    'input_lag_s':np.exp(point[4]),'command_curvature':point[5]}
                error=0.
                for g in sorted(set([1-fit['band'],1.,1+fit['band']])):
                    for ambient in [-2.,2.]:
                        values=direct(record,parameters,kk,g,ambient)[witness]
                        error=max(error,float(np.max(low[q]-values)),float(np.max(values-high[q])))
                assert error<2e-8
                component_checks.append({'draw_index':q,'independent_dimensional_witness_containment_error_C':max(0.,error)})
        rows.append({'name':record['name'],'nobs_per_channel':n,'recomputed_metrics':metric,
            'independent_nuisance_endpoint_containment_error_C':containment,
            'channel_checks':channel,'selected_component_physics':component_checks})
    return {'status':'passed','fit_sha256':sha(fitpath),'training_fit_audit_sha256':sha(args.fit_audit),
        'prediction_sha256':sha(args.input/'transfer_predictions.json'),'arrays_sha256':sha(args.input/'transfer_predictions.npz'),
        'parameter_draw_receipt':draw_receipt,'per_run':rows,
        'scope':'All nominal dimensional trajectories, code expectations, quantile brackets, metrics and refined conditional scores; independent prefix Gaussian rectangles and nuisance-endpoint containment. Parameter-mixture CDF checks use all components at declared sparse times, with selected-draw physical containment; not all component trajectories are independently recomputed. No predictive-coverage calibration, population nuisance law or untouched-validation claim.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--fit-audit',type=Path,required=True);args=parser.parse_args()
    try:result=audit(args)
    except Exception as exc:
        write(args.output/'audit.json',{'status':'failed','exception':repr(exc)});raise
    write(args.output/'audit.json',result)


if __name__=='__main__':main()
