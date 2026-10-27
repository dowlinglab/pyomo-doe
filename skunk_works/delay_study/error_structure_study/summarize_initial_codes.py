"""Compare audited initialization sensitivities in the reported-code space."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from continuation import parent_receipt
from transfer import reported_mean
from quantized import STEP,OFFSET
from run_bounded import write


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_expected_code(mean,sd,phase):
    """Separate erfc bin integration with scalar compensated accumulation."""
    low=math.floor((mean-12*sd-OFFSET)/STEP)-3
    high=math.ceil((mean+12*sd-OFFSET)/STEP)+3
    masses=[];moments=[]
    for code in range(low,high+1):
        a=(STEP*(code+phase)+OFFSET-mean)/(sd*math.sqrt(2))
        b=a+STEP/(sd*math.sqrt(2))
        mass=(math.erfc(a)-math.erfc(b))/2 if a>=0 else (math.erfc(-b)-math.erfc(-a))/2
        masses.append(mass);moments.append(mass*round(STEP*code+OFFSET,2))
    assert abs(math.fsum(masses)-1)<1e-12
    return math.fsum(moments)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);args=p.parse_args();root=args.results
    cases={
        'fixed_nearest':('joint_conductance_b10_export_v2','joint_conductance_b10_audit'),
        'bins_nearest':('joint_b10_initial_sensor_bins','joint_b10_initial_sensor_bins_audit'),
        **{name:('initial_remaining_b10/'+name,'initial_remaining_b10/'+name+'_audit') for name in
            ['fixed_floor','bins_floor','bins_buffer_nearest','bins_buffer_floor','hidden_heater_2C','hidden_heater_5C']}}
    records=load_records();rows=[];saved={};bindings={};predictions={};means={};fits={};maxcheck=0.
    for name,(directory,auditor) in cases.items():
        directory=root/directory;auditor=root/auditor
        assert parent_receipt(directory,require_terminal=True)['exit']['exit_code']==0
        assert parent_receipt(auditor,require_terminal=True)['exit']['exit_code']==0
        fit=json.loads((directory/'fit.json').read_text());audit=json.loads((auditor/'audit.json').read_text())
        assert audit['status']=='passed' and audit['fit_sha256']==sha(directory/'fit.json')
        assert audit['arrays_sha256']==sha(directory/'trajectories.npz')
        assert fit['record_names']==[d['name'] for d in records]
        assert all(fit['raw_sha256'][d['name']]==d['sha256'] for d in records)
        bindings[name]={str(q):sha(q) for q in [directory/'fit.json',directory/'trajectories.npz',auditor/'audit.json']}
        arrays=np.load(directory/'trajectories.npz');mu=np.concatenate([arrays['predicted_'+str(j)][1:] for j in range(12)])
        observed=np.concatenate([d['y'][1:] for d in records]);assert observed.shape==(10795,2)
        assert all(np.array_equal(arrays['observed_'+str(j)],d['y']) for j,d in enumerate(records))
        assert all(np.array_equal(arrays['time_'+str(j)],d['t']) for j,d in enumerate(records))
        code=np.zeros_like(mu);metrics=[]
        for k,noise in enumerate(fit['noise_parameters']):
            sd=math.hypot(noise['latent_SD_C'],noise['measurement_SD_C'])
            code[:,k]=reported_mean(mu[:,k],sd,fit['phase'])
            for index in np.linspace(0,len(mu)-1,25).astype(int):
                independent=scalar_expected_code(float(mu[index,k]),sd,fit['phase'])
                maxcheck=max(maxcheck,abs(independent-code[index,k]));assert maxcheck<1e-10
            residual=observed[:,k]-code[:,k];bias=math.fsum(residual)/len(residual)
            rmse=math.sqrt(math.fsum(float(v)*float(v) for v in residual)/len(residual))
            latent=math.sqrt(math.fsum(float(v)*float(v) for v in observed[:,k]-mu[:,k])/len(residual))
            assert abs(latent-fit['RMSE_by_channel_C'][k])<1e-10
            metrics.append({'channel':k+1,'count':len(residual),'reported_code_RMSE_C':rmse,
                'reported_code_signed_error_C':bias,'reported_code_centered_RMSE_C':math.sqrt(max(0.,rmse**2-bias**2)),
                'latent_reference_RMSE_C':latent})
        rows.append({'case':name,'phase':fit['phase'],'nll':fit['best']['nll'],'metrics':metrics,
            'K':fit['parameters']['K_common_nominal'],'sensor_time_s':fit['parameters']['rS_s'],
            'input_time_s':fit['parameters']['input_lag_s']})
        predictions[name]=code;means[name]=mu;fits[name]=fit;saved[name+'_code_mean']=code
    comparisons=[]
    for near,floor in [('bins_nearest','bins_floor'),('bins_buffer_nearest','bins_buffer_floor')]:
        a,b=fits[near],fits[floor];errors=[]
        for k,noise in enumerate(a['noise_parameters']):
            sd=math.hypot(noise['latent_SD_C'],noise['measurement_SD_C'])
            shifted=reported_mean(means[near][:,k]+STEP/2,sd,0.)
            error=float(np.max(abs(shifted-predictions[near][:,k])));assert error<1e-10;errors.append(error)
        comparisons.append({'nearest':near,'floor':floor,'nll_difference':b['best']['nll']-a['best']['nll'],
            'actual_latent_shift_minus_half_grid_max_abs_C':np.max(abs(means[floor]-means[near]-STEP/2),axis=0).tolist(),
            'actual_reported_code_mean_max_difference_C':np.max(abs(predictions[floor]-predictions[near]),axis=0).tolist(),
            'exact_translated_nearest_code_mean_max_difference_C':errors})
    np.savez_compressed(args.output/'code_means.npz',**saved)
    write(args.output/'summary.json',{'status':'complete','cases':rows,'phase_comparisons':comparisons,
        'independent_scalar_code_expectation_max_error_C':maxcheck,'independent_scalar_witnesses':400,
        'source_bindings_sha256':bindings,'code_arrays_sha256':sha(args.output/'code_means.npz'),
        'scope':'All cases already passed independent selected-point numerical audits. Marginal reported-code expectations include stationary OU plus nugget variance, not filtered feedback or parameter uncertainty. Full scored data are in-sample. Different raw code-minus-analogue-reference RMSE under a bin-phase translation is not different physical predictive quality. This arithmetic report checks 400 independent erfc witnesses and every raw observed/time array; it does not refit or establish whitening.'})


if __name__=='__main__':main()
