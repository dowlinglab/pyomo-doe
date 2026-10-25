"""Operating-point residuals, common/differential errors and forecast horizons."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def correlation(a,b):
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else None


def forecast(t,e,sigma,ell,eta):
    slow=sigma*sigma*(1-eta);v=sigma*sigma*eta;P=slow;m=0.;z=[];one=[];post=[];variance=[]
    for j in range(len(t)):
        if j:
            a=np.exp(-(t[j]-t[j-1])/ell);m=a*m;P=a*a*P+slow*(-np.expm1(-2*(t[j]-t[j-1])/ell))
        F=P+v;innovation=e[j]-m;one.append(innovation);z.append(innovation/np.sqrt(F))
        m+=P/F*innovation;P=P*v/F;post.append(m);variance.append(P)
    return np.asarray(one),np.asarray(z),np.asarray(post),np.asarray(variance)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fits',type=Path,nargs='+',required=True);args=parser.parse_args()
    grouped=[];common=[];horizons=[];runlevels=[];bindings={};largest=0.
    for path in args.fits:
        f=json.loads((path/'fit.json').read_text());a=np.load(path/'trajectories.npz');label=path.name
        bindings[label]={n:hashlib.sha256((path/n).read_bytes()).hexdigest() for n in ['fit.json','trajectories.npz']}
        for j,name in enumerate(f['record_names']):
            t=a['time_'+str(j)][1:];y=a['observed_'+str(j)][1:];p=a['predicted_'+str(j)][1:]
            u=a['input_'+str(j)][1:];e=y-p;z=a['innovation_'+str(j)]
            for cutoff in [0,450]:
                keep=t>=cutoff;ee=e[keep];zz=z[keep]
                common.append({'model':label,'name':name,'cutoff_s':cutoff,'nobs':int(keep.sum()),
                    'residual_channel_correlation':correlation(ee[:,0],ee[:,1]),
                    'innovation_channel_correlation':correlation(zz[:,0],zz[:,1]),
                    'common_residual_RMS_C':float(np.sqrt(np.mean(((ee[:,0]+ee[:,1])/2)**2))),
                    'differential_residual_RMS_C':float(np.sqrt(np.mean(((ee[:,0]-ee[:,1])/2)**2))),
                    'standardized_common_variance':float(np.var((zz[:,0]+zz[:,1])/np.sqrt(2))),
                    'standardized_difference_variance':float(np.var((zz[:,0]-zz[:,1])/np.sqrt(2)))})
            for channel in [0,1]:
                ns=f['shape_count'];ell=float(np.exp(f['best']['point'][ns+2*channel]));eta=f['best']['point'][ns+2*channel+1]
                sigma=f['sigma1_C'] if channel==0 else f['sigma2_C']
                one,computed,posterior,variance=forecast(t,e[:,channel],sigma,ell,eta)
                discrepancy=float(np.max(abs(computed-z[:,channel])));largest=max(largest,discrepancy)
                assert discrepancy<1e-7,discrepancy
                for cutoff in [0,450]:
                    keep=t>=cutoff
                    runlevels.append({'model':label,'name':name,'channel':channel+1,'cutoff_s':cutoff,
                        'free_simulation_RMSE_C':float(np.sqrt(np.mean(e[keep,channel]**2))),
                        'one_step_RMSE_C':float(np.sqrt(np.mean(one[keep]**2))),
                        'one_step_error_mean_C':float(one[keep].mean()),
                        'observed_T1_mean_C':float(y[keep,0].mean()),'observed_T1_max_C':float(y[keep,0].max()),
                        'command_mean_percent':float(u[keep].mean()),
                        'Ua_run_over_nominal':f.get('parameters',{}).get('Ua_run_over_nominal',[1.]*len(f['record_names']))[j]})
                    for h in [1.,10.,30.,60.]:
                        source=np.flatnonzero(keep);target=np.searchsorted(t,t[source]+h);valid=target<len(t)
                        source=source[valid];target=target[valid];dt=t[target]-t[source];decay=np.exp(-dt/ell)
                        err=e[target,channel]-decay*posterior[source]
                        predvar=sigma*sigma*eta+sigma*sigma*(1-eta)*(1-decay*decay)+decay*decay*variance[source]
                        horizons.append({'model':label,'name':name,'channel':channel+1,'cutoff_s':cutoff,'nominal_horizon_s':h,
                            'nobs':len(err),'actual_horizon_mean_s':float(dt.mean()),'SSE_C2':float(err@err),
                            'RMSE_C':float(np.sqrt(np.mean(err**2))),'mean_error_C':float(err.mean()),
                            'noise_only_95_coverage':float(np.mean(abs(err)<=1.96*np.sqrt(predvar)))})
                for smoothing in [10.,20.]:
                    slope=(np.interp(t+smoothing/2,t,y[:,0])-np.interp(t-smoothing/2,t,y[:,0]))/smoothing
                    direction=np.where(slope>.002,'heating',np.where(slope<-.002,'cooling','flat'))
                    for cutoff in [0,450]:
                        valid=(t>=cutoff)&(t>=t[0]+smoothing/2)&(t<=t[-1]-smoothing/2)
                        for low in np.arange(15.,80.,5.):
                            for lowq in np.arange(0.,100.,20.):
                                for sign in ['heating','cooling','flat']:
                                    take=valid&(y[:,0]>=low)&(y[:,0]<low+5)&(u>=lowq)&((u<lowq+20) if lowq<80 else (u<=100))&(direction==sign)
                                    n=int(take.sum())
                                    if n<8:continue
                                    residual=e[take,channel]
                                    grouped.append({'model':label,'name':name,'channel':channel+1,'cutoff_s':cutoff,
                                        'slope_window_s':smoothing,'direction':sign,'T1_bin_low_C':low,'command_bin_low_percent':lowq,
                                        'nobs':n,'mean_residual_C':float(residual.mean()),'RMSE_C':float(np.sqrt(np.mean(residual**2))),
                                        'innovation_mean':float(z[take,channel].mean())})
    pd.DataFrame(grouped).to_csv(args.output/'operating_bins.csv',index=False)
    pd.DataFrame(common).to_csv(args.output/'common_differential.csv',index=False)
    pd.DataFrame(horizons).to_csv(args.output/'forecast_horizons.csv',index=False)
    pd.DataFrame(runlevels).to_csv(args.output/'run_levels.csv',index=False)
    receipt={'status':'complete','source_bindings':bindings,'scalar_Kalman_vs_saved_innovation_max_error':largest,
        'grouping':'Temperature and command bins are descriptive; slope is measured T1 centered secant over 10 or 20 s with 0.002 C/s deadband. Data are reused; small bin counts and confounding preclude causal claims.',
        'prediction':'Mean parameters use whole-data fits. Conditional forecasts use only earlier observations within each run for the OU state. Noise-only ranges exclude parameter, quantization and model-form uncertainty.'}
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':'complete','operating_bins':len(grouped),'scalar_Kalman_error':largest}),flush=True)


if __name__=='__main__':main()
