"""Audit selected joint means, bin probabilities, grid support and stationarity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.integrate import nquad
from scipy.stats import multivariate_normal
from physical import load_records
from joint_quantized import Model
from quantized import filter_ou,bins
from audit_bounded import independent
from audit_radiative import direct
from run_bounded import stationary,write


def audit(args):
    fit=json.loads((args.input/'fit.json').read_text());arrays=np.load(args.input/'trajectories.npz')
    if fit.get('dataset'):raise ValueError('Synthetic joint audit requires an explicit dataset-binding extension')
    lookup={d['name']:d for d in load_records()};records=[lookup[n] for n in fit['record_names']]
    assert all(d['sha256']==fit['raw_sha256'][d['name']] for d in records)
    p=np.array(fit['best']['point']);b=fit['band'];phase=fit['phase'];family=fit['family']
    model=Model(records,b,phase,128,family,fit.get('startup_band_C',0.))
    assert model.nmean==fit['nmean'] and len(p)==len(model.bounds())
    assert np.all(p>=np.array(model.bounds())[:,0]-1e-7) and np.all(p<=np.array(model.bounds())[:,1]+1e-7)
    ds=np.array(fit['parameters']['log_deviations']);g=np.exp(ds)
    assert abs(ds.sum())<1e-7 and np.all(g>=1-b-1e-7) and np.all(g<=1+b+1e-7)
    mean=model.mean(p,False);nll_current=model.objective(p)
    stat128=stationary(model,p,model.gradient(p))
    finer=Model(records,b,phase,256,family,fit.get('startup_band_C',0.))
    nll256=finer.objective(p);stat256=stationary(finer,p,finer.gradient(p))
    gradient_difference=float(np.max(abs(model.gradient(p)-finer.gradient(p))))
    legacy=dict(fit,lag=True,curve=True,ambient='constant')
    if fit.get('startup_band_C'):
        legacy['startup_offsets_C']=np.asarray(fit['coefficients'][1+len(records):]).reshape(-1,2).tolist()
    rows=[];total512=0.;total_wide=0.;failures=[];cursor=0
    for j,d in enumerate(records):
        assert np.array_equal(arrays['time_'+str(j)],d['t'])
        assert np.array_equal(arrays['input_'+str(j)],d['u'])
        assert np.array_equal(arrays['observed_'+str(j)],d['y'])
        pred=independent(d,legacy) if family=='conductance' else direct(d,legacy,j)
        saved=arrays['predicted_'+str(j)];mean_error=float(np.max(abs(pred-saved)))
        if mean_error>(2e-7 if family=='conductance' else 3e-6):failures.append(['mean',d['name'],mean_error])
        n=len(d['t'])-1
        for k in [0,1]:
            section=slice(k*model.N+cursor,k*model.N+cursor+n)
            assert np.max(abs(mean[section]-saved[1:,k]))<1e-8
            noise=fit['noise_parameters'][k];S=noise['latent_SD_C']**2;R=noise['measurement_SD_C']**2;ell=noise['ell_s']
            t=d['t'][1:];y=d['y'][1:,k];m=pred[1:,k];sigma=np.sqrt(S+R);eta=R/(S+R)
            fine=filter_ou(t,m,y,sigma,ell,eta,phase,nodes=512,tail_sd=12.)
            wide=filter_ou(t,m,y,sigma,ell,eta,phase,nodes=512,tail_sd=16.)
            total512+=fine['nll'];total_wide+=wide['nll']
            support_error=abs(fine['nll']-wide['nll']);norm=float(np.max(abs(fine['posterior_normalization_ratio']-1)))
            tt=t[:3];C=S*np.exp(-abs(tt[:,None]-tt[None,:])/ell)+R*np.eye(3)
            density=multivariate_normal(mean=m[:3],cov=C);lo,hi=bins(y[:3],phase)
            mass,quadrature_error=nquad(lambda z0,z1,z2:density.pdf([z0,z1,z2]),list(zip(lo,hi)),
                                       opts={'epsabs':1e-11,'epsrel':1e-8})
            reference=-np.log(mass)
            prefix=filter_ou(tt,m[:3],y[:3],sigma,ell,eta,phase,nodes=512,tail_sd=16.)['nll']
            rectangle_error=abs(reference-prefix)
            if support_error>1e-4 or norm>1e-6 or rectangle_error>3e-5:
                failures.append(['bin integration',d['name'],k+1,support_error,norm,rectangle_error])
            rows.append({'name':d['name'],'channel':k+1,'physical_mean_error_C':mean_error,
                'fine_nll':fine['nll'],'support_12_to_16_nll_difference':support_error,
                'posterior_normalization_error':norm,'independent_three_point_nll_error':rectangle_error,
                'rectangle_integrator_error':quadrature_error})
        cursor+=n
    changes={'current128_minus_saved':nll_current-fit['best']['nll'],
             'nodes256_minus128':nll256-nll_current,'nodes512_minus256':total512-nll256,
             'wide16_minus12':total_wide-total512}
    for label,error in changes.items():
        if abs(error)>1e-3:failures.append([label,error])
    if stat256['lagrangian_projected_gradient_max_abs']>=.05:failures.append(['stationarity256',stat256])
    result={'status':'passed' if not failures else 'failed','failures':failures,
        'fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
        'arrays_sha256':hashlib.sha256((args.input/'trajectories.npz').read_bytes()).hexdigest(),
        'nll_current128':nll_current,'nll_current256':nll256,'nll_independent_mean512':total512,
        'nll_changes':changes,'gradient128_to256_max_abs_difference':gradient_difference,
        'stationarity128':stat128,'stationarity256':stat256,'per_run':rows,
        'scope':'Frozen raw/design binding, independent dimensional mean, independent trivariate Gaussian observation-bin integrals, corrected posterior support, full 128/256/512 likelihood and 12/16-tail checks, current 256-node Fisher score stationarity. No global-optimum, noise adequacy or interval-coverage proof.'}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    try:result=audit(args)
    except Exception as exc:
        write(args.output/'audit.json',{'status':'failed','exception':repr(exc)});raise
    write(args.output/'audit.json',result)
    print(json.dumps({k:result[k] for k in ['status','failures','nll_changes']}),flush=True)
    if result['status']!='passed':raise RuntimeError('Selected fit requires numerical refinement; see audit.json')


if __name__=='__main__':main()
