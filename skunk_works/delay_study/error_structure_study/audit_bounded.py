"""Independent dimensional energy matrix and dense two-channel likelihood."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.linalg import expm,cholesky,solve_triangular
from physical import load_records
from run_mean import write_json


def independent(d,fit):
    p=fit['parameters'];ua0=.04;us=ua0*p['Us_over_Ua_nominal'];ux=ua0*p['Ux_over_Ua_nominal'];ch=ua0*p['tA_nominal_s'];cs=us*p['rS_s']
    j=fit['record_names'].index(d['name']);ua=ua0*p['Ua_run_over_nominal'][j]
    m=5 if fit['lag'] else 4;A=np.zeros((m+3,m+3));u,a,one=m,m+1,m+2
    # Construct by signed heat fluxes, independently of the normalized model.
    for h,s,other in [(0,1,2),(2,3,0)]:
        A[h,h]=-(ua+us+ux)/ch;A[h,s]=us/ch;A[h,other]=ux/ch;A[h,a]=ua/ch
        A[s,h]=us/cs;A[s,s]=-us/cs
    beta=ua0*p['K_common_nominal']
    if fit['lag']:
        A[0,4]=beta/ch;A[4,4]=-1/p['input_lag_s'];A[4,u]=1/p['input_lag_s']
    else:A[0,u]=beta/ch
    j=fit['record_names'].index(d['name']);width=1 if fit['ambient']=='constant' else 2
    offsets=np.asarray(fit['coefficients'][1+width*j:1+width*(j+1)])
    A[a,one]=(offsets[-1]-offsets[0])/d['t'][-1]
    state=np.zeros(m+3);state[2:4]=d['y'][0,1]-d['y'][0,0];state[a]=offsets[0];state[one]=1
    if fit.get('startup_offsets_C'):
        state[0]+=fit['startup_offsets_C'][j][0];state[2]+=fit['startup_offsets_C'][j][1]
    result=[state[[1,3]]+d['y'][0,0]];cache={}
    for k,step in enumerate(np.diff(d['t'])):
        if float(step) not in cache:cache[float(step)]=expm(A*step)
        command=d['u'][k];state[u]=command+p['command_curvature']*command*(1-command/100)
        state=cache[float(step)]@state;result.append(state[[1,3]]+d['y'][0,0])
    return np.array(result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    f=json.loads((args.input/'fit.json').read_text());arrays=np.load(args.input/'trajectories.npz');rows=[];nll=0.;nobs=0
    x=f['best']['point'];nt=f['shape_count']
    deviations=np.asarray(f['parameters']['log_deviations']);ratios=np.asarray(f['parameters']['Ua_run_over_nominal'])
    assert abs(sum(deviations))<1e-7 and np.max(abs(np.exp(deviations)-ratios))<1e-12
    assert np.all(ratios>=1-f['band']-1e-7) and np.all(ratios<=1+f['band']+1e-7)
    if f.get('startup_offsets_C'):assert np.max(abs(np.asarray(f['startup_offsets_C'])))<=f['startup_band_C']+1e-7
    for j,d in enumerate(load_records()):
        assert d['sha256']==f['raw_sha256'][d['name']]
        pred=independent(d,f);error=float(np.max(abs(pred-arrays['predicted_'+str(j)])))
        if error>2e-7:raise AssertionError(('Physical mean',d['name'],error))
        innovation_error=[]
        for k in range(2):
            t=d['t'][1:];residual=d['y'][1:,k]-pred[1:,k];sigma=f['sigma1_C'] if k==0 else f['sigma2_C']
            if f['kernel']=='iid':C=np.eye(len(t))
            else:
                ell,eta=np.exp(x[nt+2*k]),x[nt+2*k+1]
                C=(1-eta)*np.exp(-abs(t[:,None]-t[None,:])/ell)+eta*np.eye(len(t))
            L=cholesky(C,lower=True);z=solve_triangular(L,residual,lower=True)/sigma
            ie=float(np.max(abs(z-arrays['innovation_'+str(j)][:,k])));assert ie<2e-5
            innovation_error.append(ie);nll+=.5*(z@z+2*np.log(np.diag(L)).sum()+2*len(t)*np.log(sigma)+len(t)*np.log(2*np.pi));nobs+=len(t)
        rows.append({'name':d['name'],'mean_max_abs_C':error,'innovation_max_abs_by_channel':innovation_error})
    assert nobs==f['nobs'];error=abs(nll-f['best']['nll']);assert error<3e-5
    write_json(args.output/'audit.json',{'status':'passed','fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
        'nll_abs_error':error,'centering_abs_error':float(abs(sum(deviations))),'band':f['band'],'per_run':rows,'scope':'Dimensional energy-flux matrix, both outputs, raw bindings, dense innovation and likelihood arithmetic. No symmetry, initialization, global optimum or scientific adequacy guarantee.'})
    print(json.dumps({'status':'passed','nll_abs_error':error,'max_mean_error':max(r['mean_max_abs_C'] for r in rows)}),flush=True)


if __name__=='__main__':main()
