"""Independent QR partitions, code expectations and held-input response checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
import pandas as pd
from scipy.linalg import qr,solve_triangular
from scipy.integrate import quad
from scipy.special import ndtr
from physical import load_records
from run_frequency import period_of
from transfer import reported_mean
from quantized import STEP,OFFSET
from run_bounded import write


def held_projection(d,period,start,end):
    w=2*np.pi/period;functions=[lambda x:1.,lambda x:np.sin(w*x),lambda x:np.cos(w*x)]
    G=np.array([[quad(lambda x:f(x)*g(x),start,end,epsabs=1e-10,limit=250)[0] for g in functions] for f in functions])
    rhs=np.zeros(3)
    for j in range(len(d['t'])-1):
        a=max(start,d['t'][j]);b=min(end,d['t'][j+1])
        if b<=a:continue
        for k,f in enumerate(functions):rhs[k]+=d['u'][j]*quad(f,a,b,epsabs=1e-11)[0]
    return np.linalg.solve(G,rhs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    assert json.loads((args.input/'exit.json').read_text())['exit_code']==0
    receipt=json.loads((args.input/'receipt.json').read_text());table=pd.read_csv(args.input/'decomposition.csv')
    frequency=json.loads((args.input/'frequency.json').read_text());lookup={d['name']:d for d in load_records()}
    checked=0;partition_error=0.;expectation_error=0.;response_errors=[]
    cache={}
    for label,binding in receipt['source_bindings'].items():
        # Result folders are external inputs; committed source is frozen into
        # a temporary checkout that intentionally contains no result tree.
        path=args.input.parent/label
        for filename,key in [('fit.json','fit_sha256'),('trajectories.npz','arrays_sha256')]:assert hashlib.sha256((path/filename).read_bytes()).hexdigest()==binding[key]
        fit=json.loads((path/'fit.json').read_text());arrays=np.load(path/'trajectories.npz');cache[label]=(fit,arrays)
        for j,name in enumerate(fit['record_names']):
            d=lookup[name];assert fit['raw_sha256'][name]==d['sha256'];mean=arrays['predicted_'+str(j)];code=np.empty_like(mean)
            for k,p in enumerate(fit['noise_parameters']):
                sd=np.hypot(p['latent_SD_C'],p['measurement_SD_C']);code[:,k]=reported_mean(mean[:,k],sd,fit['phase'])
                for index in np.unique([1,len(mean)//3,2*len(mean)//3,len(mean)-1]):
                    mu=mean[index,k];codes=np.arange(int(np.floor((mu-12*sd-OFFSET)/STEP))-2,int(np.ceil((mu+12*sd-OFFSET)/STEP))+3)
                    lo=STEP*(codes+fit['phase'])+OFFSET;hi=lo+STEP
                    exact=float(np.round(STEP*codes+OFFSET,2)@(ndtr((hi-mu)/sd)-ndtr((lo-mu)/sd)))
                    expectation_error=max(expectation_error,abs(exact-code[index,k]));assert expectation_error<1e-9
            for row in table[(table.model==label)&(table.name==name)].to_dict('records'):
                t=d['t'][1:];keep=t>=row['cutoff_s'];t=t[keep];period=period_of(name)
                pred=mean if row['prediction_kind']=='analogue_physical_mean' else code;k=int(row['channel'])-1
                e=(d['y']-pred)[1:,k][keep];X=np.column_stack([np.ones(len(t)),(t-t.mean())/np.ptp(t)])
                if period:X=np.column_stack([X]+[f(2*np.pi*h*t/period) for h in [1,2,3] for f in [np.sin,np.cos]])
                Q,R=qr(X,mode='economic');assert np.min(abs(np.diag(R)))>1e-8
                c=Q.T@e;parts=[c[0]**2,c[1]**2]+([float(c[a:a+2]@c[a:a+2]) for a in [2,4,6]] if period else [0.,0.,0.])
                rem=e-Q@c;parts.append(float(rem@rem));columns=['offset','additional_trend','additional_fundamental','additional_harmonic2','additional_harmonic3','remainder']
                for key,value in zip(columns,parts):partition_error=max(partition_error,abs(value-row[key+'_SSE_C2']))
                assert abs(row['RMSE_C']-np.sqrt(e@e/len(e)))<1e-10 and int(row['nobs'])==len(e)
                assert abs(row['mean_error_C']-e.mean())<1e-10;checked+=1
    assert checked==len(table) and partition_error<1e-7
    primary=receipt['forecast_fit'];fit,arrays=cache[primary]
    for name in [fit['record_names'][0],fit['record_names'][6]]:
        d=lookup[name];period=period_of(name);cutoff=450.;t=d['t'][1:];keep=t>=cutoff;ts=t[keep]
        inp=held_projection(d,period,max(cutoff,t[0]),d['t'][-1]);u=complex(inp[1],inp[2])
        X=np.column_stack([np.ones(len(ts)),np.sin(2*np.pi*ts/period),np.cos(2*np.pi*ts/period),(ts-ts.mean())/np.ptp(ts)])
        Q,R=qr(X,mode='economic')
        for k in [0,1]:
            beta=solve_triangular(R,Q.T@d['y'][1:,k][keep]);H=complex(beta[1],beta[2])/u
            row=next(r for r in frequency if r['model']==primary and r['name']==name and r['channel']==k+1 and r['cutoff_s']==cutoff)
            target=complex(row['measured']['H_real'],row['measured']['H_imag']);error=abs(H-target);assert error<1e-10
            response_errors.append({'name':name,'channel':k+1,'independent_held_quadrature_QR_complex_gain_error':float(error)})
    aggregate=json.loads((args.input/'aggregate.json').read_text());aggregate_error=0.
    for row in aggregate:
        take=table[(table.model==row['model'])&(table.prediction_kind==row['prediction_kind'])&(table.channel==row['channel'])&(table.cutoff_s==row['cutoff_s'])]
        aggregate_error=max(aggregate_error,abs(row['RMSE_C']-np.sqrt(take.SSE_C2.sum()/take.nobs.sum())))
    assert aggregate_error<1e-12
    write(args.output/'audit.json',{'status':'passed','receipt_sha256':hashlib.sha256((args.input/'receipt.json').read_bytes()).hexdigest(),
        'decomposition_rows':checked,'independent_QR_partition_max_error_C2':partition_error,
        'independent_CDF_code_expectation_max_error_C':expectation_error,'complex_response_checks':response_errors,
        'aggregate_RMSE_error_C':aggregate_error,
        'scope':'All saved decomposition cells via independent QR, selected marginal code expectations via direct CDF sums, actual held-input quadrature and QR frequency coefficients, all pooled RMSEs. Forecast recursion has a separate independent two-observation density/leakage check. No empirical covariance/coverage or physical-model adequacy guarantee.'})
    print(json.dumps({'status':'passed','rows':checked,'partition_error_C2':partition_error,'response_max_error':max(r['independent_held_quadrature_QR_complex_gain_error'] for r in response_errors)}),flush=True)


if __name__=='__main__':main()
