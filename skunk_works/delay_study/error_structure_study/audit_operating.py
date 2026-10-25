"""Dense Gaussian conditional checks for forecast formulas and saved RMSE tables."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from operating import forecast


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--fits',type=Path,nargs='+',required=True)
    args=parser.parse_args();rng=np.random.RandomState(51903);t=np.cumsum(rng.uniform(.5,2.,30));e=rng.normal(size=30)
    errors=[]
    for eta in [0.,.1,.8,1.]:
        sigma=.6;ell=7.;slow=sigma*sigma*(1-eta)
        one,z,post,P=forecast(t,e,sigma,ell,eta)
        for origin in [0,4,14]:
            for target in [origin+1,origin+3,origin+7]:
                history=t[:origin+1];C=slow*np.exp(-abs(history[:,None]-history[None,:])/ell)+sigma*sigma*eta*np.eye(len(history))
                cross=slow*np.exp(-(t[target]-history)/ell)
                expected_mean=float(cross@np.linalg.solve(C,e[:origin+1]))
                expected_var=float(sigma*sigma-cross@np.linalg.solve(C,cross))
                decay=np.exp(-(t[target]-t[origin])/ell)
                actual_mean=decay*post[origin];actual_var=sigma*sigma*eta+slow*(1-decay*decay)+decay*decay*P[origin]
                error=max(abs(actual_mean-expected_mean),abs(actual_var-expected_var));errors.append(error);assert error<1e-10,error
    frame=pd.read_csv(args.input/'run_levels.csv');checked=0;largest=0.
    for path in args.fits:
        fit=json.loads((path/'fit.json').read_text());a=np.load(path/'trajectories.npz');rows=frame[frame.model==path.name]
        for _,row in rows.iterrows():
            j=fit['record_names'].index(row['name']);k=int(row['channel'])-1
            t=a['time_'+str(j)][1:];keep=t>=row['cutoff_s'];e=(a['observed_'+str(j)]-a['predicted_'+str(j)])[1:,k]
            value=np.linalg.norm(e[keep])/np.sqrt(keep.sum());error=abs(value-row['free_simulation_RMSE_C']);largest=max(largest,error);assert error<1e-10;checked+=1
    result={'status':'passed','synthetic_conditional_checks':len(errors),'maximum_dense_conditional_error':max(errors),
        'saved_RMSE_cells_checked':checked,'maximum_RMSE_error_C':largest,
        'input_sha256':{name:hashlib.sha256((args.input/name).read_bytes()).hexdigest() for name in ['run_levels.csv','forecast_horizons.csv','receipt.json']},
        'scope':'Independent dense Gaussian conditioning for forecast formulas, saved free-simulation RMSE arithmetic. Calibration of noise-only prediction intervals and empirical grouping causality are not verified.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
