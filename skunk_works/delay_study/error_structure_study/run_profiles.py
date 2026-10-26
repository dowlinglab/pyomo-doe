"""Likelihood profiles with all remaining mean and covariance nuisances refit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from joint_quantized import Model
from fit_task import run
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--noise1',type=Path,required=True);parser.add_argument('--noise2',type=Path,required=True)
    parser.add_argument('--coordinate',required=True);parser.add_argument('--values',type=float,nargs='+',required=True)
    parser.add_argument('--physical-values',action='store_true');parser.add_argument('--starts',type=int,default=2)
    parser.add_argument('--maxiter',type=int,default=300);args=parser.parse_args();begun=time.monotonic()
    fit=json.loads((args.input/'fit.json').read_text());assert fit['kernel']=='quantized_ou_nugget' and not fit.get('fixed_coordinates')
    if fit.get('dataset') or fit.get('startup_band_C') or fit.get('initial_sensor_bins'):raise ValueError('Declare separate synthetic/startup profile setup')
    lookup={d['name']:d for d in load_records()};records=[lookup[n] for n in fit['record_names']]
    model=Model(records,fit['band'],fit['phase'],128,fit['family'],minimum_sensor_loading=fit.get('minimum_sensor_loading',.005),condition_initial_code=fit.get('condition_initial_code',False));names=model.coordinate_names();index=names.index(args.coordinate)
    point=np.array(fit['best']['point']);values=np.array(args.values,dtype=float)
    if args.physical_values:
        if not args.coordinate.startswith('log_'):raise ValueError('Physical-value conversion only for log coordinates')
        if np.any(values<=0):raise ValueError('Positive physical values required')
        values=np.log(values)
    lo,hi=model.bounds()[index]
    if np.any(values<lo-1e-12) or np.any(values>hi+1e-12):raise ValueError('Profile value outside committed bounds')
    # Visit outward from the selected point and retain all failed/disconnected
    # points. Nearest successful profile warms the next point; a second basin
    # swaps sensor and input times. The unconstrained reference is always saved.
    values=np.unique(np.r_[np.clip(values,lo,hi),point[index]])
    ordered=sorted(values,key=lambda x:abs(x-point[index]));rows=[];successful=[]
    for j,value in enumerate(ordered):
        warm=args.input/'fit.json'
        if successful:warm=Path(min(successful,key=lambda r:abs(r['coordinate_value']-value))['output'])/'fit.json'
        row=run(args.output/('point_'+str(j)),warm,args.noise1,args.noise2,fit['family'],fit['band'],fit['phase'],
            starts=args.starts,maxiter=args.maxiter,fixed={args.coordinate:float(value)},minimum_sensor_loading=fit.get('minimum_sensor_loading',.005),condition_initial_code=fit.get('condition_initial_code',False))
        row['coordinate_value']=float(value);row['physical_value']=float(np.exp(value)) if args.coordinate.startswith('log_') else float(value)
        if row['success']:
            row['twice_nll_minus_reference']=float(2*(row['nll']-fit['best']['nll']));successful.append(row)
        rows.append(row);write(args.output/'points.json',rows)
        write(args.output/'progress.json',{'points_attempted':len(rows),'points_successful':len(successful),'elapsed_s':time.monotonic()-begun})
    write(args.output/'profile.json',{'status':'complete' if all(r['success'] for r in rows) else 'completed_with_failures',
        'coordinate':args.coordinate,'optimization_coordinate_is_log':args.coordinate.startswith('log_'),
        'reference_nll':fit['best']['nll'],'reference_coordinate':float(point[index]),'profile_bounds':[lo,hi],
        'fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),'points':sorted(rows,key=lambda r:r['coordinate_value']),
        'lower_nll_than_reference':any(r['success'] and r['nll']<fit['best']['nll']-1e-3 for r in rows),
        'elapsed_s':time.monotonic()-begun,
        'notice':'Every fixed coordinate refits all other permitted mean, ambient, conductance and noise coordinates. Two declared starts and nearest successful warm points do not prove global profiles. Negative deviances require baseline refinement. Active-bound and disconnected profiles preclude routine Wald claims; chi-squared cutoffs require simulation calibration.'})
    if not successful:raise RuntimeError('No profile point converged; retain failures')


if __name__=='__main__':main()
