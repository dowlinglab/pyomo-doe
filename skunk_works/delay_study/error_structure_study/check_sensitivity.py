"""Frechet block-exponential and independently propagated derivative checks."""
import argparse
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.linalg import expm,expm_frechet
from physical import load_records
from joint_quantized import Model
from sensitivity import generator,mean_and_jacobian
from audit_common import dimensional_mean,dimensional_fit
from audit_bounded import independent
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    fit=json.loads((args.input/'fit.json').read_text());records=load_records();rows=[]
    for band,mode in [(fit['band'],'plain'),(0.,'plain'),(fit['band'],'startup'),(fit['band'],'sensor')]:
        m=Model(records,band,startup_band=2. if mode=='startup' else 0.,initial_sensor_bins=mode=='sensor')
        original=np.array(fit['best']['point']);oldshape=6+len(records)
        p=np.r_[original[:oldshape] if band else original[:6],original[oldshape:oldshape+1+len(records)],
            np.zeros(2*len(records)) if mode!='plain' else [],original[-6:]]
        if mode!='plain':p[m.nshape+1+m.nr:m.nmean]=np.linspace(-.03,.03,2*m.nr)
        start=time.monotonic();mean,J=mean_and_jacobian(m,p);elapsed=time.monotonic()-start
        expected=m.mean(p,False);error=float(np.max(abs(mean-expected)));assert error<1e-9,(mode,error)
        direction=np.random.RandomState(2941).normal(size=m.nmean);direction/=np.linalg.norm(direction)
        for coordinate in [2,5,m.nshape, None]:
            v=np.zeros(len(p));v[:m.nmean]=direction if coordinate is None else np.eye(m.nmean)[coordinate]
            h=5e-4;fd=(m.mean(p+h*v,False)-m.mean(p-h*v,False))/(2*h)
            rel=float(np.max(abs(fd-J@v[:m.nmean])/(1+abs(fd))));assert rel<2e-6,(band,mode,coordinate,rel)
            rows.append({'band':band,'mode':mode,'coordinate':coordinate,'mean_error_C':error,'scaled_derivative_error':rel,'elapsed_s':elapsed})
    A,D=generator(original,.03,original[oldshape]);block_errors=[]
    for v in D:
        block=np.block([[A,v],[np.zeros_like(A),A]])
        error=float(np.max(abs(expm(block)[:7,7:]-expm_frechet(A,v,compute_expm=False))));assert error<1e-12
        block_errors.append(error)
    write(args.output/'check.json',{'status':'passed','checks':rows,'Frechet_block_exponential_max_error':max(block_errors),
        'scope':'Full raw trajectories, directional finite differences in ordinary/zero-band/hidden-heater/sensor-bin charts, and independent block-exponential identity. Objective and score checks are still required before numerical fit use.'})
    print(json.dumps({'status':'passed','max_mean_error_C':max(v['mean_error_C'] for v in rows),'max_scaled_derivative_error':max(v['scaled_derivative_error'] for v in rows)}),flush=True)


if __name__=='__main__':main()
