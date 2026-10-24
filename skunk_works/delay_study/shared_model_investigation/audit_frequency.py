"""Reconstruct all empirical coefficients from raw data with QR and quadrature."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from scipy.integrate import quad
from scipy.linalg import qr, solve_triangular
from data import load
from run_mean import write_json


def quadrature_input(t,u,w,start,end):
    f = (lambda s:1., lambda s:np.sin(w*s), lambda s:np.cos(w*s))
    G = np.array([[quad(lambda s:f[i](s)*f[j](s),start,end,epsabs=1e-9)[0]
                   for j in range(3)] for i in range(3)])
    rhs = np.zeros(3)
    for k in range(len(t)-1):
        a,b = max(start,t[k]),min(end,t[k+1])
        if b>a:
            for i in range(3):
                rhs[i] += u[k]*quad(f[i],a,b,epsabs=1e-10)[0]
    return np.linalg.solve(G,rhs)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=json.loads((args.input/'frequency_estimates.json').read_text())
    fit=json.loads((args.baseline/'fit.json').read_text())
    arrays=np.load(args.baseline/'trajectories.npz')
    cache={}
    checked=[]
    for row in source['rows']:
        d=load(row['name'])
        j=fit['record_names'].index(d.name)
        if not np.array_equal(d.temperature,arrays['observed_'+str(j)]) or not np.array_equal(d.input,arrays['input_'+str(j)]):
            raise AssertionError('Raw data do not match saved source arrays')
        t=d.time-d.time[0]; start=row['cutoff_s']; keep=t>=start; tr=t[keep]
        w=2*np.pi/row['period_s']
        key=(d.name,start)
        if key not in cache:
            cache[key]=quadrature_input(t,d.input,w,start,t[-1])
        inp=complex(*cache[key][1:])
        X=np.column_stack((np.ones(len(tr)),np.sin(w*tr),np.cos(w*tr)))
        if row['nuisance']=='trend':
            X=np.column_stack((X,(tr-tr.mean())/(tr[-1]-tr[0])))
        else:
            X=np.column_stack((X,np.exp(-(tr-tr[0])/fit['best']['tau']),
                               np.exp(-(tr-tr[0])/fit['best']['theta'])))
        Q,R=qr(X,mode='economic')
        errors={}
        for kind,y in [('measured',d.temperature),('matched_sampled_n1',arrays['predicted_'+str(j)])]:
            b=solve_triangular(R,Q.T@y[keep])
            H=complex(b[1],b[2])/inp
            expected=complex(row[kind]['H_real'],row[kind]['H_imag'])
            errors[kind]=float(abs(H-expected))
            if errors[kind]>1e-9:
                raise AssertionError((row['name'],row['cutoff_s'],row['nuisance'],kind,errors[kind]))
        checked.append({'name':d.name,'cutoff_s':start,'nuisance':row['nuisance'],'complex_gain_abs_errors':errors})
    result={'status':'passed','cells':len(checked),'independent_input_quadratures':len(cache),
            'maximum_complex_gain_abs_error':max(max(r['complex_gain_abs_errors'].values()) for r in checked),
            'input_sha256':hashlib.sha256((args.input/'frequency_estimates.json').read_bytes()).hexdigest(),
            'checks':checked,'scope':'All54 empirical/matched harmonic estimates, raw arrays, independent continuous held-input quadrature and QR solver. Statistical adequacy/coverage not certified.'}
    write_json(args.output/'audit.json',result)
    print(json.dumps({k:result[k] for k in ('status','cells','maximum_complex_gain_abs_error')}),flush=True)


if __name__=='__main__':
    main()
