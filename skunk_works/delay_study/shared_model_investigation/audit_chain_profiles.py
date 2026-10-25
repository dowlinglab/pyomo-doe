"""Independently reconstruct every saved pooled profile likelihood point."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from data import load
from audit_pooled import physical_reference
from run_mean import write_json


def grid_support(points,threshold,scale='log'):
    inside=[p['twice_delta_nll']<=threshold for p in points]
    intervals=[];left=None
    def crossing(a,b):
        fraction=(threshold-a['twice_delta_nll'])/(b['twice_delta_nll']-a['twice_delta_nll'])
        return float(a['value']+fraction*(b['value']-a['value'])) if scale=='linear' else float(np.exp(np.log(a['value'])+fraction*np.log(b['value']/a['value'])))
    for j,yes in enumerate(inside):
        if yes and (j==0 or not inside[j-1]):left=None if j==0 else crossing(points[j-1],points[j])
        if yes and (j==len(points)-1 or not inside[j+1]):
            right=None if j==len(points)-1 else crossing(points[j],points[j+1])
            intervals.append({'lower_interpolated':left,'upper_interpolated':right,
                              'lower_open_on_grid':left is None,'upper_open_on_grid':right is None})
    return intervals


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-curves',type=int,default=6)
    args=parser.parse_args();curves=[]
    for path in sorted(args.input.glob('*.json')):
        profile=json.loads(path.read_text())
        if not isinstance(profile,dict) or 'reference_nll' not in profile or 'points' not in profile:continue
        if not profile['status'].startswith('complete'):raise ValueError('Profile not terminal')
        folder=Path(profile['model']);fit=json.loads((folder/'fit.json').read_text())
        records=[load(n) for n in fit['record_names']]
        rows=[]
        for row in profile['points']:
            x=np.array(row['best']['point']);c=np.array(row['coefficients'])
            times=np.exp(x[:2]) if fit['kind']=='transport' else np.r_[np.exp(x[0]),np.full(fit['order'],np.exp(x[1])/fit['order'])]
            physical=np.r_[np.zeros(len(times)-1),c[0]]
            if profile['parameter']=='K' and abs(c[0]-row['value'])>1e-9:raise AssertionError('Fixed gain')
            if profile['parameter']=='theta' and abs(np.exp(x[1])-row['value'])>1e-8:raise AssertionError('Fixed theta')
            if profile['parameter']=='delay' and abs(x[2]-row['value'])>1e-8:raise AssertionError('Fixed additional delay')
            if np.any(abs(c[1:])>2+1e-7):raise AssertionError('Ambient bounds')
            ell,eta=np.exp(x[-2]),x[-1]
            n=0;quadratic=0.;logdet=0.
            for j,d in enumerate(records):
                width=fit.get('ambient_width',1)
                if fit['kind']=='transport':
                    from transport import independent_reference
                    pred=d.temperature[0]+c[0]*independent_reference(d.time,d.input,*times,x[2])
                    elapsed=d.time-d.time[0];a=-np.expm1(-elapsed/times[0])
                    if width==1:pred+=c[1+j]*a
                    else:
                        b=(elapsed-times[0]*a)/(d.time[-1]-d.time[0]);pred+=c[1+2*j]*(a-b)+c[2+2*j]*b
                else:pred=d.temperature[0]+physical_reference(d.time,d.input,times,physical,c[1+j])
                r=d.temperature[1:]-pred[1:];t=d.time[1:]
                C=(1-eta)*np.exp(-abs(t[:,None]-t[None,:])/ell)+eta*np.eye(len(t))
                L=cholesky(C,lower=True);z=solve_triangular(L,r,lower=True)
                n+=len(t);quadratic+=z@z;logdet+=2*np.log(np.diag(L)).sum()
            nll=.5*(n*(np.log(2*np.pi)+1+np.log(quadratic/n))+logdet)
            error=abs(nll-row['best']['nll'])
            if error>3e-5:raise AssertionError((path.name,row['value'],error))
            delta=2*(nll-profile['reference_nll'])
            if abs(delta-row['twice_delta_nll'])>6e-5:raise AssertionError('Profile delta arithmetic')
            rows.append({'value':row['value'],'nll_abs_error':float(error),'twice_delta_nll':float(delta),
                         'stationarity_warning':row['stationarity_warning'],'lower_than_reference':row['lower_than_reference']})
        curve={'id':profile['id'],'parameter':profile['parameter'],'reference_value':profile['reference_value'],
               'profile_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'points':rows,
               'nominal_support':grid_support(rows,profile['nominal_95_threshold'],'linear' if profile['parameter']=='delay' else 'log'),
               'numerical_warnings':sum(p['stationarity_warning'] for p in rows),
               'lower_reference_points':sum(p['lower_than_reference'] for p in rows)}
        curves.append(curve)
        write_json(args.output/(profile['id']+'_audit.json'),curve)
        print(json.dumps({'id':curve['id'],'points':len(rows),'max_nll_error':max(r['nll_abs_error'] for r in rows),'support':curve['nominal_support']}),flush=True)
    if len(curves)!=args.expected_curves:raise AssertionError('Unexpected number of profiles')
    write_json(args.output/'audit.json',{'status':'passed','curves':curves,
        'scope':'All78profile points, independent physical propagation and dense likelihood, fixed parameters, bounds and grid-support interpolation. This does not certify global profile optima or nominal coverage.'})


if __name__=='__main__':main()
