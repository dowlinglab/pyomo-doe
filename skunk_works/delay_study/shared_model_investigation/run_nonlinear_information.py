"""Conditional scaled sensitivity information for the quadratic-loss diagnostic."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from data import load
from covariance import Whitening
from nonlinear import predict_rise
from run_mean import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    fit=json.loads((args.input/'fit.json').read_text());x=np.array(fit['best']['point']);p=fit['parameters']
    records=[load(n) for n in fit['record_names']]
    W=Whitening([d.time[1:] for d in records],'ou_nugget',[p['ell_s'],p['eta']]);sigma=fit['sigma_C']
    indexes=list(range(5))+list(range(7,len(x)))
    def trajectory(point):
        return np.concatenate([predict_rise(d.time,d.input,np.exp(point[3]),np.exp(point[0]),np.exp(point[1]),point[2],.01*point[4],point[7+j],fit['max_step_s'])[1:] for j,d in enumerate(records)])
    rows=[];matrices=[]
    for step in (2e-4,1e-4):
        columns=[]
        for k in indexes:
            plus=x.copy();minus=x.copy();plus[k]+=step;minus[k]-=step
            columns.append((trajectory(plus)-trajectory(minus))/(2*step))
        J=W(np.column_stack(columns))/sigma;B=J[:,5:];mean=J[:,:5]
        mean-=B@np.linalg.lstsq(B,mean,rcond=None)[0]
        matrices.append(mean.copy())
        U,s,V=np.linalg.svd(mean,full_matrices=False)
        covariance=(V.T*(1/s**2))@V
        sd=np.sqrt(np.maximum(np.diag(covariance),0))
        rows.append({'step':step,'singular_values':s.tolist(),'condition_J':float(s[0]/s[-1]),
                     'relative_rank_1e8':int(np.sum(s>s[0]*1e-8)),
                     'coordinate_SD':sd.tolist(),'beta_SD_per_C':float(sd[4]*.01),
                     'covariance':covariance.tolist(),'weakest_scaled_direction':V[-1].tolist()})
    disagreement=float(np.linalg.norm(matrices[0]-matrices[1])/max(np.linalg.norm(matrices[1]),1.))
    write_json(args.output/'information.json',{'status':'complete','coordinate_order':['log_tau','log_theta','delay_seconds','log_K','beta_divided_by_0.01'],
                'step_comparisons':rows,'relative_J_step_difference':disagreement,
                'notice':'Fixes covariance and treats ambient nuisance directions as unrestricted even at bounds. Local numerical rank/SD do not establish global identification, coverage, physical mechanism or model adequacy; compare profiles and alternate thermal-role basins.'})
    print(json.dumps({'status':'complete','relative_J_step_difference':disagreement,'beta_SD_per_C':rows[-1]['beta_SD_per_C']}),flush=True)


if __name__=='__main__':main()
