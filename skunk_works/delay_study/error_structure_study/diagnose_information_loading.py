"""Resolve tiny loading curvature with independent scalar/score step studies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from joint_quantized import Model
from quantized import filter_ou
from audit_common import dimensional_mean
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--information',type=Path,required=True)
    args=parser.parse_args();fit=json.loads((args.input/'fit.json').read_text())
    lookup={d['name']:d for d in load_records()};records=[lookup[n] for n in fit['record_names']]
    m=Model(records,fit['band'],fit['phase'],128,'conductance');p=np.array(fit['best']['point'])
    H=np.load(args.information/'information_matrices.npz')['observed_Hessian'];rows=[]
    def coeff(q):return np.r_[np.exp(q[m.nshape]),q[m.nshape+1:m.nmean]]
    for h in [0.,.0002,.0004,.0006,.002,.004,.006,.005,.01,.015,.02,.04,.06,.05,.1,.15]:
        q=p.copy();q[2]+=h;mean=dimensional_mean(q,coeff(q),records,fit['band'])
        nll=0.
        for k,noise in enumerate(fit['noise_parameters']):
            S=noise['latent_SD_C']**2;R=noise['measurement_SD_C']**2;cursor=0
            for d in records:
                n=len(d['t'])-1;s=slice(k*m.N+cursor,k*m.N+cursor+n)
                f=filter_ou(d['t'][1:],mean[s],d['y'][1:,k],np.sqrt(S+R),noise['ell_s'],R/(S+R),fit['phase'],256,12.)
                assert np.max(abs(f['posterior_normalization_ratio']-1))<1e-6
                nll+=f['nll'];cursor+=n
        value,score,_=m.evaluate(tuple(q));model_mean=m.mean(q,False)
        row={'step':h,'dimensional_forward_nll_256':float(nll),'model_nll_128':float(value),
             'loading_score_128':float(score[2]),'dimensional_mean_max_error_C':float(np.max(abs(mean-model_mean)))}
        rows.append(row);write(args.output/'evaluations.json',rows);print(json.dumps(row),flush=True)
    by={r['step']:r for r in rows};checks=[]
    for h in [.0002,.002,.005,.01,.02,.05]:
        rr=[by[float(round(j*h,10))] for j in range(4)]
        scalar=(2*rr[0]['dimensional_forward_nll_256']-5*rr[1]['dimensional_forward_nll_256']+4*rr[2]['dimensional_forward_nll_256']-rr[3]['dimensional_forward_nll_256'])/h**2
        score=(-3*rr[0]['loading_score_128']+4*rr[1]['loading_score_128']-rr[2]['loading_score_128'])/(2*h)
        checks.append({'step':h,'scalar_curvature':scalar,'score_curvature':score,'stored_H22':float(H[2,2])})
    write(args.output/'diagnosis.json',{'fit_sha256':hashlib.sha256((args.input/'fit.json').read_bytes()).hexdigest(),
        'checks':checks,'evaluations':rows,'scope':'Step study, not an uncertainty estimate or automatic audit acceptance.'})


if __name__=='__main__':main()
