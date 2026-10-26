"""Locate failed posterior-integration gates at a preserved joint-fit seed."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from physical import load_records
from joint_quantized import Model
from smoothing import smooth_ou


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--band',type=float,required=True)
    args=parser.parse_args();starts=json.loads((args.input/'starts.json').read_text());point=np.array(starts[0]['seed']);records=load_records()
    model=Model(records,args.band);mu=model.mean(point,False);rows=[]
    for c in [0,1]:
        latent,ell,measurement=np.exp(point[model.nmean+3*c:model.nmean+3*c+3]);cursor=0
        for j,d in enumerate(records):
            n=len(d['t'])-1;section=slice(c*model.N+cursor,c*model.N+cursor+n)
            f=smooth_ou(d['t'][1:],mu[section],d['y'][1:,c],latent,ell,measurement,nodes=128)
            norm=abs(f['posterior_normalization_ratio']-1);i=int(np.argmax(norm))
            row={'name':d['name'],'channel':c+1,'latent_SD':latent,'ell':ell,'measurement_SD':measurement,
                 'normalization_error':float(norm[i]),'smoothing_closure_error':f['smoothing_closure_error'],
                 'worst_time_s':float(d['t'][i+1]),'worst_log_predictive_mass':float(f['log_bin_probability'][i]),
                 'RMSE_C':float(np.sqrt(np.mean((d['y'][1:,c]-mu[section])**2)))}
            rows.append(row);print(json.dumps(row),flush=True);cursor+=n
    (args.output/'diagnosis.json').write_text(json.dumps({'status':'diagnostic','rows':rows,'notice':'Preserved seed only; rejected numerical integration is not evidence against the shared physical model.'},indent=2)+'\n')


if __name__=='__main__':main()
