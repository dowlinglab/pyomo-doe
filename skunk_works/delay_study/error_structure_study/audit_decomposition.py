"""Check saved error partition independently through orthogonal QR coordinates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--prior',type=Path,required=True)
    args=parser.parse_args();df=pd.read_csv(args.input/'decomposition.csv')
    largest=0.;totals={};checked=0
    for label,group in df.groupby('model'):
        path=args.prior/'results'/label;fit=json.loads((path/'fit.json').read_text());a=np.load(path/'trajectories.npz')
        for _,row in group.iterrows():
            j=fit['record_names'].index(row['name']);k=int(row['channel'])-1
            t=a['time_'+str(j)][1:];keep=t>=row['cutoff_s'];t=t[keep]
            e=(a['observed_'+str(j)]-a['predicted_'+str(j)])[1:,k][keep]
            cols=[np.ones(len(t)),(t-t.mean())/max(np.ptp(t),1.)];sizes=[1,1]
            if pd.notna(row['period_s']):
                for harmonic in (1,2,3):
                    angle=2*np.pi*harmonic*t/row['period_s'];cols.extend([np.sin(angle),np.cos(angle)]);sizes.append(2)
            Q,R=np.linalg.qr(np.column_stack(cols),mode='reduced')
            projected=Q.T@e;energy=[];start=0
            for width in sizes:
                energy.append(float(np.sum(projected[start:start+width]**2)));start+=width
            if len(sizes)==2:energy.extend([0.,0.,0.])
            energy.append(float(np.linalg.norm(e-Q@projected)**2))
            names=['offset','additional_trend','additional_fundamental','additional_harmonic2','additional_harmonic3','remainder']
            expected=[row[n+'_SSE_C2'] for n in names]
            delta=float(np.max(abs(np.asarray(expected)-energy)));largest=max(largest,delta)
            assert delta<1e-8*max(1,float(e@e)), (label,row['name'],delta)
            assert abs(sum(expected)-float(e@e))<1e-8*max(1,float(e@e))
            assert abs(float(row['RMSE_C'])-np.linalg.norm(e)/np.sqrt(len(e)))<1e-10
            assert abs(float(row['centered_RMSE_C'])-np.linalg.norm(e-e.mean())/np.sqrt(len(e)))<1e-10
            checked+=1
            key=(label,k+1,int(row['cutoff_s']));totals.setdefault(key,[]).append(e)
    aggregate=pd.read_csv(args.input/'aggregate.csv')
    for _,row in aggregate.iterrows():
        es=np.concatenate(totals[(row['model'],int(row['channel']),int(row['cutoff_s']))])
        assert abs(np.linalg.norm(es)/np.sqrt(len(es))-row['RMSE_C'])<1e-10
        assert abs(sum(row[n+'_fraction'] for n in names)-1)<1e-10
    receipt={'status':'passed','rows_checked':checked,'aggregate_rows_checked':len(aggregate),'largest_QR_energy_difference_C2':largest,
        'input_sha256':{n:hashlib.sha256((args.input/n).read_bytes()).hexdigest() for n in ['decomposition.csv','aggregate.csv','reconciliation.json']},
        'scope':'Independent QR energy projections, error closure and RMSE; does not audit earlier fitted parameters, frequency uncertainty or latent noise adequacy.'}
    (args.output/'audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
