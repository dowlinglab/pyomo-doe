"""Verify lag-permutation symmetry in the vanishing sensor-loading limit."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
import matplotlib.pyplot as plt
from physical import load_records,transfers,Model as MeanModel
from joint_quantized import Model as JointModel
from audit_bounded import independent
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);args=parser.parse_args();fit=json.loads(args.warm.read_text())
    p=np.asarray(fit['best']['point'][:6]);K=fit['parameters']['K_common_nominal'];zero=p.copy();zero[2]=-np.inf
    swapped=zero.copy();swapped[1],swapped[4]=zero[4],zero[1];zero_errors=[];rows=[]
    for period in [30.,120.,300.]:
        omega=2*np.pi/period;s=1j*omega;ta,rs,_,q,L=np.exp(zero[:5])
        common=K/((1+s*rs)*(1+s*L));plus=common/(1+s*ta);minus=common/(1+2*q+s*ta)
        factorized=np.array([(plus+minus)/2,(plus-minus)/2]);direct=transfers(zero,K,omega,True)
        error=max(float(np.max(abs(direct-factorized))),float(np.max(abs(direct-transfers(swapped,K,omega,True)))))
        assert error<1e-12;zero_errors.append(error)
    original=load_records()[0];record=dict(original,y=np.full_like(original['y'],20.))
    m=MeanModel([record],True,True);n=len(record['t'])-1;max_dimensional_error=0.
    for loading in [.0001,.001,.005,.03,.1,1.]:
        shape=p.copy();shape[2]=np.log(loading);other=shape.copy();other[1],other[4]=shape[4],shape[1]
        X,_,initial=m.design(tuple(shape));answer=20+initial+K*X[:,0]
        XX,_,ii=m.design(tuple(other));permuted=20+ii+K*XX[:,0]
        physical=dict(fit['parameters']);physical.update({'Us_over_Ua_nominal':loading,'Ua_run_over_nominal':[1.]})
        f={'parameters':physical,'lag':True,'curve':True,'ambient':'constant','record_names':[record['name']],
           'coefficients':[K,0.]}
        dimensional=independent(record,f);saved=np.column_stack((np.r_[20.,answer[:n]],np.r_[20.,answer[n:]]))
        err=float(np.max(abs(dimensional-saved)));max_dimensional_error=max(max_dimensional_error,err);assert err<1e-8
        frequencies=[]
        for period in [30.,120.,300.]:
            delta=transfers(shape,K,2*np.pi/period,True)-transfers(other,K,2*np.pi/period,True)
            frequencies.append({'period_s':period,'complex_response_difference_magnitude_per_percent':abs(delta).tolist(),
                '50_percent_sine_difference_amplitude_C':(50*abs(delta)).tolist()})
        rows.append({'Us_over_Ua':loading,'forced_waveform_swapped_lag_max_difference_by_channel_C':
            [float(np.max(abs(answer[k*n:(k+1)*n]-permuted[k*n:(k+1)*n]))) for k in [0,1]],'frequencies':frequencies})
    tiny=JointModel([record],.1,minimum_sensor_loading=.0001)
    assert abs(tiny.bounds()[2][0]-np.log(.0001))<1e-14
    result={'status':'passed','zero_loading_transfer_max_error':max(zero_errors),
        'independent_dimensional_trajectory_max_error_C':max_dimensional_error,'positive_loading_scenarios':rows,
        'source_warm':str(args.warm),'scope':'Mathematical zero-loading transfer limit and positive-loading synthetic nominal forced responses at the recorded fast-sine command/time grid, cold at 20 C with zero ambient offset. No hardware refit, noise calibration or exact symmetry with unequal initial states/ambient forcing claimed.'}
    write(args.output/'audit.json',result)
    fig,ax=plt.subplots(figsize=(6,3.5))
    for k in [0,1]:ax.loglog([r['Us_over_Ua'] for r in rows],[r['forced_waveform_swapped_lag_max_difference_by_channel_C'][k] for r in rows],'o-',label='T'+str(k+1))
    ax.axvline(.005,color='gray',ls='--',label='Original lower bound');ax.set(xlabel='Sensor loading Us/Ua',ylabel='Maximum swapped-lag difference (C)',title='Synthetic response: ambiguity grows as loading vanishes')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(args.output/'loading_symmetry.png',dpi=150);plt.close(fig)
    print(json.dumps({'status':'passed','zero_loading_error':max(zero_errors),'dimensional_error':max_dimensional_error}),flush=True)


if __name__=='__main__':main()
