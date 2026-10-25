"""Summarize selected fits, frequency mismatch, zeros and impulse response."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
import pandas as pd
from scipy.linalg import expm
from data import load
from pooled_models import make_model
from run_mean import write_json


def physical_system(times,physical):
    m=len(times);A=np.zeros((m,m));B=np.zeros(m)
    for j in range(m-1):
        A[j,j]=-1/times[j+1]
        if j:A[j,j-1]=1/times[j+1]
        else:B[j]=1/times[j+1]
    A[-1,-1]=-1/times[0];A[-1,:m-1]=physical[1:]/times[0];B[-1]=physical[0]/times[0]
    return A,B


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--frequency',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    empirical=json.loads((args.frequency/'frequency_estimates.json').read_text())['rows']
    measured=[r for r in empirical if r['cutoff_s']==450 and r['nuisance']=='trend']
    rows=[];curves=[]
    omega=np.logspace(np.log10(.005),np.log10(2.),300)
    time=np.linspace(0,120,481)
    for folder in args.inputs:
        fit=json.loads((folder/'fit.json').read_text());best=fit['best']
        model=make_model([load(n) for n in fit['record_names']],fit['kind'],fit['order'],fit.get('numerator_bound',2.),fit.get('ambient_mode','constant'))
        x=np.array(best['point']);c=np.array(best['coefficients']);times=np.array(fit['time_constants_s'])
        physical=np.r_[np.zeros(model.m-1),c[0]] if fit['kind'] in ('chain','transport','unequal_chain') else np.r_[c[1:model.m],c[0]-sum(c[1:model.m])]
        polynomial=np.poly1d([0.])
        for j,coefficient in enumerate(physical):
            term=np.poly1d([coefficient])
            for value in times[j+1:]:term*=np.poly1d([value,1.])
            polynomial+=term
        coefficients=np.trim_zeros(polynomial.c,'f')
        zeros=np.roots(coefficients) if len(coefficients)>1 else np.array([])
        A,B=physical_system(times,physical)
        delay=x[2] if fit['kind']=='transport' else 0.
        impulse=np.array([(expm(A*(t-delay))@B)[-1] if t>=delay else 0. for t in time])
        H=np.array([model.frequency_response(x,c,w) for w in omega])
        state_frequency=np.array([np.exp(-1j*w*delay)*np.linalg.solve(1j*w*np.eye(model.m)-A,B)[-1] for w in omega])
        denominator=np.poly1d([1.])
        for value in times:denominator*=np.poly1d([value,1.])
        polynomial_frequency=np.exp(-1j*omega*delay)*np.polyval(polynomial,1j*omega)/np.polyval(denominator,1j*omega)
        frequency_error=float(max(np.max(abs(H-state_frequency)),np.max(abs(H-polynomial_frequency))))
        dc_error=float(abs(np.linalg.solve(-A,B)[-1]-c[0]))
        if frequency_error>1e-8 or dc_error>1e-9:raise AssertionError((folder.name,frequency_error,dc_error))
        freq=[]
        for period in (30.,120.,300.):
            theoretical=model.frequency_response(x,c,2*np.pi/period)
            selected=[r for r in measured if r['period_s']==period]
            differences=[float(np.angle(complex(r['measured']['H_real'],r['measured']['H_imag'])/theoretical,deg=True)) for r in selected]
            ratios=[r['measured']['gain_C_per_percent']/abs(theoretical) for r in selected]
            freq.append({'period_s':period,'model_gain':float(abs(theoretical)),
                         'model_principal_phase_deg':float(np.angle(theoretical,deg=True)),
                         'empirical_minus_model_phase_deg':differences,'empirical_model_gain_ratios':ratios,
                         'notice':'Measured windowed response versus continuous candidate; primary matched-window n1 sensitivity is saved separately. Settling/window errors remain.'})
        row={'id':folder.name,'kind':fit['kind'],'order':fit['order'],'K':c[0],'time_constants_s':times.tolist(),
             'additional_delay_s':delay,'ambient_mode':fit.get('ambient_mode','constant'),
             'nll':best['nll'],'RMSE_C':fit['RMSE_C'],'nominal_AICc':fit['nominal_AICc'],
             'parameter_count':fit['parameter_count'],'sigma_C':best['sigma_C'],'ell_s':float(np.exp(x[-2])),
             'eta':x[-1],'successful_starts':sum(r['success'] for r in fit['starts']),
             'total_starts':len(fit['starts']),'stationarity_warning':fit['stationarity_warning'],
             'ambient_bound_count':int(sum(abs(v)>2-1e-7 for v in c[model.ninput:])),
             'physical_input_coefficients':physical.tolist(),
             'transfer_zeros':[{'real':float(v.real),'imag':float(v.imag)} for v in zeros],
             'frequency_realization_max_abs_error':frequency_error,'DC_gain_max_abs_error':dc_error,
             'right_half_plane_zero_count':int(sum(v.real>1e-8 for v in zeros)),
             'initial_impulse_C_per_percent_s':float(impulse[0]),'minimum_impulse_first120s':float(impulse.min()),
             'fast_input_innovation_correlation_range':[min(r['max_abs_input_innovation_correlation_lags30'] for r in fit['per_run'][:3]),max(r['max_abs_input_innovation_correlation_lags30'] for r in fit['per_run'][:3])],
             'frequency_comparison':freq,'fit_sha256':hashlib.sha256((folder/'fit.json').read_bytes()).hexdigest()}
        rows.append(row);curves.append({'id':folder.name,'H':H,'impulse':impulse})
    write_json(args.output/'comparison.json',{'models':rows,'scope':'Descriptive same-record working-likelihood comparison. Flexible real-pole models can have nonminimum-phase zeros/negative early impulse; this is a benchmark, not proof of physical inverse response. No adequacy selection solely by nominal AICc.'})
    pd.DataFrame([{k:r[k] for k in ('id','K','nll','RMSE_C','nominal_AICc','parameter_count','ell_s','sigma_C','eta','right_half_plane_zero_count','initial_impulse_C_per_percent_s','stationarity_warning')} for r in rows]).to_csv(args.output/'comparison.csv',index=False)
    plot(args.output,omega,time,curves,measured)
    print(json.dumps({'status':'complete','models':[r['id'] for r in rows]}),flush=True)


def plot(output,omega,time,curves,measured):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for r in curves:
        axes[0,0].loglog(omega,np.abs(r['H']),label=r['id'])
        axes[1,0].semilogx(omega,np.rad2deg(np.unwrap(np.angle(r['H']))))
        axes[0,1].plot(time,r['impulse'],label=r['id'])
        axes[1,1].plot(time,r['impulse'])
    for run in (1,2,3):
        points=sorted([r for r in measured if r['run']==run],key=lambda r:-r['period_s'])
        w=[2*np.pi/r['period_s'] for r in points]
        axes[0,0].loglog(w,[r['measured']['gain_C_per_percent'] for r in points],'ko',markersize=3)
        axes[1,0].semilogx(w,np.rad2deg(np.unwrap(np.deg2rad([r['measured']['phase_deg'] for r in points]))),'ko',markersize=3)
    axes[0,0].set(ylabel='Gain, C / heater percent',title='Measured fundamentals (black points)')
    axes[1,0].set(xlabel='Angular frequency, rad/s',ylabel='Unwrapped phase, degrees')
    axes[0,1].set(xlabel='Time, s',ylabel='Impulse response, C / percent / s',title='Command-response shape')
    axes[1,1].set(xlabel='Time, s',ylabel='Impulse response, C / percent / s',title='Early impulse response',xlim=(0,12))
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,1].axhline(0,color='gray',linewidth=.6);axes[1,1].axhline(0,color='gray',linewidth=.6)
    fig.tight_layout();fig.savefig(output/'model_frequency_impulse.png',dpi=160);plt.close(fig)
    write_json(output/'figure_receipt.json',{'visual_review':'pending','figures':{'model_frequency_impulse.png':hashlib.sha256((output/'model_frequency_impulse.png').read_bytes()).hexdigest()}})


if __name__=='__main__':main()
