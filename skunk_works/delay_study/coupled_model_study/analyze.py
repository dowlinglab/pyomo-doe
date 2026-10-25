"""Joint-output residual, frequency-response and local information diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from scipy.signal import ss2tf,residue
from physical import Model,load_records,matrices,transfers
from frequency import held_fundamental,regression,response_summary
from run_frequency import period_of
from run_mean import write_json


def corr(a,b):
    a=a-a.mean();b=b-b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-12 else None


def diagnostics(fit,arrays):
    rows=[];frequency=[]
    for j,name in enumerate(fit['record_names']):
        t=arrays['time_'+str(j)];y=arrays['observed_'+str(j)];p=arrays['predicted_'+str(j)]
        u=arrays['input_'+str(j)];z=arrays['innovation_'+str(j)];e=y[1:]-p[1:]
        row={'name':name,'mean_residual_C':e.mean(axis=0).tolist(),
             'innovation_mean':z.mean(axis=0).tolist(),'innovation_SD':z.std(axis=0).tolist(),
             'innovation_ACF_lags1to60':[[corr(z[k:,c],z[:-k,c]) for k in range(1,61)] for c in range(2)],
             'innovation_cross_channel_lags_minus20to20':[corr(z[max(k,0):len(z)+min(k,0),0],z[max(-k,0):len(z)-max(k,0),1]) for k in range(-20,21)],
             'innovation_input_lags_minus60to60':[[corr(z[max(k,0):len(z)+min(k,0),c],u[1:][max(-k,0):len(z)-max(k,0)]) for k in range(-60,61)] for c in range(2)]}
        rows.append(row);period=period_of(name)
        if period is None:continue
        for cutoff in [300.,450.,600.]:
            keep=t>=cutoff;inp=held_fundamental(t,u,2*np.pi/period,cutoff,t[-1])
            for c in range(2):
                measured=response_summary(regression(t[keep],y[keep,c],period),inp)
                matched=response_summary(regression(t[keep],p[keep,c],period),inp)
                H=transfers(fit['best']['point'],fit['coefficients'][0],2*np.pi/period,fit['lag'])[c]
                frequency.append({'name':name,'channel':c+1,'period_s':period,'cutoff_s':cutoff,
                    'cycles':(t[-1]-cutoff)/period,'measured':measured,'matched':matched,
                    'linearized_continuous_gain':float(abs(H)),'linearized_continuous_phase_deg':float(np.angle(H,deg=True)),
                    'matched_phase_error_deg':float(np.angle(complex(matched['H_real'],matched['H_imag'])/complex(measured['H_real'],measured['H_imag']),deg=True)),
                    'matched_gain_ratio':matched['gain_C_per_percent']/measured['gain_C_per_percent']})
    return rows,frequency


def information(fit,records):
    model=Model(records,fit['lag'],fit['curve'],fit['ambient'],fit['kernel'])
    point=np.array(fit['best']['point']);_,coef,sigma,W,initial,X=model.projected(point)
    ns=model.ns;K=coef[0];columns=[]
    for j in range(ns):
        h=1e-4;unit=np.zeros(ns);unit[j]=h
        xp,yp,ip=model.design(tuple(point[:ns]+unit));xm,ym,im=model.design(tuple(point[:ns]-unit))
        columns.append(((ip+xp@coef)-(im+xm@coef))/(2*h))
    columns.append(K*X[:,0]);J=np.column_stack(columns);rho=np.exp(point[-1])
    whiten=lambda a:np.vstack([W[k](a[k*model.N:(k+1)*model.N])/(sigma*(rho if k else 1)) for k in range(2)])
    Jw=whiten(J);nuisance=whiten(X[:,1:]);Jp=Jw-nuisance@np.linalg.lstsq(nuisance,Jw,rcond=None)[0]
    U,s,V=np.linalg.svd(Jp,full_matrices=False);eig=s*s
    names=fit['coordinate_names'][:ns]+['log_K_common']
    covariance=(V.T/(s*s))@V
    # The local calculation fixes the covariance and uses unconstrained tangent
    # directions even at active bounds. It is a diagnostic, not a CI there.
    active=[names[j] for j in range(ns) if min(point[j]-fit['bounds'][j][0],fit['bounds'][j][1]-point[j])<1e-4]
    A,B,_=matrices(point,fit['lag']);poles=np.linalg.eigvals(A)
    transfer_rows=[]
    for c in [1,3]:
        C=np.eye(len(A))[c:c+1];num,den=ss2tf(A,B[:,None],C,np.zeros((1,1)))
        n=num[0];threshold=1e-9*max(abs(n));first=np.flatnonzero(abs(n)>threshold)[0]
        zeros=np.roots(n[first:]);residues,rp,direct=residue(K*n[first:],den,tol=1e-8)
        reconstruction_error=0.
        for omega in np.logspace(-4,1,50):
            reference=transfers(point,K,omega,fit['lag'])[0 if c==1 else 1]
            reconstructed=np.sum(residues/(1j*omega-rp))
            if len(direct):reconstructed+=np.polyval(direct,1j*omega)
            reconstruction_error=max(reconstruction_error,abs(reference-reconstructed))
        if reconstruction_error>1e-6:raise AssertionError(('Modal residue reconstruction',reconstruction_error))
        transfer_rows.append({'channel':1 if c==1 else 2,
            'numerator':n[first:].tolist(),'denominator':den.tolist(),
            'zeros_real':zeros.real.tolist(),'zeros_imag':zeros.imag.tolist(),
            'residue_real':residues.real.tolist(),'residue_imag':residues.imag.tolist(),
            'residue_pole_real':rp.real.tolist(),'residue_pole_imag':rp.imag.tolist(),
            'modal_reconstruction_max_abs':float(reconstruction_error)})
    return {'coordinates':names,'eigenvalues_descending':eig.tolist(),'condition_number':float(eig[0]/eig[-1]),
            'weakest_unit_direction':dict(zip(names,V[-1].tolist())),
            'working_local_SD':dict(zip(names,np.sqrt(np.diag(covariance)).tolist())),
            'active_shape_bounds':active,'ambient_active_count':int(np.sum(abs(coef[1:])>2-1e-6)),
            'pole_real':poles.real.tolist(),'pole_imag':poles.imag.tolist(),'transfers':transfer_rows,
            'notice':'Fixed-error-covariance mean-information matrix; ambient nuisance tangent projected out. No noise re-estimation, selection uncertainty, constrained/profile CI or proof of practical identifiability. Active bounds invalidate regular Wald interpretations.'}


def plots(output,allrows,models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'savefig.dpi':150})
    colors=['#2166ac','#b2182b','#1b7837','#762a83','#e08214','#111111']
    fig,axes=plt.subplots(3,2,figsize=(10,8),sharex=True)
    indices=[0,6,9]
    for r,j in enumerate(indices):
        first=next(iter(models.values()))[1];t=first['time_'+str(j)];y=first['observed_'+str(j)]
        for c in range(2):
            axes[r,c].plot(t,y[:,c],color='.65',lw=1,label='Measured')
            for k,(label,(f,a)) in enumerate(models.items()):
                if f['kernel']=='iid':continue
                axes[r,c].plot(t,a['predicted_'+str(j)][:,c],color=colors[k%len(colors)],lw=.85,label=label)
            axes[r,c].set_ylabel('Temperature, C');axes[r,c].set_title(('30-s sine','300-s sine','Step')[r]+'; T'+str(c+1));axes[r,c].grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Time, s')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(output/'joint_trajectories.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    measured_done=False
    for k,(label,entry) in enumerate(allrows.items()):
        if entry['kernel']=='iid':continue
        for run in [1,2,3]:
            for c in [1,2]:
                f=sorted([r for r in entry['frequency'] if r['channel']==c and r['cutoff_s']==450 and '_run_'+str(run)+'.csv' in r['name']],key=lambda r:-r['period_s'])
                freq=[1/r['period_s'] for r in f];phase=np.rad2deg(np.unwrap(np.deg2rad([r['matched']['phase_deg'] for r in f])))
                axes[0,c-1].plot(freq,[r['matched']['gain_C_per_percent'] for r in f],'-',color=colors[k%len(colors)],alpha=.65,label=label if run==1 else None)
                axes[1,c-1].plot(freq,phase,'-',color=colors[k%len(colors)],alpha=.65)
                if not measured_done:
                    mp=np.rad2deg(np.unwrap(np.deg2rad([r['measured']['phase_deg'] for r in f])))
                    axes[0,c-1].errorbar(freq,[r['measured']['gain_C_per_percent'] for r in f],yerr=[1.96*r['measured']['HAC']['60']['gain_SD'] for r in f],fmt='ko',markersize=3,alpha=.5)
                    axes[1,c-1].errorbar(freq,mp,yerr=[1.96*r['measured']['HAC']['60']['phase_SD_deg'] for r in f],fmt='ko',markersize=3,alpha=.5)
        measured_done=True
    for c in range(2):
        axes[0,c].set_title('T'+str(c+1));axes[0,c].set_yscale('log');axes[0,c].set_ylabel('Gain, C / command percent')
        axes[1,c].set_ylabel('Phase, degrees');axes[1,c].set_xlabel('Frequency, Hz')
        for r in range(2):axes[r,c].set_xscale('log');axes[r,c].grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('Matched finite-window response after 450 s; dots: measured, working HAC60 ranges');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(output/'joint_frequency.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fits',nargs='+',type=Path,required=True);args=parser.parse_args()
    records=load_records();rows={};models={}
    for path in args.fits:
        f=json.loads((path/'fit.json').read_text());a=np.load(path/'trajectories.npz');res,freq=diagnostics(f,a)
        rows[path.name]={'fit_sha256':hashlib.sha256((path/'fit.json').read_bytes()).hexdigest(),
            'arrays_sha256':hashlib.sha256((path/'trajectories.npz').read_bytes()).hexdigest(),
            'RMSE_by_channel_C':f['RMSE_by_channel_C'],'nll':f['best']['nll'],'AICc':f['nominal_AICc'],
            'parameters':f['parameters'],'kernel':f['kernel'],'stationarity_warning':f['stationarity_warning'],
            'per_run':res,'frequency':freq,'information':information(f,records)}
        models[path.name]=(f,a)
    write_json(args.output/'analysis.json',{'models':rows,'notice':'All twelve runs reused. Frequency estimates descriptive, HAC working errors, no independent validation. Continuous transfer is the linear command response; finite-window predicted curves include command curvature and transients.'})
    plots(args.output,rows,models)
    write_json(args.output/'figure_receipt.json',{'visual_review':'pending','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.png')}})
    print(json.dumps({'status':'complete','models':list(rows)}),flush=True)


if __name__=='__main__':main()
