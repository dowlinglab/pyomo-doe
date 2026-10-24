"""Regularized continuous impulse-bin response benchmark with exact held inputs.

The kernel is piecewise constant over lag bins. Its bin integrals are weights,
with total DC gain bounded like the thermal models. Ambient propagation and
OU/noise shape are deliberately fixed to the n=1 reference as a diagnostic.
Regularization/bin/horizon sensitivities are not probability intervals.
"""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from scipy.optimize import lsq_linear
from data import load
from covariance import Whitening
from run_mean import write_json
from diagnostics import input_correlation


def primitive(time,command,query):
    area=np.r_[0.,np.cumsum(command[:-1]*np.diff(time))]
    index=np.searchsorted(time,query,side='right')-1
    clipped=np.clip(index,0,len(time)-1)
    values=area[clipped]+command[clipped]*(query-time[clipped])
    return np.where(query<time[0],0.,values)


def impulse_bin_features(time,command,width,horizon):
    count=int(round(horizon/width))
    edges=np.arange(count+1)*width
    integrals=primitive(time,command,time[:,None]-edges[None,:])
    return (integrals[:,:-1]-integrals[:,1:])/width


def audit_features():
    from scipy.integrate import quad
    rng=np.random.RandomState(83)
    t=np.r_[0.,np.cumsum(rng.uniform(.4,1.8,70))]
    u=rng.uniform(0,100,len(t));width,horizon=3.,30.
    X=impulse_bin_features(t,u,width,horizon);errors=[]
    for row,col in ((0,0),(3,0),(13,2),(25,5),(60,7)):
        a,b=t[row]-(col+1)*width,t[row]-col*width
        total=0.
        for k in range(len(t)-1):
            lo,hi=max(a,t[k]),min(b,t[k+1])
            if hi>lo:total+=quad(lambda s:u[k],lo,hi,epsabs=1e-10)[0]
        errors.append(abs(X[row,col]-total/width))
    step=np.arange(0.,31.)
    Xstep=impulse_bin_features(step,np.full(len(step),50.),3.,30.)
    reference=50*np.minimum(step/3,1.)
    errors.append(float(np.max(abs(Xstep[:,0]-reference))))
    if max(errors)>1e-9:raise AssertionError(errors)
    return {'status':'passed','maximum_feature_abs_error':float(max(errors)),
            'scope':'Irregular held-input impulse-bin integral vs quadrature and analytic step ramp'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    write_json(args.output/'forward_audit.json',audit_features())
    config=json.loads((HERE.parent/'residual_study/config.json').read_text())
    records=[load(n) for n in config['records']]
    y=np.concatenate([d.temperature[1:]-d.temperature[0] for d in records]);N=len(y)
    tau,ell,eta,sigma=137.4796263539508,213.87280337507227,.035823790894750226,.47877001038110467
    W=Whitening([d.time[1:] for d in records],'ou_nugget',[ell,eta])
    B=np.zeros((N,len(records)));cursor=0
    for j,d in enumerate(records):
        n=len(d.time)-1;B[cursor:cursor+n,j]=-np.expm1(-(d.time[1:]-d.time[0])/tau);cursor+=n
    yw=W(y)/(sigma*np.sqrt(N));Bw=W(B)/(sigma*np.sqrt(N));rows=[];kernels={}
    for width in (3.,6.):
        for horizon in (600.,900.):
            M=int(round(horizon/width))
            F=np.vstack([impulse_bin_features(d.time,d.input,width,horizon)[1:] for d in records])
            # c=(K,w_0,...,w_{M-2}); w_last=K-sum(other weights).
            T=np.zeros((M,M));T[:-1,1:]=np.eye(M-1);T[-1,0]=1.;T[-1,1:]=-1.
            X=np.column_stack((W(F@T)/(sigma*np.sqrt(N)),Bw))
            D=np.diff(np.eye(M),n=2,axis=0)
            penalty=np.column_stack((D@T/(.71/M),np.zeros((M-2,len(records)))))
            singular=np.linalg.svd(X,compute_uv=False)
            for strength in (1e-4,.01,1.,100.):
                augmented=np.vstack((X,np.sqrt(strength)*penalty))
                target=np.r_[yw,np.zeros(M-2)]
                lower=np.r_[.2,np.full(M-1,-np.inf),np.full(len(records),-2.)]
                upper=np.r_[1.5,np.full(M-1,np.inf),np.full(len(records),2.)]
                fitted=lsq_linear(augmented,target,bounds=(lower,upper),method='bvls',tol=1e-10,max_iter=1000)
                if not fitted.success:raise RuntimeError(fitted.message)
                coefficients=fitted.x;weights=T@coefficients[:M];ambient=coefficients[M:]
                prediction=F@weights+B@ambient;residual=y-prediction;z=W(residual)/sigma
                correlations=[];cursor=0
                for d in records[:3]:
                    count=len(d.time)-1
                    cc=input_correlation(z[cursor:cursor+count],d.input[1:],30)
                    correlations.append(max(abs(r['correlation']) for r in cc if r['correlation'] is not None));cursor+=count
                data_term=float(np.sum((W(residual)/(sigma*np.sqrt(N)))**2))
                penalty_term=float(strength*np.sum((D@weights/(.71/M))**2))
                objective=float(np.sum((augmented@coefficients-target)**2))
                if abs(objective-data_term-penalty_term)>1e-7*max(1.,objective):raise AssertionError('Objective reconstruction')
                gradient=2*augmented.T@(augmented@coefficients-target)
                projected=gradient.copy()
                projected[(coefficients<=lower+1e-7)&(gradient>0)]=0.
                projected[(coefficients>=upper-1e-7)&(gradient<0)]=0.
                pg=float(np.max(abs(projected)))
                name='bin'+str(int(width))+'_horizon'+str(int(horizon))+'_lambda'+str(strength)
                row={'id':name,'bin_width_s':width,'horizon_s':horizon,'lambda':strength,'K':float(weights.sum()),
                     'RMSE_C':float(np.sqrt(np.mean(residual**2))),'weighted_mean_square':data_term,
                     'penalty':penalty_term,'objective':objective,'projected_gradient_max_abs':pg,
                     'unregularized_design_condition':float(singular[0]/singular[-1]),
                     'unregularized_relative_rank_1e8':int(np.sum(singular>singular[0]*1e-8)),
                     'design_columns':X.shape[1],'negative_kernel_mass':float(-weights[weights<0].sum()),
                     'ambient_bound_count':int(np.sum(abs(ambient)>2-1e-7)),
                     'fast_input_innovation_correlations':correlations,'weights':weights.tolist(),'ambient_offsets':ambient.tolist()}
                rows.append(row);kernels[name]=(np.arange(M)+.5)*width,weights/width
                write_json(args.output/'progress.json',{'completed':len(rows),'expected':16,'latest':name})
                print(json.dumps({k:row[k] for k in ('id','RMSE_C','K','negative_kernel_mass','projected_gradient_max_abs')}),flush=True)
    write_json(args.output/'fir_results.json',{'status':'complete','cells':rows,
        'noise_reference':{'tau_ambient_s':tau,'ell_s':ell,'eta':eta,'sigma_C':sigma},
        'penalty_definition':'Data mean squared whitened residual +lambda||D2 w/(0.71/M)||². w are impulse-bin integrals. K=sum(w) bounded0.2..1.5; other weights unrestricted; ambient±2C.',
        'notice':'Conditional regularized benchmark, not a newly estimated stochastic likelihood. Bin/horizon/lambda variation is sensitivity, not confidence. Signed kernels allowed. Finite horizon and fixed ambient/noise reference restrict interpretation; no sub-bin delay identification.'})
    plot(args.output,kernels)


def plot(output,kernels):
    import hashlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for name,(t,g) in kernels.items():
        width=3 if name.startswith('bin3_') else 6
        col=0 if width==3 else 1
        axes[0,col].plot(t,g,label=name.split('_',1)[1],linewidth=.7)
        axes[1,col].plot(t,g,linewidth=.7)
    for col,width in enumerate((3,6)):
        axes[0,col].set(title=str(width)+'-second impulse bins',xlabel='Lag, s',ylabel='Kernel, C / percent / s')
        axes[1,col].set(title='Early response',xlabel='Lag, s',ylabel='Kernel, C / percent / s',xlim=(0,45))
        axes[0,col].legend(fontsize=6)
    for ax in axes.flat:ax.axhline(0,color='gray',linewidth=.5);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'fir_sensitivity.png',dpi=160);plt.close(fig)
    write_json(output/'figure_receipt.json',{'visual_review':'pending','figures':{'fir_sensitivity.png':hashlib.sha256((output/'fir_sensitivity.png').read_bytes()).hexdigest()}})


if __name__=='__main__':main()
