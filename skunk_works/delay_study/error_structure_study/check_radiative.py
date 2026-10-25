"""Radiation propagation, complex derivatives and conditional synthetic recovery."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from physical import load_records
from radiative import Model,TREF
from nonlinear import stable_gradient


def warm_point(f,nr,gamma):
    shape=np.array(f['best']['point'][:6]);factor=1+gamma*(TREF+296.15)*(TREF*TREF+296.15**2)/(4*TREF**3)
    shape[[0,2,3]]+=np.log(factor)
    K=f['coefficients'][0]*factor
    return np.r_[shape,np.log(K),gamma,np.zeros(nr),np.full(nr,.5),f['best']['point'][f['shape_count']:]]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path,required=True);args=parser.parse_args()
    fit=json.loads(args.warm.read_text());records=load_records();short=[]
    for d in records[:3]:
        v=dict(d)
        for key in ['t','u','y']:v[key]=d[key][:121].copy()
        short.append(v)
    model=Model(short);p=warm_point(fit,3,.8);p[2]=np.log(.03);p[8:11]=[-.02,0.,.02]
    predicted=model.prediction(p).reshape(2,-1).T
    # Direct adaptive integration uses dimensional capacities/conductances and
    # explicit radiative coefficient; it does not call the RK4 balance routine.
    ua0=.04;ta,rs,r,q,L=np.exp(p[:5]);K=np.exp(p[6]);gamma=p[7]
    CH=ua0*ta;US=ua0*r;UX=ua0*q;CS=US*rs;beta=ua0*K;rad=ua0*gamma/(4*TREF**3)
    independent=[]
    for j,d in enumerate(short):
        ambient=d['y'][0,0]+p[11+j];ua=ua0*np.exp(p[8+j]);state=np.r_[np.repeat(d['y'][0],2),0.]
        # repeat gives [T1,T1,T2,T2,0], matching hidden cold-start baseline.
        for k,dt in enumerate(np.diff(d['t'])):
            u=d['u'][k]+p[5]*d['u'][k]*(1-d['u'][k]/100)
            def balance(t,z):
                h1,s1,h2,s2,power=z
                return [(-ua*(h1-ambient)+US*(s1-h1)+UX*(h2-h1)+beta*power-rad*((h1+273.15)**4-(ambient+273.15)**4))/CH,
                        US*(h1-s1)/CS,(-ua*(h2-ambient)+US*(s2-h2)+UX*(h1-h2)-rad*((h2+273.15)**4-(ambient+273.15)**4))/CH,
                        US*(h2-s2)/CS,(u-power)/L]
            f=solve_ivp(balance,[0,dt],state,rtol=2e-11,atol=2e-12);assert f.success
            state=f.y[:,-1];independent.append(state[[1,3]])
    ode_error=float(np.max(abs(predicted-np.array(independent))))
    step_error=float(np.max(abs(model.prediction(p)-model.prediction(p,max_step=.125))))
    assert ode_error<2e-6 and step_error<2e-6,(ode_error,step_error)
    gradient=model.gradient(p);fd=stable_gradient(lambda q:model.profile_noise(q,model.y-model.prediction(q))[0],p,model.bounds(),step=2e-6)
    derivative_error=float(np.max(abs(gradient-fd)/(1+abs(fd))))
    assert derivative_error<3e-3,derivative_error
    # Separate three actual-input protocols; hold shared dynamic ratios fixed
    # while recovering gain, radiation and ambient offsets from noiseless data.
    records=[records[j] for j in (0,3,6)];generator=Model(records);truth=warm_point(fit,3,.8);truth[2]=np.log(.03)
    truth[11:14]=[.5,-.7,.9];synthetic=[];mean=generator.prediction(truth);cursor=0
    for d in records:
        v=dict(d);v['y']=d['y'].copy();n=len(d['t'])-1
        for k in [0,1]:v['y'][1:,k]=mean[k*generator.N+cursor:k*generator.N+cursor+n]
        synthetic.append(v);cursor+=n
    m=Model(synthetic);indices=np.array([6,7,11,12,13]);attempts=[]
    def objective(v):
        point=truth.copy();point[indices]=v;residual=m.prediction(point)-m.y
        J=[]
        for index in indices:
            point_complex=point.astype(complex);point_complex[index]+=1e-25j
            from radiative import forward
            J.append(forward(point_complex,m.time,m.u,m.initial,m.ends,m.max_step).imag/1e-25)
        return float(residual@residual),2*np.column_stack(J).T@residual
    for gamma in [0.,1.5]:
        seed=truth[indices].copy();seed[1]=gamma;seed[2:]=0.
        fitrec=minimize(objective,seed,jac=True,method='L-BFGS-B',bounds=[m.bounds()[i] for i in indices],
            options={'maxiter':300,'ftol':1e-15,'gtol':1e-7,'maxls':40})
        error=float(np.max(abs(fitrec.x-truth[indices])))
        assert fitrec.success and error<2e-5,(fitrec.message,error)
        attempts.append({'gamma_seed':gamma,'recovered':fitrec.x.tolist(),'maximum_error':error,'SSE_C2':float(fitrec.fun)})
    full=load_records();zero=Model(full);nr=len(full)
    zero_point=np.r_[fit['best']['point'][:6],np.log(fit['coefficients'][0]),0.,fit['parameters']['log_deviations'],fit['coefficients'][1:1+nr],fit['best']['point'][fit['shape_count']:]]
    zero_mean=zero.prediction(zero_point);arrays=np.load(args.warm.parent/'trajectories.npz')
    saved=np.concatenate([np.concatenate([arrays['predicted_'+str(j)][1:,k] for j in range(nr)]) for k in [0,1]])
    nesting_error=float(np.max(abs(zero_mean-saved)))
    nesting_nll_error=abs(zero.profile_noise(zero_point,zero.y-zero_mean)[0]-fit['best']['nll'])
    assert nesting_error<2e-6 and nesting_nll_error<.001,(nesting_error,nesting_nll_error)
    result={'status':'passed','zero_radiation_full_record_prediction_max_error_C':nesting_error,'zero_radiation_nll_abs_error':nesting_nll_error,'direct_dimensional_ODE_max_error_C':ode_error,'step_halving_max_error_C':step_error,
        'full_profile_gradient_max_scaled_error':derivative_error,'conditional_recovery':attempts,
        'scope':'Radiative balances at one interior point, time-step convergence, complex mean/noise derivatives and noiseless gain/radiation/ambient recovery with dynamics fixed. Does not establish all-parameter identifiability.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
