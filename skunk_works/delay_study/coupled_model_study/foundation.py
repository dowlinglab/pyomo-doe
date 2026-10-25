"""Independent balance integration, modal transfer and data checks."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from scipy.integrate import solve_ivp
from physical import Model,matrices,transfers,load_records
from run_mean import write_json


def independent_temperature(d,shape,K,offsets,lag,curve):
    ta,rs,r,q=np.exp(shape[:4]);Ua=.04;CH=Ua*ta;Us=Ua*r;CS=Us*rs;Ux=Ua*q
    ref=d['y'][0,0];v=np.array([0,0,d['y'][0,1]-ref,d['y'][0,1]-ref]+([0] if lag else []),float)
    initial=v.copy();rows=[v[[1,3]]+ref];duration=d['t'][-1]
    for j,(a,b) in enumerate(zip(d['t'][:-1],d['t'][1:])):
        u=d['u'][j];u=u+shape[-1]*u*(1-u/100) if curve else u
        def rhs(t,x):
            H1,S1,H2,S2=x[:4];ambient=offsets[0]+(offsets[-1]-offsets[0])*t/duration
            power=K*Ua*(x[4] if lag else u)
            cross=Ux*(H2-H1);sensor1=Us*(H1-S1);sensor2=Us*(H2-S2)
            out=[(Ua*(ambient-H1)-sensor1+cross+power)/CH,sensor1/CS,
                 (Ua*(ambient-H2)-sensor2-cross)/CH,sensor2/CS]
            if lag:out.append((u-x[4])/np.exp(shape[4]))
            return out
        solution=solve_ivp(rhs,(a,b),v,rtol=2e-10,atol=2e-11,method='DOP853')
        if not solution.success:raise RuntimeError(solution.message)
        v=solution.y[:,-1];rows.append(v[[1,3]]+ref)
    return np.array(rows)


def audit():
    rng=np.random.RandomState(20261025);cases=[]
    for lag,curve in [(False,False),(True,False),(True,True)]:
        for ambient in ['constant','linear']:
            t=np.r_[0,np.cumsum(rng.uniform(.4,1.4,32))]
            d={'name':'synthetic','t':t,'u':rng.uniform(0,100,len(t)),
               'y':np.tile([22.,22.32],(len(t),1))}
            shape=np.r_[np.log([150.,12.,.4,.35]),[np.log(3.)] if lag else [],[.12] if curve else []]
            model=Model([d],lag,curve,ambient,'iid');X,y,initial=model.design(tuple(shape))
            offsets=[-.7] if ambient=='constant' else [-.7,1.2];coef=np.r_[1.02,offsets]
            predicted=(initial+X@coef).reshape(2,-1).T+d['y'][0,0]
            ode=independent_temperature(d,shape,coef[0],offsets,lag,curve)
            error=float(np.max(abs(predicted-ode[1:])));assert error<2e-7
            A,B,_=matrices(shape,lag)
            frequency_error=max(np.max(abs(transfers(shape,coef[0],w,lag)-np.linalg.solve(1j*w*np.eye(len(A))-A,B)[[1,3]]*coef[0])) for w in np.logspace(-4,1,30))
            assert frequency_error<1e-9
            cases.append({'lag':lag,'curve':curve,'ambient':ambient,'ODE_max_abs_C':error,'frequency_max_abs':float(frequency_error)})
    # Algebraic identifiability in the common/differential chart, without alpha.
    K,ta,rs,r,q=1.02,150.,12.,.4,.35;l=1+2*q
    kp,km=K,K/l;ap=ta+rs*(1+r);am=(ta+rs*(l+r))/l;bp=ta*rs
    recovered_rs=(l*am-ap)/(l-1);recovered_ta=bp/recovered_rs
    recovered=np.array([kp,recovered_ta,recovered_rs,(ap-recovered_ta)/recovered_rs-1,(kp/km-1)/2])
    assert np.max(abs(recovered-[K,ta,rs,r,q]))<1e-10
    worst=-np.inf
    for j in range(30):
        shape=np.log([rng.uniform(20,400),rng.uniform(.1,80),rng.uniform(.01,10),rng.uniform(.01,3)])
        imaginary=np.array([transfers(shape,1.,w)[0].imag for w in np.logspace(-4,2,100)])
        assert np.max(imaginary)<0;worst=max(worst,float(imaginary.max()))
    return {'status':'passed','synthetic_cases':cases,'recovered_scale_free_parameters':recovered.tolist(),
            'phase_constraint':'For positive parameters Im(G+) and Im(G-) are strictly negative for omega>0, so Im(T1/Q1)<0. This analytical bound is checked numerically but does not depend on the sampled parameter points.',
            'largest_base_T1_imaginary_in_synthetic_sweep':worst}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    write_json(args.output/'foundation.json',audit());records=load_records();rows=[]
    for d in records:
        rows.append({'name':d['name'],'raw_sha256':d['sha256'],'n':len(d['t']),
                     'initial_T1_C':float(d['y'][0,0]),'initial_T2_C':float(d['y'][0,1]),
                     'T2_maximum_rise_C':float(d['y'][:,1].max()-d['y'][0,1])})
    write_json(args.output/'data_audit.json',{'records':rows,'status':'passed','notice':'Both channels, same twelve distinct raw records, Q2 identically zero. Initial heater temperatures remain assumed.'})
    print(json.dumps({'status':'passed','synthetic_cases':6,'records':len(rows)}),flush=True)


if __name__=='__main__':main()
