"""Exact ZOH linear-mean sensitivities from matrix-exponential Frechet derivatives.

Physical rates are assembled directly in shared nominal coordinates. The
seven-state augmentation holds heater command and ambient constant between
updates. This avoids differencing eigensystems in weak-loading directions.
"""
import numpy as np
from scipy.linalg import expm,expm_frechet
from numba import njit


@njit(cache=True)
def propagate(time,u,indices,transition,derivative,state,sensitivity,curvature,curve_local):
    n=len(time);npar=sensitivity.shape[1];out=np.empty((n-1,2));J=np.empty((n-1,2,npar))
    for j in range(n-1):
        state[5]=u[j]+curvature*u[j]*(1-u[j]/100.)
        sensitivity[5,:]=0.
        sensitivity[5,curve_local]=u[j]*(1-u[j]/100.)
        M=transition[indices[j]];D=derivative[indices[j]]
        next_sensitivity=M@sensitivity
        for k in range(npar):next_sensitivity[:,k]+=D[k]@state
        state=M@state;sensitivity=next_sensitivity
        out[j,0]=state[1];out[j,1]=state[3]
        J[j,0,:]=sensitivity[1,:];J[j,1,:]=sensitivity[3,:]
    return out,J


def generator(point,run_delta,log_gain):
    ta,rs,r,q,L=np.exp(point[:5]);g=np.exp(run_delta);K=np.exp(log_gain)
    A=np.zeros((7,7));D=np.zeros((7,7,7))
    for h,s,other in [(0,1,2),(2,3,0)]:
        A[h,h]=-(g+r+q)/ta;A[h,s]=r/ta;A[h,other]=q/ta;A[h,6]=g/ta
        A[s,h]=1/rs;A[s,s]=-1/rs
        D[1,s,:]=-A[s,:]
        D[2,h,h]=-r/ta;D[2,h,s]=r/ta
        D[3,h,h]=-q/ta;D[3,h,other]=q/ta
        D[5,h,h]=-g/ta;D[5,h,6]=g/ta
    A[0,4]=K/ta;A[4,4]=-1/L;A[4,5]=1/L
    D[0,[0,2],:]=-A[[0,2],:];D[4,4,:]=-A[4,:];D[6,0,4]=K/ta
    return A,D


def mean_and_jacobian(model,point):
    assert model.family=='conductance'
    p=np.asarray(point);m=model;means=[];jacobians=[];cursor=0
    nmean=m.nmean
    for j,d in enumerate(m.records):
        delta=p[6+j] if m.band else 0.;A,local_D=generator(p,delta,p[m.nshape])
        global_indices=[0,1,2,3,4,5]+([6+j] if m.band else [])+[m.nshape,m.nshape+1+j]
        global_D=list(local_D[:5])+[np.zeros((7,7))]+([local_D[5]] if m.band else [])+[local_D[6],np.zeros((7,7))]
        state=np.zeros(7);state[2:4]=d['y'][0,1]-d['y'][0,0];state[6]=p[m.nshape+1+j]
        if m.startup_band or m.initial_sensor_bins:
            global_indices += [m.nshape+1+m.nr+2*j+k for k in [0,1]]
            global_D += [np.zeros((7,7)),np.zeros((7,7))]
        S=np.zeros((7,len(global_indices)));S[6,global_indices.index(m.nshape+1+j)]=1.
        for k in [0,1]:
            if m.startup_band or m.initial_sensor_bins:
                index=m.nshape+1+m.nr+2*j+k;column=global_indices.index(index)
                targets=[2*k] if m.startup_band else [2*k,2*k+1]
                state[targets]+=p[index];S[targets,column]=1.
        intervals=np.unique(np.diff(d['t']));indices=np.searchsorted(intervals,np.diff(d['t']))
        transition=[];derivatives=[]
        for dt in intervals:
            transition.append(expm(A*dt));derivatives.append(np.array([
                expm_frechet(A*dt,v*dt,compute_expm=False) if np.any(v) else np.zeros((7,7)) for v in global_D]))
        out,J=propagate(d['t'],d['u'],indices,np.ascontiguousarray(transition),np.ascontiguousarray(derivatives),
            state,S,p[5],global_indices.index(5))
        out+=d['y'][0,0];means.append(out)
        full=np.zeros((len(out),2,nmean));full[:,:,global_indices]=J;jacobians.append(full)
    mean=np.concatenate([np.concatenate([v[:,k] for v in means]) for k in [0,1]])
    J=np.concatenate([np.concatenate([v[:,k,:] for v in jacobians]) for k in [0,1]])
    return mean,J
