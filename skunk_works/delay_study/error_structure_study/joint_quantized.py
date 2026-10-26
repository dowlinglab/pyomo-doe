"""Joint physical-mean and quantized OU likelihood using smoothing scores."""
from functools import lru_cache
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from bounded import Model as MeanModel
from smoothing import smooth_ou


class Model:
    def __init__(self,records,band=.1,phase=-.5,nodes=128,family='conductance',startup_band=0.):
        self.records=records;self.band=band;self.phase=phase;self.nodes=nodes;self.family=family
        self.nr=len(records);self.N=sum(len(d['t'])-1 for d in records);self.startup_band=startup_band
        self.y=np.concatenate([np.concatenate([d['y'][1:,k] for d in records]) for k in [0,1]])
        if family=='conductance':
            if startup_band:
                from effects import Model as StartupModel
                self.mean_model=StartupModel(records,band,startup_band)
            else:self.mean_model=MeanModel(records,band)
            self.nshape=self.mean_model.ns
            self.ncoef=len(self.mean_model.coefficient_bounds()[0]);self.nmean=self.nshape+self.ncoef
            self.reference=np.concatenate([np.concatenate([np.repeat(d['y'][0,0],len(d['t'])-1) for d in records]) for k in [0,1]])
        elif family=='radiation':
            if startup_band:raise ValueError('Declare and audit combined radiation/startup chart before using it')
            from radiative import Model as RadiationModel
            self.mean_model=RadiationModel(records,band);self.nmean=self.mean_model.nmean;self.nshape=None;self.ncoef=None
        else:raise ValueError('Unknown physical family')
        self.evaluate=lru_cache(maxsize=1)(self._evaluate)

    def bounds(self):
        if self.family=='conductance':
            low,high=self.mean_model.coefficient_bounds()
            mean=self.mean_model.shape_bounds()+[(np.log(low[0]),np.log(high[0]))]+list(zip(low[1:],high[1:]))
        else:mean=self.mean_model.bounds()[:self.nmean]
        return mean+[(np.log(.002),np.log(3.)),(np.log(.05),np.log(3000.)),(np.log(.0001),np.log(1.))]*2

    def centered_constraint(self,p):
        start=6 if self.family=='conductance' else 8
        count=self.mean_model.nv if self.family=='conductance' else self.nr
        return float(np.sum(p[start:start+count]))

    def centered_jacobian(self,p):
        start=6 if self.family=='conductance' else 8
        count=self.mean_model.nv if self.family=='conductance' else self.nr
        a=np.zeros(len(p));a[start:start+count]=1.
        return a

    def mean(self,p,jacobian=True):
        if self.family=='conductance':
            shape=p[:self.nshape];coef=np.r_[np.exp(p[self.nshape]),p[self.nshape+1:self.nmean]]
            X,_,initial=self.mean_model.design(tuple(shape));m=self.reference+initial+X@coef
            if not jacobian:return m
            J=np.zeros((2*self.N,self.nmean));bounds=self.bounds();h=2e-5
            for j in range(self.nshape):
                d=np.zeros(self.nshape);d[j]=h;lo,hi=bounds[j]
                def prediction(q):
                    x,_,i=self.mean_model.design(tuple(q));return self.reference+i+x@coef
                if shape[j]-lo>=h and hi-shape[j]>=h:J[:,j]=(prediction(shape+d)-prediction(shape-d))/(2*h)
                elif hi-shape[j]<h:J[:,j]=(3*m-4*prediction(shape-d)+prediction(shape-2*d))/(2*h)
                else:J[:,j]=(-3*m+4*prediction(shape+d)-prediction(shape+2*d))/(2*h)
            J[:,self.nshape]=X[:,0]*coef[0];J[:,self.nshape+1:]=X[:,1:]
            return m,J
        m=self.mean_model.prediction(p[:self.nmean])
        if not jacobian:return m
        from radiative import forward
        J=np.zeros((2*self.N,self.nmean))
        for j in range(self.nmean):
            q=np.asarray(p[:self.nmean],dtype=complex);q[j]+=1e-25j
            J[:,j]=forward(q,self.mean_model.time,self.mean_model.u,self.mean_model.initial,self.mean_model.ends,self.mean_model.max_step).imag/1e-25
        return m,J

    def _evaluate(self,key):
        p=np.asarray(key);mean,J=self.mean(p)
        mean_score=np.zeros(2*self.N);gradient=np.zeros(len(p));nll=0.;details=[]
        for channel in [0,1]:
            latent,ell,measurement=np.exp(p[self.nmean+3*channel:self.nmean+3*channel+3]);cursor=0
            noise_score=np.zeros(3)
            for j,d in enumerate(self.records):
                n=len(d['t'])-1;section=slice(channel*self.N+cursor,channel*self.N+cursor+n);nodes=self.nodes
                while True:
                    f=smooth_ou(d['t'][1:],mean[section],d['y'][1:,channel],latent,ell,measurement,self.phase,nodes)
                    error=float(np.max(abs(f['posterior_normalization_ratio']-1)))
                    if error<1e-6 and f['smoothing_closure_error']<1e-6:break
                    if nodes>=512:
                        bad=int(np.argmax(abs(f['posterior_normalization_ratio']-1)))
                        raise FloatingPointError('Joint posterior integration failed: '+str({'name':d['name'],'channel':channel+1,'nodes':nodes,'normalization_error':error,'smoothing_closure':f['smoothing_closure_error'],'time':float(d['t'][bad+1]),'log_bin_mass':float(f['log_bin_probability'][bad])}))
                    nodes*=2
                nll+=f['nll'];mean_score[section]=f['mean_nll_gradient'];noise_score+=f['noise_nll_gradient_logsd_logell_logsd']
                details.append({'name':d['name'],'channel':channel+1,'nodes':nodes,'nll':f['nll'],
                    'normalization_error':error,'smoothing_closure_error':f['smoothing_closure_error']})
                cursor+=n
            gradient[self.nmean+3*channel:self.nmean+3*channel+3]=noise_score
        gradient[:self.nmean]=mean_score@J
        return float(nll),gradient,details

    def objective(self,p):return self.evaluate(tuple(p))[0]
    def gradient(self,p):return self.evaluate(tuple(p))[1]
    def details(self,p):return self.evaluate(tuple(p))[2]
