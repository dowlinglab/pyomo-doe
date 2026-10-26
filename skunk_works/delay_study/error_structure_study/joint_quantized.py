"""Joint physical-mean and quantized OU likelihood using smoothing scores."""
from functools import lru_cache
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from bounded import Model as MeanModel
from smoothing import smooth_ou
from conditional_initial import smooth_record


class Model:
    def __init__(self,records,band=.1,phase=-.5,nodes=128,family='conductance',startup_band=0.,fixed=None,initial_sensor_bins=False,initial_noise_buffer=0.,minimum_sensor_loading=.005,condition_initial_code=False,score_backend='reference'):
        self.records=records;self.band=band;self.phase=phase;self.nodes=nodes;self.family=family
        self.nr=len(records);self.N=sum(len(d['t'])-1 for d in records);self.startup_band=startup_band
        self.fixed=dict(fixed or {})
        self.initial_sensor_bins=initial_sensor_bins;self.initial_noise_buffer=initial_noise_buffer
        if not 0<minimum_sensor_loading<30:raise ValueError('Positive sensor loading minimum below 30 required')
        self.minimum_sensor_loading=minimum_sensor_loading
        self.condition_initial_code=condition_initial_code
        if score_backend not in ['reference','compiled']:raise ValueError('Unknown score backend')
        self.score_backend=score_backend
        if initial_noise_buffer and not initial_sensor_bins:raise ValueError('An initial-noise buffer requires sensor-bin sensitivity')
        self.y=np.concatenate([np.concatenate([d['y'][1:,k] for d in records]) for k in [0,1]])
        if family=='conductance':
            if initial_sensor_bins:
                if startup_band:raise ValueError('Declare a separate combined hidden-heater/sensor-start sensitivity')
                from initial_sensor import Model as InitialSensorModel
                self.mean_model=InitialSensorModel(records,band,phase,initial_noise_buffer)
            elif startup_band:
                from effects import Model as StartupModel
                self.mean_model=StartupModel(records,band,startup_band)
            else:self.mean_model=MeanModel(records,band)
            self.nshape=self.mean_model.ns
            self.ncoef=len(self.mean_model.coefficient_bounds()[0]);self.nmean=self.nshape+self.ncoef
            self.reference=np.concatenate([np.concatenate([np.repeat(d['y'][0,0],len(d['t'])-1) for d in records]) for k in [0,1]])
        elif family=='radiation':
            if startup_band or initial_sensor_bins:raise ValueError('Declare and audit combined radiation/startup chart before using it')
            from radiative import Model as RadiationModel
            self.mean_model=RadiationModel(records,band);self.nmean=self.mean_model.nmean;self.nshape=None;self.ncoef=None
        else:raise ValueError('Unknown physical family')
        self.evaluate=lru_cache(maxsize=1)(self._evaluate)

    def bounds(self):
        if self.family=='conductance':
            low,high=self.mean_model.coefficient_bounds()
            mean=self.mean_model.shape_bounds()+[(np.log(low[0]),np.log(high[0]))]+list(zip(low[1:],high[1:]))
        else:mean=self.mean_model.bounds()[:self.nmean]
        bounds=mean+[(np.log(.002),np.log(3.)),(np.log(.05),np.log(3000.)),(np.log(.0001),np.log(1.))]*2
        bounds[2]=(np.log(self.minimum_sensor_loading),bounds[2][1])
        names=self.coordinate_names()
        for name,value in self.fixed.items():
            if name not in names:raise ValueError('Unknown fixed coordinate '+name)
            j=names.index(name);lo,hi=bounds[j]
            if not lo<=value<=hi:raise ValueError('Fixed coordinate outside declared bounds '+name)
            bounds[j]=(float(value),float(value))
        return bounds

    def coordinate_names(self):
        shared=['log_tA','log_rS','log_Us_over_Ua','log_Ux_over_Ua','log_input_lag','command_curvature']
        deviations=['log_Ua_ratio_'+str(j) for j in range(self.nr)]
        ambient=['ambient_offset_'+str(j) for j in range(self.nr)]
        if self.family=='conductance':
            names=shared+(deviations if self.band else [])+['log_K']+ambient
            if self.startup_band:names+=['initial_H'+str(k+1)+'_offset_'+str(j) for j in range(self.nr) for k in [0,1]]
            if self.initial_sensor_bins:names+=['initial_sensor'+str(k+1)+'_offset_'+str(j) for j in range(self.nr) for k in [0,1]]
        else:names=shared+['log_K','radiation_gamma']+deviations+ambient
        return names+['log_latent_SD_T1','log_ell_T1','log_measurement_SD_T1',
                      'log_latent_SD_T2','log_ell_T2','log_measurement_SD_T2']

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
            if jacobian:
                from sensitivity import mean_and_jacobian
                return mean_and_jacobian(self,p)
            shape=p[:self.nshape];coef=np.r_[np.exp(p[self.nshape]),p[self.nshape+1:self.nmean]]
            X,_,initial=self.mean_model.design(tuple(shape));m=self.reference+initial+X@coef
            return m
        m=self.mean_model.prediction(p[:self.nmean])
        if not jacobian:return m
        from radiative import forward
        J=np.zeros((2*self.N,self.nmean))
        for j in range(self.nmean):
            q=np.asarray(p[:self.nmean],dtype=complex);q[j]+=1e-25j
            J[:,j]=forward(q,self.mean_model.time,self.mean_model.u,self.mean_model.initial,self.mean_model.ends,self.mean_model.max_step).imag/1e-25
        return m,J

    def initial_mean_and_index(self,p,record,channel):
        value=float(self.records[record]['y'][0,channel]);index=None
        if self.initial_sensor_bins:
            index=self.nshape+1+self.nr+2*record+channel;value+=p[index]
        return value,index

    def _evaluate(self,key):
        p=np.asarray(key);mean,J=self.mean(p)
        if self.score_backend=='compiled':
            from fast_smoothing import smooth_record as score_record
        else:score_record=smooth_record
        mean_score=np.zeros(2*self.N);gradient=np.zeros(len(p));initial_score=np.zeros(self.nmean);nll=0.;details=[]
        for channel in [0,1]:
            latent,ell,measurement=np.exp(p[self.nmean+3*channel:self.nmean+3*channel+3]);cursor=0
            noise_score=np.zeros(3)
            for j,d in enumerate(self.records):
                n=len(d['t'])-1;section=slice(channel*self.N+cursor,channel*self.N+cursor+n);nodes=self.nodes
                first,first_index=self.initial_mean_and_index(p,j,channel);full_mean=np.r_[first,mean[section]]
                while True:
                    f=score_record(d,full_mean,channel,latent,ell,measurement,self.phase,nodes,self.condition_initial_code)
                    error=float(np.max(abs(f['posterior_normalization_ratio']-1)))
                    if error<1e-6 and f['smoothing_closure_error']<1e-6:break
                    if nodes>=512:
                        bad=int(np.argmax(abs(f['posterior_normalization_ratio']-1)))
                        index=bad if self.condition_initial_code else bad+1
                        raise FloatingPointError('Joint posterior integration failed: '+str({'name':d['name'],'channel':channel+1,'nodes':nodes,'normalization_error':error,'smoothing_closure':f['smoothing_closure_error'],'time':float(d['t'][index]),'log_bin_mass':float(f['log_bin_probability'][bad])}))
                    nodes*=2
                nll+=f['nll'];mean_score[section]=f['mean_nll_gradient'];noise_score+=f['noise_nll_gradient_logsd_logell_logsd']
                if first_index is not None:initial_score[first_index]+=f['initial_mean_nll_gradient']
                details.append({'name':d['name'],'channel':channel+1,'nodes':nodes,'nll':f['nll'],
                    'normalization_error':error,'smoothing_closure_error':f['smoothing_closure_error']})
                cursor+=n
            gradient[self.nmean+3*channel:self.nmean+3*channel+3]=noise_score
        gradient[:self.nmean]=mean_score@J+initial_score
        return float(nll),gradient,details

    def objective(self,p):return self.evaluate(tuple(p))[0]
    def gradient(self,p):return self.evaluate(tuple(p))[1]
    def details(self,p):return self.evaluate(tuple(p))[2]
