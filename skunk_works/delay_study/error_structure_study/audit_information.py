"""Independent information arithmetic, dimensional sensitivities and local SEs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.linalg import cholesky,solve_triangular,eigh,null_space,cho_factor,cho_solve
from physical import load_records
from joint_quantized import Model
from quantized import filter_ou
from audit_common import dimensional_mean
from run_bounded import write


def conditional_covariance(information,basis):
    small=basis.T@information@basis;small=(small+small.T)/2
    return basis@cho_solve(cho_factor(small,lower=True),np.eye(small.shape[0]))@basis.T


def basis_invariants(basis,centering,active,conditional):
    """Check the complete null space, including an empty active-bound set."""
    basis=np.asarray(basis);centering=np.asarray(centering);active=np.asarray(active,dtype=bool)
    n=len(centering);assert basis.ndim==2 and basis.shape[0]==n and active.shape==(n,)
    rows=[centering] if np.any(centering) else []
    if conditional:rows += list(np.eye(n)[active])
    rank=np.linalg.matrix_rank(np.array(rows)) if rows else 0
    assert basis.shape[1]==n-rank,'Constraint basis omits or adds feasible directions'
    orth=float(np.max(abs(basis.T@basis-np.eye(basis.shape[1])))) if basis.shape[1] else 0.
    centered=float(np.max(abs(centering@basis))) if basis.shape[1] else 0.
    active_error=float(np.max(abs(basis[active]))) if conditional and np.any(active) and basis.shape[1] else 0.
    assert orth<1e-12 and centered<1e-12 and active_error<1e-12
    return {'orthogonality_error':orth,'centering_error':centered,
        'active_bound_restriction_error':active_error,'constraint_rank':int(rank),
        'expected_dimension':n-int(rank),'active_bound_count':int(np.sum(active)) if conditional else 0}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--fit',type=Path,required=True)
    args=parser.parse_args();info=json.loads((args.input/'information.json').read_text());fit=json.loads((args.fit/'fit.json').read_text())
    assert info['fit_sha256']==hashlib.sha256((args.fit/'fit.json').read_bytes()).hexdigest()
    assert json.loads((args.input/'exit.json').read_text())['exit_code']==0
    assert fit['family']=='conductance' and not fit.get('dataset') and not fit.get('fixed_coordinates')
    assert not fit.get('startup_band_C') and not fit.get('initial_sensor_bins') and not fit.get('condition_initial_code')
    lookup={d['name']:d for d in load_records()};records=[lookup[n] for n in fit['record_names']]
    assert all(fit['raw_sha256'][d['name']]==d['sha256'] for d in records)
    m=Model(records,fit['band'],fit['phase'],128,'conductance',minimum_sensor_loading=fit.get('minimum_sensor_loading',.005))
    p=np.array(info['point']);assert np.array_equal(p,np.array(fit['best']['point']))
    names=m.coordinate_names();assert names==info['coordinate_names'];mat=np.load(args.input/'information_matrices.npz')
    H=mat['observed_Hessian'];raw=mat['raw_Hessian'];Q=mat['centered_basis'];F=mat['active_face_basis'];J=mat['mean_Jacobian']
    assert np.max(abs(H-(raw+raw.T)/2))<1e-12 and J.shape==(2*m.N,m.nmean)
    a=m.centered_jacobian(p);bounds=np.array(m.bounds());active=(p-bounds[:,0]<1e-5)|(bounds[:,1]-p<1e-5)
    basis_checks={}
    for label,basis in [('centered_tangent',Q),('conditional_active_face',F)]:
        invariants=basis_invariants(basis,a,active,label=='conditional_active_face')
        B=basis.T@H@basis;values,vectors=eigh((B+B.T)/2,driver='evr')
        difference=float(np.max(abs(values-info[label]['eigenvalues'])/(1+abs(values))));assert difference<1e-7
        residual=float(np.linalg.norm(B@vectors-vectors*values,ord='fro')/np.linalg.norm(B));assert residual<1e-12
        basis_checks[label]={**invariants,
            'relative_eigenpair_residual':residual,'scaled_eigenvalue_reconstruction_error':difference}
    # Compare stored normalized-model sensitivities with central differences of
    # independently assembled dimensional heat-flux matrices over every sample.
    def coefficients(point):return np.r_[np.exp(point[m.nshape]),point[m.nshape+1:m.nmean]]
    directions=[]
    for index in [0,m.nshape]:
        v=np.zeros(len(p));v[index]=1.;directions.append(v)
    if fit['band']:
        v=np.zeros(len(p));v[6]=1/np.sqrt(2);v[7]=-1/np.sqrt(2);directions.append(v)
    B=F.T@H@F;ev,vec=eigh((B+B.T)/2,driver='evr');directions.append(F@vec[:,0])
    derivative_checks=[];h=1e-4
    for j,v in enumerate(directions):
        high=p+h*v;low=p-h*v
        fd=(dimensional_mean(high,coefficients(high),records,fit['band'])-dimensional_mean(low,coefficients(low),records,fit['band']))/(2*h)
        expected=J@v[:m.nmean];error=float(np.max(abs(fd-expected)/(1+abs(fd))));assert error<5e-5,(j,error)
        derivative_checks.append({'direction':j,'scaled_dimensional_sensitivity_error':error})
    analogue=np.zeros((m.nmean,m.nmean));iid=np.zeros_like(analogue)
    for k,noise in enumerate(fit['noise_parameters']):
        S=noise['latent_SD_C']**2;R=noise['measurement_SD_C']**2;ell=noise['ell_s'];cursor=0
        for d in records:
            t=d['t'][1:];n=len(t);section=slice(k*m.N+cursor,k*m.N+cursor+n);x=J[section]
            C=S*np.exp(-abs(t[:,None]-t[None,:])/ell)+R*np.eye(n)
            w=solve_triangular(cholesky(C,lower=True),x,lower=True);analogue+=w.T@w;iid+=x.T@x/(S+R);cursor+=n
    analogue_error=float(np.linalg.norm(analogue-mat['analogue_mean_known_noise'])/np.linalg.norm(analogue))
    iid_error=float(np.linalg.norm(iid-mat['iid_mean_same_marginal_variance'])/np.linalg.norm(iid))
    assert analogue_error<1e-9 and iid_error<1e-10
    # Nominal/deviation centering removes an exact chart-reference freedom.
    # This algebraic test is off the constrained face, not a feasible CI move.
    gauge=np.zeros(len(p));gauge[[0,2,3,m.nshape]]=1.
    if fit['band']:gauge[6:m.nshape]=1.
    gauge_error=None
    if fit['band']:
        before=dimensional_mean(p,coefficients(p),records,fit['band']);after=p+.001*gauge
        gauge_error=float(np.max(abs(before-dimensional_mean(after,coefficients(after),records,fit['band']))));assert gauge_error<1e-8
    # Direct scalar likelihood differences use forward filtering, independent
    # of the smoothing-score Hessian. Larger one-sided steps resolve tiny
    # loading curvature without subtracting nearly equal whole-data scores.
    cache={}
    def forward_nll(loading_step):
        if loading_step in cache:return cache[loading_step]
        q=p.copy();q[2]+=loading_step;mean=dimensional_mean(q,coefficients(q),records,fit['band']);value=0.
        for k,noise in enumerate(fit['noise_parameters']):
            S=noise['latent_SD_C']**2;R=noise['measurement_SD_C']**2;cursor=0
            for d in records:
                n=len(d['t'])-1;s=slice(k*m.N+cursor,k*m.N+cursor+n)
                f=filter_ou(d['t'][1:],mean[s],d['y'][1:,k],np.sqrt(S+R),noise['ell_s'],R/(S+R),fit['phase'],256,12.)
                assert np.max(abs(f['posterior_normalization_ratio']-1))<1e-6
                value+=f['nll'];cursor+=n
        cache[loading_step]=value;return value
    f0=forward_nll(0.);loading_checks=[]
    # The saved scalar step study resolves cancellation at tiny steps and
    # truncation bias at 0.05. Use its two stable smaller stencils, without
    # relaxing the original 0.001 absolute curvature agreement gate.
    for step in [.005,.02]:
        curvature=(2*f0-5*forward_nll(step)+4*forward_nll(2*step)-forward_nll(3*step))/step**2
        error=float(abs(curvature-H[2,2]));assert error<.001,(step,curvature,H[2,2],error)
        loading_checks.append({'step':step,'scalar_one_sided_curvature':float(curvature),'stored_H22':float(H[2,2]),'absolute_error':error})
    # Nonpositive observed curvature cannot generate a local-normal covariance.
    # Retain the independent checks rather than failing with only a Cholesky
    # exception or changing the fitted active set to force positive curvature.
    if ev[0]<=0:
        write(args.output/'audit.json',{'status':'failed','failure_kind':'nonpositive_observed_curvature',
            'fit_sha256':info['fit_sha256'],
            'information_sha256':hashlib.sha256((args.input/'information.json').read_bytes()).hexdigest(),
            'matrix_sha256':hashlib.sha256((args.input/'information_matrices.npz').read_bytes()).hexdigest(),
            'basis_and_eigen_checks':basis_checks,'independent_dimensional_derivative_checks':derivative_checks,
            'dense_analogue_information_relative_error':analogue_error,'IID_information_relative_error':iid_error,
            'nominal_deviation_chart_gauge_mean_error_C':gauge_error,'loading_scalar_curvature_checks':loading_checks,
            'active_coordinates':[names[j] for j in np.flatnonzero(active)],
            'minimum_observed_active_face_eigenvalue':float(ev[0]),
            'notice':'Observed curvature is nonpositive at the saved accepted fit. No local covariance or parameter draw is accepted. Investigate the fit and weak direction without snapping bounds, changing tolerances or clipping eigenvalues.'})
        raise RuntimeError('Nonpositive observed curvature; local-normal uncertainty is unsupported at this fit')
    # Conditional local comparisons include all permitted mean nuisances and
    # freeze exactly the detected active bounds. They are not calibrated CIs.
    mean_constraints=[a[:m.nmean]] if np.any(a[:m.nmean]) else []
    mean_constraints += [np.eye(m.nmean)[j] for j in np.flatnonzero(active[:m.nmean])]
    face=null_space(np.array(mean_constraints)) if mean_constraints else np.eye(m.nmean)
    cov_iid=conditional_covariance(iid,face);cov_analogue=conditional_covariance(analogue,face);cov_bin=conditional_covariance(H,F)
    rows=[]
    for name in ['log_K','log_tA','log_rS','log_input_lag','command_curvature']:
        j=names.index(name);s0=np.sqrt(cov_iid[j,j]);s1=np.sqrt(cov_analogue[j,j]);s2=np.sqrt(cov_bin[j,j])
        rows.append({'coordinate':name,'IID_known_analogue_variance_SE':float(s0),'correlated_known_analogue_covariance_SE':float(s1),
            'bin_observed_mean_and_noise_nuisance_SE':float(s2),'correlated_to_IID_analogue_SE_ratio':float(s1/s0)})
    with (args.output/'local_SE_comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    diffusion=[]
    for k,noise in enumerate(fit['noise_parameters']):
        j=m.nmean+3*k;v=np.zeros(len(p));v[j]=2.;v[j+1]=-1.
        diffusion.append({'channel':k+1,'latent_diffusion_variance_rate_C2_per_s':2*noise['latent_SD_C']**2/noise['ell_s'],
            'conditional_local_log_diffusion_SE':float(np.sqrt(v@cov_bin@v)),
            'conditional_local_log_latent_SD_SE':float(np.sqrt(cov_bin[j,j])),
            'conditional_local_log_length_SE':float(np.sqrt(cov_bin[j+1,j+1])),
            'conditional_log_SD_log_length_correlation':float(cov_bin[j,j+1]/np.sqrt(cov_bin[j,j]*cov_bin[j+1,j+1]))})
    result={'status':'passed','fit_sha256':info['fit_sha256'],
        'information_sha256':hashlib.sha256((args.input/'information.json').read_bytes()).hexdigest(),
        'matrix_sha256':hashlib.sha256((args.input/'information_matrices.npz').read_bytes()).hexdigest(),
        'basis_and_eigen_checks':basis_checks,'independent_dimensional_derivative_checks':derivative_checks,
        'dense_analogue_information_relative_error':analogue_error,'IID_information_relative_error':iid_error,
        'nominal_deviation_chart_gauge_mean_error_C':gauge_error,'loading_scalar_curvature_checks':loading_checks,
        'active_coordinates':[names[j] for j in np.flatnonzero(active)],'local_comparison':rows,'diffusion':diffusion,
        'scope':'Full-data dimensional sensitivity directions, independent dense covariance information, basis/eigenpair arithmetic, chart reference symmetry and forward-likelihood loading curvature. SEs condition on active bounds and the selected basin; no regular covariance for boundary loading, global profile, unknown thresholds, model selection or empirical coverage is claimed.'}
    write(args.output/'audit.json',result);print(json.dumps({'status':'passed','dense_information_error':analogue_error,'local_comparison':rows,'diffusion':diffusion}),flush=True)


if __name__=='__main__':main()
