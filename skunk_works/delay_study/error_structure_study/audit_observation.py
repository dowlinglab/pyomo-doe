"""Selected hardware noise estimates: direct trivariate rectangles and finer grids."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import nquad
from scipy.stats import multivariate_normal
from quantized import bins,filter_ou


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--mean-fit',type=Path,required=True)
    args=parser.parse_args();fit=json.loads((args.input/'observation_fit.json').read_text())
    assert fit['all_selected_quadrature_converged']
    for name,key in [('fit.json','mean_fit_sha256'),('trajectories.npz','mean_arrays_sha256')]:
        assert hashlib.sha256((args.mean_fit/name).read_bytes()).hexdigest()==fit[key]
    a=np.load(args.mean_fit/'trajectories.npz');c=fit['channel']-1;p=fit['parameters'];phase=fit['phase']
    sigma=p['sigma_total_analogue_C'];eta=p['independent_variance_fraction'];ell=p['ell_s'];rows=[];total=0.
    for j,old in enumerate(fit['per_run']):
        t=a['time_'+str(j)][1:];mean=a['predicted_'+str(j)][1:,c];y=a['observed_'+str(j)][1:,c]
        fine=filter_ou(t,mean,y,sigma,ell,eta,phase,nodes=512);difference=abs(fine['nll']-old['bin_nll']);total+=fine['nll']
        norm=float(np.max(abs(fine['posterior_normalization_ratio']-1)))
        assert difference<1e-4 and norm<1e-6,(j,difference,norm)
        tt=t[:3];C=sigma*sigma*((1-eta)*np.exp(-abs(tt[:,None]-tt[None,:])/ell)+eta*np.eye(3))
        density=multivariate_normal(mean=mean[:3],cov=C)
        lo,hi=bins(y[:3],phase)
        mass,error=nquad(lambda z0,z1,z2:density.pdf([z0,z1,z2]),list(zip(lo,hi)),opts={'epsabs':1e-11,'epsrel':1e-8})
        reference=-np.log(mass);direct=filter_ou(tt,mean[:3],y[:3],sigma,ell,eta,phase,nodes=512)['nll']
        rect_error=abs(reference-direct);assert rect_error<3e-5,(j,reference,direct,rect_error)
        rows.append({'name':old['name'],'fine_grid_nll_difference':difference,'normalization_error':norm,
                     'direct_three_point_nll_error':rect_error,'rectangle_quadrature_reported_error':error})
    result={'status':'passed','channel':fit['channel'],'full_fine_grid_bin_nll':total,
        'total_nll_difference':abs(total-fit['validated_bin_nll']),'per_run':rows,
        'input_sha256':hashlib.sha256((args.input/'observation_fit.json').read_bytes()).hexdigest(),
        'scope':'Full-record 512-node score/normalization checks, and independent trivariate observation-covariance rectangle integrals for each run prefix at selected parameters. Not a second optimizer, global optimum proof, mean refit or inferential coverage calibration.'}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','channel':fit['channel'],'max_direct_rectangle_error':max(v['direct_three_point_nll_error'] for v in rows),'nll_difference':result['total_nll_difference']}),flush=True)


if __name__=='__main__':main()
