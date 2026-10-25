"""Convergence and randomized predictive residuals for saved-model bin likelihoods."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from covariance import Whitening
from quantized import STEP,filter_ou,randomized_residual
from decompose import diagnostic


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fit',type=Path,required=True);parser.add_argument('--records',type=int,nargs='+',default=[0,6,9])
    parser.add_argument('--nodes',type=int,nargs='+',default=[64,128,256]);parser.add_argument('--phases',type=float,nargs='+',default=[-.5,0.])
    args=parser.parse_args();fit=json.loads((args.fit/'fit.json').read_text());a=np.load(args.fit/'trajectories.npz')
    results=[];saved={};count=0
    for j in args.records:
        t=a['time_'+str(j)][1:];y=a['observed_'+str(j)][1:];mean=a['predicted_'+str(j)][1:]
        for c in [0,1]:
            sigma=fit['sigma1_C'] if c==0 else fit['sigma2_C'];ns=fit['shape_count']
            ell=np.exp(fit['best']['point'][ns+2*c]);eta=fit['best']['point'][ns+2*c+1]
            w=Whitening([t],'ou_nugget',[ell,eta]);v=w(y[:,c]-mean[:,c])/sigma
            density_nll=.5*(v@v+w.logdet+2*len(t)*np.log(sigma)+len(t)*np.log(2*np.pi))
            for phase in args.phases:
                filters={}
                for nodes in args.nodes:
                    f=filter_ou(t,mean[:,c],y[:,c],sigma,ell,eta,phase,nodes)
                    filters[nodes]=f
                    row={'name':fit['record_names'][j],'record_index':j,'channel':c+1,'phase':phase,'nodes':nodes,
                        'bin_nll':f['nll'],'posterior_normalization_max_abs_error':float(np.max(abs(f['posterior_normalization_ratio']-1))),
                        'continuous_density_nll':float(density_nll),'density_nll_minus_N_log_step':float(density_nll-len(t)*np.log(STEP))}
                    results.append(row);print(json.dumps(row),flush=True)
                keys=sorted(filters);difference=abs(filters[keys[-1]]['nll']-filters[keys[-2]]['nll'])
                last=filters[keys[-1]];normalization=float(np.max(abs(last['posterior_normalization_ratio']-1)))
                converged=difference<1e-4 and normalization<1e-5
                diag=[]
                for seed in [10371,10372,10373,10374,10375]:
                    z=randomized_residual(last,seed)
                    diag.append({'seed':seed,**diagnostic(z,[1,5,10,20,60])})
                    saved['PIT_'+str(count)+'_'+str(seed)]=z
                for key in ['log_bin_probability','cdf_lower','cdf_upper','posterior_mean','posterior_SD']:
                    saved[key+'_'+str(count)]=last[key]
                results.append({'case':count,'name':fit['record_names'][j],'record_index':j,'channel':c+1,'phase':phase,
                    'quadrature_converged':converged,'largest_two_node_nll_difference':difference,'nodes_used':keys[-1],
                    'normalization_error':normalization,'PIT_diagnostics':diag})
                count+=1
                (args.output/'progress.json').write_text(json.dumps({'completed_cases':count,'latest_name':fit['record_names'][j],'latest_channel':c+1,'latest_phase':phase})+'\n')
    result={'fit_sha256':hashlib.sha256((args.fit/'fit.json').read_bytes()).hexdigest(),
        'arrays_sha256':hashlib.sha256((args.fit/'trajectories.npz').read_bytes()).hexdigest(),
        'rows':results,'all_quadrature_converged':all(r['quadrature_converged'] for r in results if 'quadrature_converged' in r),
        'notice':'Mean and OU covariance fixed at continuous-data estimates. Bin phase is assumed. PMF likelihood differs in measure from density likelihood; density minus N log step is only a small-bin approximation. No noise refit, parameter uncertainty or hardware-noise adequacy claim.'}
    (args.output/'observation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output/'predictive_arrays.npz',**saved)
    print(json.dumps({'status':'complete','cases':count,'all_quadrature_converged':result['all_quadrature_converged']}),flush=True)


if __name__=='__main__':main()
