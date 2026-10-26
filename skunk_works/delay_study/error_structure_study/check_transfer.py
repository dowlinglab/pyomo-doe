"""Independent means, no test-output leakage, discrete expectations and mixtures."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coupled_model_study'))
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from physical import load_records
from audit_bounded import independent
from quantized import STEP,OFFSET,bins,report
from transfer import checked_scenarios,reported_mean,reported_quantile,mixture_quantile
from run_transfer_fit import folds
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--warm',type=Path,required=True)
    args=parser.parse_args();fit=json.loads(args.warm.read_text());original=load_records()[0]
    d=dict(original,t=original['t'][:81],y=original['y'][:81].copy(),u=original['u'][:81]);shape=np.array(fit['best']['point'][:6]);K=fit['parameters']['K_common_nominal']
    config={'coarse_points':41,'fine_points':81,'refinement_points':161,'maximum_envelope_change_C':1e-4}
    scenario=checked_scenarios(d,shape,K,.1,config)
    altered=dict(d,y=d['y'].copy());altered['y'][1:]=np.random.RandomState(3117).normal(50,25,altered['y'][1:].shape)
    same=checked_scenarios(altered,shape,K,.1,config);leakage_error=max(float(np.max(abs(scenario[k]-same[k]))) for k in ['nominal','lower','upper'])
    assert leakage_error==0.
    dimensional_error=0.
    for g in [.9,1.,1.1]:
        for ambient in [-2.,0.,2.]:
            p=dict(fit['parameters'],Ua_run_over_nominal=[g]);f={'parameters':p,'lag':True,'curve':True,'ambient':'constant','record_names':[d['name']],'coefficients':[K,ambient]}
            direct=independent(d,f)[1:]
            assert np.all(direct>=scenario['lower']-1e-8) and np.all(direct<=scenario['upper']+1e-8)
            if g==1 and ambient==0:dimensional_error=float(np.max(abs(direct-scenario['nominal'])))
    assert dimensional_error<1e-8
    observation=[]
    for phase in [-.5,0.]:
        for mean,sd in [(19.94,.04),(35.17,.5),(70.54,1.)]:
            low,high=mean-11*sd,mean+11*sd;codes=np.arange(np.floor((low-OFFSET)/STEP)-2,np.ceil((high-OFFSET)/STEP)+3)
            edges=STEP*(codes+phase)+OFFSET;edges=edges[(edges>low)&(edges<high)]
            reference=quad(lambda x:float(report(x,phase))*norm.pdf(x,mean,sd),low,high,points=edges,epsabs=1e-10,limit=1000)[0]
            value=float(reported_mean(np.array([mean]),sd,phase)[0]);error=abs(value-reference);assert error<1e-8
            qlo=reported_quantile(np.array([mean]),sd,.025,phase);qhi=reported_quantile(np.array([mean]),sd,.975,phase)
            lower=bins(qlo,phase)[0][0];upper=bins(qhi,phase)[1][0];coverage=norm.cdf((upper-mean)/sd)-norm.cdf((lower-mean)/sd)
            assert coverage>=.95-1e-12
            observation.append({'phase':phase,'mean_C':mean,'sd_C':sd,'expectation_integral_error_C':error,'exact_marginal_code_coverage':float(coverage)})
    # Identical normal components recover a closed-form quantile. A symmetric
    # nonidentical mixture has its median at the symmetry center.
    means=np.array([[1.,2.,3.]]*3);sd=np.array([.2,.2,.2]);q=mixture_quantile(means,sd,.975)
    error=float(np.max(abs(q-(means[0]+.2*norm.ppf(.975)))));assert error<1e-12
    median=mixture_quantile(np.array([[-1.,2.],[1.,4.]]),np.array([.5,.5]),.5);assert np.max(abs(median-[0.,3.]))<1e-12
    declared=json.loads((HERE/'transfer_config.json').read_text());primary=folds(declared,'primary');shared=folds(declared,'shared')
    assert len(primary)==16 and len(shared)==4
    for fold in primary+shared:
        assert not set(fold['train_indices'])&set(fold['test_indices'])
        assert sorted(fold['train_indices']+fold['test_indices'])==list(range(12))
    write(args.output/'audit.json',{'status':'passed','future_test_output_leakage_max_error_C':leakage_error,
        'independent_dimensional_nominal_error_C':dimensional_error,'scenario_grid_checks':scenario['grid_checks'],
        'reported_observation_checks':observation,'identical_normal_mixture_quantile_error_C':error,
        'fold_counts':{'primary':len(primary),'shared':len(shared)},
        'scope':'Independent dimensional endpoint containment and nominal trajectory, exact no-leakage perturbation, numerical integration of the actual reporting function, closed-form/mixture limits and fold separation. No transfer-model adequacy, parameter-draw calibration or simultaneous predictive-coverage claim.'})
    print(json.dumps({'status':'passed','dimensional_error':dimensional_error,'leakage_error':leakage_error,
                      'observation_integral_max_error':max(r['expectation_integral_error_C'] for r in observation)}),flush=True)


if __name__=='__main__':main()
