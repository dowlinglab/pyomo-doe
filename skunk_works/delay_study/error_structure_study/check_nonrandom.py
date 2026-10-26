"""Independent scaled-density quadrature and exact joint-code arithmetic."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr
from quantized import log_probability
from nonrandom import expected_gaussian_pit,joint_transition_checks
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    cases=[(-np.inf,np.inf),(-np.inf,-3.),(3.,np.inf),(-2.,1.),(-.2,.2),(0.,.00001),
        (10.,10.01),(-10.01,-10.),(40.,40.001),(-40.001,-40.),(1.,1.+1e-10)]
    rows=[]
    for a,b in cases:
        f={'log_CDF_lower':np.array([log_ndtr(a)]),'log_SF_upper':np.array([log_ndtr(-b)]),
            'log_bin_probability':np.array([float(log_probability(a,b))])}
        got,thin=expected_gaussian_pit(f)
        anchor=max(a,0.) if b>0 else b
        def weight(x):return np.exp(-.5*(x-anchor)*(x+anchor))
        den=quad(weight,a,b,epsabs=1e-12,epsrel=1e-12)[0]
        expected=quad(lambda x:x*weight(x),a,b,epsabs=1e-12,epsrel=1e-12)[0]/den
        error=float(abs(got[0]-expected));assert error<2e-8,(a,b,got,expected,error)
        rows.append({'bounds':[str(a),str(b)],'score':float(got[0]),'quadrature_score':float(expected),'absolute_error':error,'thin_approximation_count':thin})
    p=np.tile([.2,.5,.3],(6,1));q=np.tile([.1,.6,.3],(6,1))
    a=np.array([0,1,2,0,1,2]);b=np.array([2,1,0,2,1,0]);y=np.vstack([np.zeros(2),np.cumsum(np.column_stack([a-1,b-1]),axis=0)])
    result=joint_transition_checks([{'name':'synthetic','y':y}],[[{'transition_probabilities':p},{'transition_probabilities':q}]])
    expected=np.zeros((3,3));center=np.zeros((3,3));variance=np.zeros((3,3))
    for t in range(6):
        for i in range(3):
            for j in range(3):
                expected[i,j]+=p[t,i]*q[t,j]
                center[i,j]+=(float(a[t]==i)-p[t,i])*(float(b[t]==j)-q[t,j])
                variance[i,j]+=p[t,i]*(1-p[t,i])*q[t,j]*(1-q[t,j])
    assert np.max(abs(np.array(result['per_run'][0]['expected_joint_counts'])-expected))<1e-15
    assert np.max(abs(np.array(result['pooled_centered_event_cross_standardized'])-center/np.sqrt(np.maximum(variance,1.))))<1e-15
    write(args.output/'check.json',{'status':'passed','Gaussian_PIT_expectation_quadrature':rows,'joint_code_arithmetic':'passed'})
    print(json.dumps({'status':'passed','max_score_error':max(r['absolute_error'] for r in rows)}),flush=True)


if __name__=='__main__':main()
