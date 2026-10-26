"""Known precision, constrained subspace and box-truncation sampling checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from draw_local import conditional_draws
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    point=np.array([.1,.2,-.2,0.]);H=np.diag([4.,9.,16.,25.]);basis=null_space(np.array([[0.,1.,1.,0.],[0.,0.,0.,1.]]))
    wide=np.tile([-100.,100.],(4,1));draws,proposals=conditional_draws(point,wide,H,basis,100000,7711)
    exact=basis@np.linalg.inv(basis.T@H@basis)@basis.T
    error=float(np.max(abs(np.cov(draws.T)-exact)));assert error<.004
    assert np.max(abs(draws[:,1]+draws[:,2]))<1e-12 and np.max(abs(draws[:,3]))==0.
    narrow=wide.copy();narrow[0]=[0.,.2]
    truncated,count=conditional_draws(point,narrow,H,basis,10000,9917)
    assert np.all((truncated[:,0]>=0)&(truncated[:,0]<=.2)) and count>10000
    assert abs(truncated[:,0].mean()-.1)<.003
    write(args.output/'check.json',{'status':'passed','untruncated_covariance_max_error':error,
        'centering_max_error':float(np.max(abs(draws[:,1]+draws[:,2]))),'fixed_face_error':0.,
        'truncated_proposals':count,'truncated_symmetric_coordinate_mean':float(truncated[:,0].mean()),
        'scope':'Known Gaussian precision, exact affine constraints and bounded rejection. No statistical calibration for the hardware information matrix.'})
    print(json.dumps({'status':'passed','covariance_max_error':error}),flush=True)


if __name__=='__main__':main()
