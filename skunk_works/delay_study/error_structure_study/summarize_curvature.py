"""Recover arithmetic from completed diagnostic evaluations without rerunning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--information',type=Path,required=True)
    args=parser.parse_args();path=args.input/'evaluations.json';rows=json.loads(path.read_text())
    by={round(r['step'],10):r for r in rows};checks=[];missing=[]
    H=np.load(args.information/'information_matrices.npz')['observed_Hessian']
    for h in [.0002,.002,.005,.01,.02,.05]:
        points=[round(j*h,10) for j in range(4)]
        if any(v not in by for v in points):missing.append({'step':h,'missing_evaluations':[v for v in points if v not in by]});continue
        rr=[by[v] for v in points]
        scalar=(2*rr[0]['dimensional_forward_nll_256']-5*rr[1]['dimensional_forward_nll_256']+4*rr[2]['dimensional_forward_nll_256']-rr[3]['dimensional_forward_nll_256'])/h**2
        score=(-3*rr[0]['loading_score_128']+4*rr[1]['loading_score_128']-rr[2]['loading_score_128'])/(2*h)
        checks.append({'step':h,'scalar_forward_curvature':float(scalar),'score_curvature':float(score),'stored_H22':float(H[2,2])})
    write(args.output/'diagnosis.json',{'status':'arithmetic_recovered_with_missing_stencil' if missing else 'complete',
        'evaluations_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checks':checks,'missing_stencils':missing,
        'notice':'No optimization or filtering rerun. Original diagnostic evaluated all requested points then failed final assembly because the 0.01 stencil omitted 0.03. This receipt retains the remaining valid stencils and explicitly records the missing one. No acceptance threshold was relaxed.'})
    print(json.dumps({'checks':checks,'missing':missing}),flush=True)


if __name__=='__main__':main()
