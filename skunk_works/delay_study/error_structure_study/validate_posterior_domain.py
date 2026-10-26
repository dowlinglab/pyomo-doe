"""Regression checks after correcting the rare-bin posterior integration domain."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--study',type=Path,required=True);args=parser.parse_args();begun=time.monotonic();rows=[]
    checks=[('quantizer','check_quantized',[]),('smoother','check_smoothing',[]),
            ('joint_scores','check_joint_quantized',['--warm',str(args.study/'results/band_10_constant/fit.json')])]
    for channel in [1,2]:
        checks.append(('hardware_T'+str(channel),'audit_observation',['--input',str(args.study/('results/bin_noise_band10_T'+str(channel)+'_v2')),'--mean-fit',str(args.study/'results/band_10_constant')]))
    for label,module,extra in checks:
        output=args.output/label;output.mkdir(exist_ok=False);sys.argv=[module+'.py','--output',str(output)]+extra
        importlib.import_module(module).main()
        path=output/'audit.json';receipt=json.loads(path.read_text());assert receipt['status']=='passed'
        rows.append({'check':label,'audit_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps({'check_complete':label,'elapsed_s':time.monotonic()-begun}),flush=True)
    (args.output/'audit.json').write_text(json.dumps({'status':'passed','checks':rows,'elapsed_s':time.monotonic()-begun,
        'scope':'Ordinary/rare-domain source change regression: bin probabilities, independent smoother moments/scores, both physical score charts and full selected hardware covariance scores. Broader model adequacy and parameter uncertainty remain separate.'},indent=2)+'\n')


if __name__=='__main__':main()
