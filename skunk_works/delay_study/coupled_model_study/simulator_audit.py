"""Compare the inspected upstream simulator with an independent linear model."""
import argparse
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'))
import numpy as np
from scipy.linalg import expm
from run_mean import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--sha256',required=True);args=parser.parse_args()
    raw=args.source.read_bytes();assert hashlib.sha256(raw).hexdigest()==args.sha256
    tree=ast.parse(raw);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TCLabModel')
    selected={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['update','measurement','quantize']}
    # Inspect and execute only the pure arithmetic update method. No package
    # import, serial port discovery, constructor, realtime clock or hardware call.
    node=selected['update'];namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<checked-upstream-update>','exec'),namespace)
    A=np.array([[-.06,0,.01,0],[1/140,-1/140,0,0],[.01,0,-.06,0],[0,0,1/140,-1/140]])
    B=np.array([200/5720,0,0,0]);C=np.eye(4)[[1,3]];steps=[];rng=np.random.RandomState(617)
    time=np.r_[0,np.cumsum(rng.uniform(.8,1.3,180))];command=rng.uniform(0,100,len(time))
    exact=np.zeros(4);reference=[exact[[1,3]]+21]
    for j,h in enumerate(np.diff(time)):
        augmented=np.zeros((5,5));augmented[:4,:4]=A;augmented[:4,4]=B
        E=expm(augmented*h);exact=E[:4,:4]@exact+E[:4,4]*command[j];reference.append(exact[[1,3]]+21)
    for step in [.2,.1,.05]:
        lab=SimpleNamespace(Ta=21.,_P1=200.,_P2=100.,_Q1=0.,_Q2=0.,_H1=21.,_H2=21.,_T1=21.,_T2=21.,maxstep=step,tlast=0.,synced=False)
        trajectory=[[lab._T1,lab._T2]]
        for j,t in enumerate(time[1:]):
            lab._Q1=float(command[j]);namespace['update'](lab,t=float(t));trajectory.append([lab._T1,lab._T2])
        error=float(np.max(abs(np.array(trajectory)-reference)));steps.append({'maxstep_s':step,'max_output_error_vs_exact_C':error})
    assert steps[0]['max_output_error_vs_exact_C']<.08 and steps[-1]['max_output_error_vs_exact_C']<steps[0]['max_output_error_vs_exact_C']/2
    frequency=[]
    for period in [300.,120.,30.]:
        omega=2*np.pi/period;H=C@np.linalg.solve(1j*omega*np.eye(4)-A,B)
        discrete=C@np.linalg.solve(np.exp(1j*omega*.2)*np.eye(4)-(np.eye(4)+.2*A),.2*B)
        frequency.append({'period_s':period,'continuous_gain':abs(H).tolist(),'continuous_phase_deg':np.angle(H,deg=True).tolist(),
            'Euler_point_input_phase_difference_deg':np.angle(discrete/H,deg=True).tolist()})
    # A floor quantizer applied to independent analogue errors has a state-
    # dependent conditional mean. It does not make the draws dependent given T.
    from scipy.special import ndtr
    delta,sigma=.3223,.043;temperature=21+np.linspace(0,delta,21);means=[]
    for value in temperature:
        bins=np.arange(np.floor(value/delta)-4,np.ceil(value/delta)+5)
        probabilities=ndtr(((bins+1)*delta-value)/sigma)-ndtr((bins*delta-value)/sigma)
        means.append(float(np.sum(bins*delta*probabilities)-value))
    write_json(args.output/'simulator_audit.json',{'status':'passed','source_url':'https://raw.githubusercontent.com/jckantor/TCLab/master/tclab/tclab.py',
        'source_sha256':args.sha256,'source_git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
        'inspected_method_AST_sha256':{k:hashlib.sha256(ast.dump(v).encode()).hexdigest() for k,v in selected.items()},
        'model':'Symmetric two coupled heater states, nonloading first-order sensor states, fixed linear command conversion; no explicit command lag or radiation term in this inspected update.',
        'common_heater_time_s':20.,'differential_heater_time_s':1/.07,'sensor_time_s':140.,
        'matrix_A':A.tolist(),'unit_Q1_B':B.tolist(),'steady_gain_C_per_percent':(-C@np.linalg.solve(A,B)).tolist(),
        'upstream_Euler_checks':steps,'frequency':frequency,'measurement':{'quantization_step_C':delta,'prequantization_normal_SD_C':sigma,
            'interior_conditional_bias_range_over_one_bin_C':[min(means),max(means)],'clipping_C':[-50,132.2]},
        'notice':'Source-inspection and synthetic arithmetic audit, not an ENV2 fit or evidence that this library version generated the records. Repeated sensor pole cancels once in each transfer. Defaults are emulator assumptions, not independently calibrated hardware properties. Euler phase comparison uses point commands on its own 0.2 s grid; actual command holds are separate.'})
    print('Upstream simulator source and numerical comparisons passed',flush=True)


if __name__=='__main__':main()
