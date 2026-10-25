"""Check weak-output phase claims in the two-dimensional complex-gain plane."""
import argparse
import hashlib
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'residual_study'));sys.path.insert(0,str(HERE.parent/'shared_model_investigation'))
import numpy as np
from physical import load_records
from frequency import held_fundamental,regression
from run_frequency import period_of
from run_mean import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();rows=[]
    for d in load_records():
        period=period_of(d['name'])
        if period is None:continue
        for cutoff in [300.,450.,600.]:
            t=d['t'];keep=t>=cutoff;inp=held_fundamental(t,d['u'],2*np.pi/period,cutoff,t[-1]);u=complex(inp[1],inp[2])
            a,b=(1/u).real,(1/u).imag;rotation=np.array([[a,-b],[b,a]])
            for channel in range(2):
                f=regression(t[keep],d['y'][keep,channel],period);h=rotation@f['beta'][1:3];values={}
                for lag,V in f['covariances'].items():
                    C=rotation@V@rotation.T;wald=float(h@np.linalg.solve(C,h))
                    values[lag]={'covariance':C.tolist(),'zero_Wald_statistic':wald,
                        'working95_ellipse_contains_zero':bool(wald<=5.991464547)}
                rows.append({'name':d['name'],'period_s':period,'channel':channel+1,'cutoff_s':cutoff,
                    'complex_gain':h.tolist(),'HAC':values,'temperature_fundamental_amplitude_C':float(np.linalg.norm(f['beta'][1:3]))})
    write_json(args.output/'weak_frequency.json',{'rows':rows,'notice':'Joint complex-gain HAC ellipses are working asymptotic diagnostics, not calibrated tests under quantization or model selection. If the ellipse includes zero, phase is not bounded by that ellipse; scalar delta-method phase bars should not be interpreted as a regular confidence interval.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    fig,axes=plt.subplots(1,3,figsize=(10,3.4),sharex=True,sharey=True)
    selected=[r for r in rows if r['period_s']==30 and r['channel']==2 and r['cutoff_s']==450]
    for ax,row in zip(axes,selected):
        C=np.array(row['HAC']['60']['covariance']);values,vectors=np.linalg.eigh(C);vec=vectors[:,-1]
        h=np.array(row['complex_gain']);width,height=2*np.sqrt(5.991464547*values[::-1])
        ellipse=Ellipse(h*1e4,width*1e4,height*1e4,angle=np.rad2deg(np.arctan2(vec[1],vec[0])),facecolor='#2166ac',alpha=.18,edgecolor='#2166ac')
        ax.add_patch(ellipse);ax.plot(*(h*1e4),'o',color='#2166ac');ax.plot(0,0,'k+');ax.axhline(0,color='.7',lw=.5);ax.axvline(0,color='.7',lw=.5)
        ax.autoscale_view();ax.set_aspect('equal',adjustable='box');ax.set_title('Run '+row['name'].split('_run_')[1].split('.')[0]);ax.set_xlabel('Real gain, 1e-4 C/percent');ax.grid(alpha=.2)
    axes[0].set_ylabel('Imaginary gain, 1e-4 C/percent');fig.suptitle('Fast T2 response after 450 s; working joint HAC60 ellipses');fig.tight_layout(rect=(0,0,1,.9));fig.savefig(args.output/'fast_T2_complex.png',dpi=160);plt.close(fig)
    write_json(args.output/'figure_receipt.json',{'visual_review':'pending','sha256':{'fast_T2_complex.png':hashlib.sha256((args.output/'fast_T2_complex.png').read_bytes()).hexdigest()}})
    print('Joint complex-gain diagnostics complete',flush=True)


if __name__=='__main__':main()
