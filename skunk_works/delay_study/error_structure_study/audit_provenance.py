"""Numerical dependencies required before reusing saved selected-point audits.

All matches for these module names are bound, including conservative duplicate
names in sibling studies. Adding a new import to one of these modules changes
its hash and invalidates a prior binding. Controller/output-only code is outside
this numerical dependency graph.
"""
import json
from pathlib import Path

MODULES={'audit_joint_quantized.py','joint_quantized.py','bounded.py','physical.py',
    'covariance.py','smoothing.py','quantized.py','conditional_initial.py',
    'fast_smoothing.py','sensitivity.py','initial_sensor.py','effects.py',
    'radiative.py','nonlinear.py','audit_bounded.py','audit_radiative.py',
    'run_bounded.py','envelope.py','covariance_fit.py','run_mean.py'}


def named_binding(directory,modules):
    source=json.loads((Path(directory)/'source_snapshot.json').read_text())
    files={k:v for k,v in source['sha256'].items()
        if Path(k).name in modules or k.startswith('data/') or Path(k).name=='environment.json'}
    assert modules<=set(Path(k).name for k in files),'Missing audit dependency'
    return {'files':files,'python':source['python'],'packages':source['packages']}


def numerical_binding(directory):return named_binding(directory,MODULES)


def diagnostic_binding(directory):return named_binding(directory,{'nonrandom.py','bin_diagnostics.py','quantized.py'})
