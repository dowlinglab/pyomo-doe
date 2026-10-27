"""Independent finite-dimensional controls for information constraint bases."""
import argparse
from pathlib import Path
import numpy as np
from audit_information import basis_invariants
from run_bounded import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    a=np.array([0.,1.,1.,0.]);e=np.eye(4)
    Q=np.column_stack([e[:,0],(e[:,1]-e[:,2])/np.sqrt(2),e[:,3]])
    inactive=np.zeros(4,dtype=bool);active=np.array([True,False,False,False])
    cases={
        'no_active_bounds':basis_invariants(Q,a,inactive,True),
        'one_active_bound':basis_invariants(Q[:,1:],a,active,True),
        'centered_tangent_ignores_bound_rows':basis_invariants(Q,a,active,False),
        'no_constraints':basis_invariants(e,np.zeros(4),inactive,True)}
    assert cases['no_active_bounds']['expected_dimension']==3 and cases['no_active_bounds']['active_bound_restriction_error']==0.
    assert cases['one_active_bound']['expected_dimension']==2 and cases['one_active_bound']['constraint_rank']==2
    assert cases['no_constraints']['expected_dimension']==4
    invalid={
        'missing_feasible_direction':(Q[:,1:],a,inactive,True),
        'missing_active_constraint':(Q,a,active,True),
        'wrong_centered_subspace':(np.column_stack([e[:,1],e[:,3]]),a,active,True),
        'nonorthonormal_basis':(Q*2,a,inactive,True)}
    rejected=[]
    for name,values in invalid.items():
        try:basis_invariants(*values)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Invalid basis accepted: '+name)
    assert len(rejected)==4
    write(args.output/'check.json',{'status':'passed','valid_cases':cases,'invalid_cases_rejected':rejected,
        'scope':'Explicit four-coordinate algebra checks the empty and nonempty active-set branches, rank/dimension, equality restrictions and orthogonality. Not a measured-fit curvature or covariance result.'})


if __name__=='__main__':main()
