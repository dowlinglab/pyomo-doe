"""Check family boundary functions through physical reconstruction."""
import numpy as np
import pytest

from family_uncertainty import boundary_functions
from models import original_transfer


def test_boundary_functions_are_pole_label_invariant_and_bound_actual_members():
    physical = np.array([.04, .02, 5., .3])
    K, a, b = original_transfer(physical)
    slow = (a+np.sqrt(a*a-4*b))/2
    fast = b/slow
    result = boundary_functions([K, slow, fast])[0]
    assert np.allclose(result, boundary_functions([K, fast, slow])[0], atol=1e-14)
    Ua = physical[0]
    members = []
    for r in np.r_[np.linspace(fast+1e-5, slow-1e-5, 201), np.sqrt(b), 2*b/a]:
        CpH = Ua*b/r
        CpS = Ua*(a-r-b/r)
        members.append([Ua, CpS/r, CpH, CpS])
    members = np.array(members)
    assert np.all(members[:, 2] > result[6]) and np.all(members[:, 2] < result[7])
    assert abs(members[:, 3].max()-result[8]) < 1e-13
    assert abs(members[:, 1].max()-result[9]) < 1e-13
    assert np.max(abs(np.array([original_transfer(p) for p in members])-[K, a, b])) < 1e-10


def test_heater_calibration_scales_only_physical_boundaries():
    p = [.78, 150., 18.]
    baseline = boundary_functions(p)[0]
    changed = boundary_functions(p, calibration=.064)[0]
    assert np.allclose(changed[[0, 1, 2, 4, 5]], baseline[[0, 1, 2, 4, 5]])
    assert np.allclose(changed[[3, 6, 7, 8, 9]], 2*baseline[[3, 6, 7, 8, 9]])
    with pytest.raises(ValueError, match="distinct poles"):
        boundary_functions([.78, 18., 18.])
