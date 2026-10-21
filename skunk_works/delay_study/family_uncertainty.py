"""Identifiable boundary functions of the original model's positive family."""
import numpy as np


FUNCTIONS = ("K", "a_s", "b_s2", "Ua_W_K", "fast_pole_s", "slow_pole_s",
             "CpH_inf_J_K", "CpH_sup_J_K", "CpS_sup_J_K", "Ub_sup_W_K")


def boundary_functions(io_parameters, calibration=.032):
    """Map either pole label to the same positive-family boundary functions.

    Infima for CpH and lower-zero CpS/Ub endpoints are open. Supremal CpS/Ub
    values are attained at interior family coordinates. Bounds are for the full
    strictly positive family, before rectangular numerical fitting restrictions.
    """
    p = np.atleast_2d(np.asarray(io_parameters))
    if p.shape[1] != 3 or np.any(p <= 0) or calibration <= 0:
        raise ValueError("Positive K, tau, theta and calibration required")
    K, first, second = p.T
    fast, slow = np.minimum(first, second), np.maximum(first, second)
    if np.any(slow-fast <= 1e-10*np.maximum(slow, 1.)):
        raise ValueError("Strictly positive original family requires distinct poles")
    a, b, Ua = first+second, first*second, calibration/K
    return np.column_stack([K, a, b, Ua, fast, slow, Ua*fast, Ua*slow,
                            Ua*(a-2*np.sqrt(b)), Ua*(a*a/(4*b)-1)])
