"""Check the scope of the SISO ambiguity against an ideal symmetric two-channel model.

This is a new analytic/forward-simulation check, not a fit or validation of the
four-state model on the laboratory data. Us means heater-sensor coupling and Ux
means interheater coupling, avoiding the notebooks' differing Ub/Uc conventions.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data import load
from models import propagate_lti, simulate_original


def coefficients(p, c=.032):
    ua, us, ux, ch, cs = p
    result = []
    for loss in [ua, ua + 2 * ux]:
        result.append([c / loss, ch / loss + cs / us + cs / loss,
                       ch * cs / (loss * us)])
    return np.array(result)


def recover(transfer, c=.032):
    kp, ap, bp = transfer[0]
    km, am, bm = transfer[1]
    ua, loss_m = c / kp, c / km
    ux = (loss_m - ua) / 2
    r = (loss_m * am - ua * ap) / (loss_m - ua)
    ch = ua * bp / r
    cs = ua * (ap - r) - ch
    us = cs / r
    assert np.isclose(ua * bp, loss_m * bm)
    return np.array([ua, us, ux, ch, cs])


def forward(p, data):
    ua, us, ux, ch, cs = p
    A = np.array([[-(ua + us + ux) / ch, us / ch, ux / ch, 0],
                  [us / cs, -us / cs, 0, 0],
                  [ux / ch, 0, -(ua + us + ux) / ch, us / ch],
                  [0, 0, us / cs, -us / cs]])
    B = np.array([.032 / ch, 0, 0, 0])
    states = propagate_lti(A, B, np.zeros(4, dtype=A.dtype), data.time, data.input)
    return states[:, [1, 3]]


def main():
    data = load('sine')
    p = np.array([.04, .015, .003, 6.9, .32])
    transfer = coefficients(p)
    reconstructed = recover(transfer)
    assert np.allclose(reconstructed, p, rtol=1e-12)
    y = forward(p, data)
    yp = simulate_original(data, p[[0, 1, 3, 4]]) - data.ambient
    pm = p[[0, 1, 3, 4]].copy()
    pm[0] += 2 * p[2]
    ym = simulate_original(data, pm) - data.ambient
    error = max(np.max(np.abs(y[:, 0] + y[:, 1] - yp)),
                np.max(np.abs(y[:, 0] - y[:, 1] - ym)))
    assert error < 1e-10
    J = []
    for j in range(5):
        pc = p.astype(complex)
        pc[j] += 1e-20j
        J.append((forward(pc, data)[1:].imag / 1e-20 * p[j]).ravel())
    singular = np.linalg.svd(np.column_stack(J), compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-8))
    assert rank == 5
    result = {'status': 'analytic_reconstruction_and_forward_checks_passed',
              'parameter_roles': ['Ua', 'Us: heater-sensor', 'Ux: interheater', 'CpH', 'CpS'],
              'example_parameters': p.tolist(), 'reconstructed_parameters': reconstructed.tolist(),
              'common_differential_coefficients_K_a_b': transfer.tolist(),
              'maximum_common_differential_prediction_error_C': float(error),
              'log_sensitivity_singular_values': singular.tolist(), 'local_rank_at_example': rank,
              'known_c_W_per_percent': .032,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'forward_core_sha256': hashlib.sha256((HERE.parent / 'models.py').read_bytes()).hexdigest(),
              'input_sha256': hashlib.sha256((HERE.parents[2] / 'data/tclab_sine_test_5min_period.csv').read_bytes()).hexdigest(),
              'scope': 'Ideal symmetric four-state model, two sensor outputs, known heater conversion, positive interheater coupling, common ambient, zero initial deviations, and one driven heater. Generic structural recoverability follows from the displayed coefficient inversion; this numerical example checks algebra and a full-rank local point. No four-state data fit, practical-estimability claim, uncertainty calculation or hardware validation.'}
    (HERE / 'four_state_scope_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
