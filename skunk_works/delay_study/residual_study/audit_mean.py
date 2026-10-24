"""Independent forward checks against augmented matrix exponentials and solve_ivp."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from mean import ambient_basis, unit_response


def reference(data, K, tau, theta, endpoints, hold, integration=False):
    time = data.time-data.time[0]
    duration = time[-1]
    slope = (endpoints[1]-endpoints[0])/duration
    state = np.array([0., data.temperature[0], data.temperature[0]+endpoints[0], 1.])
    result = [state[1]]
    A = np.array([[-1/theta, 0, 0, 0], [K/tau, -1/tau, 1/tau, 0],
                  [0, 0, 0, slope], [0, 0, 0, 0.]])
    for k, dt in enumerate(np.diff(time)):
        u = data.input[k+int(hold == "right")]
        B = np.array([u/theta, 0, 0, 0])
        if integration:
            sol = solve_ivp(lambda t, x: A@x+B, [0, dt], state,
                            rtol=2e-11, atol=2e-12, method="DOP853")
            if not sol.success:
                raise RuntimeError(sol.message)
            state = sol.y[:, -1]
        else:
            augmented = np.zeros((5, 5))
            augmented[:4, :4] = A
            augmented[:4, 4] = B
            state = (expm(augmented*dt)@np.r_[state, 1.])[:4]
        result.append(state[1])
    return np.asarray(result)


def audit():
    from data import Experiment
    rng = np.random.RandomState(40321)
    d = Experiment("audit", np.r_[0., np.cumsum(rng.uniform(.03, 4, 49))],
                   np.full(50, 22.3), rng.uniform(0, 100, 50), 22.3)
    cases = []
    for tau, theta in [(150, 16), (16, 150), (80, 80), (80, 80*(1+1e-9)), (5, .01), (600, 200)]:
        for hold in ("left", "right"):
            for endpoints in ((0., 0.), (-1.7, -1.7), (-1.5, 1.8)):
                p = d.temperature[0]+.83*unit_response(d, tau, theta, hold)
                p += ambient_basis(d.time, tau, "linear")@endpoints
                matrix = reference(d, .83, tau, theta, endpoints, hold)
                integration = reference(d, .83, tau, theta, endpoints, hold, True)
                error = float(np.max(np.abs(p-matrix)))
                numerical = float(np.max(np.abs(p-integration)))
                if error > 2e-8 or numerical > 2e-7:
                    raise AssertionError((tau, theta, hold, endpoints, error, numerical))
                cases.append({"tau": tau, "theta": theta, "hold": hold,
                              "ambient_endpoints": endpoints, "matrix_error_C": error,
                              "integration_error_C": numerical})
    original = unit_response(d, 150, 16)
    swapped = unit_response(d, 16, 150)
    cold_error = float(np.max(np.abs(original-swapped)))
    ambient_difference = float(np.max(np.abs(ambient_basis(d.time, 150, "constant")-
                                              ambient_basis(d.time, 16, "constant"))))
    if cold_error > 1e-9 or ambient_difference < .1:
        raise AssertionError("Pole/ambient test failed")
    return {"status": "passed", "cases": cases, "cold_unit_response_swap_max_C": cold_error,
            "same_offset_pole_swap_response_difference_per_C": ambient_difference,
            "scope": "Forward propagation only; ambient difference at fixed offsets does not prove identifiability after nuisance refitting"}
