"""Separate observable pole coordinates from the original model's physical gauge.

For a fixed Ua and zero initial temperature deviations, the objective depends
on two pole times only. A third coordinate selects a feasible physical family
member and occurs only in the original physical box constraints.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from inference import ModelSpec, parameter_box, predict, sensitivities
from models import original_transfer


def physical_from_chart(chart, Ua):
    fast, gap = np.exp(chart[:2])
    slow = fast+gap
    fraction = 1/(1+np.exp(-chart[2]))
    complement = 1/(1+np.exp(chart[2]))
    sensor_time = fast+gap*fraction
    # Factored expression avoids cancellation near an endpoint of the family.
    CpS = Ua*gap*gap*fraction*complement/sensor_time
    return np.asarray([Ua, CpS/sensor_time, Ua*fast*slow/sensor_time, CpS])


def chart_from_physical(parameters):
    _, a, b = original_transfer(parameters)
    slow = .5*(a+np.sqrt(a*a-4*b))
    fast = b/slow
    gap = slow-fast
    fraction = (parameters[3]/parameters[1]-fast)/gap
    fraction = np.clip(fraction, 1e-14, 1-1e-14)
    return np.array([np.log(fast), np.log(gap), np.log(fraction/(1-fraction))])


def solve_fixed_Ua(Ua, starting, datasets, maxiter=2000):
    if any(data.temperature[0] != data.ambient for data in datasets):
        raise ValueError("Pole-chart recovery requires the recorded zero-state initial condition")
    if any(data.power != datasets[0].power for data in datasets):
        raise ValueError("A shared IO gain requires the same heater calibration")
    original = ModelSpec(family="original", order=0, coordinates="physics")
    lower, upper = parameter_box(original)
    if not lower[0] <= Ua <= upper[0]:
        raise ValueError("Ua outside original physical box")
    start = np.array(starting, dtype=float, copy=True)
    start[0] = Ua
    initial = chart_from_physical(start)
    io_spec = ModelSpec(order=1)
    gain = .00016*datasets[0].power/Ua
    log_lower, log_upper = np.log(lower[1:]), np.log(upper[1:])

    def objective(chart):
        fast, gap = np.exp(chart[:2])
        slow = fast+gap
        parameters = np.array([gain, slow, fast])
        residual = np.concatenate([(predict(data, parameters, io_spec)-data.temperature)[1:] for data in datasets])
        J = sensitivities(parameters, datasets, io_spec)
        derivative = np.column_stack([J[:, 1]*fast/slow+J[:, 2], J[:, 1]*gap/slow, np.zeros(len(residual))])
        return .5*float(residual@residual), derivative.T@residual

    def constraints(chart):
        values = np.log(physical_from_chart(chart, Ua)[1:])
        return np.r_[values-log_lower, log_upper-values]

    def constraint_jacobian(chart):
        columns = []
        for j in range(3):
            point = np.array(chart, dtype=complex)
            point[j] += 1e-20j
            columns.append(np.log(physical_from_chart(point, Ua)[1:]).imag/1e-20)
        J = np.column_stack(columns)
        return np.vstack([J, -J])

    solution = minimize(objective, initial, method="SLSQP", jac=True,
                        bounds=[(np.log(1e-5), np.log(1e6)), (np.log(1e-6), np.log(1e6)), (-50., 50.)],
                        constraints=[{"type": "ineq", "fun": constraints, "jac": constraint_jacobian}],
                        options={"ftol": 1e-9, "maxiter": maxiter})
    parameters = physical_from_chart(solution.x, Ua)
    actual_residual = np.concatenate([(predict(data, parameters, original)-data.temperature)[1:] for data in datasets])
    actual_sse = float(actual_residual@actual_residual)
    violation = float(max(0., -np.min(constraints(solution.x))))
    prediction_error = abs(actual_sse-2*float(solution.fun))
    return {"success": bool(solution.success and violation < 1e-7 and prediction_error < 1e-6),
            "solver_success": bool(solution.success), "status": int(solution.status), "message": str(solution.message),
            "sse": actual_sse, "observable_chart_sse": 2*float(solution.fun),
            "maximum_log_box_violation": violation, "sse_representation_difference": prediction_error,
            "parameters": parameters.tolist(), "chart": solution.x.tolist(), "initial_chart": initial.tolist(),
            "iterations": int(solution.nit), "function_evaluations": int(solution.nfev),
            "solver": "SLSQP in pole times plus physical family coordinate",
            "notice": "Exact objective equivalence verified in the original simulator; feasible family coordinate enters only physical constraints. Successful local optimization is not a global certificate."}
