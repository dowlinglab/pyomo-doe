"""Intersect the exact original-model equivalence curve with numerical boxes.

The one-dimensional root partition is separate from likelihood uncertainty.
It supplies feasible witnesses that a numerical profile set must include at
its recorded optimum, subject to explicitly checked floating-point tolerances.
"""
import numpy as np

from inference import ModelSpec, parameter_box
from models import equivalent_original, original_transfer


def real_quadratic_roots(A, B, C):
    discriminant = B*B-4*A*C
    tolerance = 64*np.finfo(float).eps*max(B*B, abs(4*A*C))
    if discriminant < -tolerance:
        return []
    root = np.sqrt(max(0., discriminant))
    # Stable formula retains the small root when the two differ greatly.
    q = -.5*(B+np.copysign(root, B))
    return [q/A, C/q] if q else [-B/(2*A)]


def fixed_value_roots(Ua, a, b, parameter_index, value):
    if parameter_index == 1:
        return real_quadratic_roots(1+value/Ua, -a, b)
    if parameter_index == 2:
        return [Ua*b/value]
    if parameter_index == 3:
        return real_quadratic_roots(1., -(a-value/Ua), b)
    raise ValueError("Only the redundant Ub,CpH,CpS coordinates have family roots")


def family_context(parameters, lower=None, upper=None):
    parameters = np.asarray(parameters)
    if np.any(parameters <= 0):
        raise ValueError("Positive original parameters required")
    if lower is None or upper is None:
        lower, upper = parameter_box(ModelSpec(family="original", order=0, coordinates="physics"))
    lower, upper = np.asarray(lower), np.asarray(upper)
    if lower.shape != (4,) or upper.shape != (4,) or np.any(lower <= 0) or np.any(upper <= lower):
        raise ValueError("Strict positive four-coordinate numerical boxes required")
    K, a, b = original_transfer(parameters)
    Ua = parameters[0]
    discriminant = a*a-4*b
    if discriminant <= 0:
        raise ValueError("Distinct positive poles not numerically resolved")
    slow = .5*(a+np.sqrt(discriminant))
    fast = b/slow
    return Ua, K, a, b, fast, slow, lower, upper


def feasible_intervals(parameters, lower=None, upper=None):
    Ua, K, a, b, fast, slow, lower, upper = family_context(parameters, lower, upper)
    if not lower[0] <= Ua <= upper[0]:
        return []
    cuts = [fast, slow]
    for j in (1, 2, 3):
        for value in (lower[j], upper[j]):
            cuts.extend(r for r in fixed_value_roots(Ua, a, b, j, value) if fast < r < slow)
    cuts = np.unique(cuts)
    result = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        middle = (left+right)/2
        if middle in (left, right):
            continue
        member = equivalent_original(K, a, b, middle)
        if np.all(member >= lower) and np.all(member <= upper):
            if result and result[-1][1] == left:
                result[-1][1] = float(right)
            else:
                result.append([float(left), float(right)])
    return result


def parameter_extrema(parameters, parameter_index, lower=None, upper=None):
    Ua, K, a, b, fast, slow, lower, upper = family_context(parameters, lower, upper)
    intervals = feasible_intervals(parameters, lower, upper)
    if not intervals:
        raise ValueError("No equivalent member lies within this numerical box")
    candidates, component_ranges = [], []
    for left, right in intervals:
        start = len(candidates)
        coordinates = [left, right]
        if parameter_index == 1 and left < 2*b/a < right:
            coordinates.append(2*b/a)
        elif parameter_index == 3 and left < np.sqrt(b) < right:
            coordinates.append(np.sqrt(b))
        for r in coordinates:
            member = equivalent_original(K, a, b, r)
            if np.any(member < lower-1e-9*np.maximum(lower, 1e-10)) or np.any(member > upper+1e-9*upper):
                raise RuntimeError("Family endpoint violates the declared box beyond numerical tolerance")
            candidates.append({"r_s": float(r), "parameters": member.tolist(), "value": float(member[parameter_index])})
        component_ranges.append([min(row["value"] for row in candidates[start:]), max(row["value"] for row in candidates[start:])])
    return {"parameter_index": parameter_index, "feasible_r_intervals_s": intervals,
            "compatible_parameter_components": component_ranges,
            "minimum": min(candidates, key=lambda row: row["value"]),
            "maximum": max(candidates, key=lambda row: row["value"]), "witnesses": candidates,
            "notice": "Root-partition extrema of the exact equivalence class at a fixed fitted transfer function, restricted by the numerical box. These are not likelihood-confidence endpoints. Floating-point endpoint feasibility is explicitly checked."}


def fixed_value_members(parameters, parameter_index, value, lower=None, upper=None):
    Ua, K, a, b, fast, slow, lower, upper = family_context(parameters, lower, upper)
    members = []
    for r in fixed_value_roots(Ua, a, b, parameter_index, value):
        if not fast < r < slow:
            continue
        member = equivalent_original(K, a, b, r)
        if np.all(member >= lower-1e-9*np.maximum(lower, 1e-10)) and np.all(member <= upper+1e-9*upper):
            if abs(member[parameter_index]/value-1) > 1e-8:
                raise RuntimeError("Family root does not match the fixed native value")
            members.append(member)
    return members
