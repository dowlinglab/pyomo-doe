"""Refine observed LR crossings and preserve exact feasible family witnesses.

Refinement does not certify all global minima or discover every missed mode.
The old profile remains an input artifact; every new witness and nuisance solve
is retained in a separate phase.
"""
from copy import deepcopy
import hashlib

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from inference import ModelSpec, parameter_box, residuals
from models import original_transfer
from original_family_bounds import parameter_extrema, fixed_value_members
from profiles import interval_components


class ProfileRefiner:
    def __init__(self, record, saved_profile, datasets, starts=4, seed=532, max_nfev=600):
        self.record, self.original = record, saved_profile
        self.datasets = datasets
        self.spec = ModelSpec(**record["fit"]["spec"])
        self.p = np.array([record["fit"]["parameters"][name] for name in self.spec.names])
        self.center = np.log(self.p)
        self.j = saved_profile["parameter_index"]
        self.nuisance = np.arange(len(self.p)) != self.j
        lower, upper = parameter_box(self.spec)
        self.lower, self.upper = lower, upper
        self.lo, self.hi = np.log(lower), np.log(upper)
        self.starts, self.seed, self.max_nfev = starts, seed, max_nfev
        self.baseline = record["fit"]["sse"]
        self.sigma = record["uncertainty"]["known_sigma_C"]
        self.count = record["fit"]["n"]
        self.cache = {point["log_value"]: deepcopy(point) for point in saved_profile["points"]}
        self.new_evaluations = 0
        self.family = None
        if self.spec.family == "original" and self.j in (1, 2, 3):
            if any(d.temperature[0] != d.ambient for d in datasets):
                raise ValueError("Exact family witnesses require zero initial temperature deviations")
            self.family = parameter_extrema(self.p, self.j, lower, upper)
            # Challenge every original sampled point in the known flat family,
            # not just its extrema, while preserving its old optimizer attempts.
            for q in list(self.cache):
                for member in fixed_value_members(self.p, self.j, np.exp(q), lower, upper):
                    self.add_witness(q, member)
            for witness in self.family["witnesses"]:
                self.add_witness(float(np.log(witness["value"])), np.asarray(witness["parameters"]))

    def add_statistics(self, point):
        point["lr_known_sigma_iid"] = float((point["sse"]-self.baseline)/self.sigma**2)
        point["lr_unknown_sigma_iid"] = float(self.count*np.log(point["sse"]/self.baseline))
        return point

    def add_witness(self, q, member):
        member = np.array(member, copy=True)
        member[self.j] = np.exp(q)
        relative_violation = max(0., float(np.max((self.lower-member)/self.lower)),
                                 float(np.max((member-self.upper)/self.upper)))
        transfer_error = float(np.max(abs(original_transfer(member)/original_transfer(self.p)-1)))
        residual = residuals(np.log(member), self.datasets, self.spec)
        sse = float(residual@residual)
        if relative_violation > 1e-8 or transfer_error > 1e-8 or abs(sse-self.baseline) > 1e-6:
            raise RuntimeError("Analytic equivalence witness failed independent feasibility/forward checks")
        attempt = {"kind": "verified_equivalence_witness", "success": True, "sse": sse,
                   "parameters": member.tolist(), "relative_box_violation": relative_violation,
                   "relative_transfer_difference": transfer_error,
                   "notice": "Feasible fitted-transfer witness, not an optimizer convergence or a global-minimum certificate."}
        point = {"log_value": q, "parameter_value": float(np.exp(q)), "sse": sse,
                 "parameters": member.tolist(), "attempts": [attempt], "selected_kind": "verified_equivalence_witness"}
        if q in self.cache:
            previous = self.cache[q]
            if previous["sse"] <= sse:
                previous["attempts"].append(attempt)
                self.add_statistics(previous)
                return previous
            point["attempts"] = previous["attempts"]+[attempt]
            point["previous_saved_sse"] = previous["sse"]
        self.cache[q] = self.add_statistics(point)
        return point

    def evaluate(self, q):
        if q in self.cache:
            return self.cache[q]
        self.new_evaluations += 1
        if self.family is not None:
            members = fixed_value_members(self.p, self.j, np.exp(q), self.lower, self.upper)
            if members:
                for member in members:
                    self.add_witness(q, member)
                return self.cache[q]
        nearest = min(self.cache, key=lambda value: abs(value-q))
        warm = np.log(self.cache[nearest]["parameters"])
        starts = [self.center[self.nuisance], warm[self.nuisance]][:self.starts]
        seed = self.seed ^ int(hashlib.sha256(float(q).hex().encode()).hexdigest()[:8], 16)
        if self.starts > len(starts):
            starts.extend(qmc.scale(qmc.LatinHypercube(int(self.nuisance.sum()), seed=seed).random(self.starts-len(starts)),
                                     self.lo[self.nuisance], self.hi[self.nuisance]))

        def assemble(other):
            values = self.center.astype(np.result_type(self.center, other), copy=True)
            values[self.j] = q
            values[self.nuisance] = other
            return values

        attempts, best = [], None
        for start in starts:
            options = {"jac": "3-point", "max_nfev": self.max_nfev}
            if self.spec.family != "pure_delay":
                options["jac"] = "cs"
            if self.spec.family == "original":
                options.update(x_scale="jac", max_nfev=10*self.max_nfev)
            fitted = least_squares(lambda other: residuals(assemble(other), self.datasets, self.spec),
                                   np.clip(start, self.lo[self.nuisance]+1e-10, self.hi[self.nuisance]-1e-10),
                                   bounds=(self.lo[self.nuisance], self.hi[self.nuisance]),
                                   ftol=1e-10, xtol=1e-10, gtol=1e-7, **options)
            row = {"kind": "nuisance_optimization", "success": bool(fitted.success), "sse": float(fitted.fun@fitted.fun),
                   "status": int(fitted.status), "nfev": int(fitted.nfev), "optimality": float(fitted.optimality),
                   "parameters": np.exp(assemble(fitted.x)).tolist(), "solver_options": options}
            attempts.append(row)
            if fitted.success and (best is None or row["sse"] < best["sse"]):
                best = row
        if best is None:
            raise RuntimeError("All refined nuisance solves failed at "+str(np.exp(q)))
        point = {"log_value": float(q), "parameter_value": float(np.exp(q)), "sse": best["sse"],
                 "parameters": best["parameters"], "attempts": attempts, "selected_kind": "nuisance_optimization"}
        self.cache[q] = self.add_statistics(point)
        return point

    def refine(self, statistic_tolerance=1e-3, log_width_tolerance=1e-7, maximum_steps=26):
        threshold = self.original["chi2_95_df1"]
        crossings = []
        for statistic in ("lr_known_sigma_iid", "lr_unknown_sigma_iid"):
            # Initial brackets include algebraic extrema and any points added
            # for the other likelihood convention; all source modes remain.
            points = [self.add_statistics(self.cache[q]) for q in sorted(self.cache)]
            brackets = [(left, right) for left, right in zip(points[:-1], points[1:])
                        if (left[statistic] <= threshold) != (right[statistic] <= threshold)]
            for first, last in brackets:
                left, right = first["log_value"], last["log_value"]
                evaluated = None
                converged = False
                for iteration in range(maximum_steps):
                    q = (left+right)/2
                    evaluated = self.evaluate(q)
                    error = abs(evaluated[statistic]-threshold)
                    if error <= statistic_tolerance or right-left <= log_width_tolerance:
                        converged = True
                        break
                    if (evaluated[statistic] <= threshold) == (self.cache[left][statistic] <= threshold):
                        left = q
                    else:
                        right = q
                crossings.append({"statistic": statistic, "initial_bracket_values": [first["parameter_value"], last["parameter_value"]],
                                  "last_bracket_log_values": [left, right], "new_point_value": evaluated["parameter_value"],
                                  "statistic_error": abs(evaluated[statistic]-threshold), "iterations": iteration+1,
                                  "converged": converged, "termination": "statistic_tolerance" if error <= statistic_tolerance else
                                      ("log_bracket_tolerance" if right-left <= log_width_tolerance else "iteration_budget")})
        points = [self.add_statistics(self.cache[q]) for q in sorted(self.cache)]
        grid = [point["log_value"] for point in points]
        result = deepcopy(self.original)
        result.update(points=points, point_count=len(points), refinement_crossings=crossings,
                      new_point_evaluations=self.new_evaluations, analytic_family=self.family,
                      refinement_requires_revisit=any(not crossing["converged"] for crossing in crossings),
                      minimum_profile_minus_baseline_sse=min(point["sse"] for point in points)-self.baseline,
                      baseline_requires_revisit=min(point["sse"] for point in points) < self.baseline*(1-1e-6),
                      refinement_notice="Observed threshold crossings were bisected; this does not prove all modes/global minima were discovered. Original physical flat regions include independently verified feasible transfer-equivalence witnesses. Source profiles and attempts remain preserved separately.")
        result["intervals_known_sigma_iid"] = interval_components(grid, [p["lr_known_sigma_iid"] for p in points], threshold, self.lo[self.j], self.hi[self.j])
        result["intervals_unknown_sigma_iid"] = interval_components(grid, [p["lr_unknown_sigma_iid"] for p in points], threshold, self.lo[self.j], self.hi[self.j])
        return result
