"""Scaled local information and full-noise likelihood curvature diagnostics."""
import numpy as np

from covariance import Whitening
from mean import fitted_records


def mean_point(result):
    best = result["best"]
    return np.r_[np.log([best["coefficients"][0], best["tau"], best["theta"]]),
                  np.asarray(best["coefficients"][1:])/2.]


def mean_from_point(records, result, point):
    candidate = dict(result, best={"tau": np.exp(point[1]), "theta": np.exp(point[2]),
                                  "coefficients": np.r_[np.exp(point[0]), 2*point[3:]]})
    return np.concatenate([p[1:] for p in fitted_records(records, candidate)])


def jacobian(records, result, step):
    point = mean_point(result)
    columns = []
    for j in range(len(point)):
        delta = np.zeros(len(point)); delta[j] = step
        columns.append((mean_from_point(records, result, point+delta)-
                        mean_from_point(records, result, point-delta))/(2*step))
    return np.column_stack(columns)


def spectrum(information):
    values, vectors = np.linalg.eigh((information+information.T)/2)
    scale = max(float(np.max(np.abs(values))), 1e-30)
    ranks = {str(t): int(np.sum(values > scale*t)) for t in (1e-8, 1e-10, 1e-12)}
    covariance = (vectors/values)@vectors.T if values.min() > scale*1e-10 else None
    return {"eigenvalues": values.tolist(), "eigenvectors_columns": vectors.tolist(),
            "rank_by_relative_threshold": ranks,
            "condition_number": float(values[-1]/values[0]) if values[0] > 0 else None,
            "covariance": None if covariance is None else covariance.tolist()}


def profiled_thermal_information(J):
    nuisance = J[:, 3:]
    if nuisance.shape[1]:
        Q, _ = np.linalg.qr(nuisance, mode="reduced")
        thermal = J[:, :3]-Q@(Q.T@J[:, :3])
    else:
        thermal = J[:, :3]
    return spectrum(thermal.T@thermal)


def hessian(function, point, step):
    n = len(point)
    H = np.zeros((n, n))
    f = function(point)
    for j in range(n):
        delta_j = np.zeros(n); delta_j[j] = step
        H[j, j] = (function(point+delta_j)-2*f+function(point-delta_j))/step**2
        for k in range(j):
            delta_k = np.zeros(n); delta_k[k] = step
            H[j, k] = H[k, j] = (function(point+delta_j+delta_k)-function(point+delta_j-delta_k)
                                 -function(point-delta_j+delta_k)+function(point-delta_j-delta_k))/(4*step**2)
    return H


def local_analysis(records, result, ordinary):
    best = result["best"]
    thermal = np.array([best["coefficients"][0], best["tau"], best["theta"]])
    J = jacobian(records, result, 1e-5)
    coarse = jacobian(records, result, 2e-5)
    sigma = best["sigma_total_ML_C"]
    W = Whitening([d.time[1:] for d in records], result["kernel"], best["noise_parameters"])
    Jw = W(J)/sigma
    conditional = profiled_thermal_information(Jw)
    whole = spectrum(Jw.T@Jw)
    observations = np.concatenate([d.temperature[1:] for d in records])
    ordinary_point = mean_point(ordinary)
    Jo = jacobian(records, ordinary, 1e-5)
    ro = observations-mean_from_point(records, ordinary, ordinary_point)
    n, p = Jo.shape
    variance_iid = float(ro@ro/(n-p))
    iid = profiled_thermal_information(Jo/np.sqrt(variance_iid))
    ordinary_information = spectrum(Jo.T@Jo)
    sandwich = {}
    if ordinary_information["covariance"] is not None:
        bread = np.asarray(ordinary_information["covariance"])
        scores = Jo*ro[:, None]
        for lag in (10, 30, 60):
            meat = scores.T@scores
            cursor = 0
            for data in records:
                count = len(data.time)-1
                section = scores[cursor:cursor+count]
                for k in range(1, min(lag+1, count)):
                    cross = section[k:].T@section[:-k]
                    meat += (1-k/(lag+1))*(cross+cross.T)
                cursor += count
            covariance = bread@meat@bread
            sandwich[str(lag)] = {"covariance_scaled_mean": covariance.tolist(),
                                  "thermal_std_native": (np.exp(ordinary_point[:3])*np.sqrt(np.maximum(np.diag(covariance)[:3], 0))).tolist()}
    output = {"mean_coordinate_order": ["logK", "logtau", "logtheta"]+["ambient_over_2C_"+str(j) for j in range(p-3)],
              "conditional_GLS_thermal": conditional, "conditional_GLS_all_mean": whole,
              "IID_OLS_thermal": iid, "IID_OLS_variance_C2_nominal_dof": variance_iid,
              "OLS_HAC_Bartlett_by_sample_bandwidth": sandwich,
              "jacobian_relative_step_disagreement": float(np.linalg.norm(J-coarse)/max(np.linalg.norm(J), 1e-30)),
              "notice": "Local information ignores box geometry and discrete pole modes. GLS conditional matrices hold fitted covariance fixed; HAC is an asymptotic sample-lag sensitivity, independent across runs. These are diagnostics, not calibrated intervals under unresolved mean/error misspecification."}
    for key, coordinate_values in [("conditional_GLS_thermal", thermal), ("IID_OLS_thermal", np.exp(ordinary_point[:3]))]:
        covariance = output[key]["covariance"]
        output[key]["thermal_std_native"] = None if covariance is None else (coordinate_values*np.sqrt(np.maximum(np.diag(covariance), 0))).tolist()
    if result["kernel"] != "ou_nugget":
        output["observed_curvature"] = {"status": "not implemented for this scoped kernel"}
        return output
    length, eta = best["noise_parameters"]
    if eta <= 1e-7 or eta >= 1-1e-7:
        output["observed_curvature"] = {"status": "withheld at mixture boundary", "eta": eta}
        return output
    point = np.r_[mean_point(result), np.log(length), np.log(sigma*np.sqrt(1-eta)), np.log(sigma*np.sqrt(eta))]
    m = len(mean_point(result))
    def objective(x):
        residual = observations-mean_from_point(records, result, x[:m])
        slow2, measure2 = np.exp(2*x[-2:])
        scale2 = slow2+measure2
        whiten = Whitening([d.time[1:] for d in records], "ou_nugget", [np.exp(x[-3]), measure2/scale2])
        r = whiten(residual)
        return .5*(len(r)*np.log(2*np.pi*scale2)+whiten.logdet+(r@r)/scale2)
    H = hessian(objective, point, 1e-4)
    Hfine = hessian(objective, point, 5e-5)
    direction = np.zeros(len(point)); direction[-2:] = 1.
    scale_identity = float(direction@Hfine@direction)
    spectral = spectrum(Hfine)
    spectral.update({"status": "local diagnostic", "hessian": Hfine.tolist(),
                     "coordinate_order": output["mean_coordinate_order"]+["logell", "log_sigma_slow", "log_sigma_measurement"],
                     "relative_step_disagreement": float(np.linalg.norm(H-Hfine)/max(np.linalg.norm(Hfine), 1e-30)),
                     "common_noise_scale_curvature": scale_identity, "expected_common_scale_curvature_2N": 2*n,
                     "common_scale_relative_identity_error": abs(scale_identity-2*n)/(2*n),
                     "ambient_bound_hits": int(np.sum(np.abs(best["coefficients"][1:]) >= 2-1e-6)),
                     "notice": "Includes noise and ambient coupling. At active bounds this unconstrained curvature is not a regular Gaussian sampling covariance; use profiles/bootstrap and retain limitations."})
    if spectral["common_scale_relative_identity_error"] > 2e-4:
        raise AssertionError("Observed-Hessian variance-scale identity failed")
    covariance = spectral["covariance"]
    spectral["thermal_std_native_diagnostic"] = None if covariance is None else (thermal*np.sqrt(np.maximum(np.diag(covariance)[:3], 0))).tolist()
    output["observed_curvature"] = spectral
    return output
