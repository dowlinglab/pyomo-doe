"""Audit saved pointwise bands independently of their producer's band helpers.

Reuses the separately verified simulator/sensitivities and source fit records.
Assembles covariance/OU whitening directly, checks every saved band value and
inverts no mixture: saved mixture endpoints must satisfy their defining CDF.
Run after the prediction phase, outside another long numerical pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm

from data import load
from inference import ModelSpec, parameter_box, predict, sensitivities
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_values(actual, expected, label, errors, atol=1e-8, rtol=1e-7):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        raise RuntimeError("Shape or undefined-value disagreement: "+label)
    defined = np.isfinite(b)
    if not np.isfinite(a[defined]).all() or np.isinf(b).any():
        raise RuntimeError("Nonfinite value: "+label)
    maximum = float(np.max(abs(a[defined]-b[defined]))) if defined.any() else 0.
    errors[label] = max(errors.get(label, 0.), maximum)
    if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
        raise RuntimeError("Saved numeric values disagree: "+label+"; maximum error "+str(maximum))


def covariance_basis(J):
    _, singular, V = np.linalg.svd(J, full_matrices=False)
    selected = singular > singular[0]*1e-8
    bread = (V[selected].T/singular[selected]**2)@V[selected]
    return bread, V[~selected].T, int(selected.sum())


def direct_local_covariances(J, residual, sizes, sigma, bandwidths):
    bread, null, rank = covariance_basis(J)
    dof = len(residual)-rank
    variance = float(residual@residual/max(1, dof))
    covariances = {"iid_known_sigma":bread*sigma**2,
                   "iid_unknown_quotient_dof":bread*variance}
    for bandwidth in bandwidths:
        # Reset lagged products at each experiment boundary. The finite-sample
        # factor uses the identifiable rank, including original-model gauges.
        meat = np.zeros_like(bread)
        start = 0
        for count in sizes:
            score = J[start:start+count]*residual[start:start+count,None]
            meat += np.einsum("ti,tj->ij",score,score)
            for lag in range(1,min(bandwidth,count-1)+1):
                product = np.einsum("ti,tj->ij",score[lag:],score[:-lag])
                meat += (1-lag/(bandwidth+1))*(product+product.T)
            start += count
        if start != len(residual):
            raise RuntimeError("Experiment sizes do not partition the retained residuals")
        covariances["hac"+str(bandwidth)+"_quotient_dof"] = len(residual)/max(1,dof)*bread@meat@bread
    return covariances, null, rank, dof, variance


def direct_ou_information(parameters, calibration, spec, length, sigma):
    transformed = []
    for data in calibration:
        time = data.time[1:]
        J = sensitivities(parameters,[data],spec)
        phi = np.exp(-np.diff(time)/length)
        whitened = np.vstack([J[0],(J[1:]-phi[:,None]*J[:-1])/np.sqrt(1-phi**2)[:,None]])/sigma
        transformed.append(whitened)
    return covariance_basis(np.vstack(transformed))


def target_variance(J, covariance, null, globally_undefined=False):
    covariance = np.asarray(covariance)
    covariance = (covariance+covariance.T)/2
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] < -1e-9*max(np.max(abs(eigenvalues)),1e-30):
        raise RuntimeError("Indefinite prediction covariance")
    allowed = np.ones(len(J),dtype=bool)
    if null.size:
        allowed = np.sqrt(np.sum((J@null)**2,axis=1)) <= 1e-8*np.maximum(np.sqrt(np.sum(J*J,axis=1)),1e-30)
    if globally_undefined:
        allowed[:] = False
    variance = np.maximum(np.sum((J@covariance)*J,axis=1),0.)
    variance[~allowed] = np.nan
    return variance, allowed


def audit_group(frame, data, role, method, mean, variance, allowed, errors,
                lower=None, upper=None, median=None, future_variance=None, mixture_means=None,
                mixture_sigma=None):
    if len(frame) != len(data.time):
        raise RuntimeError("Missing or duplicated target times")
    if abs(mean[0]-data.temperature[0]) > 1e-10:
        raise RuntimeError("The supplied mean does not condition on the first temperature")
    for name, values in [("time_s",data.time),("input_percent",data.input),
                         ("measured_C",data.temperature),("mean_C",mean)]:
        compare_values(frame[name],values,name,errors)
    if not frame.role.eq(role).all() or not frame.method.eq(method).all():
        raise RuntimeError("Target group labels changed")
    conditioned = np.r_[True,np.zeros(len(data.time)-1,dtype=bool)]
    expected_estimable = np.r_[False,allowed]
    if not np.array_equal(frame.first_temperature_conditions_state.to_numpy(),conditioned):
        raise RuntimeError("The initial temperature must be marked as conditioned")
    if not np.array_equal(frame.estimable.to_numpy(),expected_estimable):
        raise RuntimeError("Saved target estimability disagrees with global/local checks")
    compare_values(frame.mean_variance_C2,np.r_[np.nan,variance],"mean_variance_C2",errors)
    z = norm.ppf(.975)
    lo = mean[1:]-z*np.sqrt(variance) if lower is None else lower
    hi = mean[1:]+z*np.sqrt(variance) if upper is None else upper
    compare_values(frame.mean_lower_C,np.r_[np.nan,lo],"mean_lower_C",errors)
    compare_values(frame.mean_upper_C,np.r_[np.nan,hi],"mean_upper_C",errors)
    if median is not None:
        compare_values(frame.bootstrap_mean_median_C,np.r_[np.nan,median],"bootstrap_mean_median_C",errors)
    prediction_defined = future_variance is not None or mixture_means is not None
    if future_variance is not None:
        half = z*np.sqrt(variance+future_variance)
        compare_values(frame.prediction_lower_C,np.r_[np.nan,mean[1:]-half],"prediction_lower_C",errors)
        compare_values(frame.prediction_upper_C,np.r_[np.nan,mean[1:]+half],"prediction_upper_C",errors)
    elif mixture_means is not None:
        endpoints = np.vstack([frame.prediction_lower_C.to_numpy()[1:],frame.prediction_upper_C.to_numpy()[1:]])
        cdf = ndtr((endpoints[:,None,:]-mixture_means[None,:,:])/mixture_sigma).mean(axis=1)
        compare_values(cdf,np.broadcast_to(np.array([.025,.975])[:,None],cdf.shape),
                       "mixture_cdf",errors,atol=2e-8,rtol=0.)
        if not frame.iloc[0][["prediction_lower_C","prediction_upper_C"]].isna().all():
            raise RuntimeError("An observation band was assigned to the conditioned first point")
    else:
        for column in ("prediction_lower_C","prediction_upper_C"):
            if column in frame and not frame[column].isna().all():
                raise RuntimeError("An observation band was assigned without a future error law")
    widths = hi-lo
    summary = {"role":role,"experiment":data.name,"method":method,
               "conditional_points":len(data.time)-1,"estimable_points":int(allowed.sum()),
               "mean_band_median_width_C":float(np.nanmedian(widths)) if allowed.any() else None,
               "mean_band_maximum_width_C":float(np.nanmax(widths)) if allowed.any() else None}
    if prediction_defined:
        inside = ((frame.measured_C >= frame.prediction_lower_C)&(frame.measured_C <= frame.prediction_upper_C)).to_numpy()[1:]
        summary["observed_fraction_inside_prediction_band_defined_points"] = float(inside[allowed].mean()) if allowed.any() else None
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=HERE/"results")
    parser.add_argument("--prediction",type=Path)
    parser.add_argument("--allow-trial",action="store_true")
    args = parser.parse_args()
    root = args.prediction or args.results/"prediction_uncertainty"
    manifest_path = root/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["failed"]:
        raise RuntimeError("Only a terminal complete prediction run can be audited")
    if manifest["filter"] and not args.allow_trial:
        raise RuntimeError("A filtered run requires explicit trial-audit mode")
    config_path = HERE/"prediction_config.json"
    config = json.loads(config_path.read_text())
    identity = manifest["identity"]
    if sha(config_path) != identity["config_sha256"]:
        raise RuntimeError("Prediction configuration changed")
    for name,digest in identity["source_sha256"].items():
        if sha(HERE/name) != digest:
            raise RuntimeError("Audit requires the measured scientific source: "+name)
    jobs = []
    for path in sorted((args.results/"fits").glob("*.json")):
        record = json.loads(path.read_text())
        spec = ModelSpec(**record["fit"]["spec"])
        if spec.coordinates != "io" and spec.family != "original":
            continue
        jobs.append(("local__"+path.stem,"local",path,None))
        if spec.family != "original":
            jobs.append(("ou__"+path.stem,"ou",path,None))
            if record["calibration"] in config["bootstrap_calibrations"]:
                jobs.extend(("bootstrap__"+path.stem+"__"+m,"bootstrap",path,m) for m in config["bootstrap_methods"])
    jobs = [j for j in jobs if manifest["filter"] in j[0]]
    if not manifest["filter"] and len(jobs) != 131:
        raise RuntimeError("The full prediction matrix requires all 131 declared cases")
    if len(jobs) != manifest["expected_jobs"] or set(manifest["completed"]) != {j[0] for j in jobs}:
        raise RuntimeError("Prediction matrix differs from its recorded task set")
    errors,inputs,rows = {},{str(manifest_path):sha(manifest_path)},[]
    total_points,group_count = 0,0
    for key,phase,fit_path,method in jobs:
        path = root/(key+".json")
        saved,record = json.loads(path.read_text()),json.loads(fit_path.read_text())
        spec = ModelSpec(**record["fit"]["spec"])
        if saved["identity"] != identity or saved["spec"] != record["fit"]["spec"] or saved["case"] != fit_path.stem or saved["phase"] != phase or saved["method"] != method:
            raise RuntimeError("Prediction case identity mismatch: "+key)
        if saved["working_known_sigma_C"] != config["known_sigma_C"] or saved["null_target_relative_tolerance"] != 1e-8:
            raise RuntimeError("Saved working-noise or estimability convention changed")
        used = {str(fit_path):sha(fit_path)}
        parameter_record = record["fit"]["parameters"]
        source,draws = None,None
        if phase == "ou":
            source_path = args.results/"ou"/fit_path.name
            source = json.loads(source_path.read_text())
            if source["fit_sha256"] != sha(fit_path):
                raise RuntimeError("OU source fit changed")
            used[str(source_path)] = sha(source_path)
            parameter_record = source["fit"]["parameters"]
        elif phase == "bootstrap":
            source_path = args.results/"bootstrap"/(fit_path.stem+"__"+method+".json")
            source = json.loads(source_path.read_text())
            arrays = source_path.with_suffix(".npz")
            if source["fit_sha256"] != sha(fit_path) or source["arrays_sha256"] != sha(arrays):
                raise RuntimeError("Bootstrap source identities changed")
            used.update({str(source_path):sha(source_path),str(arrays):sha(arrays)})
            with np.load(arrays) as data:
                if not data["completed"].all() or not data["success"].all() or len(data["parameters"]) != 400:
                    raise RuntimeError("Bootstrap prediction does not have 400 successful recorded draws")
                draws = data["parameters"].copy()
        normalize = lambda d:{str(Path(p).parent.name+"/"+Path(p).name):v for p,v in d.items()}
        if normalize(used) != normalize(saved["input_sha256"]):
            raise RuntimeError("Prediction source file hashes differ")
        inputs.update(used)
        parameters = np.array([parameter_record[n] for n in spec.names])
        calibration = [load(n) for n in record["datasets"]]
        covariance,null,noise_variance = {},None,None
        if phase == "local":
            J = sensitivities(parameters,calibration,spec)
            residual = np.concatenate([(predict(d,parameters,spec)-d.temperature)[1:] for d in calibration])
            covariance,null,rank,dof,noise_variance = direct_local_covariances(J,residual,[len(d.time)-1 for d in calibration],config["known_sigma_C"],config["hac_bandwidth_samples"])
            if saved["information"]["rank"] != rank or saved["information"]["residual_dof"] != dof:
                raise RuntimeError("Incorrect identifiable quotient rank or residual degrees of freedom")
            compare_values(saved["information"]["singular_values"],np.linalg.svd(J,compute_uv=False),"information_singular_values",errors)
            compare_values(np.array(saved["information"]["estimated_future_iid_variance_C2"]),np.array(noise_variance),"estimated_iid_variance_C2",errors)
        elif phase == "ou":
            bread,null,rank = direct_ou_information(parameters,calibration,spec,source["fit"]["correlation_length_s"],source["fit"]["sigma_stationary_ML_C"])
            covariance = {"ou_conditional_noise":bread}
            observed = source["uncertainty"]["observed_cov_log_mean_and_length"]
            if observed is not None:
                covariance["ou_observed_noise_profiled"] = np.asarray(observed)[:3,:3]
            if saved["information"]["rank"] != rank:
                raise RuntimeError("OU information rank changed")
            if saved["information"]["observed_hessian_covariance_available"] != (observed is not None):
                raise RuntimeError("OU observed-covariance availability flag changed")
        band_path = root/saved["bands_file"]
        if not band_path.resolve().is_relative_to(root.resolve()) or sha(band_path) != saved["bands_sha256"]:
            raise RuntimeError("Saved band file path or bytes changed")
        frame = pd.read_csv(band_path)
        experiments = [("calibration",d) for d in calibration]+[("validation",load(v["dataset"])) for v in record["validation"]]
        expected_groups = set()
        recomputed_summaries = []
        max_cdf_error = 0.
        for role,data in experiments:
            mean = predict(data,parameters,spec)
            if phase == "bootstrap":
                means = np.stack([predict(data,p,spec)[1:] for p in draws])
                quantiles = np.percentile(means,[2.5,50,97.5],axis=0)
                methods = {"bootstrap_"+method:None}
            else:
                J = sensitivities(parameters,[data],spec)
                methods = covariance
            for name,C in methods.items():
                expected_groups.add((role,data.name,name))
                part = frame[(frame.role == role)&(frame.experiment == data.name)&(frame.method == name)]
                kwargs = {}
                if phase == "bootstrap":
                    variance,allowed = np.var(means,axis=0,ddof=1),np.ones(len(mean)-1,dtype=bool)
                    kwargs.update(lower=quantiles[0],upper=quantiles[2],median=quantiles[1])
                    if method == "parametric":
                        kwargs.update(mixture_means=means,mixture_sigma=config["known_sigma_C"])
                else:
                    globally_undefined = ("theta" in record["uncertainty"]["nonestimable_native_parameters"] and
                        not(spec.initialization == "input_equilibrated" and np.all(data.input == data.input[0])))
                    variance,allowed = target_variance(J,C,null,globally_undefined)
                    if name == "iid_known_sigma":
                        kwargs["future_variance"] = config["known_sigma_C"]**2
                    elif name == "iid_unknown_quotient_dof":
                        kwargs["future_variance"] = noise_variance
                    elif phase == "ou":
                        kwargs["future_variance"] = source["fit"]["sigma_stationary_ML_C"]**2
                summary = audit_group(part,data,role,name,mean,variance,allowed,errors,**kwargs)
                if "mixture_means" in kwargs:
                    endpoints = np.vstack([part.prediction_lower_C.to_numpy()[1:],part.prediction_upper_C.to_numpy()[1:]])
                    max_cdf_error = max(max_cdf_error,float(np.max(abs(ndtr((endpoints[:,None,:]-means[None,:,:])/config["known_sigma_C"]).mean(axis=1)-np.array([.025,.975])[:,None]))))
                if phase == "local" and name == "iid_known_sigma" and "theta" in record["uncertainty"]["nonestimable_native_parameters"]:
                    lower,upper = parameter_box(spec)
                    grid = np.unique(np.r_[np.geomspace(lower[2],upper[2],config["null_delay_envelope_points"]),parameters[2]])
                    envelope = np.stack([predict(data,np.r_[parameters[:2],theta],spec) for theta in grid])
                    compare_values(part.theta_exploration_lower_C,np.min(envelope,axis=0),"theta_exploration_lower_C",errors)
                    compare_values(part.theta_exploration_upper_C,np.max(envelope,axis=0),"theta_exploration_upper_C",errors)
                else:
                    for column in ("theta_exploration_lower_C","theta_exploration_upper_C"):
                        if column in part and not part[column].isna().all():
                            raise RuntimeError("A delay exploration envelope was assigned to an undeclared target/method")
                recomputed_summaries.append(summary)
                rows.append(dict(case=key,phase=phase,**summary))
                total_points += len(part)
                group_count += 1
        groups = set(map(tuple,frame[["role","experiment","method"]].drop_duplicates().to_numpy()))
        if groups != expected_groups or len(saved["summaries"]) != len(recomputed_summaries):
            raise RuntimeError("Missing or extra prediction target groups")
        for actual,expected in zip(saved["summaries"],recomputed_summaries):
            for field,value in expected.items():
                if isinstance(value,float):
                    compare_values(np.array(actual[field]),np.array(value),"summary_"+field,errors)
                elif actual[field] != value:
                    raise RuntimeError("Prediction summary label/count/undefined value differs")
        compare_values(np.array(saved["maximum_gaussian_mixture_cdf_residual"]),np.array(max_cdf_error),"recorded_maximum_mixture_cdf_error",errors,atol=1e-10,rtol=1e-4)
        inputs.update({str(path):sha(path),str(band_path):sha(band_path)})
    pd.DataFrame(rows).to_csv(root/"audited_summary.csv",index=False,float_format="%.12g")
    report={"status":"passed_trial" if manifest["filter"] else "passed","cases":len(jobs),
            "target_groups":group_count,"saved_points_checked":total_points,"maximum_errors":errors,
            "input_sha256":inputs,"auditor_sha256":sha(Path(__file__)),"checked_measurement_identity":identity,
            "summary_sha256":sha(root/"audited_summary.csv"),
            "scope":"Every saved mean/variance/band point, conditioned-state/estimability flag, target group and descriptive summary. Independent covariance/HAC algebra, manual OU whitening and mixture CDF checks reuse separately verified simulation/sensitivities and recorded observed OU covariance. No repeated-experiment, simultaneous or model-discrepancy coverage claim."}
    save_json(root/"audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in {"input_sha256","scope","checked_measurement_identity"}}))


if __name__ == "__main__":
    main()
