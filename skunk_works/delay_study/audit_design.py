"""Audit parameter-design information, resources and selected local solutions.

Uses direct simulation/sensitivities rather than cached input maps. All model
points are fixed fitted scenarios, not a probabilistic uncertainty set. Dense
temperature checks do not establish continuous-time or hardware safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_prediction_uncertainty import compare_values
from data import Experiment, load
from inference import ModelSpec, predict, sensitivities
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_gram(J, time, length, sigma):
    if length is not None:
        phi = np.exp(-np.diff(time)/length)
        J = np.vstack([J[0],(J[1:]-phi[:,None]*J[:-1])/np.sqrt(1-phi**2)[:,None]])
    return J.T@J/sigma**2


def direct_scores(F, target):
    """Evaluate the four declared log utilities without optimizer helpers."""
    values,vectors = np.linalg.eigh((F+F.T)/2)
    scores = dict.fromkeys(["D","E","theta","prediction"])
    if values[-1] <= 0:
        return scores
    selected = values > 1e-16*values[-1]
    if selected.all():
        scores["D"] = float(np.linalg.slogdet(F)[1])
        scores["E"] = float(np.log(values[0]))
        inverse = np.linalg.inv(F)
    else:
        inverse = (vectors[:,selected]/values[selected])@vectors[:,selected].T
    for name,weight in [("theta",np.diag([0.,0.,1.])),("prediction",target)]:
        if not selected.all() and np.linalg.norm(weight@vectors[:,~selected]) > 1e-8*max(np.linalg.norm(weight),1e-30):
            continue
        loss = float(np.trace(weight@inverse))
        if loss > 0:
            scores[name] = -float(np.log(loss))
    return scores


def command_resources(amplitudes,knots):
    return {"mean_input_percent":float(np.trapz(amplitudes,knots)/(knots[-1]-knots[0])),
            "maximum_slew_percent_per_s":float(np.max(abs(np.diff(amplitudes))/np.diff(knots))),
            "minimum_input_percent":float(np.min(amplitudes)),"maximum_input_percent":float(np.max(amplitudes))}


def dense_command(amplitudes,knots,config):
    horizon = config["horizon_s"]
    end = horizon+config["audit_tail_s"]
    shutdown = horizon+float(amplitudes[-1])/config["maximum_slew_percent_per_s"]
    if shutdown > end:
        raise RuntimeError("Shutdown exceeds the declared audit tail")
    time = np.unique(np.r_[np.arange(0.,end+config["audit_interval_s"]/2,config["audit_interval_s"]),knots,shutdown])
    command = np.interp(time,knots,amplitudes)
    tail = time > horizon
    command[tail] = np.maximum(0.,amplitudes[-1]-config["maximum_slew_percent_per_s"]*(time[tail]-horizon))
    return time,command


def scenario_label(order,initialization,length):
    return ("pure" if order == 0 else "n"+str(order))+"__"+initialization+"__"+("iid" if length is None else "ou"+str(length))


class DirectStudy:
    def __init__(self,config,fits):
        self.config = config
        self.time = np.arange(0.,config["horizon_s"]+config["observation_interval_s"],config["observation_interval_s"])
        self.knots = np.arange(0.,config["horizon_s"]+config["knot_interval_s"],config["knot_interval_s"])
        self.models,self.scenarios,self.inputs = {},{},{}
        self.derivative_cache,self.mean_cache,self.dense_cache = {},{},{}
        for order in config["ensemble_orders"]:
            for initialization in config["initializations"]:
                spec = ModelSpec(family="pure_delay" if order == 0 else "chain",order=order,initialization=initialization)
                path = Path(fits)/("joint__"+spec.identifier+".json")
                record = json.loads(path.read_text())
                parameters = np.array([record["fit"]["parameters"][name] for name in spec.names])
                model_key = (order,initialization)
                self.models[model_key] = (parameters,spec)
                self.inputs[path.name] = sha(path)
                reference = Experiment("mean_target",self.time,np.full(len(self.time),config["ambient_C"]),
                                       np.full(len(self.time),config["mean_input_percent"]),config["ambient_C"])
                J = sensitivities(parameters,[reference],spec)
                target = J.T@J/len(J)
                for length in config["noise_lengths_s"]:
                    prior = np.zeros((3,3))
                    for name in record["datasets"]:
                        data = load(name)
                        prior += weighted_gram(sensitivities(parameters,[data],spec),data.time[1:],length,config["sigma_C"])
                    self.scenarios[scenario_label(order,initialization,length)] = (model_key,length,prior,target)

    def experiment(self,amplitudes,time=None,command=None):
        time = self.time if time is None else time
        command = np.interp(time,self.knots,amplitudes) if command is None else command
        return Experiment("independent_design_audit",time,np.full(len(time),self.config["ambient_C"]),command,self.config["ambient_C"])

    def information(self,name,amplitudes,prior):
        model,length,historical,target = self.scenarios[name]
        key = (model,np.asarray(amplitudes,dtype=float).tobytes())
        if key not in self.derivative_cache:
            parameters,spec = self.models[model]
            data = self.experiment(amplitudes)
            self.derivative_cache[key] = sensitivities(parameters,[data],spec)
        F = weighted_gram(self.derivative_cache[key],self.time[1:],length,self.config["sigma_C"])
        return F+historical if prior else F,target

    def coarse_violation(self,amplitudes):
        violations = [abs(command_resources(amplitudes,self.knots)["mean_input_percent"]-self.config["mean_input_percent"]),
                      float(np.max(-amplitudes)),float(np.max(amplitudes-100)),
                      float(np.max(abs(np.diff(amplitudes))-self.config["maximum_slew_percent_per_s"]*np.diff(self.knots)))]
        for model,(parameters,spec) in self.models.items():
            key = (model,np.asarray(amplitudes,dtype=float).tobytes())
            if key not in self.mean_cache:
                self.mean_cache[key] = predict(self.experiment(amplitudes),parameters,spec)
            violations.append(float(np.max(self.mean_cache[key])-self.config["maximum_temperature_C"]))
        return max(violations)

    def dense(self,amplitudes):
        key = np.asarray(amplitudes,dtype=float).tobytes()
        if key not in self.dense_cache:
            time,command = dense_command(amplitudes,self.knots,self.config)
            data = self.experiment(amplitudes,time,command)
            rows = {}
            for (order,initialization),(parameters,spec) in self.models.items():
                temperature = predict(data,parameters,spec)
                index = int(np.argmax(temperature))
                rows[scenario_label(order,initialization,None)] = {"maximum_C":float(temperature[index]),
                    "time_of_maximum_s":float(time[index]),"exceeds_assumed_limit":bool(temperature[index] > self.config["maximum_temperature_C"]+1e-6)}
            self.dense_cache[key] = rows
        return self.dense_cache[key]


def audit_record(record,task,study,errors):
    key,names,criterion,prior,aggregation = task
    config = study.config
    if record["key"] != key or record["utility_scenarios"] != names or record["requires_revisit"]:
        raise RuntimeError("Design identity or unresolved review flag: "+key)
    compare_values(record["knots_s"],study.knots,"knots_s",errors)
    optimization,evaluations = record["optimization"],record["evaluations"]
    if (optimization["criterion"],optimization["include_prior"],optimization["aggregation"]) != (criterion,prior,aggregation):
        raise RuntimeError("Optimization criterion/prior/aggregation changed")
    expected_starts = {"constant","sine_300s","sine_150s","multisine","random_binary_knots","alternating_knots"}
    if set(evaluations) != expected_starts|{"selected_local_design"} or {a["start"] for a in optimization["attempts"]} != expected_starts or len(optimization["attempts"]) != 6:
        raise RuntimeError("Missing common candidate or optimization attempt")
    candidate_scores = {}
    for waveform,evaluation in evaluations.items():
        amplitudes = np.asarray(evaluation["amplitudes_percent"])
        resources = command_resources(amplitudes,study.knots)
        for field in ("mean_input_percent","maximum_slew_percent_per_s"):
            compare_values(np.array(evaluation[field]),np.array(resources[field]),field,errors)
        if resources["minimum_input_percent"] < -1e-7 or resources["maximum_input_percent"] > 100+1e-7 or abs(resources["mean_input_percent"]-config["mean_input_percent"]) > 1e-6 or resources["maximum_slew_percent_per_s"] > config["maximum_slew_percent_per_s"]+1e-6:
            raise RuntimeError("A declared candidate violates amplitude/dose/slew assumptions")
        scenario_rows = evaluation["scenario_information"]
        if len(scenario_rows) != len(study.scenarios) or {r["scenario"] for r in scenario_rows} != set(study.scenarios):
            raise RuntimeError("Missing or duplicated information scenario")
        candidate_scores[waveform] = {}
        for row in scenario_rows:
            F,target = study.information(row["scenario"],amplitudes,prior)
            # Near an exact null direction, finite differencing a sum and
            # summing differenced input basis responses can leave different
            # tiny cross terms. Check the matrix at its declared IO scale,
            # and check individual eigenvalues and target estimability below.
            saved_F = np.asarray(row["information_log_io"])
            scale = max(1.,np.linalg.norm(F))
            compare_values(saved_F/scale,F/scale,"information_log_io_relative",errors,atol=1e-8,rtol=0.)
            errors["information_log_io_absolute"] = max(errors.get("information_log_io_absolute",0.),float(np.max(abs(saved_F-F))))
            values,vectors = np.asarray(row["eigenvalues_ascending"]),np.asarray(row["eigenvectors_columns"])
            compare_values(values,np.linalg.eigvalsh(F),"information_eigenvalues",errors,atol=1e-6,rtol=1e-7)
            compare_values(vectors.T@vectors,np.eye(3),"eigenvector_orthogonality",errors,atol=1e-9,rtol=1e-9)
            # A normwise residual respects the information scale. Entrywise
            # tolerances on a weak eigenvector component can reject harmless
            # cancellation from direct versus input-mapped finite differences.
            eigen_residual = (F@vectors-vectors*values)/max(1.,np.linalg.norm(F))
            compare_values(eigen_residual,np.zeros((3,3)),"eigenpair_relative_residual",errors,atol=1e-7,rtol=0.)
            scores = direct_scores(F,target)
            for name,value in scores.items():
                if value is None:
                    if row["scores"][name] is not None:
                        raise RuntimeError("A singular/unestimable utility was assigned a finite score")
                else:
                    compare_values(np.array(row["scores"][name]),np.array(value),"score_"+name,errors,atol=1e-6,rtol=1e-7)
            candidate_scores[waveform][row["scenario"]] = scores[criterion]
        dense = evaluation["dense_audit"]
        if dense["interval_s"] != config["audit_interval_s"] or dense["tail_s"] != config["audit_tail_s"]:
            raise RuntimeError("Dense audit resources changed")
        expected_dense = study.dense(amplitudes)
        if len(dense["scenarios"]) != len(expected_dense) or {r["scenario"] for r in dense["scenarios"]} != set(expected_dense):
            raise RuntimeError("Dense audit omits or duplicates a fitted mean")
        for row in dense["scenarios"]:
            expected = expected_dense[row["scenario"]]
            for field in ("maximum_C","time_of_maximum_s"):
                compare_values(np.array(row[field]),np.array(expected[field]),"dense_"+field,errors,atol=1e-7,rtol=1e-9)
            if row["exceeds_assumed_limit"] != expected["exceeds_assumed_limit"] or (waveform == "selected_local_design" and expected["exceeds_assumed_limit"]):
                raise RuntimeError("Dense temperature flag or selected constraint failed")
    reference_name = "constant" if all(candidate_scores["constant"][n] is not None for n in names) else "multisine"
    compare_values(optimization["reference_amplitudes"],evaluations[reference_name]["amplitudes_percent"],"reference_amplitudes",errors)
    reference = np.array([candidate_scores[reference_name][n] for n in names])
    compare_values(optimization["reference_scores"],reference,"reference_scores",errors)
    for attempt in optimization["attempts"]:
        if attempt.get("status") == "singular_start_skipped":
            if attempt["success"] or all(candidate_scores[attempt["start"]][n] is not None for n in names):
                raise RuntimeError("A finite candidate was labelled a singular skipped start")
            continue
        if not 0 <= attempt["iterations"] <= config["maximum_iterations"]:
            raise RuntimeError("An optimizer attempt exceeds its declared iteration budget")
        amplitudes = np.asarray(attempt["amplitudes"])
        scores = [direct_scores(*study.information(n,amplitudes,prior))[criterion] for n in names]
        if any(s is None for s in scores):
            if attempt["objective"] is not None:
                raise RuntimeError("Finite objective for a singular attempt")
        else:
            relative = np.asarray(scores)-reference
            objective = -float(np.mean(relative) if aggregation == "mean" else np.min(relative))
            compare_values(np.array(attempt["objective"]),np.array(objective),"attempt_objective",errors,atol=1e-6,rtol=1e-7)
        compare_values(np.array(attempt["maximum_constraint_violation"]),np.array(study.coarse_violation(amplitudes)),"coarse_constraint_violation",errors,atol=1e-7,rtol=1e-7)
    accepted = [a for a in optimization["attempts"] if a["success"] and a.get("objective") is not None and a["maximum_constraint_violation"] < 1e-6]
    if not accepted or optimization["best"] != min(accepted,key=lambda a:a["objective"]):
        raise RuntimeError("Selected solution is not the best recorded converged feasible start")
    compare_values(evaluations["selected_local_design"]["amplitudes_percent"],optimization["best"]["amplitudes"],"selected_amplitudes",errors)
    selected = np.asarray(evaluations["selected_local_design"]["amplitudes_percent"])
    if study.coarse_violation(selected) >= 1e-6:
        raise RuntimeError("Selected design violates the actual coarse constraints")
    return {"case":key,"criterion":criterion,"aggregation":aggregation,"include_historical_information":prior,
            "relative_utility":-optimization["best"]["objective"],"successful_feasible_starts":len(accepted),
            "dense_maximum_C":max(r["maximum_C"] for r in study.dense(selected).values()),
            **command_resources(selected,study.knots)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=HERE/"results")
    parser.add_argument("--design",type=Path)
    parser.add_argument("--allow-trial",action="store_true")
    args = parser.parse_args()
    root = args.design or args.results/"design"
    manifest_path = root/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["failed"] or (manifest["filter"] and not args.allow_trial):
        raise RuntimeError("Completed production or explicit completed trial required")
    config_path = HERE/"design_config.json"
    config = json.loads(config_path.read_text())
    identity = manifest["identity"]
    if identity["config_sha256"] != sha(config_path) or any(sha(HERE/n) != h for n,h in identity["sources"].items()):
        raise RuntimeError("Audit requires measured source/configuration")
    study = DirectStudy(config,args.results/"fits")
    if study.inputs != identity["fit_sha256"]:
        raise RuntimeError("Fitted scenario inputs changed")
    tasks = []
    for order in config["nominal_orders"]:
        for length in config["noise_lengths_s"]:
            name = scenario_label(order,"cold",length)
            for criterion in config["criteria"]:
                for prior in config["include_prior"]:
                    tasks.append(("nominal__"+name+"__"+criterion+"__prior"+str(int(prior)),[name],criterion,prior,"mean"))
    for aggregation in config["aggregations"]:
        for criterion in config["criteria"]:
            for prior in config["include_prior"]:
                tasks.append(("ensemble__"+aggregation+"__"+criterion+"__prior"+str(int(prior)),list(study.scenarios),criterion,prior,aggregation))
    tasks = [t for t in tasks if manifest["filter"] in t[0]]
    if len(tasks) != manifest["expected_jobs"] or set(manifest["completed"]) != {t[0] for t in tasks} or (not manifest["filter"] and len(tasks) != 112):
        raise RuntimeError("Parameter-design task matrix is incomplete")
    errors,inputs,rows = {},{str(manifest_path):sha(manifest_path)},[]
    for task in tasks:
        path = root/(task[0]+".json")
        record = json.loads(path.read_text())
        if record["identity"] != identity or manifest["completed"][task[0]]["requires_revisit"]:
            raise RuntimeError("Design source identity/review flag mismatch")
        rows.append(audit_record(record,task,study,errors))
        inputs[str(path)] = sha(path)
    pd.DataFrame(rows).to_csv(root/"audited_summary.csv",index=False,float_format="%.12g")
    report = {"status":"passed_trial" if manifest["filter"] else "passed","designs":len(rows),"maximum_errors":errors,
        "auditor_sources_sha256":{n:sha(HERE/n) for n in ["audit_design.py","audit_prediction_uncertainty.py"]},
        "checked_measurement_identity":identity,"input_sha256":inputs,"summary_sha256":sha(root/"audited_summary.csv"),
        "scope":"Direct-input sensitivity/information scores and eigenpairs for all saved waveform/scenario evaluations; every attempt objective/coarse constraint, selected-start rule, amplitude/dose/slew resources and all-model dense maxima through shutdown. Reuses separately verified simulator/sensitivities. No global optimality, continuous-time temperature bound, probabilistic scenario coverage or hardware certification."}
    save_json(root/"audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in {"input_sha256","checked_measurement_identity","scope"}}))


if __name__ == "__main__":
    main()
