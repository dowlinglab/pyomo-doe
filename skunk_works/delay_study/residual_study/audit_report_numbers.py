"""Check bootstrap order statistics and saved prediction arithmetic independently."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered)-1)*probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo]+(position-lo)*(ordered[hi]-ordered[lo])


def best_values(best):
    K,tau,theta=best["coefficients"][0],best["tau"],best["theta"]
    ell,eta=best["noise_parameters"];sigma=best["sigma_total_ML_C"]
    d={"K":K,"tau":tau,"theta":theta,"slow_pole_s":max(tau,theta),"fast_pole_s":min(tau,theta),
       "pole_sum_s":tau+theta,"pole_product_s2":tau*theta,"ell_s":ell,"eta_variance_fraction":eta,
       "sigma_total_C":sigma,"sigma_slow_C":sigma*math.sqrt(1-eta),"sigma_measurement_C":sigma*math.sqrt(eta)}
    d.update({"ambient_"+str(j+1)+"_C":v for j,v in enumerate(best["coefficients"][1:])})
    return d


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--matrix",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    summary=read(a.summary/"uncertainty.json")
    maximum_quantile_error=maximum_sd_error=maximum_prediction_error=0.
    distribution_count=prediction_record_count=0
    bindings={}
    for pair,methods in summary["pairs"].items():
        for method,declared in methods.items():
            cp=a.matrix/(pair+"_"+method)/"checkpoint.json"
            raw=read(cp)
            if hashlib.sha256(cp.read_bytes()).hexdigest()!=declared["binding"]["checkpoint_sha256"]:
                raise AssertionError("Checkpoint changed")
            rows=[r for r in raw["draws"] if r["status"]=="complete"]
            values=[best_values(r["fit"]["best"]) for r in rows]
            if len(rows)!=declared["usable"] or len(raw["draws"])!=declared["requested"]:
                raise AssertionError("Draw denominator mismatch")
            for name,d in declared["distributions"].items():
                x=[r[name] for r in values]
                q=[percentile(x,p) for p in (.025,.5,.975)]
                mean=math.fsum(x)/len(x)
                sd=math.sqrt(math.fsum((v-mean)**2 for v in x)/(len(x)-1))
                maximum_quantile_error=max(maximum_quantile_error,max(abs(v-w)/max(1.,abs(w)) for v,w in zip(q,d["quantiles_025_50_975"])))
                maximum_sd_error=max(maximum_sd_error,abs(sd-d["standard_deviation"])/max(1.,sd))
                distribution_count+=1
            switches=sum(v["tau"]<v["theta"] for v in values)
            if switches!=declared["pole_label_modes"]["tau_lt_theta"]["count"]:
                raise AssertionError("Pole-mode count mismatch")
        path=a.predictions/pair/"summary.json";pred=read(path)
        data=np.load(a.predictions/pair/"predictions.npz")
        for j,row in enumerate(pred["records"]):
            y=data["observed_"+str(j)][1:]
            mean=data["point_"+str(j)][1:]
            differences=[float(t-m) for t,m in zip(y,mean)]
            rmse=math.sqrt(math.fsum(v*v for v in differences)/len(differences))
            bias=math.fsum(differences)/len(differences)
            maximum_prediction_error=max(maximum_prediction_error,abs(rmse-row["RMSE_C"]),abs(bias-row["bias_C"]))
            for method in ("gaussian","innovation_blocks"):
                for family in ("observation","ambient_observation"):
                    low=data[method+"_"+family+"_lower_"+str(j)][1:]
                    high=data[method+"_"+family+"_upper_"+str(j)][1:]
                    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low>high):
                        raise AssertionError("Invalid band")
                    inside=sum(l<=v<=h for l,v,h in zip(low,y,high))/len(y)
                    width=math.fsum(float(h-l) for l,h in zip(low,high))/len(y)
                    expected=row["bands"][method][family]
                    maximum_prediction_error=max(maximum_prediction_error,abs(inside-expected["fraction_inside"]),abs(width-expected["mean_width_C"]))
                if np.any(data[method+"_ambient_observation_lower_"+str(j)]>data[method+"_observation_lower_"+str(j)]+1e-7) or np.any(data[method+"_ambient_observation_upper_"+str(j)]<data[method+"_observation_upper_"+str(j)]-1e-7):
                    raise AssertionError("Ambient envelope fails containment")
            prediction_record_count+=1
        bindings[pair]=hashlib.sha256(path.read_bytes()).hexdigest()
    if max(maximum_quantile_error,maximum_sd_error,maximum_prediction_error)>1e-10 or prediction_record_count!=60:
        raise AssertionError("Summary arithmetic failed")
    result={"status":"passed","bootstrap_distributions_checked":distribution_count,"prediction_records_checked":prediction_record_count,
            "maximum_scaled_quantile_error":maximum_quantile_error,"maximum_scaled_sd_error":maximum_sd_error,
            "maximum_prediction_arithmetic_error":maximum_prediction_error,"prediction_summary_sha256":bindings,
            "uncertainty_summary_sha256":hashlib.sha256((a.summary/"uncertainty.json").read_bytes()).hexdigest(),
            "notice":"Independent scalar order statistics, sums and band inclusion/envelope checks. This validates numerical reporting, not model adequacy, simultaneous coverage or every optimization basin."}
    (a.output/"audit.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
