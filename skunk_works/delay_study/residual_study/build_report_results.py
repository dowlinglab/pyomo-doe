"""Generate report tables/figures exclusively from terminal, hash-bound results."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values, digits=3):
    return "["+format(values[0], "."+str(digits)+"f")+", "+format(values[2], "."+str(digits)+"f")+"]"


def table(headers, rows, widths=None):
    columns = "l"+r">{\raggedleft\arraybackslash}X"*(len(headers)-1) if widths is None else widths
    return ("\\begin{center}\\small\n\\begin{tabularx}{\\linewidth}{@{}"+columns+"@{}}\n\\toprule\n"+
            " & ".join(headers)+"\\\\\n\\midrule\n"+
            "\n".join(" & ".join(row)+"\\\\" for row in rows)+
            "\n\\bottomrule\n\\end{tabularx}\n\\end{center}\n")


def prediction_metrics(summary):
    rows = summary["records"]
    count = sum(r["n"] for r in rows)
    result = {"n": count, "records": len(rows),
              "RMSE_C": float(np.sqrt(sum(r["n"]*r["RMSE_C"]**2 for r in rows)/count)),
              "ordinary_RMSE_C": float(np.sqrt(sum(r["n"]*r["ordinary_mean_RMSE_C"]**2 for r in rows)/count)),
              "conditional_reconstruction_RMSE_C": float(np.sqrt(sum(r["n"]*r["conditional_ambient_reconstruction"]["RMSE_C"]**2 for r in rows)/count)),
              "per_record_RMSE_range_C": [min(r["RMSE_C"] for r in rows), max(r["RMSE_C"] for r in rows)]}
    for method in ("gaussian", "innovation_blocks"):
        result[method] = {}
        for family in ("observation", "ambient_observation"):
            result[method][family] = {name: float(sum(r["n"]*r["bands"][method][family][name] for r in rows)/count)
                                      for name in ("fraction_inside", "mean_width_C")}
            result[method][family]["per_record_fraction_range"] = [min(r["bands"][method][family]["fraction_inside"] for r in rows),
                                                                 max(r["bands"][method][family]["fraction_inside"] for r in rows)]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--uncertainty-summary", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--profile-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output/"figures"; figures.mkdir(exist_ok=True)
    U = read(args.uncertainty_summary/"uncertainty.json")
    profiles = read(args.profile_audit/"audit.json")
    if profiles["status"] != "passed":
        raise ValueError("Resolve or explicitly revise the profile-audit gate before report generation")
    P, I, bindings = {}, {}, {}
    for pair in ["P"+str(i) for i in range(1, 7)]:
        P[pair] = read(args.predictions/pair/"summary.json")
        audit = read(args.predictions/pair/"audit.json")
        if audit["status"] != "passed for declared numerical checks" or digest(args.predictions/pair/"predictions.npz") != audit["prediction_sha256"]:
            raise ValueError("Prediction audit/binding mismatch")
        fit_hash = digest(args.results/"pairs"/pair/"fit.json")
        I[pair] = read(args.results/"information"/pair/"information.json")
        if any(x != fit_hash for x in [P[pair]["binding"]["reference_fit_sha256"], I[pair]["binding"]["reference_fit_sha256"],
                                      U["pairs"][pair]["gaussian"]["binding"]["reference_fit_sha256"]]):
            raise ValueError("Incompatible pair references")
        for method, key in [("gaussian", "gaussian_checkpoint_sha256"), ("blocks60", "innovation_blocks_checkpoint_sha256")]:
            if P[pair]["binding"][key] != U["pairs"][pair][method]["binding"]["checkpoint_sha256"]:
                raise ValueError("Prediction and uncertainty used different simulation states")
        bindings[pair] = {"fit_sha256": fit_hash, "prediction_summary_sha256": digest(args.predictions/pair/"summary.json"),
                          "information_sha256": digest(args.results/"information"/pair/"information.json")}
    metrics = {pair: prediction_metrics(s) for pair, s in P.items()}
    common_names = set.intersection(*[{r["record"] for r in s["records"]} for s in P.values()])
    if len(common_names) != 3 or any("step" in name for name in common_names):
        raise ValueError("The declared six-pair design should leave three common sine evaluation records")
    for pair, summary in P.items():
        selected = [r for r in summary["records"] if r["record"] in common_names]
        metrics[pair]["common_three_sine_RMSE_C"] = float(np.sqrt(sum(r["n"]*r["RMSE_C"]**2 for r in selected)/sum(r["n"] for r in selected)))
    first = U["pairs"]["P1"]
    rows = []
    range_rows = []
    for k, name in enumerate(("K", "tau", "theta")):
        rows.append(["$"+name+"$", format(I["P1"]["IID_OLS_thermal"]["thermal_std_native"][k], ".4g"),
                     format(I["P1"]["OLS_HAC_Bartlett_by_sample_bandwidth"]["30"]["thermal_std_native"][k], ".4g"),
                     format(I["P1"]["conditional_GLS_thermal"]["thermal_std_native"][k], ".4g"),
                     format(I["P1"]["observed_curvature"]["thermal_std_native_diagnostic"][k], ".4g")])
        range_rows.append(["$"+name+"$",
                     interval(first["gaussian"]["distributions"][name]["quantiles_025_50_975"], 3),
                     interval(first["blocks60"]["distributions"][name]["quantiles_025_50_975"], 3)])
    tier1 = table(["Parameter", "IID SD", "HAC(30) SD", "GLS SD", "Full-curvature SD"], rows)
    tier1 += ("SD units are $\\C/\\%$ for $K$ and seconds for the time constants. HAC changes the estimated OLS covariance while retaining its mean fit. IID and GLS use their own fitted means; their comparison includes the change in fitting criterion. GLS holds the covariance fixed. Full curvature includes noise estimation but remains a local, single-basin diagnostic.\n")
    tier1 += table(["Parameter", "Gaussian 95\\% range", "Block 95\\% range"], range_rows)
    tier1 += "Bootstrap refits include noise and ambient estimation, boundaries and pole alternatives. Percentile ranges summarize the working refit distributions; coverage has not been established.\n"
    mode = first["gaussian"]["pole_label_modes"]["tau_lt_theta"]["count"]
    blocks_mode = first["blocks60"]["pole_label_modes"]["tau_lt_theta"]["count"]
    tier1 += ("The opposite pole labeling occurs in "+str(mode)+"/400 Gaussian and "+str(blocks_mode)+"/400 block refits. A central interval can miss that secondary basin. The step ambient bound is active in "+str(first["gaussian"]["bound_hit_counts_among_usable"]["ambient_2_C"])+"/400 Gaussian refits.\n")
    profile_p1 = [r for r in profiles["curves"] if r["pair"] == "P1"]
    parts = []
    for name in ("K", "tau", "theta"):
        row = next(r for r in profile_p1 if r["parameter"] == name)
        parts.append("$"+name+"$: "+" or ".join("["+format(a, ".4g")+", "+format(b, ".4g")+"]" for a,b in row["included_grid_components"]))
    tier1 += "Nominal profile support is "+"; ".join(parts)+". Finite grid and crossing brackets are retained in the numerical archive.\n"
    tier1 += "\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{generated/figures/P1_predictions.pdf}\\caption{P1 predictions for three evaluation records. Blue bands are pointwise Gaussian working observation ranges at zero future ambient correction; gray envelopes also allow ambient offsets in $\\pm2\\C$. Ambient envelopes have no assigned probability. Black points are recorded temperatures.}\\end{figure}\n"
    (args.output/"tier1.tex").write_text(tier1)
    rows, pole_rows, eigen_rows, fit_rows, band_rows, length_rows = [], [], [], [], [], []
    for pair, info in I.items():
        u = U["pairs"][pair]
        rows.append([pair, interval(u["gaussian"]["distributions"]["K"]["quantiles_025_50_975"], 3),
                     interval(u["blocks60"]["distributions"]["K"]["quantiles_025_50_975"], 3),
                     str(u["gaussian"]["pole_label_modes"]["tau_lt_theta"]["count"])+"/"+str(u["blocks60"]["pole_label_modes"]["tau_lt_theta"]["count"])])
        pole_rows.append([pair, interval(u["gaussian"]["distributions"]["slow_pole_s"]["quantiles_025_50_975"], 1),
                          interval(u["gaussian"]["distributions"]["fast_pole_s"]["quantiles_025_50_975"], 1)])
        length_rows.append([pair]+[format(u[method]["distributions"]["K"]["quantiles_025_50_975"][2]-u[method]["distributions"]["K"]["quantiles_025_50_975"][0], ".4f")
                                    for method in ("blocks30", "blocks60", "blocks120")])
        spectral = info["conditional_GLS_thermal"]
        vector = np.asarray(spectral["eigenvectors_columns"])[:, 0]
        if vector[np.argmax(np.abs(vector))] < 0: vector = -vector
        eigen_rows.append([pair, format(spectral["eigenvalues"][0], ".3g"), format(spectral["condition_number"], ".1f"),
                           "("+", ".join(format(v, ".2f") for v in vector)+")"])
        m = metrics[pair]
        fit_rows.append([pair]+[format(m[k], ".3f") for k in ("ordinary_RMSE_C", "RMSE_C", "common_three_sine_RMSE_C", "conditional_reconstruction_RMSE_C")])
        band_rows.append([pair]+[format(100*m[method][family]["fraction_inside"], ".1f") for method,family in
                                  [("gaussian","observation"),("innovation_blocks","observation"),("gaussian","ambient_observation")]])
    tier2 = table(["Pair", "Gaussian $K$ range", "Block $K$ range", "Swaps G/B"], rows)
    tier2 += "G/B counts use 400 draws per method. They are bootstrap selection frequencies under the working mechanism, not posterior probabilities of a physical pole assignment.\n"
    tier2 += table(["Pair", "Unlabeled slow pole, s", "Unlabeled fast pole, s"], pole_rows)
    tier2 += "These are Gaussian percentile ranges. Unlabeled poles avoid combining two swapped parameter basins in a single interval for a physical time constant.\n"
    tier2 += table(["Pair", "$K$ width: 30-s blocks", "60-s blocks", "120-s blocks"], length_rows)
    tier2 += "Widths are in $\\C/\\%$. Blocks count samples, approximately seconds. The 30/120-s sensitivities use 100 draws each; their 2.5\\% tails involve only two or three order statistics. The 400-draw primary methods have about ten draws per tail. Rare pole switches can move endpoints; the archive retains order-statistic Monte Carlo ranges and half-batch discrepancies.\n"
    tier2 += table(["Pair", "Smallest information eigenvalue", "Condition", "Weak log direction"], eigen_rows)
    tier2 += "Directions use $(\\log K,\\log\\tau,\\log\\theta)$ after eliminating ambient columns; eigenvector signs are arbitrary. These are local fixed-covariance diagnostics and ignore box geometry and separated modes.\n"
    tier2 += "All six conditional thermal information matrices have rank three at relative eigenvalue thresholds $10^{-8}$, $10^{-10}$ and $10^{-12}$. P2's full observed curvature is indefinite, so its regular Gaussian covariance is withheld. P4 has an active ambient bound despite positive observed curvature. Rank and positive curvature alone do not settle bounded or global inference.\n"
    tier2 += table(["Pair", "OLS RMSE", "GLS RMSE", "Common sines", "Ambient reconstruction"], fit_rows)
    tier2 += "RMSE is in degrees Celsius. OLS/GLS columns pool each pair's ten evaluation records; the exclusions differ, so those aggregates are not a controlled same-test-set ranking. Common sines restrict GLS evaluation to 30-s run 1, 120-s run 3 and 300-s run 2, which are excluded from all six calibrations. Primary predictions use zero future ambient correction. The last column fits an offset using the complete evaluation response and is retrospective reconstruction.\n"
    tier2 += table(["Pair", "Gaussian inside, \\%", "Block inside, \\%", "Gaussian + ambient, \\%"], band_rows)
    tier2 += "Fractions count measured temperatures inside pointwise future-observation ranges, pooled by sample count. Ambient envelopes are scenarios. Overlapping evaluation records and serial dependence prevent independent-binomial coverage claims.\n"
    tier2 += "\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{generated/figures/prediction_RMSE.pdf}\\caption{Zero-ambient prediction RMSE in degrees Celsius. White cells are the two calibration records for that pair. Records are labeled by sine period or step (S), then run number.}\\end{figure}\n"
    (args.output/"tier2.tex").write_text(tier2)
    plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    examples = ["sin_50_50_run_1", "sin_5_50_run_2", "step_50_run_3"]
    numeric = np.load(args.predictions/"P1"/"predictions.npz")
    fig, axes = plt.subplots(3,1,figsize=(5,5.4))
    for ax, token in zip(axes, examples):
        j = next(j for j,r in enumerate(P["P1"]["records"]) if token in r["record"])
        t=numeric["time_"+str(j)]/60
        ax.fill_between(t, numeric["gaussian_ambient_observation_lower_"+str(j)], numeric["gaussian_ambient_observation_upper_"+str(j)],color="0.88",label="Ambient envelope")
        ax.fill_between(t, numeric["gaussian_observation_lower_"+str(j)], numeric["gaussian_observation_upper_"+str(j)],color="#93bce0",label="Working observation range")
        ax.plot(t,numeric["point_"+str(j)],color="#175a8e",lw=1,label="Zero-ambient mean")
        ax.plot(t,numeric["observed_"+str(j)],".",color="0.15",markersize=1.2,label="Temperature")
        ax.set_ylabel("Temperature, C");ax.set_title(token.replace("sin_50_50", "30 s sine").replace("sin_5_50", "300 s sine").replace("step_50", "Step").replace("_run_", ", run "),loc="left",fontsize=8)
    axes[0].legend(loc="best",fontsize=6,ncol=2,frameon=False)
    axes[-1].set_xlabel("Time, min");fig.tight_layout();fig.savefig(figures/"P1_predictions.pdf");plt.close(fig)
    names = read(Path(__file__).parent/"config.json")["records"]
    matrix = np.full((6,12),np.nan)
    for i,pair in enumerate(P):
        for row in P[pair]["records"]:matrix[i,names.index(row["record"])]=row["RMSE_C"]
    labels=[period+"/"+str(run) for period in ("30","120","300","S") for run in (1,2,3)]
    fig,ax=plt.subplots(figsize=(5,2.8)); im=ax.imshow(np.ma.masked_invalid(matrix),aspect="auto",cmap="YlOrRd",vmin=0)
    for i in range(6):
        for j in range(12):
            if np.isfinite(matrix[i,j]):ax.text(j,i,format(matrix[i,j],".1f"),ha="center",va="center",fontsize=6,color="black" if matrix[i,j]<np.nanmax(matrix)*.55 else "white")
    ax.set_xticks(range(12));ax.set_xticklabels(labels,rotation=45,ha="right");ax.set_yticks(range(6));ax.set_yticklabels(list(P));fig.colorbar(im,ax=ax,label="RMSE, C",fraction=.04,pad=.025);fig.tight_layout();fig.savefig(figures/"prediction_RMSE.pdf");plt.close(fig)
    output={"prediction_metrics":metrics,"common_evaluation_records":sorted(common_names),"binding":bindings,"uncertainty_summary_sha256":digest(args.uncertainty_summary/"uncertainty.json"),
            "profile_audit_sha256":digest(args.profile_audit/"audit.json"),"figures":{p.name:digest(p) for p in figures.glob("*.pdf")},
            "notice":"Generated tables and descriptive figures; final prose interpretation and all-page rendered layout review remain required."}
    (args.output/"report_numbers.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
