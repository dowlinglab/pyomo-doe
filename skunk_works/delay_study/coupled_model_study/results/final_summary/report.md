# Can coupled thermal balances explain the unequal lags?

The second temperature confirms substantial heat transfer to the unpowered heater. A symmetric four-state energy-balance model can represent that coupling, but it cannot by itself explain the fast T1 phase. Adding one shared first-order input lag tests an additional mechanism; fitting it does not identify its hardware origin.

## What was fitted

All twelve distinct ENV2 records, both T1 and T2, with Q2 identically zero. Commands use exact left holds on the recorded irregular timestamps. The initial hidden heater temperatures equal their respective first sensor readings; first observations are conditioned on. One common physical ambient offset per run is bounded by ±2 C relative to its first T1. The affine sensitivity bounds both ambient endpoints by the same ±2 C. Channel errors use separate OU-plus-measurement covariance parameters and scales; independence between channels is a working assumption. A shared constant alpha is absorbed into the gain.

The positive reciprocal balances are, for heater i and the other heater j,

\[C_H\dot H_i=U_a(A-H_i)+U_s(S_i-H_i)+U_x(H_j-H_i)+\beta Q_i,\qquad C_S\dot S_i=U_s(H_i-S_i).\]

The estimated scale-free parameters are K=beta/Ua, tA=CH/Ua, rS=CS/Us, r=Us/Ua and q=Ux/Ua. Absolute capacities and conductances still need an independent power calibration. The optional command map is u+c*u*(1-u/100); the optional lag satisfies L*dz/dt=u-z after that map.

## Why four thermal states do not settle the phase question

For fixed ambient, common and differential sensor responses have transfers

\[G_+=\frac{K}{t_A r_Ss^2+[t_A+r_S(1+r)]s+1},\quad G_-=\frac{K}{t_A r_Ss^2+[t_A+r_S(1+r+2q)]s+1+2q}.\]

\[T_1/Q_1=(G_++G_-)/2,\qquad T_2/Q_1=(G_+-G_-)/2.\]

Each modal transfer has strictly negative imaginary part for positive frequency and positive parameters. Their sum does too. The base T1 phase therefore stays between -180 and 0 degrees. The fast sine measurements are about -193 degrees on the frequency-unwrapped branch. This is a restriction of the stated symmetric linear model, not a proof against every possible thermal network. Its transfer zeros also matter: four thermal poles do not make it equivalent to the earlier all-pole cascade.

## Matched two-output results

| Model | T1 RMSE, C | T2 RMSE, C | Nominal AICc | Projected gradient | Successful starts |
| --- | ---: | ---: | ---: | ---: | ---: |
| main_base_ou | 1.1074 | 0.2342 | -36085.64 | 0.006953 | 8/8 |
| main_lag_ou | 1.1561 | 0.2383 | -36951.37 | 0.001592 | 8/8 |
| main_lag_curve_ou | 1.0221 | 0.2312 | -37018.26 | 0.000643 | 8/8 |
| sensitivity_lag_curve_affine_ou | 1.0324 | 0.1242 | -37301.64 | 4.665e-05 | 4/4 |

The IID pilot pushed the extra input lag to its 0.01 s lower bound, with T1/T2 RMSE 1.103/0.233 C. The colored-error analysis supports a finite short lag while accepting more slow temperature mismatch. The error model changes the fitted dynamics as well as their uncertainty; a lag estimate should always name the fitting criterion.

AICc compares the same two-output observations under the working Gaussian covariance. It is not compared with the earlier T1-only likelihood. A lower correlated-error objective need not lower raw temperature RMSE. These are retrospective fits; all records have informed model selection.

The smallest nominal AICc in this bounded comparison is **sensitivity_lag_curve_affine_ou**. Its parameters are:

| Parameter | Estimate |
| --- | ---: |
| K_common | 0.892331 |
| tA_s | 179.078 |
| rS_s | 14.6866 |
| Us_over_Ua | 0.005 |
| Ux_over_Ua | 0.408947 |
| input_lag_s | 2.72662 |
| command_curvature | 0.12791 |

K_common is the gain of T1+T2. Before the command map, the individual steady gains are 0.69160 C/percent for T1 and 0.20074 C/percent for T2. It should not be read as the earlier single-channel gain. The four thermal modal time constants are 14.68, 14.68, 98.56, 179.2 s; the optional input lag is additional.

## Residual structure and frequency response

| Model | Max absolute innovation ACF, lags 1–60 | Fast T1 input/innovation correlation range | Max absolute cross-channel innovation correlation, lags ±20 |
| --- | ---: | ---: | ---: |
| main_base_ou | 0.294 | 0.460–0.482 | 0.117 |
| main_lag_ou | 0.199 | 0.113–0.144 | 0.121 |
| main_lag_curve_ou | 0.200 | 0.113–0.149 | 0.120 |
| sensitivity_lag_curve_affine_ou | 0.172 | 0.115–0.151 | 0.126 |

These are descriptive maxima across many lags and runs, not single prespecified hypothesis tests. Residual cycles and cross-channel dependence must be assessed before treating the assumed covariance as an adequate stochastic law.

| Model | Fast T1 matched phase error range, degrees | Fast T1 matched/measured gain range |
| --- | ---: | ---: |
| main_base_ou | 28.36 to 31.52 | 0.847–0.943 |
| main_lag_ou | 3.05 to 6.20 | 0.894–0.994 |
| main_lag_curve_ou | 2.79 to 5.94 | 0.887–0.986 |
| sensitivity_lag_curve_affine_ou | 2.98 to 6.13 | 0.886–0.986 |

The fast T2 phase has working HAC60 standard errors of 26.4–68.4 degrees across repeats. Its small oscillation cannot reliably locate an additional fast mode. The slow rise still supplies coupling information. Frequency plots use identical finite windows for measured and simulated temperatures; the JSON retains 300, 450 and 600 s cutoffs.

![Joint trajectories](results/final_analysis/joint_trajectories.png)

![Joint frequency response](results/final_analysis/joint_frequency.png)

## What can be interpreted physically

Unequal decay rates are physically plausible because different thermal capacities and conductances produce different eigenmodes. Their numerical values should not be assigned to particular hardware parts solely from a fitted cascade.

| Model | Mean-information condition number | Active shape bounds | Weakest local coordinate |
| --- | ---: | --- | --- |
| main_base_ou | 2.03e+09 | log_Us_over_Ua | log_Us_over_Ua |
| main_lag_ou | 3.76e+05 | None within 1e-4 | log_Us_over_Ua |
| main_lag_curve_ou | 1.28e+10 | log_Us_over_Ua | log_Us_over_Ua |
| sensitivity_lag_curve_affine_ou | 2.28e+09 | log_Us_over_Ua | log_Us_over_Ua |

These matrices fix the fitted covariance and project out the ambient nuisance tangent. They diagnose weak directions; they are not profile likelihood intervals and do not include noise re-estimation or model-selection uncertainty. Regular Wald intervals are inappropriate at active bounds.

In the limit of negligible sensor heat loading, the common sensor factor 1/(1+rS*s) and common command factor 1/(1+L*s) commute. Swapping their time constants leaves both command-to-temperature transfers unchanged. Ambient and startup assumptions can break this equivalence, so any apparent allocation can depend on those assumptions. An extra lag that improves the fit is therefore evidence for a missing dynamic contribution, not a calibrated actuator or sensor time constant.

## Open questions

- Does relaxing heater/sensor symmetry improve both outputs while preserving the fast phase evidence? The present comparison does not answer that.
- How much of the remaining cross-channel dependence comes from common environmental disturbance versus mean-model error?
- Can independently measured electrical power, sensor dynamics or ambient temperature distinguish the competing lag interpretations?
- Would separately driving heater 2 identify modes that Q1-only excitation leaves weak? No new experiment was performed or designed here.

## Verification and scope

The frozen foundation checks compare exact holds with independent dimensional energy-balance integration, transfer inversion and the common/differential identities. Every selected fit has a separate dimensional-matrix forward audit and dense-Cholesky covariance/likelihood audit. The analysis checks its pole/residue reconstruction against the analytic transfer. Source commits, raw hashes, multistarts, solver outcomes and audit bindings are retained in `run_manifest.json` and `results/`. Numerical agreement verifies implementation, not the physical assumptions or scientific adequacy. The previous completed study and email delivery are unchanged.

The reciprocal balances follow the [Notre Dame four-state TC-Lab notes](https://ndcbe.github.io/controls/tclab/03.06-Four-State-Model.html). The [APMonitor state-space treatment](https://apmonitor.com/pdc/index.php/Main/TCLabStateSpace) also discusses thermal coupling and a sensor-lag approximation. The phase restriction and interpretation above are derived here. No unreturned literature-search result is used as evidence.
