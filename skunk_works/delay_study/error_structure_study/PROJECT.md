# TC-Lab remaining-error study

Alex authorized this sustained investigation on October 25, 2026. The plan is
[the shared discussion note](../review/next_investigation_discussion.md), committed
before goal launch as 346a03d. The goal is active. Its ten-plus-hour research
scope has flexible timing and no idle minimum. No hardware, DoE or email is
part of this goal.

Read run_manifest.json and a fresh direct-tool process snapshot before dispatch.
Only terminal audited results support report claims. All twelve ENV2 records
informed structure selection; transfer is retrospective.

## Audited scientific results

The old coupled short-lag/curved-command fit has T1 RMSE 1.022 C. Orthogonal
projections assign 74.6% of T1 SSE to run offsets, another 19.3% to trends and
0.64% to the fundamental. These projections are not independent physical causes.

Matched Gaussian constant-ambient conductance bands give T1/T2 RMSE:
5% = 0.646/0.147 C; 10% = 0.639/0.115 C; 20% = 0.641/0.115 C.
Two 10% deviations reach the lower bound. The 5% fit retains three failed starts.
Affine ambient gives 0.728/0.111 C. Hidden heater starts bounded at +/-2 C give
0.615/0.111 C; +/-5 C gives the same result, every selected offset below 0.35 C.
Cold starts remain plausible, not verified equilibrium.

Radiation calculations use explicit emissivity/area/surface-temperature
scenarios. Metal does not determine emissivity. A linearized contribution can
already be in Ua. The Gaussian free-radiation fit gives 0.633/0.115 C, gamma
0.0344 and nll improvement 1.285. The linear-command radiation fit is worse.
All dimensional ODE, nesting, step-size and dense likelihood audits passed.

Joint physical-plus-quantized fits now have independent trajectory, Gaussian
observation-rectangle, corrected-domain, 128/256/512-node, tail-12/16 and
stationarity audits:

| Case | Audited export | T1/T2 RMSE, C | Bin nll |
| --- | --- | --- | --- |
| Shared, no run deviations | joint_conductance_b00_export_v3 | 1.010 / 0.239 | 6359.91818 |
| 10% conductance band | joint_conductance_b10_export_v2 | 0.605 / 0.118 | 6238.29865 |
| 20% conductance band | joint_conductance_b20_export_v2 | 0.606 / 0.118 | 6238.21668 |
| Radiation, 10% convective band | joint_radiation_b10_export_v2 | 0.590 / 0.119 | 6234.48899 |

The 5% joint fit is still running. A wider band cannot be selected from training
likelihood alone. Radiation's bin-likelihood improvement is 3.810, larger than
in the Gaussian approximation, but its boundary/nuisance profile and transfer
remain necessary. Convective and effective total loss are different quantities.

The 10% joint gain is 0.87322 C/%, within 0.1% of its Gaussian value; tA=174.586 s,
rS=14.790 s, input L=2.730 s and command curvature=0.12182. Loading Us/Ua reaches
0.005. One run conductance reaches 0.9. Its analogue latent SD / length /
measurement SD are T1 = 0.53621 C / 658.28 s / 0.02575 C and
T2 = 0.06813 C / 62.56 s / 0.03248 C. These are working noise parameters, not
sensor calibration measurements or final uncertainty claims.

## Whitening evidence and remaining checks

All 21,614 raw hardware temperatures match round(0.3223*n - 50, 2). This is
not just simulator quantization. Firmware thresholds and pre-quantization noise
are not established by the grid.

joint_b10_predictive_diagnostics is terminal: twenty independent PIT streams
and fifty fixed-model simulations, no failed simulation. Median maximum pooled
ACF over lags 1..60 is 0.02481 for T1 and 0.02454 for T2; median maximum
cross-channel correlation over lags -10..10 is 0.02235, comparable to references.
T1 nonrandomized transition discrepancy is 2.3356 and maximum phase projection
fraction has median 0.02029; both have upper-tail simulation rank 0.0588.
The plot was inspected. These are fixed-parameter references, not refitted
diagnostic p-values, model-selection adjustment or proof of IID errors.

Before dismissing cross-channel disturbances, check nonrandomized joint code
transitions and conditional expected Gaussian-PIT scores. Randomization can
dilute information in coarse bins, especially T2 plateaus. Reuse the fifty saved
synthetic observation collections for those references. The full fifty refitted
recovery draws can also provide fit-aware diagnostic references. Earlier marginal
pilots reused seeds and are unsuitable for joint cross-channel/run testing.

Sequential Gaussian GLS from an IID mean converged in nine cycles, all block
solvers successful; nll differs from the saved joint fit by 1.29e-7. The
saved-joint initialization is also terminal. Both need the full independent
dense/trajectory and trace audit before final reporting. See sequential_methods.md.
The short synthetic dense nll/score and constrained-mean checks passed.

## Identifiability and initial conditions

A synthetic limit check verifies that sensor/input lag labels exchange exactly
in zero-loading forced transfers. At the Gaussian boundary Us/Ua=0.005, swapping
14.98 s and 2.69 s changes nominal fast-sine T1/T2 responses only
0.00430/0.00185 C. Unequal starts and ambient forcing can break this exchange.
This is not a hardware zero-capacity claim or an absolute quantization-resolution
limit. The lower positive loading bound 0.0001 is a declared three-start refit
sensitivity, now running.

Initial sensor-bin coordinates move each cold heater/sensor pair together,
preserve the original ambient reference/bounds and are separate from hidden
heater-only starts. Dimensional propagation and score audits passed for nearest
and floor phases, with/without a 0.1 C buffer. Run the full data sensitivities
after capacity opens. These bounds do not integrate an initial-noise posterior.

The phase-reference check passed: nearest-to-floor bins translate by 0.16115 C.
With both cold pairs and ambient shifted equally and slack bounds, the linear
mean and bins translate together; nll changes below 1.4e-12. The phase is not
separate dynamic information under that freedom. Fixed initial states, active
ambient limits and absolute-temperature radiation need not share it.

The new --condition-initial-code sensitivity is running. It uses
p(y0,...,yN)/p(y0), leaves nobs unchanged and subtracts the one-observation
smoothing score. All mean/noise coordinates, including initial sensor-bin
coordinates, passed independent conditional-likelihood finite differences.
Its current fitting/export/profile/audit support is exact under quadrature.
Stationary simulation and transfer drivers explicitly refuse this mode until
its conditional truncated-Gaussian initial-state sampler/marginals are supplied.
Do not silently use a mismatched simulation convention.

## Numerical failures and recovery

The original zero-band attempt failed a rare-bin integration domain gate.
Component-conditional posterior ranges repaired it. Rare rectangles, tail checks
and the full score/regression suite passed. Its original failure is retained.

The terminal numerical fits for 0/10/20% and radiation each converged three
starts, then failed JSON export because parameter_count was numpy.int64.
Commit 6e9e654 casts it to a Python int and adds --resume-starts. The exported
directories above re-evaluate those saved points under the current score and
do not repeat optimization. Original exits, failed trial points and solver
traces remain evidence. Recovery receipts bind the old saved starts/source.
The still-running 5% job uses a pre-fix frozen snapshot; if it reaches the same
export failure, recover its saved starts instead of launching another fit.

Use direct pgrep -fl tclab-error. Python subprocess process enumeration is blocked
by macOS sysmon and an empty result is not process absence. reconcile.py consumes
a fresh direct-tool process_check.json, checks exact PID/command identities and
terminal exit artifacts, and never launches/restarts anything. Older exit receipts
without finish timestamps remain terminal with unknown timestamps.

## Current execution and next unfinished phases

Use the manifest for current identities and controller sessions. At the latest
dispatch, four long jobs are active: joint_conductance_b05,
profile_b10_gain, joint_b10_loading_lower and joint_b10_condition_first_code.
No duplicate live retry. At most four long numerical processes, one BLAS thread
each. Short independent arithmetic checks are recorded separately.

The information job is terminal: its full bin-score Hessian is positive on the
centered tangent at the selected point, with a weak loading direction and a
variance/length tradeoff. Relative asymmetry is 1.30e-6; three weak active-face
directions agree with direct two-step-size curvature checks to about 1.3e-4.
It still needs an independent matrix/information audit; do not interpret the
tiny boundary-loading eigenvalue as a regular Wald variance. The gain profile
is now running. Other profiles and refitted recovery are not measured yet.
uncertainty_config.json declares grids, starts, bounds,
fifty recovery draws and gain-LR coverage before measurement. Source drivers:
run_profiles.py, run_information.py, run_recovery.py.

transfer_config.json declares sixteen primary folds (twelve runs + four
protocols) and four shared-model protocol folds. Training/prediction code is
committed; independent dimensional, no-future-output-leakage, reported-code
expectation/quantile, mixture and fold-separation checks passed. No fold fit is
measured yet. Parameter-draw propagation must inherit a validated approximation;
noise-only hard-nuisance ranges are not total predictive intervals. If profiles
change the candidate set, extend transfer support explicitly rather than using
a linear predictor for a radiative or conditioned-initial-state fit.

Still required: nonrandomized cross-channel checks; full sequential audits;
all pending mean/initial/bin sensitivities and selected-point audits; updated
frequency/operating diagnostics; supported nuisance profiles; eigendecomposition
interpretation; fifty refitted recovery/coverage draws with failure denominators;
retrospective run/protocol transfer with parameter/noise/bin/model-form limits;
and the final report and requirement-by-requirement completion audit.

## Reporting and continuation

report/main.tex and main.pdf are a four-page work-in-progress layout, not a
finished report. Initial layout and numeric checks passed; recompile, rerender
and recheck after each meaningful revision. Geometry is 0.25-inch left and
3.25-inch right, with 35 college-spaced annotation lines per page. Final evidence,
figures, uncertainty/transfer and the companion concise Markdown report remain
pending. Keep no-ai-slop and the user's research practices in the writing check.

The hourly tc-lab-overnight-research heartbeat stays active. Read this file, the
committed plan and manifest on resume. Freeze committed code/data/configuration
with freeze.py under pinned Python/dependencies from ../environment.json.
Save starts, failures, source/input hashes, processes and exits. Commit coherent
source before measurement, audited results afterward; push progress and final
results on reformulation-exploration. Preserve raw data and prior deliveries.

Finish only when the full scoped gates have terminal audited evidence, figures
and the final tablet PDF have been inspected, the Markdown/LaTeX/PDF findings
are committed and pushed, and no required job remains live. Report unresolved
identifiability or whitening honestly. Then pause the heartbeat for Alex's review.
