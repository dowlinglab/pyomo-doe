# Shared-model discrepancy investigation

The goal is active. Follow [the scientific plan](../review/tier0_followup_investigation.md)
and [methods](methods.md). Alex prioritizes understanding the discrepancies and
set no duration constraint. Preserve the completed residual_study and its email.

| Phase | Current state |
|---|---|
| 0. Reproduce reference | Complete; independent propagation and dense likelihood agree |
| 1. Locate discrepancies | Per-run plots, phase folds and descriptive energy decomposition reviewed |
| 2. Frequency response | 54 estimates independently reconstructed; joint-harmonic correction audited |
| 3. LTI benchmarks | Two/three/four real-pole fits audited; numerator-bound sensitivities establish a weak fast pole |
| 4. Equal-stage comparison | n=1/2/3 joint fits and fixed-weight/IID sensitivities audited; all six profiles/78 points audited |
| 5. Targeted explanations | Positive unequal lags, transport and linear-ambient fits audited; FIR16 cells audited; nonlinear/transport checks audited; revised-mean stage comparison closing |
| 6. Report and review | report.md is a draft; numerical/source/figure/requirement audit pending |

## Established evidence

The n=1 reference reproduces at RMSE 0.48320 C. Its fast-sine response lacks
about 28–32 degrees of phase lag across repeats and startup treatments. Retuning
a positive two-real-pole transfer cannot match the estimated fast phase under
the steady LTI interpretation. n=2 suppresses some cycling, but n=3 performs
worse under the working joint likelihood. Better likelihood can accompany worse
raw temperature error because the fitted disturbance also changes.

Flexible three/four-state real-pole fits reduce fast mismatch but reach numerator
bounds and have poorly estimated fast modes. Widening bounds moves the three-
state fast pole from 0.14 to 0.07 to 0.04 s with nearly unchanged likelihood and
K. Do not interpret those states physically. Positive unequal lag stages and
an explicit extra delay provide simpler alternatives. The transport d estimate
is about 2.53 s; nominal profiled support is about 2.41–2.65 s under that model,
not a hardware delay calibration.

Use results/reference_diagnostics_v2 for harmonics/forecasts. The original
separate input-frequency projections leaked the fundamental on partial-cycle
windows; v2 projects jointly. Forecasts/plots were unaffected. The reference
one-second conditional RMSE is 0.117 C versus 0.484 C free-run on identical
targets; both use retrospectively fitted parameters. No independent forecasting
claim follows from that comparison.

The FIR sensitivity permits mostly positive response shapes with modest
smoothing, while showing tail/onset sensitivity to bins, horizon and penalty.
It conditions on the n=1 ambient/noise reference and is not another jointly
estimated likelihood competitor. Selected kernels were independently audited
using command-step convolution and dense covariance weighting.

## Final bounded comparison and remaining work

The targeted quadratic-loss and monotone command-map checks have selected-fit
and profile audits. A command map plus transport and linear ambient fits at
0.38794 C RMSE. A command map alone with n=1 and linear ambient fits at
0.33413 C, but retains a much poorer working likelihood. Raw RMSE, stochastic
weighting and dynamic adequacy must be assessed together.

Complete the matched command-map/linear-ambient n=1/2/3 and unequal-positive
three-pole comparisons before assigning a stage preference. These are the final
mean families in scope. Run their independent selected-fit audits, scaled
information/response uncertainty, K/theta profiles, and innovation checks.
The existing phase, FIR and nonlinear results explain why these follow-ups
were selected; no additional broad model search is planned.

Reconcile [run_manifest.json](run_manifest.json), exact process commands and
terminal files before launching or retrying. The manifest identifies current
jobs; it is not proof that a process remains alive. Never restart based on an
observation timeout. Keep at most four long numerical processes.

Final diagnostics must retain the quantized step response, remaining slow
trends, per-run innovation scales and input association. One-step conditioning
can hide mean error. Complete the report, figure review and requirement audit
before marking this goal finished.

## Execution and completion rules

Use committed-source/data snapshots and the pinned scientific runtime. Keep at
most four long numerical processes, each with one BLAS thread. This does not
authorize extra agents. Retain failures and recoveries; commit coherent progress.
The existing hourly thread heartbeat is active for this goal.

Finish with a concise report linking numerical evidence, methods and reviewed
figures. Distinguish physical hypotheses, working likelihood support and actual
model adequacy. All ENV2 records have informed selection; evaluation remains
retrospective. Stop for Alex's review after the report and requirement audit.
Mark the goal complete and pause the heartbeat only then. Do not enumerate
additional calibration pairs, run DoE/hardware, expand to a full two-temperature
model or send another email.
