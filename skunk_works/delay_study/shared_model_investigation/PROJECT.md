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
| 5. Targeted explanations | Positive unequal lags, transport and linear-ambient fits audited; FIR16 cells audited; nonlinear/transport checks audited; all16 main fits audited; revised-stage profiles pending |
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

## Final comparison and remaining checks

All sixteen main fits and their selected means/innovations/dense likelihoods
are audited. The revised unequal-positive model has RMSE0.37397 C, shared
K0.688645, time constants142.84/2.99/13.47 s and command curvature0.10184.
It removes much of the fast input-linked error but retains maximum innovation
ACF0.163 and slow/quantized discrepancies. It is a candidate reduced reference,
not an established universal stochastic description.

Revised equal-stage n=1/2/3 use the same command-map and affine ambient treatment.
Their raw RMSEs are0.334/0.514/0.741 C; revised n=1 retains fast input-innovation
cross-correlations0.458–0.482 despite its low raw error. Scaled information and
pole-permutation checks are complete. The revised K/theta profiles are the only
remaining numerical fits. No new mean families are planned.

Finish those profiles and independently audit all saved points. Then run the
committed summarize_evidence.py to check prediction arithmetic, AICc/counts,
artifact bindings and the report table, and review its two figures. Complete
verification.md, commit results and report, reconcile processes and stop for
Alex's review. report.md already contains the scientific synthesis; its status
and final profile discussion still need updating after these checks.

Reconcile [run_manifest.json](run_manifest.json), exact process commands and
terminal files before launching or retrying. Never restart from an observation
timeout. Keep at most four long numerical processes with one BLAS thread each.

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
