# Current n=1 residual teaching study

Goal active from 2026-10-24 16:39:24 UTC. Agreed scope: `../review/approach.md`.
Implementation: `execution_plan.md`; records/pairs: `config.json`.
This phase leaves the completed October 21 study and deliveries unchanged.

| Phase | State | Next gate |
|---|---|---|
| Data/input and exact mean audit | Passed in frozen source | Preserve conditional startup/alignment assumptions |
| Tier 0 mean/error exploration | Calculations complete; adequacy unresolved | Working structure frozen with explicit limitations |
| Tier 1 P1 uncertainty/evaluation | Point fit and local information complete; preflights passed | Bootstrap, profiles and evaluation |
| Tier 2 P2-P6 | All five point fits and local information complete | Finish exactly six pairs and uncertainty/evaluation |
| Report/Gmail/push | Pending | Checked ruled-margin PDF; one verified email |

No TC-Lab numerical process was live at startup reconciliation. Read the current
manifest before launching work. Reuse the old hourly heartbeat for this goal.
No arbitrary waiting is required: advance research until deliverables finish.
Recheck pole equivalence with ambient forcing; do not impose time-scale ordering.
Evaluation is retrospective after Tier 0. Ambient bounds are scenarios, not a
probability distribution. Technical defaults are identified in the plan.

Hourly continuation is active on `tc-lab-overnight-research` and targets this
thread. The OLS baseline and independent audit are terminal with exit code zero.
The covariance comparison, independent audit and diagnostic plotting are terminal
with exit code zero. All nine assumption-sensitivity cases are also terminal.
The diagnostic simulations and their audits are complete. All six restricted-pair fits and local information checks are terminal. The
profile and prediction preflights passed; uncertainty matrices are next. Reconcile
`run_manifest.json` before launching another worker.

The acquisition notebook at b114b13f updates lab.Q1 immediately before
p.update(t), supporting left-held inputs. This is supporting source code, not
a per-file execution log. `audit_mean.py` independently compares propagation
against augmented matrix exponentials and DOP853, including irregular grids,
repeated poles, endpoint ambient drift and both hold conventions. All 36
checkout checks passed (maximum matrix discrepancy 1.4e-11 C); the measured
baseline reruns this check from committed sources.

## Current findings and next calculation

`results/tier0_mean` contains twelve-record, 24-start-per-model pooled fits from
2cb60ff. All 48 starts terminated successfully; two distinct pole-label basins
were reached. `results/tier0_mean_audit` independently reconstructs the selected
trajectories/objectives with matrix exponentials and verifies raw files/bounds.
Maximum saved-trajectory discrepancy was 1.6e-12 C. This does not prove a global
minimum or scientific adequacy.

| Ambient treatment | Pooled RMSE, C | Runs touching ambient bounds | Raw residual lag-one ACF range |
|---|---:|---:|---:|
| Constant offset | 0.46574 | 7/12 | 0.879 to 0.966 |
| Linear endpoints | 0.43702 | 9/12 | 0.809 to 0.964 |

Linear drift lowers pooled SSE by 11.95%, with a 6.17% RMSE reduction. Several
nuisance bounds are active and residuals remain strongly correlated. These are
unweighted exploratory fits; no covariance-aware uncertainty or adequate noise
model is claimed. Do not select linear drift solely from this training decrease.
The completed covariance/iterative-GLS comparison uses the four declared kernels
under both ambient treatments, with 24 joint starts per cell and one GLS history.
`audit_covariance.py` passed sixteen synthetic dense-Cholesky comparisons,
including stationary initialization, irregular sampling and boundary limits.
The first source-bound four-start pilot retained a stationarity warning;
`methods.md` explains the central-derivative refinement and preserved records.
The refined pilot gives projected gradient about 1e-4, with GLS/joint likelihood
agreement within 4e-7. All full cells ran from b84306e. The 192 start-level
solutions finished successfully; 208 solver attempts include four unsuccessful
attempts retained alongside recovery. Selected projected gradients are below
0.0034. All eight alternating sequences stabilized; their likelihoods agree with
joint fits within 1.2e-6. This is numerical evidence, not residual adequacy.

`results/tier0_covariance_audit/audit.json` reconstructs all 96 selected run
trajectories and dense-covariance likelihoods. Maximum NLL discrepancy is
3.6e-10. `generation_audit.json` checks ten independent state-space generation
factors against closed-form covariance, ready for refit diagnostics/uncertainty.
All nine figures in `results/tier0_diagnostics` have been inspected for internal
use; the final narrow-column report still needs its own visual checks.

| Constant ambient, pooled error model | Raw RMSE, C | Nominal AICc | Innovation lag-one ACF range |
|---|---:|---:|---:|
| IID | 0.46574 | 14169.35 | 0.879 to 0.966 |
| OU | 0.46709 | -12276.51 | -0.541 to -0.307 |
| OU + measurement noise | 0.48320 | -15818.37 | -0.239 to 0.238 |
| Matern-3/2 + measurement noise | 0.47981 | -15520.30 | -0.186 to 0.241 |

With OU plus measurement noise, linear drift improves NLL by 6.05 but adds
twelve parameters; its nominal AICc is 12.00 higher than constant offsets.
This conditional comparison makes the OLS drift improvement less persuasive;
it is not a calibrated hypothesis test or proof of constant physical ambient.
Both drift treatments retain the fast-sine innovation pattern.

The OU-plus-measurement working fit has K=0.71001 C/percent, tau=137.48 s,
theta=18.37 s, correlation length 213.87 s, and marginal residual SD 0.47877 C.
Its fitted measurement SD is 0.09062 C and slow-component SD 0.47012 C. The
measurement scale is close to the 0.092-0.095 C SD of hypothetical uniform
rounding errors on a 0.32-0.33 C grid. That is an interpretation to investigate;
the observed discretization does not establish independent uniform noise.

## Assumption checks and diagnostic calibration

The thirty-second sine runs retain an approximately thirty-second innovation
cycle and maximum absolute command/innovation correlations 0.457-0.481 over the
declared +/-30 sample-lag search. Linear ambient drift barely changes this.
The reference Ljung-Box p-values are descriptive, not simulation-calibrated.
Do not treat eleven small p-values as eleven independent calibrated tests.

All nine declared checks from 38d1cc3 are complete with independent forward and
dense-likelihood reconstruction in each fit. All 216 OLS starts and 216 joint
start-level solutions converged. `results/tier0_sensitivity_summary_final` owns
the numerical comparison and checked figure. Preserve the earlier summary import
failure and legend-overlap receipt; neither required rerunning a fit.

| Same-record sensitivity | K, C/percent | NLL change from primary | Fast-sine max input correlation |
|---|---:|---:|---:|
| Right hold | 0.71057 | +256.30 | 0.519 to 0.540 |
| Initial T -0.165 C | 0.70967 | +0.07 | 0.457 to 0.481 |
| Initial T +0.165 C | 0.70999 | +1.38 | 0.456 to 0.481 |
| One second input prehistory | 0.71145 | +5.92 | 0.457 to 0.479 |
| Ambient bounds +/-4 C | 0.69250 | -23.42 | 0.456 to 0.480 |

The checks support retaining left hold and cold startup as the primary choices;
they do not resolve the fast-sine cycle. Widening the ambient band changes K by
about -2.47%, removes active ambient bounds, and leaves the cycle. Retain the
agreed +/-2 C primary bound and report that assumption sensitivity explicitly.
Protocol-specific fits give K values 0.655, 0.689, 0.724 and 0.753 for 30 s,
120 s, 300 s and step data respectively. Their likelihoods use different
observations and must not be compared directly. The 30-second fit chooses the
other pole labeling and still has input correlations 0.321-0.341. These are
exploratory point comparisons, not significance tests of thermal heterogeneity.

`results/tier0_direct_fit_check/audit.json` compares the leading pooled solution
to two direct 17-coordinate fits (ambient/gain not variable-projected). Both
converge; best NLL agreement is 2.1e-11. There is no new global certificate.

Two 12-draw diagnostic pilots generated Gaussian measurement errors or replaced
that component with grid rounding, refitting all thermal/ambient/noise terms.
All draws completed without stationarity flags. Four selected draws (0 and 11 per
mechanism) reproduce their saved statistics, pass independent dense-likelihood checks and
match 24-start refits within 1.4e-11 NLL. The three-start refit budget is supported
by this scoped check, not proved globally optimal for every future replicate.
`results/diagnostic_pilot_*_audit` owns those receipts.

The inferred grid spacing is 0.32229984 C; all 158 distinct pooled temperatures
are reproduced after two-decimal rounding of that grid. This supports a rounding
sensitivity, not an independent hardware calibration. Rounding raises pilot ACF
statistics but does not reproduce the observed strong command association. The
pilots are too small for the requested final diagnostic calibration.

## Tier 0 decision and restricted analysis

Both 200-draw diagnostic comparisons from d0ce06f completed with all 400 refits
usable. `tier0_calibration_audit` regenerates every draw and independently checks
correlation/Q arithmetic, likelihoods/scales and summary values. Maximum statistic
error is below 1.5e-13 and NLL error below 3.4e-11. Four additional selected draws
(99/199 per mechanism) match 24-start refits; the earlier pilot checks cover 0/11.

Observed maximum innovation/command association is 0.481. Its simulated 99th
percentile is 0.113 under Gaussian measurement errors and 0.110 under rounding.
Neither of 200 replicates per mechanism reaches the observed value; plus-one
tail fractions are 1/201. The same contrast persists for maximum ACF and Q30.
These are fitted-model diagnostic calibrations, not exact universal-model tests.

`structure_selection.json` freezes left hold, cold startup, constant +/-2 C
ambient offsets and OU plus measurement noise as the working n=1 description.
It explicitly retains inadequate fast-sine residuals and all uncertainty/evaluation
qualifications; it is agent-selected for the authorized tiers and awaits human
scientific review. No additional thermal order or pair enumeration is authorized.

The information preflight verifies fixed-gain projection analytically and the
observed noise-scale Hessian identity. At the pooled exploratory point, conditional
GLS thermal standard deviations are about 6.3/8.6/3.4 times nominal IID OLS for
K/tau/theta, with ambient nuisance terms eliminated in both. This still holds
the covariance fixed. The full observed curvature is indefinite at the bounded
fit, so no regular Gaussian covariance is released from it. Use profiles and
bootstrap, and retain the model inadequacy and bound qualifications.

Restricted fits run from 8a0a4e2. Each loads only its two training records, refits
all thermal/ambient/noise values, preserves OLS and alternating histories, and
independently reconstructs selected means and dense likelihoods. All six fits and local information checks are terminal with exit code zero.
P1 Gaussian and 60-s innovation-block pilots completed twelve refits each; selected
draws match 24-start checks. Small-nugget noise derivatives, profiles at the MLE,
and ten-record P1 prediction arithmetic passed their independent preflights.
Launch the declared uncertainty/profile matrices, then evaluate the ten records
per pair with the completed uncertainty distributions.

Full six-pair matrices launched from 37df1ef: 24 bootstrap cells (6000 refits,
three numerical workers) and 18 profile scans (one worker). Combined numerical
cap remains four. Durable child status is in each matrix manifest. No new pairs
or thermal orders are included.

P1 Gaussian and L60 block uncertainty cells completed 400 refits each, all usable.
The Gaussian fit selects the opposite pole labeling in 8/400 draws; L60 blocks do
so in 3/400. A central 95% interval can hide this small second basin. Preserve
mode counts and unlabeled pole summaries. The step ambient bound is active in
162/400 and 165/400 respectively. P2 theta profiling flags two remote grid points
(theta 0.0228 and 0.271 s, twice-NLL differences above 818) for noisy stationarity
checks despite near-identical stationary attempts. Original flags are retained;
audit those attempts before reporting reconciled profile support.

Report sources now compile. The first-page ruled-margin placement was rendered,
corrected to overlay coordinates and visually checked. This is an unfinished
report scaffold, not a deliverable. Native literature intake published the
reviewed Hartikainen/Sarkka author manuscript with local byte readback; final
claim/page grounding and new report packet export remain pending.

All 18 profiles are terminal. The independent audit from a991414 reconstructs
minima and crossing likelihoods and reconciles all five remote flagged points
using saved stationary attempts within 1e-8 NLL. Exact dense covariance gradients
and independent mean derivatives pass the unchanged stationarity criterion. No
lower reference minimum was found. Original flags and solver records remain.
The fourth numerical slot now runs the prediction supervisor; it waits for each
pair's terminal Gaussian and L60 block cells, then evaluates its ten other records.

Full 400-draw P1 predictions are terminal and pass independent propagation/CDF
checks. Across ten evaluation records, zero-ambient RMSE is 2.21935 C, compared
with 2.17516 C for OLS; Gaussian observation-range inclusion is 19.02%, rising
to 58.51% with the bounded ambient envelope. Fitting an ambient correction to
the complete evaluation response lowers RMSE to 0.92980 C but is retrospective
reconstruction. These are descriptive fractions under reused records/serial
dependence, not independent-binomial coverage tests. The report now leads with
poor transfer despite narrow conditional parameter ranges.
