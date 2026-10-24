# Current n=1 residual teaching study

Goal active from 2026-10-24 16:39:24 UTC. Agreed scope: `../review/approach.md`.
Implementation: `execution_plan.md`; records/pairs: `config.json`.
This phase leaves the completed October 21 study and deliveries unchanged.

| Phase | State | Next gate |
|---|---|---|
| Data/input and exact mean audit | Passed in frozen source | Preserve conditional startup/alignment assumptions |
| Tier 0 mean/error exploration | Eight cells complete and independently audited | Timing/startup/bound sensitivities and refit-calibrated adequacy diagnostics |
| Tier 1 P1 uncertainty/evaluation | Pending | Full restricted refits and checks |
| Tier 2 P2-P6 | Pending | Five further pairs; no full enumeration |
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
with exit code zero. No numerical worker remains live. Reconcile receipts in
`run_manifest.json` before launching the next phase.

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

## Next gates before structure selection

The thirty-second sine runs retain an approximately thirty-second innovation
cycle and maximum absolute command/innovation correlations 0.457-0.481 over the
declared +/-30 sample-lag search. Linear ambient drift barely changes this.
The reference Ljung-Box p-values are descriptive, not simulation-calibrated.
Do not treat eleven small p-values as eleven independent calibrated tests.

1. Compare left/right command alignment under the same n=1 model, starting with
   the mean and the OU-plus-measurement working kernel. This is a declared
   sensitivity, not authorization to replace the documented acquisition order
   merely because another convention fits better.
2. Examine input-related discrepancies, startup/first-reading and active ambient
   bounds. If useful, protocol-specific fits may diagnose the shared-mean
   assumption; they are exploratory Tier 0 checks, not extra calibration pairs.
   Do not absorb deterministic input-related bias into a new random kernel just
   to force whiteness.
3. Run refit-calibrated simulation diagnostics for the leading working error
   description, retaining the max-lag search and per-protocol statistics. Check
   whether remaining covariance/mean inadequacy warrants a bounded comparison
   or must remain unresolved in the report.
4. Verify a selected variable-projection result against direct parameter fitting.
   Then document the Tier 0 structure decision and limitations before Tier 1/2.
   No adequate kernel or ambient structure has yet been selected. Subsequent
   uncertainty must retain any unresolved model/input-dependence qualifications.
