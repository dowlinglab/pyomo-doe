# Current n=1 residual teaching study

Goal active from 2026-10-24 16:39:24 UTC. Agreed scope: `../review/approach.md`.
Implementation: `execution_plan.md`; records/pairs: `config.json`.
This phase leaves the completed October 21 study and deliveries unchanged.

| Phase | State | Next gate |
|---|---|---|
| Data/input and exact mean audit | Passed in frozen source | Preserve conditional startup/alignment assumptions |
| Tier 0 mean/error exploration | Eight covariance cells running after audited OLS | Independently reconstruct full likelihoods; diagnose innovations |
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
The covariance comparison is in progress; reconcile its live sessions and output
receipts in `run_manifest.json` before launching new workers.

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
The covariance/iterative-GLS comparison uses the four declared kernels under both
ambient treatments, with 24 joint starts per cell and one alternating-GLS history.
`audit_covariance.py` passed sixteen synthetic dense-Cholesky comparisons,
including stationary initialization, irregular sampling and boundary limits.
The first source-bound four-start pilot retained a stationarity warning;
`methods.md` explains the central-derivative refinement and preserved records.
The refined pilot gives projected gradient about 1e-4, with GLS/joint likelihood
agreement within 4e-7. All full cells run from b84306e. Preliminary innovation
diagnostics still show protocol-dependent correlation; no adequate kernel or
ambient structure has been selected. Independent saved-likelihood audits,
per-run plots, calibrated diagnostics and assumption sensitivities remain next.
Do not start restricted-pair fits until the Tier 0 structure decision is
documented, including unresolved limitations.
