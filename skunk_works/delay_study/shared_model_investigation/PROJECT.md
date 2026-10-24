# Shared-model discrepancy investigation

The goal is active. Scientific scope and rationale are in
[the investigation plan](../review/tier0_followup_investigation.md), committed
at b4f7ef7 after preserving the original brainstorming note at 8cb58c0.
Alex requested continued work after voice chat and removed any duration limit.

## Current state

Phase 0 passed: independent matrix propagation and dense-covariance likelihood
reproduce all twelve saved runs. Pooled RMSE is 0.48320369 C; likelihood differs
by 4.4e-11. Orders 1/2/3 forward preflight passed against ODE and repeated-pole
analytic responses. See results/baseline_v2/audit.json. The first attempt retained
a process-inspection sandbox failure; reconciliation now belongs to the controller. The original residual_study is read-only input to
this goal. Its report was already delivered; send no new email.

| Phase | Status |
|---|---|
| 0. Reproduce and verify | Passed independent reconstruction |
| 1. Decompose errors and compare repeats | Scalar decomposition and phase folds complete; per-run plots/interpretation next |
| 2. Frequency-response estimates | 54 estimates independently reconstructed; harmonics/limitations remain |
| 3. Flexible shared LTI benchmarks | Pending |
| 4. n=1/2/3 matched comparison | Pending |
| 5. Evidence-guided follow-ups | Pending earlier diagnostics |
| 6. Investigation report and review | Pending |

Reconcile [run_manifest.json](run_manifest.json) and actual process identities
before launching jobs. Use committed-source snapshots and the pinned scientific
Python runtime. Keep at most four long numerical processes, each with one BLAS
thread. This is a computation limit, not permission to spawn additional agents.
User-authorized note-editing workers have finished their scoped edits.

The existing hourly thread heartbeat has been updated and read back as ACTIVE
for this investigation. It replaces its old Tier0–2 instructions. Mark the goal
complete and pause that heartbeat after the scoped report and checks finish.
The remaining calibration pairs, DoE, hardware and new email remain deferred.

## Current evidence and next work

The 54 frequency cells cover startup cutoffs of 300, 450 and 600 seconds, two
nuisance treatments and all nine sine records. Independent continuous-command
quadrature and QR solves agree within 5.7e-16 in complex gain. The fast 30-second
records show 28.3–32.3 degrees more lag than matched n=1 across these choices.
Slower 120-second discrepancies are about −1.8 to +0.95 degrees; 300-second
ones about −0.02 to +2.04 degrees, with only 1–2 settled cycles. These are
per-record descriptive estimates with working HAC uncertainty. They do not
identify a unique mechanism. See report.md for current interpretation.

Use results/frequency_figures for reviewed plots. The original phase display
wrapped at 180 degrees; the revised display unwraps increasing frequency
independently for measured/model curves, preserving all numerical estimates.
The raw estimates and independent audit retain input bindings and audit scope.

Next finish Phase 1 per-run plots/energy interpretation, examine harmonics, then
implement the shared flexible linear benchmark and matched n=1/2/3 fits. All
current numerical jobs are terminal. Reconcile process IDs before launching.
