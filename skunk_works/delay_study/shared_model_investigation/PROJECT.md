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
| 1. Decompose errors and compare repeats | Driver prepared |
| 2. Frequency-response estimates | Synthetic/input-integral/HAC audits prepared |
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
