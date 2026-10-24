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
| 1. Decompose errors and compare repeats | Per-run and phase-fold plots reviewed; descriptive decomposition complete |
| 2. Frequency-response estimates | 54 estimates audited; joint harmonic input correction audited; limitations retained |
| 3. Flexible shared LTI benchmarks | Two/three-state real-pole fits audited; four-state fit running |
| 4. n=1/2/3 matched comparison | 24 starts each plus fixed-weight/IID sensitivities audited; profiles running |
| 5. Evidence-guided follow-ups | Numerator-bound and transport-delay tests motivated; not yet implemented |
| 6. Investigation report and review | report.md updated with qualified findings; incomplete |

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

## Latest comparisons

All three chain fits, two/three-state real-pole benchmarks and six secondary
chain fits are terminal and independently audited. Primary fits use 24 starts;
secondary fixed-reference OU and IID fits use eight. The n=1 reference agrees
with the original engine. n=2 suppresses part of the fast residual cycle but
increases raw RMSE under joint noise fitting; n=3 is worse under that criterion.
The real three-state benchmark reduces fast phase mismatch to about two degrees
and input/innovation association to 0.038–0.064. It also hits a numerator bound,
uses a weakly estimated 0.14-second pole, and has nonminimum-phase zeros and an
early sign-changing impulse. These features motivate bound sensitivity and an
explicit transport-delay candidate; do not interpret the states physically.

Per-run plots and harmonic/causal forecasting diagnostics are reviewed. Use
reference_diagnostics_v2: its input harmonics are projected jointly over partial
cycles. The original separate-frequency input projections leaked the fundamental
into higher harmonics; forecasts and plots were unaffected. Local sensitivity
and projected-curvature results are in information_initial and
information_flexible3. Mean information can have full numerical rank while the
flexible model's fast pole and numerator remain practically unestimable.

At the last checkpoint flexible4 and chain_profiles were still live. Reconcile
actual process identities and terminal artifacts before continuing. Finish and
audit those outputs; complete physical/bound sensitivities and profile audits,
then the report requirement audit. The active goal is not complete.
