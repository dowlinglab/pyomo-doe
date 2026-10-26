# Faster scores and preserved numerical work

The compiled scalar quadrature implementation retains the reference observation
bins, stationary OU law, posterior-domain construction, Legendre order, tail
cutoff and Fisher-score identities. It changes how arithmetic is executed.
The independent NumPy/SciPy implementation remains the audit reference.

The forward calculation uses stable normal log-CDF differences, including
negative-tail asymptotics below -30 SD. The backward joint grid weights are a
smoothed new-state weight times the conditional probability of an old grid
state. Every such term is bounded by one because its predictive denominator is
the same positive mixture containing its numerator. Directly accumulating those
probabilities avoids repeated dense log-pair arrays without changing the
smoothing law. Closure and posterior-normalization gates are retained.

Two-implementation checks cover all returned moments and mean/noise scores,
both bin phases, a rare two-code jump, and normal tails. The existing independent
Gaussian-rectangle moment integration and finite-difference score checks also
pass with the compiled kernel, as does the full joint score suite across mean
families and initial-state conventions. The full-data benchmark covers five
audited hardware fits and two fixed generated datasets. At order 128, likelihood
differences are below 3e-12 and score differences below 3e-8, with warm elapsed
speed ratios of about 2.5–2.9 under concurrent study jobs.

The 64/96-node pilots do not consistently pass the strict full-data agreement
gate. New measured fits therefore retain 128 nodes and the existing adaptive
increase to 256/512. The speed improvement is not a lower-precision shortcut.
Backend names, frozen source and actual per-record node choices are recorded.

## Continuation protocol

A backend change is permitted only after the checks above and read-only
rechecks of completed starts on the actual generated/hardware records. Existing
source snapshots and outputs remain intact. Before continuing an incomplete
task, its original parent must have an authoritative terminal exit. A deliberate
performance handoff receives an explicit superseded.json reason, linked
benchmark/control/preflight receipts and replacement directory. No restart is
based on an observation timeout or a favorable/unfavorable fit result.

Input SHA-256 bindings and scientific CLI arguments must match, including
the data, warm point, model, band, bins, starts, iteration limit, fixed
coordinates and seed/design configuration. Only output location, checked score
backend and continuation metadata may differ. Completed tasks and completed
starts retain successful or failed status. Successful saved points are
re-evaluated against current likelihood and stationarity before reuse. A
completed failure is never silently promoted or granted an extra data-dependent
attempt. Completed coverage draws, including failed draws, keep their outcome.

Atomic point checkpoints now accompany new solver callbacks. A resumed saved
iterate consumes only the original remaining iteration budget; exhaustion does
not become convergence merely because a point looks stationary. Earlier frozen
runs saved only scalar iteration traces, not unfinished parameter vectors.
Those partial branches cannot be reconstructed: if a performance handoff is
used, their abandoned scalar traces remain preserved and the unfinished branch
restarts from its original declared seed. That extra computational attempt is
disclosed separately from the fifty distinct simulated datasets and their
coverage outcomes. All completed prefixes are retained.

The generated all-fixed control tests exercise completed-prefix reuse, pending
branch execution, changed-protocol rejection, preservation of failed starts,
terminal-task reuse and exhausted-checkpoint handling. They are control-flow
tests, not statistical calibration. Actual recovery datasets are regenerated
under their original seed and checked for exact equality before their original
bytes are copied. Profile grids and nearest-successful warm selection remain
unchanged. Final simulation aggregation must count every declared draw once,
retain failed outcomes and separately list any abandoned computations/retries.
