# Retrospective transfer design

The declared primary candidate has a 10% conductance band: twelve leave-run-out
and four leave-protocol-out fits. A shared-parameter baseline has four
leave-protocol-out fits. The protocols are 30/120/300-second sine and step.
All twelve records informed structure selection. This is internal transfer,
not untouched validation. The full-data fit supplies an optimization warm start;
only the training records enter each fold's likelihood and nuisance estimates.

The centering reference changes with the retained records. Removing a protocol
changes the training geometric nominal conductance, so raw K0, tA, Us/Ua0 and
Ux/Ua0 values are not directly comparable across folds as if their Ua0 were
fixed. Under a reference shift, all four scale together. K0/tA=beta/CH,
(Us/Ua0)/tA=Us/CH, (Ux/Ua0)/tA=Ux/CH, rS and L are reference invariant.
Compare those combinations and actual forecasts when discussing stability.
The unknown-run hard band is explicitly relative to the regressed training
reference; it does not infer a population reference or variability law.

## Reference transport stress check

The first omitted-protocol results exposed a distinction between the original
cohort reference and the re-centered training reference. Before adding any
forecasts for this question, transfer_config.json declares an additional check
for all four primary protocol folds. It preserves the canonical forecasts and
all training fits.

Suppose N=12 original log deviations sum to zero and lie in [l,u], with m
retained and k=N-m omitted. The training mean is minus the omitted sum divided
by m. An omitted ratio to the training nominal is therefore
exp(delta_test + sum(delta_omitted)/m). Feasible extreme assignments give bounds
[(1-b)^(N/m),(1+b)^(N/m)]. The bounds depend on cohort/training sizes and the
declared band; fitted test outputs do not choose them. A separate all-data-fit
diagnostic shows how its fitted deviations re-center, without treating them as
measured conductance or supplying them to blind predictions.

These wider forecast ranges leave the training-band constraints unchanged.
They measure sensitivity to the reference convention, rather than transporting
the whole cohort likelihood. Coverage remains descriptive; this does not
calibrate an unknown-run distribution. The default forecast arithmetic must
match its saved canonical outputs before the extra check is measured.

## Canonical forecasts and parameter sensitivities

For an unknown run, set its conductance ratio and ambient offset to 1 and 0 for
the nominal forecast. Initialize cold heater/sensor pairs from the first reported
temperature pair. No later test output estimates conductance, ambient or physical
parameters. Compare with an outer scenario envelope over the training nominal's
permitted conductance band and ambient ±2 C. The ambient forcing is linear and
positive, so its two endpoints suffice at each conductance ratio. Conductance
grids of 201/401 points, or 801 if needed, must agree within 1e-4 C. This is a
numerically converged scenario range, not a distribution over run conditions.

Propagate the training noise to the reported measurement law. The predicted
reported mean sums code levels times their Gaussian bin probabilities. It can
differ from the physical latent temperature, especially for floor bins. Marginal
reported-code quantiles follow the monotone quantizer applied to the analogue
Gaussian quantile. They account for stationary correlated disturbances,
independent analogue noise and reporting bins at each time; they are not
simultaneous trajectory intervals. Show free simulation separately from the
filtered one-step bin score, whose residual state uses past test temperatures.

Parameter uncertainty is a separate component. The prediction driver can read
training-fit-bound parameter draws, checking their hashes, coordinate dimensions,
bounds and centering. It propagates every draw's physical and noise parameters,
then computes empirical Gaussian-mixture quantiles. Taking each draw's worst
allowed ambient/conductance means gives an outer scenario sensitivity. These
ranges inherit the generating draw method's approximation and calibration
limits. A mixture of bootstrap parameter estimates is not automatically a
frequentist predictive interval, especially at bounds and across lag basins.
Do not label noise-only scenarios as total parameter-inclusive uncertainty.
Validate the chosen parameter approximation against the refitted recovery study;
report failure or unresolved calibration explicitly. Model-form and bin/startup
sensitivities remain separate from within-model sampling uncertainty.

Independent checks compare dimensional trajectories at nuisance endpoints,
perturb every future test output to prove prediction invariance, integrate the
actual reporting function to check expectations, verify discrete coverage and
mixture limits, and check every fold's train/test separation. Numerical checks
do not establish scientific transfer or uncertainty calibration.

Each measured fold needs a terminal selected-fit audit before a scientific
forecast claim. audit_transfer_prediction.py checks full nominal dimensional
trajectories, reported-code expectations, every saved RMSE/coverage count,
quantile CDF brackets, nuisance endpoints and refined conditional bin scores.
Independent Gaussian prefix rectangles check the observation probability.
The four primary omitted-protocol canonical forecasts have passed this full
audit. Parameter-mixture forecasts and the other folds remain unfinished.

Parameter files must bind to the draw receipt and its information audit.
Prediction output saves component means/noise at nine evenly spaced sample
indices plus the nearest samples to 60, 180 and 450 s. The auditor checks all
mixture components' discrete-quantile CDF brackets at those times and selected
draws' physical containment. It does not independently recompute every component
trajectory at every time. The final report must retain that limitation.

The execution manifest owns transfer_analysis, the source index for
summarize_transfer.py. Primary leave-run, primary leave-protocol and shared
leave-protocol groups remain separate: each contains a record only once when
complete. Their groups reuse the same observations. The aggregation checks
weighted SSE/RMSE, counts and interval widths; it never treats group reuse as
new validation replication. It also reports changes from 64 to 256 local draws,
using the same declared random prefix, before interpreting mixture coverage.
Missing or failed folds stay visible. A complete numeric accounting table is
still not a calibration result or an untouched-validation result.

## Sequential uncertainty execution

The committed run_transfer_uncertainty_queue.py runs one fold at a time. Each
fold needs a terminal fit collection and its passed full selected-point audit.
The queue reuses a canonical forecast only through a snapshotted manifest index
with matching fit, prediction and independent-audit hashes. Otherwise it creates
and audits that forecast. It then runs the declared reference stress check for
primary protocol folds, observed information, independent information audit,
local parameter draws and the 64/256-draw forecast/audit pair. Failed curvature,
sampling or prediction checks stop the queue and retain the failing stage.
The queue does not modify the manifest; the controller reviews its completed
outputs before adding those paths to transfer_analysis.

For the first measured pipeline, freeze this driver with --input pointing to
results/transfer_primary_protocols, --fit-audit pointing to
results/transfer_primary_protocols_audit, --folds protocol_30s and
--reuse-noise-index pointing to the absolute run_manifest.json. Use a new
absolute output directory. Inspect its actual information/draw audits before
dispatching the other three protocol folds. For the later run/shared groups,
the same driver creates any canonical forecasts not present in that index.

stage_execution.py launches each stage as a real child of the frozen queue,
with its own immutable-source receipt, PID/command, log and actual child exit.
It rejects existing output directories. The queue waits for one stage at a
time, so it and its one numerical child occupy one of the four worker slots.
reconcile.py includes nested receipts and reports live_worker_slots separately
from all live process identities. The stage_execution_check control passed
real successful/invalid-CLI exits and no-overwrite checks; it is not a scientific
uncertainty result.

The --forecast-only execution mode allows the canonical simulation/noise and
hard-condition forecasts to be audited before the longer information/draw
stages. It reports complete_forecasts_parameter_pending and explicitly leaves
parameter_uncertainty_complete false. Use it to finish the primary-run and
shared-protocol mean comparisons first, then index those canonical forecasts
and reuse them in the full default pipeline. It does not replace any of the
nineteen remaining parameter pipelines. The final transfer summary still
requires all declared 64/256-draw components and the primary-protocol reference
checks; the existing completion-control fixture rejects twenty noise-only folds
as complete. The already validated full default execution path is unchanged.

## Nonpositive-curvature investigation

The first remaining protocol pipeline (omitting the 120 s sine records) stopped
before accepting parameter draws. Its independent information auditor initially
crashed on an empty active-bound set. The corrected audit checks complete
constraint rank/dimension, equality and active restrictions, and orthogonality.
Explicit four-coordinate controls cover empty/nonempty active sets and reject
missing directions or constraints. No bound tolerance was enlarged.

The corrected full audit confirms a separate negative observed-curvature
direction, eigenvalue -0.00735, dominated by log sensor loading. Producer
directional checks are stable at two steps; independent dimensional, dense
analogue information and direct loading-coordinate curvature checks passed.
The diagonal loading curvature is positive, so the weak negative direction
includes compensating nuisances. The saved fit's first-order check had passed;
that did not establish a local minimum or support a Gaussian covariance.

Before any new optimization, transfer_config.json declares four additional
loading seeds, 0.03, 0.3, 3 and 30, within the original bounds. Each uses the
usual two lag-branch starts and 300-iteration budget. All mean, ambient,
conductance and noise nuisances remain free, with the same training subset,
criterion, feasibility/stationarity rules and tolerances. The original fit is
also a candidate. Selection uses the lowest successful likelihood objective,
never positive covariance or test-output performance. Seed files explicitly
assert no objective value or optimizer success, and preserve every other
starting coordinate and input. A provenance/mutation/bounds control precedes
the measured refinement.

refine_transfer_loading.py is gated by the terminal failed curvature audit and
matching fit/information/matrix hashes. It preserves all attempts and writes a
single-fold collection for a new independent selected-point audit. New
information and its independent audit must follow before parameter draws.
If the selected fit changes, regenerate that fold's canonical forecasts and
keep the original forecasts for comparison. No eigenvalue clipping, bound
snapping, covariance substitution or unreported tolerance change is allowed.

### Loading-curvature convergence after the refined fit

The refined protocol120 point has H22=249.20545; the old forward .005
stencil fails the unchanged .001 gate by0.0375. Before changing the auditor,
diagnose_loading_stencils.py will compare independent dimensional forward
likelihood curvatures on steps .02,.01,.005,.002,.001. It retains the old
second-order one-sided stencil and adds a fourth-order centered stencil only
where both sides are inside the original loading bounds. All evaluations and
source/fit bindings are saved. A stable high-order range must agree with the
stored H before any audit retry. No tolerance, fitted bound or eigenvalue changes.

The shared-model protocol120 fold independently fails positive curvature in
the same weak loading direction. The existing four-seed, two-branch objective-only
refinement policy applies to that fold too, with its own original fit/audit and
failed information bindings. It does not change the held-out records or criterion.

The five-step refined120 convergence study is terminal, exit0. Forward
curvature errors0.6199,0.1516,0.0375,0.00598,0.00152 shrink quadratically;
fourth-order centered errors remain1.2e-5--2.2e-5 across .02 through .001.
The correction therefore uses centered fourth-order scalar differences at
.01 and .02 only if both +/-0.04 lie within the original loading bounds.
Near a bound it keeps the original .005/.02 forward stencils. Selection uses
feasible stencil support, never positivity or passing error. The .001 absolute
gate, stored Hessian, fitted point and all other checks remain unchanged.
A fresh full information audit must pass before any draw is accepted.
