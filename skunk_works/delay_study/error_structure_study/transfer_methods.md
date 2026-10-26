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
Full measured-fold use of this new auditor is still pending.

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
