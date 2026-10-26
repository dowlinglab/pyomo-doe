# Retrospective transfer design

The declared primary candidate has a 10% conductance band: twelve leave-run-out
and four leave-protocol-out fits. A shared-parameter baseline has four
leave-protocol-out fits. The protocols are 30/120/300-second sine and step.
All twelve records informed structure selection. This is internal transfer,
not untouched validation. The full-data fit supplies an optimization warm start;
only the training records enter each fold's likelihood and nuisance estimates.

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
