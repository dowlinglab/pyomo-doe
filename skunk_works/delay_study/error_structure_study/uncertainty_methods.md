# Uncertainty and recovery design

The numerical design is declared in uncertainty_config.json before measurement.
The primary reference is the joint 10% conductance-band fit after its independent
selected-point audit. Radiation and wider/narrower bands remain model-form
sensitivities. Neither a likelihood improvement nor a tighter covariance matrix
establishes a physically adequate or transferable model.

## Profiles and information

Profiles fix one explicitly named optimization coordinate and refit all other
physical, ambient, conductance and noise coordinates. Report the gain, short lag,
sensor loading, a representative conductance deviation, long T1 correlation
length and radiative coefficient. Log-coordinate grids are converted to physical
values only when explicitly requested. The fitted point is included. Profiles
use two starts, including the swapped sensor/input-lag basin, and retain failed
points. A profile below the saved unconstrained optimum triggers refinement;
negative likelihood-ratio statistics must not be presented as evidence.

Finite differences of the full bin-likelihood score give an observed Hessian.
Report the coordinate definitions, symmetry error and eigenvalues both in the
centered tangent space and conditional on the active constraint face. Test weak
free-face directions with two step sizes. A derivative outside a constraint face
does not authorize an infeasible confidence direction. Poorly resolved curvature,
negative eigenvalues, bounds and disconnected profiles preclude automatic Wald
intervals. Distinguish exact scale symmetries from finite-data weakness.

For comparison, compute mean-Jacobian information for unreported analogue
temperatures with covariance held known, and for IID errors with the same
marginal variance. These are conditional matrices. The analogue matrix is an
upper bound on expected mean information after quantization: the code score is
the conditional expectation of the analogue score given the code, so the law of
total variance removes a positive-semidefinite conditional-score variance. This
argument assumes fixed known covariance, regular differentiation and the same
mean parameters. It does not replace the observed bin Hessian or include
covariance estimation, uncertain thresholds or model selection.

The coarse short-lag grid also needs a theory-driven exchange check. After its
point audits, fix log input lag at the original unconstrained log sensor time,
warm from the nearest successful coarse secondary-region point, and refit every
other nuisance with the same two-start/300-iteration protocol. The zero-loading
exchange argument motivates this coordinate; a low coarse region near 15 s
alone does not resolve its minimum near the fitted sensor time. Preserve the
coarse grid, audit the added point and refine the unconstrained reference if a
constrained point beats it. This check does not supply interpolated interval
endpoints or physical labels for the two lags.

## Lower loading and lag identity

The baseline repeatedly reaches Us/Ua=0.005. A declared sensitivity lowers that
positive bound to 0.0001 with the same three starting branches. This does not
change the baseline comparison or make a zero-capacity physical state valid.
It tests whether the reported sensor/input lag identities depend on a numerical
lower bound. A short independent synthetic check verifies the transfer limit
and positive-loading trajectories before that refit.

For zero loading, the normalized common/differential heater modes factor as

\[
 G_\pm(s)=\frac{K}{(t_A s+1+\{0,2q\})(1+r_S s)(1+L s)}.
\]

The measured channels are (G_plus+G_minus)/2 and
(G_plus-G_minus)/2. Sensor lag rS and input lag L therefore exchange exactly in
the forced transfer functions at this limit. Unequal initial conditions and
ambient forcing need not share that symmetry, and finite positive loading
breaks it. Quantify how much on a synthetic nominal waveform; do not turn this
limit into a claim that the hardware has a zero sensor heat capacity.

## Fifty refitted recovery/coverage draws

Generate observations from the fitted correlated latent/analogue-noise model
and hardware-bin convention at the actual twelve command/time grids. Keep first
readings fixed consistently with the working likelihood. Each draw refits all
permitted mean and covariance parameters under the same bounds and centering.
Use two starts, one failed-fit retry with three starts, fixed recorded seeds,
single-threaded BLAS and at most four simultaneous numerical workers. Shards
have disjoint draw indices; a pilot is labeled separately and does not silently
replace the fifty-draw study.

To check an interval procedure without a Hessian calculation for every draw,
also fit each draw with gain fixed at its generating value. Other coordinates
remain free. Coverage of the nominal scalar likelihood-ratio region is the
fraction with twice the constrained-minus-free nll no greater than
3.841458820694124. This estimates how the usual 95% scalar cutoff behaves with
the actual nuisance bounds and design. It does not assume the cutoff is valid.
Other parameters have empirical recovery summaries; this single-gain calibration
does not establish coverage for every parameter.

The constrained fit has one start and one two-start retry if needed. If it beats
the free fit by more than 0.001 nll units, refit the free model from that basin
and retain the correction. An unresolved negative difference is a failed
coverage draw. Record all starts, failures, synthetic observations, numerical
barriers, source/input hashes, exact solver tolerances and wall times.

Report the complete-draw coverage with binomial Monte Carlo uncertainty and the
full attempted denominator. Also show worst-case coverage bounds that count
failed draws as either covered or uncovered. Do not discard failures to improve
coverage. Begin with fifty; expansion to two hundred requires a numerical and
coverage review. These conditional simulations do not protect against an
incorrect mean/noise model, unknown threshold phase or a new device population.
Retrospective transfer and initial-condition/bin-phase sensitivities remain
separate required checks.

## Refitted summaries and numerical accounting

audit_recovery.py checks immutable terminal shards, seed/data regeneration,
selected free and constrained fits, and LR arithmetic. Accepted free fits still
receive a numerical audit if the constrained fit failed. Saved diagnostics must
match their row metrics and the frozen diagnostic implementation. Reused point
audits require matching numerical dependencies, raw design, runtime and
fit/trajectory/data hashes. A partial controller pilot cannot replace the
fifty-draw audit or change any fitted outcome.

summarize_recovery.py retains failed and missing draws in worst-case bounds,
reports Monte Carlo uncertainty for complete-draw gain coverage, and compares
empirical estimator spread with the audited generating-point active-face SEs.
The final summary requires that information audit. Changes of active face or
lag basin can invalidate a local-normal interpretation; matching one scalar
coverage statistic cannot validate every parameter mixture or training design.
Estimator quantiles are sampling summaries under the working model, not
hardware confidence intervals.

Refitted deterministic code diagnostics use the same definitions as the
hardware and fixed-parameter references. Diagnostic failures keep a separate
denominator. The references account for nuisance refitting within the selected
family, but not structure selection, threshold uncertainty or a new device.
The short accounting check uses independent binomial-tail roots to verify the
beta-quantile Monte Carlo limits, including all-covered and none-covered cases.
The pinned SciPy emitted an internal beta-quantile overflow warning; every
reported limit was finite, ordered and agreed with the independent roots to
about 1e-15. The warning is retained in the check output.
