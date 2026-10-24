# Shared TC-Lab model investigation

Status: baseline diagnostics and five selected dynamic models are audited. The
four-state benchmark, profiles and evidence-guided follow-ups remain in progress.
The [plan](../review/tier0_followup_investigation.md) defines the scope.

| Question | Evidence so far | Status |
|---|---|---|
| Does the saved reference reproduce? | All 12 trajectories and dense likelihood reconstructed; pooled RMSE 0.48320 C | Numerical audit passed |
| Is the fast residual cycle reproducible? | Phase-aligned residuals and innovations show a similar cycle in all three fast-sine repeats | Descriptive evidence; per-run decomposition interpretation pending |
| Does n=1 miss amplitude or phase? | Fast-sine response has approximately 28–32 degrees additional lag across windows, repeats and nuisance treatments | 54 harmonic estimates independently reconstructed |
| Can flexible shared LTI dynamics explain it? | Three-state real-pole benchmark reduces fast phase error; its parameters are weakly determined | Four-state fit and bound sensitivity remain |
| Do n=2 or 3 improve the explanation? | n=2 reduces the fast innovation cycle; n=3 worsens it under the working likelihood | Profiles and synthesis pending |
| Are nonlinear or disturbance extensions needed? | Slow trends and small harmonics remain; neither identifies a unique mechanism | Follow evidence |

## Numerical reference

The independent reference uses augmented matrix exponentials, followed by dense
covariance Cholesky whitening. Its likelihood differs from the saved fit by
4.4e-11. This verifies reproduction, without certifying model adequacy.
See [baseline audit](results/baseline_v2/audit.json).

## Frequency response

The nine sine records were analyzed after 300, 450 and 600 seconds. Each window uses
both an offset-plus-trend nuisance treatment and a sensitivity with exponential
nuisance terms at the pooled reference rates. The latter is conditional on those
rates. The input reference integrates the recorded left-held command exactly.

There are 18 estimates per period: three repeats, three windows and two nuisance
treatments. The fast-sine circular phase difference from the matched n=1
prediction ranges from −32.25 to −28.26 degrees. For 120-second tests it ranges
from −1.82 to +0.95 degrees; for 300-second tests from −0.02 to +2.04 degrees.
Long-period estimates retain only 1–2 cycles, so startup and slow
variation require particular care.

The fast response points toward a reproducible dynamic mismatch. It does not
identify pure delay, an additional thermal state, nonlinear behavior or a
timestamp mechanism uniquely. Those alternatives motivate the next comparisons.

There is also a structural constraint: with positive parameters, the n=1
continuous transfer function has phase between 0 and −180 degrees at positive
frequency. The fast measured response is near −193 degrees on the displayed
continuity branch, equivalently about +167 degrees as a principal phase. Under
the steady LTI interpretation, retuning its three positive parameters cannot
match that phase. The finite-window estimates, command timing, nonlinear effects
and disturbance assumptions still qualify this interpretation.

![Measured gain and phase across startup windows](results/frequency_figures/frequency_windows.png)

Measured curves use working HAC with 60 sample lags, with 30 and 120 stored as
sensitivities. These are per-record asymptotic uncertainty calculations under a
working residual description; they do not establish calibrated coverage or
independent repetitions. Displayed phase is unwrapped by increasing frequency
independently for each measured/model curve; principal phases remain in the
saved estimates. The chosen branch does not identify a unique transport delay.

![Phase-aligned residuals and innovations](results/frequency_figures/phase_repeats.png)

The phase-fold plots remove jointly fitted offset/trend terms and show 12 phase
bins after 300 seconds. Bin counts measure sampling density, not independent
experimental replication.

Independent quadrature of the held input and QR regression reconstructed all 54
measured/matched estimates; maximum complex-gain discrepancy is 5.7e-16.
See [estimates](results/frequency/frequency_estimates.json),
[frequency audit](results/frequency_audit/audit.json), and
[error decomposition](results/frequency/error_decomposition.csv).

## Offset, trend and oscillating error

A descriptive, nested projection assigns about 38% of reference residual SSE to
per-run means, another 36% to linear trends, 10% to the sine fundamentals and
16% to the remainder. The sequence is mean, then trend, then fundamental; these
fractions depend on that sequence. They are not causal allocations to ambient,
physics and measurement noise. Ambient changes act through thermal dynamics,
and a residual trend can also reflect an incorrect thermal model.

The [per-run plots](results/reference_diagnostics_v2/runs_step_50.png) show slow
errors alongside quantized temperature changes. The phase-fold plots show a
separate reproducible fast cycle, even though it contributes less total SSE.

## Prediction versus simulation

With the reference thermal mean and covariance fixed at their pooled estimates,
conditioning on temperature history gives the following within-record results.

| Horizon | Conditional prediction RMSE, C | Free-run RMSE on the same targets, C |
|---|---:|---:|
| 1 s | 0.117 | 0.484 |
| 10 s | 0.182 | 0.485 |
| 60 s | 0.292 | 0.494 |
| 300 s | 0.483 | 0.529 |

The forecasting calculation uses temperatures only through each forecast origin.
Its parameters and ambient offsets were fitted retrospectively to all twelve
records. This demonstrates the effect of measurement conditioning; it is not
an independent forecasting test. Dense Gaussian conditioning independently
verified the recursive predictor's mean and variance.

## Current dynamic comparisons

The following selected fits use the same records, ambient bounds, input holds,
startup and OU-plus-measurement family. All numerical covariance parameters are
reestimated. Each model used 24 starts; selected trajectories and likelihoods
passed independent physical-state and dense-covariance audits.

| Model | Raw temperature RMSE, C | Negative log likelihood | Fast input/innovation correlation range |
|---|---:|---:|---:|
| n=1 chain | 0.483 | −7927.2 | 0.457–0.481 |
| n=2 chain | 0.972 | −8237.0 | 0.196–0.227 |
| n=3 chain | 1.188 | −7744.8 | 0.438–0.456 |
| Flexible two-state real-pole model | 1.100 | −8322.0 | 0.217–0.259 |
| Flexible three-state real-pole model | 0.911 | −8426.8 | 0.038–0.064 |

Lower negative log likelihood is better under the working Gaussian law. It need
not mean lower raw temperature error: the selected models also change the
estimated disturbance. For example, its correlation length increases from about
214 s for n=1 to about 1391 s for n=2. Fixed-reference-covariance sensitivities
and IID fits preserve the conclusion that these constrained higher-order chains
do not improve raw temperature RMSE. These results do not establish adequacy
of any candidate's noise law.

The three-state real-pole benchmark matches the fast measured phase within about
two degrees at the 450-second cutoff. It uses a 0.14-second pole and a numerator
coefficient at its declared −2 C/percent bound, with a pair of right-half-plane zeros and
a sign-changing early impulse response. Those features support a more flexible
linear explanation of the observed phase, while leaving physical interpretation
unresolved. Beyond the three measured frequencies, response curves are
extrapolations.

Its scaled mean sensitivity matrix has full numerical rank but a condition
number of about 141,000. The local conditional SD of the log fast time constant
is about 64; two normalized numerator SDs are about 185. A full-rank matrix does
not make those parameters usefully estimable. The result motivates numerator-
bound sensitivities and an explicit transport-delay alternative, whose early
response can remain physically positive. The four-state benchmark is still
running.

## Harmonics and uncertainty qualifications

Joint input-harmonic projection corrects finite-window leakage in the original
diagnostic implementation; use results/reference_diagnostics_v2. The forecast
results were unaffected. Small output harmonics persist, particularly in the
large-excursion 300-second tests. Input discretization, quantization, startup,
slow variation and the short windows prevent unique attribution to nonlinearity.

All three chain mean sensitivity matrices have rank three at the declared
relative thresholds, with Jacobian condition numbers around five. Projected
observed curvature is locally positive at the selected active constraint face.
This differs from an unconstrained full-coordinate Hessian that allows ambient
parameters to leave their active bounds. Neither establishes regular interval
coverage. Gain and total-delay profiles are in progress to examine the bounded
working likelihood directly.

## Next decisions

Finish per-run error and harmonic checks, then test flexible shared LTI dynamics
and n=1/2/3 chains with matched input, ambient and error conventions. Separate
short-horizon temperature-conditioned prediction from free-run simulation.
Use those results to decide which impulse-response, operating-condition or
disturbance comparisons are needed. Additional calibration pairs remain deferred.
