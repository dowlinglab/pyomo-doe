# Tier 0 follow-up investigation

Status: proposal recorded from Alex's review discussion. This note authorizes
no new calculations; the completed Tier 0–2 study remains at its researcher
review checkpoint. See [current study state](../residual_study/PROJECT.md) and
the [agreed approach](approach.md).

Alex's latest priority is to get to the bottom of the model discrepancies and
understand the system. Teaching remains a later use of the findings; it should
not constrain this diagnostic investigation. The original teaching objective
and completed study remain recorded in the linked approach and project files.

## Why return to Tier 0

Alex's concern is that substantial out-of-sample errors are less surprising if
the shared model already misses systematic behavior when fitted to all twelve
ENV2 records. Before expanding the calibration-pair study, investigate whether
one shared mean and disturbance description can explain these records.

The current n = 1 fit follows the pooled temperature trajectories with about
0.483°C RMSE, using shared K, τ and θ, left zero-order-held inputs, bounded
per-run ambient offsets, and an OU disturbance plus independent measurement
noise. The fastest sine tests retain an input-related residual cycle. Under
the working covariance comparison, linear ambient drift received less support
than the initial unweighted fit suggested. Timing, startup, rounding and ambient
bound checks did not eliminate the cycle.

These findings leave the shared model's adequacy unresolved. Ambient conditions
also affect predictions in a new run, so the pooled residual cycle should not
be assigned responsibility for every Tier 1–2 prediction discrepancy.

## Proposed investigations

### 1. Separate offsets from oscillating mismatch

For each run, plot the measured and fitted temperatures, raw residuals, and
whitened innovations. Summarize average residual, slow trend, and oscillating
component alongside the fitted ambient offset and any active bound. Account
for startup when interpreting ambient effects; an ambient change need not
produce a constant temperature shift throughout the transient.

Compare prospective predictions under declared ambient scenarios with the
existing retrospective ambient reconstructions. Keep those two quantities
separate: estimating an offset from the full evaluation temperature record
uses information unavailable before the experiment. The ±2°C bounds specify
an envelope, without specifying a probability distribution.

Deliverable: a per-run error table and plots showing how much discrepancy is
consistent with ambient uncertainty and how much remains tied to the input.

### 2. Check whether the residual cycle repeats

For each sine period (30, 120 and 300 seconds), align the three recorded repeats
by actual input phase. Show individual repeat residuals and their phase average,
with startup treated separately. Apply the same check to innovations so that
raw mean mismatch and effects of whitening remain visible.

Compare the shared phase pattern with variation among repeats. Serial samples
within one cycle do not supply independent replications; any uncertainty
calculation must respect within-run dependence and the small number of runs.

Deliverable: phase-resolved plots that distinguish a reproducible response
error from variation among repeated experiments.

### 3. Measure amplitude and phase mismatch

Use classical frequency-response estimation at the known drive frequency
ω = 2π/P for P = 30, 120 and 300 seconds. On a settled portion of each
temperature run, fit

\[
T(t)=c+a_T\sin(\omega t)+b_T\cos(\omega t)+r(t).
\]

For this sine convention, amplitude is A_T = √(a_T² + b_T²) and phase is
φ_T = atan2(b_T, a_T). Estimate the heater-input fundamental with the same time
origin and analysis window. Use the continuous left-zero-order-held command,
or explicitly correct a sampled-input estimate for that hold; do not substitute
an ideal sine for the recorded commands. Then estimate gain A_T/A_u and relative
phase φ_T − φ_u. A temperature lag has negative relative phase under this
convention.

Document how startup is excluded and check sensitivity to the selected window,
or explicitly model the transient. Count the settled cycles available,
especially for the 300-second records; assess whether those records support
useful estimates. Account for correlated residuals and the three-repeat
structure in uncertainty estimates. Samples within a run are not independent
experimental repetitions.

A five-second lag at a 30-second period corresponds to a 60-degree phase lag.
That phase could include thermal time constants as well as transport delay.
Show the phase convention and wrapping explicitly; three measured frequencies
cannot uniquely distinguish pure delay from different lag orders.

Inspect harmonics and dependence on operating temperature where the records
support them. Use the resulting frequency pattern to investigate whether gain,
timing, delay representation, or additional dynamics could explain the residual.
Preserve the existing caution about exchanging the two pole labels.

Deliverable: per-repeat measured gain and phase with uncertainty at all three
periods, compared with n = 1, n = 2 and n = 3 model predictions under the same
input convention. Include a short list of mechanisms supported or weakened by
the observations.

### 4. Compare a few targeted extensions

Choose a small set of candidate changes after the first three checks. Possible
changes include another dynamic state, a different delay representation, or
operating-temperature dependence if the diagnostics support it. Keep the n = 1
model as the reference. These are candidate extensions for discussion.

Compare candidates on the same records with the same ambient assumptions,
startup conventions and error treatment wherever feasible. Report changes in
per-run fit, input-related residual structure, uncertainty and estimability,
alongside model complexity. More flexible covariance alone does not establish
that a systematic input-response error has been explained.

Deliverable: a compact comparison identifying which changes reduce the
reproducible discrepancy and which leave it unresolved.

## Additional system-identification checks

### Flexible linear state-space benchmark

Fit a small set of low-order linear state-space models, using subspace
estimation for initialization and prediction-error refinement. Seek stable,
causal dynamics shared across runs, with run-specific initial conditions and
nuisance terms handled consistently with the baseline. State count is separate
from the delay-chain order n.

Use the same records, ambient assumptions and held inputs, and compare model
complexity and innovations. If a flexible linear model repeatedly removes the
input-related pattern, investigate which restrictions of the delay-chain model
prevent that behavior. If it also fails, check excitation, identifiability,
disturbance confounding and estimation before assigning the failure to
nonlinearity.

### Regularized impulse-response estimate

Estimate an impulse response or finite impulse response (FIR) model to inspect
the response's onset and shape without imposing the delay-chain form. Account
for finite records, sparse excitation and ill-conditioning. Report the effect
of regularization on the estimate and its uncertainty. Avoid differentiating
the noisy step response directly or interpreting an apparent onset as a
uniquely identified physical delay.

### Operating-condition and nonlinear diagnostics

Examine residuals and estimated frequency responses against temperature,
heater level, and heating versus cooling. Compare output harmonics with the
actual held input and the recorded temperature quantization before attributing
them to nonlinear thermal behavior. Use the repeats to check whether the
associations persist. This small dataset can confound operating condition,
input protocol and disturbances; an association alone does not identify a
mechanism.

### Prediction horizon versus free-run simulation

Compare one-step and short-horizon predictions conditioned on measured
temperature history with free-run simulations using commands and declared
initial conditions. Match initialization and information available at each
forecast origin; exclude future temperature responses from predictor inputs.
A noise predictor can compensate for deterministic mean-model mismatch, so
report both tasks. A gap between their errors is useful evidence but does not
uniquely diagnose its cause.

Deliverable: a short benchmark matrix and diagnostic plots showing which
response features each method explains, where uncertainty remains large, and
which findings reproduce across runs.

## Next sequence: diagnostics, linear benchmark and delay-order comparison

Begin with amplitude and phase estimation, supported by the offset and
repeatability checks, then use the flexible linear state-space benchmark to
test how much the constrained dynamics explain. Let those findings guide the
remaining diagnostic methods and targeted extensions.

Revisit n = 1, n = 2 and n = 3 delay-stage models as part of that comparison.
Here n is the number of stages in the delay chain. The later question of
fitting with one sine record and one step record concerns calibration data
quantity and is a separate study.

Compare all three delay orders on the same twelve ENV2 records with matched
ambient, startup, zero-order-hold and error treatments. Define θ consistently
as the total mean delay across the chain; for n identical first-order stages,
each stage then has time constant θ/n. Preserve this convention when setting
bounds, generating initial guesses and reporting fitted delays.

Use comparable multistart coverage and optimization effort, documenting any
remaining convergence differences. Compare amplitude and phase mismatch,
input-related innovations, per-run fit, model complexity, estimability and
conditional uncertainty. More stages need not improve the explanation, and a
better fit would not establish the device's physical delay order.

This sequence is a discussion proposal. Numerical work remains pending the
next agreed analysis scope.

## Decision before another calibration-pair study

Review whether the proposed mean and disturbance descriptions explain the
shared trajectories and the reproducible residual patterns. Specify remaining
limitations before returning to the question of what one sine test plus one
step test can identify. Any subsequent use of these records remains
retrospective after pooled model exploration.

Agree on the candidate changes and diagnostic criteria before launching work.
The remaining 21 calibration pairs, Pyomo.DoE, new hardware experiments, and
another report email remain outside this proposal.
