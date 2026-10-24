# Investigating the shared TC-Lab model

Status: execution plan requested by Alex during review of the completed Tier 0–2
study. Commit this plan before starting the new goal. This file owns the new
scientific scope; the completed study and its delivered report remain unchanged.

## Objective

Determine why one shared model leaves reproducible, input-related errors across
the twelve ENV2 experiments, and identify the simplest supported explanation.
Alex's immediate priority is understanding the system. Teaching material and
restricted-data calibration can follow once the discrepancies are understood.
There is no prescribed duration. Advance through the scientific decisions below;
report unresolved questions when these records cannot distinguish explanations.

## Starting evidence

The completed n = 1 fit uses shared positive K, τ and θ, left-zero-order-held
heater commands, per-run constant ambient offsets bounded by ±2°C, and an OU
process plus independent measurement noise. Its pooled temperature RMSE is about
0.483°C. The fastest sine tests retain a roughly 30-second, input-related
innovation cycle. Linear ambient drift received less support after accounting
for correlated errors. Timing, startup, rounding and ambient-bound sensitivities
did not eliminate that cycle.

These results establish a useful shared trajectory fit, while leaving the joint
mean/error description inadequate. Ambient conditions also affect new-run
predictions; the repeating residual cannot explain every Tier 1–2 discrepancy.

Use the completed [project](../residual_study/PROJECT.md),
[run manifest](../residual_study/run_manifest.json),
[execution plan](../residual_study/execution_plan.md), and
[agreed approach](approach.md) for implementation details and previous results.
Their old launch instructions describe completed work. Preserve that work and
reconcile terminal records before launching anything new.

## Data and comparison rules

- Use the twelve distinct same-device ENV2 records: three repeats each of
  30-second, 120-second and 300-second sines, and three 50% step tests. Calibration
  aliases are duplicate records. Q2 is zero in these records. Retain Alex's
  accepted same-device provenance assumption.
- Keep the primary left-hold timing, first-reading ambient reference, ±2°C
  constant ambient offsets and declared initial-state conventions. Any departure
  is an explicit sensitivity, with its rationale and effect recorded.
- A shared constant alpha is absorbed into K. Fit no extra alpha multiplier;
  external calibration is needed only for absolute physical interpretation.
- Use the same observations and comparable ambient, startup and stochastic
  treatments when comparing structures. Re-estimate their numerical nuisance
  and covariance parameters; do not silently fix them to the old best fit.
- Separate raw temperature residuals from whitened innovations. Keep trajectory
  fit, predictive performance and innovation adequacy as distinct assessments.
- These records have all informed model selection. Any later held-run assessment
  is retrospective; no untouched validation set is created by relabeling them.

## Investigation sequence

| Phase | Scientific question | Required output | Decision |
|---|---|---|---|
| 0. Reconcile and reproduce | Are the inputs, saved fits and numerical tools consistent? | New execution manifest and independently reconstructed reference | Resolve discrepancies before interpreting new fits |
| 1. Locate the discrepancy | Is it offset, slow drift, reproducible cycling, or a mixture? | Per-run error decomposition and phase-aligned repeat plots | Identify response features that a candidate must explain |
| 2. Measure frequency response | Does the shared model miss amplitude, phase, or both? | Measured gain/phase with uncertainty at three frequencies | Assess which dynamic restrictions merit testing |
| 3. Establish a linear benchmark | Can a more flexible shared LTI model explain those features? | Low-order state-space comparison and prediction/simulation diagnostics | Distinguish evidence against the delay-chain restriction from broader ambiguity |
| 4. Compare delay stages | Do n = 1, 2 and 3 reproduce the measured response and residuals? | Matched pooled fits and estimability/uncertainty comparison | Retain supported orders and document indistinguishable ones |
| 5. Test remaining explanations | Do response shape or operating conditions explain what remains? | Evidence-guided FIR, nonlinear or disturbance checks | Identify a supported revision or a limitation of these data |
| 6. Synthesize | What explains the discrepancies, and what remains unresolved? | Investigation report, figures and reproducible comparison table | Researcher review before another calibration-pair study |

Phases 0–4 and synthesis form the main comparison. Preliminary nonlinear and
harmonic diagnostics belong in Phases 1–2; they can change the later priority.
Phase 5 follows specific evidence rather than an exhaustive search over models.
A failed or unidentified benchmark is an informative result, not a reason to
force a preferred explanation. No phase has a wall-clock deadline.

## Phase 0: reconcile and reproduce

Inspect running processes and saved manifests before starting new work. Create a
separate investigation directory and manifest linked to this plan. Record source
commit, environment, raw-file hashes, run labels, input timestamps and numerical
assumptions. Reuse existing audited routines where they apply.

Independently reconstruct the saved pooled n = 1 trajectory and likelihood.
Reproduce the reported residual cycle and per-run statistics before adding
models. Verify synthetic amplitude/phase recovery and the higher-order forward
solver before interpreting experimental estimates. Keep new outputs separate
from completed Tier 0–2 results.

## Phase 1: separate ambient effects and reproducible response errors

For each run, show measured/fitted temperatures, raw residuals and innovations.
Summarize mean residual, slow variation, oscillating component, fitted ambient
offset and active bounds. Include startup explicitly: an ambient offset affects
a transient through the dynamics and need not create a constant output shift.
State the decomposition method so its components are reproducible.

For each sine period, align the three repeats by actual input phase. Show each
repeat and the phase average, treating startup separately. Compare the shared
cycle with variation across repeats. Apply the same analysis to residuals and
innovations so whitening does not hide the original temperature discrepancy.

Use the previous prospective ambient scenarios and retrospective ambient
reconstructions to explain their different information requirements. A correction
estimated from the complete evaluation response is retrospective. The ±2°C
bound describes an envelope and supplies no probability distribution.

Deliver a per-run table and repeat plots. Identify whether the error persists
at the same input phase and which protocols contribute most. Serial samples
within a cycle are not independent experimental repetitions.

## Phase 2: estimate amplitude and phase from the records

At each known drive frequency ω = 2π/P, fit the temperature fundamental using

\[
T(t)=c+a_T\sin(\omega t)+b_T\cos(\omega t)+r(t).
\]

For this convention, amplitude is A_T = √(a_T²+b_T²) and phase is
φ_T = atan2(b_T,a_T). Estimate the input fundamental against the same time
origin and compatible analysis window. Use the recorded continuous left-held
commands, or derive and verify the corresponding hold correction for sampled
input coefficients. Do not replace the recorded input with an ideal sine.
Estimate gain A_T/A_u and relative phase φ_T−φ_u; lag has negative relative phase.

Declare a transient treatment before choosing the final estimates. Compare
reasonable settling windows, count available settled cycles, and inspect whether
the 300-second records support useful precision. Where startup cannot be
excluded reliably, compare an explicitly modeled transient and describe the
additional assumptions. Fit with offsets or justified slow nuisance terms;
check that these do not absorb the oscillation of interest.

Account for serial dependence, repeat-to-repeat variability and small output
amplitudes when estimating uncertainty. Preserve the three repeats separately.
Report when phase or gain is weakly determined. Document wrapping and phase
reference rather than selecting branches to agree with a candidate model.

A five-second lag at a 30-second period is a 60-degree phase lag. Thermal time
constants also produce phase lag, so this does not identify five seconds of
pure transport delay. Three frequencies cannot uniquely establish a delay
mechanism or model order.

Plot measured gain and phase with uncertainty at 30, 120 and 300 seconds. Overlay
candidate predictions using the same input/sampling convention. Inspect harmonics
relative to those in the actual held command and the temperature quantization.
Record which discrepancies indicate amplitude, timing or operating-condition
mismatch; do not assign a unique mechanism from these plots alone.

## Phase 3: benchmark shared linear dynamics

Fit a small, declared set of stable causal linear state-space models, with
subspace initialization where supported and prediction-error refinement. Start
with two to four dynamic states as a proposed computational choice, expanding
only if excitation and diagnostics justify it. State count differs from delay
stage count. Use available Python tools or a verified implementation; a named
MATLAB method is a methodological reference, not a software requirement.

Preserve run boundaries, initialization and ambient forcing consistently across
records. For a black-box model, specify exactly how an ambient offset enters;
an arbitrary output intercept is not automatically equivalent to the thermal
ambient transient. If equivalence cannot be imposed, label that comparison as
conditional on a different nuisance treatment and run a matched sensitivity.

Compare free-run response, amplitude/phase and residual input association with
the n = 1 reference. Report complexity, numerical stability and weakly estimated
modes. A flexible LTI model that removes the repeatable discrepancy supports
revising the chain restriction. Failure also permits weak excitation, estimation
problems or disturbance confounding; it does not establish nonlinearity.

Compare one-step/short-horizon predictions using available temperature history
with free-run simulations driven by commands and initial conditions. Declare
forecast origins and information sets. Exclude future temperatures. Match
initialization and distinguish measurement conditioning from changes in physics.
A noise predictor can compensate for a wrong mean response; a prediction versus
simulation gap is evidence to investigate, not a unique diagnosis.

## Phase 4: revisit n = 1, 2 and 3 delay stages

For each order, fit one shared model to the same twelve records. Use θ as total mean
chain delay, so each identical stage has time constant θ/n. The command-to-
temperature transfer function for the reference structure is

\[
G_n(s)=\frac{K}{(1+\tau s)(1+\theta s/n)^n}.
\]

Its continuous-time fundamental phase for positive K is
−atan(ωτ)−n atan(ωθ/n), before accounting for the input/sampling reference.
Verify the implementation against independent propagation and synthetic cases.
Each fixed n retains three shared thermal parameters; increasing state dimension
does not by itself add freely fitted thermal parameters. Count actual nuisance,
noise and any additional free parameters when discussing complexity.

Use comparable multistart coverage, report optimizer failures and check possible
alternative basins. Preserve the n = 1 pole-exchange caveat: input-response
symmetry alone does not assign physical pole labels, while ambient forcing can
alter that symmetry. Check the corresponding ambiguities for each higher order.

Compare per-run fit, frequency response, input-related innovations, covariance
behavior and parameter/response estimability. Use local sensitivity singular
values or information eigendecompositions with declared scaling; use profiles
where curvature or bounds make local ranges unreliable. Begin resampling with
an audited pilot if needed, and expand only to resolve an identified inferential
question. Avoid repeating the previous large uncertainty matrix by default.

Improvement must reproduce across repeats and reduce the diagnosed discrepancy,
rather than merely lower the pooled objective. Use nominal likelihood criteria
with their working-model qualifications. A better-fitting n is not proof of the
physical number of delay stages; report orders the data cannot distinguish.

## Phase 5: follow the evidence

Select follow-ups from the preceding results and record why each was needed.

- **Regularized impulse response:** estimate a causal FIR/impulse-response
  benchmark for onset and shape. Assess excitation and conditioning before
  interpreting it. Show sensitivity to regularization and response horizon;
  regularization adds assumptions and trades variance for bias. Avoid directly
  differentiating noisy step records or calling an apparent onset a unique delay.
- **Operating conditions and nonlinearity:** examine residual/frequency-response
  dependence on temperature, heater level and heating versus cooling. Compare
  output harmonics with the actual held input and quantization effects. Seek
  repetition across runs. Protocol, operating condition and disturbance can be
  confounded in these data; report associations accordingly. Fit a small physical
  extension only when these diagnostics motivate a specific term.
- **Disturbances and ambient treatment:** revisit linear ambient drift or a
  parsimonious disturbance representation when remaining errors are slow or
  repeat-variable. Use the existing constant-offset fit as reference. Any drift
  limits are declared technical choices, not previously measured bounds. Assess
  mean/error confounding. A flexible covariance alone cannot establish that an
  input-linked response error has been explained.

If a full two-temperature physical model or a substantially wider model search
appears necessary, document that scope extension for review. It is not an
unbounded continuation of the reduced-model comparison.

## Evidence, reproducibility and stopping decisions

Use Python source files and reproducible entry points. Freeze source and raw
inputs for long jobs; save seeds, configurations, solver termination, failed
attempts and recoveries. Reconcile running jobs at each continuation. Commit
coherent source/results changes without overwriting the completed study.
Independently verify selected propagation, likelihood and diagnostic arithmetic.
Inspect plots before interpreting or reporting them.

Define diagnostic statistics and window/lag sensitivities before selecting
winning models. Existing input-correlation searches and phase diagnostics can
supply descriptive checks. If resampling calibrates new statistics, repeat the
actual fitting/selection steps it is intended to calibrate and state the assumed
noise law. Preserve model-inadequacy and data-reuse qualifications. No nominal
95% interval or residual test guarantees coverage under a wrong mean model.

Stop a candidate calculation when its assumptions fail, its response is not
identified by these inputs, or an independent numerical check fails. Diagnose
and record the failure before spending more computation. Complete the goal when
the comparisons and audits are reported, including unresolved alternatives and
data limitations; identifying a uniquely correct model is not required.

## Report and researcher checkpoint

Produce one concise Markdown investigation report linked to reproducible figures
and tables. Start with question, evidence and status for each phase. Explain:

1. Which part of the discrepancy is ambient-related versus input-related.
2. Whether amplitude/phase and residual patterns reproduce across runs.
3. What flexible linear models and n = 1/2/3 comparisons establish.
4. Whether nonlinear or disturbance explanations receive additional support.
5. Which uncertainty statements survive the diagnostics and which remain
   conditional on an inadequate model.
6. What these findings imply for the previous two-record calibration results,
   and what evidence would distinguish the remaining explanations.

Report prediction and simulation separately. Keep any retrospective response-
conditioned reconstruction separate from prospective prediction. End with a
specific recommendation and open questions for Alex's review before another
calibration-pair study.

The remaining 21 pairings, Pyomo.DoE and new hardware experiments remain deferred.
Do not resend the completed Tier 0–2 email. This investigation plan requests no
new email delivery or recurring report sends.

## Method references

These references support methods, not claims that any candidate fits ENV2.
Sources consulted October 24, 2026. Ground any new manuscript claims in the
user-authorized Lab Literature workflow when preparing publication material.

- [Residual analysis and input dependence](https://www.mathworks.com/help/ident/ug/what-is-residual-analysis.html).
- [Sinusoidal amplitude/phase estimation](https://www.mathworks.com/help/slcontrol/ug/frequencyresponseestimator.html).
- [Settling and frequency-response estimation](https://www.mathworks.com/help/slcontrol/ug/frequency-response-estimation-using-simulation-based-techniques.html).
- [Subspace and prediction-error state-space estimation](https://www.mathworks.com/help/ident/ug/state-space-model-estimation-algorithms.html).
- [Regularized impulse-response estimation](https://www.mathworks.com/help/ident/ref/impulseest.html).
- [Linear identification in a nonlinear setting](https://arxiv.org/abs/1804.09587).
