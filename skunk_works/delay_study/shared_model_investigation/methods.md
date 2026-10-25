# Models, inference and numerical conventions

This file records implementation choices for the shared-model investigation.
The [plan](../review/tier0_followup_investigation.md) owns scientific scope;
[PROJECT.md](PROJECT.md) and [run_manifest.json](run_manifest.json) own execution.
The completed residual_study and its delivery records are preserved.

## Shared data and nuisance treatment

Fit the twelve distinct ENV2 records jointly, with independent process resets
at run boundaries. Inputs are the recorded Q1 values, held on the left between
updates; Q2 is zero. Temperature and input-lag initial conditions match the
completed reference. First T1 is conditioned on and its residual is omitted.
Use each first T1 as the ambient reference, with an offset δ bounded by ±2 C.
These references are assumptions, not measured ambient temperatures.

The linear-ambient sensitivity uses an affine trajectory whose two endpoint
offsets both lie within ±2 C. This is an explicit computational convention;
Alex requested trying linear drift without specifying a physical slope bound.
The initial comparisons retain constant offsets; revised-mean comparisons use
the explicitly labeled affine endpoints. Mean propagation uses actual
irregular intervals, without resampling or changing command timestamps.

K absorbs a shared constant alpha calibration. No separate alpha multiplier is
fitted or added to K's prediction uncertainty. A command-dependent response map
changes the mean-response family and is a different hypothesis from a constant
alpha bias. Absolute physical conductance/capacity still requires calibration.

## Dynamic families

For a fixed delay order n=1,2,3, use

\[
G_n(s)=K/[(1+\tau s)(1+\theta s/n)^n].
\]

θ is total mean chain delay; every equal stage has time constant θ/n. Ambient
forcing enters the thermal equation through τ. For n=1, exchanging τ and θ
preserves the command response, but ambient forcing can change that equivalence.
Do not assign physical pole labels from the command response alone.

The flexible real-pole benchmark uses two to four dynamic states. Write z_0=u,
with successive input lags z_j, and

\[
\tau\dot T=-T+T_{amb}+\sum_j c_j z_j,\qquad \sum_j c_j=K.
\]

It has unequal positive time constants and a free numerator, without direct
feedthrough to measured temperature. Signed c_j can generate inverse response
and nonminimum-phase zeros. This is a stable real-pole benchmark, not unrestricted
LTI identification or a claim that its internal states are physical. We used
declared thermal/logarithmic multistarts, not a subspace estimator. The first
m−1 coefficients have technical bounds ±2 C/percent; the final coefficient
enforces their sum. Sensitivities widen the first bounds to ±4 and ±8.

The unequal positive three-pole model keeps only the final input path:

\[
G(s)=K/[(1+\tau s)(1+\theta_1s)(1+\theta_2s)].
\]

It relaxes the equal-stage restriction without numerator zeros. Interchanging
the two input lags preserves the response, so their labels are not physically
identified. It has four shared thermal parameters, like the transport extension

\[
G(s)=K e^{-ds}/[(1+\tau s)(1+\theta s)].
\]

d is an effective additional delay bounded between 0 and 15 s. The interval
before the run has zero command. Delay affects the command path, not ambient
forcing. Recorded command updates shifted by d are integrated as events; there
is no interpolation or rounding to sample delays. Acquisition timing and physical
transport are not separated by this parameter.

## Two bounded nonlinear checks

The reduced heat-loss diagnostic uses

\[
\theta\dot z=u(t-d)-z,\qquad
\tau\dot v=-v-\beta v^2+Kz,\qquad
T=T_{ref}+\delta+v,
\]

with z(0)=0 and v(0)=−δ. β is shared, bounded between 0 and 0.02 per C. Setting
β=0 recovers the transport model exactly. This is a phenomenological curvature
term, not an identified radiation coefficient. The location of the input lag
relative to the nonlinear balance is part of the model; nonlinear blocks do not
inherit the commutativity of linear lags.

The command-map diagnostic uses

\[
u_{eff}=u+c u(1-u/100),\qquad -0.8\le c\le0.8,
\]

followed by the positive transport model. It preserves the 0 and 100 percent
endpoints and is monotone on that range. K and c are shared across all records.
For an ideal 50+50 sin(ωt) command, the map adds c[12.5+12.5 cos(2ωt)]; for a
50 percent step it adds 25c. Thus it can test the observed step/sine gain
discrepancy and harmonics without adding another constant gain calibration.
Actual fitting uses the recorded held commands, not the ideal sine formula.
This is a static input nonlinearity followed by linear dynamics, a Hammerstein
structure. An estimated map does not establish a firmware or heater defect.

After the command-map/ambient result, repeat n=1/2/3 and the unequal positive
three-pole model with that same shared map and linear-ambient convention. This
closes the stage comparison under revised mean assumptions. Equal-stage cases
then have 31 nominal parameters; the unequal case has 32. Do not infer stage
preference solely from comparisons that used the earlier mean formulation.

The nonlinear checks are separately nested alternatives, not a search over
arbitrary nonlinear functions. Evaluate them before proposing a larger physical
model. No full two-temperature fit is authorized by this phase.

## Likelihood and comparison criteria

Use additive stationary OU disturbance plus independent measurement noise,
with separate process resets and one shared variance scale, correlation length
and measurement-variance fraction. Profile the common variance scale. The
Gaussian objective includes both the whitened residual quadratic and covariance
log determinant. Thermal/ambient/noise values are reestimated in the primary
fits. Equal-stage models have 18 nominal parameters; unequal-three-pole and
transport models have 19; free two/three/four-state models have 19/21/23;
the single nonlinear extensions have 20. Affine ambient adds twelve parameters.

Conditional sensitivities fix the OU shape to the previous n=1 fit, or use IID
errors. They separate mean weighting from covariance reestimation. Fixed-shape
likelihood criteria are not independently estimated model-selection scores.
Fitted nominal AICc counts the declared free parameters and profiled scale;
bounds, misspecification and weak identification qualify its interpretation.
Report raw temperature error, frequency mismatch and residual input association
alongside the likelihood. Improved likelihood alone does not establish adequacy.

## Frequency response and residual diagnostics

Fit sine/cosine coefficients at the known 30,120,300-second frequencies after
300,450,600 seconds. Compare trend nuisances with exponential nuisances at the
reference rates. Derive the input fundamental by exact integration of held
commands over the same window. Report phase differences circularly; display
frequency unwrapping independently for each measured/model curve. Only three
frequencies do not identify a unique phase branch or delay mechanism.

Working harmonic-regression uncertainty uses HAC with 30,60,120 sample lags,
approximately seconds here. It does not treat serial samples as independent
replicates, and does not calibrate coverage under the observed model discrepancy.
Higher input harmonics must be projected jointly over windows with partial
cycles; separate projections leak the fundamental into other frequencies.
Compare observed/model harmonics with deterministic rounding sensitivity and
repeat variability before attributing them to nonlinearity.

Residual energy decomposition is sequential: per-run mean, additional linear
trend, additional fundamental, remainder. The fractions depend on that order
and are descriptive, not causal attributions to ambient or measurement noise.
Phase folds use twelve bins with startup separated. No bin-count confidence
claim treats within-run samples as experimental repetitions.

Conditional OU forecasting uses measurements through each forecast origin, then
propagates the filtered disturbance without future temperatures. Compare against
free-run errors on the same targets. Since parameters/ambient were estimated
from the full records, this is retrospective within-record conditioning, not
independent forecast validation.
For each origin and nominal horizon, the target is the first observation at or
after origin time plus that horizon. Irregular observations can therefore give
a slightly longer actual horizon. Origins, target counts and free-run rows
are identical in each paired forecast comparison.

## Information, profiles and nonparametric response

Whiten scaled mean sensitivities, then eliminate ambient nuisance directions by
projection. Report singular values, ranks at multiple thresholds, weak directions
and conditional SDs. All ambient directions are allowed locally even at bounds,
so this calculation is not a regular constrained confidence analysis. Observed
projected curvature uses logit measurement fraction and two differencing steps.
Full rank alone does not establish useful estimability, unique pole labels or
calibrated uncertainty.

Profiles refit all remaining thermal, bounded ambient and noise parameters.
Retain multiple starts and recovery records. Nominal support uses 2ΔNLL=3.84146
with stated grid interpolation; it has no guaranteed 95 percent coverage under
misspecification/bounds. Additional-delay profiles include zero explicitly.
Independently reconstruct every saved profile point before reporting support.

The FIR benchmark represents impulse response as constant lag bins of 3 or 6 s,
with horizons of 600 or 900 s. It integrates held input exactly. DC gain remains
bounded 0.2–1.5; bin weights can be signed. Ambient dynamics and OU shape/scale
are fixed to the n=1 reference as an explicit diagnostic. Compare four declared
second-difference penalties, with the exact normalization saved in results.
Bin/horizon/penalty sensitivity is not a probability interval or sub-bin delay
estimate. The FIR fit is not a primary jointly estimated likelihood competitor.

## Numerical evidence and sources

Freeze committed Python/data and the pinned scientific runtime for measured jobs.
Keep raw CSVs and completed-study outputs unchanged. Verify selected means with
independent physical-state propagation and dense covariance likelihoods; audit
profile/FIR arithmetic separately. The quadratic solver uses exact linear and
quadratic split steps. Its selected trajectories require independent accurate
ODE and integration-step checks, with declared errors, before interpretation.

All model selection uses the same previously inspected ENV2 records. Keep any
later transfer assessment explicitly retrospective. The report does not establish
an untouched validation set or identify a unique physical mechanism.

Method sources (accessed during the October 24–25,2026 investigation):

- [MathWorks: static input maps with linear dynamic blocks](https://www.mathworks.com/help/ident/ref/idnlhw.html).
- [APMonitor: nonlinear TC-Lab heat-loss balance](https://apmonitor.com/pdc/index.php/Main/TCLabRadiative). This motivates curvature only; no published calibration values were imported.
- [Frequency-response estimation](https://www.mathworks.com/help/slcontrol/ug/frequencyresponseestimator.html), [residual analysis](https://www.mathworks.com/help/ident/ug/what-is-residual-analysis.html), and [regularized impulse response](https://www.mathworks.com/help/ident/ref/impulseest.html).
- Previously grounded covariance, HAC and profile sources are recorded in [the completed literature review](../residual_study/literature_review.md). The equations and additional diagnostic bounds above are this investigation's choices.
