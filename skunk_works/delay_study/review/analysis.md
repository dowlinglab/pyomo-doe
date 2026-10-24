# Discussion brief for the TC-Lab review

Prepared October 24, 2026, after transcribing [your comments](comments.md).
This is a response and discussion brief, not a revision of the manuscript.
Observed repository facts, new derivations and proposed investigations are
distinguished below. No new parameter fitting or laboratory experiment was run.

## What I would put first in our conversation

The report answers a narrower model-identifiability question than the wording
“original 4-state model” in Q1. It proves an ambiguity for a **two-state,
four-parameter, single-output model**. The repository also contains a **four-state,
five-parameter, two-channel model**. The latter needs its own analysis. A new
common/differential-mode derivation below shows that its ideal symmetric version
can generically identify all five parameters when both sensors are observed and
heater conversion is known. Practical estimability on these data remains open.

For the reduced single-output exercise, n = 1 remains a defensible teaching
baseline. The next evidence to obtain concerns the thermal mean, measurement
process and data provenance. The present analysis has not established a correct
error model, and a larger delay order should not be selected to absorb ambient
drift, cross-channel heat transfer or device differences.

## Q1. What is identifiable, and which original model?

| Model | Dynamic states | Physical/transfer unknowns | Observations considered here | Conclusion |
|---|---:|---:|---|---|
| Original single-channel thermal model | 2 | 4 physical | First sensor | Exact continuous physical ambiguity |
| Reduced n = 1 chain | 2 | 3 transfer | First sensor | Two interchangeable pole labels |
| Reduced n = 2 chain | 3 | 3 transfer | First sensor | Distinct repeated pole generically labels delay; coincidence is nonregular |
| Full symmetric two-channel thermal model | 4 | 5 physical | Both sensors | Generic recovery possible under the assumptions below; empirical precision untested |

### The result already established in the report

With known heater conversion c = αP, zero initial deviations and only the first
sensor output, the two-state model has

\[
G(s)=\frac{K}{1+as+bs^2},\quad
K=\frac{c}{U_a},\quad
a=\frac{C_H}{U_a}+\frac{C_S}{U_s}+\frac{C_S}{U_a},\quad
b=\frac{C_HC_S}{U_aU_s}.
\]

Here U_s denotes heater-to-sensor coupling (U_b in the report). Temperature data
identify three transfer coefficients, while four physical parameters are sought.
For any admissible r = C_S/U_s between the two pole times,

\[
U_a=c/K,\quad C_H=bU_a/r,\quad
C_S=U_a(a-r-b/r),\quad U_s=C_S/r
\]

gives the same response. More varied inputs cannot distinguish these physical
members through the same observation equation and initial-state specification.
This is an exact ambiguity, beyond a badly conditioned fit.

For a concrete teaching example, the saved joint n = 1 fit has K ≈ 0.77443,
a ≈ 167.2515 s and b ≈ 2494.7349 s². Two equivalent physical members are:

| Chosen r (s) | U_a (W/K) | U_s (W/K) | C_H (J/K) | C_S (J/K) |
|---|---:|---:|---:|---:|
| 30 | 0.04132 | 0.07451 | 3.4361 | 2.2352 |
| 60 | 0.04132 | 0.04523 | 1.7181 | 2.7136 |

This answers C18: fit identifiable transfer coefficients, then choose r to
reconstruct a compatible physical member. The measured trace does not choose r.
If your “c” in that comment means heater conversion, c also needs independent
calibration; it is not an extra identifiable temperature-response coefficient.

The n = 1 transfer denominator is (1 + τs)(1 + θs). Exchanging τ and θ leaves
it unchanged. A short-delay convention labels the poles without demonstrating
which is a physical transport delay.

A profile likelihood fixes one parameter at a succession of values and refits
the remaining parameters at each value. A physical equivalence family can give
a flat profile, while exchanged pole labels can give two separate equal minima.
These are different explanations for an uncertain parameter.

### The genuine four-state model is a different question

`notebooks/parmest_four_states.ipynb` fits five parameters and both sensor
channels. Its equations include interheater heat exchange. The current Python
implementation also supports this model. The notebook narrative and Python
implementation use U_b/U_c for different coupling roles, so I use U_s for
heater–sensor exchange and U_x for interheater exchange here.

For identical channels, known common c, common ambient, zero initial deviations
and U_x > 0, form sum and difference sensor outputs. They obey two two-state
transfers with effective ambient-loss conductances

\[
L_+=U_a,\qquad L_-=U_a+2U_x,
\]

and coefficients

\[
K_\pm=c/L_\pm,\qquad
a_\pm=r+(C_H+C_S)/L_\pm,\qquad
b_\pm=C_Hr/L_\pm.
\]

The two mode responses reveal additional information. Recover

\[
r=\frac{L_-a_- - L_+a_+}{L_- - L_+},\quad
C_H=\frac{L_+b_+}{r},\quad
C_S=L_+(a_+-r)-C_H,\quad U_s=C_S/r,
\]

and U_x = (L_- − L_+)/2. This is a generic structural inversion, conditional on
the assumptions above and informative observations of both modes. With only
heater 1 driven, both sum and difference inputs are still excited. As coupling
vanishes, the inversion degenerates; weak coupling can make it numerically fragile.

The new [scope check](four_state_scope_check.json) reconstructs a synthetic
positive parameter point, matches the four-state simulation to the two derived
mode simulations within 1.3e−12 °C, and gives local sensitivity rank five.
Its smallest singular value is much smaller than the largest: full rank is
not a demonstration of useful precision. This check does not fit or validate
the four-state model on the experimental records.

There is useful second-channel signal already in the CSVs: T2 rises about
12.89 °C above its first reading in the calibration sine record and 11.92 °C in
the step record, while Q2 is zero. T2 is the second channel's sensor, not a direct
measurement of the first heater state. It should be connected through the
four-state observation model before using it to resolve physical ambiguity.

**Discussion decision:** does Q1 target the reduced single-channel model, the
full symmetric four-state model, or both? A teaching presentation should state
the modeled inputs, outputs, states and unknown parameters before claiming
nonidentifiability.

## Q2. Appropriate delay formulation and the next input

Your assessment is consistent with the saved cold-start comparison. Joint
calibration RMSE is 0.4115 °C for the original single-output/n = 1 response and
0.4630 °C for n = 2. Pooled held-out RMSE is 3.3588 versus 3.3608 °C. A 2.8%
step-only improvement for n = 2 does not establish a general predictive advantage.
The n = 1 reduction removes a redundant physical direction while preserving
the original single-output predictions.

For a teaching baseline, assume zero command prehistory and near-ambient thermal
startup, as you suggest in C05–C06. The historical cooldown code supports that
choice. Keep an initial-state sensitivity check because a scheduled wait is not
a measurement of thermal equilibrium. “Input-equilibrated” in the report means
z_i(0) = u(0), not “the device cooled to ambient.” When u(0) = 50%, those two
conditions describe different prehistories.

C07 is also well motivated: the acquisition code sets a command at each clock
iteration, which supports a left zero-order hold between updates. Exact command
and sensor timestamps still matter. The report's linear-hold results remain an
audited comparison, but a revised primary analysis should justify its hold from
the acquisition loop rather than default to interpolation.

Your random-step proposal (C13) is a useful experimental candidate: vary levels
and dwell times, include zero-command periods, and record the cooldown. Compare
that input against a repeatable reference and the optimized input under the
same duration, dose and temperature constraints. Repeated waveforms can help
separate repeatable mean-response errors from stochastic variation. The exact
amplitudes and dwell times should be chosen after clarifying the error model
and whether both sensor channels will be fitted.

A sequential Pyomo.DoE loop (C02) would estimate → diagnose → design → collect
→ validate → update. It can target transfer parameters or model separation.
It cannot resolve an exact equivalence under an unchanged observation model.
Initially choose the next complete waveform between runs. Within-run feedback
is a separate extension because it changes the relationship between input and
disturbance; the open-loop design assumptions need reconsideration.

## Q3. What error model is appropriate?

**Unresolved.** The saved diagnostics challenge both independent errors and the
one-component OU covariance. For joint cold n = 1, OU-filtered innovation ACF1
is about −0.533 in the sine record and −0.492 in the step record. Across the
four displayed formulations the effect persists. The covariance fit therefore
has not delivered white innovations. Its assumed Gaussian marginals and
stationarity have not been established either.

Three distinctions should guide the next analysis:

1. A correlated simulation residual can contain omitted deterministic thermal
   dynamics or input-history error. Correlation alone does not identify a
   stochastic disturbance model.
2. An appropriate disturbance model should yield useful one-step innovations;
   deterministic mean-model simulation residuals can remain correlated.
3. A better in-sample covariance likelihood is insufficient evidence for
   calibration or prediction performance on a new run.

The [MathWorks residual-analysis documentation](https://www.mathworks.com/help/ident/ug/what-is-residual-analysis.html)
separates innovation whiteness from independence of residuals and past inputs,
and explains why output-error dynamics estimation can remain useful with
colored residuals. Its feedback caveat also matters for the proposed adaptive
experiment. This is methodological guidance, not validation of our data.

### An ordered investigation to discuss

First account for measured or explainable mean changes: command hold, startup,
ambient, channel coupling and device grouping. Then compare a small set of
working residual models on the same thermal mean families:

| Candidate | What it tests | Evidence required |
|---|---|---|
| IID reference | Independent residual approximation | Mean, ACF and variance diagnostics |
| OU | One smooth correlated component | Existing fit fails innovation-whiteness diagnostics |
| OU plus independent observation noise | Slow variation plus a fast component | Fit and held-out innovation checks; not yet performed |
| Small ARMA/state-space noise model | More flexible short-lag structure | Parsimony, innovation/input checks and predictive validation |
| Quantized measurement likelihood | Recorded temperature resolution | Verify sensor conversion/rounding and inspect likelihood sensitivity |

The CSVs have distinct reported T1 levels separated by roughly 0.32/0.33 °C.
This is direct evidence of coarse recorded resolution, not a measured standard
deviation or proof of independent uniform quantization error. It should be
checked before treating 0.25 °C Gaussian noise as an instrument calibration.

An OU-plus-observation-noise candidate is motivated by Appendix B.6. For an IID
component v_t, applying v_t − φv_(t−1) gives lag-one correlation
−φ/(1 + φ²), near −1/2 when φ is close to one. The observed negative spike is
compatible with that mechanism, but also with a misspecified slow mean. It
does not select the mixture model by itself.

For each candidate, inspect residual/innovation time plots, ACF and PACF,
variance versus temperature/time, input–innovation cross-correlation, marginal
distributions and recording resolution. Use chronological held-out segments
and separate runs; a random split of correlated samples leaks neighboring
information. Fitted-model portmanteau tests need appropriate calibration, and
passing a test would not establish a physical noise mechanism.

### How to teach HAC (C09, C10 and C19)

Ordinary local least squares uses curvature to translate observation errors into
parameter errors. With dependent residuals, neighboring score contributions
q_t = J_tᵀe_t can move together. HAC estimates their long-run covariance,
including weighted lagged products, then places it between two inverse-curvature
factors: Cov(p̂) ≈ B S B. Bartlett weights taper the retained lag products.

This adjusts an uncertainty estimate; it does not whiten the data, change the
mean fit, identify a disturbance mechanism or repair a structural null direction.
It still needs a suitable mean/exogeneity condition, enough data, finite moments
and controlled dependence. The synthetic coverage experiment supports using it
as a sensitivity comparison here, rather than assuming it gives calibrated
real-data confidence intervals. The block-length width factors 6.66 and 6.23
measure sensitivity to retained dependence, not a universal correction factor.

## Q4. What can Git establish about provenance?

The [file-by-file provenance receipt](data_provenance.json) records current file
hashes and local Git history. All current CSV versions were last changed by
Daniel Laky, but their repository history spans several dates:

| Records | Repository event | Author |
|---|---|---|
| Original step and an older sine file | Added May 1, 2024, `d5fd738` | Alex Dowling |
| `validation_experiment_sin_5_50.csv` | Added August 23, 2024, `f69322b` | Daniel Laky |
| Twelve `validation_experiment_env_2_*` files | Added together August 27, 2024, `b114b13` | Daniel Laky |
| Current calibration sine and step | Sine added/step replaced September 10, 2024, `e643686` | Daniel Laky |

The September calibration records are byte-identical to the earlier env_2
`sin_5_50_run_1` and `step_50_run_1` files. Their September commit is therefore
not evidence of September acquisition or a new independent experiment. The
study already excluded those two duplicate validation files.

At `b114b13`, the saved acquisition notebook imports TCLab and uses
`with TCLab() as lab`, P1 = 200, Historian CSV output, repeated trials and
25-minute cooldown waits. Its saved loop covers only some of the filename
patterns. It cannot establish the complete run sequence or prove that every
CSV followed that exact code version.

The CSVs contain Time, T1, T2, Q1 and Q2. They do not contain device serials,
wall-clock acquisition timestamps, an independent ambient trace or a heater
calibration identifier. Initial T1 ranges from 19.94 to 24.77 °C across the
supplied files; those readings are not independent measurements of ambient.
Your different-device/different-ambient explanation (C14) remains plausible,
with more direct evidence needed to distinguish it from thermal-model error.

**Discussion decision:** recover device/channel IDs, acquisition dates/order,
cooldown records and calibration ownership from the researcher who collected
the records. Git author identity alone does not establish experiment operator
or device identity. No messages have been sent to anyone.

## Q5. Infer ambient and α without creating hidden ambiguity

Your note that α was calibrated on one device and reused for others is new
human-provided context. The existing calculations fixed α = 0.00016 and P = 200;
they did not establish cross-device calibration uncertainty.

A proposed multi-run measurement model is

\[
y_{d,r}(t)=T_{d,r}(t)+b_d+v_{d,r}(t),\qquad
T_{amb,d,r}(t)=A_{d,r}+g_{d,r}t
\]

as a first ambient-drift sensitivity, with a measured ambient trace preferred
over a free drift whenever available. Here d identifies a physical device,
r identifies a run and b_d is an optional sensor-offset term. Use device-level
α_d only when device IDs are available; specify calibration observations or
an explicit prior from the external calibration. Estimate initial temperatures
as latent quantities if their measurement error matters, instead of treating
the first noisy sensor reading as exact ambient and exact initial state.

This is a proposed model family, not an instruction to fit all nuisance terms
simultaneously. Ambient drift, sensor offset and a flexible residual process
can explain the same low-frequency variation. Independent ambient or calibration
measurements are needed to distinguish those mechanisms reliably.

Temperature-versus-percent-command data alone preserve the scale transformation

\[
(\alpha,U_a,C_p)\mapsto
(\lambda\alpha,\lambda U_a,\lambda C_p).
\]

Thus an uncalibrated α is confounded with absolute conductance and capacity.
The corresponding four-state scaling includes both couplings and capacities.
Transfer gain and time constants can remain estimable. An informative prior
can select a scale, but that is calibration information rather than identification
from these thermal traces.

For the reduced model, log U_a = log α + log P − log K and
log C_p = log α + log P + log τ − log K. If independent calibration uncertainty
in log(αP) has variance v_c, it adds v_c vvᵀ to the transformed physical
covariance, with v = (1,1,0). This creates shared conductance/capacity uncertainty
while leaving the delay coordinate unchanged. That statement is conditional
on the independence and model assumptions; shared calibration data require the
corresponding cross-covariances instead.

For constant u, unknown ambient and gain can also enter through the same
combination A + Ku. Varying input and an independent ambient measurement address
different parts of this ambiguity. Cooler runs and repeated waveforms would
help assess ambient drift without forcing the delay order to absorb it.

## Responses to the remaining clarification comments

| Comments | Explanation or proposed revision |
|---|---|
| C01, C18, C22 | Teach the two-state equivalence with a pair of numerical physical examples; distinguish four parameters from four states and distinguish a continuous family from the discrete pole exchange. |
| C04 | Manchester lifts quadratic information in u to a matrix U = uuᵀ, drops a rank-one constraint and solves a convex relaxation. It can provide an upper-bound benchmark and candidate starts. Our dose, slew, one-sided amplitude, output constraints and adaptive protocol require their own formulation; its approximation guarantee does not automatically transfer. |
| C08 | Ambient/history choices can shift fitted gain and delay and can invalidate transfer to another run. An extra per-record fitted offset improves calibration without establishing its value before a new run is observed. |
| C03, C11, C21, C23 | Diagnose a thermal mean plus an explicit noise model, and judge held-out predictions and innovations; no currently validated “correct” error model has been selected. |
| C12 | The main ensemble study simulated 300 s after the 900 s active input, ramping the final command to zero, to check delayed peaks. The later native Pyomo.DoE teaching design checked only its 900 s window. Those checks should be labeled separately. |
| C15 | “Filtered” is the normalized OU innovation transform: e₁ unchanged, then (eᵢ − φᵢeᵢ₋₁)/sqrt(1 − φᵢ²). It is a whitening attempt using the assumed covariance, not arbitrary smoothing of the observations. |
| C16 | Command reconstruction specifies the input between timestamps: straight ramps, previous-sample hold or next-sample hold. The propagator is a separate choice of how to solve the ODE under that input. |
| C17 | D² = (μᵢ − μⱼ)ᵀR⁻¹(μᵢ − μⱼ), the covariance-weighted squared difference of predicted means. It is defined in Section 3.8, page 8, but should be repeated briefly near Figure 19 and distinguished from D-optimality. Refitting competitors reduces this separation. |
| C20 | Introduce α through external calibration observations/uncertainty; do not present a jointly fitted absolute thermal scale as uniquely data-identified. |

Manchester's lifting and constraint extensions were checked in the
[author preprint](https://arxiv.org/pdf/1009.5614), physical PDF pages 2–6.
This brief adds no claim that our design algorithm implements that relaxation.

## Suggested conversation order and decisions

1. Resolve Q1's model scope and whether the next comparison should include both
   recorded sensor outputs.
2. Establish what is known about device IDs, calibration and acquisition order.
3. Choose the next mean/noise diagnostic comparison, with explicit ambient and
   command-hold assumptions.
4. Agree on a teaching explanation for the established SISO equivalence.
5. Decide whether the first new experiment should be a repeatable multilevel
   step/cooldown protocol, followed by an adaptive design comparison.

The error-model comparisons, four-state data fits and proposed experiments
remain work to scope after this discussion. No manuscript, released result,
automation or email was changed during this review preparation.
