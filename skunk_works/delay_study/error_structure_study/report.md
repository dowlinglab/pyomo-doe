# TC-Lab remaining-error investigation

Working draft for Alex Dowling, October 26, 2026. The goal remains active.
The fifty-draw refitted recovery study and initial/bin comparisons are audited.
Remaining profiles and sixteen parameter-inclusive transfer components are
unfinished; two transfer information audits stopped before covariance acceptance. The
[completion register](completion_audit.md) lists the remaining gates. This
Markdown companion will be checked with the final tablet PDF after those gates.

| Question | Finding so far | Status |
| --- | --- | --- |
| Does a shared model with modest run variation help? | A shared nominal conductance with a 10% centered run band reduces joint-bin T1/T2 RMSE from 1.010/0.239 to 0.605/0.118 C. | Selected fits audited; nominal transfer improves much less. |
| Does radiation explain the rest? | The joint radiative fit gives 0.590/0.119 C and improves bin negative log likelihood by 3.810. | Conditional improvement; radiation profile pending. |
| Were cold starts a bad assumption? | Quantized hidden-heater offsets remain below 0.515 C with either +/-2 or +/-5 C bounds, but select the opposite lag labels. | Fits audited; consequential startup profile pending. |
| Are the hardware readings quantized? | All 21,614 temperatures lie on round(0.3223*n - 50, 2). | Hardware-data evidence; exact thresholds and analogue noise remain assumptions. |
| Can we assign the short and long lags physically? | Sensor/input factors exchange in the zero-loading limit; the fitted small-loading model retains a second likelihood region. | Lag labels weakly assigned; no calibrated exclusion of the second region. |
| Are the innovations adequate for inference? | Deterministic code diagnostics retain input, phase and cross-channel structure. | IID adequacy has not been established. |
| Does the nominal model transfer? | Omit one run: 1.750/0.754 C; omit a protocol: 2.131/0.875 C. | All twenty canonical folds audited; parameter propagation unfinished. |

## Model and comparisons

We accept the same-device ENV2 provenance. There are nine sine records
(three repeats at each of 30, 120 and 300 s) and three step records. Commands
are held from each timestamp to the next update. Missing inter-run dates do
not establish recovery times or equilibrium.

Both outputs enter a coupled heater/sensor model with an additional input lag
and the existing empirical command curve. One ambient offset per experiment
is bounded by +/-2 C. One physical coefficient varies:

\[
U_{a,j}=U_{a,0}e^{\delta_j},\qquad
\sum_j\delta_j=0,\qquad
\log(1-b)\leq\delta_j\leq\log(1+b).
\]

The geometric nominal is regressed jointly. This preserves coupling between
experiments without estimating a population variability distribution. Gain,
thermal time and conductance ratios change together when Ua changes; they
are not independently fitted run effects. Constant shared heater calibration
is absorbed into gain. Absolute conductance and heat capacity need an
independent power/capacity scale.

| Joint quantized fit | T1 RMSE, C | T2 RMSE, C | Negative log likelihood |
| --- | ---: | ---: | ---: |
| Shared conductance | 1.010 | 0.239 | 6359.9182 |
| 5% run band | 0.637 | 0.153 | 6279.4250 |
| 10% run band | 0.605 | 0.118 | 6238.2987 |
| 20% run band | 0.606 | 0.118 | 6238.2167 |
| Radiation, 10% convective band | 0.590 | 0.119 | 6234.4890 |

The 10% fit has nominal gain 0.8732 C/%, thermal time 174.59 s, sensor time
14.79 s and input lag 2.73 s. Its sensor-loading estimate reaches the lower
bound. Widening the conductance band adds little training-likelihood benefit
beyond 10%. Protocol-aligned fitted deviations could reflect several model
discrepancies; they are not airflow measurements. Training likelihood alone
does not choose the band width.

The earlier T1-only positive three-pole benchmark, with command curvature and
affine ambient, reached 0.374 C RMSE. Its different output set and ambient
formulation prevent a matched likelihood or AIC comparison with these joint
fits. It remains historical prediction context, without a unique physical
assignment of its three poles. See the
[completed single-output study](../shared_model_investigation/report.md).

The old model's roughly 1 C T1 error was total RMSE. Orthogonal projections
assign 74.6% of its squared error to run offsets and another 19.3% to trends;
the additional fundamental sine term accounts for only 0.64%. Describing the
whole error as a one-degree sine-amplitude mismatch was incorrect.

For an illustrative 0.0012 m2 surface at 70.54 C, surroundings at 23 C and
emissivity scenarios 0.05--0.9, radiation is 0.021--0.383 W. Against a linear
radiative loss matched at 50 C, the nonlinear remainder is 0.0021--0.0370 W.
The fitted linear conductance can already absorb the linear part. Metal does
not specify emissivity; finish, effective area, view factors and actual heater
surface temperature remain unknown. Radiation fits refit convection and gain
to avoid double counting.

Evidence: [initial decomposition](initial_findings.md),
[joint selected-point audits](PROJECT.md), [radiation calculation and sources](radiation_sources.md),
[quantized likelihood](quantized_methods.md).

## Residual structure and uncertainty

The likelihood integrates a correlated latent OU disturbance plus independent
analogue noise over each reported code's interval. It does not multiply
marginal bin probabilities or import emulator noise. Sequential Gaussian GLS
converges to the matched joint fit; the discrete observation law requires its
own refitting and checks.

Randomized predictive residuals looked comparable with fixed-model references.
Removing the added PIT randomization exposes remaining structure: T1 input
correlation is 0.04429 versus a reference 97.5% value 0.02261; maximum
cross-channel correlation is 0.04895 versus 0.02540. Trend and harmonic
projection leaves cross correlation 0.04719 versus 0.02603. These are
descriptive ranks from fitted, fixed-model references. Refitted references
are now complete; all four selected hardware diagnostics still exceed every
refitted reference. The multiple-diagnostic and structure-selection limitations remain.

Using earlier reported codes to update the disturbance state gives whole-data
one-step RMSE 0.103/0.087 C, and at least 60 s ahead 0.255/0.110 C. These
parameters were fitted to all twelve records. Feedback-assisted prediction
does not establish free-simulation accuracy or independent innovations.

Correlation changes parameter uncertainty by direction. Holding the same
analogue marginal variance, it increases local gain and thermal-time standard
errors by factors 4.55 and 5.83, while reducing the short-lag standard error
to 0.645 of its IID value. These local calculations condition on one basin
and its active constraint face. The fifty-draw refitted recovery study and all 100 free/true-gain point audits
are complete. The nominal gain likelihood-ratio region covers 49/50 generated
collections. The exact-binomial 95% Monte Carlo interval is 89.35--99.95%;
this checks the fitted model and algorithm rather than real-device adequacy.

Input and sensor lags nearly exchange at weak sensor loading. The audited
coarse profile rises strongly between the two regions, then falls again near
15 s. A fixed input lag of 14.79006 s fits sensor time 2.7314 s and raises
twice negative log likelihood by 4.78730. It neither improves the reference
nor locates the free secondary minimum. Profiles and initialization
sensitivities must accompany physical lag labels and local covariance.

Fifteen of the fifty fits reverse the sensor/input lag labels. Their unordered
fast and slow time scales have 2.5--97.5% repeated-fit ranges of
2.547--2.898 s and 14.419--15.576 s. These are estimator quantiles under
the generating model, not physical confidence intervals. Gain's empirical
standard deviation is 0.898 times its local standard error; log thermal
time spreads 4.780 times farther than its local approximation. A small
selected-basin covariance misses the lag-label ambiguity.

The hidden-heater fits bounded at +/-2 and +/-5 C coincide, with largest
offset 0.514 C, sensor time 2.667 s and input time 14.898 s. They give
reported-code RMSE 0.682/0.109 C, versus 0.605/0.114 C for the baseline.
A better likelihood did not improve T1 free prediction. Free-initial
nearest/floor fits give practically identical expected reported codes
(maximum difference below 0.00030 C), despite different analogue-reference
RMSE. Exact half-grid translation preserves expected codes within
3.1e-12 C. Bin phase alone is not dynamic information under that reference
freedom. Fixed starts and absolute-temperature radiation can break it.

Evidence and limits: [interim uncertainty analysis](uncertainty_interim.md),
[lag argument](lag_identifiability.md), [sequential method](sequential_methods.md),
[uncertainty protocol](uncertainty_methods.md).

## Retrospective transfer

| Comparison group | T1 nominal RMSE, C | T2 nominal RMSE, C |
| --- | ---: | ---: |
| 10% band, omit one run | 1.750 | 0.754 |
| 10% band, omit a protocol | 2.131 | 0.875 |
| Shared conductance, omit a protocol | 2.171 | 0.894 |

Each group scores the same twelve records once: 10,795 readings per channel.
Groups reuse observations; all records informed model selection. No later
test temperatures fit ambient, conductance or physical parameters. The first
temperature pair initializes the physical states.

The large all-data fitting improvement produces little nominal omitted-protocol
gain. For the band model, run mean errors explain 85.7/73.2% of omitted-protocol
SSE. Removing those means leaves RMSE 0.806/0.453 C, but uses test outcomes
and cannot serve as a blind correction.

Band-model omitted-protocol noise-only marginal ranges cover 32.07/36.66% of
observations. Allowing +/-2 C ambient and +/-10% conductance raises coverage
to 99.57/99.95%, with mean widths 9.072/5.173 C. These are hard scenario
envelopes combined with noise quantiles. They do not establish calibrated
total-95% predictions or a distribution over run conditions.

The first complete local-parameter propagation, omitting the three fast sine
runs, changes mean scenario-plus-noise widths from 9.119/5.134 to
9.223/5.206 C. Its local normal fixes the active loading bound and one basin.
The 64/256 shared-prefix comparison changes some endpoints by one code.
Four full components are audited; sixteen remain, including two failed
information gates under investigation. Changing the training set also changes the nominal
conductance reference; compare invariant combinations such as gain/thermal
time before interpreting raw gain shifts physically.

Evidence: [all twenty canonical folds](results/transfer_all_canonical_summary/summary.json),
[first parameter propagation](results/transfer_first_protocol_parameter_summary/summary.json),
[transfer definitions and audit scope](transfer_methods.md).

## Questions for scientific review

Refitting preserves the residual inadequacy finding. The remaining analyses
must establish how consequential startup assumptions and incomplete transfer
uncertainty affect the conclusions. The
existing data have not identified an environmental cause, emissivity, an
electrical power law or a unique physical assignment of both lags.

A later fume-hood experiment should record ambient temperature, absolute time,
preceding run/end and recovery, airflow and sash settings, firmware/device
identity, heater P settings and electrical power where available. A hood can
add forced or fluctuating convection. This goal performs no hardware or DoE
work and sends no new email.
