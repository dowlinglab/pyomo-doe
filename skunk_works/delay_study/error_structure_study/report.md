# TC-Lab remaining-error investigation

Prepared for Alex Dowling, October 27, 2026.

A centered 10% conductance band improves the joint fit substantially. It gives
little improvement in nominal predictions for omitted protocols. Quantization-aware
refitting retains input-linked and cross-channel dependence, and a checked
local Gaussian approximation fails away from its reference. The analyses
support conditional comparisons; calibrated independent innovations remain
unresolved. The [tablet PDF](report/main.pdf) contains the equations and figures.

| Question | Finding | Status |
| --- | --- | --- |
| Does a shared model with modest run variation help? | A shared nominal conductance with a 10% centered run band reduces joint-bin T1/T2 RMSE from 1.010/0.239 to 0.605/0.118 C. | Selected fits audited; nominal transfer improves much less. |
| Does radiation explain the rest? | The joint radiative fit gives 0.590/0.119 C and improves bin negative log likelihood by 3.810. | Profile audited; no calibrated radiation significance. |
| Were cold starts a bad assumption? | Quantized hidden-heater offsets remain below 0.515 C with either +/-2 or +/-5 C bounds, but select the opposite lag labels. | Startup profile checked; equilibrium remains unverified. |
| Are the hardware readings quantized? | All 21,614 temperatures lie on round(0.3223*n - 50, 2). | Hardware-data evidence; exact thresholds and analogue noise remain assumptions. |
| Can we assign the short and long lags physically? | Sensor/input factors exchange in the zero-loading limit; the fitted small-loading model retains a second likelihood region. | Lag labels weakly assigned; no calibrated exclusion of the second region. |
| Are the innovations adequate for inference? | Deterministic code diagnostics retain input, phase and cross-channel structure. | IID adequacy has not been established. |
| Does the nominal model transfer? | Omit one run: 1.750/0.754 C; omit a protocol: 2.137/0.878 C. | All twenty full numerical pipelines audited; local-normal uncertainty remains exploratory. |

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

## Nuisance profiles and startup

All 28 loading, step-run conductance, disturbance-length and radiation points
have independent numerical audits. Raising sensor loading from 0.005 to 0.1
or 0.3 costs twice nll 0.577 or 2.316. Raising step run 1's conductance ratio
from 0.9 to 0.925 costs 17.215. That bound constrains a consequential fitted
effect, without establishing a population distribution. The radiation optimum
is 0.11530; zero costs twice nll 7.619 after nuisance refitting. No calibrated
boundary test or emissivity estimate follows.

The startup profile fixes H1's offset in the first 300 s sine run, with all
other 23 hidden starts and permitted dynamics/noise coordinates refitted.
Five of six points are accepted and independently audited. Fixing this offset
to zero costs twice nll 6.447 and retains sensor/input times near
2.678/14.881 s. Both declared starts at -2 C exhausted their iteration budgets;
no profile value is assigned there. No accepted point beats the startup
reference. This adaptively selected coordinate and coarse grid do not supply
confidence endpoints or test an all-cold-start model.

Evidence: [28 nuisance profiles](results/remaining_profiles_figure_v3/figure_values.json),
[startup profile](results/startup_profile_summary/summary.json),
[profile interpretation](nuisance_profile_findings.md).

## Retrospective transfer

| Comparison group | T1 nominal RMSE, C | T2 nominal RMSE, C |
| --- | ---: | ---: |
| 10% band, omit one run | 1.750 | 0.754 |
| 10% band, omit a protocol | 2.137 | 0.878 |
| Shared conductance, omit a protocol | 2.178 | 0.896 |

| Range and group | Coverage, T1/T2 | Mean width, T1/T2, C |
| --- | ---: | ---: |
| 10% band, omit one run, noise | 46.01/47.56% | 2.093/0.291 |
| 10% band, omit one run, conditions + noise | 100.00/99.94% | 9.160/5.170 |
| 10% band, omit one run, parameters | 100.00/99.94% | 11.120/6.048 |
| 10% band, omit a protocol, noise | 31.84/36.48% | 2.006/0.302 |
| 10% band, omit a protocol, conditions + noise | 99.57/99.95% | 9.068/5.174 |
| 10% band, omit a protocol, parameters | 99.64/99.95% | 9.274/5.346 |
| 10% band, omit a protocol, reference stress | 100.00/99.95% | 10.380/5.737 |
| Shared conductance, omit a protocol, noise | 49.87/53.11% | 3.610/0.812 |
| Shared conductance, omit a protocol, conditions + noise | 94.01/99.81% | 6.771/3.968 |
| Shared conductance, omit a protocol, parameters | 94.65/100.00% | 7.115/4.286 |

| 64/256 comparison | Maximum endpoint change, T1/T2, C | Endpoints changed, T1/T2 |
| --- | ---: | ---: |
| 10% band, omit one run | 1.61/0.65 | 19.11/11.08% |
| 10% band, omit a protocol | 0.65/0.33 | 8.71/5.39% |
| Shared conductance, omit a protocol | 1.62/0.65 | 16.54/13.74% |

Each group scores the same twelve records once: 10,795 readings per channel.
All records informed structure selection and initial guesses; comparisons are
retrospective. No later test temperatures fit ambient, conductance or physical
parameters. The first reported pair initializes physical states.

For the band model, run means explain 85.7/73.3% of omitted-protocol SSE.
Removing them leaves RMSE 0.809/0.454 C, using test outcomes rather than a blind
correction. Nominal transfer improves little relative to shared conductance.
Noise-only ranges cover 31.84/36.48% of omitted-protocol observations.
Hard conditions (+/-2 C ambient and +/-10% conductance) raise coverage to
99.57/99.95%, with mean widths 9.068/5.174 C. These hard scenario envelopes
and marginal noise quantiles do not define calibrated total-95% predictions.

The parameters rows mix 256 box-truncated local normal draws of physical and
noise nuisance coordinates, fixing active bounds and one likelihood basin.
Their mean widths for omitted protocols become 9.274/5.346 C. The first
fast-sine-omission prototype changed 9.119/5.134 to 9.223/5.206 C; the complete
64/256 table shows larger endpoint changes in other folds. Numerical audits
verify the calculations, rather than the sampling law.

A declared likelihood-shape check examined 16 saved draws for each of three
training fits. For the run-omission fit with largest conditional log-thermal-time
SE, a representative draw has actual twice nll increase **201.731**, versus
local Taylor prediction **34.276**. Independent propagation and up to 1024-node
integration confirm the **167.454** discrepancy. The most extreme saved draw
still fails the declared integration gate; its precise deviance is withheld.
No draws were clipped or reweighted. A positive Hessian and more mixture draws
cannot establish calibration when the local shape approximation fails. The
parameter-inclusive ranges are exploratory sensitivities.

Changing the training set also changes its geometric nominal conductance.
Compare gain/thermal-time and conductance-ratio/thermal-time invariants before
interpreting raw gain shifts physically. The four primary-protocol stress
checks allow ratios [0.86894,1.13551] to the nine-run training reference,
yielding widths 10.380/5.737 C. Training constraints remain unchanged; this is
not a transported-cohort likelihood or a corrected coverage calibration.

Evidence: [complete twenty-fold summary and bindings](results/transfer_final_summary/summary.json),
[local-shape check](results/local_taylor_quality/summary.json),
[independent quadrature followup](results/local_taylor_quadrature_followup/summary.json),
[transfer methods and audit scope](transfer_methods.md).

## Questions for scientific review

One bounded conductance effect captures much of the between-run mismatch,
but transfers weakly across protocols. Radiation offers modest conditional
improvement. Small hidden starts change lag labels. Refitted residual checks
and a failed local-normal shape check limit inference despite precise local
standard errors. The existing data have not identified an environmental cause,
emissivity, an electrical power law or a unique physical assignment of both lags.

For a next analysis, compare a quantized multichannel disturbance model and
less symmetric physical balances using the same residual checks. Neither a
higher likelihood nor better one-step filtering alone establishes adequate
whitening. See the [completion evidence](completion_audit.md) for the scope
of each numerical and scientific check.

A later fume-hood experiment should record ambient temperature, absolute time,
preceding run/end and recovery, airflow and sash settings, firmware/device
identity, heater P settings and electrical power where available. A hood can
add forced or fluctuating convection. This goal performs no hardware or DoE
work and sends no new email.
