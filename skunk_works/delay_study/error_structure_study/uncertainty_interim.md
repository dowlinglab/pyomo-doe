# Interim uncertainty and whitening findings

This file preserves intermediate analyses and their original partial scopes.
The completed findings and current limitations are in [report.md](report.md)
and [the completion register](completion_audit.md); later results supersede
the pending statements and original-fit comparisons below.
The selected mean/bin fits and information results below have terminal
independent audits. The gain profile and all seven point audits are complete;
remaining nuisance profiles, complete parameter-inclusive transfer and several
initialization sensitivities remain unfinished. The fifty-draw recovery and
refitted-diagnostic study is now complete and independently audited below.

## The run band and loading boundary

The joint bin-likelihood fits give T1/T2 RMSE 0.637/0.153 C for a 5% conductance
band, 0.605/0.118 C for 10%, and 0.606/0.118 C for 20%. Negative log likelihood
is 6279.4250, 6238.2987 and 6238.2167 respectively. Four 5% deviations reach
their bounds. These are source-bound same-data comparisons, not validation
of a population variability law. See joint_conductance_b05_audit and the earlier
10/20% selected-point audits.

Lowering the positive loading bound Us/Ua from 0.005 to 0.0001 sends the
estimate to the new lower bound. Negative log likelihood improves only 0.01291;
gain changes from 0.873222 to 0.873202 C/%, and the short lag from 2.73048 to
2.73038 s. The weak loading estimate does not establish zero sensor capacity.
The structural separation argument and its zero-loading exception are in
lag_identifiability.md; practical profiles still matter.

## Correlation changes uncertainty differently in different directions

The gain profile refits all other permitted mean/noise nuisances. Fixing K
1.5% below/above 0.873222 C/% raises twice nll by 18.674/18.022. All seven
sampled points passed dimensional, rectangle, quadrature/support and stationarity
audits; none improves the reference. This is conditional precision under the
chosen model. The coarse grid does not resolve interval endpoints, and the
nominal scalar cutoff still needs the fifty-draw recovery check. Evidence:
profile_b10_gain and profile_b10_gain_audit_v2. The first collection's two
missing output folders caused saved-check failures, retained separately;
those calculations were repeated rather than declared successful from memory.

The corrected information matrix passed its independent dimensional,
dense-covariance, basis/eigenpair and scalar-curvature audit. The old finite-
difference information matrix failed the weak-loading gate and is retained
only as a numerical diagnostic. The replacement uses checked matrix-exponential
Frechet sensitivities; its loading curvature is 0.078773. Independent forward
likelihood differences give 0.078761 and 0.078592 at two declared steps.

The following are local standard errors in optimization coordinates. They
condition on the selected basin and fix the currently active loading and run-9
conductance bounds. All other permitted mean nuisances enter the comparison.
They are not calibrated confidence intervals or substitutes for profiles.

| Coordinate | IID, known analogue marginal variance | Correlated, known analogue covariance | Joint bin curvature, noise nuisances free |
| --- | ---: | ---: | ---: |
| log K | 0.0007263 | 0.0033010 | 0.0033620 |
| log tA | 0.0005152 | 0.0030056 | 0.0030888 |
| log rS | 0.0105674 | 0.0107208 | 0.0113612 |
| log short lag | 0.0479381 | 0.0309354 | 0.0324303 |
| Command curvature | 0.0022933 | 0.0108948 | 0.0112435 |

Holding the same analogue marginal variance, correlation increases the local
gain SE 4.55-fold and thermal-time SE 5.83-fold. It reduces the short-lag SE to
0.645 of the IID value. Correlation changes frequency-specific information;
it does not uniformly inflate every parameter error. The bin column also
estimates the covariance nuisances, so differences between it and the known-
covariance columns do not isolate quantization alone.

The active-face weak direction is mainly a latent-SD/correlation-length
tradeoff. T1 log SD and log length have local correlation 0.9845. The local SE
of log(2*S/ell) is 0.0472, versus 0.1331 for log latent SD and 0.2692 for log
length. Avoid treating the tiny boundary-loading eigenvalue as a regular Wald
variance. Evidence: joint_b10_information_v2 and its independent audit.

Conditioning the residual process on the initial reported code also passed its
selected-point audit. It changes T1 latent SD/length from 0.5362 C/658.3 s to
0.8763 C/1761.9 s, while mean parameters barely change. The diffusion rate
2*S/ell remains almost unchanged. This supports the variance/length ambiguity;
it does not select the initial-state convention or justify transplanting an
unconditional stationary sampler to the conditional fit.

Letting each cold heater/sensor pair vary within its first reported bin also
passed the full selected-point audit. Gain changes by -0.403% to 0.869704 C/%,
the short lag changes to 2.72054 s, and nll decreases by 28.7450 with 24 extra
bounded initial coordinates. Several hit bin edges. This is a fit sensitivity;
T1 free-simulation RMSE increases from 0.605 to 0.655 C, while T2 decreases
from 0.118 to 0.113 C. The correlated-bin objective and unweighted
free-simulation error assess different features. There is no established
whitening or transfer improvement, and these bin
bounds do not integrate a posterior for the analogue initial temperature.

## Added PIT randomness can obscure a diagnostic

Fifty fixed-model simulations and all prefix Gaussian-CDF audits completed
without failure for deterministic conditional expected Gaussian-PIT scores.
These scores average over PIT jitter within each observed code. They are
martingale diagnostics under a correct known predictive law, but are discrete,
generally non-Gaussian and not unit variance. Their data SDs are 0.5795 and
0.4981. The same deterministic calculation is applied to every reference.

| Statistic | Data | Fixed-model reference 97.5% quantile | Upper-tail rank |
| --- | ---: | ---: | ---: |
| T1 maximum score ACF, lags 1–60 | 0.02461 | 0.03110 | 0.471 |
| T2 maximum score ACF, lags 1–60 | 0.04695 | 0.04039 | 1/51 |
| T1 input/score correlation magnitude | 0.04429 | 0.02261 | 1/51 |
| T1 maximum run phase-projection fraction | 0.02983 | 0.01971 | 1/51 |
| Maximum cross-channel score correlation, lags ±10 | 0.04895 | 0.02540 | 1/51 |

Cross dependence peaks at lag -2 in the stated T1-versus-T2 convention.
Nonrandomized joint transition counts also flag a particular run more strongly
than pooled centered cross-event counts. No observed statistic is a unique
diagnosis of shared environmental noise: reproducible mean errors can produce
cross-channel dependence. The references hold fitted parameters fixed, do not
refit them or repeat model selection, and these ranks are descriptive rather
than calibrated composite-null p-values. Multiple diagnostics were examined.
Evidence: joint_b10_nonrandom_diagnostics; its figure was inspected.

Whitening is therefore unfinished. The larger randomization-based check looked
less concerning, but cannot overrule the code-based input/phase and cross
structure. The completed recovery sweep below adds refitted diagnostic references within
the chosen working model, with every declared outcome retained.

## Next discriminating checks

The updated mean diagnostics also passed an independent audit of all 960
decomposition cells, selected code expectations and actual held-input complex
responses. After 450 s, the three fast T1 phase errors remain about -2.8 to
-6.0 degrees; the fast T2 complex regions still include zero, so precise phase
intervals are inappropriate. The inspected phase-fold figure shows reproducible
T1 structure at the slower sine periods as well.

Using earlier codes to update the residual state reduces whole-data one-step
reported-temperature RMSE to 0.103/0.087 C, compared with physical free-simulation
RMSE 0.605/0.118 C. At a 60 s horizon it is 0.255/0.110 C. These use parameters
estimated from all twelve records, and are not retrospective omitted-run scores.
Good feedback-assisted prediction does not validate the physical free simulation.

The cross-localization references are now complete, fifty successes and no
failure. Removing each run's trend and three sine harmonics changes pooled
maximum cross correlation only from 0.04895 to 0.04719; the matched reference
97.5% quantile is 0.02603, with upper-tail rank 1/51. The largest run-wise
correlation falls from 0.1598 to 0.1506, still above the matched reference range.
The figure was inspected. Projection has not resolved the dependence.

1. Carry the completed localization into the disturbance-model comparisons.
   Projection is descriptive; it does not repair the physical model or uniquely
   identify an environmental mechanism.
2. Compare shared OU driving noise with shared analogue measurement noise in
   matched continuous-Gaussian joint refits. Independent mode must reproduce
   the previous Gaussian fit under a dense covariance audit, which passed.
   Shared process driving noise gives correlation 0.0908 and improves Gaussian
   nll by 4.3501; its full-data dense audit passed. Shared measurement noise
   gives correlation -0.00692 and improves nll only 0.16748, also audited.
   The combined fit and dense audit also passed: process/measurement
   correlations are 0.11587/-0.02463, with total nll improvement 6.14891.
   These comparisons
   are sensitivities on discrete measurements, not the quantized joint likelihood.
3. Complete profiles, initial sensor-bin sensitivities, the declared fifty
   nuisance-refitted recovery draws, and retrospective run/protocol transfer.
   Local active-face mixtures remain explicitly approximate and must be paired
   with profile, nuisance-band, initialization and model-form limits.

The final report will distinguish a useful physical mean, a useful short-term
predictor and a validated stochastic model. Current evidence does not establish
all three at once.

## Audited coarse lag profile

All nine fixed-input-lag points passed the independent selected-point audit.
Each point refits the other mean, per-run nuisance and noise coordinates. The
grid does not establish a globally optimized continuous profile or confidence
endpoints.

| Fixed input lag, s | Twice the nll increase from the reference |
| ---: | ---: |
| 1 | 455.660 |
| 2 | 77.879 |
| 2.7 | 0.119 |
| 2.73048 (reference) | approximately 0 |
| 4 | 141.575 |
| 7 | 552.445 |
| 12 | 219.869 |
| 15 | 6.253 |
| 20 | 775.555 |

At fixed L=15 s, the fitted sensor time is 2.643 s. The second region is
consistent with the near exchange of input/sensor lag labels, but the coarse
15 s point does not locate its minimum or exclude it at a calibrated cutoff.
The declared refinement at the original exact log sensor-time coordinate is
now terminal and independently audited. At L=14.7900558 s it fits sensor time
2.7313751 s, with nll 6240.69230385 and T1/T2 RMSE 0.615/0.118 C. Both starts
converged. Twice the nll increase is 4.78730, smaller than the coarse 15 s value
but still above the reference. This fixed point does not locate the free
secondary minimum or establish a calibrated exclusion of the lag assignment.
The existing baseline therefore remains the reference.

Evidence: profile_b10_short_lag_v2, profile_b10_short_lag_audit and the inspected
profile_gain_lag_summary_v2. The latter also plots the seven audited gain-grid
points. Plot lines guide the eye; detail panels show only the actual grid.

## First complete protocol-transfer group

All four primary omitted-protocol fits and their canonical forecasts passed
independent numerical audits. Each fit used the nine retained records; its three
omitted records initialized the physical states from their first readings only.
No later test temperatures estimated ambient, conductance or physical parameters.
Each of the twelve records appears once in this group (10,795 scored readings
per channel). All records previously informed structure selection, so this is
retrospective transfer.

| Pooled quantity | T1 | T2 |
| --- | ---: | ---: |
| Nominal reported-mean RMSE, C | 2.131 | 0.875 |
| Noise-only marginal range coverage | 32.1% | 36.7% |
| Noise-only mean range width, C | 2.010 | 0.303 |
| Permitted-condition plus noise coverage | 99.57% | 99.95% |
| Permitted-condition plus noise mean width, C | 9.072 | 5.173 |

Nominal conditions set the unknown run's conductance ratio to one and ambient
offset to zero. The outer ranges allow the declared ±10% conductance band and
±2 C ambient range. They are hard scenario envelopes combined with marginal
noise quantiles, not probability distributions over conditions or calibrated
total-95% predictions. Parameter draws are still pending. High coverage from
these wide ranges does not identify whether changing conditions or model error
caused the nominal discrepancies. The twelve forecast figures were inspected.
Evidence: transfer_primary_protocols, its four point audits, the four
transfer_primary_protocol*_noise forecasts/audits, and
transfer_primary_protocols_noise_summary. That summary remains explicitly
partial while other folds and uncertainty components are unfinished.

Run mean errors account for 85.7% of T1 and 73.2% of T2 squared prediction
error. Removing each run's mean error reduces the pooled RMSE to 0.806/0.453 C.
This decomposition uses the omitted observations and cannot be applied as a
blind prediction correction. It separates run-level shifts from changing error
without assigning either to a physical mechanism. The independently checked
summary is transfer_primary_protocols_noise_decomposition; its arithmetic
fixture transfer_accounting_check_v3 includes an all-offset limiting case.

Centering also affects interpretation. Each training subset defines a new
geometric nominal Ua, so K0, tA, Us/Ua0 and Ux/Ua0 can shift together merely
because the reference changed. Compare beta/CH=K0/tA, Us/CH, Ux/CH and the
lag times instead. The source-bound reference_transport_check verifies this
cohort/training issue and exact preservation of the canonical forecasts.

For a centered twelve-run cohort with deviations bounded by ±10%, an omitted
run's ratio to a nine-run training nominal can range from 0.86894 to 1.13551.
This is centering algebra, not an observed conductance range. The all-data fit's
re-centered reference ratios fall outside the canonical training band in one
300 s sine record and two step records; those are fitted-reference diagnostics,
not measured airflow or blind forecast inputs. A separately declared forecast
stress check will propagate the algebraic range for all four primary protocols.
Training constraints and all canonical forecasts stay unchanged. This check
cannot be presented as a transported cohort likelihood or corrected coverage.

## First parameter-inclusive protocol check

The omitted-30-s protocol pipeline is terminal. Its information matrix,
independent information audit, 256 local draws, 64/256-draw forecasts and full
forecast audits all passed. The selected active face fixes sensor loading at
its bound; it does not include alternate lag basins or released-bound/profile
uncertainty. All nine new forecast figures were inspected.

Across these three fast-sine records, the canonical hard-condition-plus-noise
mean widths are 9.119/5.134 C. Adding the 256-draw local parameter mixture gives
9.223/5.206 C, with observed marginal coverage unchanged at 100/99.963 percent.
This is a modest change to an already broad hard-condition envelope, not a
calibrated total-95-percent prediction or a repair of the nominal mean error.
The separate centering-reference stress check gives 10.407/5.683 C widths and
the same coverage. It preserves all training fits.

Using 256 rather than the shared-prefix 64 draws changes at most one reported
code (0.33 C) at any envelope endpoint; 2.26/2.69 percent of T1/T2 endpoints
change. Mean widths decrease by about 0.015/0.016 C. This checks finite-mixture
resolution only, not the validity of the local approximation. The aggregate
transfer study is still partial: the other nineteen declared fold pipelines
have not all completed their uncertainty components.

The audited local joint-bin log-input-lag SE is 0.1232 without the fast-sine
records, versus 0.0324 in the twelve-record fit. These are conditional local
comparisons. In this omitted-protocol fit, correlated analogue covariance
increases the input-lag SE relative to the same-marginal IID calculation by
1.293; in the full-data fit the ratio is 0.645. A single blanket uncertainty
inflation factor cannot describe both designs.

Evidence: transfer_uncertainty_protocol30 and the manifest-indexed
transfer_first_protocol_parameter_summary. The four remaining profile
coordinates are running in profile_remaining_b10; other initial-state,
recovery and retrospective-transfer gates remain open.

## Matched protocol comparison: shared versus bounded conductance

The four shared-conductance selected fits and all four canonical forecast/audit
pairs are terminal and passed. Each protocol group predicts the same twelve
records once (10,795 scored readings per channel). These groups reuse the same
observations and are retrospective because structure selection used all runs.

| Pooled reported-temperature quantity | Shared Ua, T1/T2 | 10% Ua band, T1/T2 |
| --- | ---: | ---: |
| Nominal RMSE, C | 2.171 / 0.894 | 2.131 / 0.875 |
| RMSE after removing each run's mean error, C | 0.815 / 0.453 | 0.806 / 0.453 |
| Noise-only marginal coverage | 50.3% / 53.5% | 32.1% / 36.7% |
| Noise-only mean width, C | 3.606 / 0.811 | 2.010 / 0.303 |
| Hard conditions plus noise coverage | 94.01% / 99.81% | 99.57% / 99.95% |
| Hard conditions plus noise mean width, C | 6.769 / 3.969 | 9.072 / 5.173 |

The all-data fitting improvement from allowing Ua variation has little
counterpart in nominal omitted-protocol RMSE. Both nominal forecasts still
set new-run ambient offset to zero and conductance to the training nominal.
The broader band model's higher outer-envelope coverage includes an additional
unknown-conductance range; it is not evidence of calibrated 95% probabilities
or a causal measurement of airflow. The shared model estimates larger residual
variance, so its noise-only ranges are broader despite similar nominal errors.
Removing run mean errors is descriptive and cannot be used as a blind correction.

All twelve new shared-model forecast figures and the two-group summary were
inspected. Evidence: transfer_shared_protocols, transfer_shared_protocols_audit,
transfer_shared_protocols_forecasts and
transfer_protocol_candidates_canonical_summary. The last remains partial:
twelve primary run forecasts and nineteen parameter pipelines are unfinished.

## Canonical transfer is now complete for all twenty folds

All twenty selected fits and canonical forecast/audit pairs have terminal
passed evidence. The twelve primary leave-run-out forecasts give pooled
reported-mean RMSE 1.750/0.754 C, compared with 2.131/0.875 C when an entire
protocol is omitted. Other repeats of the same protocol remain in training in
the first comparison. These groups reuse all twelve records and are retrospective.

For the leave-run group, run mean errors account for 84.9/72.5 percent of SSE;
removing them leaves RMSE 0.680/0.395 C. Nominal-condition noise ranges cover
46.0/47.6 percent of the scored temperatures, with mean widths 2.093/0.291 C.
The hard-condition-plus-noise ranges cover 100/99.94 percent, at mean widths
9.160/5.170 C. Those broad scenario ranges still do not define calibrated
probabilities for new-run conditions. Offset removal still uses test outcomes.

The twelve new run-forecast figures and the three-group summary were inspected.
transfer_all_canonical_summary accounts for all twenty folds, with no missing
fold IDs, but remains **partial** because nineteen parameter pipelines are
unfinished. The remaining six bin/initial/hidden-heater cases are now running
in initial_remaining_b10, alongside the remaining profiles and recovery shards.

## Complete fifty-draw recovery and refitted diagnostics

All fifty declared generated collections have accepted free and true-gain fits,
exact seed/data regeneration, saved refitted diagnostics and independently
passed selected-point numerical checks. The full-design audit binds both current
continuation ledgers and all one hundred cached point checks. No failed or missing
outcome is omitted. The source-bound summary and an independent arithmetic audit
also passed: one hundred saved objective values, all thirty-seven coordinates,
manual order-statistic quantiles, active-bound/lag counts and fourteen refitted
diagnostic statistics. Its four-panel figure was inspected.

The nominal 95% gain likelihood-ratio region covers the generating value in
49/50 collections. Direct binomial-tail inversion confirms the finite beta-quantile
output despite its retained SciPy overflow warning: the 95% Monte Carlo interval
is 89.353--99.949%. Fifty draws do not tightly establish coverage, and this is
conditional on the specified generating model, initial/bin conventions, bounds
and fitting algorithm. It does not calibrate hardware uncertainty or every
training-fold parameter mixture.

| Coordinate | Refitted estimator SD | Generating-point active-face local SE | SD / local SE |
| --- | ---: | ---: | ---: |
| log gain | 0.003018 | 0.003362 | 0.898 |
| log thermal time | 0.014765 | 0.003089 | 4.780 |
| log sensor time | 0.782163 | 0.011361 | 68.845 |
| log input time | 0.781136 | 0.032430 | 24.087 |
| Command curvature | 0.010054 | 0.011243 | 0.894 |

Fifteen of fifty fits put the sensor time below the input time, reversing their
generating labels. The estimator distribution crosses lag basins, so the large
labeled SDs must not be read as a single Gaussian uncertainty interval. Unordered
fast/slow time quantiles (2.5/50/97.5%) are 2.547/2.733/2.898 s and
14.419/14.805/15.576 s. These are conditional repeated-fit quantiles, not confidence
intervals for the device. The lag scales are much more stable than their physical
assignment. The thermal-time spread also exceeds the fixed-face approximation;
gain and command-curvature spread are closer to their conditional local SEs.

| Hardware statistic | Data | Refitted reference 97.5% | Descriptive upper-tail rank |
| --- | ---: | ---: | ---: |
| T1 input/score correlation | 0.04429 | 0.01689 | 1/51 |
| T1 maximum phase projection | 0.02983 | 0.02146 | 1/51 |
| T2 maximum score ACF, lags 1--60 | 0.04695 | 0.03894 | 1/51 |
| Cross-channel maximum, lags +/-10 | 0.04895 | 0.03014 | 1/51 |

All four hardware statistics exceed every refitted reference. Accounting for
parameter fitting has not explained the input/phase and cross dependence. These
ranks account for fitting within the declared family/algorithm, but not structure
selection or a joint multiple-diagnostic calibration. Approximately IID
innovations have not been established.

Evidence: [full numerical/data audit](results/recovery_b10_full50_audit/audit.json),
[full summary](results/recovery_b10_full50_summary/summary.json),
[independent scalar/quantile/diagnostic audit](results/recovery_b10_full50_summary_audit/audit.json)
and [inspected figure](results/recovery_b10_full50_summary/recovery_summary.png).
The separate protocol120 information failure and targeted loading-start
investigation are recorded in transfer_methods.md; no covariance from its
nonpositive-curvature point is accepted.
