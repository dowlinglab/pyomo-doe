# Interim uncertainty and whitening findings

This is a progress note, not the completed report. The goal remains active.
The selected mean/bin fits and information results below have terminal
independent audits. The gain profile and all seven point audits are complete;
remaining nuisance profiles, refitted coverage, retrospective transfer and
several initialization sensitivities remain unfinished.

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
structure. The recovery sweep will add refitted diagnostic references within
the chosen working model and retain every failed draw.

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
