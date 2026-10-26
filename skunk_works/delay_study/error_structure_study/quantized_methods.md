# Joint fitting with discrete temperature measurements

The first bin-likelihood fits held the physical trajectories fixed. They changed
the fitted noise split substantially, so fitting the mean again under that
likelihood is necessary. Adding a quantizer to an unchanged Gaussian noise fit
can count some measurement variation twice.

Write the unreported analogue observation as

\[
 Y_i^*=m_i(\psi)+X_i+v_i,\qquad
 v_i\sim N(0,R),\qquad X_1\sim N(0,S),
\]
\[
 X_i=a_iX_{i-1}+w_i,\quad
 a_i=e^{-\Delta t_i/\ell},\quad
 w_i\sim N\{0,S(1-a_i^2)\}.
\]

The recorded code gives an interval for Y*. The intervals have width 0.3223 C
before final decimal reporting. Their phase remains an assumption: nearest-code
and ADC-style floor bins are separate sensitivities. No simulator noise or
clipping law is supplied as hardware evidence.

Forward quadrature evaluates the sequence of conditional bin probabilities.
Backward smoothing gives posterior latent-state moments and adjacent-state
cross moments. Independent Gaussian rectangle integrals and node-count checks
verify this numerical approximation. Its posterior is generally non-Gaussian;
a Gaussian Kalman update on rounded bin centers is a different likelihood.

The likelihood score can be computed without differentiating every filter
operation. With fixed bin limits and positive variances, differentiating under
the complete-data integral gives

\[
 \nabla\log p_\theta(Q)
 =E_\theta[\nabla\log p_\theta(Y^*,X)\mid Q].
\]

This is Fisher's identity. In particular,

\[
 \frac{\partial\log p(Q)}{\partial m_i}
 =\frac{E[v_i\mid Q]}{R}.
\]

Multiplying these components by the physical mean Jacobian gives the mean-model
score. The OU covariance score uses E[X_i^2], E[X_i X_{i-1}] and the corresponding
expected squared process innovations. Measurement-variance updates use E[v_i^2].
The implementation uses log standard deviations and log correlation length;
it includes the stationary initial-state density separately for each run.
All ambient offsets, centered conductance deviations and noise parameters can
therefore be re-estimated within the same bin likelihood. Constant alpha remains
absorbed in gain.

[ Poyiadjis, Doucet and Singh (2011), section 2.1, equations 4–6 ](https://www.stats.ox.ac.uk/~doucet/poyiadjis_doucet_singh_particlescoreparameterestimation.pdf)
provides the score and observed-information identities for state-space models,
under regularity permitting exchange of integration and differentiation. The
paper's particle algorithms are not reproduced here; our scalar state integrals
use deterministic quadrature. The formulas above are a specialization derived
for this OU/quantized observation model. The author-hosted 16-page published PDF
was inspected October 25, 2026; source PDF bytes were not downloaded or hashed.

A valid gradient is not a proof of a global optimum or physical adequacy. Check
quadrature closure, likelihood convergence, independent moment integrals,
finite-difference scores, optimizer stationarity, parameter bounds and
alternative starting points. Hardware interpretations and interval claims
require the later identifiability, simulation and transfer analyses.

Randomized predictive residuals must use independent streams for different
channels and runs. Earlier marginal pilot outputs reused seeds and remain valid
as marginal illustrations; their joint randomization is unsuitable for testing
cross-channel or cross-run dependence. Regenerate those joint diagnostics with
independent recorded streams. Inspect bin-transition and predictive checks as
well as randomized Gaussian residuals, since coarse-bin randomization can
obscure a mean discrepancy.


Posterior-domain correction: a predictive mean-plus/minus-SD window can omit a
material fraction of the posterior for a rare observed bin. Increasing node
count does not repair that domain error. The corrected grid uses the Gaussian
conditional state distribution for each mixture component and each analogue
value within the observed bin. Component conditional means at the bin endpoints,
plus conditional SD tails, bound the integration region. Joint score calculations
use twelve conditional SDs and require wider-support convergence at problematic
seeds. Initial covariance inflation, when requested, is an initialization
heuristic based on residual RMS; its use of step-squared/12 is not an assumption
that quantization error is independent uniform noise in the fitted likelihood.


Initial-condition sensitivity remains part of the uncertainty analysis. The
current working likelihood fixes physical initial sensor states from the first
reported readings, omits those readings from the score, and starts the residual
OU process with a stationary prior at the first scored time. That differs from
conditioning the latent error state on the initial quantized measurement. Test
initial sensor values within their code bins, consistent with bin phase, and
assess the residual initial-state convention. Hidden heater offsets alone do
not represent initial sensor uncertainty. For long correlation lengths,
a=exp(-dt/ell) approaches 1 and process variance approaches 2*S*dt/ell;
profiles must check whether the data distinguish S and ell or mainly their
ratio. A zero-start or integrating-disturbance limit is a possible sensitivity,
not an established explanation for this device.

## Predictive checks after joint fitting

Use 20 independent randomized-PIT streams, with different recorded seeds for
each run and channel. Measure within-run serial correlation through sample lag
60, per-run PACF through lag 20 and Ljung–Box statistics at 1/5/10/20/60,
cross-channel correlation through lags -10 to 10, marginal shape and
correlation with the command held over the preceding interval. Pooled lag
products never join the end of one experiment to the start of another. Check
tail fractions, fixed command/temperature bins, and projections on each sine's
fundamental and first two higher harmonics. Simulated references, rather than
unadjusted chi-squared p-values, accompany these descriptive statistics.

Also evaluate nonrandomized probabilities that the next reported code decreases,
stays the same or increases. These integrate the predictive observation mixture
over thresholds at the preceding code. They use the previous filtering state;
neither the current posterior nor a smoothed state is a valid forecast. Compare
observed transition counts with the sum of their conditional probabilities.
The sum of p(1-p) supplies a descriptive martingale variance scale. It is not
used to assert an independently calibrated Gaussian test after fitting.

Before interpreting these diagnostics, simulate 50 data sets conditional on
the fitted mean, noise, recorded commands, sampling times and first readings.
Generate stationary OU errors at the first scored time, independent analogue
noise and the assumed hardware bins. Retain synthetic observations, seeds,
every failure and denominators. Compare statistics with those fixed-model
simulation references, alongside the spread across PIT randomizations.
These simulations do not refit parameters or repeat model selection. Their
tail ranks are conditional discrepancy measures, not bootstrap p-values or
interval-coverage calibration. Separate refitted recovery/coverage work remains.

Validate down/same/up probabilities against an independent conditional
bivariate Gaussian observation integral for both declared bin phases. Gaussian
PIT appearance can hide coarse-bin discrepancies, so serial, cross-channel and
nonrandomized checks must be reported together.

## Physical initial sensor-bin sensitivity

In a separate cold-start sensitivity, add two bounded initial-state coordinates
per run. Each moves the heater and its sensor together from the first reported
temperature, preserving H_i(0)=S_i(0). The bounds are the first code's analogue
interval under the same phase used for all subsequent observations. An optional
0.1 C buffer widens each endpoint to assess dependence on treating the initial
reading as a noise-free code for the physical state. It is a scenario buffer,
not a simultaneous confidence interval or calibrated sensor-noise limit.

These initial departures propagate through the exact homogeneous thermal
solution. They do not shift ambient temperature: its reference remains the first
reported T1 and its fitted offset retains the original ±2 C bounds. Initial
sensor shifts and hidden heater-only shifts are separate comparisons. The first
reading is omitted from the score and the OU state still starts stationary at
the first scored time. This sensitivity does not integrate a joint physical-state
and initial-measurement posterior; that distinction must stay visible.
