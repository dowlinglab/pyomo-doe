# Residual likelihood: assumptions and checks

Mode: autonomous preparation. The numerical choices below are agent-authored;
Alex requested covariance modeling but has not reviewed the selected kernels.
This note is a derivation/implementation record, not a finished literature review.

For each run, e=y-m follows a stationary zero-mean Gaussian process independent
of other runs. Write C=sigma^2 R(ell,eta), where R has unit marginal variance.
OU uses R_ij=exp(-|ti-tj|/ell). OU+nugget uses
R_ij=(1-eta) exp(-|ti-tj|/ell)+eta I_ij. Matern-3/2+nugget replaces that
exponential with (1+a)exp(-a), a=sqrt(3)|ti-tj|/ell. The fraction eta is
measurement variance divided by total variance, not a standard-deviation ratio.
Bounds ell=[0.05,3000] seconds, eta=[0,1] are numerical defaults; record boundary
hits. IID is explicit. No variance positivity bound substitutes for a zero fit.

Innovations filtering gives W such that WRW^T=I. A full temporal transform is
applied to the response and every design column. For fixed nonlinear parameters,
minimize ||W(Xb-y)||^2 over shared K and bounded ambient terms. Profile
sigma^2=||W(Xb-y)||^2/N, and minimize

    nll = (N*(log(2*pi)+1+log(sigma^2)) + log(det R))/2.

The first fitted residual has stationary error covariance. Each run resets the
filter. This is conditional on treating the first T1 as fixed initialization and
omitting its residual, not a latent noisy-start model. Startup sensitivity remains
required. Alternating GLS estimates noise from current residuals, refits the
thermal/ambient mean with fixed temporal covariance, and repeats. Joint nonlinear
likelihood uses variable projection over the same bounded linear mean terms.

For Matern-3/2 define lambda=sqrt(3)/ell and scaled state x=(f,f'/lambda).
Stationary covariance is (1-eta) I. For a=lambda*dt, the transition is

    A = exp(-a) [[1+a, a], [-a, 1-a]],
    Q = (1-eta) (I-AA^T).

Small-a series prevent cancellation in Q00. The scalar measurement variance is
eta. Innovations gains and scales define the linear whitening of every column;
their log variances sum to log(det R). The OU filter is the one-state case.
`audit_covariance.py` compares the recurrence to dense kernel Cholesky on two
independent irregular grids, including zero/unit nugget and long/short lengths.

## Inspected theoretical source and remaining literature work

Hartikainen and Särkkä, *Kalman filtering and smoothing solutions to temporal
Gaussian process regression models* (2010), author-hosted
[PDF](https://users.aalto.fi/~ssarkka/pub/gp-ts-kfrts.pdf), inspected October 24,
2026, physical PDF pages 2-3 and 5: the Matérn half-integer kernels have exact
finite state-space representations, including the second-order p=1 case;
stationary initialization recovers the covariance-based Gaussian-process model.
Nonuniform sampling is allowed. Our scaled-state matrices above follow by a
coordinate change. This source motivates computational representation, not a
claim that these TC-Lab residuals follow that kernel.

Särkkä, *Bayesian Filtering and Smoothing* (2013), author-hosted
[PDF](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf), inspected
physical pages 196-197, gives recursive marginal likelihood.
The source-bound report packet must verify the final cited context; these source
notes alone do not constitute completed passage grounding.

Lab Literature metadata search for the exact Hartikainen/Särkkä title returned
zero indexed matches in snapshot 2530e8c89d532b5f5772c9b39b22fab6b615caa9331a6ac7c71de5cd211efce2
(2947 records, no metadata issues; remote sync and PDF validation not checked).
This is an acquisition gap, not evidence the paper is absent from the wider
literature. Use the inspected external primary source now, retain it for report
intake/grounding, and preserve the original report packet separately.
Current serving runtime: shared-literature-rag 0.5.9, installed/source fingerprint
8aa3bbb3ec0cb812e72f2c4413aef11221e14bde48f9e905de08d7b47743f985,
private state `state-0.5.9-maintenance-2026-10-22` under Application Support.
The literature-synthesis workflow is available; native passage/packet audit for
the new report is still pending. No shared library or client configuration was
changed by this inspection.

The author-hosted Hartikainen/Särkkä PDF was retained at
`/private/tmp/tclab_gp_kalman_source.pdf`, SHA-256
bba85ed12fee67d1dfc581ba67669906efa98070caaf1a9ac1f26aef04f3bdfd.
Rendered physical pages 3 and 5 were inspected: p=1 companion-state equations,
stationary initialization and nonuniform sampling agree with the implementation
derivation. No source PDF is redistributed in this repository.

## Numerical refinement prompted by the first pilot

`results/covariance_pilot_constant_ou_nugget` preserves the first four-start pilot
from 576da82, including a projected-gradient warning (0.334). Central difference
steps from 2e-5 to 2e-7 confirmed a nonzero derivative despite the optimizer's
function-decrease stopping status. Subsequent fits supply central derivatives
(step 2e-6) and retain any SLSQP refinement after an unsuccessful or nonstationary
L-BFGS-B result. This changes numerical optimization, not the likelihood, data,
bounds or adequacy criteria. Retain the pilot rather than replacing its record.

## Declared Tier 0 assumption checks

Before outcome inspection, `run_sensitivity.py` defines nine diagnostic cases:
right-held commands; fixed initial-temperature shifts of -0.165 and +0.165 C;
one second of constant-command prehistory in the input lag; ambient-offset
bounds widened to +/-4 C; and separate three-run fits for each of the 30-second,
two-minute, five-minute and step protocols. Each case repeats 24-start OLS,
alternating GLS and 24-start joint fitting with constant ambient and OU plus
measurement noise. Primary +/-2 C, left-held fits are preserved unchanged.
The optional startup terms default to zero, preserving prior predictions and
likelihood. Thirty-nine forward checks include the new startup scenarios.

The initial-temperature scenarios set true thermal T(0)=recorded T1(0)+dinit,
while retaining the recorded first reading as the ambient reference. Their mean
contribution is dinit*exp(-t/tau); the first residual remains excluded. The 0.165 C
magnitude is half the larger observed 0.33 C level spacing. It is a bounded
diagnostic choice, not a measured error distribution. The same-sign shifts in
all runs are not a worst-case envelope over independent run errors. Input
prehistory sets z(0)=u(0)*(1-exp(-1/theta)) while conditioning on the recorded
thermal initial state. Neither startup quantity becomes a new fitted parameter.

Agent hypotheses: a timing explanation should change the residual harmonic and
command association when the hold changes; active ambient bounds should matter
if widening them reduces the discrepancy; a shared-mean limitation may become
clearer in protocol-specific fits. A better-fitting endpoint convention alone
does not overturn the acquisition-code evidence. Do not use a covariance kernel
to disguise deterministic input dependence. Protocol fits diagnose Tier 0 and
do not add one-sine/one-step pairs to the six-pair Tier 1/2 budget.

`fit_diagnostics.py` reconstructs selected means with an independent augmented
matrix exponential and computes likelihoods/innovations with dense Cholesky.
It also reports descriptive residual harmonics after t=300 s, fitting a
constant, linear and quadratic trend plus the known sine/cosine. The five-minute
window contains only two cycles; no IID uncertainty claim accompanies these
harmonic coefficients.

## Refit diagnostic calibration and rounding sensitivity

Use a source-bound pilot before the 200-replicate diagnostic comparisons. Generate
from the leading pooled constant-ambient, OU-plus-measurement fit, preserving
the real commands/timestamps and fixed first temperatures. Refit thermal,
ambient and noise parameters for every replicate. Three starts include the
generating neighborhood, the other pole basin and an additional thermal seed;
noise values initialize optimization and are re-estimated, never fixed.
Check selected refits with the full 24-start procedure before relying on the
shorter refit budget. Preserve failed and nonstationary fits separately.

Use global maximum absolute innovation ACF (lags 1-30), maximum absolute
innovation/command correlation (lags -30 to 30) and maximum per-run Q30, as well
as run-level statistics. Refit simulation calibrates those selected searches;
finite plug-in tail fractions remain approximate. Two hundred draws give a
minimum plus-one fraction of 1/201, not arbitrarily small significance.

The alternative measurement mechanism retains the fitted slow OU process but
replaces the independent Gaussian measurement component with temperature-grid
rounding followed by two-decimal CSV reporting. Infer the grid only from the
reference fitting records and require it to reconstruct every observed distinct
level after two-decimal rounding. This is a diagnostic explanation to challenge,
not an established hardware calibration. The Gaussian working likelihood is
refitted in both simulations. Comparing their residual statistics can distinguish
a generic Gaussian-kernel failure from effects compatible with discretized
observations; neither simulation establishes the physical mean model as true.

Save per-replicate seeds, all optimizer outcomes and fitted nuisance/noise terms.
Source/config/reference hashes bind a resumable checkpoint. Resume only a
matching identity into a fresh source-bound output directory. Failed draws stay
in denominators and are not silently replaced. This Tier 0 diagnostic experiment
is separate from the 400-draw uncertainty analysis for each of the six pairs.
