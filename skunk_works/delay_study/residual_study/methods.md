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
