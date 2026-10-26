# Sequential fitting comparison

This comparison addresses the requested fit–diagnose–reweight sequence. It uses
the same 10% centered conductance band, constant ambient offsets and two-channel
physical mean as the audited Gaussian joint fit. Quantized likelihood fitting
is a separate comparison; a continuous Gaussian density and a discrete bin mass
are not directly comparable scores.

The alternating procedure has two blocks:

1. Hold the current physical mean fixed. Estimate OU length, independent-noise
   fraction and channel scale ratio from its residuals, including the covariance
   determinant. Profile the common variance by maximum likelihood.
2. Hold those covariance shape parameters fixed. Fit the physical mean by
   constrained GLS, profiling gain and bounded ambient coefficients. Profile the
   common variance again. Recompute residuals and repeat.

Both blocks minimize the same Gaussian likelihood. The monotone relationship
between weighted squared error and its profiled variance makes the mean block
a GLS update. Reprojecting the bounded linear coefficients immediately after a
covariance update is recorded separately. No residual crosses a run boundary.

Two declared initializations are used: fit the physical mean with IID errors,
then start OU covariance at 30 s and independent fraction 0.5; and start at the
saved joint Gaussian solution to check that it is a fixed point. The explicit
interior covariance initialization can change likelihood relative to the IID
fit. Later block updates must be nonincreasing to within 2e-5 nll units.

Stop after at most 30 cycles. Declare convergence only if both block optimizers
succeed, the cycle's nll improvement is below 1e-5, the largest coordinate change
is below 2e-4, centering error is below 1e-7, and the raw projected Lagrangian
gradient is below 0.05. Save every block, parameter vector, objective, failure and
stationarity diagnostic. An iteration limit is a result, not convergence. The
Gaussian reference may be a local optimum; disagreement requires investigation.

Before the full run, an independent dense covariance calculation checks the
determinant, quadratic form, covariance finite differences and a constrained
mean update on short synthetic outputs with actual recorded input/time grids.
The full-record result will be checked against the same dense arithmetic used
for the original joint fit. This verifies computation, not the adequacy of a
Gaussian approximation to the hardware's quantized readings.
