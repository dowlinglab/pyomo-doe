"""Pointwise mean/prediction uncertainty with explicit estimability checks."""
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from inference import _sandwich


def local_covariances(J, residual, experiment_sizes, sigma=.25, bandwidths=(10, 30, 60)):
    """Covariances on the estimable quotient; null directions stay explicit.

    A proven redundant original coordinate must not add a residual degree of
    freedom. These baseline cases have algebraically established regular
    identifiable quotients; numerical rank alone would not justify this rule
    at a nonregular coincident-pole point.
    """
    J, residual = np.asarray(J), np.asarray(residual)
    _, singular, Vh = np.linalg.svd(J, full_matrices=False)
    keep = singular > 1e-8*singular[0]
    rank = int(keep.sum())
    bread = (Vh[keep].T/singular[keep]**2)@Vh[keep]
    variance = float(residual@residual/max(1, len(residual)-rank))
    matrices = {"iid_known_sigma": bread*sigma**2, "iid_unknown_quotient_dof": bread*variance}
    # The shared score helper uses the supplied column count in its finite-
    # sample factor. Adjust only that factor to the identifiable quotient rank.
    correction = max(1, len(residual)-J.shape[1])/max(1, len(residual)-rank)
    matrices.update({"hac"+str(b)+"_quotient_dof": _sandwich(J, residual, experiment_sizes, b, bread)*correction for b in bandwidths})
    return matrices, Vh[~keep].T, {"rank": rank, "residual_dof": len(residual)-rank,
                                      "known_sigma_C": sigma,
                                      "estimated_future_iid_variance_C2": variance,
                                      "singular_values": singular.tolist()}


def estimable_variance(target_sensitivities, covariance, null_space=None, relative_tolerance=1e-8):
    """Local quotient variance; global target identifiability needs another check."""
    J, C = np.asarray(target_sensitivities), np.asarray(covariance)
    if C.shape != (J.shape[1], J.shape[1]) or not np.isfinite(C).all():
        raise ValueError("Finite covariance with matching target dimension required")
    C = (C+C.T)/2
    values = np.linalg.eigvalsh(C)
    if values[0] < -1e-9*max(abs(values).max(), 1e-30):
        raise ValueError("Covariance is not positive semidefinite")
    defined = np.ones(len(J), dtype=bool)
    if null_space is not None and np.asarray(null_space).size:
        null = np.asarray(null_space)
        defined = np.linalg.norm(J@null, axis=1) <= relative_tolerance*np.maximum(np.linalg.norm(J, axis=1), 1e-30)
    variance = np.maximum(np.einsum("tp,pq,tq->t", J, C, J), 0.)
    variance[~defined] = np.nan
    return variance, defined


def constant_equilibrated_target(data, spec):
    """An algebraically delay-independent target under equilibrated lag states."""
    return spec.initialization == "input_equilibrated" and bool(np.all(data.input == data.input[0]))


def normal_bands(mean, mean_variance, future_error_variance=None):
    """Local normal bands; future error law is an additional working assumption."""
    mean, variance = np.asarray(mean), np.asarray(mean_variance)
    z = norm.ppf(.975)
    result = {"mean_lower": mean-z*np.sqrt(variance), "mean_upper": mean+z*np.sqrt(variance)}
    if future_error_variance is not None:
        width = z*np.sqrt(variance+future_error_variance)
        result.update(prediction_lower=mean-width, prediction_upper=mean+width)
    return result


def gaussian_mixture_quantiles(mean_draws, sigma, probabilities=(.025, .975), steps=36):
    """Invert the exact equal-weight Gaussian mixture, without future-noise MC.

    Parameter bootstrap draws form the empirical mean distribution. Each mixture
    component adds an independent future Gaussian error of the stated sigma.
    This is a conditional plug-in prediction distribution, not a posterior.
    """
    means = np.asarray(mean_draws)
    probabilities = np.asarray(probabilities)
    if means.ndim != 2 or not np.isfinite(means).all() or sigma <= 0:
        raise ValueError("Finite draw-by-time means and positive future sigma required")
    lower = np.repeat((means.min(axis=0)-9*sigma)[None, :], len(probabilities), axis=0)
    upper = np.repeat((means.max(axis=0)+9*sigma)[None, :], len(probabilities), axis=0)
    for _ in range(steps):
        middle = (lower+upper)/2
        cdf = ndtr((middle[:, None, :]-means[None, :, :])/sigma).mean(axis=1)
        below = cdf < probabilities[:, None]
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    result = (lower+upper)/2
    residual = ndtr((result[:, None, :]-means[None, :, :])/sigma).mean(axis=1)-probabilities[:, None]
    return result, float(np.max(abs(residual)))
