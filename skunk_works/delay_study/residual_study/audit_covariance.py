"""Check recursive whitening/likelihood against dense covariance Cholesky."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular

from covariance import Whitening, dense_covariance, profiled_likelihood


def audit():
    rng = np.random.RandomState(817)
    grids = [np.r_[0., np.cumsum(rng.uniform(.1, 3, 39))],
             np.r_[4., 4.+np.cumsum(rng.uniform(.5, 2, 31))]]
    matrix = rng.normal(size=(sum(map(len, grids)), 4))
    cases = []
    for kernel in ("iid", "ou", "ou_nugget", "matern32_nugget"):
        settings = [[]] if kernel == "iid" else ([[.05], [3.], [3000.]] if kernel == "ou" else
                    [[.05, .2], [3., 0.], [3., .05], [3., .9], [3., 1.], [3000., .01]])
        for parameters in settings:
            W = Whitening(grids, kernel, parameters)
            expected = []
            logdet = 0.
            start = 0
            for t in grids:
                C = dense_covariance(t, kernel, parameters)
                L = cholesky(C, lower=True)
                expected.append(solve_triangular(L, matrix[start:start+len(t)], lower=True))
                logdet += 2*np.log(np.diag(L)).sum()
                start += len(t)
            expected = np.vstack(expected)
            recursive = W(matrix)
            error = float(np.max(np.abs(expected-recursive)))
            delta = float(abs(logdet-W.logdet))
            limit = 2e-6 if kernel == "matern32_nugget" and parameters[1] == 0 else 2e-8
            if error > limit or delta > limit:
                raise AssertionError((kernel, parameters, error, delta))
            direct_sigma = float(np.sqrt(np.mean(expected[:, 0]**2)))
            direct_nll = .5*(len(matrix)*(np.log(2*np.pi)+1+2*np.log(direct_sigma))+logdet)
            nll, sigma, _ = profiled_likelihood(matrix[:, 0], W)
            if abs(nll-direct_nll) > limit or abs(sigma-direct_sigma) > limit:
                raise AssertionError("Profile likelihood disagreement")
            cases.append({"kernel": kernel, "parameters": parameters,
                          "whitening_max_abs": error, "logdet_abs": delta,
                          "nll_abs": abs(nll-direct_nll), "tolerance": limit})
    # A long almost-coincident interval limit is checked without an ill-conditioned
    # dense comparison; it must retain a finite positive innovation variance.
    stress = Whitening([np.array([0., 1e-6, 1., 901.])], "matern32_nugget", [3000., 0.])
    if not np.all(np.isfinite(stress(np.ones(4)))):
        raise AssertionError("Small-interval covariance limit failed")
    return {"status": "passed", "scope": "Unit-shape covariance, multiple runs, matrix/vector whitening, profile variance, stationary first state, IID/OU/nugget limits",
            "cases": cases, "small_interval_limit_finite": True}
