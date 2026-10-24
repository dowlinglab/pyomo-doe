"""Compare finite differences with exact dense Gaussian covariance derivatives."""
import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from covariance import Whitening, profiled_likelihood
from covariance_fit import covariance_bounds, numerical_gradient
from run_mean import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(19217)
    time = np.r_[0., np.cumsum(rng.uniform(.5, 1.5, 39))]
    residual = rng.normal(size=len(time))
    distance = np.abs(time[:, None]-time[None, :])
    rows = []
    for length, eta in [(20., .05), (3000., 1e-4), (3000., .001), (100., .5)]:
        R = np.exp(-distance/length)
        C = (1-eta)*R+eta*np.eye(len(time))
        factor = cho_factor(C, lower=True)
        v = cho_solve(factor, residual)
        q = residual@v
        derivatives = [(1-eta)*R*distance/length, np.eye(len(time))-R]
        exact = np.array([.5*(np.trace(cho_solve(factor, D))-len(time)*(v@D@v)/q) for D in derivatives])
        def objective(x):
            W = Whitening([time], "ou_nugget", [np.exp(x[0]), x[1]])
            return profiled_likelihood(residual, W)[0]
        computed = numerical_gradient(objective, np.array([np.log(length), eta]), covariance_bounds("ou_nugget"), 2e-6)
        relative = float(np.max(np.abs(computed-exact)/np.maximum(1., np.abs(exact))))
        if relative > 2e-4:
            raise AssertionError((length, eta, computed, exact, relative))
        rows.append({"length_s": length, "eta": eta, "finite_difference": computed.tolist(),
                     "dense_analytic_derivative": exact.tolist(), "max_scaled_error": relative})
    write_json(args.output/"audit.json", {"status": "passed", "cases": rows,
               "scope": "Profile-variance Gaussian covariance derivative with fixed residuals, including small measurement fraction. Does not establish all profiled minima or asymptotic validity."})


if __name__ == "__main__":
    main()
