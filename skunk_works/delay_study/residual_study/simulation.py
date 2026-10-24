"""Independent state-space error generation for refit/bootstrap diagnostics.

Transitions use scipy matrix exponentials rather than the whitening recurrence.
The generator conditions on the same fixed first temperature as the likelihood;
its supplied time vector is the fitted residual times, not the omitted T0.
"""
import numpy as np
from numba import njit
from scipy.linalg import expm


@njit(cache=True)
def _draw(transitions, roots, initial_scale, measurement_scale, normals):
    n, d, _ = transitions.shape
    latent = initial_scale*normals[0, :d]
    result = np.empty(n)
    result[0] = latent[0]+measurement_scale*normals[0, d]
    for k in range(1, n):
        latent = transitions[k]@latent+roots[k]@normals[k, :d]
        result[k] = latent[0]+measurement_scale*normals[k, d]
    return result


class NoiseProcess:
    def __init__(self, time, kernel, parameters, sigma):
        self.time = np.asarray(time)
        self.kernel = kernel
        self.sigma = sigma
        if kernel not in {"iid", "ou", "ou_nugget", "matern32_nugget"} or sigma <= 0:
            raise ValueError("Invalid noise generator")
        if kernel == "iid":
            self.dimension = 0
            return
        length = parameters[0]
        eta = parameters[1] if kernel.endswith("_nugget") else 0.
        if length <= 0 or not 0 <= eta <= 1:
            raise ValueError("Invalid shape parameters")
        self.dimension = 2 if kernel == "matern32_nugget" else 1
        d = self.dimension
        rate = np.sqrt(3.)/length if d == 2 else 1/length
        F = rate*np.array([[0., 1.], [-1., -2.]]) if d == 2 else np.array([[-rate]])
        slow_variance = sigma*sigma*(1-eta)
        stationary = slow_variance*np.eye(d)
        self.initial_scale = np.sqrt(slow_variance)
        self.measurement_scale = sigma*np.sqrt(eta)
        self.transitions = np.zeros((len(time), d, d))
        self.roots = np.zeros_like(self.transitions)
        for k, dt in enumerate(np.diff(time), 1):
            A = expm(F*dt)
            Q = stationary-A@stationary@A.T
            eigenvalues, vectors = np.linalg.eigh((Q+Q.T)/2)
            if eigenvalues.min() < -1e-12*max(1., slow_variance):
                raise ValueError("Transition covariance is not positive semidefinite")
            self.transitions[k] = A
            self.roots[k] = vectors@np.diag(np.sqrt(np.maximum(eigenvalues, 0.)))

    def draw(self, rng):
        if self.kernel == "iid":
            return self.sigma*rng.normal(size=len(self.time))
        normals = rng.normal(size=(len(self.time), self.dimension+1))
        return _draw(self.transitions, self.roots, self.initial_scale,
                     self.measurement_scale, normals)

    def covariance_factor_for_audit(self):
        n = len(self.time)
        if self.kernel == "iid":
            return self.sigma*np.eye(n)
        d = self.dimension
        state = np.zeros((d, n*(d+1)))
        state[:, :d] = self.initial_scale*np.eye(d)
        factor = np.zeros((n, n*(d+1)))
        for k in range(n):
            if k:
                state = self.transitions[k]@state
                state[:, k*d:(k+1)*d] += self.roots[k]
            factor[k] = state[0]
            factor[k, n*d+k] += self.measurement_scale
        return factor


def audit():
    from covariance import dense_covariance
    from scipy.linalg import block_diag
    rng = np.random.RandomState(9073)
    grids = [np.r_[0., np.cumsum(rng.uniform(.2, 3, 29))],
             np.r_[0., np.cumsum(rng.uniform(.5, 2, 19))]]
    rows = []
    for kernel in ("iid", "ou", "ou_nugget", "matern32_nugget"):
        for eta in ((0.,) if kernel in {"iid", "ou"} else (0., .05, .5, 1.)):
            parameters = [] if kernel == "iid" else [40.]+([eta] if kernel.endswith("_nugget") else [])
            factors = []
            covariances = []
            for time in grids:
                process = NoiseProcess(time, kernel, parameters, .47)
                factor = process.covariance_factor_for_audit()
                factors.append(factor)
                covariances.append(.47**2*dense_covariance(time, kernel, parameters))
                if not np.all(np.isfinite(process.draw(rng))):
                    raise AssertionError("Nonfinite generated sample")
            B = block_diag(*factors)
            C = block_diag(*covariances)
            error = float(np.max(np.abs(B@B.T-C)))
            if error > 1e-10:
                raise AssertionError((kernel, parameters, error))
            rows.append({"kernel": kernel, "parameters": parameters, "covariance_factor_error": error})
    return {"status": "passed", "cases": rows,
            "scope": "Exact linear generation factor against closed-form kernel, independent-run reset, zero/unit nugget; not a coverage study"}
