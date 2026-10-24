"""Stationary covariance kernels and exact innovations whitening on irregular grids.

For Matern 3/2 use state (f, f'/lambda), stationary covariance (1-eta)*I,
lambda=sqrt(3)/ell. eta is the independent measurement variance fraction;
the overall variance is profiled separately. See methods.md for source/context.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _filter_plan(time, kind, length, nugget):
    n = len(time)
    transitions = np.zeros((n, 2, 2))
    gains = np.zeros((n, 2))
    scales = np.ones(n)
    slow = 1-nugget
    p00, p01, p11 = slow, 0., slow if kind == 2 else 0.
    logdet = 0.
    for k in range(n):
        if k > 0:
            h = time[k]-time[k-1]
            if kind == 1:
                a = np.exp(-h/length)
                transitions[k, 0, 0] = a
                p00 = a*a*p00+slow*(-np.expm1(-2*h/length))
            else:
                a = np.sqrt(3.)*h/length
                decay = np.exp(-a)
                a00, a01 = decay*(1+a), decay*a
                a10, a11 = -decay*a, decay*(1-a)
                transitions[k, 0, 0] = a00
                transitions[k, 0, 1] = a01
                transitions[k, 1, 0] = a10
                transitions[k, 1, 1] = a11
                if a < 1e-3:
                    q00 = a**3*(4/3-2*a+8/5*a*a-8/9*a**3+8/21*a**4-2/15*a**5)
                else:
                    q00 = -np.expm1(-2*a)-2*a*(1+a)*decay*decay
                q01 = 2*a*a*decay*decay
                q11 = -np.expm1(-2*a)+2*a*(1-a)*decay*decay
                v00 = a00*a00*p00+2*a00*a01*p01+a01*a01*p11+slow*q00
                v01 = a00*a10*p00+(a00*a11+a01*a10)*p01+a01*a11*p11+slow*q01
                v11 = a10*a10*p00+2*a10*a11*p01+a11*a11*p11+slow*q11
                p00, p01, p11 = v00, v01, v11
        variance = p00+nugget
        if variance <= 0 or not np.isfinite(variance):
            raise ValueError("Nonpositive innovation variance")
        gains[k, 0], gains[k, 1] = p00/variance, p01/variance
        scales[k] = np.sqrt(variance)
        logdet += np.log(variance)
        # The first two updates avoid subtractive cancellation at nugget=0.
        p11 = p11-p01*p01/variance
        p01 = p01*nugget/variance
        p00 = p00*nugget/variance
    return transitions, gains, scales, logdet


@njit(cache=True)
def _apply(values, transitions, gains, scales):
    result = np.empty_like(values)
    for column in range(values.shape[1]):
        m0, m1 = 0., 0.
        for k in range(len(values)):
            a = transitions[k]
            v0, v1 = a[0, 0]*m0+a[0, 1]*m1, a[1, 0]*m0+a[1, 1]*m1
            innovation = values[k, column]-v0
            result[k, column] = innovation/scales[k]
            m0, m1 = v0+gains[k, 0]*innovation, v1+gains[k, 1]*innovation
    return result


@njit(cache=True)
def _inverse(values, transitions, gains, scales):
    result = np.empty_like(values)
    for column in range(values.shape[1]):
        m0, m1 = 0., 0.
        for k in range(len(values)):
            a = transitions[k]
            v0, v1 = a[0, 0]*m0+a[0, 1]*m1, a[1, 0]*m0+a[1, 1]*m1
            innovation = values[k, column]*scales[k]
            result[k, column] = v0+innovation
            m0, m1 = v0+gains[k, 0]*innovation, v1+gains[k, 1]*innovation
    return result


class Whitening:
    def __init__(self, times, kernel, parameters):
        if kernel not in {"iid", "ou", "ou_nugget", "matern32_nugget"}:
            raise ValueError("Unknown covariance kernel")
        self.times = times
        self.kernel = kernel
        self.parameters = parameters
        self.plans = []
        self.logdet = 0.
        if kernel == "iid":
            return
        kind = 2 if kernel == "matern32_nugget" else 1
        ell = float(parameters[0])
        eta = float(parameters[1]) if kernel.endswith("_nugget") else 0.
        if ell <= 0 or not 0 <= eta <= 1:
            raise ValueError("Invalid covariance parameters")
        for time in times:
            plan = _filter_plan(time, kind, ell, eta)
            self.plans.append(plan[:3])
            self.logdet += plan[3]

    def __call__(self, values):
        values = np.asarray(values)
        if self.kernel == "iid":
            return values.copy()
        vector = values.ndim == 1
        matrix = np.ascontiguousarray(values[:, None] if vector else values)
        result = np.empty_like(matrix)
        offset = 0
        for time, plan in zip(self.times, self.plans):
            section = slice(offset, offset+len(time))
            result[section] = _apply(matrix[section], *plan)
            offset += len(time)
        if offset != len(values):
            raise ValueError("Whitening dimensions do not match run boundaries")
        return result[:, 0] if vector else result

    def inverse(self, values):
        values = np.asarray(values)
        if self.kernel == "iid":
            return values.copy()
        vector = values.ndim == 1
        matrix = np.ascontiguousarray(values[:, None] if vector else values)
        result = np.empty_like(matrix)
        cursor = 0
        for time, plan in zip(self.times, self.plans):
            section = slice(cursor, cursor+len(time))
            result[section] = _inverse(matrix[section], *plan)
            cursor += len(time)
        if cursor != len(values):
            raise ValueError("Inverse whitening run boundaries differ")
        return result[:, 0] if vector else result


def profiled_likelihood(residual, whitening):
    innovation = whitening(residual)
    n = len(innovation)
    scale2 = float(innovation@innovation/n)
    if not np.isfinite(scale2) or scale2 <= 0:
        raise ValueError("Invalid profiled variance")
    nll = .5*(n*(np.log(2*np.pi)+1+np.log(scale2))+whitening.logdet)
    return float(nll), float(np.sqrt(scale2)), innovation/np.sqrt(scale2)


def dense_covariance(time, kernel, parameters):
    if kernel == "iid":
        return np.eye(len(time))
    distance = np.abs(time[:, None]-time[None, :])
    ell = parameters[0]
    eta = parameters[1] if kernel.endswith("_nugget") else 0.
    if kernel == "matern32_nugget":
        d = np.sqrt(3.)*distance/ell
        C = (1+d)*np.exp(-d)
    else:
        C = np.exp(-distance/ell)
    return (1-eta)*C+eta*np.eye(len(time))
