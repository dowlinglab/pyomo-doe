"""Frequency identities checked by perturbation, determinants and phase sums."""
import numpy as np

from frequency_information import frequency_response, lag_sensitivity_area


def test_frequency_sensitivities_match_log_parameter_perturbations():
    p = np.array([.8, 40., 12.])
    frequencies = np.r_[0., np.logspace(-4, 1, 20)]
    for order in [1, 2, 12, 0]:
        G, derivative = frequency_response(p, order, frequencies)
        assert G[0] == p[0]
        for j in range(3):
            step = np.eye(3)[j]*1e-6
            plus = frequency_response(p*np.exp(step), order, frequencies)[0]
            minus = frequency_response(p*np.exp(-step), order, frequencies)[0]
            assert np.allclose((plus-minus)/2e-6, derivative[:, j], atol=1e-9, rtol=2e-7)


def test_lag_area_identity_and_repeated_pole_rank_loss():
    for order in [1, 2, 12, 0]:
        for tau in [6., 40.]:
            for omega in [.001, .1, 2.]:
                G, derivative = frequency_response([.8, tau, 12.], order, omega)
                h = derivative[0, 1:]/G[0]
                area = np.linalg.det(np.array([h.real, h.imag]))
                assert np.allclose(area, lag_sensitivity_area(tau, 12., order, omega), atol=1e-13, rtol=1e-12)
        if order:
            _, derivative = frequency_response([.8, 12./order, 12.], order, [.01, .1, 1.])
            assert np.allclose(derivative[:, 2], order*derivative[:, 1], atol=1e-15)


def test_exact_periodic_information_needs_more_than_one_unoffset_tone():
    phases = np.arange(128)*2*np.pi/128
    sigma, amplitude, dc = .25, 20., 50.
    for order in [1, 2, 12, 0]:
        _, derivative = frequency_response([.8, 40., 12.], order, .08)
        d = derivative[0]
        J_tone = amplitude*np.real(np.exp(1j*phases[:, None])*d)
        F_tone = J_tone.T@J_tone/sigma**2
        exact = len(phases)*amplitude**2/(2*sigma**2)*np.real(np.outer(d, d.conj()))
        assert np.allclose(F_tone, exact, atol=1e-9, rtol=1e-12)
        assert np.linalg.matrix_rank(J_tone, tol=1e-10) == 2
        J = J_tone+np.array([dc*.8, 0., 0.])
        assert np.linalg.matrix_rank(J, tol=1e-10) == 3
        assert np.allclose(J.T@J/sigma**2, F_tone+len(phases)/sigma**2*np.diag([(dc*.8)**2, 0., 0.]), atol=1e-8)
    for order in [1, 2, 12]:
        _, derivative = frequency_response([.8, 12./order, 12.], order, .08)
        J = amplitude*np.real(np.exp(1j*phases[:, None])*derivative[0])+[dc*.8, 0., 0.]
        assert np.linalg.matrix_rank(J, tol=1e-10) == 2
