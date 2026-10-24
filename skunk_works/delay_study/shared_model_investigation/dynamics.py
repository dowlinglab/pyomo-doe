"""Independent exact held-input propagation for fixed delay-chain orders."""
import numpy as np
from scipy.linalg import expm


def matrices(n, K, tau, theta):
    if n not in (1, 2, 3) or min(K, tau, theta) <= 0:
        raise ValueError('Positive parameters and n=1,2,3 required')
    rate = n/theta
    A = np.zeros((n+1, n+1))
    for j in range(n):
        A[j, j] = -rate
        if j:
            A[j, j-1] = rate
    A[n, n-1] = K/tau
    A[n, n] = -1/tau
    B = np.zeros((n+1, 2))
    B[0, 0] = rate
    B[n, 1] = 1/tau
    return A, B


def response(time, command, n, K, tau, theta, ambient=0., initial_temperature=0.):
    """Return temperature rise: initial lag stages zero, left-held input."""
    A, B = matrices(n, K, tau, theta)
    augmented = np.zeros((n+3, n+3))
    augmented[:n+1, :n+1] = A
    augmented[:n+1, n+1:] = B
    state = np.zeros(n+1)
    state[-1] = initial_temperature
    result = [state[-1]]
    propagators = {}
    for k, dt in enumerate(np.diff(time)):
        if dt <= 0:
            raise ValueError('Strictly increasing time required')
        h = float(dt)
        if h not in propagators:
            propagators[h] = expm(augmented*h)[:n+1]
        state = propagators[h]@np.r_[state, command[k], ambient]
        result.append(state[-1])
    return np.asarray(result)


def audit():
    from scipy.integrate import solve_ivp
    from scipy.special import gammainc
    rng = np.random.RandomState(20261025)
    t = np.r_[0., np.cumsum(rng.uniform(.1, 3., 20))]
    u = rng.uniform(0., 100., len(t))
    cases = []
    for n in (1, 2, 3):
        for tau, theta in [(137., 18.), (20., n*20.), (5., .3), (600., 200.)]:
            A, B = matrices(n, .71, tau, theta)
            state = np.zeros(n+1)
            expected = [0.]
            for k, h in enumerate(np.diff(t)):
                forcing = B@np.array([u[k], -1.3])
                sol = solve_ivp(lambda s, x: A@x+forcing, (0., h), state,
                                rtol=2e-11, atol=1e-12, method='DOP853')
                if not sol.success:
                    raise AssertionError(sol.message)
                state = sol.y[:, -1]
                expected.append(state[-1])
            actual = response(t, u, n, .71, tau, theta, ambient=-1.3)
            error = float(np.max(np.abs(actual-expected)))
            if error > 2e-8:
                raise AssertionError((n, tau, theta, error))
            cases.append({'n': n, 'tau': tau, 'theta': theta, 'ode_max_abs_C': error})
        regular = np.linspace(0., 250., 101)
        equal = response(regular, np.full(len(regular), 50.), n, .71, 20., n*20.)
        error = float(np.max(np.abs(equal-.71*50*gammainc(n+1, regular/20.))))
        if error > 2e-10:
            raise AssertionError(('Repeated-pole Erlang step', n, error))
        cases.append({'n': n, 'repeated_pole_step_max_abs_C': error})
    return {'status': 'passed', 'cases': cases,
            'scope': 'Orders1/2/3, irregular held commands, ambient forcing, repeated-pole analytic response and independent ODE'}
