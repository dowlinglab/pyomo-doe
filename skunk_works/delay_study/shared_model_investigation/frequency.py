"""Descriptive sinusoidal regression, continuous held-input reference, HAC errors."""
import numpy as np


def held_fundamental(time, command, omega, start, end):
    """L2 project the continuous left-held input onto 1,sin,cos over a window."""
    left = np.maximum(time[:-1], start)
    right = np.minimum(time[1:], end)
    keep = right > left
    left, right, u = left[keep], right[keep], command[:-1][keep]
    if abs(np.sum(right-left)-(end-start)) > 1e-7:
        raise ValueError('Window not covered by command intervals')
    moments = np.array([np.sum(u*(right-left)),
                        np.sum(u*(np.cos(omega*left)-np.cos(omega*right))/omega),
                        np.sum(u*(np.sin(omega*right)-np.sin(omega*left))/omega)])
    a, b, w = start, end, omega
    gram = np.empty((3, 3))
    gram[0, 0] = b-a
    gram[0, 1] = gram[1, 0] = (np.cos(w*a)-np.cos(w*b))/w
    gram[0, 2] = gram[2, 0] = (np.sin(w*b)-np.sin(w*a))/w
    gram[1, 1] = (b-a)/2-(np.sin(2*w*b)-np.sin(2*w*a))/(4*w)
    gram[2, 2] = (b-a)/2+(np.sin(2*w*b)-np.sin(2*w*a))/(4*w)
    gram[1, 2] = gram[2, 1] = (np.cos(2*w*a)-np.cos(2*w*b))/(4*w)
    return np.linalg.solve(gram, moments)


def held_harmonics(time,command,omega,start,end,order=3):
    """Joint continuous-time projection; avoid cross-harmonic window leakage."""
    basis=[{0:1.}]
    for k in range(1,order+1):
        basis += [{k:-.5j,-k:.5j},{k:.5,-k:.5}]
    def integral(k,a,b):
        return b-a if k==0 else (np.exp(1j*k*omega*b)-np.exp(1j*k*omega*a))/(1j*k*omega)
    G=np.array([[sum(c*d*integral(k+l,start,end) for k,c in f.items() for l,d in g.items()).real
                 for g in basis] for f in basis])
    left=np.maximum(time[:-1],start);right=np.minimum(time[1:],end);keep=right>left
    left,right,u=left[keep],right[keep],command[:-1][keep]
    if abs(np.sum(right-left)-(end-start))>1e-7:raise ValueError('Uncovered window')
    rhs=np.array([np.sum(u*sum(c*integral(k,left,right) for k,c in f.items())).real for f in basis])
    return np.linalg.solve(G,rhs)


def regression(time, values, period, nuisance='trend', tau=None, theta=None, lags=(30, 60, 120)):
    t = np.asarray(time)
    w = 2*np.pi/period
    X = np.column_stack((np.ones(len(t)), np.sin(w*t), np.cos(w*t)))
    if nuisance == 'trend':
        X = np.column_stack((X, (t-t.mean())/(t[-1]-t[0])))
    elif nuisance == 'reference_exponentials':
        # Explicitly conditional sensitivity; these rates came from pooled n=1.
        X = np.column_stack((X, np.exp(-(t-t[0])/tau), np.exp(-(t-t[0])/theta)))
    elif nuisance != 'constant':
        raise ValueError(nuisance)
    beta, _, rank, singular = np.linalg.lstsq(X, values, rcond=None)
    e = values-X@beta
    bread = np.linalg.pinv(X.T@X, rcond=1e-12)
    score = X*e[:, None]
    covariances = {}
    for lag in lags:
        S = score.T@score
        for k in range(1, min(lag, len(t)-1)+1):
            cross = score[k:].T@score[:-k]
            S += (1-k/(lag+1))*(cross+cross.T)
        V = bread@S@bread*len(t)/(len(t)-X.shape[1])
        covariances[str(lag)] = V[1:3, 1:3]
    return {'beta': beta, 'residual': e, 'rank': int(rank),
            'condition': float(singular[0]/singular[-1]), 'covariances': covariances,
            'nuisance_component': X@beta-X[:, 1:3]@beta[1:3]}


def response_summary(fit, input_coefficient):
    a, b = fit['beta'][1:3]
    amplitude = np.hypot(a, b)
    u = complex(input_coefficient[1], input_coefficient[2])
    H = complex(a, b)/u
    summaries = {}
    for lag, V in fit['covariances'].items():
        ga = np.array([a, b])/amplitude/abs(u)
        gp = np.array([-b, a])/amplitude**2
        summaries[lag] = {'gain_SD': float(np.sqrt(max(0., ga@V@ga))),
                          'phase_SD_deg': float(np.sqrt(max(0., gp@V@gp))*180/np.pi)}
    return {'temperature_amplitude_C': float(amplitude), 'input_amplitude_percent': float(abs(u)),
            'gain_C_per_percent': float(abs(H)), 'phase_deg': float(np.angle(H, deg=True)),
            'H_real': float(H.real), 'H_imag': float(H.imag), 'HAC': summaries}


def audit():
    from scipy.integrate import quad
    rng = np.random.RandomState(672)
    t = np.r_[0., np.cumsum(rng.uniform(.7, 1.3, 220))]
    u = rng.uniform(0, 100, len(t))
    w, start, end = 2*np.pi/30, 13.2, 192.7
    coefficient = held_fundamental(t, u, w, start, end)
    f = (lambda s: 1., lambda s: np.sin(w*s), lambda s: np.cos(w*s))
    G = np.array([[quad(lambda s: f[i](s)*f[j](s), start, end, epsabs=1e-10)[0]
                   for j in range(3)] for i in range(3)])
    m = np.zeros(3)
    for k in range(len(t)-1):
        a, b = max(start, t[k]), min(end, t[k+1])
        if b > a:
            for j in range(3):
                m[j] += u[k]*quad(f[j], a, b, epsabs=1e-11)[0]
    integration_error = float(np.max(np.abs(coefficient-np.linalg.solve(G, m))))
    regular = np.linspace(0, 700, 701)
    input_phase, lag, A, B = .3, -.8, 2.7, 50.
    y = 22+1.1*(regular-regular.mean())/700+A*np.sin(w*regular+input_phase+lag)
    fit = regression(regular, y, 30)
    summary = response_summary(fit, [50., B*np.cos(input_phase), B*np.sin(input_phase)])
    gain_error = abs(summary['gain_C_per_percent']-A/B)
    phase_error = abs(summary['phase_deg']-lag*180/np.pi)
    if integration_error > 1e-9 or gain_error > 1e-12 or phase_error > 1e-9:
        raise AssertionError((integration_error, gain_error, phase_error))
    # Verify HAC arithmetic against direct sums of outer products, independently
    # of the vectorized lag products used in regression().
    noisy = y+rng.normal(0, .2, len(y))
    fit = regression(regular, noisy, 30, lags=(7,))
    X = np.column_stack((np.ones(len(y)), np.sin(w*regular), np.cos(w*regular),
                         (regular-regular.mean())/700))
    score = X*fit['residual'][:, None]
    S = np.zeros((4, 4))
    for i in range(len(y)):
        S += np.outer(score[i], score[i])
        for k in range(1, min(7, i)+1):
            cross = np.outer(score[i], score[i-k])
            S += (1-k/8)*(cross+cross.T)
    bread = np.linalg.inv(X.T@X)
    V = bread@S@bread*len(y)/(len(y)-4)
    hac_error = float(np.max(np.abs(V[1:3, 1:3]-fit['covariances']['7'])))
    if hac_error > 1e-12:
        raise AssertionError(hac_error)
    return {'status': 'passed', 'held_input_quadrature_coefficient_error': integration_error,
            'synthetic_gain_abs_error': gain_error, 'synthetic_phase_abs_error_deg': phase_error,
            'HAC_direct_sum_max_abs': hac_error,
            'scope': 'Analytic ZOH projection vs independent quadrature, known-phase recovery with trend, HAC direct arithmetic'}
