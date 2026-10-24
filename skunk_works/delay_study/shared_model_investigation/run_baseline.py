"""Reconstruct the saved baseline without refitting or mutating prior results."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent/'residual_study'))
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from data import load
from diagnostics import input_correlation
from dynamics import response, audit
from run_mean import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output/'forward_audit.json', audit())
    paths = [args.baseline/p for p in ('fit.json', 'trajectories.npz', 'source_snapshot.json')]
    fit = json.loads(paths[0].read_text())
    saved = np.load(paths[1])
    binding = json.loads(paths[2].read_text())
    best = fit['best']
    K = best['coefficients'][0]
    tau, theta = best['tau'], best['theta']
    ell, eta = best['noise_parameters']
    rows = []
    quadratic, logdet = 0., 0.
    nobs = 0
    for j, name in enumerate(fit['record_names']):
        d = load(name)
        raw = HERE.parents[2]/d.source
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        if digest != binding['sha256'][d.source]:
            raise AssertionError('Raw input hash changed')
        if np.any(pd.read_csv(raw).Q2.values != 0):
            raise AssertionError('Nonzero Q2')
        prediction = d.temperature[0]+response(d.time, d.input, 1, K, tau, theta, best['coefficients'][1+j])
        error = float(np.max(np.abs(prediction-saved['predicted_'+str(j)])))
        residual = d.temperature[1:]-prediction[1:]
        t = d.time[1:]
        C = (1-eta)*np.exp(-np.abs(t[:, None]-t[None, :])/ell)+eta*np.eye(len(t))
        L = cholesky(C, lower=True)
        innovation = solve_triangular(L, residual, lower=True)
        standardized = innovation/best['sigma_total_ML_C']
        innovation_error = float(np.max(np.abs(standardized-saved['standardized_innovation_'+str(j)])))
        if error > 1e-7 or innovation_error > 2e-5:
            raise AssertionError((name, error, innovation_error))
        correlations = input_correlation(standardized, d.input[1:], 30)
        valid = [abs(r['correlation']) for r in correlations if r['correlation'] is not None]
        quadratic += float(innovation@innovation)
        logdet += float(2*np.log(np.diag(L)).sum())
        nobs += len(t)
        rows.append({'name': name, 'raw_sha256': digest, 'n': len(d.time),
                     'duration_s': float(d.time[-1]-d.time[0]),
                     'dt_range_s': [float(np.diff(d.time).min()), float(np.diff(d.time).max())],
                     'ambient_offset_C': best['coefficients'][1+j],
                     'mean_residual_C': float(residual.mean()),
                     'RMSE_C': float(np.sqrt(np.mean(residual**2))),
                     'max_abs_innovation_input_correlation_lags30': max(valid) if valid else None,
                     'prediction_max_abs_C': error, 'innovation_max_abs': innovation_error})
    nll = .5*(nobs*(np.log(2*np.pi)+1+np.log(quadratic/nobs))+logdet)
    if abs(nll-best['negative_log_likelihood']) > 2e-5:
        raise AssertionError('Independent likelihood differs')
    result = {'status': 'passed', 'baseline_bindings': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'baseline_producer_commit': binding['commit'], 'python': platform.python_version(),
              'nobs': nobs, 'negative_log_likelihood': nll,
              'nll_abs_error': abs(nll-best['negative_log_likelihood']),
              'pooled_RMSE_C': float(np.sqrt(sum((r['n']-1)*r['RMSE_C']**2 for r in rows)/nobs)),
              'per_run': rows,
              'interpretation': 'Independent numerical reproduction; not a model adequacy certificate'}
    write_json(args.output/'audit.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'nobs', 'pooled_RMSE_C', 'nll_abs_error')}), flush=True)


if __name__ == '__main__':
    main()
