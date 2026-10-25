#!/usr/bin/env python3
"""Audit unchanged CSV temperature grids; no model fitting or hardware access.

Run with the pinned Python and --output pointing to a new directory.
Only a documented candidate conversion is tested; it does not identify firmware.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[3]
ENV_CONFIG = ROOT / "skunk_works/delay_study/residual_study/config.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_check(values, step, offset):
    levels = np.arange(1024)
    grid = step * levels + offset
    # Two-decimal serial/CSV reporting; no claim about a particular half-tie rule.
    reported = np.round(grid, 2)
    distances = np.abs(values[:, None] - reported[None, :])
    nearest = np.argmin(distances, axis=1)
    error = distances[np.arange(len(values)), nearest]
    unrounded = grid[nearest]
    rounding_distance = np.abs(values - unrounded)
    return {
        "step_C": float(step), "offset_C": float(offset),
        "samples": len(values), "matches_to_1e_minus_8_C": int(np.sum(error < 1e-8)),
        "max_reported_error_C": float(np.max(error)),
        "rms_reported_error_C": float(np.sqrt(np.mean(error**2))),
        "rounding_bin_compatibility_tolerance_C": 1e-6,
        "rounding_bin_compatible_samples": int(np.sum(rounding_distance <= .005 + 1e-6)),
        "max_rounding_distance_C": float(np.max(rounding_distance)),
        "inferred_code_range": [int(nearest.min()), int(nearest.max())],
    }


def describe(values, text_values):
    unique = np.unique(values)
    increments = np.round(np.diff(unique), 8)
    temporal = np.round(np.diff(values), 8)
    plateau_lengths, count = [], 1
    for delta in temporal:
        if delta == 0:
            count += 1
        else:
            plateau_lengths.append(count)
            count = 1
    plateau_lengths.append(count)
    codes = np.rint((values + 50) / .3223).astype(int)
    raw_error = values - (.3223 * codes - 50)
    return {
        "samples": len(values), "unique_temperatures": len(unique),
        "range_C": [float(unique[0]), float(unique[-1])],
        "distinct_increment_counts_C": {str(k): v for k, v in sorted(Counter(increments).items())},
        "temporal_increment_counts_C": {str(k): v for k, v in sorted(Counter(temporal).items())},
        "fraction_unchanged_consecutive": float(np.mean(temporal == 0)),
        "longest_plateau_samples": max(plateau_lengths),
        "decimal_places_in_CSV_counts": dict(Counter(len(v.split('.')[1]) if '.' in v else 0 for v in text_values)),
        "candidate_rounding_error_range_C": [float(raw_error.min()), float(raw_error.max())],
        "candidates": {
            "firmware_coefficient": grid_check(values, .3223, -50),
            "exact_3V3_10bit_10mV_per_C": grid_check(values, 330/1024, -50),
            "alternative_endpoint_1023": grid_check(values, 330/1023, -50),
            "5V_10bit_10mV_per_C": grid_check(values, 500/1024, -50),
            "emulator_zero_offset_grid": grid_check(values, .3223, 0),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(ENV_CONFIG.read_text())
    env_names = set(config['records'])
    selected = sorted((ROOT / 'data').glob('*.csv'))
    files, raw_hashes, env_values, all_values = [], {}, [], []
    for path in selected:
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not {'T1', 'T2'} <= set(rows[0]):
            continue
        sha = digest(path)
        raw_hashes[str(path.relative_to(ROOT))] = sha
        channels = {}
        for channel in ['T1', 'T2']:
            texts = [row[channel] for row in rows]
            values = np.asarray([float(v) for v in texts])
            channels[channel] = describe(values, texts)
            all_values.append(values)
            if path.name in env_names:
                env_values.append(values)
        provenance = subprocess.check_output(
            ['git', 'log', '-1', '--format=%H%n%an%n%aI%n%s', '--', str(path.relative_to(ROOT))],
            cwd=ROOT, text=True).splitlines()
        files.append({'path': str(path.relative_to(ROOT)), 'sha256': sha,
                      'primary_ENV2': path.name in env_names, 'last_git_commit': provenance,
                      'channels': channels})
    combined = np.concatenate(env_values)
    unique = np.unique(combined)
    n = np.rint((unique + 50) / .3223).astype(int)
    # Bounds assume rounding to nearest .01C and code assignment from candidate.
    lower = (unique - .005 + 50) / n
    upper = (unique + .005 + 50) / n
    # Free offset: slope bounds from all rounded levels with same code labels.
    matrix = np.column_stack([n, np.ones(len(n))])
    lp_bounds = [(0, None), (None, None)]
    lo = linprog([1, 0], A_ub=np.vstack([matrix, -matrix]),
                 b_ub=np.r_[unique + .005, -(unique - .005)], bounds=lp_bounds, method='highs')
    hi = linprog([-1, 0], A_ub=np.vstack([matrix, -matrix]),
                 b_ub=np.r_[unique + .005, -(unique - .005)], bounds=lp_bounds, method='highs')
    assert lo.success and hi.success
    table = []
    for value, code in zip(unique, n):
        table.append({'temperature_C': float(value), 'candidate_ADC_code': int(code),
                      'unrounded_firmware_value_C': float(code*.3223-50),
                      'firmware_reported_C': float(np.round(code*.3223-50, 2))})
    summary = {
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'audit_source_sha256': digest(Path(__file__)), 'config_sha256': digest(ENV_CONFIG),
        'scope': 'Both channels, twelve distinct ENV2 records; three other CSV paths separately described. Raw files unchanged.',
        'ENV2_files': sum(f['primary_ENV2'] for f in files),
        'ENV2_observations_both_channels': len(combined), 'ENV2_unique_levels_both_channels': len(unique),
        'ENV2_pooled': describe(combined, [str(v) for v in combined]),
        'ENV2_candidate_slope_interval_offset_fixed_minus50': [float(lower.max()), float(upper.min())],
        'ENV2_candidate_slope_interval_offset_free': [float(lo.x[0]), float(hi.x[0])],
        'all_paths_pooled_candidates': {k: grid_check(np.concatenate(all_values), step, offset)
            for k, step, offset in [('firmware',.3223,-50), ('emulator',.3223,0), ('exact_3V3',330/1024,-50)]},
        'duplicate_raw_hashes': [[f['path'] for f in files if f['sha256'] == sha]
                                 for sha in set(raw_hashes.values()) if list(raw_hashes.values()).count(sha)>1],
        'files': files,
        'interpretation_limits': [
            'A rounded temperature grid does not establish simulator or hardware provenance.',
            'ADC code labels are inferred from a candidate law, not logged codes.',
            'Slope intervals condition on candidate code labels; they do not prove lattice uniqueness against finer submultiple grids.',
            'Exact np.round matches assume float64 ties-to-even; bin compatibility allows 1e-6 C for firmware arithmetic/half-bin differences.',
            'Rounding-bin compatibility is not a calibration or accuracy measurement.',
            'Analogue noise and dynamics can both cause code transitions; no analogue noise variance is estimated.',
            'Pooled temporal statistics concatenate runs/channels; use per-file channel values for time structure.',
        ],
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (args.output / 'ENV2_levels.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    assert raw_hashes == {name: digest(ROOT/name) for name in raw_hashes}
    print(json.dumps({k: summary[k] for k in ['ENV2_files','ENV2_observations_both_channels',
          'ENV2_unique_levels_both_channels','ENV2_candidate_slope_interval_offset_fixed_minus50',
          'ENV2_candidate_slope_interval_offset_free','all_paths_pooled_candidates']}, indent=2))


if __name__ == '__main__':
    main()
