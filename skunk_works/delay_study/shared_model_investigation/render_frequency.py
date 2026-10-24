"""Regenerate figures from saved estimates without rerunning inference."""
import argparse
import hashlib
import json
from pathlib import Path
from run_frequency import make_plots


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.input/'frequency_estimates.json').read_text())['rows']
    folded = json.loads((args.input/'phase_folds.json').read_text())['runs']
    make_plots(args.output, rows, folded)
    path = args.output/'figure_receipt.json'
    receipt = json.loads(path.read_text())
    receipt['input_bindings'] = {name: hashlib.sha256((args.input/name).read_bytes()).hexdigest()
                                 for name in ('frequency_estimates.json','phase_folds.json')}
    receipt['phase_display'] = 'Unwrap increasing frequency independently for each measured/model/run/window curve; principal values preserved in source estimates; three frequencies do not uniquely determine branch/delay.'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
