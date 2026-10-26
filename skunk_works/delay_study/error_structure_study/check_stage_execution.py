"""Check actual sequential-stage success/failure receipts without fitting data."""
import argparse
import json
import os
from pathlib import Path
from stage_execution import run_stage
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    good=run_stage(args.output,'check_transfer_accounting.py',args.output/'success',[])
    assert json.loads((good/'check.json').read_text())['status']=='passed'
    assert json.loads((good/'exit.json').read_text())['exit_code']==0
    failed=args.output/'failure'
    try:run_stage(args.output,'check_transfer_accounting.py',failed,['--deliberately-invalid-option'])
    except RuntimeError:pass
    else:raise AssertionError('A failed stage was promoted')
    assert json.loads((failed/'exit.json').read_text())['exit_code']==2
    before=(good/'source_snapshot.json').read_bytes()
    try:run_stage(args.output,'check_transfer_accounting.py',good,[])
    except FileExistsError:pass
    else:raise AssertionError('Existing output accepted')
    assert (good/'source_snapshot.json').read_bytes()==before
    for directory in [good,failed]:
        process=json.loads((directory/'process.json').read_text())
        source=json.loads((directory/'source_snapshot.json').read_text())
        assert process['command']==source['command'] and process['parent_pid']==os.getpid()
        assert source['sequential_parent']==str(args.output)
        assert (directory/'run.log').exists()
    write(args.output/'check.json',{'status':'passed','actual_success_and_failure_exits':True,
        'existing_output_rejected_unchanged':True,'command_and_parent_source_bound':True,
        'notice':'Process-control and arithmetic fixtures only; no measured fit or transfer calibration.'})


if __name__=='__main__':main()
