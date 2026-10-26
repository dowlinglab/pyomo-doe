"""Execute sequential stages from one immutable frozen source tree.

A queue and its one child occupy one numerical worker slot. Each child has an
actual process/exit receipt and its own log; the parent never manufactures a
successful stage from an exception or reuses an existing output directory.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from run_bounded import write

HERE=Path(__file__).resolve().parent


def run_stage(parent,driver,destination,arguments):
    parent=Path(parent);destination=Path(destination)
    assert Path(driver).name==driver and driver.endswith('.py')
    assert destination.is_absolute() and parent.resolve() in destination.resolve().parents
    receipt=json.loads((parent/'source_snapshot.json').read_text())
    source=HERE/driver
    candidates=[(k,v) for k,v in receipt['sha256'].items() if Path(k).name==driver and Path(k).parent.name==HERE.name]
    assert len(candidates)==1 and source.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==candidates[0][1]
    assert '--output' not in arguments
    destination.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(source),'--output',str(destination)]+[str(v) for v in arguments]
    receipt.update({'command':command,'sequential_parent':str(parent),
        'sequential_parent_source_sha256':hashlib.sha256((parent/'source_snapshot.json').read_bytes()).hexdigest()})
    write(destination/'source_snapshot.json',receipt)
    with (destination/'run.log').open('w') as stream:
        child=subprocess.Popen(command,cwd=Path(receipt['snapshot']),stdout=stream,stderr=subprocess.STDOUT)
        write(destination/'process.json',{'pid':child.pid,'parent_pid':os.getpid(),
            'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),
            'sequential_parent':str(parent)})
        code=child.wait()
    write(destination/'exit.json',{'exit_code':code,'finished_utc':datetime.now(timezone.utc).isoformat()})
    if code:raise RuntimeError('Sequential stage exited '+str(code)+': '+str(destination))
    return destination
