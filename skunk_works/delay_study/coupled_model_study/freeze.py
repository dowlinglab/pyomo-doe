"""Run committed investigation code with pinned dependencies and frozen raw data."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    if len(sys.argv) < 4 or Path(sys.argv[1]).name != sys.argv[1] or '--output' not in sys.argv:
        raise SystemExit('Usage: freeze.py DRIVER.py --output ABSOLUTE_DIRECTORY [arguments]')
    driver, args = sys.argv[1], sys.argv[2:]
    output = Path(args[args.index('--output')+1])
    if not output.is_absolute():
        raise SystemExit('Output must be absolute')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    prefix = HERE.relative_to(ROOT)
    parent = prefix.parent
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision,
                                      '--', str(parent), 'data'], cwd=ROOT, text=True).splitlines()
    selected = [p for p in tracked if (
        (Path(p).parent in {prefix, parent/'residual_study', parent/'shared_model_investigation', parent} and p.endswith('.py')) or
        (Path(p).parent in {prefix, parent/'residual_study'} and Path(p).name in {'config.json', 'structure_selection.json'}) or
        p == str(parent/'environment.json') or (Path(p).parent == Path('data') and p.endswith('.csv')))]
    if str(prefix/driver) not in selected:
        raise SystemExit('Commit driver before running measurements')
    expected = json.loads(subprocess.check_output(['git', 'show', revision+':'+str(parent/'environment.json')], cwd=ROOT))
    packages = {p: importlib.metadata.version(p) for p in expected['packages']}
    if packages != expected['packages'] or platform.python_version() != expected['python']:
        raise SystemExit('Runtime differs from pinned environment')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = Path(tempfile.mkdtemp(prefix='tclab-coupled-'+revision[:8]+'-'))
    hashes = {}
    for name in selected:
        content = subprocess.check_output(['git', 'show', revision+':'+name], cwd=ROOT)
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(0o444)
        hashes[name] = hashlib.sha256(content).hexdigest()
    command = [sys.executable, str(snapshot/prefix/driver)]+args
    receipt = {'commit': revision, 'sha256': hashes, 'packages': packages,
               'python': platform.python_version(), 'snapshot': str(snapshot), 'command': command}
    (output/'source_snapshot.json').write_text(json.dumps(receipt, indent=2)+'\n')
    env = dict(os.environ)
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = '1'
    env['MPLBACKEND'] = 'Agg'
    env['MPLCONFIGDIR'] = str(snapshot/'runtime_cache'/'matplotlib')
    env['XDG_CACHE_HOME'] = str(snapshot/'runtime_cache')
    with (output/'run.log').open('w') as log:
        child = subprocess.Popen(command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT)
        (output/'process.json').write_text(json.dumps({'pid': child.pid, 'parent_pid': os.getpid(), 'command': command}, indent=2)+'\n')
        code = child.wait()
    (output/'exit.json').write_text(json.dumps({'exit_code': code})+'\n')
    print(json.dumps({'output': str(output), 'exit_code': code}), flush=True)
    print((output/'run.log').read_text()[-5000:], flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
