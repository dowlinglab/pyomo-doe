"""Read-only final evidence and preservation audit, separate from inference.

Requires a fresh direct-tool process reconciliation. Does not perform network,
email, hardware work or infer calibrated uncertainty from successful jobs.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
BASE='346a03d97a4a20d4b60423d18a574dec2b087f7d'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    bindings={}
    def read(path):
        bindings[str(path.relative_to(REPO))]=sha(path)
        return json.loads(path.read_text())
    manifest=read(HERE/'run_manifest.json');snapshot=read(HERE/'process_check.json')
    checked=datetime.strptime(snapshot['checked_utc'],'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    age=(datetime.now(timezone.utc)-checked).total_seconds();assert -5<=age<=180,age
    assert not snapshot['processes'] and snapshot['exit_code']==1
    assert not manifest['live_jobs'] and not manifest['live_worker_slots']
    assert not any(manifest['reconciliation_warnings'].values())
    assert all(r['status']=='terminal' for r in manifest['jobs'])
    failures=[];failure_logs={}
    for row in manifest['jobs']:
        d=REPO/row['output'];exit=read(d/'exit.json');source=read(d/'source_snapshot.json')
        assert source['commit']==row['source_commit'] and exit['exit_code']==row['exit_code']
        if exit['exit_code']!=0:
            log=d/'run.log'
            if log.exists():failure_logs[row['id']]=log
            failures.append({'id':row['id'],'exit_code':exit['exit_code'],'source_commit':source['commit'],
                'finished_utc':exit.get('finished_utc'),'directory':row['output']})
    # Preserve all prior studies and raw data, rather than only checking the
    # twelve selected CSV names. The one parent status file is allowed to move.
    changed=subprocess.check_output(['git','diff','--name-only',BASE],cwd=REPO,text=True).splitlines()
    prefix=str(HERE.relative_to(REPO))+'/'
    unexpected=[p for p in changed if not(p.startswith(prefix) or p=='skunk_works/delay_study/PROJECT.md')]
    assert not unexpected,unexpected
    prior_paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'--','data','skunk_works/delay_study'],cwd=REPO,text=True).splitlines()
    preserved=[]
    for name in prior_paths:
        if name.startswith(prefix) or name=='skunk_works/delay_study/PROJECT.md':continue
        path=REPO/name
        content=subprocess.check_output(['git','show',BASE+':'+name],cwd=REPO)
        digest=hashlib.sha256(content).hexdigest();assert sha(path)==digest,name
        preserved.append({'path':name,'sha256':digest})
    report=HERE/'report';numeric=read(report/'final_numeric_audit.json');layout=read(report/'layout_audit.json');visual=read(report/'visual_review.json')
    assert numeric['status']==layout['status']=='passed' and visual['status']=='reviewed'
    assert numeric['source_sha256']==layout['source_sha256']==visual['source_sha256']==sha(report/'main.tex')
    assert numeric['markdown_source_sha256']==sha(HERE/'report.md')
    assert layout['pdf_sha256']==visual['pdf_sha256']==sha(report/'main.pdf')
    assert len(layout['per_page'])==len(visual['pages_inspected'])==len(visual['render_sha256'])==11
    assert not visual['unresolved_citations_or_references'] and not visual['empty_tables_or_missing_figures']
    for name,digest in numeric['source_bindings_sha256'].items():
        path=HERE.parent/name;assert sha(path)==digest,str(path)
        bindings[str(path.relative_to(REPO))]=digest
    for path in [report/'main.tex',report/'main.pdf',HERE/'report.md']:
        bindings[str(path.relative_to(REPO))]=sha(path)
    for name in ['transfer_final_summary','startup_profile_summary','recovery_b10_full50_summary','local_taylor_quadrature_followup']:
        directory=HERE/'results'/name;exit=read(directory/'exit.json');summary=read(directory/'summary.json')
        assert exit['exit_code']==0
        if name=='transfer_final_summary':
            assert summary['status']=='complete' and summary['all_declared_folds_accounted'] and summary['all_required_uncertainty_components_accounted']
        if name=='startup_profile_summary':assert summary['accepted_points']==5 and summary['declared_points']==6 and not summary['baseline_refinement_required']
        if name=='recovery_b10_full50_summary':assert summary['status']=='complete' and summary['coverage']['covered']==49
        if name=='local_taylor_quadrature_followup':assert summary['status']=='completed_with_unresolved_quadrature'
    args.output.mkdir(exist_ok=False)
    (args.output/'failure_logs').mkdir()
    for row in failures:
        if row['id'] in failure_logs:
            original=failure_logs[row['id']];target=args.output/'failure_logs'/(row['id'].replace('/','__')+'.txt')
            target.write_bytes(original.read_bytes())
            row['original_run_log_sha256']=sha(original);row['preserved_log']=str(target.relative_to(args.output))
    failed_tasks=[]
    for task in sorted((HERE/'results').rglob('task_exit.json')):
        value=json.loads(task.read_text())
        if not value.get('success',True):
            failed_tasks.append({'path':str(task.relative_to(REPO)),'sha256':sha(task),'result':value})
    (args.output/'failed_tasks.json').write_text(json.dumps(failed_tasks,indent=2)+'\n')
    (args.output/'preserved_files.json').write_text(json.dumps(preserved,indent=2)+'\n')
    (args.output/'terminal_nonzero_exits.json').write_text(json.dumps(failures,indent=2)+'\n')
    output={'status':'passed','checked_utc':datetime.now(timezone.utc).isoformat(),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'checker_source_sha256':sha(Path(__file__)),'starting_plan_commit':BASE,
        'terminal_jobs':len(manifest['jobs']),'nonzero_exits_retained':len(failures),'failed_task_receipts_retained':len(failed_tasks),'live_jobs':0,
        'preserved_prior_files':len(preserved),'changed_paths_outside_allowed_scope':unexpected,
        'preserved_inventory_sha256':sha(args.output/'preserved_files.json'),
        'nonzero_exit_inventory_sha256':sha(args.output/'terminal_nonzero_exits.json'),
        'source_bindings_sha256':bindings,
        'scope':'Read-only source/report/result and prior-file preservation checks. Terminal failures are retained. Does not assert physical identification, calibrated uncertainty, remote push or heartbeat/goal completion.'}
    (args.output/'audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='source_bindings_sha256'}))


if __name__=='__main__':main()
