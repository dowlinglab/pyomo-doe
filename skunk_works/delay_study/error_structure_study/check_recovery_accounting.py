"""Failure-denominator semantics, binomial tails and exact free-fit selection."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from scipy.optimize import brentq
from scipy.stats import binom
from audit_recovery import interval,free_output
from audit_provenance import numerical_binding
from summarize_recovery import coverage
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draw',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args()
    rows=[{'status':'passed','coverage_complete':True,'covered':True},
          {'status':'passed','coverage_complete':True,'covered':False},
          {'status':'passed','coverage_complete':False},
          {'status':'failed','coverage_complete':True,'covered':True}]
    value=coverage(rows,5)
    assert value['complete']==2 and value['covered']==1 and value['missing']==1
    assert value['failed_or_incomplete_audited']==2 and value['coverage_among_complete']==.5
    assert value['failure_worst_case_bounds_all_declared']==[.2,.8]
    checks=[]
    for k,n in [(0,50),(1,50),(25,50),(49,50),(50,50),(1,2)]:
        expected=[brentq(lambda p:binom.sf(k-1,n,p)-.025,0.,1.,xtol=1e-14) if k else 0.,
                  brentq(lambda p:binom.cdf(k,n,p)-.025,0.,1.,xtol=1e-14) if k<n else 1.]
        actual=interval(k,n);error=max(abs(a-b) for a,b in zip(actual,expected));assert error<1e-11
        checks.append({'covered':k,'complete':n,'beta_quantile_vs_binomial_tail_root_error':error})
    row=json.loads((args.draw/'recovery.json').read_text());original=free_output(row)
    older=dict(row);older.pop('selected_free_output',None)
    assert free_output(older)==original
    invalid=copy.deepcopy(older);invalid['recovered_point'][0]+=.123;rejected=False
    try:free_output(invalid)
    except AssertionError:rejected=True
    assert rejected
    before=numerical_binding(args.source)
    with tempfile.TemporaryDirectory(prefix='tclab-audit-binding-fixture-') as temporary:
        p=Path(temporary);source=json.loads((args.source/'source_snapshot.json').read_text())
        key=next(k for k in source['sha256'] if Path(k).name=='quantized.py')
        source['sha256'][key]='0'*64;(p/'source_snapshot.json').write_text(json.dumps(source))
        assert numerical_binding(p)!=before
    write(args.output/'check.json',{'status':'passed','failure_and_missing_denominators':value,
        'independent_binomial_tail_checks':checks,'exact_saved_free_point_selection':True,
        'altered_recovered_point_rejected':True,'quantized_dependency_drift_detected':True,
        'notice':'Generated accounting fixtures and a read-only existing fitted draw. No optimizer run, synthetic coverage sweep or hardware model-adequacy claim.'})


if __name__=='__main__':main()
