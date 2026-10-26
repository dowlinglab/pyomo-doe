"""Check weighted transfer summaries and prevent cross-fold double counting."""
import argparse
import copy
import math
from summarize_transfer import aggregate_groups,completion_state
from run_bounded import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();from pathlib import Path
    a={'name':'r1','nobs_per_channel':1,'latent_SSE_by_channel_C2':[1.,4.],
        'reported_SSE_by_channel_C2':[1.,4.],'reported_signed_error_sum_C':[1.,2.],
        'reported_run_offset_SSE_C2':[1.,4.],
        'noise_only_covered_counts':[0,1],'noise_only_width_sum_C':[1.,2.]}
    b={'name':'r2','nobs_per_channel':3,'latent_SSE_by_channel_C2':[27.,48.],
        'reported_SSE_by_channel_C2':[27.,48.],'reported_signed_error_sum_C':[9.,12.],
        'reported_run_offset_SSE_C2':[27.,48.],
        'noise_only_covered_counts':[3,1],'noise_only_width_sum_C':[12.,15.]}
    rows=[{'candidate':'primary','fold_type':'leave_run_out','status':'audited','per_run':[a]},
          {'candidate':'primary','fold_type':'leave_run_out','status':'audited','per_run':[b]},
          {'candidate':'primary','fold_type':'leave_protocol_out','status':'audited','per_run':[a,b]},
          {'candidate':'shared','fold_type':'leave_protocol_out','status':'failed'}]
    expected={('primary','r1'):{'candidate':'primary','type':'leave_run_out'},
              ('primary','r2'):{'candidate':'primary','type':'leave_run_out'},
              ('primary','p'):{'candidate':'primary','type':'leave_protocol_out'},
              ('shared','p'):{'candidate':'shared','type':'leave_protocol_out'}}
    result=aggregate_groups(rows,expected,['r1','r2'])
    for row in result[:2]:
        assert row['nobs_per_channel']==4 and row['unique_test_records']==2
        assert max(abs(a-b) for a,b in zip(row['reported_RMSE_by_channel_C'],[math.sqrt(7),math.sqrt(13)]))<1e-12
        assert row['noise_only_marginal_coverage']==[.75,.5]
        assert row['noise_only_mean_width_C']==[3.25,4.25]
        assert row['reported_run_offset_SSE_fraction']==[1.,1.] and row['reported_per_run_centered_RMSE_C']==[0.,0.]
    assert result[2]['declared_folds']==1 and result[2]['audited_folds']==0 and result[2]['nobs_per_channel']==0
    duplicate=copy.deepcopy(rows);duplicate[1]['per_run']=[a];rejected=False
    try:aggregate_groups(duplicate,expected,['r1','r2'])
    except AssertionError:rejected=True
    assert rejected
    noise_only=[{'status':'audited','required_uncertainty_complete':False} for _ in range(20)]
    assert completion_state(noise_only,20)['status']=='partial'
    complete=[dict(r,required_uncertainty_complete=True) for r in noise_only]
    assert completion_state(complete,20)['status']=='complete'
    assert completion_state(complete[:-1],20)['status']=='partial'
    complete[-1]={'status':'failed'}
    assert completion_state(complete,20)['status']=='complete_with_failures'
    write(Path(args.output)/'check.json',{'status':'passed','groups':result,
        'weighted_SSE_not_mean_RMSE':True,'record_reuse_between_groups_kept_separate':True,
        'within_group_duplicate_rejected':True,'failed_fold_preserved':True,
        'twenty_noise_only_folds_cannot_be_complete':True,'missing_fold_cannot_be_complete':True,
        'notice':'Generated arithmetic fixtures, not measured transfer performance or predictive calibration.'})


if __name__=='__main__':main()
