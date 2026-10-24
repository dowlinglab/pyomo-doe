# N=1 teaching study: execution plan

Alex authorized this goal on October 24, 2026. `../review/approach.md` owns the
agreed scope. This plan owns implementation choices; `PROJECT.md` and
`run_manifest.json` own execution state. Update files in place; Git keeps history.

## Scope and data

Use the twelve unique ENV2 recordings, accepting Alex's same-device assumption.
Fit the n=1 T1-only model with shared K, tau and theta, zero-order-held Q1,
run-specific ambient corrections and temporal covariance. Alpha is absorbed
into K; do not estimate it separately or add its uncertainty again to fitted K.
Exclude duplicate calibration aliases and the standalone non-ENV2 sine file.
`sin_50_50` denotes 30 seconds. Hash CSVs, record timestamps/quantization/Q2 and
check acquisition-code command alignment. Never modify raw data or old results.

| Tier | Goal | Numerical fitting data |
|---|---|---|
| 0 | Test a shared thermal model across ENV2; assess ambient drift and residual structure | All twelve unique records |
| 1 | Assess accuracy/uncertainty from one sine and one step, conditional on Tier 0 structure | P1 only, including refitted noise and ambient terms |
| 2 | Assess sensitivity to input period and particular repeated records | Five additional pairs, fitted separately |

Tier 0 inspects all records. Subsequent evaluation is retrospective conditional
on structural selection, even when the pair excludes ten records from numerical
fitting. The six overlapping pairs are not independent experimental replications.
No DoE, hardware, extra delay orders, four-state fits or remaining 21 pairs.
No artificial minimum elapsed duration applies to this new multi-day study.

## Mean model and technical defaults

    dz/dt = (u-z)/theta
    dT/dt = (-T + Tamb(t) + K*z)/tau

Use actual irregular timestamps. Condition on T(0)=first recorded T1=Tref,
set z(0)=0 (cooled startup), omit the first residual, and state that conditioning
in uncertainty claims. Tref is a sensor reading, not measured ambient. Assess
initial-reading/startup sensitivity before claiming general calibration.

Begin with Tamb=Tref+d0, d0 in [-2,2] C. Compare the nested linear trajectory
Tamb=Tref+d0+(d1-d0)*t/H. **Agent implementation choice:** bound both endpoints
d0,d1 in [-2,2] C, keeping the full trajectory in that band; disclose this choice
and assess sensitivity if bounds bind. Alex did not prescribe slope bounds.
Fitted offsets are nuisance terms, not independently measured room temperatures.

Use exact constant-input propagation and exact affine ambient forcing. Start
with left hold pending acquisition-code audit; compare right hold if alignment
cannot be settled. For fixed tau/theta, bounded linear least squares in K and
ambient terms permits variable projection. Cross-check against direct fitting.
Initial bounds: K=[0.2,1.5] C/command-percent, tau=[5,600] s,
theta=[0.01,200] s. Use 24 deterministic starts, including swapped pole seeds;
retain every start, bound hit and failure. Investigate bound-driven conclusions.

Exchanging poles preserves the cold zero-offset input response, but ambient
forcing enters through tau and may change equivalence. Retain both basins and
check equivalent responses under the new nuisance family. Never impose tau>=theta
or infer physical assignment merely from a labeling convention.

## Tier 0 sequence and gates

1. Audit membership/aliases, actual timestamps, command alignment and quantized
   temperatures. Verify exact mean propagation independently using matrix
   exponentials and numerical integration, including repeated poles, offsets,
   drift and both constant-hold conventions.
2. Fit pooled OLS under constant offsets, then linear ambient trajectories.
   Save per-run responses/residuals and all starts. Diagnose residual time,
   temperature/input dependence, startup, variance, quantization, ACF/PACF and
   command cross-correlation. Give approximate time-lag meaning for sample ACF.
3. Compare IID, stationary OU, OU plus independent measurement noise, and
   Matern-3/2 plus independent measurement noise, for both ambient treatments.
   Reset error state per run; retain stationary initial-state likelihood.
   Prefer O(N) state-space filters; independently check selected likelihoods
   against dense covariance calculations and boundary limits.
4. Demonstrate feasible GLS: estimate covariance, whiten full temporal residuals
   and design vectors, refit mean and repeat. Save iterations. Initial stopping
   rules: relative likelihood change <1e-7 and maximum scaled parameter change
   <1e-4 for two iterations; cap at 50. Stability is not scientific adequacy.
   Include Gaussian log determinant and normalization. Cross-check selected
   solutions with joint likelihood optimization and multiple starts.
5. Inspect innovations separately from raw residuals: per-run ACF, input
   association, variance/marginal plots, and simulation-based diagnostic
   envelopes with parameters refitted. AICc/likelihood support selection;
   persistent input-related bias cannot be repaired by declaring a kernel adequate.
6. Freeze the mean/covariance structure before restricted fits. If no candidate
   is adequate, carry a documented working model and its limitations forward;
   do not force whiteness or silently expand the thermal model. Optimization
   failure alone does not disprove a shared model.

The first eight-cell comparison left a repeatable 30-second innovation cycle
and command association in the fast-sine records. Before freezing the working
structure, prioritize the declared command-alignment sensitivity, startup and
ambient-bound checks, refit-calibrated diagnostics, and a direct-parameter
optimization check. Protocol-specific mean fits can diagnose the sharing
assumption within Tier 0 without adding calibration pairs. These are responses
to observed residual patterns; no new thermal order or extra pair enumeration
is authorized by this diagnostic refinement.

## Restricted fits, uncertainty and predictions

`config.json` fixes these agent-selected balanced assignments within Alex's
budget. Each step appears twice, each sine period twice, and six distinct sine
records give two uses of each sine repetition index.

| Pair | Tier | Sine period/run | Step run |
|---|---:|---|---:|
| P1 | 1 | 5 min / 1 | 1 |
| P2 | 2 | 30 s / 2 | 1 |
| P3 | 2 | 2 min / 1 | 2 |
| P4 | 2 | 5 min / 3 | 2 |
| P5 | 2 | 30 s / 3 | 3 |
| P6 | 2 | 2 min / 2 | 3 |

Refit all thermal, calibration-ambient and covariance parameters per pair under
the selected structure. Tier 0 values may initialize, but must not become fixed
noise values, extra observations or prior information. Save OLS, alternating GLS
and joint likelihood results for teaching. Count P1 once.

Compare IID/local curvature to covariance-aware curvature, scaled/log-coordinate
information eigendecompositions, selected nuisance-refitted likelihood profiles
and bootstrap intervals. State eigenvector scaling; retain competing minima.
Do not span unidentified label modes with a single Gaussian interval.

Initial bootstrap budget: 400 parametric datasets per pair with joint
thermal/ambient/noise refits, fixed saved seeds and explicit failure denominators.
Check batch stability and Monte Carlo limits. Use block-residual sensitivity
if innovations remain misspecified, choosing a block-length range from the
calibration diagnostics. This does not guarantee coverage. Pilot costs first;
independently verify generated covariance and selected refits.

Evaluate the other ten records per pair using only the declared first reading,
startup and commands. Separate mean-response bands from future-observation
bands. Unknown ambient gives a bounded scenario envelope; [-2,2] alone supplies
no probability law. Do not invent an ambient prior. Ambient estimates using full
evaluation temperatures are separately labeled conditional reconstructions,
never primary predictions. Report per-run/protocol RMSE, bias, coverage/width
and correlated exceedance patterns with all conditioning qualifications.

## Source binding, recovery and delivery

Use `../environment.json` and `/Users/adowling/opt/anaconda3/bin/python`.
At most four long numerical processes, each BLAS/OpenMP thread count one;
begin with one until cost/stability are measured. Commit code before measured
runs. Run immutable copies of committed sources/raw data with configuration,
runtime and source hashes. Preserve old outputs. Distinguish mutable pilots
from source-bound measurements. Write atomic terminal outputs and failures.

Before recovery read this plan, PROJECT and manifest; reconcile PIDs, process
identity, lock files and terminal artifacts before retrying. Reuse the existing
hourly thread heartbeat. Advance substantive work rather than idle to fill time.
Keep state current, avoid diaries and superseded drafts. Commit coherent changes
with `Co-Authored-By: Codex <noreply@openai.com>`, review diffs, and push completed
work on reformulation-exploration. No amend or force push.

Prepare a concise LaTeX/PDF for Alex: opening tier/goal/status/result-or-open-issue
table, Tier 0/1/2 results covering all six pairs, and questions before Tier 3.
Distinguish completed calculations from residual-model adequacy. Generate report
values from saved artifacts, audit numerical bindings and primary-source claims,
preserve literature-access limitations, and apply no-ai-slop editing. Lab
Literature use is already authorized for this session.

US Letter, left margin 0.25 in, right margin 3.25 in, body width 5 in. Agent layout
choices: 0.6-in top/bottom margins; light right-margin rules every 9/32 in on every
page. Keep body/figures/tables/page furniture clear of writing space. Compile,
verify exact geometry and inspect every rendered page. Save TeX/PDF and checks.

After Tier 2 and verification, push and send the finished PDF through Gmail to
adowling@nd.edu exactly once. Reconcile delivery intent/receipt and Sent before
sending or retrying uncertain outcomes. Verify recipient and attachment filename,
size and message readback; save a hash-bound receipt. Do not resend October 21
reports. Mark the goal complete and pause the heartbeat after every delivery
gate succeeds. Wait for Alex's review before Tier 3.
