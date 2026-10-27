# TC-Lab remaining-error study

Alex authorized this sustained investigation on October 25, 2026. The plan is
[the shared discussion note](../review/next_investigation_discussion.md), committed
before goal launch as 346a03d. The goal is active. Its ten-plus-hour research
scope has flexible timing and no idle minimum. No hardware, DoE or email is
part of this goal.

Read run_manifest.json and a fresh direct-tool process snapshot before dispatch.
Only terminal audited results support report claims. All twelve ENV2 records
informed structure selection; transfer is retrospective.

## Current checkpoint: October 27, 02:17 UTC

The full fifty-draw recovery and independent summary audit passed: gain coverage
49/50, exact-binomial MC interval89.35--99.95%, fifteen lag-label exchanges.
All four selected hardware input/phase/T2-ACF/cross statistics exceed all fifty
refitted references. The eight-page WIP PDF and Markdown now contain these
results; numeric checks bind295 files, all eight pixels/geometry were reviewed.
Final pagination and remaining inference are still pending.

All initial/bin cases passed selected-point audits. In reported-code space,
freely initialized floor/nearest fits differ below0.00030C. Their changed
analogue-reference RMSE is not predictive improvement. Hidden +/-2/5C fits
coincide (maximum0.514C) and reverse the lag labels, triggering the declared
six-point startup profile. Its independent score/propagation control passed.

Four full transfer components are indexed and reviewed: primary30/300/step
and shared30. Primary120 extra loading starts improve nll, but the old scalar
curvature stencil needs a saved convergence study at the larger loading.
Shared120 has confirmed negative weak-loading curvature and needs the existing
additional-start policy. Both original failed audits remain. No covariance
was accepted on those failures. Shared300/step propagation remains undispatched.

Four long slots are allocated: remaining radiation profile, startup profile,
primary runs1--6 and runs7--12. Reconcile exact identities before any dispatch.
Next free slot: committed diagnose_loading_stencils.py on primary refined120,
then consider a justified auditor stencil update without relaxing the .001 gate;
use another slot for shared120 additional starts. Consult run_manifest for
actual receipts and sessions rather than historical pending prose below.

## Audited scientific results

The old coupled short-lag/curved-command fit has T1 RMSE 1.022 C. Orthogonal
projections assign 74.6% of T1 SSE to run offsets, another 19.3% to trends and
0.64% to the fundamental. These projections are not independent physical causes.

Matched Gaussian constant-ambient conductance bands give T1/T2 RMSE:
5% = 0.646/0.147 C; 10% = 0.639/0.115 C; 20% = 0.641/0.115 C.
Two 10% deviations reach the lower bound. The 5% fit retains three failed starts.
Affine ambient gives 0.728/0.111 C. Hidden heater starts bounded at +/-2 C give
0.615/0.111 C; +/-5 C gives the same result, every selected offset below 0.35 C.
Cold starts remain plausible, not verified equilibrium.

Radiation calculations use explicit emissivity/area/surface-temperature
scenarios. Metal does not determine emissivity. A linearized contribution can
already be in Ua. The Gaussian free-radiation fit gives 0.633/0.115 C, gamma
0.0344 and nll improvement 1.285. The linear-command radiation fit is worse.
All dimensional ODE, nesting, step-size and dense likelihood audits passed.

Joint physical-plus-quantized fits now have independent trajectory, Gaussian
observation-rectangle, corrected-domain, 128/256/512-node, tail-12/16 and
stationarity audits:

| Case | Audited export | T1/T2 RMSE, C | Bin nll |
| --- | --- | --- | --- |
| Shared, no run deviations | joint_conductance_b00_export_v3 | 1.010 / 0.239 | 6359.91818 |
| 5% conductance band | joint_conductance_b05_export_v2 | 0.637 / 0.153 | 6279.42504 |
| 10% conductance band | joint_conductance_b10_export_v2 | 0.605 / 0.118 | 6238.29865 |
| 20% conductance band | joint_conductance_b20_export_v2 | 0.606 / 0.118 | 6238.21668 |
| Radiation, 10% convective band | joint_radiation_b10_export_v2 | 0.590 / 0.119 | 6234.48899 |

The 5% joint fit and selected-point audit are terminal and passed.
A wider band cannot be selected from training
likelihood alone. Radiation's bin-likelihood improvement is 3.810, larger than
in the Gaussian approximation, but its boundary/nuisance profile and transfer
remain necessary. Convective and effective total loss are different quantities.

The 10% joint gain is 0.87322 C/%, within 0.1% of its Gaussian value; tA=174.586 s,
rS=14.790 s, input L=2.730 s and command curvature=0.12182. Loading Us/Ua reaches
0.005. One run conductance reaches 0.9. Its analogue latent SD / length /
measurement SD are T1 = 0.53621 C / 658.28 s / 0.02575 C and
T2 = 0.06813 C / 62.56 s / 0.03248 C. These are working noise parameters, not
sensor calibration measurements or final uncertainty claims.

## Whitening evidence and remaining checks

All 21,614 raw hardware temperatures match round(0.3223*n - 50, 2). This is
not just simulator quantization. Firmware thresholds and pre-quantization noise
are not established by the grid.

joint_b10_predictive_diagnostics is terminal: twenty independent PIT streams
and fifty fixed-model simulations, no failed simulation. Median maximum pooled
ACF over lags 1..60 is 0.02481 for T1 and 0.02454 for T2; median maximum
cross-channel correlation over lags -10..10 is 0.02235, comparable to references.
T1 nonrandomized transition discrepancy is 2.3356 and maximum phase projection
fraction has median 0.02029; both have upper-tail simulation rank 0.0588.
The plot was inspected. These are fixed-parameter references, not refitted
diagnostic p-values, model-selection adjustment or proof of IID errors.

The deterministic check joint_b10_nonrandom_diagnostics is now terminal, with
all fifty saved synthetic observation collections and prefix CDF audits passed.
It exposes remaining structure: T1 input/score correlation 0.04429 (reference
97.5% quantile 0.02261), T1 maximum phase projection 0.02983 (0.01971), T2
maximum ACF 0.04695 (0.04039), and cross-channel maximum 0.04895 (0.02540).
All four have descriptive upper-tail rank 1/51. No IID claim. The fixed-model
references do not refit parameters or repeat selection. Its figure was inspected.
Code-score SDs are not 1 and are not supposed to be Gaussian. Randomized PIT
jitter can dilute their structure. Cross localization v2 applies matched run
centering, trends and phase projections to data and references. A constant-input
correlation NaN caused the first localization export to fail; explicit null
repairs it and the failed source/exit are retained. Earlier marginal pilots
reused seeds and are unsuitable for joint cross-channel/run testing.

Sequential Gaussian GLS from an IID mean converged in nine cycles, all block
solvers successful; nll differs from the saved joint fit by 1.29e-7. The
saved-joint initialization is also terminal. Both independent full-data
dense/trajectory and trace audits passed. The initial IID mean stage holds
the relative channel SD fixed; later covariance blocks estimate it. This is
not an initial joint IID variance-component MLE. See sequential_methods.md.
The short synthetic dense nll/score and constrained-mean checks passed.

## Identifiability and initial conditions

A synthetic limit check verifies that sensor/input lag labels exchange exactly
in zero-loading forced transfers. At the Gaussian boundary Us/Ua=0.005, swapping
14.98 s and 2.69 s changes nominal fast-sine T1/T2 responses only
0.00430/0.00185 C. Unequal starts and ambient forcing can break this exchange.
This is not a hardware zero-capacity claim or an absolute quantization-resolution
limit. The lower positive loading bound 0.0001 is a declared three-start refit
sensitivity, now terminal and independently audited. Loading again reaches the
lower bound; nll improves only 0.01291, while K and the two fitted time scales
barely change. This supports weak loading information, not physical zero
capacity. See lag_identifiability.md for the restricted structural argument.

Initial sensor-bin coordinates move each cold heater/sensor pair together,
preserve the original ambient reference/bounds and are separate from hidden
heater-only starts. Dimensional propagation and score audits passed for nearest
and floor phases, with/without a 0.1 C buffer. The nearest-bin full-data fit and
its independent 128/256/512-node, dimensional, rectangle and stationarity audit
are now terminal and passed. Gain changes from 0.873222 to 0.869704 C/%, short
lag from 2.73048 to 2.72054 s, and nll from 6238.29865 to 6209.55360 with 24
additional bounded initial-state coordinates. Several reach bin edges. The
T1 free-simulation RMSE increases from 0.605 to 0.655 C, while T2 decreases from
0.118 to 0.113 C. Better correlated-bin likelihood does not imply better
noise-free simulation. The fit is an initialization sensitivity; improved
whitening/transfer has not been established. Fixed-initial floor and buffered/floor sensor-bin sensitivities
remain. These bounds do not integrate an initial-noise posterior.

The phase-reference check passed: nearest-to-floor bins translate by 0.16115 C.
With both cold pairs and ambient shifted equally and slack bounds, the linear
mean and bins translate together; nll changes below 1.4e-12. The phase is not
separate dynamic information under that freedom. Fixed initial states, active
ambient limits and absolute-temperature radiation need not share it.

The new --condition-initial-code sensitivity and independent selected-point
audit are terminal and passed. It uses
p(y0,...,yN)/p(y0), leaves nobs unchanged and subtracts the one-observation
smoothing score. All mean/noise coordinates, including initial sensor-bin
coordinates, passed independent conditional-likelihood finite differences.
Its current fitting/export/profile/audit support is exact under quadrature.
Stationary simulation and transfer drivers explicitly refuse this mode until
its conditional truncated-Gaussian initial-state sampler/marginals are supplied.
Do not silently use a mismatched simulation convention.

Conditioning on the first code changes T1 latent SD/length from 0.5362 C/658.3 s
to 0.8763 C/1761.9 s; 2*S/ell changes by only about 0.22%, while physical mean
parameters barely change. This is a working-model initial-state sensitivity,
not evidence that either stationary SD/length is accurately known.

## Numerical failures and recovery

The original zero-band attempt failed a rare-bin integration domain gate.
Component-conditional posterior ranges repaired it. Rare rectangles, tail checks
and the full score/regression suite passed. Its original failure is retained.

The terminal numerical fits for 0/10/20% and radiation each converged three
starts, then failed JSON export because parameter_count was numpy.int64.
Commit 6e9e654 casts it to a Python int and adds --resume-starts. The exported
directories above re-evaluate those saved points under the current score and
do not repeat optimization. Original exits, failed trial points and solver
traces remain evidence. Recovery receipts bind the old saved starts/source.
The 5% job also reached that export failure. joint_conductance_b05_export_v2
recovers the saved starts without re-optimization; joint_conductance_b05_audit
passed. Original failure/exit/source evidence is retained.

Use direct pgrep -fl tclab-error. Python subprocess process enumeration is blocked
by macOS sysmon and an empty result is not process absence. reconcile.py consumes
a fresh direct-tool process_check.json, checks exact PID/command identities and
terminal exit artifacts, and never launches/restarts anything. Older exit receipts
without finish timestamps remain terminal with unknown timestamps.

## Current execution and next unfinished phases

Use the manifest for current identities and controller sessions. Recovery and
profile inference are underway; cross-localization and continuous-Gaussian
disturbance comparisons are audited. Do not infer live state from this prose;
use fresh process and terminal receipts.
No duplicate live retry. At most four long numerical processes, one BLAS thread
each. Short independent arithmetic checks are recorded separately.

The original information job is terminal but failed its independent scalar
loading-curvature gate. Finite differences of mean eigensystems contaminate
that tiny second derivative. Its scalar step study and failure are retained.
The replacement uses matrix-exponential Frechet mean derivatives, verified by
independent block exponentials, full-data trajectory/directional checks and
the complete joint mean/noise score suite. joint_b10_information_v2 and its
independent audit passed. Active-face local gain and thermal-time SEs under
known correlated analogue covariance are 4.55 and 5.83 times the matching IID
same-marginal-variance values; the short-lag SE is 0.645 times. The full bin
curvature also estimates noise nuisances. These are conditional local
comparisons, not calibrated intervals. Keep the original matrix as a numerical diagnostic;
do not report its tiny centered eigenvalue as a variance. The original gain
profile completed naturally with seven successful points and no point below
the reference. Its corrected collection audit is terminal and passed for all
seven points. The first collection omitted per-point output folders, losing
two calculated-but-unsaved checks; that failed run remains intact. Five saved
checks were reused with identical numerical source/config/data and fit/array
hashes, and the two missing checks were recomputed. Each original recovery
shard retained seven completed draws before approved termination (exit -15).
The gain profile exited 0 before that termination request was executed.
The compiled continuation shards retain all fourteen completed outcomes and
continue only unfinished work. The original even branch had no point checkpoint
and is explicitly restarted from its original seed; its 119 abandoned callback
iterations remain recorded. The odd branch retains its completed start.
Numerical equivalence, independent score checks, exact input/protocol gates and
continuation control checks passed before the handoff. The scientific protocol,
quadrature order, draw count, retry allowance and tolerances are unchanged.
See compiled_score_methods.md. This conditional algorithm check cannot establish
real-data noise adequacy. The nine-point short-lag profile is terminal as
profile_b10_short_lag_v2, with nine successful points and all nine independent
point audits passed. Twice the nll increase rises to 552.445 at L=7 s and falls
again to 6.253 at L=15 s, where the fitted sensor time is 2.643 s. The inspected
profile_gain_lag_summary_v2 shows this second region and the seven-point gain
grid without confidence-endpoint interpolation. It does not locate or rule out
the secondary minimum. The declared lag-exchange fixed-point refinement and
independent audit are terminal and passed. At L=14.7900558 s it fits sensor
time 2.7313751 s, nll 6240.69230385 and T1/T2 RMSE 0.615/0.118 C. Twice the nll
increase is 4.78730, compared with 6.25274 at the original coarse 15 s point.
Both starts converged to this value. The point does not improve the baseline
and does not locate the unconstrained secondary minimum or calibrate a cutoff.
The first profile command used an incorrect coordinate name
and failed before fitting; that invocation is retained. Remaining profiles are
queued. See sensitivity_methods.md and
uncertainty_interim.md for the source-bound findings and limits.
uncertainty_config.json declares grids, starts, bounds,
fifty recovery draws and gain-LR coverage before measurement. Source drivers:
run_profiles.py, run_information.py, run_recovery.py.

The generated-data auditor pilot checks draw 0 only. Its free and true-gain
fits both passed full independent selected-point audits, exact seed/observation
regeneration and LR arithmetic. It is not a coverage estimate. A metadata label
in that frozen pilot refers to available rather than selected draws; its README
clarifies the denominator of one and later source corrects the labels. Its
two passed numerical point checks can be reused by the final fifty-draw audit
only with matching numerical auditor and fit/array/dataset hashes.

Recovery accounting and cached-audit controller checks are terminal and passed.
They validate failed/missing denominators, exact old free-point selection,
numerical dependency bindings and independent binomial-tail/quantile agreement.
The corrected one-draw summary-controller figure was inspected; it is not a
sampling or coverage result. summarize_recovery.py requires all fifty audited
draws for its final mode, binds refitted diagnostic definitions, and compares
their sampling spread with generating-point active-face information. Its final
measured summary is still pending.

Both continuations are now terminal with all fifty declared outcomes. Each has
an accepted free/null fit and saved refitted diagnostics; these are unverified
recovery outcomes until the full numerical/data audits pass. The odd25 audit
is now terminal with all twenty-five odd rows and all fifty selected free/null
numerical checks passed. Its partial status is retained; this is not the full
fifty-draw calibration. The even25 audit remains live, using both current
terminal ledgers and only the passed, source/data-bound draw-0 cache.
The odd audit's original even declaration witness is not conflated with the
current complete ledger. The final full audit must consume both current
complete continuation ledgers and source-bind any reused parity checks. No
partial-audit coverage estimate is a fifty-draw calibration result.

transfer_config.json declares sixteen primary folds (twelve runs + four
protocols) and four shared-model protocol folds. Training/prediction code is
committed; independent dimensional, no-future-output-leakage, reported-code
expectation/quantile, mixture and fold-separation checks passed. All four primary
leave-protocol-out fits and canonical noise/scenario forecasts passed full
independent audits. transfer_primary_runs is now terminal with all twelve
primary leave-run-out fits successful. Their independent selected-point audits
and all twelve canonical forecast/audit pairs are terminal and passed. The
twelve figures were inspected. Their pooled nominal RMSE is 1.750/0.754 C;
the corresponding primary leave-protocol RMSE is 2.131/0.875 C. The four shared-model protocol fits, independent
selected-point audits and canonical forecast/audit pairs are now terminal and
passed. Their twelve figures and the two-candidate protocol summary were
inspected. Shared versus bounded-conductance nominal protocol RMSE is
2.171/0.894 versus 2.131/0.875 C. The large all-data fitting improvement gives
little improvement in these nominal omitted-protocol predictions. The shared
model's larger fitted residual variance gives broader noise-only ranges;
the bounded model's additional unknown-conductance scenarios give broader
outer ranges. Neither establishes calibrated total-95-percent predictions.
Evidence: transfer_protocol_candidates_canonical_summary (still partial while
nineteen full parameter pipelines remain unfinished). transfer_all_canonical_summary
now accounts for every one of the twenty canonical folds, with no missing fold
IDs, while retaining partial status for the missing parameter components.
Parameter-draw propagation must inherit a validated approximation;
noise-only hard-nuisance ranges are not total predictive intervals. If profiles
change the candidate set, extend transfer support explicitly rather than using
a linear predictor for a radiative or conditioned-initial-state fit.

The first complete omitted-protocol group gives pooled reported-mean RMSE
2.131/0.875 C, with noise-only marginal coverage 32.1/36.7 percent. The hard
condition-plus-noise envelopes cover 99.57/99.95 percent, but are 9.072/5.173 C
wide on average. These are scenario envelopes, not calibrated total-95-percent
intervals. Per-run mean errors account for 85.7/73.2 percent of SSE; removing
those means leaves RMSE 0.806/0.453 C. That decomposition uses test outcomes
and is not a blind correction. The accounting fixture and source-bound summary
transfer_primary_protocols_noise_decomposition passed; figures were inspected.

Training subsets define new geometric nominal conductances. The arithmetic
reference_transport_check passed, including bit-identical canonical forecasts
under the optional-ratio refactor. A separately declared primary-protocol
forecast stress check uses ratios [0.86894,1.13551], derived without test-output
fitting, and preserves all training fits/canonical forecasts. It does not
transport the training likelihood or calibrate coverage. Read transfer_methods.md.

The manifest's transfer_analysis index owns selected fit/prediction paths.
New prediction output binds
parameter files to draw receipts and stores sparse component witnesses.
audit_transfer_prediction.py checks complete nominal means/metrics and selected
mixture witnesses; full measured-fold validation is pending.
summarize_transfer.py keeps the three declared comparison groups separate and
checks 64/256-draw changes. Its weighted-SSE, duplicate-record, run-offset and
failed-fold accounting fixtures passed. Do not treat controller checks as
transfer results. run_transfer_uncertainty_queue.py sequences reference stress,
information, independent information audit, local draws and 64/256 forecasts
with their full audits. A failed approximation stops the queue for review.
stage_execution.py gives every sequential child its actual process, frozen
source, log and exit. Its success/failure/no-overwrite fixture passed. A waiting
queue and its one numerical child share one of the four worker slots; reconcile
the nested process receipts as well as the top-level parent. Passing the
queue-control fixture alone does not establish parameter uncertainty.
The first measured pipeline,
transfer_uncertainty_protocol30, is now terminal with every stage passed:
information, independent curvature audit, local draws, reference stress and
64/256 forecasts with full independent audits. All nine new figures were
inspected. Its 256-draw mixture changes hard-condition-plus-noise mean widths
from 9.119/5.134 C to 9.223/5.206 C, with the same 100/99.963 percent observed
coverage. Coarse/fine mixture endpoints differ by at most one code, 0.33 C,
at 2.26/2.69 percent of endpoints. The local approximation remains conditional
on one basin and its active face. transfer_first_protocol_parameter_summary
keeps this one completed fold separate from the nineteen unfinished pipelines.
The remaining four profile coordinates are running in profile_remaining_b10,
which stops if a point improves the reference or fails a numerical audit.

Still required: synthesize the completed cross-localization, disturbance,
frequency/operating and information/eigenvector findings in the final report;
finish pending initial/bin sensitivities, their selected-point audits and the
remaining nuisance profiles; finish fifty refitted recovery/coverage draws with failure denominators;
retrospective run/protocol transfer with parameter/noise/bin/model-form limits;
and the final report and requirement-by-requirement completion audit.

updated_diagnostics.py is committed for the audited final mean candidates. It
compares analogue means and marginal reported-code expectations, matched complex
frequency responses with working HAC regions, phase-aligned repeats, operating
groups and conditional reported-code forecasts. Its independent bivariate
observation-bin forecast and no-future-output-leakage check passed.
updated_mean_diagnostics is terminal. Its first full-table audit failed before
numeric checks because sibling result paths used the frozen source directory;
the corrected updated_mean_diagnostics_audit_v2 uses the supplied input root
and passed all 960 QR decomposition rows, code expectations, held-input complex
responses and aggregates. The phase figure was inspected.
The recovery driver now also saves deterministic
diagnostic metrics after each accepted free fit; diagnostic failures remain
separate from the interval-coverage denominator.

The new coupled_noise.py comparison uses correlated OU Brownian driving noise
and/or correlated analogue measurement noise, with unequal-time stationary
cross covariance derived from the OU Lyapunov equation. Dense toy covariance,
innovation and all-coordinate score checks passed. Its independent Gaussian
mode reproduces the prior Gaussian objective within 3e-11 under a full-data
dense audit. The process-correlation fit and full-data dense audit passed;
its driving correlation is 0.0908 and nll improves 4.3501 versus the independent
Gaussian fit. Measurement correlation and its audit also passed: rho=-0.00692,
Gaussian nll improvement 0.16748. gaussian_both_correlations and its full-data
dense audit are now terminal and passed. Process/measurement correlations are
0.11587/-0.02463 and nll improves 6.14891 over the independent Gaussian model.
These are working correlations, not measured
airflow or electronics. Their likelihood
is continuous Gaussian on discrete temperatures, not the
quantized multichannel likelihood. They cannot establish calibrated bin
probabilities or identify an environmental cause by themselves. Retain the
quantized baseline's input/phase failures in uncertainty and transfer discussion.

## Reporting and continuation

report/main.tex and main.pdf are a seven-page work-in-progress report, not a
finished study. Current numeric checks bind 265 source files and independently check the concise
Markdown companion as well as the LaTeX values; all seven rendered pages and the requested geometry were inspected. Recompile, rerender
and recheck after each meaningful revision. Geometry is 0.25-inch left and
3.25-inch right, with 35 college-spaced annotation lines per page. Final evidence,
figures and uncertainty/transfer remain pending. report.md is now a concise
working companion with source-checked values; its final update remains pending. Current nonrandomized, local-information and initialization findings
are included in this PDF and uncertainty_interim.md, together with the audited
all twenty canonical folds, group-specific offset decomposition, the first parameter-inclusive
fold and whole-data feedback-assisted forecast diagnostics. The latter are in-sample
and do not validate physical free simulation. The bibliography currently continues
onto the last page; final pagination is still pending. Profiles, calibrated
recovery and parameter-inclusive/full-fold transfer remain unfinished. Keep no-ai-slop
and the user's research practices in the writing check.

The hourly tc-lab-overnight-research heartbeat stays active. Read this file, the
committed plan and manifest on resume. Freeze committed code/data/configuration
with freeze.py under pinned Python/dependencies from ../environment.json.
Save starts, failures, source/input hashes, processes and exits. Commit coherent
source before measurement, audited results afterward; push progress and final
results on reformulation-exploration. Preserve raw data and prior deliveries.

Finish only when the full scoped gates have terminal audited evidence, figures
and the final tablet PDF have been inspected, the Markdown/LaTeX/PDF findings
are committed and pushed, and no required job remains live. Report unresolved
identifiability or whitening honestly. Then pause the heartbeat for Alex's review.

The requirement register in completion_audit.md maps every scientific, execution
and delivery gate to evidence and explicit remaining work. It is not a completion
claim. Reinspect all rows against terminal artifacts at the final checkpoint.

The historical T1-only positive three-pole/curved-command/affine-ambient benchmark
is retained in both report drafts at 0.374 C RMSE. Its old producer/independent
audit exits, fit/trajectory/source hashes, all raw T1/time/input arrays and
10,795-point scalar SSE were rechecked. Different outputs and ambient definitions
prevent a matched likelihood/AIC ranking with the current two-output fits.

The remaining-protocol uncertainty queue stopped at protocol_120s. Its
information producer exited 0; the independent auditor failed before numerical
checks because it took max over an empty active-bound set. The failed auditor
and parent exits/logs are retained. The correction treats the empty restriction
as vacuous while independently checking constraint rank, complete basis dimension,
centering and orthogonality. It does not snap a fitted parameter to a bound or
change any tolerance. A declared four-coordinate control must pass first.
The saved information also has a negative weak loading eigenvalue (-0.00735),
with stable producer directional checks. Its covariance is not accepted; the
corrected full audit will retain an explicit nonpositive-curvature failure if
confirmed, before any fit refinement or parameter propagation. No existing
canonical forecast or selected fit is overwritten.

The corrected protocol120 audit completed with explicit nonpositive-curvature
failure after its numerical checks. Four loading seeds (.03, .3, 3, 30), usual
two lag branches, original bounds and 300-iteration budgets are declared in
transfer_config.json and transfer_methods.md before any refinement fit. The
first new seed control failed on a Python dict-call syntax error before
numerical work. A separate compilation preflight had caught it, but the
controller incorrectly continued instead of checking that exit before dispatch.
The frozen failed control and source commit remain intact. The corrected
four-file compilation/configuration preflight now passes; a new seed-control
output must pass before the refinement starts. No failed control or fit is
promoted by correction, and no original fit/forecast is overwritten.

Both parity audits are now terminal: twenty-five rows and fifty selected free/null
numerical checks passed in each. Their partial statuses remain unchanged. The
final full50 audit must bind both current terminal continuation ledgers, exact
seeds/data/diagnostics and all reused numerical checks; it uses no subset or
partial mode. The even audit's retained SciPy beta-quantile overflow warning did
not change its zero exit or finite JSON output; final Monte Carlo interval
arithmetic must still be checked independently before reporting calibration.

All six remaining initial/bin/hidden-heater cases are terminal and independently
audited. Hidden-heater +/-2 and +/-5 C fits coincide: max selected offset 0.514 C,
T1/T2 latent-reference RMSE 0.682/0.113 C, rS=2.667 s and input L=14.898 s.
Their opposite lag assignment makes the consequential-startup-profile condition
apply. Declare a representative nuisance profile before measuring it.
Free-initial nearest/floor fits have matching likelihoods and nearly identical
dynamics but different raw code-minus-latent-mean RMSE; compare reported-code
expectations before interpreting that as a prediction difference.

The targeted protocol120 loading refinement also exited 0. Three additional
loading starts converge to r=0.341 and nll4595.33824, improving the original
near-bound point by0.25791. The fourth reaches the other lag region. Selection
used objective value only; the new selected-point and information audits remain
required. Original fits, canonical forecasts and failed information artifacts
remain unchanged. The fresh process reconciliation found two free worker slots;
the earlier controller's exact-three-active expectation was protective but too
restrictive and launched nothing. More free slots do not violate the limit.

## October 27, 02:10 UTC execution checkpoint

The startup-profile control passed: dimensional propagation agrees within
3.6e-15 C and all four hidden-heater scores agree with independent likelihood
differences within 2.3e-9 scaled error. Fixed-state and CLI routing checks passed.
The six-point representative profile is now released for measurement.

The shared protocol120 information audit confirms negative curvature
(-0.00475); no covariance was accepted. The primary refined120 audit instead
stopped at its independent scalar-curvature gate: the old one-sided .005
stencil differs by0.0375 from H22=249.20545. This can be truncation at the
new, much larger loading, and requires a saved convergence study with unchanged
.001 agreement tolerance before changing the stencil. Both failures remain.
The protocol300/step full pipelines are terminal and passed; their figures and
final index integration remain. The shared30 full pipeline passed numerically.
The first six run pipelines and remaining radiation profile are still active.

## October27 remaining nuisance profiles complete

The four-coordinate queue exited0 at03:03:58UTC. All28 points and their
independent numerical checks passed; no point improves its reference. The
inspected source-bound figure v3 retains every point. Read
nuisance_profile_findings.md for the conditional comparisons and boundary
limits. The primary refined120 scalar-curvature convergence study now occupies
the released long slot; its first three centered fourth-order stencils agree
within2.2e-5, while the old forward errors shrink by about4 on step halving.
Full study termination and unchanged .001 audit agreement are still required
before accepting a covariance. Startup and the two run queues remain live.

The refined120 five-step scalar convergence study exited0. Fourth-order
centered errors stay below2.2e-5; the forward error falls quadratically but
still exceeds.001 at h=.001. Source-committed interior centered stencils now
replace only that numerically biased audit check, with the same .001 gate.
Fresh full information audit52318 is live. The stored matrix, fit, bounds and
all other checks are unchanged. The failed old audit and full convergence
evaluations are retained. Then finish primary120 parameter forecasts before
using the released slot for shared120 additional starts.
