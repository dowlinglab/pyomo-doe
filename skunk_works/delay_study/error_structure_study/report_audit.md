# Final report audit

Audit performed 2026-10-27T05:36:10.674964+00:00. The subjects are `report/main.tex` SHA-256
`c1b0fc70420b00f33e87b197cf2f213ae447ae413365d62fbab116af0b856e75`, `report/main.pdf` SHA-256
`6e7cff8f3b7d2356a8bfc77b9d809e84d9dcebe3bbf100f28d427f4dc5114c92`, and `report.md` SHA-256
`75d52345e160f683ead43b89fe5eb69e3c04e4562d893b7e8f9fa850721ec93b`. Numerical sources were read only. This is agent-authored verification;
independent numerical checks mean separate arithmetic/solver paths. Alex's
scientific review remains open. The final report
supersedes the frozen `report/wip_oct26` intermediate draft; it does not alter
that draft or any prior study. Editing a bound source requires a new audit.

## Arithmetic

`report/final_numeric_audit.json` passes, binding755 source/evidence files.
It recomputes the Gaussian table's RMSE, joint fits' RMSE, all twenty preferred
folds' RMSE/coverage/width,64/256 endpoint changes, all four reference-stress
forecasts and the28 nuisance-profile differences. Separate independent point
audits check dimensional dynamics, observation rectangles, likelihood support
and derivatives. Their scope is narrower than a proof of statistical adequacy.

A worked example: the printed joint 10% T1 RMSE0.605 C comes from
`results/joint_conductance_b10_export_v2/trajectories.npz`. Remove the initial
row of each of the twelve records; sum squared differences between observed
and latent mean. SSE=3945.038945387214 C2 over10795 points.
The square root of3945.038945387214/10795 is0.604525081 C, rounded0.605 C.
The checker verifies the raw CSV hashes, matching arrays and a passed
independent selected-point audit before using this quantity. Reported-code
RMSE integrates the observation law and is a separately labeled metric.

| Claim group | Authoritative evidence | Arithmetic status |
| --- | --- | --- |
| All 21,614 hardware readings on shifted grid | Raw records and hardware audit | VERIFIED exact decimal-grid reconstruction |
| Matched Gaussian and joint-bin comparisons | Audited selected fits and full trajectories | VERIFIED direct SSE; unlike density/bin likelihoods not ranked together |
| Radiation0.021--0.383 W and remainder0.0021--0.0370 W | `radiation_scale/scenarios.csv` and dimensional audit | VERIFIED stated geometry/emissivity/temperature scenarios |
| Local correlation effects4.55/5.83/0.645 | `joint_b10_information_v2_audit` | VERIFIED named-coordinate, known-covariance comparison |
| Recovery49/50 and15 label reversals | Full 50-draw numerical/data audit and independent summary | VERIFIED denominators, scalar binomial-tail limits and saved points |
| Startup five accepted, one failed | `startup_profile_summary` and five real point audits | VERIFIED; rejected endpoints receive no deviance |
| Loading, conductance, length and radiation profiles | `profile_remaining_b10`, figurev3 | VERIFIED all 28 selected point/audit bindings and difference arithmetic |
| Full transfer groups and local64/256 sensitivities | `transfer_final_summary`, saved arrays/curvature/draw chains | VERIFIED all 20 preferred numerical components |
| Taylor201.731 versus34.276 | `local_taylor_quality`, independent quadrature followup | VERIFIED scalar Hessian/gradient sum, actual independent forward likelihood and convergence arithmetic |
| Extreme-draw precise likelihood | Declared1024-node/support followup | CANNOT VERIFY within declared convergence gate; precise deviance withheld |

## Method

| Method question | Status and scope |
| --- | --- |
| Physical run coupling | VERIFIED centered boundedUa with shared beta/CH/CS/Us/Ux; scale-free ratios change together. Band does not estimate a population variance. |
| Actual input timing | VERIFIED left-held commands and actual within-run timestamps; missing inter-run recovery remains unknown. |
| Radiation interpretation | VERIFIED Kelvin calculation, refitted convection/gain and nonnegative nuisance profile; CANNOT VERIFY physical emissivity or exact geometry. |
| Startup/threshold law | VERIFIED declared nearest/floor, bins/buffer, hidden starts and first-code sensitivities; CANNOT VERIFY actual cold equilibrium or hardware thresholds. No initial-noise posterior claim. |
| Sequential versus joint fitting | VERIFIED Gaussian feasible-GLS comparison, normalization retained; initial IID stage fixes relative SD. Not a joint initial IID variance-componentMLE. |
| Quantized likelihood | VERIFIED scalar OU-plus-nugget bin integration against independent rectangles and convergence tests. Emulator noise unused. |
| Approximate IID innovations | NOT ESTABLISHED: hardware input/phase/ACF/cross diagnostics exceed every one of50 refitted references. Repeated randomized PIT agreement is insufficient. |
| Cross-channel disturbance | VERIFIED dense continuous-Gaussian sensitivities on discrete data; CANNOT VERIFY a calibrated joint two-channel binned law. |
| Local covariance and sampling | VERIFIED numerical Hessian; local normal fails away-from-reference shape check. Parameter mixtures remain exploratory. |
| Recovery coverage | VERIFIED scalar gain/working-model/algorithm check with exact Monte Carlo uncertainty; CANNOT VERIFY hardware or selection-adjusted coverage. |
| Transfer | VERIFIED no later-output fitting of unknown nuisances; all12 records informed structure and initial guesses, hence retrospective. No untouched validation claim. |
| Mixture independence | Checks cover all components' quantile brackets at declared times and selected trajectories; do not independently recompute every component history. |
| Reported high coverage | VERIFIED descriptive pointwise coverage and hard scenarios; CANNOT VERIFY total95% calibration or an unknown-run distribution. |

Radiation, startup and threshold results are all-data sensitivities. Transfer's
declared candidates are shared conductance and the10% band. No claim says that
radiation or hidden-start changes improve held-out predictions. The affine
ambient comparison is Gaussian; it does not rule out drift under the joint
quantized law. FastT2 complex response can include zero, so its phase is not
reported as precisely identified. AbsoluteUa/capacities and a unique lag
assignment remain unidentified under the current evidence.

## Transcription and presentation

VERIFIED: generated transfer tables bind the complete preferred summary,
including the objective-selected120s refinements. Original canonical comparisons
and failures remain distinct. The final numeric helper checks both Markdown
and TeX against numeric inputs rather than only matching their prose. The
fifty-refit and initial-code helpers bind their independent audits.

VERIFIED: actual TeX build succeeds, no unresolved citation/reference or
overfull box, eleven page images inspected. Final geometry and visual receipts
bind the exact PDF/source. The requested0.25in left and3.25in right margins,
35 college-ruled lines and9/32in pitch pass on every page. No empty table or
missing figure. Figure1 is the four-profile panel plus startup; Figure2 is
full50 recovery. Their PDF bytes match reviewed figure receipts.

VERIFIED source claims: Park etal2020 supports prior convection/radiation/sensor
models; Poyiadjis etal2011§2.1 supports Fisher's score identity; Dunn--Smyth1996
supports randomized quantile residuals. Exact inspected versions and access
limits are in `radiation_sources.md`, `quantized_methods.md` and
`observation_sources.md`. No current source was silently substituted for
Dan's unknown firmware. The prior ten-family literature review retains four
incomplete leads and does not claim exhaustive coverage.

Writing check against no-ai-slop/eval.md: editing principles PASS; empty/banned
phrases PASS; named pattern check PASS; concrete scientific voice and final
read PASS. The artifact is the full edited draft. Changes since WIP: replaced
partial/original-fit transfer values with all 20 preferred components; added
28-point/startup profiles; stated failed local-normal shape and unresolved
extreme quadrature; fixed pagination and removed stale pending claims.

## Self-corrections and retained failures

The study corrected earlier misdescription of1 C totalRMSE as a pure amplitude
error. It also distinguished latent-reference RMSE from expected reported-code
RMSE under a half-grid temperature-reference shift. A prior positive-Hessian
assumption was tested and failed; final ranges are labeled exploratory.

Execution corrections retained in their original directories include rare-bin
integration support, numpy-int JSON export/resume, tiny-loading numerical
curvature, missing profile audit folders, incorrect coordinate/controller
assumptions, a true weak negative curvature at the first120s training point,
and scalar forward-difference truncation after refinement. Curvature corrections
kept the original gates and did not clip eigenvalues or move bounds.

The final-check preflight stopped before the startup summary existed. A new
reference-stress assertion initially referenced Markdown before initialization;
its code correction was committed and the whole checker rerun successfully.
A report-edit command used the wrong working directory and failed before
editing; its subsequent compile used unchanged source. The edit was applied
with an absolute path and the actual final source recompiled, rechecked and
rerendered. PDF text was inspected during an earlier compilation for pagination
only; all saved final hashes/renders were collected after terminal compilation.
No such controller/edit failure changes a scientific fit.

The final delivery audit separately inventories every nonzero process exit,
failed task and exact retained log. These include deliberately failing
controller fixtures and approved recovery-continuation SIGTERM exits; none are
silently counted as successful scientific fits. The startup-2C task and extreme
quadrature witness remain declared unsuccessful outcomes.

## Author-decision register

| Decision for Alex | Evidence now available | Status |
| --- | --- | --- |
| Whether to revise physical symmetry or disturbance structure next | Persistent refitted input/phase/cross failures | OPEN scientific review |
| How to assign sensor/input time scales physically | Near-commuting transfer factors, startup sensitivity and15/50 swaps | OPEN; current data do not establish unique labels |
| Whether to run controlled-environment hardware later | Quantified run offsets, radiation scale and proposed logging | OPEN; no hardware action in this goal |
| Whether to seek a global sampling approximation | Local-normal shape failure and conditional scalar recovery | OPEN; current mixtures remain sensitivities |

A reviewer can trace each table to its linked JSON/arrays and rerun the two
report checks. Record discrepancies as `claim | source | recomputed value |
status | evidence`; leave unverifiable mechanisms open rather than filling
them by interpretation. Push verification and heartbeat/goal closure have
separate receipts and are not established by this manuscript audit.

## Preservation check

The terminal read-only delivery audit passes for8090 pre-existing files,
including raw data and prior reports/delivery receipts. All398 indexed process
receipts are terminal;27 nonzero exits and one failed subtask remain visible,
with exact log copies. Input snapshots preserve the finalizing manifest and
process check even when closure metadata advances. No scientific subject was
changed by the audit.
