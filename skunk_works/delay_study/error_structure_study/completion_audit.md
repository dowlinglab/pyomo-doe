# Remaining-error study: completion evidence

This is the requirement register for the plan committed as `346a03d`. The
scientific analyses are complete with the stated solver, quadrature and
statistical limitations. Final preservation/push/closure checks are in progress. The register separates finished numerical checks from
scientific adequacy and from unfinished deliverables. A passed controller
fixture is not evidence that its measured study passed. The execution manifest
owns live process state; this file does not infer it from older prose.

Status vocabulary: `VERIFIED` means the stated scope has direct evidence;
`PARTIAL` means some required components remain; `PENDING` means the final
evidence has not been collected; `UNRESOLVED` is a scientific question the
completed analysis may leave open. An unresolved mechanism can be reported
honestly; a missing calculation cannot be relabeled as that outcome.

## Scientific requirements

| Plan requirement | Evidence to inspect for closure | Current status |
| --- | --- | --- |
| Same-device hardware provenance; twelve distinct ENV2 records, no alias replication | Plan, hardware review, frozen raw CSV hashes and selected arrays in every fit/audit | VERIFIED within the agreed provenance assumption; firmware and inter-run recovery remain unknown |
| Both outputs, actual command timestamps and left-held input | `results/bounded_check`, selected dimensional audits, transfer trajectory/no-future-output checks | VERIFIED for the implemented models; no reconstruction of missing dates |
| Retain the earlier empirical three-pole prediction benchmark | Prior `shared_model_investigation/results/mapped_unequal3`, its independent selected-point audit, current report's raw-SSE/source binding | VERIFIED historical source/array/SSE check and both report transcriptions; different output and affine-ambient definitions prohibit a matched likelihood/AIC ranking |
| Constant alpha absorbed into gain; no unjustified absolute Ua or capacity estimate | Scale-free equations, `lag_identifiability.md`, report equations and physical-interpretation audit | VERIFIED algebra and final report interpretation; absolute conductance/capacity remains unscaled |
| Four prior model families; offsets, trends, fundamentals, higher harmonics and remainder | `results/decomposition`, `results/decomposition_audit`; 384 rows, 32 aggregates, prespecified 0/60/180/450 s cutoffs | VERIFIED arithmetic; descriptive projections retain their order dependence |
| Operating-point, heating/cooling, common/differential and frequency-response diagnostics | `results/operating_first`, `results/operating_first_audit`, `results/updated_mean_diagnostics`, `results/updated_mean_diagnostics_audit_v2`; inspected figures | VERIFIED declared tables and checks; causal attribution UNRESOLVED |
| Free simulation versus one-step and short-horizon prediction | `results/updated_mean_diagnostics/forecast_horizons.csv`, `results/updated_diagnostics_check`, report numeric audit | VERIFIED whole-data in-sample comparison; toy independent bin recursion has narrower scope than recomputing every hardware history |
| Regress shared nominal Ua with centered, bounded per-run deviations | `bounded_check`, all Gaussian band audits, joint 0/5/10/20% fits and audits | VERIFIED equations, centering, bounds and selected fits; hard band is not a population distribution |
| Conditional synthetic recovery of run deviations and full joint numerical score | `bounded_check`, positive/zero-band joint score checks, Frechet and compiled-score checks | VERIFIED declared synthetic/numerical scope; full repeated-data calibration remains separate |
| Radiation magnitude, Kelvin, finish/area/view/surface-temperature uncertainty and nonlinear remainder | `results/radiation_scale`, its audit, `radiation_sources.md`, radiative nesting and dimensional checks | VERIFIED conditional scenarios; device emissivity/area and heater-surface temperature UNRESOLVED |
| Matched radiation fit, refitted convection/gain, linear-command sensitivity and nonnegative profile | Gaussian radiation/linear-command audits, `joint_radiation_b10_audit`, `profile_remaining_b10/radiation_gamma` and its collection audit | VERIFIED Gaussian and joint-bin fits, linear-command comparison and all ten radiation profile points; no calibrated boundary significance or emissivity estimate |
| Cold-start and hidden-heater sensitivity, +/-2 then +/-5 C | Gaussian startup audits, terminal quantized `initial_remaining_b10` and independent code-space summary | VERIFIED specified sensitivities; maximum0.514C and opposite lag labels, equilibrium remains unverified |
| Initial measurement bins, threshold phase, analogue buffer and residual first-code conditioning | `initial_sensor_check`, `phase_reference_check`, all initial/bin/conditional audits, `initial_reported_code_summary` | VERIFIED declared cases; independent erfc checks and full arrays confirm free-initial phase-reference equivalence; no initial-noise posterior integration |
| Profile startup terms if they affect conclusions | Compare all audited initialization cases with the primary model, record which conclusions move, then profile a consequential nuisance or document why the conditional trigger is absent | VERIFIED declared six-point attempt: five accepted points have independent audits; -2 C exhausted both 300-iteration starts and receives no deviance. No lower accepted point; no global profile or confidence endpoints |
| Sequential covariance estimation/refit and comparison with joint likelihood | `sequential_methods.md`, `sequential_b10_iid_audit`, `sequential_b10_joint_audit` | VERIFIED Gaussian comparison; initial IID stage fixes relative SD and is not joint variance-component MLE |
| Quantization-aware correlated likelihood; no emulator noise law | `quantized_methods.md`, hardware audit, rare-bin/smoothing/rectangle/full-score checks and independent selected-point audits | VERIFIED implemented independent-channel OU-plus-nugget/bin law; actual thresholds and analogue noise are assumptions |
| Cross-channel stochastic alternatives, with honest approximation limits | `coupled_noise_check`, four dense Gaussian covariance audits, deterministic cross-localization | VERIFIED continuous-Gaussian sensitivities on discrete data; a calibrated joint binned two-channel law is not established |
| ACF/PACF, declared Ljung-Box lags, input/phase/cross dependence, variance/QQ/tails | `joint_b10_predictive_diagnostics`, `joint_b10_nonrandom_diagnostics`, matched localization and inspected plots | VERIFIED fixed-model diagnostics; deterministic checks retain structure, so IID adequacy is not established |
| Refitted diagnostic references, failed-draw accounting and Monte Carlo uncertainty | Both terminal continuation ledgers, both terminal parity audits, final full fifty-draw audit and summary | VERIFIED full fifty-draw numerical/data and independent summary audits, with failed/missing counts zero and explicit algorithm/model/selection limits |
| Gain, lag, loading, representative run-deviation, noise-length and radiation profiles with all permitted nuisances refit | Gain and lag audits, fixed lag-exchange refinement, terminal `profile_remaining_b10`; preserve failed starts and disconnected branches | VERIFIED seven gain, nine lag and all 28 remaining nuisance points plus exchanged-lag refinement; startup five accepted/one failed. Nuisances refit, rejected starts retained; no routine boundary cutoff |
| Information eigendirections in named coordinates and active-face limits | `joint_b10_information_v2`, independent audit, old failed curvature result, invariant coordinate interpretation | VERIFIED numerical/local analysis; active-face covariance does not cover other basins or released bounds |
| Local quadratic shape at saved draws, beyond numerical Hessian correctness | `local_taylor_quality`,16 declared saved indices for full data, omit30s and the run with largest conditional log_tA SE; independent worst-point physical/256-node witnesses | VERIFIED three cases × 16 saved draws and independent dimensional/128–1024-node followup. Representative 201.731 versus 34.276 confirms shape failure. Extreme witness fails highest-order/tail gates; precise value withheld |
| Fifty refitted recovery draws, true-gain constrained fits, declared retries and failure denominators | Current even/odd design and outcome ledgers, final complete independent audit, source-bound recovery summary | VERIFIED all fifty numerical/data-audited outcomes, one hundred saved-objective checks, independent binomial Monte Carlo limits and summary arithmetic; nominal coverage is only a conditional algorithm check |
| Retrospective leave-run and leave-protocol transfer; small declared candidate set | Twelve primary-run, four primary-protocol and four shared-protocol fits, their selected-point audits | VERIFIED all twenty selected fits; all records informed structure selection |
| Unknown-run nuisance propagation without later-output fitting | All twenty canonical forecasts/audits, source-bound transfer index, `transfer_all_canonical_summary` | VERIFIED canonical noise/hard-scenario arithmetic; hard ranges do not define a probability law over conditions |
| Parameter-inclusive transfer and 64/256 mixture sensitivity | Every fold's audited information, supported local approximation, draw receipt and independent forecast audits; full source-bound summary | VERIFIED all 20 preferred numerical components, full curvature/audit/draw chains and raw-array aggregate; 64/256 changes reported. Local-normal calibration UNRESOLVED/unsupported after shape check; mixtures explicitly exploratory |
| Training-reference change and invariants | `reference_transport_check`, invariant ratios in saved folds, all four primary-protocol reference-stress forecasts/audits | VERIFIED invariant chart and all four preferred reference-stress forecasts/audits, complete raw-array summary; stress is not a transported cohort likelihood or calibration |
| Quantization, startup, noise and model-form limits included in uncertainty interpretation | Completed sensitivity/profile/recovery/transfer evidence compared in final report | VERIFIED final Markdown/TeX compare these limits, lag reversal, refitted recovery and Taylor failure. Calibration and physical mechanism remain UNRESOLVED |
| Recommendations for a future controlled-environment experiment | Final report records ambient, absolute time, preceding recovery, airflow/hood settings, firmware/device identity and electrical-power logging; no new experiment in this goal | VERIFIED final report logging recommendations; no new hardware or DoE |

## Reproducibility and deliverables

| Requirement | Evidence to inspect for closure | Current status |
| --- | --- | --- |
| Scientific plan committed before measured work | `346a03d`, goal start and immutable source/data receipts | VERIFIED |
| Source committed before measurements; pinned runtime, one BLAS thread, at most four numerical slots | `freeze.py`/stage receipts and fresh process reconciliation, numerical dependency bindings | VERIFIED execution receipts and final reconciliation: 398 terminal jobs, zero live science jobs, no missing-exit/process warnings |
| Save failures, starts, optimization exits and corrections | Original rare-bin/export/curvature/controller failures, migration exits, source-bound continuation receipts and retained logs | VERIFIED delivery audit inventories 27 nonzero exits, exact logs and one failed subtask; fixtures/approved migrations distinguished from accepted fits |
| Independent arithmetic, method and transcription checks have distinct scope | Numerical audits, method limits and final report claim audit; do not equate matching text with independent recomputation | VERIFIED final 755-source numeric checks and report_audit.md separate arithmetic, method appropriateness and transcription; model adequacy explicitly unestablished |
| Primary literature claims tied to inspected sources and access limits | `radiation_sources.md`, `observation_sources.md`, prior-modeling review and source receipts | VERIFIED final three report citations checked against inspected primary versions; source/access limits retained, prior review not claimed exhaustive |
| Concise Markdown findings plus checked LaTeX/PDF | Final Markdown companion and LaTeX/PDF, numeric/source hashes and inspected final figures | VERIFIED final Markdown and eleven-page tablet LaTeX/PDF with complete results, source-bound tables, 755-input numerical check and explicit failed approximations |
| Tablet layout: 0.25-inch left, 3.25-inch ruled right margin | Final `report/layout_audit.json` and all eleven inspected page renders | VERIFIED final hash-matched layout audit: 0.25in left/3.25in right, five-inch body, 35 lines and 9/32in pitch on all 11 pages |
| Inspect rendered figures/PDF, resolved references, labels and pagination | Per-result figure reviews, final PDF review with exact render/source/PDF hashes | VERIFIED all 11 page pixels, source-bound figure PDFs and references; changed pagination inspected, no empty tables/missing figures/unresolved citations |
| no-ai-slop and claude-for-researchers practices | Final writing check, independent numeric/method/transcription audit and explicit self-correction ledger | VERIFIED end-to-end writing check and report_audit.md arithmetic/method/transcription/author-decision/self-correction registers |
| Preserve prior studies, raw CSVs and email receipts | Final diff against recorded starting evidence plus raw hashes and prior delivery artifacts | VERIFIED exact contents of 8090 prior files versus 346a03d, including rawCSV and earlier delivery receipts; no out-of-scope tracked change |
| Commit coherent progress and push reformulation-exploration | Local commits, final clean scoped status and verified remote commit | PARTIAL; newer local work remains unpushed |
| No required job remains live; pause heartbeat and hand back for review | Fresh direct process snapshot, terminal exits for every required job, full requirement audit, verified push, heartbeat pause receipt | PENDING; goal stays active |
| No hardware, DoE, pair enumeration or new email | Scope and final action/artifact audit | VERIFIED scope/action review; no hardware, DoE, pair enumeration or email action in this goal |

## Closure protocol

Reinspect every row against current artifacts after the numerical jobs finish.
Replace pending rows with exact source-bound evidence, including failed or
unavailable scientific approximations. Check what each auditor actually
recomputes. Regenerate report values from terminal evidence, then check the
Markdown and rendered PDF separately. Preserve earlier failures and explicitly
record any corrected claim. The completion claim requires the full original
scope, a verified push and the final process audit; this register by itself
does not prove any of those outcomes.

## Final numerical gate outcomes

`results/transfer_final_summary` is complete: all twenty selected fits and
preferred information/draw/forecast chains are terminal and independently
audited. Both120s folds were refined by declared multistarts chosen on training
likelihood only. Old candidates and failed audits remain intact. The final
aggregate binds the local-normal shape failure; complete numerical accounting
is distinct from an adequate uncertainty approximation.

`results/startup_profile_summary` reports five accepted/audited points and
one failed endpoint, with all starts preserved. The extreme Taylor witness
remains unconverged at the declared 1024-node/support gate, while a representative
point's independently converged discrepancy confirms local-shape failure.
These unsuccessful branches are reported outcomes, not missing undisclosed work.
No more computation of the same local-normal approximation is justified as a
calibration remedy. Scientific mechanism and IID adequacy remain open for review.

## Final preservation result

`results/delivery_audit/audit.json` passes:8090 prior file contents unchanged,
398 terminal scientific/control processes,27 retained nonzero exits and one
failed startup subtask. Exact failed-job logs are preserved as text beside the
inventory. The manifest and process inputs are copied byte-for-byte with a
receipt before later closure metadata updates. The audit deliberately does not
claim remote push or heartbeat/goal completion; those are verified separately.
