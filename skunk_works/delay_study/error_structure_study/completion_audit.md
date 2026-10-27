# Remaining-error study: completion evidence

This is the requirement register for the plan committed as `346a03d`. The
study is **incomplete**. The register separates finished numerical checks from
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
| Constant alpha absorbed into gain; no unjustified absolute Ua or capacity estimate | Scale-free equations, `lag_identifiability.md`, report equations and physical-interpretation audit | VERIFIED algebra; final report audit PENDING |
| Four prior model families; offsets, trends, fundamentals, higher harmonics and remainder | `results/decomposition`, `results/decomposition_audit`; 384 rows, 32 aggregates, prespecified 0/60/180/450 s cutoffs | VERIFIED arithmetic; descriptive projections retain their order dependence |
| Operating-point, heating/cooling, common/differential and frequency-response diagnostics | `results/operating_first`, `results/operating_first_audit`, `results/updated_mean_diagnostics`, `results/updated_mean_diagnostics_audit_v2`; inspected figures | VERIFIED declared tables and checks; causal attribution UNRESOLVED |
| Free simulation versus one-step and short-horizon prediction | `results/updated_mean_diagnostics/forecast_horizons.csv`, `results/updated_diagnostics_check`, report numeric audit | VERIFIED whole-data in-sample comparison; toy independent bin recursion has narrower scope than recomputing every hardware history |
| Regress shared nominal Ua with centered, bounded per-run deviations | `bounded_check`, all Gaussian band audits, joint 0/5/10/20% fits and audits | VERIFIED equations, centering, bounds and selected fits; hard band is not a population distribution |
| Conditional synthetic recovery of run deviations and full joint numerical score | `bounded_check`, positive/zero-band joint score checks, Frechet and compiled-score checks | VERIFIED declared synthetic/numerical scope; full repeated-data calibration remains separate |
| Radiation magnitude, Kelvin, finish/area/view/surface-temperature uncertainty and nonlinear remainder | `results/radiation_scale`, its audit, `radiation_sources.md`, radiative nesting and dimensional checks | VERIFIED conditional scenarios; device emissivity/area and heater-surface temperature UNRESOLVED |
| Matched radiation fit, refitted convection/gain, linear-command sensitivity and nonnegative profile | Gaussian radiation/linear-command audits, `joint_radiation_b10_audit`, pending radiation profile | PARTIAL; fits audited, profile unfinished; model choice must not be inferred from training likelihood alone |
| Cold-start and hidden-heater sensitivity, +/-2 then +/-5 C | Gaussian startup audits, terminal quantized `initial_remaining_b10` and independent code-space summary | VERIFIED specified sensitivities; maximum0.514C and opposite lag labels, equilibrium remains unverified |
| Initial measurement bins, threshold phase, analogue buffer and residual first-code conditioning | `initial_sensor_check`, `phase_reference_check`, all initial/bin/conditional audits, `initial_reported_code_summary` | VERIFIED declared cases; independent erfc checks and full arrays confirm free-initial phase-reference equivalence; no initial-noise posterior integration |
| Profile startup terms if they affect conclusions | Compare all audited initialization cases with the primary model, record which conclusions move, then profile a consequential nuisance or document why the conditional trigger is absent | PARTIAL; quantized hidden-start lag reversal triggers six-point profile, now live after independent score/propagation controls |
| Sequential covariance estimation/refit and comparison with joint likelihood | `sequential_methods.md`, `sequential_b10_iid_audit`, `sequential_b10_joint_audit` | VERIFIED Gaussian comparison; initial IID stage fixes relative SD and is not joint variance-component MLE |
| Quantization-aware correlated likelihood; no emulator noise law | `quantized_methods.md`, hardware audit, rare-bin/smoothing/rectangle/full-score checks and independent selected-point audits | VERIFIED implemented independent-channel OU-plus-nugget/bin law; actual thresholds and analogue noise are assumptions |
| Cross-channel stochastic alternatives, with honest approximation limits | `coupled_noise_check`, four dense Gaussian covariance audits, deterministic cross-localization | VERIFIED continuous-Gaussian sensitivities on discrete data; a calibrated joint binned two-channel law is not established |
| ACF/PACF, declared Ljung-Box lags, input/phase/cross dependence, variance/QQ/tails | `joint_b10_predictive_diagnostics`, `joint_b10_nonrandom_diagnostics`, matched localization and inspected plots | VERIFIED fixed-model diagnostics; deterministic checks retain structure, so IID adequacy is not established |
| Refitted diagnostic references, failed-draw accounting and Monte Carlo uncertainty | Both terminal continuation ledgers, both live parity audits, final full fifty-draw audit and summary | VERIFIED full fifty-draw numerical/data and independent summary audits, with failed/missing counts zero and explicit algorithm/model/selection limits |
| Gain, lag, loading, representative run-deviation, noise-length and radiation profiles with all permitted nuisances refit | Gain and lag audits, fixed lag-exchange refinement, pending `profile_remaining_b10`; preserve failed starts and disconnected branches | PARTIAL; no interpolated confidence endpoints or routine boundary chi-square claims |
| Information eigendirections in named coordinates and active-face limits | `joint_b10_information_v2`, independent audit, old failed curvature result, invariant coordinate interpretation | VERIFIED numerical/local analysis; active-face covariance does not cover other basins or released bounds |
| Fifty refitted recovery draws, true-gain constrained fits, declared retries and failure denominators | Current even/odd design and outcome ledgers, final complete independent audit, source-bound recovery summary | VERIFIED all fifty numerical/data-audited outcomes, one hundred saved-objective checks, independent binomial Monte Carlo limits and summary arithmetic; nominal coverage is only a conditional algorithm check |
| Retrospective leave-run and leave-protocol transfer; small declared candidate set | Twelve primary-run, four primary-protocol and four shared-protocol fits, their selected-point audits | VERIFIED all twenty selected fits; all records informed structure selection |
| Unknown-run nuisance propagation without later-output fitting | All twenty canonical forecasts/audits, source-bound transfer index, `transfer_all_canonical_summary` | VERIFIED canonical noise/hard-scenario arithmetic; hard ranges do not define a probability law over conditions |
| Parameter-inclusive transfer and 64/256 mixture sensitivity | Every fold's audited information, supported local approximation, draw receipt and independent forecast audits; full source-bound summary | PARTIAL; nineteen full components indexed/reviewed; shared120 is the sole unfinished component, including two information audit failures under investigation |
| Training-reference change and invariants | `reference_transport_check`, invariant ratios in saved folds, all four primary-protocol reference-stress forecasts/audits | PARTIAL; all four preferred primary-protocol stress forecasts are independently audited and indexed; summary/final interpretation pending |
| Quantization, startup, noise and model-form limits included in uncertainty interpretation | Completed sensitivity/profile/recovery/transfer evidence compared in final report | PENDING; numerical convergence alone cannot validate sampling coverage or physical mechanism |
| Recommendations for a future controlled-environment experiment | Final report records ambient, absolute time, preceding recovery, airflow/hood settings, firmware/device identity and electrical-power logging; no new experiment in this goal | Draft recommendations present; final review PENDING |

## Reproducibility and deliverables

| Requirement | Evidence to inspect for closure | Current status |
| --- | --- | --- |
| Scientific plan committed before measured work | `346a03d`, goal start and immutable source/data receipts | VERIFIED |
| Source committed before measurements; pinned runtime, one BLAS thread, at most four numerical slots | `freeze.py`/stage receipts and fresh process reconciliation, numerical dependency bindings | VERIFIED execution controls so far; final no-live-job check PENDING |
| Save failures, starts, optimization exits and corrections | Original rare-bin/export/curvature/controller failures, migration exits, source-bound continuation receipts and retained logs | VERIFIED retention so far; final failure ledger PENDING |
| Independent arithmetic, method and transcription checks have distinct scope | Numerical audits, method limits and final report claim audit; do not equate matching text with independent recomputation | PARTIAL; final claim/method audit unfinished |
| Primary literature claims tied to inspected sources and access limits | `radiation_sources.md`, `observation_sources.md`, prior-modeling review and source receipts | VERIFIED reviewed sources; final reference/claim audit PENDING |
| Concise Markdown findings plus checked LaTeX/PDF | Final Markdown companion and LaTeX/PDF, numeric/source hashes and inspected final figures | PARTIAL; eight-page PDF and independently source-checked Markdown companion are explicitly WIP; final evidence remains pending |
| Tablet layout: 0.25-inch left, 3.25-inch ruled right margin | Current `report/layout_audit.json` and eight rendered pages; rerun against final PDF | VERIFIED current WIP geometry; final hash-matched check PENDING |
| Inspect rendered figures/PDF, resolved references, labels and pagination | Per-result figure reviews, final PDF review with exact render/source/PDF hashes | PARTIAL; current eight pages inspected, final pagination and final figures unfinished |
| no-ai-slop and claude-for-researchers practices | Final writing check, independent numeric/method/transcription audit and explicit self-correction ledger | PARTIAL; current prose checked, final version unfinished |
| Preserve prior studies, raw CSVs and email receipts | Final diff against recorded starting evidence plus raw hashes and prior delivery artifacts | VERIFIED measured inputs so far; final preservation audit PENDING |
| Commit coherent progress and push reformulation-exploration | Local commits, final clean scoped status and verified remote commit | PARTIAL; newer local work remains unpushed |
| No required job remains live; pause heartbeat and hand back for review | Fresh direct process snapshot, terminal exits for every required job, full requirement audit, verified push, heartbeat pause receipt | PENDING; goal stays active |
| No hardware, DoE, pair enumeration or new email | Scope and final action/artifact audit | VERIFIED current goal actions; final check PENDING |

## Closure protocol

Reinspect every row against current artifacts after the numerical jobs finish.
Replace pending rows with exact source-bound evidence, including failed or
unavailable scientific approximations. Check what each auditor actually
recomputes. Regenerate report values from terminal evidence, then check the
Markdown and rendered PDF separately. Preserve earlier failures and explicitly
record any corrected claim. The completion claim requires the full original
scope, a verified push and the final process audit; this register by itself
does not prove any of those outcomes.
