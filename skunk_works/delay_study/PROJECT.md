# TC-Lab delay study

This study compares distributed-delay descriptions of a single-heater TC-Lab.
The deliverable is a reproducible Python analysis and draft 0 of a LaTeX
manuscript, including unfavorable results and explicit uncertainty assumptions.
This file is the living entry point; `research_brief.md` records the authorized
scope. The preserved `results/manifest.json` identifies the baseline fit sweep;
each later phase has its own manifest. A final `results/study_manifest.json`
must index those phases and manuscript artifacts without replacing their records.

## Current state

- Branch: `reformulation-exploration`; fetched `origin/main` merged without
  conflicts in `02ab3d9`.
- Research started **2026-10-21 01:39:03 UTC** (October 20, 9:39 PM Eastern).
  The requested 12-hour window ends no earlier than October 21, 9:39 AM Eastern.
  Do substantive research throughout the window; do not pad it with idle waits.
  That minimum was met at 13:39:03 UTC; unfinished phases and delivery still
  determine completion.
- Hourly recovery automation: `tc-lab-overnight-research`, attached to this chat.
- Operational recovery: `get_goal` reported `blocked` at 07:35 UTC while jobs
  remained live. The user-authorized heartbeat continued the study; a fresh
  14:04 UTC goal read confirmed `active`. Preserve that goal and reconcile jobs
  rather than recreating the objective. Completion still requires every delivery
  gate, independent of elapsed time.
- Unchanged legacy simulations were run twice per dataset. `baseline/manifest.json`
  records source identity, versions and numerical results; duplicate trajectories
  were byte-identical. This is a behavior baseline, not a correctness certificate.
- Maintainer Pyomo 6.10.0 has `cov_est` but lacks the custom
  `theta_est_multistart` used by the inherited script. The new numerical study
  must not silently assume that API exists or change a shared environment.
- Notebook migration committed in `58b4fa5`: 13 deterministic Python exports,
  scientific-cell AST equality, and a successful migrated simulation run.
- Exact LTI models committed in `451749f`; inference runner in `275fe0a`.
  One hundred and thirty-eight focused checks pass after adding exact-draft expansion,
  citation-offset, independent nuisance/prediction/design/discrimination audits, and explicit
  profile-catalog selection checks. They cover independent integrator/analytic agreement,
  transformations, synthetic recovery, null directions, profile recovery,
  resampling, OU likelihoods, coverage accounting, input-design gradients and
  exact frequency-sensitivity/periodic-information identities.
- First full sweep finished 95/99 cases; four uncertainty serialization failures
  exposed a rank-deficient covariance roundoff issue. `baseline/initial_sweep_*`
  preserves the original manifest and numerical summary. Complete original
  outputs remain locally in ignored `_runs/initial_sweep_undefined_covariance/`.
  The correction uses SVD of the sensitivity matrix directly and reports undefined
  full covariance and null standard deviations along nonestimable directions.
  Fitting itself is unchanged. The corrected sweep completed **99/99** cases.
  `results/comparison_receipt.json` verifies 95 prior objectives agree within
  1.14e-13. `results/fit_summary.csv` and `coordinate_comparison.csv` are generated
  by `summarize_fits.py`, not manually maintained tables.
  Full profiles, bootstrap, structural checks, OU calibration and synthetic
  coverage have since finished. Sensitivity and prediction audits passed; see the single phase-status
  section below. The overnight study remains in progress; do not mark it
  complete early.
- Literature packet: `6f11d88ebd8741c892fea8fffc243d7d`. Initial revision:
  `d88745b5615020ca2f74fa981cd87f94d867723803a0f0efb5a668e0a09f2262`.
  Current observed revision after two searches:
  `76843554a1e57656e98b3b1ff3530f57a502823a57346e9bb74d981efd3ef80d`.
  A subsequent scope/interpretation note advanced the observed revision to
  `38a5811a157d16b7d0ffb039919d3516a445e2d087ac957e4fbebf0749fddab1`.
  Inspect the current revision before further mutations. Primary Raue (2009)
  physical pages 1-5 were inspected; profile method is on physical page 4,
  printed page 1926. Search hits that only cite a paper are not supporting evidence.
- Gmail profile verified. Finished report recipient: `adowling@nd.edu` (self).
  Send once after verification and push; save the resulting message identity.

## Layout and authority

- `notebooks/` at the repository root: original teaching notebooks;
  preserve their content during migration.
- `../tclab_reformulated.py`, `../tclab_objective_contours.py`: inherited models;
  preserve as baseline authorities until deltas have been explained.
- `baseline/`: frozen numerical behavior captures from unchanged code.
- `notebook_scripts/`: executable exports with source hashes and cell markers.
- `results/`: new computations; one run manifest is the numerical authority.
- `manuscript/`: split LaTeX sources, generated figures/tables and final PDF.
- `literature/`: search receipts, source/page mappings and literature synthesis;
  do not put a large paper collection in Git.

## Working rules

Follow `resources/practices/` in the sibling `claude-for-researchers` repository:
`working_with_ai_agents.md`, `scientific_computing_workflow.md`,
`literature_review.md`, `technical_writing.md`, `writing_style_guide.md` and
`scientific_figures_tables.md`. That repository is read-only for this task.
User authorization here explicitly permits commits, a final push, and a Gmail
message to the user; it supersedes the general practice of reserving those actions.

Keep the study inside this branch. Preserve raw CSVs and reference outputs.
The user additionally requested the **no-ai-slop** skill on October 20. Apply its
editing rules and `eval.md` checks to the manuscript, summaries and completion
email. Read each full draft before editing, remove filler and formulaic patterns,
retain the lab's voice and technical precision, and preserve uncertainty and
unfavorable findings. An editorial pass must not change a result or citation.
Separate code migration from scientific changes and explain every numerical delta.
Freeze source during measurements; record code/configuration/data hashes, seeds,
versions, failures, and convergence evidence. Keep compute separate from plotting.
Never fabricate a result, citation, confidence interval, or completion claim.
Condition numbers depend on coordinates: report units/scaling and prediction-space
comparisons. Clearly separate structural identifiability, practical estimability,
and uncertainty conditional on a noise/discrepancy model. Mark all manuscript
sections as AI drafted and awaiting human scientific review.

Commit coherent, inspected changes with an AI attribution trailer. Reconcile live
jobs after interruption rather than restarting blindly. Keep a single current-state
section here and a single sourcing manifest; Git preserves superseded states.

Reproduction dependencies, source-checkout rules, fresh-output conventions and
artifact checks are described in `REPRODUCING.md`.

## Environment

The existing base Python has NumPy 1.21.5, SciPy 1.7.3, Pandas 1.4.2,
Matplotlib 3.5.1 and Numba 0.55.1. The maintainer environment
`pyomo-doe-maint` has NumPy 2.4.3, SciPy 1.17.1, Pandas 3.0.2, Pyomo 6.10.0,
pytest 9.0.3 and a working Ipopt. Use the maintainer environment for legacy
Pyomo verification. Pin the new runner's verified environment before production.
Use `MPLBACKEND=Agg` and a writable `MPLCONFIGDIR` for headless runs.
Local `pdflatex`, `latexmk` and `pdftoppm` are available; confirm compilation and
inspect rendered pages before delivering the report.

From the repository root, run the numerical tests and fits with:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q skunk_works/delay_study/test_models.py skunk_works/delay_study/test_inference.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python skunk_works/delay_study/run_fits.py
```

The runner caps workers at four, uses a nonblocking lock, checks measured source
against HEAD, and refuses reuse across source/configuration changes. Prediction
CSVs are gzip-compressed with a fixed timestamp; all attempt/convergence records
and conditional noise assumptions remain in the case JSONs.

For long later phases, use `frozen_run.py`: it copies committed Python sources,
configuration and raw data into a temporary snapshot with read-only source files,
checks the pinned runtime, and records hashes and a reproduction command at the
output. This makes the measured source independent of subsequent edits in this
checkout. Never change the snapshot or the committed fit inputs during a job.
Run at most four long-running numerical workers total, including across different
phases. Brief unit checks may use one additional thread with BLAS capped at one;
do not use elapsed times from overlapping work as performance benchmarks.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python skunk_works/delay_study/frozen_run.py run_profiles.py --fits skunk_works/delay_study/results/fits --output skunk_works/delay_study/results/profiles
```

There are 300 profile jobs (three parameters per delayed model, four per original
physical model). Grids include global exploration and local refinement; the
one-stage cold model additionally includes the exchanged pole-label mode.
Intervals preserve disconnected components. IID Gaussian likelihood thresholds
are conditional diagnostics, not coverage guarantees for autocorrelated residuals.

## Phase status and next work

The 99 baseline fits are committed in `f30451e`. The immutable `e267580`
profile pass completed 297/300 jobs; all three original-model Ua failures were
recovered from source `d2f4f15` in a separate `results/profile_retries` phase.
Keep both manifests. No completed profile found a lower baseline minimum.
Retry records retain failed starts before complex-step Jacobians, Jacobian
scaling and a larger evaluation budget. The later experimental pole-chart
helper was not used for these measured retries.

The immutable `1b01061` bootstrap finished **240 jobs, 96,000 successful draws**
with no failed jobs. Its session 11820 has exited. The producer audit checks
saved arrays, fit identities, selected converged minima and independently
recomputed physical transforms. Four hundred draws per method cover residual
blocks 1, 10, 30, 60 and known-sigma IID Gaussian simulation for each of 48 IO
fits. Eight constant-step input-equilibrated cases correctly have undefined
delay intervals. These are conditional intervals; per-experiment centering does
not capture run-to-run mean discrepancy. Baseline/trial outputs are preserved.

The `d2f4f15` structural phase is complete in `results/structural`: 403 sampled
positive physical equivalents per calibration, analytic null tangents,
repeated-pole perturbations and a fixed-parameter chain-to-delay limit. Maximum
equivalent prediction differences are below 1.8e-12 C. Sampled parameter ranges
are not confidence intervals. Source/output receipts are committed in `9cb14b6`.

All **48 OU likelihood calibrations finished without revisit flags**, from
source `9cb14b6`, in `results/ou`; session 17895 has exited. Audit and summarize
their noise lengths, bounds, likelihoods, mean predictions and conditional/full
Hessians before extending their interpretation. The independent saved-trajectory
audit passed: maximum NLL difference 2.6e-12, selected converged minima verified,
and physical transforms checked. Joint cold innovation lag-one correlations
remain near -0.5, challenging the single-component OU covariance. All results
and the audit are committed in `89c3e8c`. No push or completion email has been sent.

The independent coverage production **finished at 14:07:12 UTC**, execution
session **19175** exited successfully, from immutable source **23629a2**, output
`results/coverage`. All **3,600 synthetic datasets** and 360 chunks finished
without failed draws or chunks:
300 per combination of n1/n2/n12/pure delay and IID/OU10/OU60 noise, with
three-coordinate local intervals and nuisance-reoptimized delay profiles.
`results/coverage_pilot` preserves the 12-draw pilot and production-size receipt.
The initial draft's 500 was changed to 300 using solver timing and a binomial
precision calculation before inspecting pilot coverage. Production seed 726533
differs from pilot seed 726532. Nominal Monte Carlo standard error near 95%
coverage is about 1.3 percentage points. Do not mix pilot and production draws.
The frozen `84cf569` summary passed its 384-row arithmetic/source audit, with
no failed/missing draws, zero LR arithmetic errors, physical-transform error
3.33e-16 and maximum selected-minimum discrepancy 7.62e-11 squared C. Its CSV
hash is bound in the audit. Figure generation from `19306cd` now accepts a
byte-identical configuration at a different path only when it matches the
production identity and an audited source digest; original inputs are still
checked. Production numeric audits passed (5,796 figure values, 48 table counts and three prose macros); color/grayscale figures and working PDF pages 7/9/17/18/24/25 passed scoped visual review. Receipts are saved with the manuscript. Final full-document QA remains pending.

The 112-job experiment-design driver and configuration are now written. Run a
frozen short trial and the full study once the coverage pool frees resources.
It considers D, E, delay and prediction criteria, prior information on/off,
nominal cold models with three working covariances, and mean/min utility over
48 model/startup/noise scenarios. Temperature assumptions are checked across
16 fitted mean models and re-simulated densely through a ramp-down tail.
These remain local numerical designs under assumed constraints, not hardware
operating instructions or globally optimal solutions.

The prepared `audit_design.py` reconstructs information directly from each
candidate command rather than the cached input-to-sensitivity map. It checks
all scenario matrices, eigenpairs and four target scores; reference normalization;
every recorded attempt objective and actual coarse resource violation; the
best-converged-feasible-start rule; and dense all-model temperature maxima through
shutdown. Normwise information/eigenpair residuals use the declared log-IO metric
to avoid treating finite-difference cancellation at an exact null as a physical
cross sensitivity; individual spectra and finite/undefined target utilities
remain separately checked. Four new checks cover analytic utilities/null targets,
dense OU precision, irregular-knot command dose/shutdown and a saved synthetic
design, including deliberate information/temperature corruption. The 116-check
suite passed in 7.31 s; the added attempt-budget guard passed targeted checks.
Production design outputs still await the queued trial and full run.
The model section now distinguishes recorded command from calibrated heat input,
identifies the exact delay factor in the pure-delay limit, and separates
input-equilibrated memory from whole-system thermal equilibrium. Working pages
2/3 were inspected in a warning-free 34-page build; the scoped receipt is
`manuscript/model_assumption_review.json`. Final full-document QA remains pending.

Additional committed follow-up phases have passed their bounded trials: the 241-case
`run_sensitivity.py` matrix (192 ambient/history cases and 49 input-hold/
discretization cases), and 131 `run_prediction_uncertainty.py` jobs (51 local
quotient cases, 48 OU cases, 32 joint bootstrap cases). All baseline linear-hold
pure-delay predictions remain unchanged after adding exact left/right holds.
Prediction variance is checked for original/n1 quotient invariance, with residual
degrees of freedom based on the proven identifiable quotient rather than an
extra redundant coordinate. A varying-input target receives no uncertainty band
when calibration leaves delay structurally unobservable, even where a local
delay derivative happens to vanish. Mean-estimator uncertainty and an additional
future observation error law are distinct outputs.

The prepared `audit_sensitivity.py` separately propagates chain states at
the fitted ambient/history initial conditions, checks every stored trajectory,
SSE, selected converged start, input identity and held-out metric, and compares
a finite-difference information matrix and numerical rank with the saved nuisance
SVD. It checks IID covariance arithmetic and exact pole-exchange predictions without claiming
a HAC audit or coverage guarantee. It refuses active/incomplete runs and
requires an explicit flag for a filtered trial. Four new checks include
regular and rank-deficient cases and deliberate covariance corruption. Production
sensitivity completed 241/241 cases without failures or revisit flags. Its independent audit passed all 241 cases: maximum trajectory difference 1.8e-12 C, SSE difference 3.4e-11 squared C, mixed information relative difference 1.4e-7 and covariance difference 1.2e-16. Full result interpretation and manuscript integration remain pending.

The prepared `audit_prediction_uncertainty.py` checks every saved mean,
variance, interval endpoint, conditioning/estimability flag and target summary.
It assembles local/HAC covariance independently, resets score lags at record
boundaries, uses identifiable-quotient degrees of freedom, and whitens OU
sensitivities directly. Bootstrap endpoints are checked against saved draws;
Gaussian-mixture prediction endpoints must satisfy their defining CDF rather
than merely reproduce the producer's inversion helper. It reuses the separately
verified simulator/sensitivities and recorded observed OU covariance. Eight
new checks cover dense irregular-time OU agreement, HAC/rank algebra, deliberate
band corruption and saved synthetic artifacts from all four output branches.
The full 112-check suite passed in 7.78 s. The full 131-case production run and its independent audit have now passed; all 4,220,249 saved points were checked. Reporting still distinguishes mean-estimator bands, future-observation bands and structurally undefined varying-input targets.

One-shot follow-up queue **session 48705** exited after all seven trials at
14:08:40 UTC. It used immutable commit **5178dc5**, starting only after the
coverage owner released its OS lock and all chunk/draw counts reconciled.
Original trials are preserved in `/private/tmp/tclab-followup-trials` and copied
to `results/followup_trials`. Both sensitivity trials and all four prediction
trials passed their independent audits. The design trial converged under its
1 s constraints but its dense audit flagged peaks up to 85.000274 C. That
flagged source/result is preserved. A common 0.01 C optimization guard now
tightens parameter/discrimination design constraints; dense acceptance remains
85 C. It has no continuous-time or hardware guarantee. The fresh guarded trial passed its separate audit in results/design_guard_trial; preserve any later flag for review.

The full **241-case sensitivity phase** from **84cf569** and the full **131-case prediction phase** from **19306cd** are terminal and independently audited. The selected 18-profile refinement from **c071c1f** has also exited successfully; its full audit is next before catalog publication. No numerical pool is live at this reconciliation. The guarded parameter-design trial passed; full design and the separate discrimination trial/design/refit pools remain.

`discrimination.py` and its driver prepare a separate model-order design phase:
12 local fixed-mean designs compare all six pairs among n1/n2/n12/pure, using
both startup assumptions and all three working covariances. A bounded 72-case
refit diagnostic lets n2/n12/pure change parameters against a fixed n1 generating
mean under constant, multisine and selected mean/min designs. It does not cover
every generating model or parameter point. Direct covariance and gradient checks
pass. An equilibrated constant-input counterexample has positive separation at
the original fitted parameters but zero separation after refitting, demonstrating
why fixed-mean scores alone can exaggerate discrimination. This phase also waits
for an inspected trial after the active pool releases; nothing runs on hardware.

The prepared `audit_discrimination.py` checks all saved fixed-pair distances
using direct mean simulation and independently assembled Gaussian precision.
It checks all candidate resources/dense maxima, each optimization objective and
coarse constraint, and best-converged-feasible selection. The refit branch
recomputes every attempt distance, declared box/active-bound and evaluation-budget
flags, the smallest converged selection and fixed-parameter upper bound. It also
requires the exact zero-distance witness for equilibrated constant input. Selected
waveforms bind to the twelve complete source designs and their independent audit.
Six short checks include dense irregular-time covariance agreement, real saved
synthetic design/refit artifacts and deliberate distance, probability, budget,
active-bound and source-waveform corruption. These prepare validation; no
production discrimination outputs have yet been computed or audited. The scope
excludes global optimality and an independent derivative check of each saved
refit optimality scalar.

The frequency-information appendix now derives log-transfer sensitivities, the
low-frequency combined-lag expansion and an exact ideal periodic phase-sum
information matrix. A DC component plus one nonzero tone gives local full rank
away from the finite-chain repeated pole; a lone unoffset tone has rank at most
two. At tau=theta/n the two lag sensitivity columns are proportional at every
frequency, so more tones do not remove that first-order singularity. This does
not resolve the n1 global pole exchange, prove finite-record precision, or replace
the initialized constrained-design simulations. Three analytic checks compare
parameter perturbations, determinants and explicit real phase sensitivities;
there are no new fits or numerical pools. The full suite is 131/131 in 8.26 s.
Working pages 32/33 of a warning-free 35-page build were visually read; the
partial receipt is `manuscript/frequency_information_review.json`. Final full
manuscript review remains pending.

A bounded working-manuscript review clarified notation and terminology: the
initial-history deficit is now d_n, distinct from the delay kernel h_n; the IO,
SSE, RMSE, IID, HAC, ACF, OU, SLSQP and DC terms are defined where used. The
SLSQP expansion was checked against the installed SciPy 1.7.3 documentation.
Methods refer to recorded heater command rather than measured heat input, and
the held-out comparison explicitly concerns pooled error rather than every
file. Two generic explanatory sentences were cut using no-ai-slop guidance.
Across all 22 expanded TeX inputs, numeric-literal sequences are unchanged
apart from the explicit gain unit-vector definition. All 24 citation occurrences
are unchanged. Eleven affected working pages were read in a warning-free
35-page build; `manuscript/notation_editorial_review.json` records the limited
review. It does not replace final empirical writing, exact-draft citation
binding or full-document review after all phases finish.

Discussion of the completed bootstrap, working-OU and validation phases is now
drafted. It distinguishes block-length sensitivity from contrasts that also
change the error scale, and explains why within-record centering omits the
between-record offsets observed in validation. The discussion retains the
negative OU innovation diagnostic and the absence of a prospective file-mean
correction. No new intervals, coverage rates or future-design outcomes were
introduced. The measured bootstrap source/configuration and existing artifact
audits support this bounded interpretation. Working pages 25/26 were read in
a warning-free 36-page build; `manuscript/discussion_partial_review.json`
records the source/citation checks and no-ai-slop pass. The final discussion
still needs the remaining empirical phases and author review.

While coverage runs: complete the design-driver review, literature synthesis,
noise/validation and structural summaries, result figures/tables, ambient and
heater-calibration uncertainty derivations, and manuscript editing. Further
long numerical phases wait for the current pool; brief single-thread unit
checks may run. Remaining model-mismatch computations can follow coverage.

The merged profile catalog and arithmetic audit cover **300/300** profiles,
including the three documented retries. An analytic cross-check found that all
nine redundant original-parameter profiles have coarse interpolated endpoints
that exclude some exact equivalent family members. The sampled flat objectives
are correct, but those endpoints must not be presented as confidence bounds.
`profile_audit.json` records both arithmetic success and these interpretation
failures. Refine crossings if numerical endpoint claims are needed. Results and
catalog are committed in `183e6b7`.

`original_family_bounds.py` now partitions the equivalent physical family at
all box-boundary roots and obtains exact compatible-coordinate ranges, including
disconnected feasible intervals. `profile_refinement.py` inserts independently
checked feasible witnesses and bisects observed likelihood crossings under both
variance conventions. Four focused numerical tests and the full 87-test suite
pass. The prepared 18-profile phase targets the nine offending physical profiles,
all six cold one-stage delay profiles, and joint cold n2/n12/pure-delay examples.
It has **not been launched** while coverage owns the four-worker budget. The
separate auditor must pass before `selected_profile_index.json` can replace any
profile selection for later tables or plots; the original catalog remains intact.

Profile-dependent table/figure producers and their paired auditors now accept
`--profile-index selected_profile_index.json` explicitly. Their default remains
the preserved first-pass catalog for reproduction; no automatic switch occurs
when refinement files appear. `profile_catalog.py` checks all 300 selections,
consumed file hashes, the complete audited 18-profile refinement and unchanged
282 selections, and rejects flagged coarse physical endpoints or a mismatched
artifact catalog. Refinement catalogs retain receipt paths as well as hashes,
so an audited retry directory can be selected. The baseline table producer and
auditor own exactly five files; they no longer claim other producers' tables.
A missing or duplicate model row now fails the table audit. Six new checks pass;
the full suite is 122/122. `baseline/profile_catalog_migration_receipt.json`
records five byte-identical regenerated tables, 150 printed cells and 434
preserved figure values checked against the original selection. Existing
manuscript artifacts were not replaced. After the refinement audit passes,
regenerate the five tables and four uncertainty figures with the explicit
selected catalog, rerun their paired numeric audits with the same flag, and
perform fresh visual QA before updating manuscript interval claims.

The prepared coverage summarizer refuses an active or interrupted run and checks
all replicate identities, generating truths, seeds, physical transformations and
likelihood arithmetic. It reconciles saved files with each terminal chunk so
file loss cannot be relabeled as an optimizer failure. Its 384 planned summary
rows retain failed/undefined outcomes in unconditional bounds and give
method-defined coverage, Wilson intervals, bias/empirical SD, widths and a
secondary grid-interpolation sensitivity. Production summarization began only
after the prespecified run became terminal. Four new accounting tests
pass. A separate design check verifies that log D, delay-target and transformed
prediction utilities and gradients agree in IO and physical metrics, while E
changes with the metric. The methods section now records the complete planned
coverage, parameter-design and discrimination matrices.

The prepared `plot_coverage.py` refuses active, partial or unbound summaries.
After terminal summarization, it selects all 84 IO delay rows (four model
points, three true covariances and seven interval methods), with conditional
Wilson bars and a separate median-width view. Hollow symbols mark incomplete
method denominators; NA remains undefined. Companion JSON/CSV retain exact
counts, unconditional failure bounds, component-sum versus envelope widths,
exploration flags and interpolation sensitivity. The summary now binds its CSV
hash in its audit. `audit_coverage_figures.py` independently recomputes these
quantities from saved interval endpoints, solves Wilson score roots and a local
quadratic interpolation system, and checks both exported data formats. Seven new
checks include a full all-worker-failed artifact fixture and altered undefined
values; the full suite passed 138/138 in 8.17 s. Synthetic-only color/grayscale
layout inspection corrected overlapping width labels. Shared width limits use
all panels' data, with a regression check against clipping later noise rows.
Its scoped receipt is
`baseline/coverage_artifact_preflight.json`; it is not production outcome or
final figure validation. Production outcome summarization started only after
the coverage run was terminal.

The small deterministic onset phase from `a0f01fe` is complete in
`results/onset_limit`: 18 exact Erlang-CDF calculations, independently checked
against quadrature and state propagation. The derived onset error is n^-1/2;
its scaled value at n=65,536 is within 0.03% of the CLT coefficient. This is a
cold constant-step result at fixed parameters, not a refitted trajectory error.

The `75f7da6` family-function phase is also complete in
`results/family_uncertainty`: it reuses 6,000 saved cold one-stage draws from
15 case/method jobs, with no new refits, to quantify uncertainty in identifiable
transfer coefficients and analytic positive-family boundary functions. These
are not intervals for an arbitrarily selected original physical parameter.
For joint block 60, for example, CpS supremum is 2.783 J/K with conditional
percentiles [2.647, 2.910], and Ub supremum is 0.07451 W/K with percentiles
[0.06801, 0.08156]. The full positive-family bounds precede the original numerical
fit boxes. Two independent reconstruction/calibration-scaling tests pass.
Add the boundary-function interpretation and formulas to the manuscript/figures.

Four public PDFs were acquired in `/private/tmp/tclab-literature`, registered as
source `tclab-lti-study-20261021`. The local review plan is
`c49dff1f62c741bbb3d942f65df59e8c`; its four sources are distinct from the local
library. Cross-check `b9f398dcc5274c3bc5272bc53e931922` used an OCR judge and
recommends pypdf for Hjalmarsson, Hurtado, and Manchester; pdftotext for Huang.
Its unresolved Huang page 35 is not used for a claim. Other disputed pages remain
bounded extraction evidence, not verified mathematical text. Hjalmarsson physical
pages 8-9 and Hurtado pages 11-12 were inspected as rendered images as well as text.
The paper hashes and private staging workspace are in
`literature/acquired_sources.json`. No shared-library publication was performed;
this task needs a source-grounded review, not maintenance of the shared corpus.
Literature next: ground Raue's method passages, inspect Garnier/Wang chapter
bylines, obtain/check the published Gerencser/Hjalmarsson/Huang (2017) version
(the downloaded 2019 arXiv revision differs in author order), and review primary
sources for dependent-data uncertainty. `literature/source_review.json` needs
its pending discovery statuses reconciled with these later source inspections.
The chapter-one residual-source passage at physical page 28 (printed page 2)
was also rendered, checked, grounded as fine shard
`c6d3d2dabc24623261caeabeaa3fbed5c7151bf187cbe2d807913f9ab9006b0c`,
and read back with the selected source event and exact stored-text offsets.
`literature/garnier_residual_evidence_receipt.json` records the scoped receipt
without publishing source text. The book carrier does not replace the chapter
authors/key. Final exact-draft packet claim bindings remain pending.
Raue's nuisance-reoptimization passage is now grounded in fine shard
`2ec259a5dedd82b02a891e141629a060ac4cb8beabf6ff26a7c1077636036afe`.
The published Newey-West (1987) scan was acquired from a University of Wisconsin
course source, OCR-read and visually checked at physical pages 1-3 (printed
703-705). Equation (5) supplies the Bartlett weights; Theorem 2 states conditions
that our finite-bandwidth sensitivity checks do not establish. The source hashes,
preprint/journal distinction, and access gaps for Kunsch and the adaptive-design
journal paper are recorded in the literature receipts. A parallel bibliography
lookup was cancelled after several minutes without a result; do not interpret
that attempt as evidence of absence from the corpus.
The configured literature server is maintenance 0.5.8; do not change its runtime,
models, taxonomy, existing source records or configuration to bypass limitations.

## Manuscript argument and evidence map

Split LaTeX sources now exist in `manuscript/`. The desktop compiler succeeded
on the initial working draft, which explicitly marks unfinished analyses and
AI authorship. It is not the completed report. Generate result tables/figures
from saved numerical outputs and replace status placeholders before delivery.

- Transfer-family equivalence and physical ambiguity: model equations and an
  explicit equivalence class in `sections/structural_derivations.tex`, to be
  accompanied by a parameter-manifold/prediction figure.
- Order and initialization comparison: generated fit table, RMSE/order plot,
  calibration overlays, and a separate held-out experiment plot.
- Coordinate effects and estimability: scaled information spectra/directions,
  transformed confidence geometry, and numerical coordinate-equivalence table.
- Conditional uncertainty and its limits: disconnected profiles, block-length
  comparisons, parametric/residual bootstrap intervals, residual diagnostics,
  and sensitivity to ambient/prehistory/noise assumptions.
- Prospective design: candidate input/temperature traces, audited resource
  constraints, information improvements, model-discrimination scores, and
  parameter/model sensitivity of the proposed inputs.
- Literature: source/claim/page audit distinguishing theoretical assumptions,
  our derivations, and observed TC-Lab results. Do not cite metadata-only leads.

## Verification and manuscript work

The lab's DOI checker verified **4/4 cited entries as MATCH** against Crossref;
`literature/doi_results.json` preserves the result. This checks reference identity,
not whether a source supports a particular manuscript statement.
Three added references were checked directly against publisher/source pages.
The expanded checker command was rejected by automatic review before execution
because its User-Agent would transmit the user's email to Crossref. No workaround
or indirect retry was performed. `added_reference_identity_review.json` records
the completed safer identity checks and their scope; do not claim a seven-entry
Crossref run. The added Politis--Romano journal scan was OCR-read and all three
method pages rendered and checked. Its Remark 2 describes the fixed circular
relative; geometric stationary-bootstrap guarantees are not transferred to it.

The initial 8-page manuscript was rendered and all eight pages inspected. A long
source filename caused one overfull line, which was corrected with breakable
paths. Later mathematical additions compile without overfull-line or unresolved
reference warnings, but they and the final results still require rendered review.
The working `main.pdf` is ignored; save a separately named final report only
after the full manuscript and no-ai-slop/source/result audits are complete.

Baseline figures now exist and are included in the working results section.
`plot_baseline.py` reads only saved outputs. It uses the lab's supplied figure
style module, with centralized additions for dense traces and a multipanel size.
Its latest source is `f8ee8c1`; phase-specific
`manuscript/figures/baseline_source_snapshot.json` is authoritative for restyling
(the directory's generic `source_snapshot.json` preserves the earlier first launch).
`audit_baseline_figures.py` checked 48 RMSE points against independent case JSONs
and 32 trajectory arrays against hash-verified prediction files; all deltas were
zero. Color and grayscale figures and their rendered manuscript pages 6-7 were
inspected. `baseline_numeric_audit.json` and `baseline_visual_audit.json` retain
the separate checks. The remaining full document and future figures need their
own final audit; this is not human scientific review.

The last compiled working manuscript is **34 pages**, without warnings. New
generated tables cover calibration errors, parameters, coordinate metrics,
conditional delay uncertainty and OU innovations. `audit_tables.py` checked
150 printed cells using independent result paths, including independent physical
fits and covariance-based moment conditioning. Pages 6, 9 and 10 were visually
checked for tables; pages 14-16 for the added derivations. This is partial draft
QA, not the final full-document review. The final PDF has not been named or sent.
Four additional uncertainty figures display pole modes, conditional delay
intervals, positive information spectra and weak directions. Their producer
source is `6dd8633`, and an independent audit checks 434 plotted values. Color,
grayscale and working PDF pages 10/12/13/14 were inspected. Vector heatmap cells
avoid raster seams observed in the earlier PDF attempt. This adds partial
figure QA, not a full-document final review or human scientific review.
Three structural figures and two additional tables now report the physical
equivalence family, identifiable boundary-function uncertainty, parameter
correlations, quartic repeated-pole curvature and the step-onset limit. Their
producer is `c900ff9`; independent audits verify 2,781 plotted values and 44
table numbers. The latter uses original bootstrap draws and independently
fitted physical-coordinate correlations (maximum difference 9.3e-9). Color,
grayscale and rendered working pages 9/11/13/14/18 were inspected; receipts
identify the exact temporary 25-page PDF snapshot. A wording correction
clarifies that the onset rate excludes a uniform O(n^-1) bound, rather than
claiming the temperature response fails to converge uniformly.
The literature comparison now has seven question-driven rows that separate
inspected theory, assumptions and our TC-Lab use. Rendered pages 26/27 were
checked after fixing stretched narrow-column text. Raue physical page 4,
printed 1926, was additionally rendered to verify equation (10) and its nuisance
reoptimization paragraph. `literature/visual_source_checks.json` retains the
source-page and working-PDF hashes. The exact final-draft packet still needs
all citation occurrences, claim bindings, available fine evidence, explicit
external-source gaps and scoped export. No full-document final QA, final
no-ai-slop pass, push or completion email has occurred.
Held-out error and working-OU correlation figures now use source `66d9f4d`.
Their independent audit recomputes 701 values from saved prediction trajectories
and raw file metadata; maximum error is 7.2e-15. Four rendered working pages
11/12/13/23 and both grayscale previews pass partial QA. Pooled held-out RMSE is
3.36-3.41 C for the four representative joint cold models, compared with
0.41-0.53 C on calibration. Per-file mean residuals contribute 80.1-84.8 percent
of pooled squared error algebraically; this is not a causal diagnosis or
prospective bias correction. Generated tables retain every file code and the
pooling definition. A derived IID-component filtering example explains how an
OU filter can induce lag-one correlation near -1/2 without identifying the
actual residual mechanism. The source-snapshot and visual-QA receipts identify
the exact 32-page temporary PDF; the final document still needs full review.

The nuisance appendix now derives the admissible one-stage history/pole
exchange and the constant-command gain/ambient/history continuum, including
the zero-initial-command exception and explicit parameter-domain restrictions.
The methods specify the 241-case matrix before its production outcomes. Eight
existing independent nuisance checks pass. Working pages 4/5/29/30/31 were read
and the final compile has no warnings; `manuscript/nuisance_derivation_review.json`
identifies this partial 34-page PDF review. It does not establish uncertainty
coverage or finish the pending nuisance computation.

The exact-draft preflight expands controlled LaTeX inputs without executing
them, hashes every included source, checks citation offsets and bibliography
keys, and retains the chapter-to-book carrier alias. The current working draft
has 24 citation occurrences across seven works and 21 included TeX files.
This is a preflight count, not a completed evidence or scientific-support audit.
Run `prepare_manuscript_audit.py` again after the final editorial pass, then save
that exact flattened text with `set_packet_draft` and audit every occurrence.
Two works have mapped ingested carriers; five primary PDFs were read externally.
Keep source-availability and fine-grounding gaps explicit rather than relabeling
an externally inspected PDF as an ingested source. Raw evidence exports may
contain copyrighted passages and belong in ignored `_private/`, with a scoped
public receipt and the manuscript bibliography retaining checked metadata.

For the final packet, bounded planning/grounding tools were discovered in the
configured runtime: `prepare_packet_evidence_batch` saves exact revision/claim-
bound requests, and `run_packet_evidence_batch` plans them from one validated
source view without shared grounding. Inspect every resulting plan before using
`prepare_packet_grounding_batch` and `run_packet_grounding_batch`; the latter
accepts at most 20 pending items per call and requires the current revision.
Use the existing Raue/Garnier fine passages where they support the actual clause,
without transferring one passage's support to unrelated source/metadata claims.
The five externally read works still need explicit carrier/fine-availability
issues, not invented source events. Read `research-common/bounded-work.md` in
the literature skill for recovery after uncertain calls. These are discovered
capabilities, not executed final-draft batches; no batch or claim has been saved
against the unfinished draft. Recheck tool availability and the packet head.

Do not expand the numerical comparison matrix further by default. Complete the
already declared coverage, sensitivity, prediction, parameter-design, selected
profile refinement and discrimination phases, resolve numerical issues found by
their trials/audits, and finish the evidence packet, report and delivery. An
OU-plus-nugget fit, per-file retrospective refits and hardware experiments remain
future questions rather than additional completion requirements.
Added derivations cover heater-calibration scale ambiguity, ambient/history
nuisance information, zero-mean colored-error least-squares interpretation,
quartic behavior at coincident poles, and the onset convergence coefficient.
The saved literature packet still has no exact draft/citation inventory; finish
its synthesis/claim/grounding/export workflow before reporting it complete.
A fresh metadata-only regex search (`snapshot_id` starting `1978ed7b`) found
no current title matches for the five external cited sources across 2,899
indexed papers, with no metadata issues. It does not validate passage/PDF state
or remote synchronization. The full response remains in the active tool store
as `studyCitedBibliographySearch`. Those works were read from hash-identified
primary PDFs; do not imply they have saved fine grounding in the shared corpus.

Coverage reporting must retain method-specific undefined intervals and optimizer
failures, unconditional coverage bounds for failures, binomial uncertainty,
bias/width summaries and the distinction between component and envelope widths
for disconnected profiles. A valid result at these chosen generating points
does not establish coverage throughout parameter space or for real discrepancy.

Priority work after these phases: ambient and prehistory sensitivity, a correlated
residual likelihood using actual time intervals, per-file validation diagnostics,
and finite-horizon input design. The extra sensor trace cannot be assumed to
measure the original model's internal heater state. The supplied validation
filenames mostly identify environment 2; a between-environment explanation for
their errors is not established. Data resolution and unrecorded run conditions
also need inspection before assigning all residual structure to white sensor noise.

## Current phase reconciliation

At this reconciliation, no prior numerical process remained live (process inventory checked). The full 131-case prediction-uncertainty run is terminal and independently audited: all 131 cases passed, with Gaussian-mixture CDF residual below 7.9e-12. The 241-case sensitivity and 3,600-dataset coverage phases are also terminal and audited. Their immutable output identities remain in their own manifests.

The original design trial is preserved because it passed its 1 s constraints but exceeded 85 C by 0.000274 C on the independent 0.25 s grid. The committed 0.01 C optimization guard retains the independent 85 C acceptance limit. The guarded trial passed its separate audit; full parameter design is eligible after the selected-profile pool exits. The discrimination trial remains required before its full design/refit pools.

The selected physical CpH refinement trial passed its producer flags and analytic-family inclusion check. The full 18-profile refinement exited successfully from source c071c1f, execution session 23765. Its separate full audit must pass before publishing selected_profile_index.json or regenerating profile-dependent manuscript figures/tables. First-pass files and trial outputs remain preserved.
