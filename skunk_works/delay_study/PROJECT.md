# TC-Lab delay study

This study compares distributed-delay descriptions of a single-heater TC-Lab.
The deliverable is a reproducible Python analysis and draft 0 of a LaTeX
manuscript, including unfavorable results and explicit uncertainty assumptions.
This file is the living entry point; `research_brief.md` records the authorized
scope, and `results/manifest.json` will identify manuscript results.

## Current state

- Branch: `reformulation-exploration`; fetched `origin/main` merged without
  conflicts in `02ab3d9`.
- Research started **2026-10-21 01:39:03 UTC** (October 20, 9:39 PM Eastern).
  The requested 12-hour window ends no earlier than October 21, 9:39 AM Eastern.
  Do substantive research throughout the window; do not pad it with idle waits.
- Hourly recovery automation: `tc-lab-overnight-research`, attached to this chat.
- Unchanged legacy simulations were run twice per dataset. `baseline/manifest.json`
  records source identity, versions and numerical results; duplicate trajectories
  were byte-identical. This is a behavior baseline, not a correctness certificate.
- Maintainer Pyomo 6.10.0 has `cov_est` but lacks the custom
  `theta_est_multistart` used by the inherited script. The new numerical study
  must not silently assume that API exists or change a shared environment.
- Notebook migration committed in `58b4fa5`: 13 deterministic Python exports,
  scientific-cell AST equality, and a successful migrated simulation run.
- Exact LTI models committed in `451749f`; inference runner in `275fe0a`.
  Forty-eight focused checks passed on October 21 before the full design-driver
  checks were added. They cover independent integrator/analytic agreement,
  transformations, synthetic recovery, null directions, profile recovery,
  resampling, OU likelihoods, coverage accounting and input-design gradients.
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
  Full profiles, bootstrap, structural checks and OU calibration have since
  finished. Independent synthetic coverage is live; see the single phase-status
  section below. The overnight goal remains active; do not mark it complete early.
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
Hessians before citing them. The OU covariance is a working assumption and
cannot repair mean-model bias. No push or completion email has been sent.

The independent coverage production is **live**, execution session **19175**,
from immutable source **23629a2**, output `results/coverage`. It has four workers;
do not start another numerical pool. There are **3,600 synthetic datasets**:
300 per combination of n1/n2/n12/pure delay and IID/OU10/OU60 noise, with
three-coordinate local intervals and nuisance-reoptimized delay profiles.
`results/coverage_pilot` preserves the 12-draw pilot and production-size receipt.
The initial draft's 500 was changed to 300 using solver timing and a binomial
precision calculation before inspecting pilot coverage. Production seed 726533
differs from pilot seed 726532. Nominal Monte Carlo standard error near 95%
coverage is about 1.3 percentage points. Do not mix pilot and production draws.

The 112-job experiment-design driver and configuration are now written. Run a
frozen short trial and the full study once the coverage pool frees resources.
It considers D, E, delay and prediction criteria, prior information on/off,
nominal cold models with three working covariances, and mean/min utility over
48 model/startup/noise scenarios. Temperature assumptions are checked across
16 fitted mean models and re-simulated densely through a ramp-down tail.
These remain local numerical designs under assumed constraints, not hardware
operating instructions or globally optimal solutions.

While coverage runs: complete the design-driver review, literature synthesis,
noise/validation and structural summaries, result figures/tables, ambient and
heater-calibration uncertainty derivations, and manuscript editing. Further
long numerical phases wait for the current pool; brief single-thread unit
checks may run. Remaining model-mismatch computations can follow coverage.

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
