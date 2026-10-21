# Reproducing the TC-Lab study

Read `PROJECT.md` for current completion state and live process ownership.
This guide describes the calculation dependencies; a command listed here is
not evidence that its phase has finished. `research_brief.md` fixes the study
scope. Preserve the supplied CSVs, saved first-pass results and numerical
failures when reproducing or extending a phase.

## Runtime and short checks

`environment.json` records Python 3.9.12 and exact numerical package versions.
The verified interpreter on the study host is
`/Users/adowling/opt/anaconda3/bin/python`. Use a separate compatible environment
on another machine. The inherited Pyomo behavior checks used a different
environment, recorded in `baseline/manifest.json`; they do not require changing
the new study's numerical runtime.

From the repository root:

```sh
study_python=/Users/adowling/opt/anaconda3/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/tclab-matplotlib
"$study_python" -m pytest -q skunk_works/delay_study/test_*.py
```

The focused suite uses small synthetic examples and committed reference fits.
It checks simulation, transformations, structural counterexamples, derivatives,
uncertainty accounting, design criteria and artifact auditing. It does not rerun
the full Monte Carlo study. The notebook exporter and its independent scientific
cell checks are described in `notebook_scripts/README.md`.

`frequency_information.py` implements transfer-function derivative and determinant
identities used by the frequency-information appendix. Its three short checks
compare numerical parameter perturbations and exact real periodic phase sums;
they do not fit more data or assume the prospective-design records are in
periodic steady state.

## Measured source and fresh output directories

Each phase manifest identifies its numerical source/configuration and input
hashes. For frozen runs, the corresponding `source_snapshot.json` also names
the commit, runtime and reproduction command. Artifact directories can contain
several generations: use the phase-specific source snapshot named by their
figure/table manifest rather than assuming the directory's first generic
snapshot produced every later artifact.

`frozen_run.py` snapshots **HEAD**, not an arbitrary revision. To reproduce a
historical phase, create a detached checkout at that phase's measured commit
and invoke its driver there. For example, the baseline fit source is the
`identity.commit` in `results/manifest.json`; the commit that later added the
result files is a different identity. A clean detached checkout is required
for `run_fits.py`, whose source guard uses Git. The later drivers can run from
the read-only-file snapshots created by `frozen_run.py`.

Write recomputations to fresh directories outside the saved production outputs.
For later phases, explicitly pass absolute input/output paths: default paths in
a frozen snapshot refer to that temporary snapshot, which does not contain the
production result collection. This example starts a new parameter-design run;
it must wait until the four-worker budget is free and its trial is inspected:

```sh
"$study_python" skunk_works/delay_study/frozen_run.py run_design.py \
  --fits /absolute/path/to/saved/results/fits \
  --output /absolute/path/to/fresh/design
```

The snapshot's scientific source hashes must match the measured phase when
reusing results. A later commit that only changes prose may retain identical
scientific hashes. Do not replace a recorded source identity with a later commit
merely because both commits contain the driver. Retain source/hash receipts
when an inspected trial is reused in production.

The runners enforce per-output locks and source/input identity. A lock file's
existence does not prove ownership; inspect the held OS lock and the live
execution session. Limit long numerical work to four worker processes in total
across phases. A filtered `--only` run is a trial, not a completed comparison.
Keep failed attempts and failed chunks in subsequent summaries.

## Calculation dependencies

| Phase | Inputs and authority | Driver and checks |
| --- | --- | --- |
| Inherited behavior | Original simulations and untouched CSVs | `baseline/manifest.json`; two runs per calibration with identical trajectories |
| Baseline comparison | 99 cases in `config.json`; measured source in `results/manifest.json` | `run_fits.py`, `summarize_fits.py`; `results/comparison_receipt.json` checks the preserved initial sweep |
| Profiles | Baseline fit JSONs; initial pass plus documented original-model retries | `run_profiles.py`, `summarize_profiles.py`; retain `profile_index.json` and its interpretation flags |
| Selected profile refinement | 18 declared profiles and their fit/first-pass hashes | `run_profile_refinement.py`, `audit_profile_refinement.py`; publish `selected_profile_index.json` only after the audit passes |
| Bootstrap | 48 IO fits, five methods, 400 draws each | `run_bootstrap.py`, `summarize_bootstrap.py`; saved draw/attempt arrays and exact physical transforms |
| Structural calculations | Fitted transfer functions and fixed synthetic limits | `run_structural.py`, `run_onset_limit.py`; analytic equivalents, state/quad checks and source receipts |
| Family-function uncertainty | Saved cold n1 bootstrap draws; no new fits | `run_family_uncertainty.py`; identifiable transfer functions and family-boundary functions |
| OU likelihood | 48 IO fits and retained observation times | `run_ou.py`, `summarize_ou.py`; saved-trajectory likelihood and covariance audit |
| Synthetic coverage | Four cold generators, three true noise processes, 300 draws each | `run_coverage.py`, `summarize_coverage.py`; complete terminal accounting before summarizing 3,600 draws; `plot_coverage.py` and `audit_coverage_figures.py` select/check all 84 IO delay rows |
| Assumption sensitivity | 241 tasks in `sensitivity_config.json` and baseline fits | `run_sensitivity.py`, `audit_sensitivity.py`; independent chain initial-state propagation, finite-difference information/rank, all-start/SSE and held-out checks |
| Prediction uncertainty | Baseline/OU fits and completed bootstrap arrays | `run_prediction_uncertainty.py`, `audit_prediction_uncertainty.py`; 131 jobs, global estimability checks, separate mean/observation bands and all-point audits |
| Parameter design | Joint fits and `design_config.json` | `run_design.py`, `audit_design.py`; 112 jobs, direct information/resource/selection audits and all-model dense tail simulation |
| Order discrimination | Joint fits and both design configurations | `run_discrimination.py --phase designs`, then `--phase refits`; `audit_discrimination.py --phase designs/refits` directly checks saved means, distances, resources and all attempts; 12 designs must pass before 72 directional alternative fits |

Selected-profile refinement produces a **separate** catalog; do not overwrite
the first-pass catalog or use its flagged coarse physical endpoints as final
confidence limits. Pass `--profile-index selected_profile_index.json` to
`render_tables.py`, `plot_uncertainty.py` and their paired auditors after the
refinement audit passes. Defaults preserve the first-pass catalog; adding a
refinement file does not silently alter a figure or table. Manifests record
the selected catalog, and a paired audit rejects a different selection. The
baseline table producer owns exactly five files in the shared table directory.
A family-equivalence witness is a feasible equal-prediction
point, not proof of a globally minimized nuisance objective.

Coverage summarization rejects active, interrupted or filtered runs. Its failure
bounds keep every requested dataset in the denominator. Bootstrap summaries are
separate from synthetic coverage; the latter does not measure coverage of the
real-data block-bootstrap intervals. The sensitivity auditor needs
`--allow-trial` to label a filtered run as a trial audit.

The discrimination auditor also requires `--allow-trial` for filtered outputs.
Its refit audit requires all twelve source designs to match their completed
manifest and independent audit. It recomputes every recorded attempt's precision
distance and active bounds, but does not independently differentiate the saved
optimizer optimality scalar. A smallest converged refit distance is an upper
bound on the global closest alternative; only the fixed-mean distance receives
the ideal common-Gaussian equal-prior error calculation.

Compare recomputed objectives, trajectories and audited quantities using the
recorded tolerances. Elapsed times, timestamps, paths and some floating-point
results can differ; complete result JSONs need not be byte-identical.

## Manuscript and source audit

Figure/table producers read saved calculations rather than rerunning fits.
Their manifests, numeric audits and partial visual receipts are separate
artifacts. Regenerate a changed figure/table from its declared input selection,
run its independent auditor, inspect color/grayscale views and read its rendered
manuscript pages. A successful compiler invocation checks none of those claims.

```sh
cd skunk_works/delay_study/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The working `main.pdf` is ignored. A final named PDF requires full-document
visual review, numerical/source checks and the no-ai-slop editorial pass.
`prepare_manuscript_audit.py` expands the exact saved TeX inputs without executing
them and records source hashes, bibliography mappings and citation offsets.
After final editing, bind that exact draft to the saved evidence packet, review
every citation occurrence and claim, and export the scoped audit. A draft change
requires rebinding; bibliography identity and quote matching alone do not verify
claim support.

The configured literature runtime exposes bounded packet evidence/grounding
batches. After exact-draft claim binding, plan a reviewed quote queue from one
validated source view, inspect the plans, and apply at most 20 pending grounding
items per call with the actual packet revision. Inspect saved batch/packet state
after uncertain completion before retrying. This batches evidence operations;
it does not authorize new PDF ingestion or remove source-availability gaps.

Two cited works have ingested source carriers; five inspected primary PDFs are
externally staged. Their available text and missing fine-grounding bindings must
remain distinct. Keep copyrighted raw packet exports in ignored `_private/` and
publish only scoped receipts and checked reference metadata. Reproduction of the
analysis does not send email or publish changes to the shared literature library.

## Final report producers

The corrected full parameter-design collection is `results/design_final`;
`results/design` preserves the initial 96 completed cases and sixteen
near-coincident shutdown-grid exceptions. The numerical repair rounds the dense
report/audit grid to the simulator's existing one-nanosecond interval resolution
before deduplication. The independent acceptance limit remains 85 C.

`report_followups.py` consumes the audited sensitivity and prediction records;
`audit_followup_report.py` checks its scalar references and generated tables.
`render_coverage_table.py` and `audit_coverage_table.py` format and verify exact
coverage counts separately from the coverage-figure producer.

After the 112 parameter designs, twelve discrimination designs and 72
alternative refits have passed their own full audits,
`summarize_design_report.py` prepares saved scalar summaries and deterministic
sixteen-model temperature envelopes. Pass an absolute `--results` path and a
fresh `--output` directory. `plot_design_results.py --summary ABSOLUTE_PATH
--output FIGURE_DIRECTORY --tables TABLE_DIRECTORY` only formats those outputs.
`audit_design_report.py` checks source scalars, table cells and envelope peak
receipts; its scope excludes an independent all-time integrator and global
optimization. Design temperatures remain model-scenario calculations.

The final exact-draft evidence receipt records citation inspection and remaining
carrier/fine-grounding gaps separately from DOI identity checks and human
scientific review. Report reproduction does not publish source PDFs or email
the report.
