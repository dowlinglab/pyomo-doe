# TC-Lab delay study

The deliverable is a reproducible Python study and draft 0 of a LaTeX manuscript
with literature, methods, results and discussion. `research_brief.md` owns the
requested scope; `REPRODUCING.md` describes numerical dependencies and commands.
This file owns current execution state. Git preserves superseded states.

## Current state

- The user requested an updated report and a new self-email after the teaching
  follow-up. The current source adds Appendix G, generated tables and the four
  native Pyomo.DoE experiment plots. The updated deliverable is
  `manuscript/TC_Lab_delay_study_draft0_updated.pdf`; its separate source,
  numerical-audit and visual-review bindings are in
  `teaching_followup/report_revision.json`. The original named PDF, study release
  index and original email receipt below describe the earlier frozen release.
  This explicit request authorizes one new email containing the revised report.
- The updated build has 53 pages. The original 48 pages are pixel-identical at
  the checked rendering resolution; all five added pages passed visual review.
  No numerical optimization or fitting was rerun for this report revision.
- The revised exact draft is bound to native literature packet revision
  `e7c5813e985dc6a68fa8cc1c0432b95dadbe7f3d762f8201360d1308c5a649b1`.
  All 24 unchanged cited claims were re-anchored. Seven occurrences reuse two
  checked source passages; seventeen retain the original unavailable-carrier
  issues. The inspected export and scoped receipt are recorded in
  `teaching_followup/report_audit/packet_receipt.json`. Human scientific review
  remains pending.
- Updated report and verification records were committed and pushed as
  `fabbfc1bc302c26618a6d65f859c7b9ecab16c9c`. Gmail message
  `1a15fb844da39b96` delivered the revised PDF to `adowling@nd.edu`; readback
  confirmed its filename and 1,159,539-byte attachment. Sent search found one
  update message. `teaching_followup/delivery_receipt.json` owns this delivery.
  The user's report-update request is complete; do not resend the update.
- Branch: `reformulation-exploration`; fetched `origin/main` merged in `02ab3d9`.
- Research began **2026-10-21 01:39:03 UTC**. The requested twelve-hour minimum
  elapsed at **13:39:03 UTC**. Completion still requires every delivery gate.
- Goal and recovery-heartbeat lifecycle are managed in the app. The heartbeat
  is `tc-lab-overnight-research`; do not create a duplicate goal or automation.
- **Every declared numerical comparison is terminal and independently audited.**
  No numerical fitting/optimization pool remains live. Report-formatting sessions
  have exited. The full-PDF page render (session **19929**) is terminal; all 48 pages were inspected.
- The complete working manuscript compiles to **48 pages** without warnings,
  unresolved references or overfull/underfull lines. It has received an agent
  editorial/no-ai-slop pass; full rendered-page review has passed. It remains
  AI drafted and awaiting human scientific review.
- Exact manuscript preflight: **31 included TeX files, 24 citation occurrences,
  seven cited works**, expanded draft SHA-256
  `b61db178c8e7a3482e890245352d650237dc9e753cd9c63d87a4056fd8d52dc7`.
  Files are in `literature/manuscript_audit/`; a changed source requires a new
  expansion and private packet binding.
- The completed release is pushed as `35d48055d3de427eb965854fcb267f8b0215dd13`.
  GitHub's branch head was verified before delivery. Gmail message
  `1a15f35f374c71df` sent the PDF to `adowling@nd.edu` exactly once; readback
  confirmed its filename and 906,515-byte attachment, and Sent search found
  exactly that one completion message. **Do not resend.** `delivery_receipt.json`
  records the requirement-by-requirement completion audit.


## Numerical authorities

| Phase | Completed evidence and authority |
| --- | --- |
| Migration | Thirteen notebook Python exports; scientific-cell AST equality and migrated simulation checked. `notebook_scripts/README.md`; commit `58b4fa5`. |
| Inherited baseline | Unchanged simulations run twice with byte-identical trajectories. `baseline/manifest.json`. Behavior preservation is not correctness proof. |
| Baseline fits | 99/99 corrected cases, `results/manifest.json`. Preserve the first 95/99 sweep and documented rank-deficient covariance serialization correction; `results/comparison_receipt.json` verifies prior fitted objectives. |
| Profiles | 297 first-pass jobs plus three recovered Ua profiles. `results/profile_index.json` is immutable first-pass selection. No lower baseline minimum found under the declared tolerance. |
| Selected refinement | Eighteen declared profiles from `c071c1f`, full forward/arithmetic/exact-family inclusion audit passed. `results/selected_profile_index.json` selects 18 refined and 282 original profiles; first-pass catalog retained. Regenerated profile figures/tables explicitly use this selection. |
| Bootstrap | 240 jobs and 96,000 successful draws, no failed jobs; `results/bootstrap/manifest.json` and `summary_audit.json`. Four hundred draws per method and 48 IO cases. Undefined structural delay directions remain undefined. |
| Structural | 403 positive physical equivalents per calibration, null tangents and repeated-pole checks; `results/structural`. Maximum equivalent prediction difference below 1.8e-12 C. |
| Onset limit | Eighteen exact Erlang calculations and independent quadrature/state checks, `results/onset_limit`. The n^-1/2 onset error excludes a uniform O(n^-1) bound, not uniform convergence itself. |
| Family uncertainty | 6,000 saved cold n1 draws reused without new fitting; `results/family_uncertainty`. Intervals concern identifiable family-boundary functions, not a selected original physical member. |
| Correlated likelihood | 48/48 working OU fits, independent saved-trajectory likelihood audit; `results/ou`. Remaining negative innovations challenge the error model. |
| Synthetic coverage | 3,600 datasets / 360 chunks, zero failed draws; frozen `23629a2`. `results/coverage` and `coverage_summary`; independent figure/table audits passed. Finite-grid profile classification sensitivity remains explicit. |
| Assumption sensitivity | 241/241 cases from `84cf569`, no failures/revisit flags; `results/sensitivity/audit.json` passed. Includes ambient/history, fixed perturbations and command reconstruction/discretization. |
| Prediction uncertainty | 131/131 cases from `19306cd`, independent audit of 4,220,249 saved points; `results/prediction_uncertainty/audit.json`. Mean and future-observation bands remain distinct; unresolved varying-input targets are withheld. |
| Parameter design | 112/112 corrected cases from `bb376e4`, `results/design_final/audit.json` passed. Maximum selected dense temperature 84.990587 C across assumed scenarios, not a hardware certificate. |
| Order discrimination | Twelve designs from `d1bbc74` and 72 alternative refits from `fef98bf`, both full direct audits passed in `results/discrimination_designs` and `results/discrimination_refits`. These are fixed n1-generating directional refits, not all possible truths. |

Preserve unfavorable/failed calculations. `results/followup_trials` retains the
seven first trials; the initial design trial exceeded 85 C by 0.000274 C on its
dense grid. A 0.01 C optimization guard retains the independent 85 C acceptance
limit. `results/design` preserves the initial full run (96 complete, 16
near-coincident shutdown-grid exceptions). Rounding dense audit times to the
propagator's existing one-nanosecond resolution before deduplication resolved
those exceptions; `results/design_final` is the separately measured corrected
matrix. Neither repair establishes a continuous-time bound.

The final full numerical suite passed **141/141 in 9.03 s**, saved in results/final_tests.xml. Fresh 24-start joint fits for the finite-chain n2, pure-delay and original physical families exactly reproduced saved objectives and predictions; results/final_reproduction/audit.json and its frozen source receipt retain the scope. This is not a full-matrix rerun. No additional numerical tests are needed absent changes or a new failure. The one-job cap leaves full discrimination pools at four workers.

## Report artifacts and verification

Figures and tables read saved results; do not hand-edit them. Phase-specific
source snapshots and numeric audits are authoritative in the shared artifact
directories; a generic `source_snapshot.json` can belong to an earlier producer.
Partial visual receipts identify their exact PDF hashes and scope. A successful
compiler run or a partial receipt is not final all-page QA or human review.

- Baseline and uncertainty producers have separate direct-data auditors; selected
  profile regeneration passed 150 table cells and 653 plotted numeric values.
- Coverage audits checked 5,796 figure values, 48 printed counts and three prose
  macros. Coverage results retain failure denominators, Wilson uncertainty,
  component versus envelope widths and grid-classification diagnostics.
- Follow-up reporting passed 130 source scalars and 48 table cells. History uses
  gamma consistently with the nuisance derivations; regenerated numeric values
  are unchanged.
- Design reporting separates deterministic envelope preparation
  (`summarize_design_report.py`) from plotting (`plot_design_results.py`). Its
  audit checks 280 source scalars, 56 printed cells and 64 saved model peaks.
  This does not independently verify every trajectory with another integrator.
- The working PDF is ignored `manuscript/main.pdf`. All 48 pages are rendered in
  `/private/tmp/tclab-final-draft-qa`; all pages passed inspection. The named PDF is manuscript/TC_Lab_delay_study_draft0.pdf; final_visual_review.json binds its source/build/PDF hashes.

## Literature audit and export

Exact-draft packet `6f11d88ebd8741c892fea8fffc243d7d` is saved at revision
`353ae0c83c3252afe8b0ce5db404679f66e699615a9c71f858ec96ee3130c5e8`.
All **24 citation occurrences across seven works** have exact-span claims and
agent inventory review. Seven occurrences link two inspected fine passages:
three support relations and four context relations. Seventeen occurrences retain
explicit unavailable-ingested-carrier issues for five primary works inspected
externally. These are library availability gaps, not claims that the PDFs were
unread. Agent support assessments and mechanical linkage do not establish full
claim coverage or human scientific acceptance.

Two guarded publication/readback stops were reconciled through native operations.
The persisted passages were reused and all seven current usages/attachments
completed; no library configuration, policy or runtime was changed. No literature
operation remains live. Runtime/source/destination records and primary PDF/page
identities remain in `literature/`. Four DOI identities were checked through
Crossref and three others against primary pages; do not claim seven Crossref checks.

Native Markdown, JSON and scoped BibTeX export
`c87e20495d9a844c8c2b27a3e7a4eeb2f1ced9cc4cea8fb62b620758ecde1c01`
was inspected and its hashes verified. The files are copied to ignored
`_private/final_packet/`; raw passages and source PDFs are not tracked.
`literature/manuscript_audit/final_packet_receipt.json` records the exact counts,
links, gaps, review status and export hashes. Its alias review preserves the
Garnier/Wang/Young chapter reference separately from the book carrier, and the
inspected Manchester preprint separately from the unread proceedings version.
The manuscript bibliography retains all seven checked works; scoped library
BibTeX contains only the two ingested carriers.

## Release and delivery

`build_study_manifest.py` passed, creating `results/study_manifest.json` with
234 indexed receipts and a measured elapsed window over fourteen hours. It
verifies declared phase completion, unchanged raw CSVs, notebook migration hashes,
selected audits, source/figure/table/PDF bindings, exact-draft literature counts,
141 passing tests and the three fresh-fit reproductions. Each numerical audit
retains its own scope; this index is not a new full-matrix rerun.

The complete scientific release is committed and pushed as `35d4805`.
The self-addressed Gmail completion message and its PDF attachment were verified;
`delivery_receipt.json` records that delivery and the completion audit. The receipt
and this final handoff update are committed and pushed separately. No research or
report-delivery work remains. Close the existing goal and pause its heartbeat;
human scientific review remains pending in draft 0.

## Working rules

The subsequent teaching follow-up is in `teaching_followup/README.md`. Its
separate scripts compare the original/n1/n2 saved predictions and residual-block
intervals, and run native Pyomo.DoE for n1/n2 under IID and fixed OU covariance.
All four two-start designs and sixteen information-matrix evaluations passed
an independent exact-simulator/full-covariance audit. This supplement does not
replace the frozen overnight report or its release index.

Follow the user's sibling `claude-for-researchers` practices and the no-ai-slop
skill. That sibling repository is read-only. User authorization explicitly
permits commits, the final push and self-email. Preserve raw CSVs and inherited
source. No hardware experiment is authorized or needed. Use coherent commits
with an AI attribution trailer; do not amend or force-push.

Numerical runtime is pinned in `environment.json`; verified interpreter:
`/Users/adowling/opt/anaconda3/bin/python`. Set BLAS thread counts to one. Use
`frozen_run.py` for measured phases and absolute input/output paths. Never edit
a measured snapshot. At most four long numerical worker processes total;
independent short checks may use one thread as described in REPRODUCING.md.
Source libraries and research state outside this repository are not task outputs.

The first exact-draft API attempt was rejected as remote connector disclosure. A configuration/runtime/source-flow check proved this is the configured local stdio server and private local packet storage; the unchanged native API retry then succeeded. literature/destination_review.json preserves the evidence. No transport, shared runtime or publication setting was changed, and no rejected action was routed through another execution path.
