# TC-Lab delay study

The deliverable is a reproducible Python study and draft 0 of a LaTeX manuscript
with literature, methods, results and discussion. `research_brief.md` owns the
requested scope; `REPRODUCING.md` describes numerical dependencies and commands.
This file owns current execution state. Git preserves superseded states.

## Current state

- Branch: `reformulation-exploration`; fetched `origin/main` merged in `02ab3d9`.
- Research began **2026-10-21 01:39:03 UTC**. The requested twelve-hour minimum
  elapsed at **13:39:03 UTC**. Completion still requires every delivery gate.
- The existing thread goal is active. Recovery heartbeat:
  `tc-lab-overnight-research`. Do not create another goal or duplicate automation.
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
- **No final push or completion email has been sent.** Gmail self-recipient was
  verified as `adowling@nd.edu`; refresh the profile at delivery. Send exactly
  once after final verification and push, retaining/reconciling the sent ID.

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

## Literature state and exact resume point

Use the configured lab literature skill, without installing or reconfiguring
its runtime or ingesting/publishing new source PDFs. Packet:
`6f11d88ebd8741c892fea8fffc243d7d`. The exact draft is now saved through the native API. All 24 occurrences have exact-span agent claims and inventory coverage review; 17 explicit issues retain missing ingested carriers for five externally inspected works. Latest completed claim/issue revision: 27bbc96ec524d7f22cd6383a97ffc45b90b71c8b96ffc901d46d931e6bb70df8. Evidence planning batch 6a1c85fa62e0408b915223d72961c147 is live in functions cell 750; do not restart it. The batch contains seven claim-bound uses of the two inspected reusable passages, with support versus context relations kept distinct. Inspect every saved plan before grounding. Use the actual head after any mutation, not this earlier revision.

Runtime: configured maintenance 0.5.8; private state is
`/Users/adowling/Library/Application Support/shared-literature-rag/state-0.5.8-maintenance-2026-10-19`.
Native calls can take several minutes. Timeouts do not prove termination. Inspect
saved ownership/head/receipts before repeating an uncertain mutation; do not
start parallel writes to this packet or bypass library integrity validation.

`literature/source_review.json`, `acquired_sources.json` and the visual/evidence
receipts retain primary identities, PDF hashes, physical/printed pages, assumptions
and access gaps. Seven works are cited. Two ingested carriers have inspected fine
passages: Raue profile construction, fine shard
`2ec259a5dedd82b02a891e141629a060ac4cb8beabf6ff26a7c1077636036afe`;
Garnier/Wang/Young chapter residual decomposition, fine shard
`c6d3d2dabc24623261caeabeaa3fbed5c7151bf187cbe2d807913f9ab9006b0c`.
Five cited primary PDFs were inspected externally and have no ingested carrier;
retain explicit carrier/fine-availability issues, without calling the PDFs unread.
Chapter/book aliases and inspected preprint/journal distinctions must survive
export. Four DOI identities passed Crossref checks; three additional identities
were checked against primary pages. Do not claim seven Crossref checks.

Exact draft and claim/inventory binding has completed. Finish the live evidence-planning batch, inspect its plans, then ground the supported/context Raue/chapter clauses through normal native operations while retaining external-source availability gaps. Bounded evidence/grounding batch tools are available; inspect
plans before applying them. Export private Markdown/JSON/scoped BibTeX into
ignored `_private/`, inspect it, and save only scoped public receipts/hash metadata.
Raw source passages and PDFs do not belong in Git. Mechanical quote matching,
metadata identity, claim support and human scientific review remain separate.

## Remaining delivery gates

1. Finish all-page rendered review and any necessary fixes; preserve no-ai-slop
   and notation checks. Re-expand after any TeX change.
2. Complete the exact-draft packet workflow and scoped export with explicit gaps.
3. Commit the full audited discrimination outputs, prepared design summary,
   final figures/tables/prose, report PDF and verification receipts coherently.
4. Create `results/study_manifest.json` indexing every phase and final artifacts;
   do not replace the baseline `results/manifest.json`. Verify reproduction and
   final source/result/artifact identities against the actual committed tree.
5. Push `reformulation-exploration` normally; confirm remote HEAD. Email the
   verified finished report through Gmail exactly once and retain the sent ID.
6. Only then mark the active goal complete and pause the existing heartbeat.
   Unresolved human scientific review is disclosed in draft 0, not fabricated.

## Working rules

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
