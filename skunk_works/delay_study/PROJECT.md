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
- Next: export notebook logic into Python, verify migration, then implement and
  validate the common LTI simulation/estimation pipeline before broad sweeps.
- Literature packet: `6f11d88ebd8741c892fea8fffc243d7d`. Initial revision:
  `d88745b5615020ca2f74fa981cd87f94d867723803a0f0efb5a668e0a09f2262`.
  Inspect the current revision before further mutations. A first LTI search is
  pending in the original tool execution; do not duplicate it.
- Gmail profile verified. Finished report recipient: `adowling@nd.edu` (self).
  Send once after verification and push; save the resulting message identity.

## Layout and authority

- `../../notebooks/` from the repository root: original teaching notebooks;
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
