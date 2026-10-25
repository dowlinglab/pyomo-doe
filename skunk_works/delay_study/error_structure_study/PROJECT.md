# TC-Lab remaining-error study

Alex authorized this sustained investigation on October 25, 2026. The scientific
plan is [the shared discussion note](../review/next_investigation_discussion.md),
committed before goal launch as `346a03d`. Timing is flexible; the work spans a
ten-plus-hour research scope without an idle minimum or fixed stopping time.

## Current state

The goal is active. Stage 0 passed: all twelve unique raw records and sixteen
prior evidence bindings match, and all 26 prior jobs have successful terminal
exit artifacts. No related process was live at launch. Stage 1 has a first
source-bound decomposition of 384 run/channel/window combinations, independently
checked by QR projections (maximum energy difference 9.1e-12 C2). Both figures
were visually inspected. New thermal fits have not started.

In the short-lag/command-curve constant-ambient model, offsets explain 74.6%
and additional linear trends 19.3% of T1 squared error. The fundamental adds
0.64%. After 450 s, 96.1% is run offsets. These are descriptive nested
projections, not identified environmental effects. This favors investigating
slow between-run mismatch before adding another response lag. Detailed initial
findings are in `initial_findings.md` and `results/decomposition`.

The current manifest owns phase status and job paths. Stage 1 still needs
operating-point, heating/cooling, common/differential and prediction diagnostics.
Next implement and audit centered conductance variation and radiation/observation
small cases before measured model fits. `config.json` fixes initial choices.

## Resume instructions

Read the plan, this file and `run_manifest.json`. Inspect process identities and
exit artifacts before starting or retrying anything. Do not duplicate a job on
an unchanged heartbeat. The existing `tc-lab-overnight-research` hourly heartbeat
is active and points at this goal. Do not change unrelated automations.

Use the pinned interpreter `/Users/adowling/opt/anaconda3/bin/python` and
`../environment.json`. Freeze committed source/data using `freeze.py`; outputs
must have unique absolute paths. Use one BLAS thread per process and at most
four long numerical processes total. Save solver starts, source hashes, failures,
process identity and exit status. Commit coherent source before measurement,
audited results afterward. Preserve all earlier studies and raw data.

Start from the saved short-lag/command-curve two-output model with constant
ambient offsets. Bounded run variation changes one physical ambient conductance,
so gain, thermal time and conductance ratios change together. Radiation needs
an order-of-magnitude range before fitting; distinguish sensor and surface
temperatures. Hidden startup temperatures and quantized observations are
explicit sensitivities. All 21,614 hardware readings match the shifted 0.3223 C
grid; emulator noise and thresholds are not the observed hardware law.

The finish gate is audited numerical findings plus concise Markdown and a
compiled, visually checked LaTeX/PDF report using the tablet annotation layout.
Commit and push the branch. Report ambiguity and failed whitening honestly.
No new email, hardware, DoE or additional calibration-pair enumeration is
part of this goal. Pause the heartbeat after goal completion for Alex's review.
