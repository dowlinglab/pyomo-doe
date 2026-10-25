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
were visually inspected. The initial conductance fits are now terminal.

In the short-lag/command-curve constant-ambient model, offsets explain 74.6%
and additional linear trends 19.3% of T1 squared error. The fundamental adds
0.64%. After 450 s, 96.1% is run offsets. These are descriptive nested
projections, not identified environmental effects. This favors investigating
slow between-run mismatch before adding another response lag. Detailed initial
findings are in `initial_findings.md` and `results/decomposition`.

The three constant-ambient conductance fits and independent dense likelihood/
dimensional trajectory audits passed. At ±10%, T1/T2 RMSE is 0.639/0.115 C;
at ±20%, 0.641/0.115 C. Two ±10% deviations hit the lower band. The ±5% job
retains three failed SLSQP starts and one feasible converged start; its RMSE is
0.646/0.147 C. Training likelihood alone does not select a band or establish
that deviations measure actual airflow. The run ratios depend on protocol.

Three four-start jobs now compare affine ambient and hidden-heater startup
bands ±2/±5 C at the 10% conductance band. Their outputs are band_10_affine,
band_10_startup2 and band_10_startup5; controller sessions 32642, 5303, 93466.
Reconcile before new launches. Audit every terminal fit with audit_bounded.py.

Quantized scalar OU filtering passed independent Gaussian rectangle integrals,
node-count convergence and a known-truth synthetic randomized-PIT check. This
supports a hardware likelihood pilot; it does not establish Gaussian hardware
noise. Operating-point/common-differential diagnostics, hardware quadrature
convergence and later covariance refitting remain unfinished. Radiation's
conditional scale and startup propagation checks passed. Read the manifest and
initial_findings.md before proceeding; existing source/data receipts bind work.

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
