# TC-Lab remaining-error study

Alex authorized this sustained investigation on October 25, 2026. The scientific
plan is [the shared discussion note](../review/next_investigation_discussion.md),
committed before goal launch as `346a03d`. Timing is flexible; the work spans a
ten-plus-hour research scope without an idle minimum or fixed stopping time.

## Current state

The goal is active. Read `run_manifest.json` for authoritative job identities and
terminal artifacts. Stage 0 and the first error partition passed. The constant
ambient conductance fits at ±5/±10/±20% are terminal and independently audited.
At ±10%, T1/T2 RMSE is 0.639/0.115 C; widening to ±20% gives 0.641/0.115 C.
Two ±10% deviations reach their lower bound. Protocol-aligned ratios are fitted
behavior, not measured airflow. The ±5% fit retains three failed starts.

Matched ambient/startup tests passed dimensional trajectory and dense likelihood
audits. Affine ambient gives 0.728/0.111 C. Hidden starting offsets bounded by
±2 C give 0.615/0.111 C; ±5 C gives the same result, with every selected offset
within 0.35 C. This supports cold starts as a reasonable working assumption;
absolute run timestamps and between-run recovery times remain unknown.

The radiative implementation passed independent balances, time-step, derivative
and synthetic recovery checks. Its zero-radiation limit matches all saved
baseline trajectories within 4.4e-8 C. The original four-start free-radiation
solver ended without an accepted converged result; all four attempts and scores
are preserved. A scaled-objective recovery is running in radiation_free_scaled
(controller 11817). The original linear-command/radiation fit is terminal and
independently audited. No process was killed: both original jobs ended naturally
before the approved stop retry found no such process.

Fixed-mean bin-likelihood covariance estimates passed 512-node full-record checks
and independent trivariate Gaussian rectangle integrals for every run prefix.
T1 analogue measurement SD is 0.0255 C and correlation length 725 s; T2 is
0.0325 C and 54.8 s. These are conditional model parameters, not reference-sensor
calibrations. The initial import failures and corrected v2 runs are preserved.
Medians/normality alone do not establish independence or physical adequacy.

Backward quadrature smoothing and Fisher-identity scores passed independent
posterior-moment and derivative checks. The full physical-plus-bin score check
is running in joint_score_check (controller 66603). After it passes, implement
and run joint mean/noise optimization. See quantized_methods.md for the derivation.
Earlier marginal PIT pilots reused random seeds; joint/cross-channel diagnostics
must be regenerated using independent streams by channel and run. Do not infer
joint independence from those pilot arrays.

Operating-point, heating/cooling, common/differential and Gaussian forecast
horizon tables are in operating_first; independent conditional-Gaussian and RMSE
checks passed. Noise-only conditional forecasts use earlier temperatures and are
not free simulation or independent validation. All twelve records informed
selection. Remaining work includes updated frequency diagnostics, sequential
GLS, joint quantized fits, cross-channel checks, profiles/eigendecompositions,
simulation uncertainty, retrospective transfer and the checked report.

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
