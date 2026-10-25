# Shared-model discrepancy investigation

The scoped investigation is complete for researcher review. Read [report.md](report.md)
for findings and open questions, [methods.md](methods.md) for equations and
assumptions, and [verification.md](verification.md) for the requirement audit.
The scientific scope is the committed [plan](../review/tier0_followup_investigation.md)
at b4f7ef7. Alex set no duration constraint and prioritized understanding the
system. The completed residual_study, its report/email and raw data are unchanged.

| Phase | Result |
|---|---|
| 0. Reproduce | Independent reference means, innovations and dense likelihood agree |
| 1. Locate discrepancy | Repeatable fast cycle plus slow/offset errors; descriptive decomposition reviewed |
| 2. Frequency response | All 54 window/nuisance estimates audited; startup, repeat and HAC sensitivities retained |
| 3. Linear benchmark | Two/three/four real-pole fits audited; fast-mode/numerator-bound weaknesses established |
| 4. Stage comparison | Original and revised n=1/2/3 fits, SVD/eigenanalysis and K/theta profiles audited |
| 5. Targeted checks | Unequal positive lags, transport, affine ambient, FIR and two nonlinear hypotheses tested/audited |
| 6. Synthesis | Report table/headlines checked against source-bound artifacts; figures reviewed; requirements verified |

The revised unequal-positive model has pooled RMSE 0.37397 C versus 0.48320 C
for the original n=1 reference. Shared time constants are 142.84/2.99/13.47 s,
K 0.688645 and command curvature 0.10184. It removes much of the fast input-linked
error. It retains maximum innovation ACF 0.163, slow biases and quantized step
structure. This is a candidate reduced reference, not a universal stochastic
model or identified hardware mechanism.

Equal-stage revised n=1/2/3 retain systematic fast mismatch. Revised n=1 has low
raw RMSE 0.334 C but fast input-innovation correlations 0.458–0.482. Its nominal
K profile also shifts beyond the original formulation's support. Keep dynamic
adequacy, raw error and conditional parameter precision separate.

All sixteen main fits have independent propagation/likelihood audits. The final
consolidation verifies saved prediction arithmetic, AICc/counts, audit and
information bindings, profile support, and the report's table/headline numbers.
Current job receipts and source/data hashes are indexed in [run_manifest.json](run_manifest.json).
Terminal files and exact process commands take precedence over stale status.

Stop for Alex's review. Do not launch more calibration pairs, DoE, hardware,
a full two-temperature model or another email under this goal. A next goal can
address quantization-aware observations, ambient/thermal discrepancy and the
physical interpretation of the fast response. No new mean families or numerical
jobs remain authorized by this completed scope.
