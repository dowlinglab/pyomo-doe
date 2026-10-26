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

The radiative implementation and full-record nesting passed independent audits.
The scaled free-radiation recovery has four accepted starts and is terminal;
T1/T2 RMSE is 0.633/0.115 C and normalized radiation coefficient is 0.0344.
Its Gaussian nll improves by only 1.285 over the zero-radiation 10% band. This
is conditional evidence, not a measured emissivity or a calibrated significance
test. Linear-command radiation gives 0.924/0.119 C and does not replace the
empirical command curve in this comparison. Original failed/unreliable starts
and the natural terminal exits are retained; no process was terminated.

Fixed-mean bin covariance estimates passed 512-node full-record checks and
independent trivariate rectangles. T1 analogue measurement SD is 0.0255 C and
length 725 s; T2 is 0.0325 C and 54.8 s. These are conditional model parameters,
not sensor calibration measurements. Initial import failures remain recorded.

Full physical-plus-bin scores passed independent finite-difference checks for
both conductance and radiative means, with and without run deviations.
Four three-start joint fits are running: conductance bands 0/10/20% and radiation
with a 10% convective band. The zero-band first attempt failed a numerical gate:
a predictive-moment integration range cut off rare T2 posterior mass. Corrected
component-conditional posterior ranges passed independent rare-bin rectangles,
12/16-SD support checks, and the full quantizer/smoother/score/hardware regression
suite. The replacement joint_conductance_b00_v2 is progressing with finite
likelihoods and no initial failed points. Other live jobs are joint_conductance_b10,
joint_conductance_b20 and joint_radiation_b10. The 5% case is queued.

Use run_manifest.json and process_check.json for current identities. Controller
sessions are 28529/15102/61547/36440 in that order. The original zero-band session
55866 is terminal and must not be resumed/relaunched. Obtain process lists through
the approved direct pgrep tool: pgrep run inside a Python subprocess is blocked
by sysmon and returns no list, which is not evidence of no live jobs. Do not
exceed four long numerical processes.

The three original nonzero-band jobs use earlier frozen posterior-domain code.
Audit their accepted points against corrected tail-12 and wider tail-16 support
and finer grids; refine under the current score if needed before interpretation.
All starts and failed points remain evidence. Source matrix joint_fit_matrix.json
records the five cases; quantized_methods.md owns the score derivation and
initial-condition caveats. Warm initial covariance inflation is a heuristic,
not an independent-uniform quantization assumption in the fitted likelihood.

Earlier marginal PIT pilots reused random seeds. Their joint randomization is
unsuitable for cross-channel/run tests. The joint fitter uses independent streams
by channel and run; repeated randomizations and model-based reference checks are
still required. Gaussian marginal appearance alone is not adequate whitening.

Operating-point, heating/cooling, common/differential and Gaussian forecast
horizon tables are in operating_first; independent conditional-Gaussian and RMSE
checks passed. Noise-only conditional forecasts use earlier temperatures and are
not free simulation or independent validation. All twelve records informed
selection. Remaining work includes updated frequency diagnostics, sequential GLS, terminal
joint-fit audits, cross-channel checks, profiles/eigendecompositions, simulation
uncertainty, retrospective transfer and the checked report. Include sensitivity
to initial sensor values within their quantization bins, consistent with each
bin phase; varying only hidden heater initial states does not cover that issue.

Sequential GLS implementation and its independent short-design check are now
committed. Dense covariance nll agrees within 8.6e-14, noise scores within
2.6e-10 scaled error, and the constrained mean block decreases its objective.
The full IID-start and saved-joint-start comparisons wait for numerical capacity;
see sequential_methods.md for stopping rules and retained iteration limits.

Independent conditional observation integrals verify down/same/up code forecasts
for nearest and floor bin phases to 1.2e-16. Predictive diagnostics are implemented
but not yet measured on selected joint fits: 20 independent PIT randomizations,
50 fixed-model simulation references, ACF/PACF, prespecified Ljung–Box statistics,
input/phase/operating-bin dependence, cross-channel dependence and nonrandomized
transition counts. These conditional simulations do not refit parameters and
must not replace the planned 50 refitted uncertainty/recovery draws.

Use reconcile.py with a fresh direct-tool process_check.json to update job states.
It never launches jobs or interprets blocked Python process enumeration as an
empty machine. The selected joint-bin audit checks current corrected score
stationarity, independent physical trajectories and Gaussian bin rectangles,
128/256/512-node convergence and 12/16-tail support. Neither the predictive checks
nor these numerical audits establish a global optimum or physical adequacy.

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
