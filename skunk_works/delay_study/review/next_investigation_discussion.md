# TC-Lab: remaining-error investigation and execution plan

Alex authorized this goal on October 25, 2026 after reviewing the coupled-model
results and three source/data reviews. This file owns the scientific plan and
the evidence behind it. Commit this plan before starting the goal or numerical
work. The execution state, job identities and terminal artifacts belong in
`../error_structure_study/PROJECT.md` and its run manifest.

The requested scope is a sustained investigation suitable for ten or more hours
of research. Alex explicitly made timing secondary: advance the analyses and
decision gates below, without waiting to satisfy a clock or stopping a useful
analysis at an arbitrary deadline. Commit coherent progress throughout. Preserve
the earlier completed studies and raw data. The question is why the joint
physical model leaves structured errors, and how much credible uncertainty
remains after modeling those errors.

The three first-pass reviews are now complete. They establish a hardware-style
temperature grid in the recorded data, document the distinction between common
and channel-specific calibration, and identify prior physical and stochastic
models worth comparing. They do not establish the cause of the remaining T1
error. Source access and historical-version gaps are recorded below.

## Decisions and scope

- Treat Dan's records as physical TCLab experiments. Alex trusts that they were
  collected carefully using the same device. Simulator behavior is a source of
  hypotheses, not evidence that the simulator generated these data.
- A shared constant heater conversion factor alpha remains absorbed into the
  fitted gain. Temperature-dependent delivered power is a different question.
- A simple linear sensor scaling may be absorbed into a freely fitted SISO
  gain. Check the limits of that statement for two channels with a shared
  physical model, different calibration slopes, and ambient offsets.
- Ambient temperature and airflow can vary. Changing room temperature affects
  the ambient forcing; changing airflow can affect heat-transfer coefficients.
  Do not assume one nuisance ambient offset accounts for both.
- Consider radiation explicitly, using Kelvin and physical parameter ranges.
  Quantify its plausible size before claiming it explains the discrepancy.
- The completed worker assignments below covered review only. The newly
  authorized numerical goal follows the execution plan at the end of this file.
  It includes model fitting and a continuation heartbeat. New hardware, DoE,
  additional calibration-pair enumeration and a new email remain out of scope.

## Established results to carry forward

The [coupled-model report](../coupled_model_study/report.md),
[run manifest](../coupled_model_study/run_manifest.json), and
[independent checks](../coupled_model_study/results/final_check/audit.json)
own the numerical evidence. All twelve distinct ENV2 records informed these
fits and model selection. These results are retrospective.

| Result | Evidence and limit |
| --- | --- |
| Remaining T1 error | About 1 C pooled **RMSE**, not a measured 1 C sine-amplitude error. Separate mean offset, slow trend, amplitude, phase and remaining residual structure. |
| Base symmetric coupled model | T1/T2 RMSE 1.107/0.234 C. A positive symmetric four-state model cannot produce T1 steady phase outside (-180, 0) degrees under its stated assumptions. This does not rule out every thermal network. |
| Additional short lag | Around 2.7 s under the working colored-error likelihood. Fast T1 matched phase error drops from roughly 28–32 degrees to 3–6 degrees. The mechanism is not identified as a hardware input delay. |
| Command curve plus short lag | T1/T2 RMSE 1.022/0.231 C with constant per-run ambient offsets. The command curve is empirical, not a measured electrical power law. |
| Affine ambient sensitivity | T1/T2 RMSE 1.032/0.124 C. Both ambient endpoints remain bounded by ±2 C. Better T2 fit does not resolve the remaining T1 discrepancy. |
| Fast T2 information | Working joint complex-gain ellipses include zero in all three fast-sine repeats after 450 s. Fast T2 phase cannot reliably distinguish sensor lag from input lag. |
| Physical interpretation | Sensor heat loading is weakly estimated and often reaches its lower bound. In the negligible-loading limit, common sensor and input lag factors commute. Unequal fitted time scales do not uniquely label physical components. |
| Simulator source | The inspected Python emulator uses coupled heater states, nonloading sensor lags, analogue noise and a floor quantizer. Its defaults and noise law are not calibration results for Dan's unit. |

The earlier [shared-model investigation](../shared_model_investigation/report.md)
already tested several empirical dynamics, command transformations, startup and
timing assumptions. Its quadratic heat-loss sensitivity was not a fitted
physical radiation model for both temperature channels. Check those results
before repeating an old test under a new name.

## Worker assignments and integration rules

| Worker | Assignment | Current status | Detailed evidence |
| --- | --- | --- | --- |
| Hardware quantization | Inspect Dan's data for temperature grids; separate ADC quantization from CSV rounding; trace hardware conversion and reporting. | Complete; all ENV2 values match the shifted 0.3223 C grid. Historical firmware identity remains unknown. | [Findings and numerical audit](../hardware_quantization_review/findings.md) |
| Sensor calibration | Official documentation, installed pip package, firmware and documented calibration studies; distinguish offset, slope, reference-voltage and nonlinear effects. | Complete first pass; installed 1.0.0 inspected. No calibration of Dan's unit established. | [Calibration and source review](../sensor_calibration_review/README.md) |
| Prior modeling literature | Physical and empirical TC-Lab models, radiation, convection, disturbances, sensor dynamics, identification and validation. | Complete first pass; ten reviewed source families and four explicitly incomplete leads. | [Review and evidence matrix](../prior_modeling_review/review.md) |

Each worker owns only its named section below. Keep detailed source receipts,
code and numerical output in its separate review folder and link them here.
Populate this file with the findings needed for our discussion, not just links.
For each conclusion, state what was inspected, what it supports, what remains
unknown, and a source/evidence link. Mark preliminary or unverified leads.
Do not describe nominal tolerances or simulator defaults as observed errors in
this device. The parent owns the shared synthesis, decisions and proposed work.

## Hardware quantization findings — worker section

All **21,614 T1/T2 observations** in the twelve distinct ENV2 records exactly
match `round(0.3223*n - 50, 2)` for inferred integer codes n. The same law fits
both channels in every run. Across both channels there are 158 levels,
19.94–70.54 C, with adjacent gaps of 0.32 or 0.33 C. The 0.01 C text precision
is finer than the underlying approximately 0.3223 C grid. All 27,198 values
across the fifteen current CSV paths also match; two paths are duplicate
aliases and do not add experiments.

The zero-offset floor lattice in the inspected Python emulator has the same
spacing but a different offset: after two-decimal reporting it matches **zero**
ENV2 values. It must not be copied as the hardware measurement law. The exact
3.3 V/1024 coefficient, 0.322265625, matches only 2,331 ENV2 values after
reporting and differs by up to 0.02 C. The data therefore support the rounded
0.3223 coefficient specifically, conditional on the inferred code labels.

Repeated adjacent readings are common: T1 51.6–81.3 percent, T2 85.4–89.7
percent across runs. The longest fast-sine T2 plateau is 137 samples. These
facts support a quantization-aware measurement analysis; they do not identify
analogue-noise variance or establish that quantization explains all residual
correlation. Estimated sinusoidal amplitudes can be smaller than one grid step
if the record contains informative threshold crossings or noise.

Hardware provenance is supported by the historical acquisition notebook using
physical `TCLab()` and `Historian.to_csv`, alongside Alex's knowledge of the
experiments. The notebook has no retained text giving package/firmware version
and is not a complete per-file execution log. Official Python hardware getters
parse serial float responses; they do not apply the emulator quantizer. The
old official WebUSB firmware lists `analogRead(pin)*0.3223 - 50.0` and an
external analogue reference. The exact firmware used for Dan's device is still
unknown. The calibration worker found another official firmware variant that
uses different sampling/conversion/reporting, so version differences must be
preserved rather than inferred from a current upstream default.

The [detailed findings](../hardware_quantization_review/findings.md),
[reproducible audit source](../hardware_quantization_review/audit.py),
[per-file/channel results and raw hashes](../hardware_quantization_review/results/summary.json)
and [level table](../hardware_quantization_review/results/ENV2_levels.csv)
contain the evidence. Source was committed as `016101b` before the successful
single-threaded pinned-runtime audit. Raw CSVs are unchanged. The fixed-offset
rounding-bin step bounds [0.3222997033, 0.3223000000] C are arithmetic feasibility
bounds conditional on candidate labels, not confidence intervals, calibration
accuracy or proof against finer submultiple grids.

Source links: [official Python](https://github.com/jckantor/TCLab/blob/master/tclab/tclab.py),
[old official WebUSB firmware](https://github.com/jckantor/TCLab-sketch/blob/master/TCLab-sketch-webusb-old/TCLab-sketch-webusb-old.ino).
Remaining questions are exact firmware/board identity, reference-voltage and
sensor accuracy, analogue-noise statistics, and how an appropriate hardware
observation likelihood changes the existing uncertainty conclusions. No
thermal refitting was performed.

## Sensor calibration findings — worker section

The [calibration review](../sensor_calibration_review/README.md) documents the
inspected sources, algebra, search limits and remaining evidence requirements.
The [source inspection receipt](../sensor_calibration_review/source_inspection.json)
binds the cached upstream Python file and hardware getter methods by SHA-256.

- Official documentation identifies **TMP36GZ analog semiconductor sensors**
  with nominal T(C)=0.1 Vout(mV)-50. The manufacturer specifies typical ±1 C
  accuracy at25 C and ±2 C across -40 to125 C; G-grade maximum limits are
  ±3 C and ±4 C respectively. These specifications are not observed errors
  in Dan's device. The common ±1/±2 C shorthand is not a hard accuracy bound.
  See [Analog Devices Rev H, pp.1,3](https://www.analog.com/media/en/technical-documentation/data-sheets/tmp35_36_37.pdf).
- The inspected upstream hardware `T1`/`T2` getters request a temperature
  string from firmware and parse it as float. They apply no sensor slope or
  offset calibration. The Python constructor exposes no channel calibration
  argument. That does not establish every historical interface's behavior.
  [Upstream hardware path](https://github.com/jckantor/TCLab/blob/master/tclab/tclab.py).
- A common static temperature offset trades with ambient and initial
  temperatures in a linear model. A common positive slope trades with freely
  fitted gain and does not add phase. **Different channel slopes or offsets
  can affect the symmetric two-output fit**, because it does not contain
  unrestricted channel calibration. Sensor calibration, physical asymmetry
  and coupling can then trade off. Static calibration cannot by itself
  explain a frequency-dependent missing lag.
- ADC-reference error is a distinct mechanism: under an ideal nominal sensor,
  y=(Vnom/Vref)*(T+50)-50, so even a constant reference error creates both a
  slope and an offset. Time-varying reference or nonlinear conversion errors
  would have different consequences. No reference drift has been measured.
- The official [sensor worksheet, p.3](https://www.apmonitor.com/pdc/uploads/Main/TCLab1.pdf)
  explicitly discusses manufacturing differences, reference comparisons and
  calibration offsets. The bounded online search has not verified a
  reference-thermometer calibration study that measures errors relevant to
  ENV2. Thermal-model parameter calibration papers do not establish sensor
  calibration. Generic TMP36 forum reports were not used as device evidence.
- Firmware differences are concrete: the official historical WebUSB-old
  sketch (internal version1.3.0) uses a single ADC read,0.3223 C per count
  and a -50 C offset. The APMonitor version2.0.1 listing averages ten reads
  and actively uses0.322265625 per count. Both select an external reference;
  neither listing proves Dan's actual firmware. The current upstream
  Historian CSV export adds no explicit decimal rounding. The detailed
  review links both code listings and separates inspected web code from a
  verified pip artifact.

Source/version gaps remain explicit. PyPI lists1.0.0 (September2023), but its
wheel archive has not been downloaded and independently byte-verified. An
existing **pip-installed1.0.0** was then located in the local `controls3`
environment and inspected without imports or changes. Its hardware source is
byte-identical to the cached upstream file; inspected source/metadata hashes
match its RECORD. Its Historian confirms no explicit decimal rounding.
The [installed-package receipt](../sensor_calibration_review/installed_package_inspection.json)
records that evidence. This local installation does not establish Dan's
acquisition version. The exact firmware/package used for ENV2 is not recorded
in the CSVs; firmware versions differ in reference calculation and averaging.
We need those versions, or independent reference measurements, before selecting
a device-specific calibration law. No calibration error has been established
as the cause of the approximately1 C pooled T1 RMSE.

## Prior modeling literature — worker section

The [first source review](../prior_modeling_review/review.md) and
[search/verification record](../prior_modeling_review/search_log.md) are saved.
This is a focused comparison of ten source families, with four partial leads;
it does not claim to find every TC-Lab study. Agent-authored and awaiting Alex's
scientific review. Three findings directly affect our next choices:

- **Nonlinear heat loss has prior TC-Lab precedent.** Park et al.'s
  [2020 benchmark manuscript](https://apm.byu.edu/prism/uploads/Members/2020_park_tclab.pdf)
  uses ambient and interheater convection/radiation with separate sensor lags.
  It compares physical, FOPDT, ARX and Hammerstein models on separate step-test
  validation. Strong training fits do not always transfer. Nominal constants and
  validation scores are specific to that experiment; they are not our RMSE.
- **Temperature-dependent power is distinct from constant alpha.**
  [Kantor's documentation, §1.3.5.1](https://jckantor.github.io/cbe30338-book/tclab/00.03-using-tclab.html)
  explicitly says delivered power depends on temperature at constant P settings.
  This supports a constrained power(T,Q) sensitivity, but supplies no calibrated
  law for ENV2. Temperature-dependent loss and power may be hard to separate
  without electrical measurements.
- **Disturbance modeling needs its own validation.**
  [Kuntz & Rawlings, extended 2025 report §6.1](https://arxiv.org/html/2406.03760v3)
  fits stochastic state-space models with integrating output disturbances on
  physical TCLab. It separates noise-free response, feedback estimation/control
  and covariance-normalized innovation checks. This is a grounded alternative
  to independent-channel OU errors. A better filter or likelihood does not
  establish the deterministic model's accuracy or physical parameter meaning.

Two source qualifications guard against misleading interpretations. The
[Di Teodoro et al. 2024 experiments, §5.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11292832/)
use a fan but add their 200 s delay in software. That delay is not evidence about
native TCLab hardware. Some
[APMonitor higher-order examples](https://apmonitor.com/pdc/index.php/Main/TCLabHigherOrder)
concern Q1 to T2; their order should not be assigned to the directly heated T1
transfer.

[APMonitor nonlinear model code](https://apmonitor.com/do/index.php/Main/TCLabG)
uses area As for interheater radiation while Park's displayed balance uses A.
This may affect numerical coupling; choose and document a geometry assumption
rather than combining parameter sets. Radiation's linear contribution can be
absorbed into existing effective conductance, as the parent's calculation below
explains. An explicit radiation extension must refit the loss terms.

The review also found a monotonic Taylor-neural TCLab demonstration, but its
15 s sampling and closed-loop test do not establish the missing few-second
mechanism. A new CBA SINDy/WSINDy preprint claims weak radiation identification;
only its abstract was accessible, so its methods/results remain unverified.
These are secondary leads, not reasons to replace the current model immediately.

Proposed next comparison: retain matched free simulation and frequency response,
test constrained nonlinear loss and temperature-dependent power separately, then
assess shared environmental/noise structure. Compare errors at similar
absolute temperatures but different commands and during heating/cooling. None
of these causal explanations has been established in Dan's records.

The Lab Literature metadata lookup failed because metadata-generation retention
capacity was exhausted. No corpus completeness or source-grounding claim follows.
Official web-source inspection supplied this first review; source bytes were not
downloaded or hashed, and no new evidence packet was completed. Access failures
and exact inspected versions are retained in the detailed record.

## Radiation scale — parent calculation for discussion

Radiation is worth checking quantitatively. For a surface at absolute
temperature T exchanging with surroundings at Tr,

\[
q_{rad}=\epsilon\sigma A(T^4-T_r^4)
=A\,h_{rad}(T,T_r)(T-T_r),\qquad
h_{rad}=\epsilon\sigma(T+T_r)(T^2+T_r^2).
\]

Using the **illustrative** APMonitor radiative-example assumptions
epsilon=0.9, A=0.0012 m2, sigma=5.67e-8 W/(m2 K4), and surroundings at
23 C gives:

| Assumed surface temperature, C | Equivalent h_rad, W/(m2 K) | Radiative loss, W |
| ---: | ---: | ---: |
| 23 | 5.30 | 0 |
| 50 | 6.07 | 0.197 |
| 70.54 | 6.72 | 0.383 |

The example's convective coefficient is 5 W/(m2 K), so radiation need not be
small under those assumptions. This is an order-of-magnitude calculation,
not an estimated heat-loss split for our device. The largest measured T1 is
70.54 C, but T1 is a sensor reading, not a direct heater-surface measurement.
Actual emissivity, effective area, view factors and surrounding surface
temperatures remain unmeasured here. Radiative surroundings and room air need
not have identical temperatures.

The linear part of radiative heat loss can already be absorbed into a fitted
lumped conductance. The new information would come from its temperature
dependence and any independently known physical quantities. An explicit
convection-plus-radiation model must refit the loss parameters; adding radiation
on top of the old fitted effective Ua would risk counting the same loss twice.
This also means a large radiation fraction does not, by itself, establish that
neglecting its nonlinearity explains the remaining error.

## Questions not yet assigned to numerical investigation

| Question | Why it could matter | Candidate diagnostic; not yet launched |
| --- | --- | --- |
| Delivered electrical power | A constant alpha is absorbed by gain; power varying with heater temperature or supply conditions can distort the mean response. | Inspect logged P1/P2 and acquisition configuration; compare errors with command, temperature and heating/cooling. Identify what cannot be checked without voltage/current measurements. |
| Airflow and heat-transfer variation | A fan or changing room airflow can change Ua without a matching change in ambient temperature. | Compare shared versus run-specific loss behavior and common environmental signatures; distinguish this from simply making ambient more flexible. |
| Hidden starting temperatures | Initial heater temperatures were set equal to first sensor readings. An unequilibrated heater can bias startup and inferred time scales. | Check early-time sensitivity to independently initialized hidden heater states and prehistory, while distinguishing this from prior SISO startup tests. |
| Shared versus differential residuals | Errors common to both outputs may help diagnose environmental forcing; differential errors may help diagnose coupling or channel-specific effects. | Analyze residual sums/differences, cross-channel spectra and time structure with appropriate scaling and uncertainty. This is diagnostic evidence, not a unique causal separation. |
| Radiation magnitude and nonlinearity | A fitted linear Ua can absorb the local linear part of radiative loss. Temperature dependence may remain across operating points. | Estimate effective radiative conductance over plausible heater temperatures; compare its variation with convection and inspect temperature-dependent residuals before fitting a full extension. |
| Quantization-aware uncertainty | A smooth mean plus additive Gaussian errors may misrepresent a discrete measurement process. | Use worker evidence to decide whether a documented hardware measurement likelihood is supportable; do not import the emulator noise law without evidence. |

## Synthesis and decisions still open

The observation model now has a direct empirical constraint: the shifted
hardware-style grid fits every recorded temperature. Future likelihood work
should account for that grid without importing the emulator's unmeasured noise
variance or floor thresholds. Resolution, calibration accuracy and temporal
noise dependence remain different quantities. The grid explains the available
levels and is consistent with plateaus; it does not measure analogue noise or
attribute the full T1 RMSE to quantization.

Alex's point about linear calibration remains valid for a common static scaling
with free gain. The two-output model adds a distinction: relative channel
calibration can trade with coupling and physical asymmetry. A free calibration
parameter that lowers the residual is therefore not, by itself, a measured
calibration error. Static scaling also does not supply the missing phase.

The literature gives specific alternatives to investigate: nonlinear radiative
loss, temperature-dependent delivered power, and shared environmental/stochastic
disturbances. These are supported model structures, not diagnoses of this
device. Preserve the differences between free simulation and filtered prediction,
and between native hardware dynamics and deliberately injected delays.

No cause of the remaining T1 error has been established. Avoid adding many free
effects simultaneously: an improved objective alone will not tell us whether
the explanation is power delivery, heat loss, calibration, startup, timing or
measurement statistics. The existing empirical command curve and quadratic-loss
sensitivities should inform the next comparison, while retaining their limited
physical interpretation.

The following plan incorporates Alex's decisions. It tests these explanations
in sequence and records when the existing temperatures cannot distinguish them.

## Authorized goal and questions

Explain the remaining errors in the twelve distinct ENV2 experiments using
matched two-output fits, physically constrained variation between runs and a
measurement model consistent with the observed temperature grid. Determine
whether the resulting innovations are sufficiently independent and Gaussian
for the intended uncertainty calculations. Report unresolved discrepancies and
physical ambiguities rather than making a white-noise claim by construction.

The investigation must answer:

1. How much error is run offset, slow trend, startup, periodic amplitude/phase
   mismatch, cross-channel dependence or measurement discreteness?
2. Can modest changes in one physical heat-loss parameter explain differences
   between runs while the experiments still estimate a shared nominal model?
3. How large could radiation be for this metal heater/fin, and can its nonlinear
   contribution be distinguished from convection and delivered power?
4. Are the cold-start assumptions adequate? What changes if hidden heater
   states differ from the first sensor readings?
5. Which covariance and observation assumptions yield calibrated innovations
   and prediction intervals? Which conclusions survive retrospective transfer
   to different runs and protocols?

## Model definitions and constraints

### Common baseline and bounded run variation

Retain the positive coupled heater/sensor energy balances from the completed
coupled study. Fit both temperatures and use actual within-run sampling times
with left-held heater commands. Start from the saved short-lag plus command-curve
model with constant ambient offsets. Carry the affine-ambient fit as a matched
sensitivity. Reuse saved baselines before deciding whether a new baseline fit is
needed. Keep the earlier empirical three-pole fit as a prediction benchmark;
its better T1 fit does not supply a physical interpretation automatically.

The primary varying parameter is ambient heat conductance Ua. Write

\[
 U_{a,j}=U_{a,0}\exp(\delta_j),\qquad
 \sum_{j=1}^{12}\delta_j=0,\qquad
 \log(1-b)\leq\delta_j\leq\log(1+b).
\]

Ua0 is the regressed geometric nominal conductance; the centering constraint
prevents a free shift between nominal and deviations. Ua0 and the remaining
coefficients are represented through identifiable ratios because absolute power
calibration is unavailable. The hard band couples runs; it is not a Gaussian
random-effects model and does not estimate a population variance. Evaluate
prespecified b=0, 0.05, 0.10 and 0.20. A wider band always gives at least as much
fitting freedom, so training likelihood alone cannot select its width. Report
active bounds, deviation profiles, stability and retrospective prediction.
Use sensitivity beyond 20 percent only if an identified failure or profile
requires it, with a recorded reason before the additional run.

Hold beta, CH, CS, Us and Ux shared. For g_j=exp(delta_j), the existing
scale-free coordinates change together:

\[
 K_j=K_0/g_j,\quad t_{A,j}=t_{A,0}/g_j,\quad
 r_j=r_0/g_j,\quad q_j=q_0/g_j,\quad r_{S,j}=r_{S,0}.
\]

This is one varying physical coefficient, not independent gains and time
constants for each experiment. With radiation, Ua denotes the convective
coefficient in that model; refit it rather than reusing an effective total loss.
Do not claim an absolute Ua in W/K without an independent power/capacity scale.
Use a constrained optimizer or an exact feasible parameterization for all
twelve centered deviations; eliminating the last deviation still requires its
band constraint. Verify centering and every run's bound independently.

Give each run one common ambient offset, bounded by plus/minus 2 C around its
documented reference. Keep channel measurement offsets separate from ambient
physics. Test linear ambient trajectories only as a declared comparison with
both endpoints in the same band. Shared constant alpha stays absorbed into gain.
Retain the empirical command curve initially to match the completed baseline;
compare a fixed linear command law when assessing whether loss effects replace
the curve. Do not label the curve a measured electrical power law.

### Radiation calculation before a radiation fit

Compute q_rad=epsilon*sigma*A*(H^4-Tr^4) and its equivalent and differential
conductance using Kelvin. Show surface temperatures spanning the observed sensor
range and conditional hotter hidden-heater temperatures. Report the calculation
per unit area and emissivity, as well as using the published illustrative area.
Metal alone does not establish emissivity: preserve an explicit range across
surface finishes, with primary source values or clearly labeled scenario
assumptions. Do not transfer a painted/anodized epsilon=0.9 to bare metal.

Separate the total radiative contribution from its variation with temperature.
A locally linearized radiation term can be absorbed into a fitted linear Ua.
Estimate how much nonlinear variation remains across our operating range;
that is the contribution relevant to the current model discrepancy. Surface
temperature, effective area, view factors and radiative surroundings are not
directly measured. Calculate the sensitivity to each. The measured T1 is not a
known heater-surface temperature.

If the scale and residual dependence justify fitting radiation, use positive
convection plus normalized radiative loss in the two-output balances. Keep
unknown absolute beta absorbed in the scale-free chart. Begin with ambient
radiation; add interheater radiation only as a specified separate comparison,
resolving the published A versus As area convention. Test fixed physically
plausible coefficients and a profiled nonnegative radiation coefficient.
Refit all loss/gain terms to avoid double counting. Check correlation with
command curvature and bounded conductance deviations. An uninformative profile
is a result, not a reason to assign a precise emissivity.

### Startup and possible future measurements

Use cold starts as the baseline hypothesis, not a verified equilibrium condition.
CSV Time gives within-run timing; it does not reveal dates or recovery times
between runs. Never reconstruct missing inter-run gaps from file order or Git
commit times. Audit early transients and initial T1-T2 differences. Compare
H_i(0)=S_i(0) with bounded hidden-heater offsets, first plus/minus 2 C then a
declared plus/minus 5 C sensitivity, reporting bound hits and physical ordering.
Conditioning on the first quantized reading also carries interval uncertainty.
Use common prespecified startup windows of 0, 60, 180 and 450 s for diagnostics;
do not choose a favorable window separately for each model. Compare full-record
fitting with startup-state fitting before discarding startup observations.

Prepare recommendations for a possible later fume-hood experiment: log ambient
temperature, absolute time, previous run/end time, warm-up/cool-down, hood sash
and airflow setting, firmware, board and sensor identity, P settings and power
measurements if available. A hood can introduce forced convection or fluctuating
flow; it is not automatically a constant environment. No hardware work is part
of this goal.

## Staged investigation and decision gates

| Stage | Work and required evidence | Decision before proceeding |
| --- | --- | --- |
| 0. Recover and bind evidence | Reconcile process IDs, exit artifacts and manifests; hash the twelve unique CSVs; bind selected prior fits and trajectories; freeze committed source and configuration. | No duplicate live job, no changed raw file, no unexplained selection. |
| 1. Decompose the error | Compute per-run/channel offset, trend, early/late error, harmonic amplitude/phase, common/differential residuals, input dependence and innovation diagnostics. Quantify uncertainty with dependence-aware methods and enough settled cycles. | Rank discrepancies by size and repeatability; record which mechanisms they can discriminate. |
| 2. Radiation scale and synthetic checks | Produce the metal-surface radiation range and nonlinear remainder; verify conductance reparameterization, startup-state propagation and observation-bin arithmetic. Recover known parameters on synthetic records with the actual commands and times. | Candidate effects have a plausible scale and numerically verified implementation. Identify nonrecoverable combinations. |
| 3. Bounded common model | Fit the four conductance bands, constant ambient first. Use matched noise/command/startup assumptions, multistarts, frozen source and independent likelihood/trajectory checks. | Do deviations remove reproducible mean structure? Are bounds or one experiment driving the result? |
| 4. Targeted physical extensions | Test radiation, hidden starts and affine ambient one change at a time; assess the smallest supported combination. Compare fixed-linear and curved command laws when warranted. | Improvement survives numerical audits and per-run diagnostics; physical parameters remain interpretable or their ambiguity is explicit. |
| 5. Residual distribution and refitting | Diagnose and refit covariance, including cross-channel disturbances and quantization-aware observation likelihoods. Compare sequential feasible GLS with joint likelihood where feasible. | Innovations pass declared predictive checks or remaining failures are quantified. Do not use covariance flexibility to conceal input-linked mean errors. |
| 6. Uncertainty and transfer | Profile supported parameters/nuisances, inspect information eigendirections and simulate coverage; run retrospective leave-run/protocol-out checks for a small declared set of candidates. | Intervals include noise, nuisance and bounded-parameter uncertainty; no untouched-holdout claim. |
| 7. Report and review checkpoint | Produce concise findings, audited figures/tables, open questions and recommendations; compile and inspect a tablet-readable PDF. Commit and push all coherent results. | Scientific claims trace to terminal, audited evidence. Stop for Alex's review before a new hardware study or scope expansion. |

Stages are scientific dependencies rather than fixed hour allocations. This is
enough work for a sustained ten-plus-hour investigation; adjust the amount of
computation to evidence and convergence. Record deferred or unsuccessful branches
and why they did not resolve the question. A completed goal may conclude that
the available inputs cannot separate competing mechanisms.

### Stage 1 details: compare like with like

For all four saved physical model families, report pooled and per-run T1/T2
RMSE, signed mean error, centered RMSE, linear-trend projection and periodic
projection. Keep the decomposition orthogonal where possible and report overlap
where it is not; do not sum overlapping error categories as independent terms.
Fit sine/cosine to input and output at the known drive frequency, with offsets
and startup/trend treatment declared. Compare complex response, amplitude and
phase with uncertainty; with weak fast T2 response, retain the complex-plane
region containing zero and withhold an apparently precise phase interval.

Show the same errors versus time, command, temperature and heating/cooling.
Compare repeat phase-aligned residuals, fundamental/higher harmonics already in
the input, and residual sums/differences. Report both free simulation from the
chosen initial state and one-step/short-horizon conditional prediction. Model
feedback can improve prediction without fixing simulated dynamics. These plots
can suggest mechanisms but do not uniquely distinguish airflow, calibration,
radiation and power.

### Stage 5 details: whitening and discrete measurements

The hardware data are quantized: every ENV2 value fits the shifted 0.3223 C
grid. Do not use the emulator's zero-offset floor or its analogue-noise variance.
The grid alone does not identify thresholds or pre-quantization noise. Compare
round-to-nearest and ADC-style bins with a documented phase/offset sensitivity;
separate the conversion from final two-decimal CSV reporting.

Start with the existing IID and independent OU-plus-nugget approximations.
Estimate covariance on the current residuals, refit the mean by generalized
least squares, reassess residuals and repeat to a declared objective/parameter
convergence tolerance. Retain every iterate and compare with joint ML; changing
weights without updating the covariance normalization is not likelihood fitting.
Next test a parsimonious two-channel disturbance state with full observation
covariance or a shared environmental component, guided by residual cross-channel
structure and the reviewed Kuntz-Rawlings model. Preserve physical and stochastic
states as distinct objects.

For a quantization-aware likelihood, integrate the conditional observation
density over each candidate hardware bin. Correlated latent errors require
conditional filtering/integration; multiplying marginal bin probabilities is
not the correlated joint likelihood. Verify small cases by independent numerical
integration and compare any approximation with a converged reference. Check
small-variance, tail, threshold and vanishing-bin-width limits. Keep failures
and approximation error visible. If a faithful likelihood is computationally
prohibitive, report a validated approximation or a bounded sensitivity rather
than calling an ad hoc rounding correction exact.

Inspect per-run ACF/PACF, several prespecified Ljung-Box lags (1, 5, 10, 20, 60),
input/innovation and cross-channel correlations, phase dependence, variance
versus temperature/command, QQ behavior and tail exceedances. Correct for fitted
parameters and multiple diagnostics with simulated reference envelopes where
possible. Report effect sizes, not just a large-sample p-value. Uncorrelated
errors alone do not establish independence; a straight QQ plot alone does not
establish model adequacy.

Deterministic whitening of discrete observations cannot yield exactly continuous
IID Gaussian values. Assess latent Gaussian innovations conditionally, and use
randomized probability-integral-transform residuals from the discrete predictive
distribution with fixed seeds and repeated randomizations. State the
randomization contribution; it must not hide systematic input dependence.
Generate reference diagnostics from the whole fitted model and quantizer.
The aim is approximately calibrated independent innovations for inference;
report where Gaussian assumptions fail. Never declare success merely because
a more flexible noise model produces a higher likelihood.

### Stage 6 details: uncertainty and transfer

Use profile likelihood for gain/lags, representative conductance deviations,
radiation and startup terms that affect conclusions. Include all re-estimated
ambient and covariance nuisances. Show active bounds and disconnected or flat
regions. Use Jacobian/information singular values and eigendirections in stated
coordinates; separate structural symmetries from finite-data weakness. Wald
intervals at active bounds are not the primary uncertainty result.

Use parametric simulation through the fitted correlated latent and quantized
observation model, with refitting, to check interval behavior. A residual block
bootstrap is a sensitivity only where its assumptions are appropriate. Declare
draw counts, seeds, solver tolerances, failure denominators and Monte Carlo
uncertainty before a measured sweep. Begin with 50 recovery/coverage draws;
expand to 200 only after numerical/coverage diagnostics justify the cost.
Do not count readings as independent replicates.

For transfer, separate prediction using a known run's fitted deviation from
prediction for a run whose deviation/ambient are unknown. For the latter,
propagate the permitted band and ambient uncertainty; do not estimate nuisance
values from the test output and call that blind prediction. Compare a shared
nominal prediction with an explicit small-prefix adaptation procedure if useful.
Use retrospective leave-one-run-out and leave-one-protocol-out comparisons
(30 s, 120 s, 300 s sine and step) for the smallest supported candidate set.
All twelve runs have informed model selection, so these are internal transfer
checks. Include parameter, noise, observation-bin and model-form sensitivities
when discussing predictive coverage.

## Reproducibility, continuation and reporting

- Commit this plan first. Then create the explicit goal and record its start,
  plan commit and next unfinished phase in the execution manifest. Reuse the
  existing TC-Lab heartbeat for recovery; do not duplicate it or alter unrelated
  automations. Reconcile terminal artifacts and process identity on every resume.
- Use pinned Python/dependencies from `../environment.json`, single-threaded
  BLAS and at most four long numerical processes total. Freeze committed source,
  configuration and raw-data hashes for every measured job. Do not edit a live
  snapshot. Keep solver starts, convergence gradients, failures, exit status and
  elapsed work; retry only after understanding the failed condition.
- Reuse completed fits and audits; source-bind any analysis of their saved
  trajectories. Verify new mean calculations against a direct balance solver,
  covariance likelihoods against independent dense/small-case calculations,
  and generated tables against saved numeric output. Synthetic checks must test
  physical recovery, limiting cases and likelihood arithmetic rather than merely
  mirror the implementation.
- Preserve raw CSVs, prior reports and email receipts. Keep detailed numerical
  output in `error_structure_study`; update this file with durable decisions,
  not a running chat transcript. Commit implementation before measurement,
  results after verification and report after numerical and visual review.
- Consult primary sources for new theoretical or material-property claims;
  record exact versions, access limits and source locations. Existing literature
  authorization persists. Metadata-retention failure does not authorize library
  maintenance or eviction, and does not prevent using inspected primary sources.
- Write a concise report beginning with a question/status/result table, followed
  by error decomposition, matched model comparisons, uncertainty/transfer and
  open questions. Distinguish measured results, conditional calculations and
  hypotheses. Use the no-ai-slop writing check and the user's research practices.
  Produce Markdown plus a checked LaTeX/PDF report with the previously requested
  tablet layout: 0.25-inch left margin, 3.25-inch right annotation margin and
  college-ruled lines. No new email is authorized for this goal.
- Complete when scoped decision gates have terminal evidence, numerical audits
  and inspected figures/PDF, conclusions and limitations are committed and pushed,
  and no required job remains live. Pause the continuation heartbeat and present
  the report for Alex's scientific review. Honest non-identifiability or an
  unresolved residual mechanism can be a completed research result.
