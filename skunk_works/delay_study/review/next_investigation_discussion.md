# TC-Lab: evidence and questions for the next investigation

This is the working note for Alex's discussion following the coupled-model
study. Update it in place as the three workers return and as Alex makes
decisions. The purpose is to understand the remaining discrepancy before
writing and launching another research goal. **No new goal is authorized yet.**
Alex explicitly asked the workers and parent to keep going after voice chat
ends, populate this note, and preserve findings in the repository.

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
- The immediate work is data/source review and synthesis. No new hardware,
  experiment design, thermal-model fitting, goal, automation or email is part
  of these worker assignments.

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
| Hardware quantization | Inspect Dan's data for temperature grids; separate ADC quantization from CSV rounding; trace hardware conversion and reporting. | Running; findings pending. | To be linked below. |
| Sensor calibration | Official documentation, published pip package, firmware and documented calibration studies; distinguish offset, slope, reference-voltage and nonlinear effects. | Running; findings pending. | To be linked below. |
| Prior modeling literature | Physical and empirical TC-Lab models, radiation, convection, disturbances, sensor dynamics, identification and validation. | Running; preliminary source leads, synthesis pending. | To be linked below. |

Each worker owns only its named section below. Keep detailed source receipts,
code and numerical output in its separate review folder and link them here.
Populate this file with the findings needed for our discussion, not just links.
For each conclusion, state what was inspected, what it supports, what remains
unknown, and a source/evidence link. Mark preliminary or unverified leads.
Do not describe nominal tolerances or simulator defaults as observed errors in
this device. The parent owns the shared synthesis, decisions and proposed work.

## Hardware quantization findings — worker section

Pending. Report per-channel/run grid evidence, numerical precision and rounding,
plausible hardware conversion formulas, documentation/version limits, and
whether the data distinguish quantization from formatting.

## Sensor calibration findings — worker section

Pending. Report sensor identification, documented calibration/conversion,
reference-voltage assumptions, accuracy evidence and the consequences for the
shared model. Separate known mechanisms from an unproven calibration error in
Dan's device.

## Prior modeling literature — worker section

Pending synthesis. The worker has identified author documentation stating that
delivered heater power can depend on temperature and vary by device:
[Kantor, Python Coding for TCLab, section 1.3.5.1](https://jckantor.github.io/cbe30338-book/tclab/00.03-using-tclab.html).
This is a qualitative mechanism to investigate, not a calibrated power law for
ENV2. The worker will supply the reviewed evidence matrix and limits.

Parent source leads on radiation are
[APMonitor's radiative heat-transfer example](https://apmonitor.com/pdc/index.php/Main/TCLabRadiative),
[higher-order models](https://apmonitor.com/pdc/index.php/Main/TCLabHigherOrder),
and [two-heater parameter estimation](https://apmonitor.com/do/index.php/Main/TCLabC).
Example areas, emissivities and convection coefficients vary between models;
do not mix their constants or transfer channels without checking assumptions.

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

No cause of the remaining T1 error has been established. The reviews should
help rank hypotheses and specify discriminating analyses. Avoid adding many
free effects simultaneously: an improved objective alone will not tell us
whether the explanation is power delivery, heat loss, calibration, startup,
timing or measurement statistics.

Before drafting the next goal, agree on which findings justify a numerical
test, the order of those tests, what existing results can be reused, what
would count as a resolved discrepancy, and which questions need independent
measurements rather than more regression on the same temperatures.
