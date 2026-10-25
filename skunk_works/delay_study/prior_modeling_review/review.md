# Prior TC-Lab modeling: evidence for the next investigation

Prepared 2026-10-25 for Alex's discussion. Agent-authored autonomous preparation,
awaiting scientific review. This is a focused review of modeling approaches and
their relevance to the remaining input-linked error. It is not an exhaustive or
systematic review, and it does not identify the cause of error in Dan's device.

## What the literature changes

There is a documented basis for testing nonlinear heat loss, temperature-dependent
power and structured environmental disturbances. None requires interpreting the
fitted short lag as a measured actuator time constant. The priority is to separate
these mechanisms with existing data before adding them simultaneously.

1. A convection-plus-radiation model with separate heater and sensor temperatures
   is an established TC-Lab model. Its radiative term changes with absolute
   temperature; a locally fitted linear loss coefficient can absorb much of it.
   Refitting is necessary when radiation is introduced.
2. Author documentation says delivered power changes with temperature at fixed
   heater settings. An unknown constant conversion belongs in the gain; a
   temperature-dependent conversion can change the response shape.
3. A fitted disturbance process can improve prediction while obscuring a faulty
   deterministic response. Published stochastic TC-Lab work evaluates the filter,
   noise-free response and distribution of innovations separately. We should too.
4. Published extra delay may have been deliberately added in software. Such work
   does not establish a long physical transport delay in the hardware.

The evidence matrix below distinguishes measured hardware studies, official
educational examples and incomplete source leads. Model scores across their
different devices, inputs, outputs and error definitions are not comparable with
our pooled RMSE. The present roughly 1 C T1 result is RMSE, not a sine-amplitude
error. Our completed [coupled-model study](../coupled_model_study/report.md)
remains the source for ENV2 numerical conclusions.

## Evidence matrix

| ID and source | Type and inspected coverage | Model, data and validation | Relevance and limits |
| --- | --- | --- | --- |
| P1. Park, Martin, Kelly & Hedengren (2020), [Benchmark temperature microcontroller for process dynamics and control](https://doi.org/10.1016/j.compchemeng.2020.106736); [author manuscript](https://apm.byu.edu/prism/uploads/Members/2020_park_tclab.pdf) | Peer-reviewed article; inspected author preprint, 37 physical PDF pages. Relevant context pp. 5–9, empirical models pp. 9–15, comparison p. 16. Preprint dated 2019-10-01; publication is 2020. | Compares coupled convection/radiation balances with sensor lags, FOPDT, ARX and a static neural map followed by linear dynamics. Separate step-test validation is shown. Table 2 reports average absolute error, not our RMSE: MIMO physical 0.23 C training and 0.70 C validation; MIMO ARX 0.26/2.66 C. | Physical structure deserves a matched test. The much poorer validation of some low-training-error models cautions against selecting by fit alone. Nominal constants belong to this device/model. The paper's interheater radiation uses area A while some educational code uses As; do not merge constants or equations silently. |
| P2. Kuntz & Rawlings (2025), [Maximum Likelihood Identification of Linear Models with Integrating Disturbances for Offset-Free Control](https://doi.org/10.1109/TAC.2025.3547607); [extended report v3](https://arxiv.org/html/2406.03760v3); [author code](https://github.com/rawlings-group/mlid_2024) | Peer-reviewed article with an inspected extended report, revised September 2025. Read problem formulation, §6.1 TCLab, Table 1 and Figs. 2–8; repository README inspected. | Fits stochastic LTI plant/disturbance models with integrating output disturbances; constrains filter eigenvalues. TCLab default is two plant states and two disturbances, full innovation covariance, C=I. Shows noise-free training response, independent closed-loop tracking/disturbance tests and covariance-normalized innovation diagnostics. | Direct grounding for an alternative to independent channel OU errors. Low likelihood alone can accompany unstable or slow filters. Feedback prediction quality does not establish free-simulation accuracy or physical parameter identification. Report notes some optimizations reached the iteration cap. No specific fan/USB disturbance mechanism was verified in this source. |
| D1. [Notre Dame four-state model](https://ndcbe.github.io/controls/tclab/03.06-Four-State-Model.html) | Author educational notes; four energy balances and stated assumptions inspected. | Reciprocal heater/sensor balances plus interheater transfer produce four thermal states. Coefficients depend on capacities and conductances. | Basis for our physical model, not validation of fitted hardware constants. The displayed state matrix has capacity-symbol inconsistencies; derive it from the differential balances. Modal poles and transfer zeros must both be retained. More states do not automatically provide arbitrary phase lag. |
| D2. [APMonitor radiative heat transfer](https://apmonitor.com/pdc/index.php/Main/TCLabRadiative) | Educational first-principles example and code inspected. | Heat loss combines convection and the Kelvin fourth-power radiation difference. Nominal emissivity, area and convection coefficient are prescribed for an exercise. | Supports the parent's radiation-scale calculation in the living note. It does not measure ENV2 heat loss, emissivity or radiative surroundings. |
| D3. [APMonitor parameter estimation](https://apmonitor.com/do/index.php/Main/TCLabC) and [nonlinear MPC code](https://apmonitor.com/do/index.php/Main/TCLabG) | Author educational model/code; state and heat-transfer equations inspected. | C uses two directly measured temperature states in nonlinear balances. G uses hidden heater states with separate nonloading sensor lags. Both include ambient convection/radiation and interheater transfer. G adds measured feedback to predictions. | Reproducible structure for a targeted nonlinear extension. The inspected code uses As for interheater radiation, unlike P1's displayed A. Feedback-corrected control plots cannot substitute for open-loop validation. Fixed mass, heat capacity, area and emissivity are assumptions, not identified truth. |
| D4. Kantor, [Python coding for TCLab, §1.3.5.1](https://jckantor.github.io/cbe30338-book/tclab/00.03-using-tclab.html) | Author software/teaching documentation inspected. | Documents Q as fraction of a P-dependent power ceiling and explicitly says power at constant P settings is temperature dependent and varies across devices. | Supports checking a power-versus-temperature hypothesis. Supplies no calibrated function for Dan's unit. It does not make constant alpha separately identifiable. |
| D5. [APMonitor heater actuators](https://apmonitor.com/pdc/index.php/Main/TCLabActuator) | Author hardware teaching page/code inspected. | Describes transistor heating, PWM, power-supply/cable losses and different heater power ceilings. Demonstrates FOPDT approximation of Q2 to T2. | Power-path effects are plausible, but nominal watts and delay examples cannot be transferred across firmware/power settings. PWM period is not evidence for a 2.7 s actuator lag. |
| D6. [APMonitor higher-order models](https://apmonitor.com/pdc/index.php/Main/TCLabHigherOrder) | Author educational higher-order example inspected as a model-family source. | Separates heater/sensor states and the response of the second temperature to heating the first unit. | Some higher-order paths are Q1 to T2. Their additional order cannot be cited as evidence that the directly heated T1 path has the same transfer. |
| P3. Di Teodoro et al. (2024), [A Hybrid Control Framework for Chemical Processes with Long Time Delay](https://pmc.ncbi.nlm.nih.gov/articles/PMC11292832/) | Peer-reviewed ACS Omega article; experimental §5.2 and validation/control context inspected. DOI 10.1021/acsomega.3c10514. | Physical TCLab tests use reaction-curve FOPDT/fractional approximations and a fan disturbance. The authors explicitly add 200 s delay in software. | Confirms airflow is a practical disturbance and that effective low-order models are used. The added 200 s delay is not a native hardware delay. No radiation-identification or free-run uncertainty result follows from the control experiment. |
| P4. Nguyen & Tique (2024), [Physics-Constrained Taylor Neural Networks](https://arxiv.org/html/2410.02258v1) | Inspected arXiv v1, §V-B (physical PDF pp. 5–6). Preprint, not asserted peer-reviewed. | Learns a monotonic Taylor neural map and controls physical TCLab. Reported training has 250 samples, 15 s sampling, commands between 10–50% changed every 120 or 150 s. The displayed multistep prediction table concerns HVAC, not TCLab. | A documented flexible nonlinear benchmark. Its TCLab evidence is chiefly a closed-loop demonstration; sampling is too coarse to resolve our few-second lag well. Does not identify radiation, calibration or uncertainty coverage. |

## Partial leads that must not become established findings

- Santos, Pinheiro & Leonardi (2026), [SINDy versus WSINDy on TCLab](https://ojs.sba.org.br/index.php/cba/article/view/5980), CBA preprint posted October 5. Only the landing-page abstract was accessible. It describes linear/quartic/constant/input candidate terms and hardware plus simulation, and claims radiation terms were weakly identified. Full methods, the identifiability threshold and validation split remain unverified. Do not quote its comparative RMSE as an established result.
- Montes Lozano (2025), [Control predictivo basado en datos. Aplicación al sistema térmico TCLab](https://idus.us.es/items/11fc9f85-5aa8-4e69-9049-e47dd6e3e2be), bachelor's thesis. Repository metadata/abstract describes NARX learning and DeePC on simulation and hardware. The PDF endpoint returned HTTP 403; model order, test design and numerical claims remain unread.
- A KTH dissertation/appendix surfaced at [diva2:1603520](https://www.diva-portal.org/smash/get/diva2%3A1603520/FULLTEXT01.pdf). Search text suggests fitting separate sensor times within a nonlinear physical model. Repeated web access timed out. Bibliographic identity and full equations have not been verified, so it is an acquisition lead only.
- Kuntz's [2024 dissertation](https://sites.engineering.ucsb.edu/~jbraw/jbrweb-archives/theses/kuntz.pdf) is accessible and overlaps P2. It is a useful reading path, not an independent replication counted as another modeling study here.

## Discriminating checks suggested by this review

These are proposals for the next goal, not new fits performed in this review.

| Competing explanations | Existing-data check | What it would leave unresolved |
| --- | --- | --- |
| Temperature-dependent power versus nonlinear heat loss | Compare residual behavior at similar temperatures but different commands, and during heating versus cooling. Start with one constrained mechanism at a time and compare matched frequency/trajectory errors. | Both affect the energy balance; without electrical power or heater-surface data, temperature-only agreement need not identify their separate laws. |
| Room-temperature forcing versus varying airflow | Compare common and differential channel errors; distinguish variation in ambient input from variation in conductance. | Shared or differential structure is not a unique causal fingerprint. No ambient or airspeed record has been established. |
| Missing deterministic dynamics versus flexible disturbance filtering | Compare free simulation, short-term prediction, command/innovation correlation and cross-channel innovation dependence under the same calibration data. | A better filter can hide a deterministic mismatch. Innovation covariance fit is not physical validation. |
| Additional thermal state versus timing/measurement filtering | Preserve the measured amplitude/phase evidence and compare transfer zeros as well as poles. | A fitted cascade cannot uniquely label its hardware mechanism. Q2 is never driven in ENV2, limiting discrimination. |

## Reading path and required information

Read P1's physical balances and its separate validation table first, then P2's
TCLab section and innovation diagnostics, then D4/D5 for the electrical-power
assumptions. Inspect D3's actual equations before borrowing constants. P4 is a
later flexible-model comparison, after a matched linear benchmark.

Before assigning physical parameters, recover P1/P2 settings and firmware for
Dan's collection, any electrical supply measurements, heater/sensor geometry,
and any logged ambient or cooling history. These are requirements for interpreting
mechanisms; missing values should remain unknown rather than inferred from a
good temperature fit. No claim of novelty or exhaustive literature coverage is made.

See [search and verification record](search_log.md) and [source receipts](receipts.json)
for scope, failed access and the unavailable corpus lookup. No source PDF bytes
were captured or hashed by this review; URLs, inspected versions and page/section
locators are retained. These notes have not been grounded in a new Lab Literature
evidence packet or reviewed by Alex.
