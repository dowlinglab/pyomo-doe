# Working approach for the next TC-Lab analysis

This file records the approach agreed with Alex during the October 24, 2026
discussion. Alex requested launch on October 24. The new goal is active;
`../residual_study/execution_plan.md` records implementation choices and
`../residual_study/PROJECT.md` owns current execution state.

## Agreed teaching objective and analysis sequence

Focus the next phase on the n = 1 reduced model so that the analysis can support
teaching. Start with constant input holds and per-run constant ambient offsets;
evaluate linear ambient drift later. Alpha remains explanatory material for
physical interpretation, rather than an additional temperature-fit parameter.

Alex wants detailed analysis of the residual/error structure, following this
sequence: fit the temperature model, assess residual structure, update the
covariance/weights, perform weighted regression, and repeat. Begin with pooled
exploration across all twelve unique ENV2 records, then examine how the analysis
changes parameter uncertainty and predictions under Tier 1 restricted calibration.
The iteration is an agreed analysis goal; its numerical implementation and
stopping criteria still need to be specified.

The intended next goal, over the coming couple of days, is limited to this
analysis of existing data: n = 1 mean fitting, residual/error-covariance fitting
and iteration, uncertainty propagation, and predictions for the declared
evaluation runs. All Pyomo.DoE work, experiment-design optimization and new hardware
experiments are outside that goal. Random-step/cooldown experiments and
sequential design work belong to a separate future phase.

## Agreed Tier 0: pooled exploration

First ask whether one consistent model explains multiple experiments. Tier 0
uses all twelve unique ENV2 recordings jointly, counting the calibration aliases
only once. Fit shared n = 1 K, τ and θ with per-run ambient offsets and examine
residual consistency across runs and input protocols. Here, "all data" means
this same-device ENV2 group; it does not silently include files outside it.

Alex describes this as testing for a "universal" model: one shared thermal
parameter set that can explain all of these records, together with the declared
per-run ambient treatment and a coherent residual model. Universal here refers
to the supplied ENV2 experiments under these assumptions, rather than every
device or operating condition.

Use Tier 0 to establish what the model needs before interpreting restricted-data
results. In particular, investigate:

- Whether constant ambient offsets suffice across runs and input protocols.
- Whether linear within-run ambient drift is needed, and whether a linear trend
  is sufficient to explain the remaining systematic behavior.
- Whether the mean/covariance iteration converges to a stable, adequate residual
  description, including its ability to account for serial correlation in the
  innovations across multiple experiments.
- Where discrepancies persist, whether they point toward changes in the ambient
  treatment, thermal mean specification or stochastic error structure. Separate
  those explanations rather than letting a flexible covariance absorb every
  discrepancy.

Report per-run and per-protocol diagnostics alongside pooled fit statistics so
that agreement on average does not conceal failure on particular experiments.
Distinguish numerical convergence from adequacy of the residual model. Failure
of a fitting attempt does not prove that no shared model exists; a satisfactory
pooled fit supplies conditional evidence for consistency on these data.

Tier 0 is explanatory pooled fitting. Its fitted responses are not out-of-sample
validation, and its uncertainty calculations are exploratory rather than final
claims of calibrated uncertainty. Tiers 1 and 2 are defined below; any further
tiers are still being discussed.

## Agreed Tier 1: one sine test and one step test

Use the shared model structure established in Tier 0, once its thermal,
ambient and residual descriptions are satisfactory. Freeze those structural
choices, then regress the model using only one sine-wave experiment and one
step experiment. Quantify parameter uncertainty and predictive uncertainty
from that restricted calibration, and compare predictions with the remaining
distinct ENV2 experiments.

The scientific and teaching question is: how good a model can we build from
just one step test and one sine test, conditional on the model structure learned
in Tier 0? Compare both prediction accuracy and the usefulness of the uncertainty
bands across the different evaluation input protocols.

Carry forward the selected structure, rather than treating Tier 0's fitted
parameter values as known. Re-estimate the thermal parameters, the selected
error-model parameters and the two calibration runs' ambient nuisance terms
using only the Tier 1 calibration pair. Tier 0 estimates can supply numerical
starting points, but must not silently supply extra observations, fixed noise
parameters or prior information in the Tier 1 uncertainty calculation.

Keep parameter/thermal-mean uncertainty separate from future-observation
variation. For evaluation runs, specify the available ambient and initial-state
information and propagate unresolved conditions; fitting an ambient correction
to a complete evaluation response is a separate conditional reconstruction,
not the same prospective prediction.

This tier assesses data economy under the Tier 0-selected structure. The
retrospective model-selection qualification below remains explicit; it does not
prevent a useful comparison of what changes when fitting only two records.

### Calibration and evaluation records

Keep the current calibration pair: the five-minute sine record and the constant
50% step record. Their ENV2 run 1 aliases are duplicates of these same physical
records, so include each only once. These are the pair previously agreed in
discussion; record any later change in which sine/step records are chosen.

For the later restricted-calibration evaluation, exclude the other ten unique
ENV2 records from that tier's parameter and covariance fitting:

- Three half-minute sine records.
- Three two-minute sine records.
- Two five-minute sine records.
- Two step records.

These evaluation data are repeated sine/step hardware recordings, not waveforms
generated by a DoE optimization. The standalone August 23 file
`validation_experiment_sin_5_50.csv` is outside ENV2 and is not part of this
declared primary validation set. Any use of it requires a separate scope
decision; do not silently add it to primary validation.

After Tier 0 has inspected the ten records, they are no longer untouched for
scientific model selection. If pooled exploration guides the model, covariance
or evaluation-protocol choices, later performance on these records is
retrospective validation. Calling the pooled work exploratory does not restore
independence. Independent confirmatory evidence would require new unseen data,
or a fully nested/predeclared fold-specific selection procedure that protects
each evaluation fold from all model-selection information. New hardware data
collection is outside this goal. Report the actual information reuse in every
tier rather than implying that the later fitting split removes it.

## Agreed Tier 2: five additional sine/step pairs, then researcher review

Repeat the complete Tier 1 procedure with different choices of one sine-wave
record and one step record, keeping the Tier 0-selected structure fixed. For
each pair, refit the thermal, ambient and residual-model parameters, quantify
uncertainty and evaluate predictions on the other ten distinct ENV2 records.

The ENV2 group contains nine sine tests (three repetitions at each of the
half-minute, two-minute and five-minute periods) and three step tests. This
gives 27 possible sine/step calibration pairs. The Tier 1 reference pair is
one of those 27; reuse its saved result rather than treating it as a separate
independent experiment. Calibration aliases add no new records. The standalone
sine file outside ENV2 is not included in these counts.

The initial authorized analysis will cover six pairs total: the Tier 1 reference
pair plus five additional Tier 2 pairs. Across those six, each of the three step
records appears in exactly two pairs. Spread the sine selections across periods
and repeated records. Do not enumerate all 27 in the first pass.

The following is a proposed balanced selection for the detailed plan. It uses
six distinct sine records, each sine period twice and each sine repetition index
twice, while retaining the agreed Tier 1 pair:

| Pair | Tier | Sine period | Sine run | Step run |
|---|---|---|---:|---:|
| P1 | 1, reference | 5 min | 1 | 1 |
| P2 | 2 | 30 s | 2 | 1 |
| P3 | 2 | 2 min | 1 | 2 |
| P4 | 2 | 5 min | 3 | 2 |
| P5 | 2 | 30 s | 3 | 3 |
| P6 | 2 | 2 min | 2 | 3 |

Exact P2–P6 assignments are agent-proposed; the six-pair budget and two uses of
each step record are Alex's decisions. File pattern `sin_50_50` denotes the
half-minute sine records in this dataset.

Use this tier to assess sensitivity to sine period and to the particular
repeated runs selected. Report results across pairs and per evaluation
protocol, rather than reporting only the best pair. The pairs share records
and are not independent experimental replications. Keep the retrospective
structure-selection qualification from Tier 0 explicit.

**Researcher review checkpoint:** deliver Tier 0, Tier 1 and these first five
Tier 2 results for Alex to inspect and provide feedback. Alex anticipates an
eventual enumeration of all 27 pairs, but expansion is outside the first pass.
Wait for that feedback and a subsequent instruction before running the
remaining 21 pairs or automatically expanding the study.

## Completion deliverable: tablet report emailed through Gmail

At completion of the initial Tier 2, prepare a PDF explaining the Tier 0–2
methods and results, including residual/error-model diagnostics, uncertainty,
evaluation predictions and unresolved questions. Write it for Alex's review
before expanding the study. The next goal's deliverables include both the
checked report and its delivery, not merely completion of the fitting jobs.

### Agreed report organization and level of detail

Keep the report concise. Begin with a goals-and-status table, rather than a
separate bullet-point recap. Use one row per tier goal so that goals with
different outcomes are not collapsed into a single completion label. Columns
should identify **tier**, **goal**, **status**, and a short **result or open
issue** with a pointer to the relevant report section or figure.

Status must distinguish completed calculations from scientific adequacy. For
example, a fit may have converged while residual correlation remains unresolved.
Use clear labels such as completed, partly resolved, unresolved or not attempted,
with a short explanation as needed. Do not mark a scientific goal achieved
merely because the corresponding jobs finished.

Follow the opening table with sections titled **Tier 0**, **Tier 1** and
**Tier 2**. In each section, walk Alex through the results in relation to that
tier's goals, explain the supporting diagnostics and uncertainty, and identify
remaining questions. Cover the substantive findings and all six selected pairs,
including unfavorable or inconclusive results, without turning the report into
a run log or repeating the full methods at every tier. Summarize comparable
results together in compact tables and readable figures.

End with the open questions and decisions needed from Alex before Tier 3.
Distinguish what the current results answer, what remains uncertain and which
next steps depend on feedback. No fixed page limit has been requested; use the
shortest report that communicates the evidence and limitations clearly in the
tablet layout.

Alex explicitly requests this page layout for reading and handwriting on a tablet:

- Left margin: **0.25 inch**.
- Right margin: **3.25 inches**.
- Put **college-ruled horizontal lines in the right margin**, leaving that area
  available for handwritten comments on each report page.
- Keep report prose, figures, captions and tables inside the main content column;
  they must not encroach on the ruled notes area.

Use US Letter unless Alex specifies otherwise; this gives a 5-inch content
column. The proposed college-rule spacing is 9/32 inch, with light rules that
remain visible on a tablet without competing with the report. Top/bottom margins,
rule color and small writing-area gutters are layout implementation choices;
they do not alter the requested left/right margins.

Compile the report and inspect its rendered pages before delivery. Check
readability of plot labels and tables in the narrower content column, exact
margin dimensions, and unobstructed ruled writing space. Keep the LaTeX source
and bindings from reported values/figures to the saved analysis results.

Send the finished PDF to **adowling@nd.edu through Gmail** when Tier 2 is
complete. This instruction authorizes that future self-email; do not send a
report now or resend either earlier report. Send one completion email, verify
the message and attachment metadata, and record a delivery receipt so recovery
does not create duplicate emails. Preserve any unresolved scientific limitations
in the report rather than treating numerical completion as model validation.

## Deferred Tier 3: full pair enumeration after feedback

The new goal will cover Tier 0 through the initial Tier 2 only. Tier 3 is the
natural next progression after Alex reviews the emailed report and provides
feedback: complete the 27-pair enumeration or rerun that enumeration under a
revised analysis, as appropriate. It is documented future work, not part of
the initial execution budget. Do not start Tier 3 without a subsequent request.

## Agreed assumptions and first formulation

Alex trusts the ENV2 data provenance and considers those records to have been
collected carefully on the same experimental device. Treat that as the working
scientific assumption. The Git inspection is supporting repository evidence;
it does not independently establish device identity or acquisition timestamps.

Apply the following changes together, then reassess the fits, residuals and
uncertainty:

| Component | Agreed treatment |
|---|---|
| Heater input | Piecewise constant between command updates; replace the linear interpolation used in the report's baseline. Check command/measurement timestamp alignment before selecting the interval endpoint convention. |
| Thermal parameters | Start with n = 1 and fit shared positive K, τ and θ jointly across the selected ENV2 fitting records. Any later model-order comparison uses the same sharing convention. |
| Alpha calibration | A shared, constant alpha is absorbed into the freely fitted gain K. Do not add alpha as an extra fitting parameter or expect alpha uncertainty to improve the temperature fit. Retain calibration uncertainty only when interpreting physical conductances/capacities. |
| Ambient temperature | Start with one constant offset per experiment, bounded by ±2°C relative to that experiment's reference ambient. |
| Initial objective | Joint temperature sum of squared residuals across fitting records. Treat this as an initial estimation objective, without assuming IID uncertainty is valid. |
| Error analysis | Inspect the resulting residual structure before selecting the final error model and uncertainty procedure. |

For experiment j, write

\[
T_{\mathrm{amb},j}=T_{\mathrm{ref},j}+\delta_j,
\qquad -2\leq\delta_j\leq2\;{}^\circ\mathrm C.
\]

The previous analysis used the first sensor reading as the reference ambient.
The definition of the reference and treatment of initial-state measurement error
must be made explicit in the execution plan. A constant offset within a run is
the agreed starting point; within-run ambient drift is a later comparison.

For physical interpretation, the shared calibration can be written as

\[
\alpha=\alpha_{\mathrm{ref}}m_\alpha,
\qquad m_\alpha>0.
\]

The inherited reference is α_ref = 0.00016 with heater setting P = 200.
No multiplier or calibration distribution is needed for the next temperature-only
fit in K, τ and θ.

## How alpha enters the optimization and uncertainty

In gain/time coordinates, temperature predictions depend on

\[
K=\frac{\alpha P}{U_a},\qquad
\tau=\frac{C_p}{U_a},\qquad\theta.
\]

Therefore alpha does not add an independently estimable parameter to a fit
already parameterized by K, τ and θ. An unconstrained alpha multiplier would
have no effect on that temperature objective. In physical coordinates, jointly
scaling alpha, U_a and C_p leaves the response unchanged.

Alex's clarified decision: leave alpha out of the next optimization. A shared,
constant alpha multiplier does not expand the temperature-response family when
K is already free, so adding it cannot improve the best achievable temperature
fit over the same feasible response family. The fitted K is an overall
command-to-temperature gain; its uncertainty is inferred from the temperature
data and should not receive an additional alpha-uncertainty term merely because
alpha is one of its physical factors.

If absolute conductance and capacity are later reported, use external calibration
information or a declared calibration sensitivity range. Alpha uncertainty then
enters the physical conversion

\[
U_a=\frac{\alpha P}{K},\qquad C_p=\frac{\alpha P\tau}{K}.
\]

The implementation of that optional physical-interpretation calculation can be
chosen later. Do not claim that the thermal traces alone identify the absolute
conductance/capacity scale. K and τ are not themselves made nonidentifiable
merely because alpha is uncertain.

## Agreed extension for ambient drift

After the constant-offset formulation, evaluate a linear ambient change within
each run:

\[
T_{\mathrm{amb},j}(t)=T_{\mathrm{ref},j}+\delta_j+s_jt,
\qquad 0\leq t\leq H_j.
\]

Time t is elapsed time from that run's start. Setting s_j = 0 recovers the first
formulation. Alpha remains shared and constant. No slope bounds have been
agreed. Also decide whether the ±2°C restriction applies only to the intercept
or to the full ambient trajectory; do not silently choose either interpretation.

Evaluate the extension through fit changes, residual diagnostics, parameter
estimability and validation, rather than accepting it solely because adding
parameters lowers the fitting objective.

## Proposed implementation: mean and error-model iteration

The following are proposed technical details, rather than additional decisions
already made by Alex.

1. Run Tier 0 pooled exploration first. For each subsequent restricted tier,
   preserve its declared fitting/evaluation split during that tier's covariance
   selection and fitting, while retaining the information-reuse qualification
   above. Fit the shared n = 1 thermal parameters and bounded per-run ambient
   offsets using the agreed input holds, initially by joint ordinary least
   squares.
2. Examine raw mean residuals, y minus the predicted mean, for temporal
   correlation, dependence on heater commands, trends, heteroskedasticity and
   the effects of recorded temperature quantization. Use these checks to
   distinguish possible mean-model errors from plausible stochastic errors.
3. Compare a small set of parsimonious covariance models using fitting records
   only. Implement feasible generalized least squares with full temporal
   covariance, or equivalent whitening. Diagonal weights alone cannot account
   for temporal correlation. Keep raw residuals separate from whitened
   innovations in plots, diagnostics and interpretation.
4. Alternate noise-parameter estimation with thermal/ambient-parameter fitting.
   Specify repeatable stopping diagnostics for parameter changes, objective
   changes and numerical convergence. If optimizing a Gaussian likelihood,
   include the covariance log-determinant as well as the covariance-weighted
   residual quadratic. Compare the alternating result with a joint likelihood
   fit as a numerical and inferential check.
5. Reassess residual and innovation diagnostics after fitting. Do not promise
   decorrelation or select models merely to force whiteness tests to pass. A
   flexible covariance model must not conceal errors in the thermal or ambient
   mean response; persistent input dependence or trends can motivate revisiting
   that response, including the later linear ambient comparison.

Estimate uncertainty with methods that account for covariance estimation and
the fitted per-run ambient offsets, rather than treating either as known.
Distinguish uncertainty bands for the thermal mean from prediction bands for
future observations, which also include observation/error-process variation.
Check the n = 1 pole-label ambiguity again under the new nuisance formulation.
Exchanging time scales preserves the cold zero-offset input response, but
ambient forcing enters through the thermal pole and may change equivalence.
Do not impose ordering or claim physical assignment without that check.

Within a restricted-calibration tier, keep its evaluation runs out of that tier's
covariance selection and fitting. This does not erase their earlier Tier 0 use.
For prospective predictions, define what ambient and initial-condition
information is available before each run. Do not estimate a holdout ambient
offset from its complete temperature response and call the result a prospective
prediction. Separate predictions conditional on available pre-run measurements
from predictions that must propagate unknown future ambient conditions. The
±2°C bound alone does not specify a probability distribution for that
uncertainty; its treatment remains an execution-plan choice.

## Choices still open

- Whether and when to extend beyond the agreed n = 1 teaching analysis,
  including other delay orders or the full four-state/two-channel model.
- If absolute physical parameters are requested later, alpha calibration
  information and its applicability to the ENV2 device. This is not a prerequisite
  for the next gain/time-coordinate temperature fit.
- Ambient reference, initial-temperature treatment and zero-command prehistory
  specification. Alex considers cooled startup with zero input history reasonable.
- Exact zero-order-hold timestamp convention.
- Linear ambient slope bounds and treatment of the full-trajectory ambient limit.
- Candidate residual models, diagnostics, uncertainty methods and acceptance checks.
- Numerical stopping rules for alternating estimation, the joint likelihood
  comparison and treatment of prospective ambient uncertainty.
- Final P2–P6 record assignments within the agreed initial six-pair budget.
- How retrospective evaluation and any nested/fold-specific selection will be
  reported after Tier 0 exploration. The full enumeration is already documented
  as deferred Tier 3, to be scoped after Alex's feedback.

Alex has now requested the goal. Technical defaults and rationale are recorded
in `../residual_study/execution_plan.md`; distinguish them from Alex's agreed
scientific decisions above.

## Related review material

- [Transcribed report comments](comments.md)
- [Analysis and discussion brief](analysis.md)
- [Repository data-provenance inspection](data_provenance.json)
- [Conditional four-state identifiability check](four_state_scope_check.json)
