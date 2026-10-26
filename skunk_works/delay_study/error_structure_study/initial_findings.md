# First error decomposition

The saved coupled physical model leaves mostly slow T1 error. In the constant
ambient model with a short input lag and command curve, T1 RMSE is 1.022 C.
Nested projections assign 74.6% of squared error to per-run mean offsets,
19.3% to additional linear trends, 0.64% to the fundamental drive frequency,
0.18% to harmonics two and three, and 5.31% to the remainder. After 450 s,
96.1% is per-run offset; the total RMSE then is 1.234 C. Startup exclusion
therefore does not remove the sustained level discrepancy.

The fast-sine T1 mean errors are -0.945, -1.027 and -0.747 C. The 300 s sine
means are +1.164, +1.439 and +0.376 C. This protocol dependence can be checked
against command and temperature-dependent loss, bounded conductance variation
and the colored-error criterion. It does not prove airflow changed between runs.
The additional lag previously improved fast phase; the remaining approximately
1 C RMSE should not be described as a 1 C sine-amplitude error.

T2 changes more under affine ambient: full-record RMSE falls from 0.231 to
0.124 C while T1 remains 1.032 C. The nested partition leaves 93.6% of T2
squared error in the remainder after affine ambient. That remainder still
includes quantization, stochastic error and any unmodeled structure; it is not
an estimate of IID measurement variance.

The partition is an ordered orthogonal-projection decomposition: per-run
constant, additional linear trend, fundamental, harmonics two and three,
remaining error. Changing the order can change the attribution. Cutoffs were
prespecified at 0, 60, 180 and 450 s; these are descriptive projections and are
not physical ambient offsets, inference intervals or a new thermal regression.
All twelve records already informed model selection.

Source was committed as `c407107` before measurement. All sixteen prior fit,
trajectory and audit bindings matched; all 26 declared prior jobs were terminal
with exit code zero. Raw CSV hashes and saved input/observed arrays matched in
all four selected model families. The independent QR auditor checked 384 rows,
32 pooled aggregates and error closure, with maximum energy discrepancy
9.1e-12 C2. Figures were inspected at native reading scale. Quantization-aware
and simulated-reference whitening checks remain to be done.

Evidence: [aggregate table](results/decomposition/aggregate.csv),
[per-run partition](results/decomposition/decomposition.csv),
[innovation diagnostics](results/decomposition/innovation_diagnostics.json),
[reconciliation](results/decomposition/reconciliation.json),
[independent arithmetic audit](results/decomposition_audit/audit.json),
[error partition figure](results/decomposition/error_partition.png) and
[selected residual curves](results/decomposition/selected_residuals.png).


The bounded-conductance equations also passed a zero-band nesting check,
independent direct energy-balance integration and a full-versus-envelope
profiled-gradient check. A three-record synthetic example recovers the known
centered conductance deviations from two starts when the shared dynamics are
fixed. This verifies a recoverable conditional problem; it does not establish
all-parameter identifiability in the hardware data. The three four-start real-data comparisons and their independent audits are now complete.

The radiation calculation uses scenarios, not a fitted emissivity. With area
0.0012 m2, surroundings 23 C and surface 70.54 C, the heat loss is 0.0213 W
at epsilon=0.05 and 0.3834 W at epsilon=0.9. At those same endpoints, the
nonlinear remainder beyond a linear radiative loss matched at 50 C is 0.0021
and 0.0370 W. A fitted linear loss can absorb the larger linear contribution.
Surface 100 C raises the high-emissivity heat loss to 0.7163 W and the remainder
to 0.1552 W. Hidden heater temperatures could therefore matter. These scenarios
cannot yet be translated into degrees of model error without a fitted energy
balance, nor establish the surface finish of this device.

Sources and qualifications are in [radiation_sources.md](radiation_sources.md);
computed [scenario table](results/radiation_scale/scenarios.csv) and
[independent quadrature/derivative checks](results/radiation_scale/audit.json)
retain the calculation. The two finishes differ by a factor of eighteen in the
assumed emissivity; the purpose is to show sensitivity, not assign a physical
confidence interval.


At ±10% conductance variation, pooled T1/T2 RMSE is 0.639/0.115 C. At ±20%,
it is 0.641/0.115 C, with little likelihood improvement. Two 10% deviations
reach the lower bound. The three fast-sine ratios exceed nominal, while some
slow-sine and step ratios are below nominal; this protocol dependence prevents
calling the inferred variation an airflow measurement. Three of four 5% starts
failed SLSQP status/feasibility checks and remain recorded alongside the selected
converged fit. A larger training likelihood is not a physical explanation.

With the 10% band, allowing linear ambient gives 0.728/0.111 C, while bounded
hidden-heater starting offsets give 0.615/0.111 C. Widening those starting offsets
from ±2 to ±5 C does not improve the result; all selected offsets are within
0.35 C. Initial disequilibrium has not explained most of the remaining T1 error.

Bin-likelihood refitting with the physical mean fixed gives an estimated analogue
measurement SD of 0.0255 C for T1 and 0.0325 C for T2. The corresponding old
continuous-data values were about 0.094 and 0.072 C. T2's fitted correlation
length grows from 7.9 to 54.8 s. Both channels passed full-record 512-node score
checks and independent trivariate rectangle integrals. These changes show why
whitening and uncertainty must be revisited under the discrete observation law.
They do not measure this device's calibration accuracy. Physical parameters
have not yet been jointly re-estimated under that bin likelihood.

The working Gaussian forecast comparison also separates simulation from noise
state feedback. With the 10% band, T1 one-second conditional RMSE is 0.111 C;
at 10/30/60 seconds it is 0.145/0.202/0.260 C. Free-simulation pooled RMSE is
0.639 C. The forecasts condition on preceding observations and use parameters
estimated from all twelve records. Their current intervals include observation
noise only, excluding parameter, model-form and quantization uncertainty. The
dense Gaussian conditioning identities and all 336 printed free-simulation RMSE
cells passed independent arithmetic checks.

Joint predictive residual diagnostics still need independent random streams.
The existing pilots used repeated seeds within channels/runs, so their marginal
illustrations should not be pooled to claim cross-channel or cross-run
independence. Simulated references and nonrandomized bin-transition diagnostics
remain necessary even when randomized residuals look Gaussian.


The scaled free-radiation comparison now has four accepted starts, independent
adaptive-ODE/dense-likelihood verification and a step-halving check. Its T1/T2
RMSE is 0.633/0.115 C versus 0.639/0.115 C without radiation at the same 10%
band. Gaussian nll improves by 1.285 for one additional parameter. The normalized
differential radiation coefficient at 50 C is 0.0344 relative to nominal
convection. This is a small, conditional improvement; boundary profiles and the
quantized observation comparison remain necessary before interpreting support.
It does not identify emissivity or area. With a linear command law, radiation
alone gives 0.924/0.119 C and a worse Gaussian likelihood than the empirical
command-curve model. Both comparisons refit convection and gain.

Joint physical-plus-bin scores passed coordinate finite-difference checks for
the conductive and radiative families at positive and zero conductance bands.
The maximum scaled derivative discrepancy was 5.7e-7. Full-data joint fits are
now running; none has been selected yet. Their starts also test the possible
exchange of sensor and input time constants. The shared mean, bounded per-run
conductance and observation distribution must be considered together before
final uncertainty or new-experiment prediction claims.
