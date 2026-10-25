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
all-parameter identifiability in the hardware data. Three four-start real-data
comparisons are now running and have no selected result yet.

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
