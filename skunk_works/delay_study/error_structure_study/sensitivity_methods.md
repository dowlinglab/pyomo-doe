# Mean sensitivities and weak observed curvature

The first full information matrix used central differences of smoothing scores,
with each linear-mean sensitivity itself formed by a finite difference of an
eigensystem/exponential calculation. Most information directions were stable,
but the independent forward-likelihood loading-curvature audit failed: H22 was
0.07197 while a one-sided scalar difference gave 0.07859. The original matrix is
retained as a numerical diagnostic. It does not pass the full audit.

The replacement computes first mean derivatives from the Frechet derivative
of each held-input transition matrix. For generator A and direction E,
the upper-right block of exp([[A,E],[0,A]] dt) is the derivative of exp(A dt).
State sensitivities propagate by J_next = M J + dM x. The command and ambient
states are constant over each recorded update interval; command sensitivities
reset at every interval. Shared nominal loss coordinates and the per-run loss
deviation enter the generator separately. Hidden-heater and sensor-bin starts
enter the initial state. No finite differences of eigenvectors are needed.

`sensitivity.py` assembles the seven-state generator directly. Its check compares
all hardware trajectories with the existing normalized implementation in four
charts, compares directional derivatives with finite differences, and verifies
the Frechet derivative against the independent block-exponential identity.
The full-data mean difference is below 6.2e-13 C, and the largest scaled
directional derivative difference is below 6.9e-8. These checks establish mean
arithmetic, not likelihood-score correctness or interval coverage. The complete
joint score suite and a recomputed information matrix must pass before the
replacement supports uncertainty claims.

The saved forward-likelihood step study gives curvature 0.07876 at step 0.005
and 0.07859 at 0.02. At 0.0002, cancellation affects scalar differences; at 0.05,
one-sided truncation changes curvature to 0.07756. The independent audit uses
0.005 and 0.02 with its original 0.001 agreement threshold. It does not relax
that threshold to accept the original matrix. The original diagnostic omitted
one 0.03 evaluation from the 0.01 stencil and failed only at final assembly;
the arithmetic recovery preserves five complete stencils and records that gap.

Frozen fits already running retain their original implementation. Their saved
likelihoods and stationarity will be checked with the improved score; no process
is restarted merely to change its source. Accepted old results are not silently
replaced. The new score also supports all nuisance-refitted recovery draws and
their deterministic code diagnostics after the score suite passes.

## Consequential hidden-start nuisance profile

The quantized +/-2 and +/-5 C hidden-heater fits coincide and reverse the
preferred lag assignment relative to the cold baseline. Their largest selected
heater offset is -0.514 C in H1 of the first 300 s sine record (record index 6).
This triggers the plan's startup-profile condition.

Before measuring a profile, uncertainty_config.json declares the representative
coordinate rule: largest absolute hidden-heater offset in the audited +/-2 C
fit, with coordinate-order tie breaking. Profile it at -2, -1, 0, 1 and 2 C plus
its exact selected value. Each point refits all other dynamics, conductance,
ambient, twenty-three hidden starts and covariance nuisances from the usual two
lag branches, 300 iterations each and 128-node compiled scores. Source/data,
failed starts and the nearest successful warm start are retained. This is an
exploratory representative chosen after the observed discrepancy; no ordinary
chi-square confidence endpoint follows from the coarse grid. Fixing this one
offset to zero leaves twenty-three starts free and is not the all-cold model.

check_startup_profile.py must pass before run_startup_profile.py is measured.
It checks independent dimensional propagation, all four hidden-heater bin scores
on two short generated actual-command records, reference/compiled agreement,
fixed-coordinate objective/bounds/KKT behavior and the fitter's startup CLI flag.
The optional flag leaves old default commands unchanged. This profile uses fresh
tasks, not an unsupported startup continuation. Every accepted profile point then
needs its full independent selected-point audit. A point improving the free
startup reference by more than 0.001 requires free-baseline refinement before
interpreting relative likelihoods. Neither the physical initial states nor the
measurement-error initial posterior are established by these bounded fits.

### Initialization comparisons in observation-code space

initial_reported_code_summary binds eight independently audited initial-state
fits and all 12 raw observed/time arrays per fit. It recomputes stationary
marginal reported-code expectations and full scored RMSE. Four hundred
independent scalar erfc bin-integration witnesses agree within3.5e-13 C.
The freely initialized floor/nearest fits differ in expected reported output
by at most0.00030 C, although their analogue-reference RMSE differs. Exact
translation by half the grid preserves expected codes within3.1e-12 C.
Consequently, that latent-reference RMSE change is not a prediction benefit.
The fixed-start cases do not have the same reference freedom. All metrics are
in-sample stationary marginals, with no claim of whitening or transfer.
