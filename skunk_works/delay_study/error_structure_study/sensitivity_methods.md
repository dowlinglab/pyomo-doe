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

Frozen fits already running retain their original implementation. Their saved
likelihoods and stationarity will be checked with the improved score; no process
is restarted merely to change its source. Accepted old results are not silently
replaced. The new score also supports all nuisance-refitted recovery draws and
their deterministic code diagnostics after the score suite passes.
