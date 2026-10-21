# Authorized overnight study (frozen brief, 2026-10-20)

## User request

Switch to `reformulation-exploration`, merge main, and work autonomously for
12+ hours. Systematically compare TC-Lab n-stage delay models, quantify
uncertainty, investigate identifiability/estimability with profile likelihood,
eigendecompositions and other methods, ground the study in LTI identification
and optimal experiment design literature using the literature skill, and compare
`Ua, Cp, theta` with `K, tau, theta`. First migrate notebook logic to Python.
Prepare a LaTeX draft-0 manuscript with literature, methods, results and discussion.
Commit as work proceeds, push when complete, and email the report via Gmail.
Follow the user's `claude-for-researchers` best practices.

## Completion evidence

1. Verified branch and merge; deterministic script migration with source hashes.
2. Cross-method simulation checks, transformation checks, and negative/limiting cases.
3. A declared comparison matrix covering delay orders 1, 2, 3, 4, 6, 8 and 12,
   the pure-delay limit, and the original two-thermal-state reference. Evaluate
   sine, step and joint calibration; use untouched validation records where
   their metadata support comparison. Include both fixed initial lag-state
   assumptions: zero and first recorded input. Report metadata limitations.
4. Fits with broad bounds, repeatable multistart, boundary/convergence reporting,
   trajectory and residual diagnostics, and sensitivity to input interpolation,
   initialization and numerical discretization.
5. Local covariance and prediction intervals, relative-parameter FIM/SVD and
   eigenvectors, parameter correlations, profile likelihood with nuisance
   reoptimization, and bootstrap checks accounting for serial residual structure.
   Distinguish IID/known-noise/unknown-noise assumptions; do not present conditional
   intervals as model-discrepancy uncertainty. Compare transformed covariances
   and prediction uncertainty, not raw condition numbers alone.
6. Derive structural identifiability from transfer functions, including the
   original physical model's redundant coordinates, pole interchange when
   appropriate, and the information supplied by fixed initial conditions.
7. Compare informative candidate LTI input designs under common horizon, input,
   sampling and safety assumptions. Distinguish parameter estimation from
   discrimination of delay order. No unvalidated physical hardware execution.
8. Source-inspected literature synthesis with search scope, assumptions,
   exclusions/access gaps, page locations, saved evidence packet, and checked
   bibliography. Include competing explanations and no unsupported novelty claims.
9. Generated manuscript tables/figures traceable to the run manifest, compiled
   LaTeX PDF with visual QA, explicit human-review status and AI disclosure.
10. Reproduction commands, verified checks, final commit/push identity, and exactly
    one completion email with the usable report (prefer PDF attachment) and branch.

## Phases and gates

- Baseline -> migration: run inherited code twice; preserve numerical behavior.
- Migration -> new methods: compare exported code cells to notebooks, compile
  Python exports, and execute representative noninteractive paths.
- New methods -> broad study: analytic LTI cases, independent integrator agreement,
  parameter round trips, derivatives, and identical-formulation prediction checks.
- Broad study -> interpretation: retain all fits, failures and configurations;
  use held-out experiments and noise/initial-condition sensitivity.
- Interpretation -> delivery: audit quantitative claims and bibliography, render
  figures/PDF, reproduce key outputs, inspect diffs, commit and push, send Gmail.

## Resource and review boundaries

Use at most four numerical worker processes and cap BLAS threads per worker.
Start with short trials, estimate throughput, then launch resumable bounded jobs.
Do not silently shrink declared comparisons or bootstrap/profile budgets to meet
time. Extend the work past 12 hours if needed to complete authorized deliverables.
Do not let an elapsed clock alone establish completion. Do not edit measured
source files while a computation is running. Do not modify other repositories,
shared environments, raw data, hardware, or remote services beyond the authorized
final push and self-email. Routine reversible methodological decisions can be made
and documented; author-level interpretation remains explicitly pending review.

## Predictions to challenge

- Physical and gain-time coordinates describe the same transfer function, so
  converged interior fits and transformed prediction uncertainty should agree.
- Raw FIM conditioning can change under a coordinate change without adding data.
- Delay order may be weakly distinguishable even when each conditional fit is precise.
- Initialization and residual correlation may dominate apparent order improvements.
- Input design cannot recover structurally redundant physical parameters from the
  same single-output zero-state transfer function; additional assumptions or
  measurements are needed. Fixed initial conditions can alter that conclusion.

These are hypotheses/derived expectations, not reported computational findings.
