# Investigation verification and scope

Status: scoped requirements verified; ready for researcher review.
The scientific requirements come from [the committed plan](../review/tier0_followup_investigation.md)
at b4f7ef7. A numerical audit verifies an implementation; it does not establish
that a mean/error model adequately describes the experimental system.

| Requirement | Evidence | Interpretation or limit |
|---|---|---|
| Reproduce pooled reference | results/baseline_v2/audit.json | Independent propagation and dense likelihood; original failed receipt retained in baseline |
| Verify holds, irregular intervals and n=1/2/3 propagation | Forward audits saved with main fits; dynamics.py synthetic ODE comparison | Left holds and specified initial states; acquisition conventions remain assumptions |
| Separate offset, trend and cycle | results/frequency/error_decomposition.csv and results/frequency_figures | Sequential descriptive decomposition; no causal ambient/noise allocation |
| Repeatable amplitude/phase and uncertainty | results/frequency; results/frequency_audit/audit.json | All 54 estimates reconstructed using independent quadrature/QR; HAC lag and startup sensitivities retained |
| Flexible shared linear benchmark | flexible2/3/4, comparison_primary and information_final | Stable real-pole family with numerator freedom; not unrestricted LTI or subspace estimation |
| Matched delay-order comparisons | chain1/2/3, secondary_matrix, mapped_chain1/2/3 | Joint reestimated covariance, fixed-shape/IID sensitivities and revised mean comparison |
| Independent selected means, innovations and likelihood | primary_audits, flexible3_audit, flexible4_audit, targeted_audits, transport_audit, power_linear_audit, mapped_audits12, mapped_audit3 and mapped_unequal_audit | Raw hashes, independent means and dense covariance/innovation checks passed |
| Estimability and response uncertainty | information_initial, information_final, information_quadratic, information_mapped | Scaled SVD/eigenvalues, weak directions, pole permutation; local conditional approximations |
| Profile support | chain_profiles_audit, transport_profiles_audit, power_profiles_audit, beta_profiles, mapped_profiles1_audit and mapped_profiles23_audit | Every reported profile point independently checked; scoped grids do not certify global optima or coverage |
| Impulse-response shape | fir_sensitivity and fir_audit/audit.json | All 16 cells; independent command-step convolution and dense weighting; fixed reference covariance |
| Targeted nonlinear/ambient checks | quadratic, quadratic_audit; power_curve, power_curve_linear and selected audits | ODE/convergence checks for nonlinear mean; candidate mechanisms remain confounded |
| Prediction versus simulation | reference_diagnostics_v2, diagnostics_power_linear, diagnostics_mapped2, diagnostics_quadratic, diagnostics_mapped_unequal | Synthetic dense conditional mean/variance audit; observations only through each origin; fitted parameters remain retrospective |
| Innovation adequacy | innovation_assessment/assessment.json | Per-run autocorrelation, input association, scale and time-block checks are descriptive, not calibrated joint tests |
| Reproducible report and reviewed figures | report.md, methods.md, terminal source/data receipts and final_evidence/evidence.json | Source revision, hashes, starts, bounds, failures and runtime preserved |
| Preserve completed study and raw files | git diff b4f7ef7 -- data skunk_works/delay_study/residual_study | Verified empty at final completion check |

Result-directory names in the table are relative to results/; explicit results/
paths start from this folder. The live manifest owns exact
producer/auditor commits and terminal states. Frozen runs use the scientific
runtime specified in ../environment.json, one BLAS thread each, and no more
than four simultaneous long numerical processes.

## Numerical corrections and retained limitations

The initial baseline calculation reached its numerical checks but failed while
attempting optional sandboxed process inspection. baseline_v2 removed that
embedded inspection; the approved controller reconciles processes. Both records
are retained.

The first harmonic diagnostic projected each input frequency separately on
partial-cycle windows. reference_diagnostics_v2 uses a joint projection verified
by independent quadrature. Earlier trajectory and forecast outputs were
unchanged. Frequency displays were also corrected to show independent phase
unwrapping; measured complex responses were unchanged.

The nonlinear optimizer needed stable differences near active bounds. The
selected quadratic fit passed the independent ODE/step and stable stationarity
checks. Some non-selected starts in the original frozen job have poor raw
stationarity; retained records show this. Later fixed-beta runs use the corrected
boundary treatment. Converged multistarts provide search evidence, not proof of
a global minimum.

No untouched validation claims, physical delay calibration, absolute alpha
calibration, or nominal 95% coverage guarantee follow from this investigation.
No remaining calibration-pair enumeration, DoE, hardware run, full two-temperature
fit or email delivery belongs to this goal. The scoped investigation stops here for researcher review.

## Final delivery checks

The consolidated source-bound check passed for all sixteen main fits. It binds
each fit to an independent selected-fit audit, reconstructs raw RMSE from saved
observations/predictions, checks counts and nominal AICc, verifies all sixteen
report table rows and the headline improvement/range statements, and records
the report SHA256. Fifteen regular profile curves and the seven-point nonlinear
beta profile are preserved. All reported profile points have independent mean
and likelihood checks.

The final report links resolve locally. The report figures, including the final
profile and innovation summaries, were visually inspected. The no-ai-slop
writing review passed: concrete claims, explicit conditional uncertainty,
source links, no inflated mechanism claims or recap ending. Raw data and the
completed residual_study have no diff from b4f7ef7. Execution and delivery state
are recorded in run_manifest.json; numerical success does not resolve the open
scientific questions in report.md.
