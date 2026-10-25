# Investigation verification and scope

Status: final mapped-stage profiles and consolidated evidence checks pending.
The scientific requirements come from [the committed plan](../review/tier0_followup_investigation.md)
at b4f7ef7. A numerical audit verifies an implementation; it does not establish
that a mean/error model adequately describes the experimental system.

| Requirement | Evidence | Interpretation or limit |
|---|---|---|
| Reproduce pooled reference | results/baseline_v2/audit.json | Independent propagation and dense likelihood; original failed receipt retained in baseline |
| Verify holds, irregular intervals and n=1/2/3 propagation | Forward audits saved with main fits; dynamics.py synthetic ODE comparison | Left holds and specified initial states; acquisition conventions remain assumptions |
| Separate offset, trend and cycle | results/frequency/error_decomposition.csv and results/frequency_figures | Sequential descriptive decomposition; no causal ambient/noise allocation |
| Repeatable amplitude/phase and uncertainty | results/frequency; results/frequency_audit/audit.json | All54 estimates reconstructed using independent quadrature/QR; HAC lag and startup sensitivities retained |
| Flexible shared linear benchmark | flexible2/3/4, comparison_primary and information_final | Stable real-pole family with numerator freedom; not unrestricted LTI or subspace estimation |
| Matched delay-order comparisons | chain1/2/3, secondary_matrix, mapped_chain1/2/3 | Joint reestimated covariance, fixed-shape/IID sensitivities and revised mean comparison |
| Independent selected means, innovations and likelihood | primary_audits, flexible3_audit, flexible4_audit, targeted_audits, transport_audit, power_linear_audit, mapped_audits12 and mapped_audit3 | Raw hashes and dense covariance checks; mapped unequal audit pending |
| Estimability and response uncertainty | information, information_final, information_quadratic, information_mapped | Scaled SVD/eigenvalues, weak directions, pole permutation; local conditional approximations |
| Profile support | chain_profiles_audit, transport_profiles_audit, power_profiles_audit, beta_profiles; mapped profiles pending | Every reported profile point independently checked; scoped grids do not certify global optima or coverage |
| Impulse-response shape | fir_sensitivity and fir_audit/audit.json | All16 cells; independent command-step convolution and dense weighting; fixed reference covariance |
| Targeted nonlinear/ambient checks | quadratic, quadratic_audit; power_curve, power_curve_linear and selected audits | ODE/convergence checks for nonlinear mean; candidate mechanisms remain confounded |
| Prediction versus simulation | reference_diagnostics_v2, diagnostics_power_linear, diagnostics_mapped2, diagnostics_quadratic | Synthetic dense conditional mean/variance audit; observations only through each origin; fitted parameters remain retrospective |
| Innovation adequacy | innovation_assessment pending | Per-run autocorrelation, input association, scale and time-block checks are descriptive, not calibrated joint tests |
| Reproducible report and reviewed figures | report.md, methods.md, terminal source/data receipts; final evidence pending | Source revision, hashes, starts, bounds, failures and runtime preserved |
| Preserve completed study and raw files | git diff b4f7ef7 -- data skunk_works/delay_study/residual_study | Verified empty before final synthesis; repeat before completion |

Paths in the table are relative to results/. The live manifest owns exact
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
calibration, or nominal95% coverage guarantee follow from this investigation.
No remaining calibration-pair enumeration, DoE, hardware run, full two-temperature
fit or email delivery belongs to this goal. The report will stop for researcher
review after the scoped comparisons and their checks are complete.
