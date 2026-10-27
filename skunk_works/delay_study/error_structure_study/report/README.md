# Remaining-error report

`main.pdf` is the checked tablet report; `../report.md` is its Markdown companion.
The complete goal closes at the scientific review checkpoint, with residual
adequacy and uncertainty calibration explicitly unresolved. No email is part
of this goal. The frozen eight-page intermediate draft and its original audits
are retained under `wip_oct26/`.

The final transfer tables come from `../results/transfer_final_summary` through
`render_transfer_tables.py`; all twenty preferred components and their
curvature/draw/forecast chains are bound. Positive numerical information and
correct mixture arithmetic do not validate the local-normal law. The reported
Taylor check rejects that approximation away from its reference.

`check_final_numbers.py` runs with the pinned scientific Python and independently
pools saved observation/forecast arrays for RMSE, coverage, width, endpoint
changes and reference stress. It checks profile arithmetic and saved numerical
point audits, and binds the recovery/initial-state source chains. The helper
checks distinguish numerical arithmetic from method appropriateness.
`final_numeric_audit.json` pins both manuscript sources and every input.

Compile with pdflatex from this directory, checking its actual exit before
reading the result. `check_layout.py` uses the bundled PDF runtime and checks
letter dimensions, body extent and every annotation rule. The requested
0.25-inch left and 3.25-inch right margins leave a five-inch body and35 ruled
lines at9/32-inch pitch in the right annotation column. `visual_review.json`
binds the final source, PDF and inspected Poppler page renders. An edit
invalidates the corresponding audit until it is rerun.

The report's source/array checks do not independently recompute every mixture
component history or calibrate model-selection-adjusted uncertainty. Detailed
numerical failures and scientific limits are in `../completion_audit.md` and
`../report_audit.md`. Earlier study reports and email receipts remain unchanged.
