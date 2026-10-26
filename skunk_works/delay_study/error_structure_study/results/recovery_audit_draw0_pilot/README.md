# Generated-data audit pilot

This is a numerical/data-accounting check of draw 0, not a coverage estimate.
The fourteen terminal original-shard draws remain available, but thirteen were
explicitly not selected for this pilot. The JSON lists their indices.

Both selected free/true-gain fits passed independent physical propagation,
Gaussian bin rectangles, 128/256/512-node, support and stationarity checks.
Regeneration from the declared seed exactly reproduces the saved observations;
the saved gain likelihood ratio agrees with the two independently checked fits.

In this frozen controller version, the fields ending `among_available` and
`failed_or_unaudited_available_draws` use the **selected audit denominator of
one**, despite their names. The following source commit renames them to refer
to selected draws. The original JSON is retained unchanged. This label repair
does not change either point audit, any fitted draw, or the fifty-draw design.
Final aggregation may reuse the two source-bound passed numerical audits.
