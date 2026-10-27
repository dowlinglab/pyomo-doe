# Audited nuisance profiles

All four remaining profile grids completed on October27 at03:03:58 UTC.
All28 selected points passed independent dimensional trajectory, observation-bin
rectangle, quadrature/support and stationarity audits. Every point refits the
other permitted dynamics, ambient, run-conductance and covariance coordinates.
Two starts and a coarse grid do not establish global minima or confidence endpoints.

| Coordinate | Evaluated comparison | Twice the nll increase | Interpretation |
| --- | --- | ---: | --- |
| Sensor loading Us/Ua0 |0.005 to0.1|0.577| The loading bound does not imply a precise physical estimate. |
| Sensor loading Us/Ua0 |0.005 to0.3|2.316| Substantial loading remains weakly separated in this working likelihood. |
| Step run1 conductance ratio |0.9 to0.925|17.215| This run presses against the declared lower band; the hard band is not a population law. |
| T1 disturbance length |658.28 to1000s|1.884| Variance/length nuisance freedom matters for local uncertainty. |
| Radiation coefficient |0.11530 to0|7.619| Positive curvature helps the specified model; no calibrated radiation significance or emissivity estimate. |

The full radiation grid extends to4; its largest point reaches the gain bound
and both disturbance-length bounds. Its large deviance is conditional on those
original fitting constraints. Radiation refits convection and gain, so the
coefficient cannot be converted to a measured surface emissivity without an
independent scale and geometry. Remaining input/phase and cross-channel
failures still prevent a claim of adequate whitening.

The source-bound four-panel figure is in
`results/remaining_profiles_figure_v3`; it shows every evaluated point and uses
symlog radiation axes to retain both the low-coefficient valley and large values.
No chi-square cutoff is drawn. The separate consequential-startup profile is
still live; these results do not replace it.
