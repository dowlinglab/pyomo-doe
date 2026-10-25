# Physical origin of the apparent extra lag

Alex authorized this follow-up after reviewing the unequal-lag explanation.
Use both measured temperature channels from the same twelve ENV2 records.
Preserve all completed studies and raw files. No new email, hardware or DoE.
Apply the existing no-ai-slop and claude-for-researchers practices.

## Question and bounded comparison

Can coupled heater/sensor energy balances explain the extra response mode?
Start with symmetric heaters and sensors, positive capacities/conductances,
one common ambient trajectory per run and Q2=0. Test the base four-state model,
then one shared first-order command lag if the fast phase is unexplained.
Compare fixed linear command conversion with the existing bounded command map.
Use constant ambient first and affine endpoints as a declared sensitivity.
Do not assume that estimated modes correspond to individual physical parts.

The equations are CH*dHi=Ua*(A-Hi)+Us*(Si-Hi)+Ux*(Hj-Hi)+beta*Qi,
CS*dSi=Us*(Hi-Si). Fit K=beta/Ua, tA=CH/Ua, rS=CS/Us,
r=Us/Ua and q=Ux/Ua. This removes the common absolute calibration scale.
Q2 is zero, so its heater conversion cannot be estimated. A second powered-off
heater can still participate thermally. Initial heater temperatures equal their
respective first sensor readings as an explicit assumption. First observations
are conditioned on and omitted from the residual vector. Inputs use left holds.

With fixed ambient, common/differential sensor transfers are
G+=K/[tA*rS*s^2+(tA+rS*(1+r))*s+1] and
G-=K/[tA*rS*s^2+(tA+rS*(1+r+2*q))*s+1+2*q].
Thus T1/Q1=(G++G-)/2 and T2/Q1=(G+-G-)/2. Both modal transfers
have negative imaginary part at positive frequency. The base symmetric model
therefore cannot produce T1 phase outside(-180,0) degrees, even with four poles.
Verify this algebra independently before interpreting fits. Extra poles alone
do not imply the all-pole cascade found in the previous study.

## Execution and verification

Freeze committed code/data with freeze.py and the pinned runtime in ../environment.json.
Keep at most four long numerical processes with one BLAS thread each. Inspect
actual process/terminal records before retries. Save multistarts and failures.
Audit exact propagation against direct energy-balance integration, analytic
common/differential transfers, dense covariance likelihoods and saved raw hashes.
Fit both outputs jointly. Begin with IID pilots, then independently colored
OU-plus-measurement errors with separate channel scales; cross-channel error
independence is a working assumption to diagnose. No likelihood comparison
against a T1-only fit uses a different observation count as if it were matched.

Report both-channel residuals, fast frequency mismatch, pole locations/residues,
common/differential response, local identifiability and the physical hypotheses
the data support or refute. Nominal within-model uncertainty is conditional on
symmetry, initialization, ambient, command timing and measurement assumptions.
All data have informed model selection; no untouched validation claim follows.

## Sources

The existing analytic four-state scope check is ../review/check_four_state_scope.py.
The two-state balance is ../models.py:simulate_original. Primary teaching sources
consulted October25,2026: the Notre Dame TC-Lab model-identification notes and
https://apmonitor.com/pdc/index.php/Main/TCLabStateSpace . The latter also
illustrates a one-way sensor-lag approximation; this study retains the sensor
heat capacity and reciprocal heater/sensor heat flow in the balance above.
The Lab Literature search did not return within the controller's bounded wait;
no result from that unreturned query is used as evidence.

The reciprocal four-state balance is also given in
https://ndcbe.github.io/controls/tclab/03.06-Four-State-Model.html . Its labels
for interheater and sensor conductances differ from some other TC-Lab notes;
Us and Ux above refer to heat paths, not those inconsistent letter assignments.

## Interpretation to preserve

Unequal time constants can arise from the eigenmodes of different heat capacities
and conductances. Coupling is physical, but a cascade with three positive poles
is not automatically equivalent to this four-state network. The network also
has zeros. For B=tA*rS, a=tA+rS*(1+r), and q=Ux/Ua, its numerators are
K*[B*s^2+(a+q*rS)*s+1+q] for T1 and K*q*(rS*s+1) for T2, over D+*D-.
Consequently T2/T1=q*(rS*s+1)/[B*s^2+(a+q*rS)*s+1+q]. This ratio cancels
a shared input lag and K. It would help separate sensor and command dynamics
with sufficiently informative two-channel data. Here the fast T2 oscillation
must first be checked against quantization and harmonic-estimation uncertainty.

Structural recoverability of scale-free parameters follows in the ideal
common/differential chart. Write G+=kp/(bp*s^2+ap*s+1) and
G-=km/(bm*s^2+am*s+1). Then K=kp, l=kp/km=1+2*q,
rS=(l*am-ap)/(l-1), tA=bp/rS and r=(ap-tA)/rS-1.
This needs nonzero interheater coupling and resolvable modal responses; it is
not a practical-estimability guarantee for the finite records.

At negligible reciprocal sensor heat loading (r approaching zero), the command
response contains the common factors 1/(1+rS*s) and 1/(1+L*s). These commute:
interchanging the sensor and command time constants leaves both command-to-
temperature transfers unchanged in that limit. Ambient/startup responses can
break that equivalence under additional assumptions. Even a supported pair of
unequal lag times therefore does not automatically identify which physical
component supplies each time constant.

All new results belong in this folder. The earlier completed research goal and
paused heartbeat stay unchanged. This follow-up does not authorize another
email or a new hardware experiment. Commit source before measured runs, then
commit selected evidence and the report, and push the branch when complete.
