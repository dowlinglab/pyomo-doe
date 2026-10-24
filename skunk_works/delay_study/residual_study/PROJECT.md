# Current n=1 residual teaching study

Goal active from 2026-10-24 16:39:24 UTC. Agreed scope: `../review/approach.md`.
Implementation: `execution_plan.md`; records/pairs: `config.json`.
This phase leaves the completed October 21 study and deliveries unchanged.

| Phase | State | Next gate |
|---|---|---|
| Data/input and exact mean audit | Forward check passed in checkout | Repeat in frozen run; audit raw records |
| Tier 0 mean/error exploration | Pending | Diagnose and freeze working structure |
| Tier 1 P1 uncertainty/evaluation | Pending | Full restricted refits and checks |
| Tier 2 P2-P6 | Pending | Five further pairs; no full enumeration |
| Report/Gmail/push | Pending | Checked ruled-margin PDF; one verified email |

No TC-Lab numerical process was live at startup reconciliation. Read the current
manifest before launching work. Reuse the old hourly heartbeat for this goal.
No arbitrary waiting is required: advance research until deliverables finish.
Recheck pole equivalence with ambient forcing; do not impose time-scale ordering.
Evaluation is retrospective after Tier 0. Ambient bounds are scenarios, not a
probability distribution. Technical defaults are identified in the plan.

The acquisition notebook at b114b13f updates lab.Q1 immediately before
p.update(t), supporting left-held inputs. This is supporting source code, not
a per-file execution log. `audit_mean.py` independently compares propagation
against augmented matrix exponentials and DOP853, including irregular grids,
repeated poles, endpoint ambient drift and both hold conventions. All 36
checkout checks passed (maximum matrix discrepancy 1.4e-11 C); the measured
baseline reruns this check from committed sources.
