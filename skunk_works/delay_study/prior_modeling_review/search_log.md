# Search, screening and verification record

Date: 2026-10-25. Mode: autonomous preparation, agent-authored, pending Alex's
scientific review. Scope: prior TC-Lab model formulations that could explain
remaining input-linked errors. First pass is physical TC-Lab models; second pass
is stochastic/data-driven identification that challenges a purely physical
explanation. No hardware work, new fit, package install or corpus mutation.

## Searches and selection

Web queries included the following, with backward/forward source links followed:

```text
"temperature control lab" "model" "radiation" paper
"TCLab" "identification" paper model nonlinear
"temperature control laboratory" model sensor identification radiation
"TCLab" "model" "Park" Hedengren paper
"temperature control lab" "parameter estimation"
"TCLab" "radiation" "identification"
"Benchmark Temperature Microcontroller" Park 2020 pdf
"Kuntz" "Rawlings" "TCLab" "identification"
"TCLab" nonlinear identification radiation parameter uncertainty
"TCLab" "supply" "Kuntz"
"TCLab" "grey box" identification
"TCLab" "Gaussian process" identification
"TCLab" "temperature-dependent"
```

Included sources were original author papers, author-hosted manuscripts,
official educational code/docs or institutional theses with relevant modeling
content. Educational examples are not counted as independent validation studies.
The review matrix has ten entries: three journal-paper families (P1/P2/P3), one
inspected preprint (P4), and six educational/source families (D1–D6). Some entries
contain related official pages. Four additional leads have explicit partial
coverage and are not treated as full-text studies. Kuntz's dissertation and P2
overlap, and are not counted as independent replication.

Excluded from evidential synthesis: Scribd reposts when author/institutional
versions were available; acronym collisions with other thermal laboratories;
unreviewed student GitHub projects; control-only search hits without inspected
model detail; search snippets without verified source context. A fractional-PID
conference lead was not developed into a modeling claim. No systematic database
screening, bibliometric completeness or literature-exhaustion claim is made.

## Verification pass

- Park preprint: model families and disturbance discussion checked in physical
  PDF pp. 5–9; equations/parameter table pp. 8–9; Hammerstein data conditions
  pp. 13–15; separate training/validation and error definition pp. 15–16.
  Table 2 figures in the review were rechecked against the source extraction.
  Its metric is average absolute error, not RMSE. Its displayed interheater
  radiation area A differs from As in some official educational code; retained
  as a modeling discrepancy rather than silently fixed.
- Kuntz–Rawlings extended report v3: read model assumptions, TCLab §6.1,
  Table 1 and discussion of Figs. 2–8. Its noise-free training response is
  explicitly distinguished from testing. Iteration-limit/filter instability
  qualifications retained. Published DOI/version read from PDF title page.
- Nguyen–Tique v1: checked TCLab §V-B data collection and sampling conditions;
  did not misattribute the HVAC table to TCLab. No comparable TCLab RMSE or
  uncertainty result claimed.
- Di Teodoro et al.: §5.2 confirms the fan and explicitly software-added 200 s
  delay. The text has a duplicated MSE label; no comparative MSE copied.
- Official code: D3 specifies As for interheater radiation and Kelvin for fourth
  powers. C uses two direct temperature states; G adds hidden heater states and
  separate sensor lags. D1’s displayed state matrix has capacity-symbol
  inconsistencies, so the differential balances are the source of equations. D4 explicitly describes temperature-dependent
  delivered power. No example coefficient imported into ENV2 as a measured value.
- New 2026 SINDy/WSINDy source: abstract-only; no full-text numerical claims used.

## Access failures and recovery

- A Lab Literature `search_bibliography(title="TCLab", limit=25)` call returned
  `Metadata generation retention capacity exhausted; no silent eviction`.
  The result was recovered from the tool-state store after the yielded cell
  disappeared. It provides no evidence of whether TC-Lab papers are present.
  No retry, index rebuild, cache eviction, maintenance or fallback corpus scan
  was attempted. Runtime/config inspections succeeded. Only the relevant runtime identity and
  inspection status are retained; unrelated configuration paths, accounts and
  device identifiers are deliberately omitted from tracked receipts.
- Author Park PDF was readable through the web tool. Terminal curl first failed
  DNS in the sandbox. The required escalated retry then waited on approval and
  was aborted by the user. No direct downloaded bytes/hash are asserted and no
  alternate terminal mechanism was used to bypass that outcome.
- CBA full-text link opened an empty viewer; guessed/download endpoints were
  unavailable. Retain abstract-only status.
- Seville thesis PDF endpoint returned HTTP 403; repository metadata available.
- KTH PDF timed out twice through the web tool; identity remains unverified.
- arXiv Taylor NN v2 HTML failed; the inspected source is explicitly v1.

No new evidence packet, ingestion, shared fine passage, citation grounding or
remote delivery was completed. A saved web review with transparent access gaps
is the deliverable; corpus-native source grounding remains unfinished.

## Open follow-up

Acquire the CBA full text before relying on its radiation identifiability claim;
inspect Kuntz's author experiment scripts before attributing a particular
disturbance mechanism or excitation protocol. Neither task blocks the documented
first comparison. The parent owns integration, decisions and the combined commit.
