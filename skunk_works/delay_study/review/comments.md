# Comments on the TC-Lab report

Transcribed from the annotated PDF supplied on October 24, 2026. This file
preserves the reviewer's comments; responses belong in `analysis.md`.

- Source: `1-TC_Lab_delay_study_draft0_updated.pdf`, supplied attachment.
- SHA-256: `447acfc69f5ce519bba785ee4a65e8bfe6f4411119177c7517b6e3886c270a6b`.
- The PDF has 55 pages: the 53-page report and two handwritten continuation pages.
- Page references below use physical PDF pages, which match printed report pages
  through page 53.
- No PDF annotation objects are present. Handwriting and marks are flattened
  into page content. All 55 pages were surveyed visually; comment-bearing pages
  were inspected at higher resolution.
- Line breaks have been joined and mathematical symbols typeset. Uncertain
  wording is identified explicitly. Text in quotation blocks is transcription;
  the separate context lines describe the marked manuscript passage.

## Five big open questions

### Q1. Original model estimability (page 53)

> Is original 4-state model not estimable and why?

Reviewer's accompanying assessment, in red:

> Yes. This document explains the reason, although the presentation is not clear
> enough for teaching yet.

Transcription note: the handwriting says “4-state.” The model terminology is
addressed separately in the analysis rather than silently changed here.

### Q2. Appropriate delay model (page 53)

> What is the appropriate delay model?

Reviewer's accompanying assessment, in red:

> Results so far show that n = 1 is reasonable. We do not have enough evidences
> yet to strongly argue for n > 1.

### Q3. Error model and time-series assumptions (page 54)

> What is the appropriate error model? Are the time series error assumptions
> satisfied?

Reviewer's accompanying assessment, in red:

> This would benefit from additional investigation and analysis.

### Q4. Experimental-data provenance (page 54)

> What is the provenance of the experimental data? Was it added to git by the
> same person around the same time?

### Q5. Ambient and heater-calibration uncertainty (pages 54–55)

> How to incorporate inferring errors in T_amb and α in the data? α was
> estimated for one device and assumed to be the same for all devices.
> Likewise, T_amb likely slowly varies with time. If experiments were a few
> hours apart, T_amb likely changed.

Transcription note: the phrase “inferring errors” is the closest visual reading;
the question clearly concerns uncertainty in ambient temperature and α.

## Comments in report order

### C01. Define the abstract's identifiability language (page 1)

> ??? Please explain/define.

Context: the highlighted exact one-dimensional equivalence family and the
underlined three-parameter representation/global pole-label ambiguity. The
arrow points into the latter sentence; both phrases need a teaching explanation.

### C02. Closed-loop Pyomo.DoE (page 1)

> What about running Pyomo.DoE in a closed loop to identify model? Nice test
> case for “self-driving lab” or digital twin.

### C03. Separate sources of residual structure (page 2)

> How to disentangle? Serial structure looks like a nice next step.

Context: underlined “serial structure and between-experiment discrepancies.”

### C04. Manchester's convex-relaxation approach (page 2)

> What did they do? Can we use their ideas here?

Context: Manchester (2010), nonconvex time-domain information design and a
convex relaxation. “Explicit transfer functions” is also highlighted on this page.

### C05. Initial thermal equilibrium (page 3)

> Reasonable to assume TC-Lab was thermally equilibrated at T_amb and u = 0
> history at startup.

Context: initial-state assumptions and absent pre-window heater history.

### C06. Zero input history (page 3)

> Okay to set u = 0 for history.

Context: the same initial-state discussion. The pure-delay expression is also
underlined. These comments state a proposed physical assumption, rather than
documenting that prehistory was measured.

### C07. Constant input hold (page 4)

> Constant hold is likely more appropriate.

Context: linear interpolation of recorded heater commands is bracketed. The
repeated-pole limiting-case paragraph near the top is also underlined.

### C08. Why ambient/history sensitivity matters (page 5)

> Why is this important?

Context: the ambient/history perturbation and held-out-transfer discussion.

### C09. Correlated errors in the local information calculation (page 5)

> What about with time-correlated data?

Transcription note: the margin handwriting is compressed; “time-correlated
data” is a normalized reading, not a claim of exact character-for-character OCR.
Context: the IID information matrix and local uncertainty subsection.

### C10. Explain HAC and its assumptions (page 5)

> I want more background on this. I am unfamiliar with these methods.

Context: heteroskedasticity and autocorrelation consistent (HAC) sandwich
estimation, Bartlett weights and dependence assumptions.

### C11. Diagnostic checks (page 6)

> Need to check assumptions via diagnostic plots.

Context: the working correlated-residual likelihood and profile/resampling
discussion.

### C12. Explain the shutdown check (page 8)

> ???

Context: “independent shutdown-tail check” is underlined.

### C13. Random step experiments and cooldown (page 12)

> What about random step tests? Would that be a helpful next experiment?

The sketch shows a sequence of varying command levels followed by zero command.
The two parts are labeled “active experiment” and “cool down.”

### C14. Different devices or ambients behind validation error (page 13)

> I wonder if this is because all of the data files correspond to different
> devices or different T_amb.

Context: Figure 3's large, file-dependent RMSE and signed mean error.

### C15. Define filtered residuals (page 27)

> What is meant by “filtered” in this context?

Context: Figure 13's OU-filtered innovation autocorrelations.

### C16. Define command reconstruction (page 30)

> What is meant by this?

Context: circled “Different command reconstructions” in Figure 15's caption.

### C17. Define model separation (page 36)

> What is the D²? Is this defined in the methods?

Context: Figure 19's “Separation D²” ordinate.

### C18. Explain regression versus physical reconstruction (pages 39–40)

> Is this saying for any choice r, a, b, and c can be regressed and then the
> parameters estimated?

Transcription note: the last coefficient appears to be “c”; the manuscript
itself uses (K, a, b). The analysis distinguishes the two readings.
Context: the bracketed physical-parameter equivalence family, Eq. (9).

### C19. Explain the sandwich calculation (page 42)

> How to dig into this?

Context: Appendix A.8, Bartlett weighting and its consistency assumptions.

### C20. Include calibration as an inferred quantity (page 42)

> How to incorporate this in analysis as a parameter?

Transcription note: normalized from slanted margin handwriting.
Context: Appendix B.1, uncertain heater conversion c = αP and propagated
physical-parameter covariance.

### C21. Choose an error model (page 44)

> What is an appropriate error model?

Context: bracket around Appendix B.5, serial dependence and least-squares bias;
Appendix B.6 discusses an OU filter anticorrelating an IID component.

### C22. Original and reduced predictions agree (page 49)

> Very interesting!

Context: Table 16's matching original/n = 1 calibration and held-out RMSE values
are boxed.

### C23. Correct error model for these data (pages 49–50)

> What is the correct error model for these data?

Context: bracket around Section G.2's residual-correlation uncertainty comparison,
including the continuation on page 50.

## Other marks

Highlights and underlines without an accompanying standalone sentence are
recorded with their neighboring comments above. No scientific conclusion is
inferred from pen color, a highlight alone or an isolated bracket. No handwritten
comment was found on pages 7, 9–11, 14–26, 28–29, 31–35, 37–38, 41, 43,
45–48 or 51–52 during the full-page survey.
