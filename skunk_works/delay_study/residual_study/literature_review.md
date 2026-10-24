# Method sources for the n=1 residual teaching study

This is a focused check of the methods used in the Tier 0-2 report. The broader
October 21 review remains in `../manuscript/sections/literature_comparison.tex`.
This update does not repeat the experiment-design review or claim a systematic
literature search. Scientific interpretations are agent preparation for Alex's
review.

| Source and inspected pages | What it supports | Limit in this study |
|---|---|---|
| [Hartikainen and Särkkä (2010)](https://users.aalto.fi/~ssarkka/pub/gp-ts-kfrts.pdf), physical PDF pp. 3 and 5; equations visually checked | Exact finite state representations of half-integer Matérn covariance kernels; stationary initialization and transitions for irregular sampling. These motivate efficient covariance/innovation calculations. | Computational equivalence does not establish which kernel describes TC-Lab errors. Thermal lag states and stochastic error states have different roles. |
| [Newey and West (1987)](https://doi.org/10.2307/1913610), physical pp. 2-3, printed pp. 704-705; publisher scan and native OCR checked | Bartlett smoothing of score autocovariances and the regularity qualifications behind HAC consistency. | Our 10/30/60-sample calculations are finite bandwidth sensitivities. Remaining command dependence challenges the mean/error assumptions; HAC does not change the OLS mean fit. |
| [Raue et al. (2009)](https://doi.org/10.1093/bioinformatics/btp358), physical p. 4, printed p. 1926 | Profiles reoptimize nuisance parameters and can reveal extended support missed by quadratic likelihood curvature. | The source does not justify regular confidence coverage for our bounded, potentially multimodal and misspecified likelihood. Our cutoff remains nominal. |

## Search and source record

The initial metadata snapshot was
`2530e8c89d532b5f5772c9b39b22fab6b615caa9331a6ac7c71de5cd211efce2`,
with 2947 indexed metadata candidates and no reported metadata issues. Exact DOI
lookup found the profile paper. The temporal-GP author manuscript and the HAC
publisher scan were not indexed at the corresponding initial lookups. Both were
reviewed and ingested through native managed staging; their publication receipts
record local byte readback. Remote Drive delivery was not independently checked.

The GP manuscript used the configured pdftotext reader. The scanned HAC paper
required an explicit Tesseract profile; its equations were checked in the rendered
scan. The native stored pages, rather than fresh extraction offsets, anchor the
final claim passages. Metadata, source availability, quote fidelity, claim support
and human scientific review remain separate checks.

The report has three citation occurrences and three method claims. Its exact
flattened source is `report/flattened.tex`. Packet identity, current revision,
source pages, fine-passage IDs and export receipt belong in the literature
checkpoint and receipt files. An initial grounding attempt encountered a changing
shared record after one passage was written. Inspection and reconciliation
recovered its missing usage/attachment before proceeding; the failed attempt is
preserved.

## Reading path and open question

Start with the report's two-state thermal model and alpha/gain scaling. Then read
the GP paper's covariance representation and initialization, the HAC paper's
assumptions, and the profile paper's nuisance reoptimization. Compare those
assumptions with the fitted innovations before interpreting a covariance matrix
or a percentile range.

The fixed-length, noncircular innovation blocks are a declared sensitivity. They
retain short error sequences while breaking alignment with the command; no
coverage theorem for this custom resampling scheme is claimed. The remaining
question is whether a revised mean/disturbance description can explain the fast
sine and transfer across runs before the remaining 21 pairs are examined.
