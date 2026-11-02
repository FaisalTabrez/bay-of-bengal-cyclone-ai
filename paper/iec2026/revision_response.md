# Response to the critical review

The first response below describes the earlier held-results revision. The completed-results update at the end supersedes those status statements while preserving the review history.

Revision date: 30 October 2026. The paper title and author order are unchanged. This revision changes documentation and manuscript framing, not model weights, calibration factors or the frozen evaluation protocol.

| Finding | Revision | Remaining limitation |
|---|---|---|
| R1 Research contribution | Introduction states a primary six-hour residual-skill question and a related uncertainty-score question. Added regional NIO intensity estimation and learned Gaussian track-uncertainty literature; differences in task, inputs and evaluation are explicit. | Empirical answers and the strength of the eventual contribution depend on held results. No first-application or new-architecture claim is made. |
| R2 Reproducibility | Added Supplement S1 v1 with all 39 features in order, units, missingness encoding, accessible population counts, equations, exact model/head/tree settings, role assignments and checksummed source/configuration snapshots. Added native LaTeX equations to the paper and mathematical images to Word. | Source acquisition times, original extraction notebooks, interpolation, daily SST support and some derived-feature definitions remain unresolved. Reproduction is scoped to the supplied derived package. |
| R3 Label populations | Main Table 1 now includes observed wind/pressure target counts. Supplement gives persistence-eligible counts and targets with missing current intensity by accessible partition. Added reporting-shift and representativeness limits. | No reconstruction of unobserved intensity or missing-at-random assumption; final-test label counts remain on hold. |
| R4 Probability references | Main Methods and Supplement describe the constant training-RMS diagonal Gaussian and uncalibrated learned Gaussian already declared in the implementation, with fixed shared means and paired proper-score comparisons. | These controls do not isolate every effect of correlation, input-dependent spread and calibration. No new fit or comparison was run. |
| R5 Word budget | Removed repeated preparation instructions and moved detailed methods to S1. Replaced the partial count with a count of all body paragraphs and table cells plus figure labels, including references, declarations and repeated captions. | Formula symbols and headers/footers are excluded; the count is not a publisher-certified total. Recheck after inserting results. |
| R6 Submission format | Synopsis revised to exactly 150 words under whitespace and regex counting. Extended abstract remains within 1000 words and explicitly describes ongoing work. | Key forecast results are still held. Final journal series and its precise submission format are not assigned. IEC Word citation style and Springer journal citation style remain intentionally distinct. |
| R7 Editorial focus | Replaced document-preparation instructions with study-method statements. Consolidated the results hold into a short reserved section, with brief abstract/conclusion qualifications. | Funding, conflicts, contributions and author approval still require human confirmation; no declarations are invented. |

The label audit read only `bob_train`, `nio_pretrain_train` and `bob_validation` from the immutable data package. It used the original validation-role CSV. It did not open final-test Parquet, execute a model, compute new forecast metrics, run outstanding scientific tests or refit calibration. Historical development results remain outside the manuscript and submission package.

The original review remains available as a review of the pre-revision snapshot; it has not been rewritten to imply a second independent endorsement of this revision.

## Completed-results update, 30 October 2026

The project subsequently completed the reviewed frozen evaluation. A fresh saved-output audit passed 110 checks, verifying 91 final and 71 preflight manifested files and reproducing analytical scores and point/paired intervals without fitting or inference. The earlier results hold is superseded only for this completed evidence.

- R1: The selected three-seed Bay MLP now answers the aggregate retrospective comparison question; no final ranking of every architecture or causal component benefit is asserted.
- R2: S1 v1.1 adds complete final metrics, reviewed plan, model-selection/calibration receipts, execution/ledger receipts and numerical audit. Historical source snapshots remain distinguished from actual frozen provenance.
- R3: Table 1 and S1 now include final observed intensity counts; paired and all-observed denominators are kept distinct.
- R4: Table 3 and Figure 2 report all three declared Gaussian forecasts and mixed outcomes, including worse wind CRPS after calibration and central overcoverage.
- R5: All prose, tables, captions, references, declarations and figure labels are recounted; decimals count as single words.
- R6: The 150-word synopsis and sub-1000-word extended abstract now include verified numerical results. The full draft retains three tables and two figures.
- R7: The held section is replaced by Results; discussion and conclusion state retrospective limits and the consumed-test boundary.

Funding, conflicts, author contributions, original source-construction gaps, code licensing and public archival deposit remain unresolved. No new author consent, public release, external peer endorsement or operational skill is asserted.

## Revision after similarity and AI writing reports, 2 November 2026

The original report scores are 13% similarity with bibliography exclusion Off, and 50% likely AI-written qualifying prose. They describe the preserved earlier manuscript. The revised manuscript has not been rechecked institutionally.

1. Preserved the exact baseline and original reports with checksums.
2. Classified all 16 visible similarity sources. Official terms, affiliations, bibliography titles and required provider wording remain accurate; no long copied Methods/Results block was identified in the supplied highlights.
3. Added three relevant 2026 studies and a primary-source access/comparison record. No unsupported full-text or cross-study performance ranking is claimed.
4. Revised the abstract, introduction, selected-model account, results interpretation, discussion and conclusion from saved methods/results. Selection reuse, paired populations and mixed uncertainty findings remain explicit.
5. Confirmed no funding and no competing interests from the user's direct reply. Individual contribution details and final author approval remain pending.
6. Expanded the AI declaration to describe drafting and code-based preparation, and identified code assistance in figure captions. The revision does not claim to eliminate AI assistance.
7. Recounted and rebuilt the deliverables. The audit maps scientific claims to frozen evidence and checks that scientific files are unchanged. A fresh institutional similarity/AI report and policy assessment remain outstanding; request text is prepared but unsent.

Detailed records are under revision_2026-11-02/. This response is a preparation record, not an external peer-review approval or a new detector result.
