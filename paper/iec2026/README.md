# IEC 2026 paper preparation

Updated 2 November 2026 for **Machine Learning for Cyclone Prediction in the Bay of Bengal**.

The frozen 2018-2025 evaluation is complete. This results manuscript supersedes the held-results draft, preserved in qa/before_completed_results/. A fresh saved-output audit passed 110 checks. No training, inference, calibration refit or original-test decoding was performed during this manuscript update.

## Deliverables

- drafts/IEC2026_Extended_Abstract.docx: initial submission with background, methods, numerical key results and conclusions; no figures or tables.
- drafts/IEC2026_Paper_Draft.docx: editable full paper with three tables, two figures, equations, references and declarations.
- drafts/IEC2026_Figures_and_Tables_Updated.docx: current three-page companion with both figures, all three editable tables, a technical model overview and explanatory notes. Generated from the current paper by build_figures_tables.py. The updated copy preserves the original document while it is open and locked.
- drafts/IEC2026_Springer_Draft.pdf: compiled Springer journal-format reading copy.
- springer/main.tex: editable native LaTeX using the unmodified official Springer Nature sn-jnl class, native equations and numbered journal citations.
- manuscript_content.json: shared prose source; edit this and run build_draft.py with the workspace Python runtime to regenerate Word, Markdown, LaTeX and figures.
- paper_draft.md: plain-text manuscript.
- supplement_v1/: Supplement S1 **v1.1**, containing the feature contract, population counts, methods and historical source snapshots, plus complete final metrics, reviewed plan, selection/calibration/ledger receipts and numerical audit in results/.
- revision_response.md: review changes and the completed-results update.
- submission_notes.md and submission_email_draft.md: preparation notes and an unsent email template.
- sources/IEI_Guidelines_for_Authors.pdf and sources/IEI_Declaration_Form.doc: congress-linked guidance and original blank declaration; no signatures or author approvals are asserted.
- IEC2026_Draft_Package.zip: current manuscripts, LaTeX and supplement with a verified internal file manifest.

## Evidence and scope

The selected model is the Bay-only snapshot MLP with absolute wind loss and seeds 17, 29 and 43. GRU and transfer models remain development comparisons. The final test contains 1018 observations from 65 cyclones. Paired errors improve by 11.87% for track, 12.41% for wind and 14.74% for pressure. All-observed-label counts remain distinct from persistence-eligible comparisons.

Uncertainty results are mixed: intensity proper scores improve over the constant training-RMS Gaussian, spatial gains over that reference are unclear, wind CRPS worsens after calibration and central regions overcover. Operational warning skill, avoided losses and energy savings have not been established. The final holdout is consumed and must not guide further tuning.

The saved-output audit verifies 91 final and 71 preflight manifested files, unchanged model/calibration bytes, and consistent execution receipts. It reproduces point errors, paired bootstrap intervals, Gaussian likelihoods, marginal CRPS and coverage from saved predictions/cache. Energy-score Monte Carlo calculations were not independently repeated. Four historical development archives and their 376 manifested entries were also rechecked without executing model binaries.

The full local research archive is dist/bob_cyclone_research_archive_v1.zip in the project root, separate from the paper ZIP. It contains original exports, frozen weights and source. It is not a new public release and has no archival DOI. The manuscript builder expects original development exports in Downloads and audited numerical evidence in S1; it does not regenerate scientific model results.

## Format and limits

Selected congress papers may be considered for the Journal of The Institution of Engineers (India) after peer review; no journal series is assigned. Series B is a provisional reference because it covers Computer Engineering. This is not an assumed LNCS proceedings volume.

The IEC-linked generic guidance specifies at most 3000 words, A4 single column, double spacing including references, a synopsis of at most 150 words and at most five illustrations. Word follows sequential superscript citations and a caption list; the Springer source retains numbered journal citations. The paper contains three tables and two figures. The extended abstract has no illustrations and remains below 1000 words.

draft_checks.json records all body/table text, references, declarations, repeated captions and figure labels. Decimal numbers count as one word. Formula symbols and headers/footers are excluded; this is not a publisher-certified count. The synopsis is exactly 150 words under whitespace and decimal-aware counting. Word equations are high-resolution mathematical images; editable equations are native in LaTeX. Native knots and hPa have SI equivalents introduced in the methods.

## Authors and unresolved fields

Author order: Md. Faisal Tabrez; K. Akhilesh; Dr. T. V. Rajinikanth. Affiliation: Department of Computer Science and Engineering, Mahatma Gandhi Institute of Technology, Hyderabad, Telangana, India. Faisal is corresponding author in this draft, faisaltabrez01@gmail.com. The supplied second email is preserved as akhlieshkancharla05@gmail.com. No third-author email was supplied. The title Dr. is included for Dr. T. V. Rajinikanth at the author's request.

The user confirmed no funding and no competing interests on 2 November 2026. Individual contributions and final author approval remain pending. Original extraction notebooks, acquisition timestamps, interpolation/daily SST timing and some derived-feature definitions remain unresolved. Code licensing and a public archival deposit are outstanding. No assumption fills these gaps.

## Rendering and guidance

Word uses Microsoft Word PDF export when the packaged renderer cannot find bundled LibreOffice, followed by page-image inspection. Springer compilation uses bundled Tectonic. qa/ contains intermediates and verification records, excluded from the package.

- IEC call: https://41iec.org/papers/
- Generic guidance: https://repos.ieindia.org/technical/publication/document/Guidelines_for_Authors.pdf
- Declaration: https://repos.ieindia.org/technical/publication/document/IEI_DECLARATION_FORM.doc
- Provisional Series B: https://link.springer.com/journal/40031/submission-guidelines
- Official template: https://www.springernature.com/gp/authors/campaigns/latex-author-support

No document has been submitted or emailed to the organisers.

## Revision after supplied similarity and AI reports

The 2 November revision preserves the original checked manuscript and reports under qa/reports_2026-11-02/baseline/. Preparation records are in revision_2026-11-02/: the 16-source visible-match audit, related-work evidence/access matrix, claim-to-metric mapping, author confirmations and an unsent institutional recheck request. The original 13% similarity and 50% AI scores describe the earlier manuscript only; no new score is available.

The main paper now cites the relevant Faiaz, Haque and Tsai 2026 studies, limited to facts established from accessible primary records and abstracts. Full-text methodological comparison remains limited by access. The introduction, results interpretation, discussion and abstracts were revised against the saved experiment records. AI-assisted drafting and code-based preparation are explicitly disclosed, including figure captions. This is still an assisted manuscript; final scientific review and approval by the authors are not asserted complete.

The journal-format reading copy, Word manuscript, extended abstract and companion are generated consistently. No model, calibration, data split or saved scientific result was changed. Only an authorised institutional checker can produce the fresh similarity/AI reports and apply the relevant institutional policy. No upload or email has been performed.
