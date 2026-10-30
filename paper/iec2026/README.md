# IEC 2026 paper preparation

Prepared 30 October 2026 for the initial submission to the 41st Indian Engineering Congress.

The author's latest instruction is controlling: **model results, missing analyses, required outstanding tests and performance conclusions remain on hold until the project work is complete**. No new model training, calibration or final-test evaluation was performed while preparing these documents. Previously verified development outputs are retained only in `working_evidence/` and are excluded from the submission package.

## Documents

- `drafts/IEC2026_Extended_Abstract.docx`: structured initial extended-abstract draft, with all three named authors, keywords, background, methods, current status and limitations. It contains no forecast-performance numbers and no figures or tables. Its evaluation-status paragraph must remain accurate when submitted.
- `drafts/IEC2026_Paper_Draft.docx`: editable full-paper scaffold following the IEC-linked author guidance, including a synopsis, numbered sections, sequential superscript citations and a separate caption list. Results are explicitly on hold.
- `drafts/IEC2026_Springer_Draft.pdf`: compiled Springer journal-format reading copy, after successful compilation.
- `springer/main.tex`: editable manuscript using the official Springer Nature `sn-jnl.cls`, with a workflow figure and bibliography. The official class and its licence notices are retained unmodified. `sn-mathphys-num` is used for numbered journal references; final series-specific instructions take precedence.
- `manuscript_content.json`: shared prose source used to generate the Word, Markdown and LaTeX drafts consistently. Edit this and run `build_draft.py` using the configured workspace runtime to update them. The builder also verifies development export checksums; it expects the original local archives in Downloads.
- `paper_draft.md`: plain-text review copy.
- `supplement_v1/`: Supplement S1 v1, with the ordered 39-feature contract, observed-label and persistence-eligible counts, explicit mathematical definitions, method/configuration snapshots, validation roles and provenance hashes. Unresolved source construction is marked rather than inferred.
- `revision_response.md`: how the review findings were addressed and which scientific/provenance items remain on hold.
- `sources/IEI_Guidelines_for_Authors.pdf`: author guidance linked by the IEC call for papers.
- `sources/IEI_Declaration_Form.doc`: original blank declaration supplied by IEC, preserved without signatures or assertions of author consent. It is a Word `.doc`, not a PDF.
- `submission_notes.md`: current requirements, author information, unresolved fields and held work.
- `submission_email_draft.md`: unsent email text for use only after a submission generates a Paper ID.

## Format and limits

The full-paper target is the Springer/IEI journal route, not an assumed LNCS proceedings volume. IEC states only that selected papers may be considered for a suitable Journal of The Institution of Engineers (India) series after peer review. Series B includes Computer Engineering and is a provisional formatting reference; the congress has not assigned this paper to a series.

The IEC-linked generic author PDF specifies a maximum of 3000 words, third-person English, A4 single-column text, double spacing including references, a synopsis of no more than 150 words and a maximum of five illustrations. It asks for sequential superscript references and a separate caption list. The Word paper follows that instruction; the LaTeX copy retains Springer's own numbered journal citation style. No journal acceptance or exact production layout is implied.

Current conservative counts are recorded in `draft_checks.json`. The revised synopsis is exactly 150 words under both whitespace and regex counting, meeting the linked IEC maximum and the provisional Series B minimum. The revised counter includes all Word body paragraphs and table cells, references, declarations, repeated captions and figure labels; formula symbols and headers/footers are excluded. It is not a publisher-certified count. Recheck the limit when held results are added. The figure contains a workflow only, with no performance data. Word equations are high-resolution mathematical images; the Springer source contains native LaTeX equations. Wind and pressure units are introduced in SI equivalents before the project's native knots and hPa units where numerical values occur.

## Author block

1. Md. Faisal Tabrez, B.Tech CSE, MGIT. Corresponding author in the draft: faisaltabrez01@gmail.com.
2. K. Akhilesh, B.Tech CSE, MGIT. Email provided: akhlieshkancharla05@gmail.com. The supplied spelling is preserved.
3. T. V. Rajinikanth, Professor, Department of CSE, MGIT. No email was supplied.

Institution expanded from the official MGIT website to Mahatma Gandhi Institute of Technology, Hyderabad, Telangana, India. Degrees and the honorific Dr. are kept in these preparation notes rather than placed before journal author names. Author order follows the user's instruction. Corresponding-author designation should be confirmed by the authors before upload.

## Verification

Scientific text was checked against the current implementation and data handoff. Four historical result archives were rechecked against their archive hashes and all 376 file-manifest entries; no model binary was executed. None of those model-performance values was inserted into the manuscript. The review revision counted observed labels in training and validation only, without producing forecast scores or fitting a model. No final-test Parquet was decoded. Literature references were checked against publisher, proceedings or provider sources; source-construction acquisition dates remain unresolved. The two added studies are Mawatwal and Das (2024) on NIO satellite intensity estimation and Fernandez et al. (2025) on probabilistic track-forecast errors, with their different tasks made explicit.

`qa/` holds rendering intermediates and a verification record. These intermediates are excluded from the submission package. Word rendering uses Microsoft Word PDF export because the packaged DOCX renderer cannot find LibreOffice on this Windows host, followed by page-image inspection. Springer compilation was checked with the desktop compiler and bundled Tectonic.

## Source links

- Live IEC call for papers: https://41iec.org/papers
- Linked IEI author guidance: https://repos.ieindia.org/technical/publication/document/Guidelines_for_Authors.pdf
- Linked declaration: https://repos.ieindia.org/technical/publication/document/IEI_DECLARATION_FORM.doc
- Provisional Series B instructions: https://link.springer.com/journal/40031/submission-guidelines
- Official Springer template: https://www.springernature.com/gp/authors/campaigns/latex-author-support
- MGIT institution name: https://mgit.ac.in/about/

No file has been uploaded or emailed to the organisers. Draft placeholders do not assert funding, competing-interest clearance, author approval, copyright transfer or submission acceptance.
