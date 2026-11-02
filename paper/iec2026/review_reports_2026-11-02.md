# Review of similarity and AI writing reports

Reviewed 2 November 2026. This plan was prepared before implementation. The completed manuscript-revision status is recorded in revision_2026-11-02/response_to_reports.md; scientific results remain unchanged.

## Findings from the supplied reports

| Evidence | Finding | Interpretation and action |
|---|---|---|
| Similarity PDF, pages 14-15 | Overall similarity 13%; internet 12%, publications 13%, student papers 10%. Sixteen primary sources; largest displayed individual matches are 1%. | Source categories overlap and must not be added. The score is a text-match measure, not a plagiarism finding or acceptance decision. |
| Similarity PDF, page 15 | Exclude bibliography, quotes and matches are all Off. | Preserve this unfiltered report. Request an additional bibliography-excluded report using the supervisor's approved settings; record all settings. No filtered score can be inferred from the PDF. |
| Similarity PDF, pages 11-13 | Every bibliography entry has some highlighted material, including titles, author lists and DOI strings. | Keep bibliographic details accurate. They should not be paraphrased to reduce a score. |
| Similarity PDF, pages 2-3 | Matches include a title fragment, affiliation and template labels, standard intensity terms, and the names of IBTrACS, OISST and ERA5. | Preserve official names and correct citations. Check any substantive borrowed wording individually. A match to a generic title fragment is not evidence of copied research. |
| Similarity PDF, pages 4-10 | No long matched Methods or Results blocks are highlighted in this supplied report. | This supports a limited conclusion about these visible matches. It is not a complete originality certification or a comparison with every possible source. |
| AI PDF, page 2 | 50% of qualifying prose is flagged as likely AI-generated or AI-generated and revised; report warns that human review is required. | This is independent of similarity and is not a 50% probability of misconduct. The draft was AI-assisted; the response must acknowledge the actual writing process. |
| AI PDF, pages 3-5 and 8-12 | Highlights include the abstract framing, most of the introduction, parts of data/evaluation prose, substantial results commentary, discussion, conclusion and declarations. | Prioritise an author-led substantive revision of these sections. Some boilerplate and necessary technical language is also highlighted, so sentence-by-sentence label chasing is inappropriate. |
| AI PDF, pages 13-14 | Bibliography largely unhighlighted, with some caption text highlighted. | The AI percentage and similarity percentage use different populations. They cannot be subtracted, combined or converted into one another. |

## Revision sequence

1. **Preserve the baseline.** Retain these reports and the exact checked paper. Keep the frozen model, data roles, calibration and final-test outputs unchanged. Writing concerns do not justify repeating or tuning on the consumed test.
2. **Review similarity by passage.** Classify each match as a bibliography item, proper name/template wording, necessary terminology, attributed paraphrase or substantive borrowed wording. Correct attribution and genuinely close paraphrases where needed. Keep original provider names, reference titles, units, equations and required acknowledgement wording accurate.
3. **Refresh related work.** The report identifies three relevant 2026 publications absent from the current bibliography. Their publisher records have now been verified. Read the available full texts where accessible, record access limitations, and compare inputs, targets, horizon, split design, references and uncertainty evaluation. Cite studies for relevance, not merely because Turnitin lists them as matching sources.
4. **Rebuild the argument from experiment records.** The authors should supply or approve their technical account of the research question, selection chronology and interpretation. Use a claim-to-evidence table connecting each result to the saved metrics and correct denominator. Revise the introduction, results interpretation, discussion and conclusion from that account. Language editing can then improve clarity without replacing technical judgment.
5. **Make interpretations specific.** Explain why the MLP is the selected model, distinguish final testing from reused selection data, and explain the practical meaning of intensity improvements, weak-regime degradation and central overcoverage. Do not invent causes such as reduced overfitting or improved environmental learning without supporting experiments. Shorten audit logistics in the main text and retain their detailed receipts in S1.
6. **Complete declarations accurately.** Retain a truthful AI-use statement describing actual drafting and other material assistance. Do not recast this workflow as grammar-only editing. Confirm funding, competing interests, contributions, availability and author approval with the authors. Only claim that authors verified and approved content after that review occurs. Check the actual congress and departmental requirements alongside the applicable journal policy.
7. **Check and reassess.** Verify citations, all numerical claims, paired populations and units; preserve the 150-word synopsis and paper length limit; regenerate Word, LaTeX and the companion consistently. Obtain one fresh report through the authorised institutional workflow, documenting filters and any earlier-draft/self-matches. Evaluate remaining passages under the actual venue/institution policy. No particular detector percentage can be promised.

## Section priorities

| Section | Required revision |
|---|---|
| Abstract | State the exact six-hour retrospective task, test population, matched-reference gains and mixed uncertainty outcome directly. |
| Introduction and related work | Replace broad framing with a focused comparison against the relevant regional and uncertainty literature; define the limited empirical contribution. |
| Data and methods | Preserve correct technical names and equations. Verify missing-label handling and provenance rather than renaming terminology. |
| Results | Lead with absolute errors, paired label counts and intervals. Explain mixed findings without unsupported causal claims. |
| Discussion and conclusion | Give the authors' interpretation of what the evaluated model supports and what information-timing and missing-label gaps prevent. |
| Declarations and captions | Complete factual placeholders, disclose actual assistance and retain required provider wording. |

## Additional related work verified

- Faiaz et al. (2026), *Physics-guided non-autoregressive transformer for lightweight cyclone track prediction in the Bay of Bengal*, Expert Systems with Applications, DOI 10.1016/j.eswa.2026.131972. Publisher: https://www.sciencedirect.com/science/article/pii/S0957417426008857. The authors' preprint describes a sequence-to-sequence track task over 48 hours, unlike this paper's six-hour track and intensity task: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5985599.
- Haque et al. (2026), *Implementation of a Convolutional Neural Network for 24-hour tropical cyclone track forecasting in the Bay of Bengal using fused historical and reanalysis data*, DOI 10.1007/s00704-026-06341-5. Publisher abstract describes IBTrACS and ERA5 inputs for a 24-hour track task: https://link.springer.com/article/10.1007/s00704-026-06341-5.
- Tsai et al. (2026), *Quantifying situation-dependent uncertainty in tropical cyclone track forecasts with a recurrent neural network approach*, DOI 10.1007/s11069-026-08023-x. Publisher abstract describes LSTM-based uncertainty using official and numerical forecast guidance: https://link.springer.com/article/10.1007/s11069-026-08023-x.

These records establish relevant prior work; full methodological equivalence or a numerical performance ranking has not been established. Publisher abstracts for the two Springer studies are available, but the full texts are subscription content. Raw errors across different lead times and populations must not be compared as a leaderboard.

## Guidance checked

- Turnitin similarity interpretation: https://helpcenter.turnitin.com/hc/en-us/articles/27811925295501-Does-Turnitin-check-for-plagiarism-and-what-is-a-good-Similarity-score
- Turnitin filters: https://guides.turnitin.com/hc/en-us/articles/23539146689549-How-exclusion-filters-refine-the-Similarity-Report
- Turnitin AI-report interpretation: https://guides.turnitin.com/hc/en-us/articles/22774058814093-Using-the-AI-Writing-Report
- Provisional IEI Series B journal instructions, including LLM disclosure and author responsibility: https://link.springer.com/journal/40031/submission-guidelines
- Springer Nature AI manuscript policy: https://www.springernature.com/gp/policies/editorial-policies/ai-manuscript-preparation

No venue-specific numerical acceptance threshold is supplied in the two reports. Their existence does not establish acceptance or rejection. The appropriate response is documented attribution, author-led interpretation and accurate disclosure; any recheck remains evidence for human assessment.
