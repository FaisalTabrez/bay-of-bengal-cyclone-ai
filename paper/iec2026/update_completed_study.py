"""Migrate the held draft to verified completed results; never run a model."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
backup = HERE/'qa/before_completed_results'
backup.mkdir(parents=True, exist_ok=True)
for name in ['manuscript_content.json','build_draft.py','README.md','submission_notes.md',
             'revision_response.md','drafts/IEC2026_Paper_Draft.docx',
             'drafts/IEC2026_Extended_Abstract.docx','drafts/IEC2026_Springer_Draft.pdf']:
    dest = backup/name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE/name, dest)

C = json.loads((HERE/'manuscript_content.json').read_text(encoding='utf-8'))
if not any(section['title']=='Results on hold' for section in C['sections']):
    raise SystemExit('One-time migration already applied. Edit manuscript_content.json and use build_draft.py for subsequent revisions.')
C['abstract'] = (
    'This study evaluates compact machine learning models for six hour cyclone track and intensity prediction in the Bay of Bengal. '
    'Historical cyclone records, atmospheric reanalysis and sea surface temperature provide 39 predictors with explicit missingness indicators. '
    'Chronological storm partitions separate training, selection, uncertainty calibration and final testing. '
    'Development selected a three seed residual multilayer perceptron ensemble. '
    'On 1018 observations from 65 cyclones during 2018 to 2025, mean track error is 35.915 kilometres, improving on constant velocity by 11.87%. '
    'Paired wind and pressure errors improve on persistence by 12.41% and 14.74%, respectively; whole cyclone bootstrap intervals favour the model. '
    'A Gaussian spread network improves intensity probability scores over constant spread, while spatial score improvement remains unclear. '
    'Calibration has mixed effects and central uncertainty regions overcover. '
    'These findings support retrospective regional prediction; operational warning skill requires latency matched inputs and operational references in a separate carefully designed prospective evaluation before practical deployment.'
)
# The linked IEC maximum and provisional Springer minimum intersect at 150 words.
assert len(C['abstract'].split()) == 150, len(C['abstract'].split())
assert len(re.findall(r"\b[\w'-]+(?:\.[0-9]+)*\b", C['abstract'])) == 150
C['sections'] = [
 {'title':'Introduction','paragraphs':[
  'Cyclone position and intensity information can inform coastal preparedness. Regional prediction studies must account for correlated observations, incomplete intensity labels and retrospective environmental products. This study asks whether compact residual models improve six-hour predictions for later, unseen Bay-affecting storms over physical references, and whether calibrated uncertainty improves probability scores around fixed point forecasts.',
  "Giffard-Roisin et al. combined trajectories and spatial reanalysis for 24-hour track forecasting [@giffard2020]. Mawatwal and Das studied North Indian Ocean satellite-based intensity estimation and contrasted chronological testing with stratified splitting [@mawatwal2024]. Their estimation target differs from six-hour prediction. Fernandez et al. estimated track-forecast uncertainty with a neural bivariate Gaussian [@fernandez2025]. Different inputs, populations and horizons preclude direct numerical comparisons.",
  'The contribution is an integrated regional evaluation of track, wind, pressure and uncertainty, with explicit label populations and separate selection, calibration and testing roles. Snapshot and tree controls assess the need for recurrent history, consistent with strong tabular baselines [@gorishniy2021]. No architectural novelty or first regional application is asserted.'
 ]},
 {'title':'Data and labelled populations','paragraphs':[
  'At an existing observation time t, targets are cyclone position, maximum sustained wind and minimum central pressure six hours later. The derived 1982-2025 package combines the International Best Track Archive for Climate Stewardship (IBTrACS) v04r01 [@knapp2010;@gahtan2024], Optimum Interpolation Sea Surface Temperature (OISST) v2.1 [@huang2021], and European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 atmospheric reanalysis [@hersbach2020]. Main tracks use synoptic hours and continuous six-hour pairs. New Delhi intensity fields are preferred, with World Meteorological Organization values as fallback.',
  'Table 1 gives whole-storm roles. North Indian Ocean (NIO) training includes all Bay training rows and adds 997 pairs from 54 storms. Bay-affecting lifecycles extend outside the geographic Bay box, 5-25 degrees north and 78-100 degrees east. Boundary-spanning storms enter the later role. Earlier baseline acceptance inspected the full validation archive; calibration was reserved from learned-model selection rather than never inspected.',
  'Only 51.2% of Bay training rows have wind targets and 49.7% have pressure targets. Some future labels lack current measurements. Supplement S1 reports persistence-eligible counts and the ordered 39 predictors describing storm state, tendencies, calendar, location and environmental conditions. SST outside the supplied regional crop retains missing values and an availability flag. Reproduction starts from the derived package; raw extraction notebooks and some acquisition details remain unavailable.'
 ]},
 {'title':'Residual models and selection','paragraphs':[
  'Training-only standardisation and zero imputation accompany 39 missingness indicators, yielding 78 inputs. Near-constant features use unit scale. Missing future labels are masked before loss arithmetic; observed targets are retained when current intensity is unavailable.',
  'The track anchor extrapolates the previous six-hour latitude/longitude changes, with persistence fallback. Anchor and target positions use the spherical logarithmic map at the current centre. Intensity anchors use current observations or training current-feature means. With transformed target y, anchor a and diagonal training scale S, residual learning and reconstruction are:',
  'The two track axes share a training residual root-mean-square (RMS) scale; wind and pressure use separate scales, floored at one native unit. Corrected displacement is decoded with the spherical exponential map. The snapshot multilayer perceptron (MLP) has widths 78-64-32-4. The gated recurrent unit (GRU) projects to 32 features, uses hidden width 64 and a current-input skip connection. Up to four consecutive observations span 18 hours; snapshot inputs already contain six-hour tendencies.',
  'Original candidates comprise Bay-only MLP, GRU and boosted trees, NIO-pretrained GRU with Bay fine-tuning, and pooled NIO GRU. Masked SmoothL1 equally weights active tasks. A bounded follow-up, defined after inspecting selection results, compares the original MLP, observed-current wind persistence replacement and absolute wind loss with other losses unchanged. This reused selection data and constitutes exploratory development.',
  'Checkpoints minimise J below using identical intensity rows for model and persistence. Families are compared by the mean individual-seed score for seeds 17, 29 and 43. T and Tcv denote track mean errors; W and P denote intensity mean absolute errors (MAEs), with persistence references Wpers and Ppers. The selected family averages three forecasts in common tangent-plane and native intensity coordinates.',
  'The completed follow-up selected the Bay-only MLP with absolute wind loss; the GRU and transfer candidates remain development comparisons. Only the frozen selected model entered final testing, so the final evidence does not rank all architectures. Supplement S1 provides schedules, settings, selection records and source identities.'
 ]},
 {'title':'Uncertainty and reference forecasts','paragraphs':[
  'A separate 78-32-5 MLP predicts four positive standard deviations and bounded track correlation around the frozen ensemble mean. Track is bivariate Gaussian; wind and pressure are independent marginals. Scale floors are 1 km per track axis, 0.2572 metres per second (0.5 knots) and 50 Pa (0.5 hPa); correlation is 0.95 times a hyperbolic tangent. Cross-task dependence is omitted.',
  'Spread training minimises scale-normalised Gaussian negative log likelihood (NLL), selecting epoch 48 of 50. In-sample training error RMS initialises spread. Reserved calibration fits three standard-deviation multipliers: 0.94898 for track, 1.05730 for wind and 0.80831 for pressure. Means and correlation remain fixed; track covariance scales by the squared multiplier.',
  'Probability references share the frozen means: a constant diagonal Gaussian using frozen training-error RMS scales, and the uncalibrated learned Gaussian. The declared final predictor is the calibrated Gaussian. The RMS reference is fitted on in-sample errors; it is not an out-of-fold estimate. Figure 1 summarises the workflow.'
 ]},
 {'title':'Evaluation and reproducibility','paragraphs':[
  'Point metrics include great-circle track errors, paired intensity MAEs and all-observed-label MAEs. Probability metrics include native-unit NLL, continuous ranked probability score (CRPS), spatial energy score using 512 draws (seed 7101), and coverage, width or area at 50%, 80% and 90%. Proper scores complement coverage [@gneiting2007]. Track density is defined in tangent-plane kilometres.',
  'Nominal 95% percentile intervals use 2000 whole-cyclone resamples (seed 1701), retaining row-weighted means and identical eligible rows for paired differences. No multiplicity correction is claimed. The geographic Bay subset and current-intensity/land-proximity diagnostics were declared before testing. Cyclones may enter several subgroups; subgroup estimates are descriptive.',
  'The final evaluation completed on 30 October 2026. A separate numerical audit verified all 91 final and 71 preflight manifested files, unchanged frozen weights/calibration, and consistent plan, receipts and ledger. Saved predictions reproduced point errors, paired intervals, Gaussian likelihoods, marginal CRPS and coverage without training, inference or original-test decoding. The audit checked energy-score file integrity but did not repeat its Monte Carlo calculation. Supplement S1 binds complete exported metrics and provenance; the consumed holdout must not guide further tuning.'
 ]},
 {'title':'Results','paragraphs':[
  'The 2018-2025 test contains 1018 observations from 65 cyclones. Table 2 reports aggregate comparisons: track error falls by 11.87%, paired wind MAE by 12.41% and paired pressure MAE by 14.74%. Each nominal paired difference interval favours the model. Track median and 90th-percentile errors are 28.802 and 69.897 km. Across all available intensity targets, MAEs are 2.719 knots (808 labels) and 1.404 hPa (852 labels); these cannot replace paired values in reference comparisons.',
  'The geographic Bay subset contains 789 observations from 62 cyclones. Relative error reductions are 11.95% for track, 13.19% for wind and 17.58% for pressure; both paired intensity counts are 656. This overlapping subset is not independent confirmation. Among observations with current wind below 34 knots, wind MAE is 1.536 versus 1.517 knots and pressure MAE is 0.714 versus 0.684 hPa. Aggregate gains therefore coexist with weak-regime degradation.',
  'Table 3 compares probability forecasts. Calibrated intensity NLL and CRPS improve over the RMS reference, with paired difference intervals excluding zero. Track NLL and energy-score intervals include zero against that reference. Compared with the uncalibrated Gaussian, calibration improves pressure CRPS and track energy score but worsens wind CRPS by 0.02124 knots (95% interval 0.01441 to 0.02842). All three calibrated-minus-uncalibrated NLL intervals include zero. The frozen calibrated predictor is retained without test-driven revision.',
  'Figure 2 shows excess central coverage: nominal 50% regions contain 64.44% of track, 67.20% of wind and 60.92% of pressure outcomes. At 90%, coverage is 89.49%, 89.11% and 92.02%, respectively. Mean 90% widths are 16.312 knots and 6.570 hPa; mean track ellipse area is 16268.3 square kilometres. Near-nominal outer coverage does not establish calibration throughout the distribution. Supplement S1 retains every declared subgroup, score and interval.'
 ]},
 {'title':'Discussion and limitations','paragraphs':[
  'The frozen selected MLP improves aggregate retrospective six-hour prediction over constant velocity and persistence. The study does not establish superiority to operational numerical forecasts or isolate the benefit of each predictor, loss or architectural component. Repeated development on selection storms limits interpretation of development rankings; final findings apply to the model fixed before testing.',
  'Best-track revision, reanalysis release latency and daily SST support prevent assuming real-time availability from causal row ordering. Same-date SST can include later observations. Operational evaluation needs a source-timing audit, latency-matched inputs and operational references. Original extraction notebooks, interpolation choices and some derived-feature definitions remain unresolved.',
  'Incomplete labels, reporting conventions and geographic/temporal shifts limit generalisation. Gaussian wind allows negative mass and pressure is unbounded; no tail clipping was applied. Row-weighted calibration from 25 cyclones gives no population-wide coverage guarantee. These findings concern retrospective prediction, without measured warning benefits, landfall-event skill, inundation, avoided losses or energy savings.'
 ]},
 {'title':'Conclusion','paragraphs':[
  'A three-seed residual snapshot MLP improves aggregate six-hour track, wind and pressure prediction on later Bay-affecting cyclones against declared physical references. Gaussian uncertainty improves intensity scores over a constant-spread reference, while spatial gains remain unclear and central regions overcover. Future operational assessment requires a separate untouched evaluation set and verified information availability.'
 ]}
]
C['declarations'][4][1] = ('Code is available at https://github.com/FaisalTabrez/bay-of-bengal-cyclone-ai. '
    'Supplement S1 v1.1 includes the feature contract, population counts, complete final metrics, reviewed plan and numerical audit. '
    'A local research archive retains frozen weights and original exports; no public archival DOI is assigned. '
    'Provider terms apply separately. Code licensing and original extraction notebooks remain unresolved.')
C['extended_abstract'] = [
 {'title':'Background','paragraphs':[
  'Reliable cyclone information can support coastal preparedness and infrastructure resilience. Regional machine learning assessments must distinguish forecast improvement from the effects of correlated observations, incomplete intensity records and retrospective information. This study evaluates compact models for six-hour prediction of cyclone position, maximum sustained wind and minimum central pressure in Bay-affecting storms. It also tests whether uncertainty estimation improves probability scores around fixed point forecasts.'
 ]},
 {'title':'Data and methods','paragraphs':[
  'The derived 1982-2025 package combines International Best Track Archive for Climate Stewardship cyclone records, ERA5 atmospheric reanalysis and Optimum Interpolation Sea Surface Temperature. Bay training contains 4560 pairs from 235 cyclones, with 2334 observed wind targets and 2267 observed pressure targets. Broader North Indian Ocean training contains 5557 pairs from 289 cyclones and includes all Bay training rows. Separate chronological roles contain 493 selection observations from 25 cyclones and 361 calibration observations from 25 cyclones. The final 2018-2025 test contains 1018 observations from 65 cyclones.',
  'Training-only standardisation and imputation accompany 39 predictors and their missingness indicators. Partially observed intensity labels are masked rather than filled. Residual models correct constant-velocity track and persistence intensity forecasts using spherical tangent-plane displacement. Original development compares snapshot neural, recurrent and tree models, including broader-basin training. A bounded wind-loss follow-up reuses selection storms and is disclosed as exploratory. The selected predictor is a Bay-only multilayer perceptron ensemble with seeds 17, 29 and 43 and hidden widths 64 and 32.',
  'A separate Gaussian spread network predicts bivariate track uncertainty and marginal wind and pressure uncertainty around the frozen ensemble mean. Three standard-deviation factors are fitted on reserved calibration storms without changing the means. Probability references are a diagonal Gaussian using frozen in-sample training-error scales and the uncalibrated learned Gaussian. Final reporting uses matched model/reference populations and 2000 whole-cyclone bootstrap resamples. The final evaluation and a separate numerical audit are complete.'
 ]},
 {'title':'Key results','paragraphs':[
  'Mean track error is 35.915 km versus 40.753 km for constant velocity, an 11.87% reduction across 1018 observations. Paired wind mean absolute error is 2.727 versus 3.113 knots across 779 observations, a 12.41% reduction. Paired pressure mean absolute error is 1.385 versus 1.625 hPa across 823 observations, a 14.74% reduction. Nominal 95% model-minus-reference intervals are -6.213 to -3.585 km, -0.628 to -0.162 knots and -0.450 to -0.046 hPa. All-observed-label intensity counts are 808 and 852; these differ from the paired comparison denominators.',
  'Calibrated wind and pressure continuous ranked probability scores are 2.322 knots and 1.037 hPa, versus 2.631 knots and 1.332 hPa for the constant-spread reference. Paired intervals favour these intensity improvements. Track energy scores are 26.784 versus 26.790 km, with a difference interval including zero. Calibration worsens wind probability score relative to the uncalibrated model. Nominal 50% regions cover 64.44% of track, 67.20% of wind and 60.92% of pressure outcomes; central uncertainty is overly broad. Aggregate point improvement does not extend to every intensity regime.'
 ]},
 {'title':'Conclusions and scope','paragraphs':[
  'The study provides a regional retrospective comparison with explicit physical/probability references, missing-label populations, experiment chronology and a reproducibility supplement. Compact residual prediction improves aggregate six-hour track and intensity errors, while uncertainty results are mixed. Best-track revisions, reanalysis latency and daily sea surface temperature timing require a source-availability audit before operational use. Warning capability, disaster-loss reduction and energy savings have not been measured. The final holdout is consumed; future model changes require a separate study and untouched evaluation set.'
 ]}
]
(HERE/'manuscript_content.json').write_text(json.dumps(C,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

# Bind the paper to the actual completed export, preserving historical snapshots.
s = HERE/'supplement_v1'
resultdir = s/'results'
resultdir.mkdir(exist_ok=True)
auditpath = HERE/'qa/final_results_review/final_test_export_verification.json'
audit = json.loads(auditpath.read_text())
assert audit['status']=='passed' and len(audit['checks'])==110 and all(audit['checks'].values())
shutil.copy2(auditpath,resultdir/'final_test_export_verification.json')
with zipfile.ZipFile(Path.home()/'Downloads/bob_cyclone_final_results.zip') as z:
    assert hashlib.sha256(Path(z.filename).read_bytes()).hexdigest()==audit['archives']['final']['sha256']
    for original,target in [
        ('execution/final_metrics.json','final_metrics.json'),
        ('prepared/reviewed_plan.json','reviewed_plan.json'),
        ('execution/result_receipt.json','result_receipt.json'),
        ('execution/ledger_receipt.json','ledger_receipt.json'),
        ('prepared/model/mean/deterministic_decision.json','deterministic_decision.json'),
        ('prepared/model/calibration/calibration.json','calibration.json')]:
        (resultdir/target).write_bytes(z.read(original))
for doc in ['final-results.md','model-card.md','reproducibility-archive.md']:
    shutil.copy2(ROOT/'docs'/doc,resultdir/doc)
shutil.copy2(ROOT/'scripts/verify_final_exports.py',resultdir/'verify_final_exports.py')
shutil.copy2(ROOT/'modelling/configs/final_evaluation_v1.json',s/'configs/final_evaluation_v1.json')
with (s/'label_populations.csv').open(newline='',encoding='utf-8') as f:
    reader=csv.DictReader(f); fields=reader.fieldnames; populations=list(reader)
if not any(r['partition']=='bob_test' for r in populations):
    # SST count is outside the result review and is left unreported, not guessed.
    populations.append(dict(zip(fields,['bob_test',1018,65,808,779,29,852,823,29,'unreported'])))
with (s/'label_populations.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(populations)
p=json.loads((s/'provenance.json').read_text())
p.update(version='1.1',scope='Derived-package methods and completed frozen-evaluation reporting',
         final_holdout_consumed=True,final_results_archive_sha256=audit['archives']['final']['sha256'],
         preflight_archive_sha256=audit['archives']['preflight']['sha256'],
         reviewed_plan_sha256=audit['reviewed_plan_sha256'],attempt_id=audit['attempt_id'],
         manuscript_update_original_test_decoding=False,manuscript_update_inference=False,
         manuscript_update_saved_output_audit_checks=110)
p['current_working_source_sha256']['modelling/configs/final_evaluation_v1.json']=hashlib.sha256((s/'configs/final_evaluation_v1.json').read_bytes()).hexdigest()
p['result_file_sha256']={str(f.relative_to(s)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(resultdir.iterdir())}
p['test_parquet_decoded']=False
p['test_parquet_decoded_note']='This field describes the original methods-documentation step. The historical final evaluator consumed the holdout; this manuscript revision only audits its saved exported cache and predictions.'
(s/'provenance.json').write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8')
text=(s/'README.md').read_text(encoding='utf-8')
text=text.replace('Version 1.0','Version 1.1')
text=text.replace('It does not supply new model results or complete the held experiments.', 'Version 1.1 incorporates the completed frozen final evaluation and its independent saved-output numerical audit. No model was refitted or rerun during this manuscript update.')
text=text.replace('computed only from training and validation partitions.','computed from training/validation in v1.0 and extended with already exported final-test counts in v1.1.')
text=text.replace('including the separate final-evaluation plan.', 'including the final-evaluation protocol updated to match the completed study. Historical method-source snapshots retain their earlier identities; the actual reviewed plan and source map are in results/.')
text=text.replace('Final model weights and research-result archives remain outside this drafting supplement.', 'The complete local research archive, described in results/reproducibility-archive.md, contains frozen weights and original exports. This supplement includes numerical metrics and receipts but does not duplicate model binaries.')
text=text.replace('| Calibration role | 361 | 25 | 298 | 291 | 303 | 297 |','| Calibration role | 361 | 25 | 298 | 291 | 303 | 297 |\n| Completed final test | 1018 | 65 | 808 | 779 | 852 | 823 |')
start=text.index('The final test is omitted from this availability audit.')
end=text.index('\n\n## Temporal',start)
text=text[:start]+('Final-test availability comes from the completed exported evaluation: 808 wind and 852 pressure targets, with 779 and 823 persistence-eligible pairs. The 29 additional labels for each intensity have missing current values. The manuscript audit uses the saved exported test cache; it does not decode the original dataset or repeat inference. Unknown labels remain unfilled. Reporting coverage changes substantially across chronological roles, and missingness is not assumed random.')+text[end:]
text=text.replace('to be assessed by the held evidence','assessed using the completed frozen final evaluation')
start=text.index('## Items still on hold')
text=text[:start]+'''## Completed result evidence and remaining limits

The final 2018-2025 holdout was consumed on 30 October 2026. The selected point model is bay_mlp_wind_absolute_loss, seeds 17, 29 and 43. The recurrent and transfer models were development comparisons, not additional final-test competitors. The three calibrated spread multipliers are approximately 0.94898128, 1.05729715 and 0.80830777 for track, wind and pressure.

`results/final_metrics.json` retains every declared cohort, score, coverage/width/area, tail diagnostic and bootstrap interval without rounding. `reviewed_plan.json`, `deterministic_decision.json`, `calibration.json`, `result_receipt.json` and `ledger_receipt.json` bind model selection, calibration and execution to the frozen study. The full plan's files map records historical source/model hashes. These historical identities take precedence over the earlier illustrative method-source snapshots.

`final_test_export_verification.json` records the fresh 110-check saved-output audit. It verified 91 final and 71 preflight manifested files, unchanged model/calibration bytes, paired point intervals and analytical Gaussian scores/coverage. Energy-score Monte Carlo calculations were not independently repeated. The copied audit script is for inspection; run it in the complete research archive as described by the reproducibility guide, not as an isolated standalone script.

Track, wind and pressure errors improve against the declared physical references, with matched denominators. Probability results are mixed: intensity scores improve over the fitted training-RMS reference, track improvements over that reference are unclear, wind CRPS worsens after calibration, and central regions overcover. Complete subgroup results include weaker or adverse regimes. These findings do not establish operational warning skill or superiority to operational numerical prediction.

Raw-source construction, latency-matched evaluation, component ablations, public archival deposit, code licensing and author declarations remain unresolved where stated. None is asserted complete. No further tuning or calibration may use the consumed final test; new model changes require a separate study and untouched evaluation set.
'''
(s/'README.md').write_text(text,encoding='utf-8')
print('Updated manuscript and Supplement S1 v1.1 from verified saved results.')
