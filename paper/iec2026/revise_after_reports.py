"""Prepare an evidence-bound manuscript revision after the supplied reports.

This edits documents only. It never trains, infers, calibrates, or decodes data.
"""
from pathlib import Path
import csv
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REV=HERE/'revision_2026-11-02'
REV.mkdir(exist_ok=True)
C=json.loads((HERE/'manuscript_content.json').read_text(encoding='utf-8'))
C['revision_date']='2026-11-02'
audit=json.loads((HERE/'supplement_v1/results/final_test_export_verification.json').read_text())
metrics=json.loads((HERE/'supplement_v1/results/final_metrics.json').read_text())
assert audit['status']=='passed' and all(audit['checks'].values())

scientific_paths=[p for p in (ROOT/'modelling').rglob('*') if p.suffix in {'.py','.json'}]
scientific_paths += list((HERE/'supplement_v1/results').glob('*'))
scientific_paths += [ROOT/'data/packages/bob_cyclone_training_v2.zip']
identity={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in scientific_paths if p.is_file()}
(REV/'unchanged_scientific_files.json').write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')

def write_csv(name,fields,rows):
    with (REV/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

# Report source numbers identify visible highlights, not definitive original sources.
matches=[
 (1,'dergipark.org.tr','11','Reference 1','bibliography'),
 (2,'people.iiti.ac.in','12','Reference 2','bibliography'),
 (3,'ouci.dntb.gov.ua','12','Reference 6','bibliography'),
 (4,'www.wpcouncil.org','12','Reference 5','bibliography'),
 (5,'ajdsa-journal.org','12','Reference 4','bibliography'),
 (6,'assets-eu.researchsquare.com','2;12','Affiliation and template labels; journal metadata','proper names and bibliography'),
 (7,'repository.library.noaa.gov','12','Reference 7','bibliography'),
 (8,'www.koreascience.or.kr','13','Reference 9','bibliography'),
 (9,'egusphere.net','13','Reference 8','bibliography'),
 (10,'Tsai et al. 2026','12','Reference 3','bibliography; relevant prior art'),
 (11,'pure.mpg.de','3','ECMWF ERA5 product name','official product name'),
 (12,'www.mdpi.com','3','IBTrACS product name','official product name'),
 (13,'Faiaz et al. 2026','2','Title fragment','generic subject phrase; relevant prior art'),
 (14,'www.research-collection.ethz.ch','3','OISST product name','official product name'),
 (15,'Ali dissertation','3','Maximum sustained wind and minimum central pressure','standard variable terminology'),
 (16,'Haque et al. 2026','2','Abstract phrase identifying the Bay of Bengal','geographic wording; relevant prior art')
]
write_csv('similarity_match_audit.csv',['report_source','source','pdf_pages','location','classification','action'],[
 dict(report_source=n,source=s,pdf_pages=pages,location=loc,classification=kind,
      action='Retain accurate names and bibliographic text; verify attribution. Relevant new studies added after source checks.' if 'prior art' in kind else 'Retain accurate wording and citation; no long borrowed prose identified in this visible match.')
 for n,s,pages,loc,kind in matches])

literature=[
 dict(study='Faiaz et al. 2026',doi='10.1016/j.eswa.2026.131972',
      source='https://www.sciencedirect.com/science/article/pii/S0957417426008857',
      additional_source='https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5985599',
      access='Publisher metadata and author preprint abstract; full text not retrieved; direct publisher/preprint fetch returned 403 on revision date',
      inputs='Best-track and recent motion information in preprint abstract',targets='Track trajectory',horizon='48 hours at 3-hour intervals in preprint abstract',
      split='Out-of-sample testing stated in abstract; exact grouping and dates not independently established',
      references='Autoregressive decoding comparison described in preprint abstract',
      uncertainty='Not established from the inspected abstract; no absence claim made',
      comparison='Different input representation, trajectory horizon and objective; no numerical ranking against this study'),
 dict(study='Haque et al. 2026',doi='10.1007/s00704-026-06341-5',
      source='https://link.springer.com/article/10.1007/s00704-026-06341-5',additional_source='',
      access='Publisher abstract and bibliographic record; full text is subscription content',
      inputs='IBTrACS history and ERA5 atmospheric fields',targets='Track',horizon='24 hours',
      split='Exact split and leakage controls not established from publisher abstract',
      references='Persistence comparison described in publisher abstract',
      uncertainty='Not established from inspected abstract; no absence claim made',
      comparison='Different lead time and input representation; direct error comparison would be misleading'),
 dict(study='Tsai et al. 2026',doi='10.1007/s11069-026-08023-x',
      source='https://link.springer.com/article/10.1007/s11069-026-08023-x',additional_source='',
      access='Publisher abstract and bibliographic record; full text is subscription content',
      inputs='CWA official guidance plus ECMWF and NCEP deterministic/ensemble forecasts',targets='Track forecast error and situation-dependent uncertainty cone',
      horizon='Exact lead-time schedule not established from inspected abstract',
      split='Exact split not established from inspected abstract',
      references='CWA official forecasts discussed in abstract',uncertainty='LSTM-based track forecast error and cone estimation',
      comparison='Operational-guidance post-processing differs from six-hour residual prediction using retrospective observations and reanalysis')
]
write_csv('related_work_evidence.csv',list(literature[0]),literature)
(REV/'literature_access.md').write_text('''# Literature access record

Checked 2 November 2026. The comparison matrix distinguishes publisher records and abstracts from full-text review. No full text was obtained for the three newly cited studies. Direct ScienceDirect and SSRN fetches returned HTTP 403; indexed primary-source text and the author publication page corroborated the metadata and preprint abstract. The two Springer pages supplied their abstracts and subscription-access notices.

Claims added to the main paper are limited to task, input family, method family and lead time supported by these records. The 48-hour task is explicitly attributed to the Faiaz author preprint and is not assumed identical in every detail to its journal version. No numerical performance ranking, precise split reconstruction or claim that a study lacks uncertainty methods is made.

The matrix records unknown split, benchmark and uncertainty details rather than treating unavailable text as evidence of absence. The full texts can support a later deeper comparison if the authors obtain them through normal institutional access.
''',encoding='utf-8')

C['abstract']=(
 'Six-hour forecasts of cyclone position, wind and pressure were evaluated for Bay-affecting storms using a frozen residual neural ensemble. '
 'Training used historical storm records, atmospheric reanalysis and sea surface temperature, with explicit masks for missing predictors and targets. '
 'Complete storms were assigned to chronological training, selection, calibration and test roles. '
 'The selected model averages three multilayer perceptrons. '
 'On 1018 observations from 65 cyclones during 2018-2025, mean track error was 35.915 kilometres, compared with 40.753 for constant velocity. '
 'Paired wind and pressure errors fell by 12.41% and 14.74% against persistence. '
 'Whole-cyclone bootstrap intervals favoured all three point comparisons. Errors varied across the current intensity regimes. '
 'Gaussian uncertainty improved intensity probability scores relative to a constant-spread reference, but track-score differences remained inconclusive. '
 'Calibration worsened wind probability score relative to the uncalibrated model, and nominal 50% regions overcovered. '
 'The evidence supports retrospective prediction; real-time warning performance requires separate evaluation with operationally available inputs.'
)

C['sections']=[
 {'title':'Introduction','paragraphs':[
  'This study tests six-hour cyclone predictions against constant-velocity track and persistence intensity forecasts. Later cyclones are kept out of model selection. A second comparison evaluates probability forecasts around the same fixed point mean.',
  'Regional deep learning already spans several forecasting tasks. Giffard-Roisin et al. combined trajectories and reanalysis for 24-hour tracks [@giffard2020]. Haque et al. described a Bay of Bengal convolutional model using storm histories and atmospheric reanalysis for 24-hour track prediction [@haque2026]. Faiaz et al. developed a physics-guided transformer for Bay tracks; their author preprint describes a 48-hour trajectory task [@faiaz2026]. Mawatwal and Das addressed satellite-based North Indian Ocean intensity estimation [@mawatwal2024]. For uncertainty, Fernandez et al. used a neural bivariate Gaussian for track errors [@fernandez2025], while Tsai et al. used recurrent processing of official and numerical guidance to estimate track uncertainty [@tsai2026]. These tasks differ in horizon, predictors and reference forecasts, so their reported errors cannot rank the present model.',
  'Tabular baseline comparisons motivate snapshot and tree controls [@gorishniy2021]; recurrent candidates additionally test sequence history. The selected point ensemble and spread model are frozen for later-storm testing. The contribution is the regional track/intensity comparison with explicit label populations; the architectures and Gaussian representation are established methods.'
 ]},
 {'title':'Data and labelled populations','paragraphs':[
  'Each case starts at an observed cyclone position and predicts the same cyclone six hours later. The supplied 1982-2025 package combines the International Best Track Archive for Climate Stewardship (IBTrACS) v04r01 [@knapp2010;@gahtan2024], Optimum Interpolation Sea Surface Temperature (OISST) v2.1 [@huang2021] and European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 reanalysis [@hersbach2020]. Targets are latitude, longitude, maximum sustained wind and minimum central pressure. Synoptic observations form a pair only when their separation is six hours. New Delhi intensity values take precedence over the World Meteorological Organization fallback.',
  'Table 1 gives whole-storm roles. North Indian Ocean (NIO) training includes Bay training and 997 cases from 54 other storms. Bay-affecting lifecycles can leave the reporting box of 5-25 degrees north and 78-100 degrees east. Boundary-spanning cyclones enter the later role. Calibration was reserved from learned-model selection, although earlier baseline acceptance inspected the full validation archive.',
  'Bay training has 2334 wind and 2267 pressure targets among 4560 cases. Missing targets are not reconstructed. The 39 predictors describe storm state, tendencies, calendar, location and environmental conditions. SST outside the crop remains missing with an availability indicator. S1 gives feature order, units and populations. Reproduction starts from the derived package; acquisition and extraction details remain incomplete.'
 ]},
 {'title':'Residual models and selection','paragraphs':[
  'Training statistics standardise the 39 predictor values. Missing standardised values become zero and are accompanied by binary indicators, giving 78 inputs. Near-constant features use unit scale. A missing future intensity is excluded from its loss, while an observed future target remains usable even if current intensity is missing.',
  'For track, the reference continues the preceding six-hour latitude and longitude changes, replacing unavailable changes with zero. A spherical logarithmic map expresses reference and target positions as east/north displacements at the current centre. Intensity references use current measurements, or training current-feature means when unavailable. For transformed target y, reference a and diagonal training scale S, the correction is:',
  'Track coordinates share a training residual root-mean-square (RMS) scale; intensity scales are separate, each floored at one native unit. The spherical exponential map decodes corrected position. The snapshot multilayer perceptron (MLP) has widths 78-64-32-4. The gated recurrent unit (GRU) uses projection width 32, hidden width 64 and a current-input skip. Its one-to-four observations span up to 18 hours; snapshot inputs already contain six-hour tendencies.',
  'Candidates are Bay-only MLP, GRU and trees, NIO-to-Bay GRU and pooled NIO GRU. Original masked SmoothL1 averages active tasks. A post-inspection follow-up compared the original MLP, observed-current wind persistence replacement and absolute wind loss with unchanged track/pressure losses. Reused selection storms make that ranking development evidence.',
  'Seeds are 17, 29 and 43. Checkpoints minimise J; family selection uses the mean seed score. T and Tcv are mean track errors; W and P are intensity mean absolute errors (MAEs), with persistence errors Wpers and Ppers on identical observed-current/target pairs. The ensemble averages members in common displacement and intensity coordinates.',
  'Follow-up mean seed scores were 0.97305 for the original MLP, 0.94253 for persistence replacement and 0.92667 for absolute wind loss. The last was selected and alone entered final testing. This does not isolate a causal mechanism or rank all architectures on the final test. S1 retains settings and receipts.'
 ]},
 {'title':'Uncertainty and reference forecasts','paragraphs':[
  'The point ensemble remains fixed while a 78-32-5 MLP estimates its spread. Four outputs parameterise positive standard deviations and one gives track correlation. The distribution is a bivariate track Gaussian with independent wind and pressure marginals. Native scale floors are 1 km, 0.2572 metres per second (0.5 knots) and 50 Pa (0.5 hPa). Correlation is bounded by 0.95 times a hyperbolic tangent. Cross-task dependence is not represented.',
  'Spread training minimises scale-normalised Gaussian negative log likelihood (NLL); selection chose epoch 48 of 50. Initial scales use in-sample training-error RMS. Reserved calibration fits standard-deviation factors of 0.94898, 1.05730 and 0.80831 for track, wind and pressure. They alter spread, preserving the means and correlation; track covariance changes by the squared track factor.',
  'Three probability forecasts are compared around identical means: the constant diagonal training-RMS Gaussian, the learned uncalibrated Gaussian and the learned calibrated Gaussian. The last is the frozen predictor declared for final evaluation. This design compares spread forecasts without changing point accuracy. The RMS reference uses fitted in-sample errors, rather than out-of-fold errors. Figure 1 shows the study sequence.'
 ]},
 {'title':'Evaluation and reproducibility','paragraphs':[
  'Track error is great-circle distance. Intensity reporting separates all observed future targets from persistence-eligible pairs. Probability evaluation uses native-unit NLL, continuous ranked probability score (CRPS), track energy score and coverage with width or ellipse area at 50%, 80% and 90%. Proper scores assess distribution quality alongside coverage [@gneiting2007]. Track density is defined in tangent-plane kilometres. Energy score uses 512 draws with seed 7101.',
  'Nominal 95% percentile intervals resample complete cyclones 2000 times with seed 1701. Means remain weighted by cases; paired differences use identical eligible cases. No multiplicity adjustment is applied. The geographic Bay subset and current-intensity and distance-to-land diagnostics were specified before testing. A cyclone can contribute to several subgroups, so subgroup counts and findings are descriptive.',
  'Final evaluation completed on 30 October 2026. A saved-output audit reproduced point errors, paired intervals, analytical Gaussian scores and coverage. Energy-score file integrity was checked without repeating its Monte Carlo calculation. S1 retains the plan, frozen identities, receipts and full metrics. The consumed test cannot support further tuning.'
 ]},
 {'title':'Results','paragraphs':[
  'Across 1018 cases from 65 test cyclones, mean track error was 35.915 km against 40.753 km for constant velocity (Table 2). Paired wind MAE was 2.727 against 3.113 knots on 779 cases; pressure MAE was 1.385 against 1.625 hPa on 823 cases. Relative reductions were 11.87%, 12.41% and 14.74%. Nominal model-minus-reference intervals were -6.213 to -3.585 km, -0.628 to -0.162 knots and -0.450 to -0.046 hPa. Each favoured the model. Track median and 90th-percentile errors were 28.802 and 69.897 km. On all observed intensity targets, MAEs were 2.719 knots (808 labels) and 1.404 hPa (852 labels); these use different populations from the paired comparisons.',
  'The geographic Bay subset contained 789 cases from 62 cyclones. Reductions were 11.95% for track, 13.19% for wind and 17.58% for pressure; both paired intensity counts were 656. These cases overlap the main test. Improvement was uneven: below 34 knots current wind, wind MAE was 1.536 versus 1.517 knots and pressure MAE was 0.714 versus 0.684 hPa. The aggregate gain therefore does not extend to every intensity regime.',
  'Against the RMS Gaussian, calibrated wind and pressure NLL and CRPS improved, with paired difference intervals excluding zero (Table 3). Track NLL and energy-score intervals included zero. Calibration also had different effects across outcomes: pressure CRPS fell from 1.0688 to 1.0371 hPa, while wind CRPS rose from 2.3008 to 2.3221 knots. The wind increase was 0.02124 knots, with interval 0.01441 to 0.02842. Track energy score improved relative to the uncalibrated model; all three NLL difference intervals against that model included zero. These test findings do not change the frozen forecast choice.',
  'Coverage helps locate the remaining spread error (Figure 2). Nominal 50% regions contained 64.44% of track, 67.20% of wind and 60.92% of pressure outcomes, indicating excessive central coverage. At 90%, coverage was 89.49%, 89.11% and 92.02%. Mean 90% wind and pressure widths were 16.312 knots and 6.570 hPa; mean track ellipse area was 16268.3 square kilometres. Agreement near the outer level does not establish agreement at central levels. S1 includes all declared scores, intervals and subgroup results.'
 ]},
 {'title':'Discussion and limitations','paragraphs':[
  'The selected MLP improves the aggregate six-hour physical-reference comparisons, but the experiment answers a narrower question than operational cyclone forecasting. It neither compares against operational numerical prediction nor separates the effects of predictor groups, history length and loss choice. The selection follow-up identifies a candidate; the later frozen test supplies the independent performance assessment for that candidate.',
  'The uncertainty comparison shows why coverage alone is insufficient. Intensity proper scores improve over the constant-spread reference even though central regions remain too broad. Calibration reduces pressure spread and increases wind spread, but it does not improve every score. The test provides no basis for choosing revised calibration factors. Future changes require a separate study and untouched evaluation data.',
  'Information timing is a further constraint. Best-track revisions and reanalysis latency do not match a live forecast stream, and a daily SST value can include observations after an early forecast origin. Missing intensity records and changing reporting coverage also restrict generalisation. Exact acquisition times, interpolation and some derived-feature definitions remain unresolved. Gaussian wind permits negative values and pressure is unbounded; no tail clipping was applied. Warning benefits, landfall-event skill, avoided losses and energy use were not measured.'
 ]},
 {'title':'Conclusion','paragraphs':[
  'A frozen three-member residual MLP reduced aggregate six-hour track, wind and pressure errors for later Bay-affecting cyclones against constant velocity and persistence. Separate spread learning improved intensity probability scores over the training-RMS reference, while spatial gains remained uncertain. Central overcoverage and mixed calibration effects limit the uncertainty claim. Operational evaluation needs latency-matched inputs, appropriate forecast references and new evaluation data.'
 ]}
]

newrefs=[
 {'key':'haque2026','text':'Haque MK, Miah MA, Hossain MA, Islam MA (2026) Implementation of a Convolutional Neural Network for 24-hour tropical cyclone track forecasting in the Bay of Bengal using fused historical and reanalysis data. Theoretical and Applied Climatology 157:408.','doi':'10.1007/s00704-026-06341-5'},
 {'key':'faiaz2026','text':'Faiaz A, Shaharin A, Rahman MM, Rashid MB, Hasan SMQ (2026) Physics-guided non-autoregressive transformer for lightweight cyclone track prediction in the Bay of Bengal. Expert Systems with Applications 317:131972. Author preprint: https://ssrn.com/abstract=5985599.','doi':'10.1016/j.eswa.2026.131972'},
 {'key':'tsai2026','text':'Tsai HC, Lin FY, Lin YL, Hsu NN, Huang TS, Elsberry RL (2026) Quantifying situation-dependent uncertainty in tropical cyclone track forecasts with a recurrent neural network approach. Natural Hazards 122:245.','doi':'10.1007/s11069-026-08023-x'}
]
lookup={r['key']:r for r in C['references']+newrefs}
order=[]
for section in C['sections']:
    for paragraph in section['paragraphs']:
        for group in re.findall(r'\[@([^\]]+)\]',paragraph):
            for key in group.split(';'):
                key=key.lstrip('@')
                if key not in order:order.append(key)
C['references']=[lookup[key] for key in order]

declarations=dict(C['declarations'])
declarations['Funding']='No funding was received for this work.'
declarations['Competing interests']='The authors declare no competing interests.'
declarations['Author contributions']='Individual contribution statements and final approval of the manuscript remain to be confirmed by the authors.'
declarations['AI assistance']=('OpenAI Codex assisted with literature searches, manuscript drafting and revision, and code-based preparation of documents, tables and figures from recorded methods and saved computational outputs. This assistance extended beyond copy-editing. The named authors are responsible for checking the sources, scientific interpretation and final text.')
C['declarations']=[[key,declarations[key]] for key,_ in C['declarations']]

C['extended_abstract']=[
 {'title':'Background','paragraphs':[
  'At a six-hour horizon, continuing recent motion and retaining current intensity are useful cyclone forecast references. This study evaluates how much a compact residual neural model improves these references for later Bay-affecting cyclones. A second comparison tests probability forecasts around the same fixed point mean. The study addresses retrospective track, wind and pressure prediction; operational warning benefits require separate assessment.'
 ]},
 {'title':'Data and methods','paragraphs':[
  'The derived 1982-2025 dataset combines International Best Track Archive for Climate Stewardship records, ERA5 atmospheric reanalysis and Optimum Interpolation Sea Surface Temperature. Bay training contains 4560 cases from 235 cyclones, including 2334 wind targets and 2267 pressure targets. Broader North Indian Ocean training contains 5557 cases from 289 cyclones and includes every Bay training case. Whole-storm roles separate 493 selection cases from 25 cyclones, 361 calibration cases from 25 cyclones and the 2018-2025 test of 1018 cases from 65 cyclones.',
  'Each model receives 39 standardised predictors and 39 missingness indicators. Missing future intensity labels are excluded from their losses. Four residual outputs correct east/north displacement, wind and pressure relative to constant-velocity track and persistence intensity anchors. Track corrections are decoded from a local spherical tangent plane. Original development compares snapshot, recurrent and tree models with Bay-only and broader-basin training. A bounded follow-up changed the wind loss after inspecting selection results; that stage is exploratory. Its selected predictor averages three snapshot multilayer perceptrons with hidden widths 64 and 32 and seeds 17, 29 and 43.',
  'A separate network estimates one bivariate track Gaussian and marginal intensity Gaussians around the frozen point mean. Three standard-deviation factors are fitted on reserved calibration storms. Constant training-error spread and uncalibrated learned spread provide probability references. Final comparisons retain identical eligible observations for each model-reference pair. Nominal 95% intervals use 2000 complete-cyclone bootstrap resamples.'
 ]},
 {'title':'Key results','paragraphs':[
  'Mean track error was 35.915 km against 40.753 km for constant velocity, an 11.87% reduction. Wind mean absolute error was 2.727 against 3.113 knots on 779 paired cases, a 12.41% reduction. Pressure mean absolute error was 1.385 against 1.625 hPa on 823 paired cases, a 14.74% reduction. Model-minus-reference intervals were -6.213 to -3.585 km, -0.628 to -0.162 knots and -0.450 to -0.046 hPa. All-observed wind and pressure populations contain 808 and 852 labels, respectively; they differ from the persistence-eligible pairs.',
  'Calibrated wind and pressure continuous ranked probability scores were 2.322 knots and 1.037 hPa, against 2.631 knots and 1.332 hPa for the constant-spread reference. Paired intervals favoured those intensity improvements. Track energy scores were 26.784 and 26.790 km, with a difference interval including zero. Calibration worsened wind score relative to uncalibrated spread. Nominal 50% regions contained 64.44% of track, 67.20% of wind and 60.92% of pressure outcomes. Outer coverage was closer to nominal, but that did not resolve excessive central coverage. Point improvements were also uneven across intensity regimes.'
 ]},
 {'title':'Conclusions and scope','paragraphs':[
  'The frozen residual ensemble improved aggregate six-hour track and intensity errors against declared physical references. Uncertainty learning improved intensity scores over a constant-spread Gaussian, while spatial improvement remained unclear and central regions overcovered. Reporting gaps, best-track revisions, reanalysis latency and daily SST timing limit operational interpretation. The work supplies a reproducible comparison from the derived package, with complete metrics and provenance in Supplement S1. Further model changes need untouched evaluation data; this final holdout is consumed.'
 ]}
]

words=lambda s:len(re.findall(r"\b[\w'-]+(?:\.[0-9]+)*\b",s))
print('Synopsis counts:',len(C['abstract'].split()),words(C['abstract']))
assert len(C['abstract'].split())==words(C['abstract'])==150
(HERE/'manuscript_content.json').write_text(json.dumps(C,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

confirmations={'confirmed_on':'2026-11-02','confirmed_by':'User in this conversation',
 'funding':'No funding','competing_interests':'None',
 'author_contributions':'Awaiting user details','final_author_approval':'Not yet asserted',
 'institutional_similarity_ai_thresholds':'Awaiting policy details; no numerical limit inferred',
 'fresh_institutional_reports':'Not run; requires authorised institutional checker access'}
(REV/'author_confirmations.json').write_text(json.dumps(confirmations,indent=2)+'\n',encoding='utf-8')

claims=[]
def claim(label,path,value,display):
    claims.append(dict(claim=label,evidence_file='supplement_v1/results/final_metrics.json',json_path=path,
                       saved_value=value,manuscript_display=display,population='Frozen final test unless stated otherwise'))
point=metrics['point_metrics']['full']
for key,display in [('rows','1018'),('storms','65'),('track_mean_km','35.915'),('track_median_km','28.802'),('track_p90_km','69.897'),
 ('wind_mae_kts_paired','2.727'),('wind_persistence_mae_kts_paired','3.113'),('wind_paired_labels','779'),
 ('pressure_mae_hpa_paired','1.385'),('pressure_persistence_mae_hpa_paired','1.625'),('pressure_paired_labels','823'),
 ('wind_mae_kts_all_observed','2.719'),('wind_observed_labels','808'),('pressure_mae_hpa_all_observed','1.404'),('pressure_observed_labels','852')]:
    v=point[key];assert (str(v)==display if isinstance(v,int) else f'{v:.3f}'==display),(key,v,display)
    claim(key,'point_metrics.full.'+key,v,display)
for key,display in [('track_skill_vs_constant_velocity','11.87%'),('wind_skill_vs_persistence','12.41%'),('pressure_skill_vs_persistence','14.74%')]:
    v=point[key];assert f'{100*v:.2f}%'==display;claim(key,'point_metrics.full.'+key,v,display)
for dist in ['training_RMS_diagonal_gaussian','frozen_uncalibrated_gaussian','frozen_calibrated_gaussian']:
    for key in ['track_joint_nll','track_energy_km','wind_kts_nll','wind_kts_crps','pressure_hpa_nll','pressure_hpa_crps']:
        v=metrics['probability_metrics'][dist]['full']['means'][key]
        claim(dist+' '+key,'probability_metrics.'+dist+'.full.means.'+key,v,f'{v:.4f}')
for key,v in audit['calibrated_coverage'].items():
    claim(key,'probability_metrics.frozen_calibrated_gaussian.full.means.'+key,v,f'{100*v:.2f}%')
write_csv('claim_evidence.csv',list(claims[0]),claims)

(REV/'institutional_recheck_request.md').write_text('''# Unsent institutional recheck request

Please check the revised IEC 2026 manuscript under the applicable MGIT and congress requirements. Please confirm any numerical similarity or AI-writing thresholds and the required filters.

Please retain the unfiltered similarity result and, if the approved workflow allows it, provide a second report excluding the bibliography. Record exclusions and any matches to our earlier submitted draft. Review substantive matches individually; provider names, reference titles and required acknowledgements remain accurate.

Please also provide the AI-writing report for human assessment. The manuscript declares drafting assistance. A detector score is not presented as proof of human-only writing or as a substitute for author review.

This is request text for the authors to use. No message has been sent and no manuscript has been uploaded by this revision process.
''',encoding='utf-8')
print('Prepared manuscript, similarity audit, literature matrix, author facts and claim evidence.')
