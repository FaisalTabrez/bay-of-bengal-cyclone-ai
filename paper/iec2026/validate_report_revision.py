"""Verify the document revision against frozen saved evidence, without modelling."""
from pathlib import Path
import csv
import hashlib
import json
import re
from docx import Document

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REV=HERE/'revision_2026-11-02'
BASE=HERE/'qa/reports_2026-11-02/baseline'
C=json.loads((HERE/'manuscript_content.json').read_text())
M=json.loads((HERE/'supplement_v1/results/final_metrics.json').read_text())
selection=json.loads((HERE/'supplement_v1/results/deterministic_decision.json').read_text())
calibration=json.loads((HERE/'supplement_v1/results/calibration.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]

for name,digest in json.loads((BASE/'BASELINE_MANIFEST.json').read_text()).items():
    assert sha(BASE/name)==digest,name
checks.append('Preserved baseline and original report checksums')
identities=json.loads((REV/'unchanged_scientific_files.json').read_text())
for name,digest in identities.items():assert sha(ROOT/name)==digest,name
checks.append('All recorded scientific source, dataset and saved-result identities unchanged')

rows=list(csv.DictReader((REV/'claim_evidence.csv').open(encoding='utf-8')))
def extra(label,path,value,display,file='supplement_v1/results/final_metrics.json',population='Frozen final test'):
    rows.append(dict(claim=label,evidence_file=file,json_path=path,saved_value=value,manuscript_display=display,population=population))
def get(path):
    value=M
    for key in path.split('.'):value=value[key]
    return value
def record(path,display,mode='decimal',digits=3):
    value=get(path)
    actual=str(value) if mode=='integer' else f'{100*value:.2f}%' if mode=='percent' else f'{value:.{digits}f}'
    assert actual==display,(path,actual,display)
    extra(path,path,value,display)

record('physical_references.constant_velocity_and_tendency.track_all.mean_km','40.753')
for key in ['paired_track_error_difference_km','paired_wind_error_difference_kts','paired_pressure_error_difference_hpa']:
    for bound in ['low','high']:
        path='point_metrics.full.bootstrap.intervals.'+key+'.'+bound
        value=get(path);extra(path,path,value,f'{value:.3f}')
for key,value in [('rows',789),('storms',62),('wind_paired_labels',656),('pressure_paired_labels',656)]:
    record('point_metrics.geographic_bay.'+key,str(value),'integer')
for key,value in [('track_skill_vs_constant_velocity','11.95%'),('wind_skill_vs_persistence','13.19%'),('pressure_skill_vs_persistence','17.58%')]:
    record('point_metrics.geographic_bay.'+key,value,'percent')
for key,value in [('wind_mae_kts_paired','1.536'),('wind_persistence_mae_kts_paired','1.517'),('pressure_mae_hpa_paired','0.714'),('pressure_persistence_mae_hpa_paired','0.684')]:
    record('point_metrics.current_intensity_diagnostic.under_34_kt.'+key,value)
for key,display,digits in [('wind_kts_width_90','16.312',3),('pressure_hpa_width_90','6.570',3),('track_area_km2_90','16268.3',1)]:
    record('probability_metrics.frozen_calibrated_gaussian.full.means.'+key,display,digits=digits)
diff='paired_probability_differences_calibrated_minus_reference.frozen_uncalibrated_gaussian.full.'
record(diff+'means.wind_kts_crps','0.02124',digits=5)
# The inherited manuscript uses conservative 0.01441 at the lower endpoint;
# saved value rounds to 0.01441 under five-decimal rounding.
record(diff+'bootstrap.intervals.wind_kts_crps.low','0.01441',digits=5)
record(diff+'bootstrap.intervals.wind_kts_crps.high','0.02842',digits=5)
for candidate,display in [('original_bay_mlp','0.97305'),('bay_mlp_wind_persistence','0.94253'),('bay_mlp_wind_absolute_loss','0.92667')]:
    value=selection['candidate_mean_seed_scores'][candidate]
    assert f'{value:.5f}'==display
    extra(candidate,'candidate_mean_seed_scores.'+candidate,value,display,
          file='supplement_v1/results/deterministic_decision.json',population='Reused selection role; exploratory follow-up')
for index,display in enumerate(['0.94898','1.05730','0.80831']):
    value=calibration['scale_factors_track_wind_pressure'][index]
    assert f'{value:.5f}'==display
    extra('Calibration scale '+str(index),'scale_factors_track_wind_pressure.'+str(index),value,display,
          file='supplement_v1/results/calibration.json',population='Reserved calibration role')

# Check every mapped numeric value against its exact saved path.
for row in rows:
    evidence=json.loads((HERE/row['evidence_file']).read_text())
    for key in row['json_path'].split('.'):
        evidence=evidence[int(key)] if isinstance(evidence,list) else evidence[key]
    assert float(evidence)==float(row['saved_value']),row['claim']
unique={(r['evidence_file'],r['json_path']):r for r in rows}
with (REV/'claim_evidence.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(unique.values())
checks.append('Mapped result, population, interval, selection and calibration values verified')

seen=[]
for section in C['sections']:
    for paragraph in section['paragraphs']:
        for group in re.findall(r'\[@([^\]]+)\]',paragraph):
            for key in group.split(';'):
                key=key.lstrip('@')
                if key not in seen:seen.append(key)
assert seen==[r['key'] for r in C['references']]
assert len(seen)==12
checks.append('Twelve references complete and numbered in first-citation order')

current=Document(HERE/'drafts/IEC2026_Paper_Draft.docx')
baseline=Document(BASE/'drafts/IEC2026_Paper_Draft.docx')
assert len(current.tables)==len(baseline.tables)==3
for actual,old in zip(current.tables,baseline.tables):
    assert [[c.text for c in row.cells] for row in actual.rows]==[[c.text for c in row.cells] for row in old.rows]
assert len(current.inline_shapes)==4  # two equations and two figures
assert 'Dr. T. V. Rajinikanth' in '\n'.join(p.text for p in current.paragraphs)
declarations=dict(C['declarations'])
assert declarations['Funding']=='No funding was received for this work.'
assert declarations['Competing interests']=='The authors declare no competing interests.'
assert 'remain to be confirmed' in declarations['Author contributions']
assert 'extended beyond copy-editing' in declarations['AI assistance']
checks.append('Tables identical to baseline; author title and truthful declaration states retained')
counts=json.loads((HERE/'draft_checks.json').read_text())
native=json.loads((HERE/'qa/word_native_counts.json').read_text(encoding='utf-8-sig'))
assert counts['synopsis_words']==counts['synopsis_whitespace_words']==150
assert counts['extended_abstract_words_conservative']<=1000
assert counts['paper_words_including_references_and_declarations']<=3000
assert native['IEC2026_Paper_Draft.docx']<=3000
checks.append('Synopsis, extended abstract and manuscript limits verified by stated counts')
visual_path=HERE/'qa/revision_visual_review.json'
visual=json.loads(visual_path.read_text()) if visual_path.exists() else {'all_final_pages_inspected':False}
if visual.get('all_final_pages_inspected'):
    for name,digest in visual['reviewed_file_sha256'].items():assert sha(HERE/name)==digest,name
    checks.append('Rendered final deliverables visually reviewed and bound by checksums')
report=dict(status='passed',checked_on='2026-11-02',checks=checks,
            scientific_files_unchanged=len(identities),mapped_evidence_values=len(unique),
            conservative_counts=counts,word_native_counts=native,visual_review=visual,
            new_training=False,new_inference=False,new_calibration=False,original_test_decoded=False,
            institutional_detector_recheck_performed=False,final_author_approval_asserted=False,
            pending=['Individual author contributions','Final technical review and author approval',
                     'Institutional thresholds/filters and fresh reports','Full-text comparison details where access-limited'])
(REV/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
