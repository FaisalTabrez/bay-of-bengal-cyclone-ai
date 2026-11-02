"""Create the figures and tables companion from the current verified paper."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

HERE=Path(__file__).resolve().parent
source=HERE/'drafts/IEC2026_Paper_Draft.docx'
out=HERE/'drafts/IEC2026_Figures_and_Tables_Updated.docx'
original=Document(source)
assert len(original.tables)==3
tables=[deepcopy(t._tbl) for t in original.tables]
captions={}
for p in original.paragraphs:
    for key in ['Table 1.','Table 2.','Table 3.','Figure 1.','Figure 2.']:
        if p.text.startswith(key) and key not in captions:
            captions[key]=p.text
assert len(captions)==5

d=Document(source)
for node in list(d._element.body):
    if node.tag!=qn('w:sectPr'): d._element.body.remove(node)
section=d.sections[0]
section.top_margin=Mm(20);section.bottom_margin=Mm(20)
section.left_margin=Mm(22);section.right_margin=Mm(22)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
    style=d.styles[name]
    style.font.name='Times New Roman';style.font.color.rgb=RGBColor(0,0,0)
    style.paragraph_format.line_spacing=1.15
    style.paragraph_format.space_after=Pt(6)
d.styles['Normal'].font.size=Pt(11)
d.styles['Title'].font.size=Pt(17)
d.styles['Title'].paragraph_format.space_after=Pt(10)
d.styles['Heading 1'].font.size=Pt(13)
d.styles['Caption'].font.size=Pt(10)
d.styles['Caption'].paragraph_format.line_spacing=1.1
d.core_properties.title='Cyclone Prediction Figures and Tables'
d.core_properties.subject='Figures and tables for the IEC 2026 cyclone prediction paper'

def caption(key):
    p=d.add_paragraph(captions[key],'Caption')
    return p

def picture(filename,width):
    p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next=True
    p.add_run().add_picture(str(HERE/'springer/figures'/filename),width=Mm(width))

def table(index):
    p=caption(f'Table {index+1}.');p.paragraph_format.keep_with_next=True
    d._element.body.insert(len(d._element.body)-1,tables[index])
    for row in d.tables[-1].rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing=1.15
                p.paragraph_format.space_before=Pt(3)
                p.paragraph_format.space_after=Pt(3)
                for r in p.runs:r.font.size=Pt(10)

d.add_paragraph('Cyclone Prediction Figures and Tables','Title')
d.add_paragraph('Companion to Machine Learning for Cyclone Prediction in the Bay of Bengal')
author_names=json.loads((HERE/'manuscript_content.json').read_text(encoding='utf-8'))['authors']
d.add_paragraph(', '.join(author_names[:-1])+' and '+author_names[-1])
d.add_paragraph('The paper\'s two figures and three editable tables are collected here with their original numbers. They describe the frozen six-hour prediction study, the eligible comparison populations and the remaining uncertainty errors.')
d.add_heading('Model workflow',1)
picture('workflow.png',163);caption('Figure 1.')
d.add_heading('Technical overview',1)
d.add_paragraph('The selected point predictor is an ensemble of three snapshot multilayer perceptrons (MLPs), with seeds 17, 29 and 43. Each model has a 78-64-32-4 architecture: 39 standardised predictor values and 39 missingness indicators feed two hidden layers and four residual outputs. The predictors include current cyclone state, recent tendencies and environmental conditions.')
d.add_paragraph('The four outputs correct eastward and northward track displacement, wind and pressure relative to constant-velocity track and persistence intensity anchors. Track correction uses a local spherical tangent plane before decoding back to latitude and longitude. Missing future intensity labels are masked during training. The chosen model uses absolute wind loss; the original GRU remains a development comparison.')
d.add_paragraph('The three point forecasts are averaged. A separate 78-32-5 MLP estimates four standard deviations and one track correlation around that fixed mean. Three spread multipliers, fitted on reserved calibration storms, scale track, wind and pressure uncertainty without changing the point predictions.')

d.add_page_break()
d.add_heading('Data populations and point predictions',1)
table(0)
d.add_paragraph('NIO denotes the North Indian Ocean. Its training population includes the Bay training rows. Wind and pressure columns count observed future labels; a persistence comparison additionally requires an observed current value.')
table(1)
d.add_paragraph('The final population contains 1018 observations from 65 cyclones in 2018-2025. Track error is a mean great-circle distance; intensity errors are mean absolute errors. Lower values are better. Wind and pressure comparisons use the same eligible observations for the model and persistence.')
d.add_paragraph('Nominal 95% paired model-minus-reference intervals are -6.213 to -3.585 km for track, -0.628 to -0.162 knots for wind and -0.450 to -0.046 hPa for pressure. They use 2000 whole-cyclone bootstrap resamples. No multiplicity adjustment is claimed.')
d.add_paragraph('All-observed-label wind and pressure errors are 2.719 knots across 808 labels and 1.404 hPa across 852 labels. These populations differ from the paired comparisons in Table 2. Aggregate improvement does not imply improvement in every intensity regime.')

d.add_page_break()
d.add_heading('Probability scores and coverage',1)
table(2)
d.add_paragraph('NLL is negative log likelihood; CRPS is continuous ranked probability score. The RMS reference uses frozen in-sample training-error scales. All three distributions share identical point means. Lower scores are better; NLL values use native units and are compared within the same outcome.')
picture('coverage.png',145);caption('Figure 2.')
d.add_paragraph('Calibration reduced pressure CRPS from 1.0688 to 1.0371 hPa but increased wind CRPS from 2.3008 to 2.3221 knots. Against the RMS reference, intensity scores improved while track-score difference intervals included zero. The 50% regions cover too many outcomes even though the 90% regions are closer to nominal.')
d.add_paragraph('These are retrospective results using best-track and reanalysis-derived inputs. Operational warning skill requires verified source availability and a separate operational evaluation. Complete metrics, intervals and provenance are retained in Supplement S1 v1.1.')

# Tables retain the exact source values and remain editable.
for actual,expected in zip(d.tables,original.tables):
    assert [[c.text for c in r.cells] for r in actual.rows]==[[c.text for c in r.cells] for r in expected.rows]
assert len(d.inline_shapes)==2 and len(d.tables)==3
try:
    d.save(out)
except PermissionError:
    # Preserve an open user document and deliver a separate updated copy.
    out=out.with_name('IEC2026_Figures_and_Tables_Revised_2026-11-02.docx')
    d.save(out)
record={'output':str(out),'source_paper_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'tables':3,'figures':2,'table_values_identical_to_paper':True,
        'training_inference_or_calibration_performed':False}
(HERE/'qa/figures_tables_checks.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
(HERE/'qa/latest_companion.json').write_text(json.dumps({'path':str(out)},indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
