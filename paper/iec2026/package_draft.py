"""Package reviewed drafts and numerical supplement; exclude weights and QA files."""
from pathlib import Path
import hashlib
import json
import zipfile

root=Path(__file__).resolve().parent
companion=Path(json.loads((root/'qa/latest_companion.json').read_text())['path'])
files=[
    'README.md','submission_notes.md','submission_email_draft.md','revision_response.md',
    'manuscript_content.json','paper_draft.md','draft_checks.json',
    'drafts/IEC2026_Extended_Abstract.docx',
    'drafts/IEC2026_Paper_Draft.docx',
    str(companion.relative_to(root)).replace('\\','/'),
    'drafts/IEC2026_Springer_Draft.pdf',
    'springer/main.tex','springer/sn-jnl.cls','springer/sn-mathphys-num.bst',
    'springer/figures/workflow.pdf','springer/figures/workflow.png',
    'springer/figures/coverage.pdf','springer/figures/coverage.png',
    'sources/IEI_Guidelines_for_Authors.pdf','sources/IEI_Declaration_Form.doc'
]
files += [str(p.relative_to(root)).replace('\\','/') for p in sorted((root/'supplement_v1').rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
files += [str(p.relative_to(root)).replace('\\','/') for p in sorted((root/'revision_2026-11-02').glob('*')) if p.is_file()]
for name in files:assert (root/name).is_file(),name
out=root/'IEC2026_Draft_Package.zip'
manifest={name:{'bytes':(root/name).stat().st_size,
                'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in files}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for name in files:z.write(root/name,name)
    z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
    for name,expected in manifest.items():
        data=z.read(name)
        assert len(data)==expected['bytes']
        assert hashlib.sha256(data).hexdigest()==expected['sha256']
qa=json.loads((root/'qa/layout_checks.json').read_text())
qa['visual_review']={
    'reviewed_on':'2026-11-02',
    'all_final_pages_inspected':True,
    'word_extended_abstract_pages':qa['IEC2026_Extended_Abstract.pdf']['pages'],
    'word_paper_pages':qa['IEC2026_Paper_Draft.pdf']['pages'],
    'springer_pdf_pages':qa['IEC2026_Springer_Draft.pdf']['pages'],
    'resolved_defects':['Removed inherited title border and theme font overrides',
                        'Shortened workflow labels to fit boxes',
                        'Prevented reference entry splitting in Springer PDF',
                        'Kept short Word results tables on one page'],
    'model_results_held':False,'completed_final_results_incorporated':True,
    'review_revision':'Revision after supplied similarity and AI reports; expanded related work, evidence-bound interpretation, confirmed funding/conflicts and explicit AI drafting disclosure.',
    'external_submission_performed':False,
    'word_renderer':'Microsoft Word export then PNG inspection; canonical LibreOffice renderer unavailable',
    'springer_compiler':'Bundled Tectonic'
}
(root/'qa/final_review.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps({'package':str(out),'bytes':out.stat().st_size,'files_verified':len(manifest),
                  'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))
