"""Build a verified local final-research handoff without rerunning model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modelling.artifacts import release_files
from modelling.final_evaluation import evaluator_fingerprint
from verify_research_archive import verify


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def build(inputs, output, review):
    inputs, output, review = Path(inputs), Path(output), Path(review)
    if output.exists():
        raise FileExistsError('Preserve the existing archive; choose a new output name')
    audit = json.loads((review/'final_test_export_verification.json').read_text(encoding='utf-8'))
    if audit['status'] != 'passed' or not all(audit['checks'].values()):
        raise ValueError('A passing independent export audit is required')
    names = {'final': 'bob_cyclone_final_results.zip', 'preflight': 'bob_cyclone_final_preflight_results.zip'}
    payloads = {}
    def add(name, payload):
        if name in payloads:
            raise ValueError('Duplicate output member: '+name)
        payloads[name] = payload
    for label, name in names.items():
        data = (inputs/name).read_bytes()
        if sha(data) != audit['archives'][label]['sha256']:
            raise ValueError('Input no longer matches passing audit: '+name)
        add('inputs/'+name, data)
    with zipfile.ZipFile(inputs/names['final']) as z:
        plan = json.loads(z.read('prepared/reviewed_plan.json'))
        if plan['evaluator_fingerprint'] != evaluator_fingerprint():
            raise ValueError('Current evaluator differs from reviewed source')
        protocol = plan['protocol']
        for name in z.namelist():
            if name.startswith('prepared/model/'):
                add(name.replace('prepared/model/', 'frozen_model/'), z.read(name))
            elif name.startswith('execution/'):
                add(name.replace('execution/', 'final_run/'), z.read(name))
        add('final_run/execution_status.json', z.read('execution_status.json'))
    calibrated = (inputs/'uncertainty_v1_results (3).zip').read_bytes()
    if sha(calibrated) != protocol['calibrated_export_sha256']:
        raise ValueError('Calibrated export differs from frozen protocol')
    add('inputs/uncertainty_v1_results (3).zip', calibrated)
    code = (ROOT/'dist/bob_cyclone_final_evaluation_code.zip').read_bytes()
    if sha(code) != 'a3aa7c1a2df88b678c64e3f20585cf3d3a04bfcca8685b38d2d55b675198fee4':
        raise ValueError('Exact reviewed final-evaluation transfer bundle is required')
    add('inputs/bob_cyclone_final_evaluation_code.zip', code)
    data = (ROOT/'data/packages/bob_cyclone_training_v2.zip').read_bytes()
    if sha(data) != protocol['dataset_sha256']:
        raise ValueError('Immutable training package differs from reviewed protocol')

    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                          cwd=ROOT, capture_output=True, text=True)
    log = test.stdout+test.stderr
    (review/'software_tests.log').write_text(log, encoding='utf-8')
    match = re.search(r'Ran (\d+) tests? in ', log)
    test_receipt = {'status': 'passed' if test.returncode == 0 and match else 'failed',
                    'command': 'python -m unittest discover -s tests -v',
                    'tests_run': int(match.group(1)) if match else None,
                    'scope': 'Software and synthetic regression tests; not a new scientific test evaluation',
                    'evaluator_fingerprint_matches_reviewed_plan': True}
    (review/'software_tests.json').write_text(json.dumps(test_receipt, indent=2)+'\n', encoding='utf-8')
    if test_receipt['status'] != 'passed':
        raise RuntimeError('Software regression checks failed; inspect review log')
    for path in release_files(ROOT):
        add('source/'+path.relative_to(ROOT).as_posix(), path.read_bytes())
    for name in ['final_test_export_verification.json', 'final_test_coverage.png', 'software_tests.log', 'software_tests.json']:
        add('evidence/'+name, (review/name).read_bytes())
    add('verify_archive.py', (ROOT/'scripts/verify_research_archive.py').read_bytes())
    add('README.md', b'''# Bay of Bengal Cyclone AI - frozen research handoff v1

The final retrospective holdout evaluation is complete. Read source/docs/final-results.md,
source/docs/model-card.md and source/docs/reproducibility-archive.md.

Verify this extracted archive with: python verify_archive.py .
For a ZIP check, use: python verify_archive.py /path/to/archive.zip

Original input exports and the exact reviewed code bundle are under inputs/.
Frozen model files are under frozen_model/. Completed results, plan, cached rows
and ledger receipts are under final_run/. Independent audit and software-test
evidence are under evidence/. source/ is the accompanying workspace snapshot;
the exact historical inference source is separately hash-bound in the exports.

The final test is consumed. Reproduce saved-score calculations only; do not use
these results to select or recalibrate a model. No operational warning-use skill
is claimed. This archive is a local handoff, not a newly published GitHub release.
Provider terms remain in source/NOTICE.md and source/data/README.md.
''')
    manifest = {'archive_id': 'bob_cyclone_research_handoff_v1', 'date': '2026-10-30',
                'test_status': 'completed_and_consumed_no_further_tuning',
                'reviewed_plan_sha256': audit['reviewed_plan_sha256'],
                'frozen_record_digest': protocol['frozen_record_digest'],
                'calibration_file_sha256': protocol['calibration_file_sha256'],
                'software_tests': test_receipt,
                'source_snapshot_note': 'Includes documentation and audit/archive tooling added after the reviewed evaluation. Evaluation sources are checked against the reviewed fingerprint; inference uses its separate original source freeze.',
                'files': {n: {'sha256': sha(b), 'bytes': len(b)} for n,b in sorted(payloads.items())}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    if temporary.exists():
        raise FileExistsError('Temporary archive exists; preserve it and choose another output')
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, payload in sorted(payloads.items()):
            entry = zipfile.ZipInfo(name, (2026,10,30,0,0,0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(entry, payload)
        z.writestr('ARCHIVE_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    verified = verify(temporary)
    temporary.replace(output)
    checksum = sha(output.read_bytes())
    output.with_suffix(output.suffix+'.sha256').write_text(checksum+'  '+output.name+'\n', encoding='utf-8')
    result = dict(verified, archive=output.name, sha256=checksum, bytes=output.stat().st_size,
                  software_tests=test_receipt, original_dataset_decoded=False, model_inference_rerun=False)
    (review/'archive_build_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-dir', type=Path, default=Path.home()/'Downloads')
    parser.add_argument('--output', type=Path, default=ROOT/'dist/bob_cyclone_research_archive_v1.zip')
    parser.add_argument('--review-dir', type=Path, default=ROOT/'reports/final_archive_review')
    args = parser.parse_args()
    print(json.dumps(build(args.inputs_dir, args.output, args.review_dir), indent=2))
