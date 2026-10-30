"""Prepare a reviewable final protocol; decode the sealed test only after its gate.

TrainingPackage.load('bob_test') remains forbidden. The dedicated decoder below
is called only after a persistent exclusive ledger claim and reviewed-plan match.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
import uuid
import zipfile

import numpy as np
import pandas as pd

from .data import TrainingPackage, validation_roles
from .experiment_io import atomic_write, digest, mirror_run, source_hashes, write_json
from .final_scores import score_all
from .geometry import log_map
from .paths import DEFAULT_PACKAGE, ROOT
from .wind_audit import ResultsArchive

CONFIG_PATH = Path(__file__).parent/'configs/final_evaluation_v1.json'
SOURCE_ARCHIVE = ROOT/'data/frozen_runtime/uncertainty_v1_source.zip'
WORKER = ROOT/'scripts/final_inference_worker.py'


def file_receipt(path):
    path = Path(path)
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def verify_files(root, expected):
    root = Path(root).resolve()
    for name, receipt in expected.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_receipt(path) != receipt:
            raise ValueError('Final artifact integrity failure: '+name)


def protocol_config():
    return json.loads(CONFIG_PATH.read_text())


def evaluator_fingerprint():
    return {'package_sources': source_hashes(), 'inference_worker_sha256': file_receipt(WORKER)['sha256']}


def calibrated_results(path, protocol=None):
    protocol = protocol or protocol_config()
    if file_receipt(path)['sha256'] != protocol['calibrated_export_sha256']:
        raise ValueError('Use the reviewed calibrated uncertainty_v1_results (3).zip; the ZIP checksum differs')
    archive = ResultsArchive(path)
    required = ['research/frozen_model.json', 'research/calibration/calibration.json', 'research/calibration/status.json']
    if not set(required).issubset(archive.manifest):
        raise ValueError('Upload the completed calibrated research export, not source or acceptance results')
    frozen, calibration, status = (archive.read(name) for name in required)
    if (digest(frozen) != protocol['frozen_record_digest'] or digest(frozen) != calibration['frozen_model_sha256']
            or status['frozen_model_sha256'] != digest(frozen) or status['status'] != 'completed'
            or calibration['status'] != 'calibrated_once'
            or archive.manifest[required[1]]['sha256'] != protocol['calibration_file_sha256']
            or status['calibration_sha256'] != protocol['calibration_file_sha256']
            or digest(frozen['source_sha256']) != protocol['inference_source_map_digest']
            or frozen['package_sha256'] != protocol['dataset_sha256']
            or frozen['calibration_labels_used'] or frozen['test_parquet_decoded']
            or calibration['means_changed'] or calibration['test_parquet_decoded']):
        raise ValueError('Model/calibration/source/data receipt differs from the reviewed anchors')
    for name, expected in frozen['files'].items():
        if archive.manifest.get('research/'+name) != expected:
            raise ValueError('Frozen core artifact differs from the results manifest')
    factors = np.asarray(calibration['scale_factors_track_wind_pressure'], float)
    if factors.shape != (3,) or not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError('Invalid fixed spread factors')
    return archive, frozen, calibration


def _source_member(name):
    return name if name.startswith(('tests/', 'requirements/')) or name == 'pyproject.toml' else 'modelling/'+name


def _copy_original_source(destination, expected, protocol):
    if file_receipt(SOURCE_ARCHIVE)['sha256'] != protocol['inference_source_archive_sha256']:
        raise ValueError('Frozen inference source archive differs')
    mapping = {_source_member(name): checksum for name, checksum in expected.items()}
    with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(mapping):
            raise ValueError('Frozen source snapshot file set differs')
        for name, checksum in mapping.items():
            payload = archive.read(name)
            path = (destination/name).resolve()
            if not path.is_relative_to(destination.resolve()) or hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError('Frozen inference source integrity failure')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)


def _worker(prepared, frame, directory):
    prepared, directory = Path(prepared), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((prepared/'model/frozen_model.json').read_text())
    targets = ['TARGET_LAT_6H', 'TARGET_LON_6H', 'TARGET_WIND_KTS_6H', 'TARGET_PRESSURE_HPA_6H']
    inputs = frame.drop(columns=[name for name in targets if name in frame])
    input_path, output_path, metadata_path = directory/'inputs.parquet', directory/'parameters.npz', directory/'inference.json'
    inputs.to_parquet(input_path, index=False)
    request = {'runtime_root': str((prepared/'runtime').resolve()), 'frozen_directory': str((prepared/'model').resolve()),
               'expected_source_hashes': frozen['source_sha256'], 'expected_frozen_digest': digest(frozen),
               'inputs': str(input_path.resolve()), 'output': str(output_path.resolve()), 'metadata': str(metadata_path.resolve())}
    write_json(directory/'request.json', request)
    result = subprocess.run([sys.executable, str((prepared/'worker.py').resolve()), str((directory/'request.json').resolve())],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
    (directory/'worker.log').write_text(result.stdout, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Frozen inference failed; inspect worker.log: '+result.stdout[-1600:])
    metadata = json.loads(metadata_path.read_text())
    if metadata['input_sha256'] != file_receipt(input_path)['sha256'] or metadata['input_rows'] != len(frame) or metadata['target_columns_received']:
        raise ValueError('Inference input identity/label isolation check failed')
    with np.load(output_path, allow_pickle=False) as stored:
        parameters = {name: stored[name] for name in stored.files}
    for name in ['point', 'mean', 'std_calibrated', 'std_uncalibrated']:
        if parameters[name].shape != (len(frame), 4) or not np.isfinite(parameters[name]).all():
            raise ValueError('Invalid frozen inference output')
    if parameters['rho'].shape != (len(frame),) or not np.isfinite(parameters['rho']).all() or (np.abs(parameters['rho']) >= 1).any():
        raise ValueError('Invalid track correlation')
    if (parameters['std_calibrated'] <= 0).any() or (parameters['std_uncalibrated'] <= 0).any():
        raise ValueError('Non-positive Gaussian spread')
    return parameters, metadata


def prepare(results, output, package_path=DEFAULT_PACKAGE):
    """No test decoder is reachable in preparation; reproduction uses selection inputs."""
    output, protocol = Path(output), protocol_config()
    if output.exists():
        if not (output/'reviewed_plan.json').is_file():
            raise FileExistsError('Preparation is incomplete; test remains sealed. Preserve this folder and use a fresh preparation directory')
        return verify_prepared(output)
    archive, frozen, calibration = calibrated_results(results, protocol)
    package = TrainingPackage(package_path)
    package.verify_checksums()  # Raw ZIP integrity only; no test Parquet decoding.
    if file_receipt(package.path)['sha256'] != protocol['dataset_sha256']:
        raise ValueError('Frozen dataset differs')
    output.mkdir(parents=True)
    with zipfile.ZipFile(results) as source:
        names = list(frozen['files'])+['frozen_model.json', 'calibration/calibration.json', 'calibration/status.json']
        for name in names:
            target = (output/'model'/name).resolve()
            if not target.is_relative_to((output/'model').resolve()):
                raise ValueError('Unsafe model member path')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read('research/'+name))
    _copy_original_source(output/'runtime', frozen['source_sha256'], protocol)
    shutil.copyfile(WORKER, output/'worker.py')
    validation = package.load('bob_validation')
    selection = validation.loc[validation_roles(validation) == 'selection'].reset_index(drop=True)
    parameters, metadata = _worker(output, selection, output/'selection_reproduction')
    original = archive.read('research/selection_distribution.csv')
    if original.SID.tolist() != selection.SID.tolist() or pd.to_datetime(original.ISO_TIME, utc=True).tolist() != selection.ISO_TIME.tolist():
        raise ValueError('Archived selection prediction identity differs')
    point_columns = ['PRED_LAT_6H', 'PRED_LON_6H', 'PRED_WIND_KTS_6H', 'PRED_PRESSURE_HPA_6H']
    differences = np.max(np.abs(parameters['point']-original[point_columns].to_numpy()), axis=0)
    if (differences > protocol['point_reproduction_tolerances_lat_lon_wind_pressure']).any():
        raise ValueError('Frozen means did not reproduce archived forecasts')
    native_std = original[['STD_EAST_KM', 'STD_NORTH_KM', 'STD_WIND_KTS', 'STD_PRESSURE_HPA']].to_numpy()
    factors = np.asarray(calibration['scale_factors_track_wind_pressure'])[[0, 0, 1, 2]]
    if not np.allclose(parameters['std_calibrated'], native_std*factors, atol=protocol['spread_reproduction_absolute_tolerance_native_units'], rtol=0):
        raise ValueError('Calibrated spread reproduction failed')
    training_summary = archive.read('research/training_summary.json')
    codec = json.loads((output/'model/mean/codec.json').read_text())
    reference_std = np.asarray(training_summary['initial_std_normalized'])*np.asarray(codec['residual_scales_km_km_kts_hpa'])
    write_json(output/'preflight.json', {'status': 'passed', 'selection_rows_checked': len(selection),
                                      'max_point_reproduction_difference': differences.tolist(),
                                      'test_parquet_decoded': False, 'source_is_original_snapshot': True,
                                      'software_reproduction_only_not_new_model_selection': True})
    # Requests/logs are operational details; all scientific inputs below are bound.
    files = [p for folder in ['model', 'runtime'] for p in (output/folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    files += [output/'worker.py', output/'preflight.json', output/'selection_reproduction/inputs.parquet',
              output/'selection_reproduction/parameters.npz', output/'selection_reproduction/inference.json']
    plan = {'status': 'prepared_for_review_test_still_sealed', 'protocol': protocol,
            'evaluator_fingerprint': evaluator_fingerprint(), 'inference_runtime': metadata['runtime'],
            'frozen_record_digest': digest(frozen), 'calibration_file_sha256': protocol['calibration_file_sha256'],
            'files': {p.relative_to(output).as_posix(): file_receipt(p) for p in files},
            'training_RMS_reference_std_native': reference_std.tolist(),
            'known_gaussian_limitations': 'Calibration central intervals over-cover; calibration NLL improvement did not uniformly improve CRPS',
            'test_parquet_decoded': False, 'calibration_refit': False, 'model_training': False}
    write_json(output/'reviewed_plan.json', plan)
    (output/'REVIEW.md').write_text(
        '# Final test plan: review before enabling\n\n'
        f'Plan SHA-256 (canonical JSON): `{digest(plan)}`\n\n'
        'Preparation has not decoded the test. The original model, calibration receipt, source snapshot, scoring code and runtime are bound.\n\n'
        '| Frozen input | Value |\n|---|---|\n'
        f'| Predictor | Three-seed absolute-wind-loss Bay MLP with frozen Gaussian spread |\n'
        f'| Model digest | `{protocol["frozen_record_digest"]}` |\n'
        f'| Calibration file SHA-256 | `{protocol["calibration_file_sha256"]}` |\n'
        f'| Dataset SHA-256 | `{protocol["dataset_sha256"]}` |\n'
        f'| Inference source snapshot SHA-256 | `{protocol["inference_source_archive_sha256"]}` |\n'
        f'| Evaluation source fingerprint | `{digest(plan["evaluator_fingerprint"])}` |\n'
        f'| Runtime | CPU, Python {metadata["runtime"]["python"]}, PyTorch {metadata["runtime"]["packages"]["torch"]}, two threads |\n\n'
        'Expected test: 1,018 observations, 65 storms, input years 2018–2025 (handoff metadata).\n\n'
        'Report the frozen point model against persistence and constant velocity/tendency; report calibrated, uncalibrated and frozen training-RMS Gaussian forecasts.\n\n'
        'Use paired observed-label denominators, full and fixed geographic-Bay cohorts, 2,000 whole-cyclone bootstrap replicates, and 50/80/90% coverage levels.\n\n'
        'Probability scores are native-unit Gaussian NLL, marginal CRPS and spatial energy score with 512 fixed common-random-number draws. No latitude/longitude density is claimed.\n\n'
        'Known calibration central-coverage mismatch remains a model limitation; the test does not choose another distribution or scale.\n\n'
        'A persistent dataset-level ledger is claimed before decoding. Resume uses the same checked cache; a new directory does not authorize another test read.\n\n'
        'After the final test, report the findings without model, scale, threshold or feature changes on this holdout.\n', encoding='utf-8')
    return plan


def verify_prepared(directory):
    directory = Path(directory)
    plan = json.loads((directory/'reviewed_plan.json').read_text())
    if plan['protocol'] != protocol_config() or plan['evaluator_fingerprint'] != evaluator_fingerprint() or plan['test_parquet_decoded']:
        raise ValueError('Prepared plan/source/config differs; no test access is permitted')
    verify_files(directory, plan['files'])
    return plan


def ledger_key(protocol):
    # Independent of model/config/run-id, so another candidate cannot reset this holdout.
    return digest({'study': protocol['study_id'], 'dataset': protocol['dataset_sha256'], 'partition': protocol['partition']})


def _update_ledger(path, value):
    write_json(path, value)


def _decode_test(package, protocol, permit):
    if permit.get('status') != 'claimed' or not permit.get('reviewed_plan_sha256') or not permit.get('attempt_id'):
        raise PermissionError('A reviewed plan and persistent one-time claim are required before test decoding')
    dataset_bytes = package.path.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != protocol['dataset_sha256']:
        raise ValueError('Dataset changed after review; no test rows will be decoded')
    with zipfile.ZipFile(io.BytesIO(dataset_bytes)) as archive:
        names = [n for n in archive.namelist() if n.startswith('bob_test/') and n.endswith('.parquet')]
        if len(names) != 1:
            raise ValueError('Expected exactly one immutable test Parquet member')
        frame = pd.read_parquet(io.BytesIO(archive.read(names[0])))
    expected = package.manifest['identifier_columns']+package.features+package.targets
    if list(frame.columns) != expected or len(frame) != protocol['expected_rows'] or frame.SID.nunique() != protocol['expected_storms']:
        raise ValueError('Test schema/count/storm contract failed; no rows may be selectively dropped')
    frame = frame.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)
    if frame.duplicated(['SID', 'ISO_TIME']).any() or not frame.TARGET_TIME_6H.sub(frame.ISO_TIME).eq(pd.Timedelta(hours=6)).all():
        raise ValueError('Test observation keys or horizon differs')
    if not frame.ISO_TIME.dt.year.between(*protocol['input_years']).all() or frame.SPLIT.nunique() != 1:
        raise ValueError('Test chronology/partition contract failed')
    if frame[package.targets[:2]].isna().any().any() or not frame.SST_AVAILABLE.eq(frame.SST_C.notna().astype(int)).all():
        raise ValueError('Test target or availability contract failed')
    for column in package.features+package.targets:
        if np.isinf(frame[column].to_numpy(float)).any():
            raise ValueError('Infinite test feature or target')
    known = set(package.load('nio_pretrain_train').SID)|set(package.load('bob_validation').SID)
    if known & set(frame.SID):
        raise ValueError('Test cyclone overlaps training/validation')
    return frame


def inspect_result(directory):
    directory = Path(directory)
    receipt = json.loads((directory/'result_receipt.json').read_text())
    verify_files(directory, receipt['files'])
    return json.loads((directory/'final_metrics.json').read_text())


def evaluate(prepared, output, ledger_directory, reviewed_plan_sha256, resume=False, mirror_directory=None,
             package_path=DEFAULT_PACKAGE):
    prepared, output, ledger_directory = Path(prepared), Path(output), Path(ledger_directory)
    plan = verify_prepared(prepared)
    plan_hash, protocol = digest(plan), plan['protocol']
    if reviewed_plan_sha256 != plan_hash:
        raise PermissionError('Read REVIEW.md and supply the exact reviewed plan SHA-256; test remains sealed')
    if (output.resolve().is_relative_to(prepared.resolve()) or prepared.resolve().is_relative_to(output.resolve())
            or ledger_directory.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(ledger_directory.resolve())
            or ledger_directory.resolve().is_relative_to(prepared.resolve()) or prepared.resolve().is_relative_to(ledger_directory.resolve())):
        raise ValueError('Prepared source, result directory and persistent ledger must be separate')
    package = TrainingPackage(package_path)
    if file_receipt(package.path)['sha256'] != protocol['dataset_sha256']:
        raise ValueError('Test dataset hash differs')
    ledger_directory.mkdir(parents=True, exist_ok=True)
    ledger = ledger_directory/(ledger_key(protocol)+'.json')
    resuming_claim = False
    if ledger.exists():
        attempt = json.loads(ledger.read_text())
        if not resume or attempt['reviewed_plan_sha256'] != plan_hash or not (output/'attempt.json').exists():
            raise FileExistsError('This dataset holdout was already claimed. Restore the same output and use resume; a new run is refused')
        local = json.loads((output/'attempt.json').read_text())
        if local['attempt_id'] != attempt['attempt_id'] or local['reviewed_plan_sha256'] != plan_hash:
            raise ValueError('Saved attempt does not match the persistent ledger')
        if attempt['status'] == 'completed':
            if file_receipt(output/'result_receipt.json')['sha256'] != attempt['result_receipt_sha256']:
                raise ValueError('Completed result receipt differs from the persistent ledger')
            return inspect_result(output)
        if not (output/'cache/test_frame.parquet').exists() or 'test_cache_receipt' not in attempt:
            raise RuntimeError('The claimed test has no verified completed cache. Preserve this attempt for review; automatic re-decoding is forbidden')
        if file_receipt(output/'cache/test_frame.parquet') != attempt['test_cache_receipt']:
            raise ValueError('Saved test cache differs from the ledger; no re-decoding is permitted')
        resuming_claim = True
    elif output.exists():
        raise FileExistsError('Use an absent result directory for the first final-test attempt')
    # Confirm inference/runtime before making a first irreversible dataset claim.
    with tempfile.TemporaryDirectory(prefix='final_runtime_probe_') as temporary:
        probe_frame = pd.read_parquet(prepared/'selection_reproduction/inputs.parquet')
        _, metadata = _worker(prepared, probe_frame, Path(temporary))
        if metadata['runtime'] != plan['inference_runtime']:
            raise ValueError('Inference runtime differs from the reviewed plan; restore it before final evaluation')
    if not resuming_claim:
        attempt = {'status': 'claimed', 'study_id': protocol['study_id'], 'dataset_sha256': protocol['dataset_sha256'],
                   'reviewed_plan_sha256': plan_hash, 'attempt_id': str(uuid.uuid4()),
                   'claimed_at_utc': datetime.now(timezone.utc).isoformat(), 'test_parquet_decoded': False}
        fd = os.open(ledger, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(attempt, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        output.mkdir(parents=True)
        (output/'cache').mkdir()
        write_json(output/'attempt.json', attempt)
        write_json(output/'reviewed_plan.json', plan)
        mirror_run(output, mirror_directory)
        try:
            attempt['test_access_started'] = True
            attempt['test_parquet_decoded'] = None
            _update_ledger(ledger, attempt)
            frame = _decode_test(package, protocol, attempt)
            atomic_write(output/'cache/test_frame.parquet', lambda p: frame.to_parquet(p, index=False))
            attempt.update(status='test_cached', test_parquet_decoded=True,
                           test_cache_receipt=file_receipt(output/'cache/test_frame.parquet'))
            _update_ledger(ledger, attempt)
            mirror_run(output, mirror_directory)
        except Exception:
            attempt['status'] = 'failed_before_verified_cache'
            _update_ledger(ledger, attempt)
            raise
    else:
        frame = pd.read_parquet(output/'cache/test_frame.parquet')
    # Resume can recompute deterministic software outputs from the same immutable cache,
    # but it never invokes the original ZIP decoder again.
    parameters, metadata = _worker(prepared, frame, output/'inference')
    if metadata['runtime'] != plan['inference_runtime']:
        raise ValueError('Actual inference runtime differs from the reviewed plan')
    frozen = json.loads((prepared/'model/frozen_model.json').read_text())
    metrics = score_all(frame, parameters, protocol, frozen['config'], plan['training_RMS_reference_std_native'], output)
    metrics.update(reviewed_plan_sha256=plan_hash, attempt_id=attempt['attempt_id'],
                   frozen_record_digest=plan['frozen_record_digest'], calibration_file_sha256=plan['calibration_file_sha256'],
                   test_parquet_decoded=True, inference_runtime=metadata['runtime'], model_training=False, calibration_refit=False)
    write_json(output/'final_metrics.json', metrics)
    (output/'README.md').write_text(
        '# Frozen final-test evaluation\n\n'
        f'Observations: {metrics["rows"]}; cyclones: {metrics["storms"]}.\n\n'
        'The model, calibration factors, scoring code and protocol were fixed before test decoding.\n\n'
        'Read final_metrics.json for all declared point and probability references, paired errors, uncertainty scores and whole-cyclone intervals.\n\n'
        'This is a retrospective best-track/reanalysis study. It does not establish operational data availability or real-time forecasting skill.\n\n'
        'Report these results as observed; do not use this holdout for further model or calibration choices.\n', encoding='utf-8')
    attempt.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'ledger_receipt.json', attempt)
    final_files = [p for p in output.rglob('*') if p.is_file() and p.name != 'result_receipt.json']
    receipt = {'status': 'completed', 'reviewed_plan_sha256': plan_hash, 'attempt_id': attempt['attempt_id'],
               'files': {p.relative_to(output).as_posix(): file_receipt(p) for p in final_files}}
    write_json(output/'result_receipt.json', receipt)
    attempt.update(result_receipt_sha256=file_receipt(output/'result_receipt.json')['sha256'])
    _update_ledger(ledger, attempt)
    mirror_run(output, mirror_directory)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--results', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('evaluate')
    p.add_argument('--prepared', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ledger', required=True)
    p.add_argument('--reviewed-plan-sha256', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--mirror-dir')
    p = sub.add_parser('inspect')
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = prepare(args.results, args.output)
        print(json.dumps({'status': plan['status'], 'reviewed_plan_sha256': digest(plan), 'test_parquet_decoded': False}, indent=2))
    elif args.command == 'evaluate':
        metrics = evaluate(args.prepared, args.output, args.ledger, args.reviewed_plan_sha256, args.resume, args.mirror_dir)
        print(json.dumps({k:metrics[k] for k in ['status', 'rows', 'storms', 'reviewed_plan_sha256', 'test_parquet_decoded']}, indent=2))
    else:
        print(json.dumps(inspect_result(args.output), indent=2))


if __name__ == '__main__':
    main()
