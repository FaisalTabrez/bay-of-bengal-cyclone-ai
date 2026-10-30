"""Final-gate tests use temporary synthetic test ZIPs, never the real bob_test."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from modelling.data import TrainingPackage
from modelling.experiment_io import digest
from modelling import final_evaluation as final
from modelling.final_scores import probability_records, storm_intervals
from modelling.uncertainty_io import native_truth, read_config
from modelling.uncertainty_math import distribution_scores

ROOT = Path(__file__).resolve().parents[1]


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        real = TrainingPackage(ROOT/'data/packages/bob_cyclone_training_v2.zip')
        # The old selection values only supply a valid schema. IDs/times mark a
        # synthetic fixture; all files and holdout outputs stay in this temp dir.
        frame = real.load('bob_validation').iloc[:6].copy().reset_index(drop=True)
        frame['SID'] = ['synthetic-a']*3+['synthetic-b']*3
        frame['ISO_TIME'] = pd.date_range('2020-01-01', periods=6, freq='6h', tz='UTC')
        frame['TARGET_TIME_6H'] = frame.ISO_TIME+pd.Timedelta(hours=6)
        frame['SPLIT'] = 'synthetic_test'
        self.frame = frame
        parquet = self.root/'fixture.parquet'
        frame.to_parquet(parquet, index=False)
        self.archive = self.root/'synthetic_test_not_research.zip'
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.write(parquet, 'bob_test/fixture.parquet')
        self.package = SimpleNamespace(path=self.archive, manifest=real.manifest, features=real.features,
                                       targets=real.targets, load=lambda name: pd.DataFrame({'SID': ['other-fixture']}))
        self.protocol = final.protocol_config()
        self.protocol.update(expected_rows=6, expected_storms=2, input_years=[2020, 2020],
                             dataset_sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest(),
                             bootstrap_replicates=64, energy_score_draws=16, study_id='synthetic_fixture_only')
        self.prepared, self.output, self.ledger = self.root/'prepared', self.root/'output', self.root/'ledger'
        (self.prepared/'model').mkdir(parents=True)
        (self.prepared/'selection_reproduction').mkdir()
        frame.drop(columns=real.targets).to_parquet(self.prepared/'selection_reproduction/inputs.parquet', index=False)
        (self.prepared/'model/frozen_model.json').write_text(json.dumps({'config': read_config(False)}))
        self.plan = {'protocol': self.protocol, 'inference_runtime': {'fixture_only': True},
                     'training_RMS_reference_std_native': [10., 10., 5., 3.],
                     'frozen_record_digest': 'fixture-not-a-real-model', 'calibration_file_sha256': 'fixture',
                     'test_parquet_decoded': False, 'evaluator_fingerprint': {'fixture_only': True}, 'files': {}}
        self.reviewed_hash = digest(self.plan)

    def tearDown(self):
        self.temp.cleanup()

    def worker(self, prepared, frame, directory):
        # Predictions depend only on current inputs, never future labels.
        point = frame[['LAT', 'LON', 'WIND_KTS', 'PRESSURE_HPA']].fillna(0).to_numpy(float)
        mean = np.column_stack([np.zeros((len(frame), 2)), point[:, 2:]])
        parameters = {'point': point, 'mean': mean, 'std_calibrated': np.tile([10., 10., 5., 3.], (len(frame), 1)),
                      'std_uncalibrated': np.tile([12., 12., 6., 4.], (len(frame), 1)), 'rho': np.zeros(len(frame))}
        return parameters, {'runtime': {'fixture_only': True}}

    def contexts(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(final, 'verify_prepared', return_value=self.plan))
        stack.enter_context(patch.object(final, 'TrainingPackage', return_value=self.package))
        stack.enter_context(patch.object(final, '_worker', side_effect=self.worker))
        return stack

    def test_review_digest_is_required_before_any_decoder_or_claim(self):
        with self.contexts(), patch.object(final, '_decode_test') as decoder:
            with self.assertRaises(PermissionError):
                final.evaluate(self.prepared, self.output, self.ledger, '')
            decoder.assert_not_called()
        self.assertFalse(self.ledger.exists())

    def test_one_claim_before_decode_and_completed_resume_is_read_only(self):
        original = final._decode_test
        def checked(package, protocol, attempt):
            self.assertEqual(package.path, self.archive)
            claim = json.loads((self.ledger/(final.ledger_key(protocol)+'.json')).read_text())
            self.assertEqual(claim['status'], 'claimed')
            self.assertTrue(claim['test_access_started'])
            return original(package, protocol, attempt)
        with self.contexts(), patch.object(final, '_decode_test', side_effect=checked) as decoder:
            metrics = final.evaluate(self.prepared, self.output, self.ledger, self.reviewed_hash)
            self.assertEqual(metrics['rows'], 6)
            self.assertEqual(metrics['storms'], 2)
            self.assertNotIn('selection_composite', metrics['point_metrics']['full'])
            with self.assertRaises(FileExistsError):
                final.evaluate(self.prepared, self.output, self.ledger, self.reviewed_hash)
            again = final.evaluate(self.prepared, self.output, self.ledger, self.reviewed_hash, resume=True)
            self.assertEqual(metrics, again)
            with self.assertRaises(FileExistsError):
                final.evaluate(self.prepared, self.root/'another-run', self.ledger, self.reviewed_hash, resume=True)
            self.assertEqual(decoder.call_count, 1)

    def test_interrupted_inference_resumes_only_from_same_verified_cache(self):
        original = final._decode_test
        calls = []
        def interrupted(prepared, frame, directory):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError('synthetic inference interruption')
            return self.worker(prepared, frame, directory)
        with self.contexts(), patch.object(final, '_worker', side_effect=interrupted), patch.object(final, '_decode_test', wraps=original) as decoder:
            with self.assertRaisesRegex(RuntimeError, 'synthetic inference'):
                final.evaluate(self.prepared, self.output, self.ledger, self.reviewed_hash)
            self.assertTrue((self.output/'cache/test_frame.parquet').exists())
            final.evaluate(self.prepared, self.output, self.ledger, self.reviewed_hash, resume=True)
            self.assertEqual(decoder.call_count, 1)

    def test_missing_cache_after_failed_access_cannot_decode_again(self):
        with self.contexts(), patch.object(final, '_decode_test', side_effect=RuntimeError('synthetic failed access')) as decoder:
            with self.assertRaisesRegex(RuntimeError, 'synthetic failed'):
                final.evaluate(self.prepared, self.output, self.ledger, self.reviewed_hash)
            with self.assertRaisesRegex(RuntimeError, 're-decoding is forbidden'):
                final.evaluate(self.prepared, self.output, self.ledger, self.reviewed_hash, resume=True)
            self.assertEqual(decoder.call_count, 1)

    def test_original_design_loader_still_denies_real_test_before_parquet(self):
        package = TrainingPackage(ROOT/'data/packages/bob_cyclone_training_v2.zip')
        with patch('modelling.data.pd.read_parquet') as decoder:
            with self.assertRaises(PermissionError):
                package.load('bob_test')
            decoder.assert_not_called()

    def test_row_scores_match_gaussian_scores_and_keep_missingness(self):
        frame = self.frame.copy()
        frame.loc[0, 'TARGET_WIND_KTS_6H'] = np.nan
        truth = native_truth(frame)
        mean = np.nan_to_num(truth, nan=20.)+1.
        std = np.tile([3., 4., 5., 2.], (len(frame), 1))
        rho = np.repeat(.3, len(frame))
        records = probability_records(frame, mean, std, rho, self.protocol, [850., 1100.])
        config = read_config(False)
        config.update(energy_score_draws=self.protocol['energy_score_draws'], diagnostic_seed=self.protocol['energy_score_seed'])
        reference = distribution_scores(truth, mean, std, rho, config)
        self.assertAlmostEqual(records.track_joint_nll.mean(), reference['track_joint_nll'], places=10)
        self.assertAlmostEqual(records.track_energy_km.mean(), reference['track_energy_score_km'], places=10)
        self.assertEqual(records.wind_kts_nll.notna().sum(), reference['marginals']['wind_kts']['labels'])
        self.assertAlmostEqual(records.wind_kts_crps.mean(), reference['marginals']['wind_kts']['crps'], places=10)
        self.assertAlmostEqual(records.wind_kts_coverage_90.mean(), reference['marginals']['wind_kts']['intervals']['0.9']['coverage'])

    def test_bootstrap_resamples_entire_storms(self):
        frame = pd.DataFrame({'SID': ['a', 'b', 'b', 'b'], 'score': [1., 5., 5., 5.]})
        intervals = storm_intervals(frame, ['score'], 1000, 17)
        self.assertEqual(intervals['intervals']['score']['low'], 1.)
        self.assertEqual(intervals['intervals']['score']['high'], 5.)

    def test_frozen_source_snapshot_is_exact_and_distinct_from_new_evaluator(self):
        protocol = final.protocol_config()
        self.assertEqual(hashlib.sha256(final.SOURCE_ARCHIVE.read_bytes()).hexdigest(), protocol['inference_source_archive_sha256'])
        with zipfile.ZipFile(final.SOURCE_ARCHIVE) as z:
            mapping = {(name.removeprefix('modelling/') if name.startswith('modelling/') else name):
                       hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}
        self.assertEqual(digest(mapping), protocol['inference_source_map_digest'])
        self.assertNotIn('final_evaluation.py', mapping)

    def test_inference_worker_receives_only_target_free_inputs(self):
        from modelling.uncertainty_io import file_receipt
        (self.prepared/'model/frozen_model.json').write_text(json.dumps({'source_sha256': {'fixture': 'only'}}))
        def fake_process(arguments, **kwargs):
            request = json.loads(Path(arguments[2]).read_text())
            inputs = pd.read_parquet(request['inputs'])
            self.assertTrue(set(self.package.targets).isdisjoint(inputs.columns))
            parameters, metadata = self.worker(self.prepared, inputs, self.root/'unused')
            with Path(request['output']).open('wb') as stream:
                np.savez(stream, **parameters)
            Path(request['metadata']).write_text(json.dumps({'input_rows': len(inputs), 'target_columns_received': False,
                                                            'input_sha256': file_receipt(request['inputs'])['sha256']}))
            return SimpleNamespace(returncode=0, stdout='synthetic inference stub')
        with patch.object(final.subprocess, 'run', side_effect=fake_process):
            parameters, metadata = final._worker(self.prepared, self.frame, self.root/'inference')
        self.assertEqual(parameters['point'].shape, (6, 4))

    def test_prepared_source_and_artifact_corruption_block_the_gate(self):
        from modelling.final_evaluation import file_receipt
        plan = copy.deepcopy(self.plan)
        target = self.prepared/'model/frozen_model.json'
        plan['files'] = {'model/frozen_model.json': file_receipt(target)}
        (self.prepared/'reviewed_plan.json').write_text(json.dumps(plan))
        with patch.object(final, 'protocol_config', return_value=self.protocol), patch.object(final, 'evaluator_fingerprint', return_value=plan['evaluator_fingerprint']):
            final.verify_prepared(self.prepared)
            target.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'integrity failure'):
                final.verify_prepared(self.prepared)
        with patch.object(final, 'protocol_config', return_value=self.protocol), patch.object(final, 'evaluator_fingerprint', return_value={'changed_source': True}):
            with self.assertRaisesRegex(ValueError, 'source/config differs'):
                final.verify_prepared(self.prepared)

    def test_concurrent_claim_during_probe_cannot_be_mistaken_for_resume(self):
        def competing(prepared, frame, directory):
            self.ledger.mkdir(exist_ok=True)
            (self.ledger/(final.ledger_key(self.protocol)+'.json')).write_text(json.dumps({'status': 'claimed', 'attempt_id': 'another-process'}))
            return self.worker(prepared, frame, directory)
        with self.contexts(), patch.object(final, '_worker', side_effect=competing), patch.object(final, '_decode_test') as decoder:
            with self.assertRaises(FileExistsError):
                final.evaluate(self.prepared, self.output, self.ledger, self.reviewed_hash)
            decoder.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
