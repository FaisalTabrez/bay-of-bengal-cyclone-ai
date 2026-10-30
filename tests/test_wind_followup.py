"""Wind-loss, causal forecast, evidence-integrity and follow-up contract tests."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modelling.artifacts import export_runs
from modelling.model import masked_multitask_loss
from modelling.train import load_config
from modelling.wind_audit import ResultsArchive, label_profile, persistence_wind, save_original_results, validate_records, wind_groups


class WindFollowupTests(unittest.TestCase):
    @staticmethod
    def research_layout_fixture():
        import hashlib
        from modelling.wind_audit import FAMILIES, SEEDS
        # Structural upload fixture only; audit() still requires real run semantics.
        members = {f'research/{family}/{member}/{name}': b'{}'
                   for family in FAMILIES
                   for member in [*(f'seed_{s}' for s in SEEDS), 'ensemble']
                   for name in ['summary.json', 'selection_predictions.csv']}
        manifest = {n: {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)} for n, b in members.items()}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as z:
            for name, payload in members.items():
                z.writestr(name, payload)
            z.writestr('RUN_FILE_MANIFEST.json', json.dumps(manifest))
        return stream.getvalue()

    def test_wrong_archive_type_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'wrong.zip'
            for name, diagnostic in [('BUNDLE_MANIFEST.json', 'source/code'),
                                     ('training_manifest_v2.json', 'training dataset'),
                                     ('wrapped/RUN_FILE_MANIFEST.json', 'nested manifest'),
                                     ('readme.txt', 'without a results manifest')]:
                with self.subTest(name=name):
                    with zipfile.ZipFile(p, 'w') as z:
                        z.writestr(name, '{}')
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        ResultsArchive(p)

    def test_invalid_upload_preserves_existing_cache_and_rejects_acceptance_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'cached.zip'
            valid = self.research_layout_fixture()
            save_original_results(valid, p)
            with self.assertRaisesRegex(ValueError, 'not a readable ZIP'):
                save_original_results(b'not a zip', p)
            self.assertEqual(p.read_bytes(), valid)
            data = io.BytesIO()
            with zipfile.ZipFile(data, 'w') as z:
                z.writestr('RUN_FILE_MANIFEST.json', '{}')
            with self.assertRaisesRegex(ValueError, 'acceptance-only'):
                save_original_results(data.getvalue(), p)
            self.assertEqual(p.read_bytes(), valid)

    def test_notebook_reprompts_for_cached_wrong_zip(self):
        import types
        from unittest.mock import Mock, patch
        import nbformat
        from modelling.make_wind_notebook import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notebook_path = root/'notebook.ipynb'
            build(notebook_path)
            notebook = nbformat.read(notebook_path, as_version=4)
            cell = next(c for c in notebook.cells if c.cell_type == 'code' and 'results_reader = None' in c.source)
            cached = root/'original_results.zip'
            with zipfile.ZipFile(cached, 'w') as z:
                z.writestr('BUNDLE_MANIFEST.json', '{}')
            google, colab = types.ModuleType('google'), types.ModuleType('google.colab')
            colab.files = types.SimpleNamespace(upload=Mock(return_value={'original.zip': self.research_layout_fixture()}))
            google.colab = colab
            context = {'PERSIST_TO_DRIVE': False, 'IN_COLAB': True, 'WORKDIR': root,
                       'RUN_ROOT': root/'runs/a', 'RUN_ID': 'a', 'RESULTS_PATH': cached, 'Path': Path, 'zipfile': zipfile}
            with patch.dict('sys.modules', {'google': google, 'google.colab': colab}):
                exec(compile(cell.source, '<notebook results upload>', 'exec'), context)
            colab.files.upload.assert_called_once()
            ResultsArchive(cached).require_research()

    def test_notebook_rejects_results_at_source_upload_before_extracting(self):
        import types
        from unittest.mock import Mock, patch
        import nbformat
        from modelling.make_wind_notebook import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notebook_path = root/'notebook.ipynb'
            build(notebook_path)
            notebook = nbformat.read(notebook_path, as_version=4)
            cell = next(c for c in notebook.cells if c.cell_type == 'code' and 'manifest_path = ' in c.source)
            google, colab = types.ModuleType('google'), types.ModuleType('google.colab')
            colab.files = types.SimpleNamespace(upload=Mock(return_value={'results.zip': self.research_layout_fixture()}))
            google.colab = colab
            workdir = root/'source'
            with patch.dict('sys.modules', {'google': google, 'google.colab': colab}):
                with self.assertRaisesRegex(ValueError, 'SOURCE bundle'):
                    exec(compile(cell.source, '<notebook source upload>', 'exec'), {'IN_COLAB': True, 'WORKDIR': workdir})
            self.assertFalse(workdir.exists())

    def test_absolute_wind_loss_keeps_nan_targets_out_of_loss_and_gradients(self):
        p = torch.tensor([[.25, .25, .25, .25], [.25, .25, .25, .25]], requires_grad=True)
        t = torch.tensor([[0., 0., 0., 0.], [float('nan')]*4])
        m = torch.isfinite(t)
        default = masked_multitask_loss(p, t, m)
        explicit = masked_multitask_loss(p, t, m, 'smooth_l1')
        self.assertEqual(default.item(), explicit.item())
        changed = masked_multitask_loss(p, t, m, 'absolute')
        self.assertAlmostEqual(changed.item(), (.03125+.25+.03125)/3, places=7)
        changed.backward()
        self.assertTrue(torch.equal(p.grad[1], torch.zeros(4)))
        self.assertGreater(p.grad[0, 2].item(), p.grad[0, 3].item())
        with self.assertRaises(ValueError):
            masked_multitask_loss(p, t, m, 'typo')

    def test_hybrid_is_target_independent_and_preserves_missing_current_predictions(self):
        f = pd.DataFrame({'WIND_KTS': [0., 25., np.nan], 'TARGET_WIND_KTS_6H': [100., 100., 100.]})
        predictions = np.arange(12., dtype=float).reshape(3, 4)
        old = predictions.copy()
        result = persistence_wind(f, predictions)
        np.testing.assert_array_equal(result[:, 2], [0., 25., 10.])
        np.testing.assert_array_equal(result[:, [0, 1, 3]], predictions[:, [0, 1, 3]])
        f.TARGET_WIND_KTS_6H = -999.
        np.testing.assert_array_equal(result, persistence_wind(f, predictions))
        np.testing.assert_array_equal(predictions, old)

    def test_diagnostic_bands_keep_missing_labels_and_future_bands_out_of_forecast(self):
        f = pd.DataFrame({'SID': ['a']*5, 'WIND_KTS': [25., 25., 40., np.nan, 50.],
                          'TARGET_WIND_KTS_6H': [25., 30., 30., 30., np.nan], 'SST_C': [28.]*5})
        groups = wind_groups(f, np.array([26., 28., 34., 32., 40.]))
        self.assertEqual(groups['steady']['paired_labels'], 1)
        self.assertEqual(groups['steady']['persistence_mae_paired_kts'], 0.)
        self.assertEqual(groups['strengthening']['mae_paired_kts'], 2.)
        self.assertEqual(groups['weakening']['mae_paired_kts'], 4.)
        self.assertEqual(groups['missing_current']['mae_all_observed_kts'], 2.)
        self.assertIsNone(groups['missing_current']['mae_paired_kts'])
        self.assertIsNone(groups['missing_target']['mae_all_observed_kts'])
        self.assertEqual(label_profile(f, 8.)['unchanged'], 1)

    def test_archive_rejects_corruption_and_never_parses_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.mkdir()
            (source/'metrics.json').write_text('{"status":"fixture"}')
            (source/'best.pt').write_bytes(b'not a model')
            archive = root/'result.zip'
            export_runs(source, archive)
            reader = ResultsArchive(archive)
            self.assertEqual(reader.read('metrics.json')['status'], 'fixture')
            with self.assertRaises(ValueError):
                reader.read('best.pt')
            with zipfile.ZipFile(archive) as z:
                entries = {n: z.read(n) for n in z.namelist()}
            entries['metrics.json'] = b'{}'
            with zipfile.ZipFile(root/'bad.zip', 'w') as z:
                for n, payload in entries.items():
                    z.writestr(n, payload)
            with self.assertRaisesRegex(ValueError, 'integrity'):
                ResultsArchive(root/'bad.zip')

    def test_prediction_binding_rejects_forged_labels_even_with_valid_archive_hash(self):
        from modelling.data import TrainingPackage, validation_roles
        from modelling.metrics import prediction_records
        root = Path(__file__).resolve().parents[1]
        p = TrainingPackage(root/'data/packages/bob_cyclone_training_v2.zip')
        f = p.load('bob_validation')
        f = f.loc[validation_roles(f) == 'selection'].reset_index(drop=True)
        forecast = f[['LAT', 'LON', 'WIND_KTS', 'PRESSURE_HPA']].fillna(0).to_numpy(float)
        records = prediction_records(f, forecast)
        validate_records(records, f)
        records.loc[0, 'TARGET_LAT_6H'] += 1
        with self.assertRaisesRegex(ValueError, 'target differs'):
            validate_records(records, f)

    def test_followup_configuration_changes_only_the_declared_loss(self):
        from modelling.wind_followup import CONFIG
        a, b = load_config(), load_config(CONFIG)
        changed = {k for k in set(a)|set(b) if a.get(k) != b.get(k)}
        self.assertEqual(changed, {'status', 'experiment_id', 'wind_loss', 'loss'})
        self.assertEqual(b['wind_loss'], 'absolute')
        self.assertEqual(b['training_plan']['seeds'], [17, 29, 43])

    def test_absolute_loss_mlp_resume_is_bitwise_equal(self):
        from modelling.train import Inputs, train_one
        from modelling.experiment_io import load_checkpoint
        from modelling.wind_followup import CONFIG
        root = Path(__file__).resolve().parents[1]
        inputs = Inputs(root/'data/packages/bob_cyclone_training_v2.zip')
        config = load_config(CONFIG, acceptance=True)
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'whole', Path(tmp)/'resumed'
            whole = train_one(inputs, config, 'bay_mlp', 17, a, torch.device('cpu'))
            train_one(inputs, config, 'bay_mlp', 17, b, torch.device('cpu'), stop_after_epochs=1)
            resumed = train_one(inputs, config, 'bay_mlp', 17, b, torch.device('cpu'), resume=True)
            sa, sb = load_checkpoint(a/'latest.pt'), load_checkpoint(b/'latest.pt')
            for name, value in sa['model_state'].items():
                self.assertTrue(torch.equal(value, sb['model_state'][name]), name)
            self.assertEqual(whole['selection_score'], resumed['selection_score'])

    def test_protocol_rejects_modified_audit_and_source(self):
        from unittest.mock import patch
        from modelling.experiment_io import digest, source_hashes
        from modelling.wind_followup import CONFIG, verify_protocol
        from modelling.wind_audit import PROTOCOL_ID
        audit = {'status': 'passed', 'original_export_sha256': 'fixture-export',
                 'package_sha256': 'fixture-package', 'calibration_labels_used': False,
                 'test_parquet_decoded': False, 'score': 1.}
        protocol = {'protocol_id': PROTOCOL_ID, 'audit_sha256': digest(audit),
                    'config_sha256': digest(json.loads(CONFIG.read_text())),
                    'source_sha256': source_hashes(), 'original_results_sha256': 'fixture-export',
                    'package_sha256': 'fixture-package', 'seeds': [17, 29, 43], 'new_training_runs': 3}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'audit.json').write_text(json.dumps(audit))
            (p/'followup_protocol.json').write_text(json.dumps(protocol))
            verify_protocol(p)
            with patch('modelling.wind_followup.source_hashes', return_value={}):
                with self.assertRaisesRegex(ValueError, 'differs'):
                    verify_protocol(p)
            audit['score'] = .1
            (p/'audit.json').write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, 'differs'):
                verify_protocol(p)


if __name__ == '__main__':
    unittest.main()
