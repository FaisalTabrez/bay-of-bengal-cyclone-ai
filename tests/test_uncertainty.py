"""Numerical and isolation tests. Temporary archive fixtures are not research results."""
import copy
import hashlib
import io
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from modelling.data import TrainingPackage
from modelling.experiment_io import digest, load_checkpoint, source_hashes
from modelling.model import ResidualCodec, ResidualMLP
from modelling.uncertainty import calibrate, predict_frozen, train, verify_frozen
from modelling.uncertainty_io import read_config, role_data, selected_results, save_selected_results
from modelling.uncertainty_math import (SpreadMLP, distribution_scores, fit_scale_factors,
                                      gaussian_nll, marginal_crps, scaled_std)

ROOT = Path(__file__).resolve().parents[1]


def synthetic_archive(destination, package, selection, roles):
    """Zero-head model fixture using real schema; fabricated receipts stay in temp tests."""
    files = {}
    normalization = package.normalizer('bob')
    codec = ResidualCodec.fit(package.load('bob_train'), normalization)
    package_hash = hashlib.sha256(package.path.read_bytes()).hexdigest()
    # Deliberately model a legacy Windows CSV; new freezes must remain portable.
    role_bytes = roles.to_csv(index=False, lineterminator='\r\n').encode()
    role_hash = hashlib.sha256(role_bytes).hexdigest()
    source = {'fixture_only': 'not a scientific training run'}
    decision = {'status': 'bounded_deterministic_selection_complete', 'selected_candidate': 'bay_mlp_wind_absolute_loss',
                'seeds': [17, 29, 43], 'candidate_mean_seed_scores': {'original': 1., 'bay_mlp_wind_absolute_loss': .9},
                'checkpoint_hashes': {}, 'package_sha256': package_hash, 'source_sha256': source,
                'calibration_labels_used': False, 'test_parquet_decoded': False}
    audit = {'fixture_only': True}
    protocol = {'audit_sha256': digest(audit), 'fixture_only': True}
    decision['protocol_sha256'] = digest(protocol)
    for seed in [17, 29, 43]:
        config = {'wind_loss': 'absolute', 'mlp_widths': [64, 32], 'dropout': .1}
        context = {'config': config, 'feature_columns': package.features,
                   'model_input_columns': package.features+[f'{c}_MISSING' for c in package.features],
                   'package_sha256': package_hash, 'source_sha256': source,
                   'validation_roles_sha256': role_hash, 'runtime': {'device': 'cpu', 'fixture_only': True}}
        model = ResidualMLP().eval()
        checkpoint = {'context': context, 'signature': digest(context), 'model_state': model.state_dict(),
                      'input_normalization': normalization.statistics, 'output_codec': codec.as_dict(), 'normalizer_stage': 'bob'}
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        payload = stream.getvalue()
        checksum = hashlib.sha256(payload).hexdigest()
        prefix = f'research/bay_mlp/seed_{seed}/'
        files[prefix+'best.pt'] = payload
        files[prefix+'validation_roles.csv'] = role_bytes
        files[prefix+'run_manifest.json'] = json.dumps(dict(context, signature=digest(context))).encode()
        files[prefix+'summary.json'] = json.dumps({'best_checkpoint_sha256': checksum, 'status': 'completed',
                                                   'run_purpose': 'research_training', 'fixture_only': True,
                                                   'checkpoint_reload_max_abs_difference': 0.}).encode()
        from modelling.metrics import prediction_records
        records = prediction_records(selection, codec.decode(selection, np.zeros((len(selection), 4))))
        files[prefix+'selection_predictions.csv'] = records.to_csv(index=False).encode()
        decision['checkpoint_hashes'][str(seed)] = {'sha256': checksum}
    for name, value in [('research/candidate_decision.json', decision), ('audit/audit.json', audit),
                        ('audit/followup_protocol.json', protocol), ('research/followup_protocol.json', protocol),
                        ('research/declaration.json', {'protocol_sha256': digest(protocol)})]:
        files[name] = json.dumps(value).encode()
    manifest = {name: {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)} for name, payload in files.items()}
    with zipfile.ZipFile(destination, 'w') as z:
        for name, payload in files.items():
            z.writestr(name, payload)
        z.writestr('RUN_FILE_MANIFEST.json', json.dumps(manifest))


class UncertaintyMathTests(unittest.TestCase):
    def test_joint_and_marginal_nll_agree_with_independent_torch_distributions(self):
        error = torch.tensor([[1., 2., 3., 4.], [2., -1., 1., 0.]], dtype=torch.float64)
        std = torch.tensor([[2., 3., 4., 5.], [1., 2., 3., 4.]], dtype=torch.float64, requires_grad=True)
        rho = torch.tensor([.4, -.3], dtype=torch.float64, requires_grad=True)
        covariance = torch.diag_embed(std[:, :2]**2)
        covariance[:, 0, 1] = covariance[:, 1, 0] = rho*std[:, 0]*std[:, 1]
        joint = -torch.distributions.MultivariateNormal(torch.zeros((2, 2)), covariance_matrix=covariance).log_prob(error[:, :2]).mean()
        wind = -torch.distributions.Normal(torch.zeros(2), std[:, 2]).log_prob(error[:, 2]).mean()
        pressure = -torch.distributions.Normal(torch.zeros(2), std[:, 3]).log_prob(error[:, 3]).mean()
        actual = gaussian_nll(error, std, rho, torch.ones_like(error, dtype=torch.bool))
        self.assertAlmostEqual(actual.item(), ((joint+wind+pressure)/3).item(), places=12)
        error[0, 2] = float('nan')
        loss = gaussian_nll(error, std, rho, torch.isfinite(error))
        loss.backward()
        self.assertTrue(torch.isfinite(std.grad).all())
        self.assertEqual(std.grad[0, 2].item(), 0.)

    def test_gaussian_crps_and_track_ellipse_have_known_values(self):
        self.assertAlmostEqual(float(marginal_crps(np.array([0.]), np.array([0.]), np.array([1.]))[0]),
                               (math.sqrt(2)-1)/math.sqrt(math.pi), places=12)
        config = read_config(True)
        truth = np.array([[1., 0., 0., 1000.], [3., 0., np.nan, 1000.]])
        mean = np.array([[0., 0., 0., 1000.]]*2)
        scores = distribution_scores(truth, mean, np.ones((2, 4)), np.zeros(2), config)
        self.assertEqual(scores['track_ellipses']['0.5']['coverage'], .5)
        self.assertAlmostEqual(scores['track_ellipses']['0.5']['mean_area_km2'], math.pi*(-2*math.log(.5)))
        self.assertEqual(scores['marginals']['wind_kts']['labels'], 1)
        self.assertEqual(scores['gaussian_tail_mass']['mean_probability_wind_below_zero'], .5)
        self.assertFalse(scores['gaussian_tail_mass']['clipping_applied'])

    def test_scale_calibration_matches_closed_form_and_covariance_rule(self):
        y = np.array([[2., 2., 1., 3.], [-2., -2., -1., -3.]])
        factors, details = fit_scale_factors(y, np.zeros((2, 4)), np.ones((2, 4)), np.zeros(2))
        np.testing.assert_allclose(factors, [2., 1., 3.])
        np.testing.assert_allclose(scaled_std(np.ones((2, 4)), factors)**2, [[4., 4., 1., 9.]]*2)
        factors, details = fit_scale_factors(y*100, np.zeros((2, 4)), np.ones((2, 4)), np.zeros(2))
        self.assertEqual(factors, [4., 4., 4.])
        self.assertTrue(details['track']['at_declared_bound'])

    def test_spread_parameters_remain_positive_and_track_covariance_positive_definite(self):
        model = SpreadMLP([.1]*4, [1.]*4)
        s, r = model(torch.randn(10, 78)*100)
        self.assertTrue((s > .1).all())
        self.assertTrue((s[:, 0]**2*s[:, 1]**2*(1-r*r) > 0).all())


class UncertaintyPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.package = TrainingPackage(ROOT/'data/packages/bob_cyclone_training_v2.zip')
        cls.selection, cls.roles = role_data(cls.package, 'selection')
        cls.archive = cls.root/'synthetic_fixture_not_research.zip'
        synthetic_archive(cls.archive, cls.package, cls.selection, cls.roles)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_invalid_export_is_rejected_without_creating_run_or_replacing_upload(self):
        wrong = self.root/'source.zip'
        with zipfile.ZipFile(wrong, 'w') as z:
            z.writestr('BUNDLE_MANIFEST.json', '{}')
        output = self.root/'invalid_run'
        with self.assertRaisesRegex(ValueError, 'completed three-seed'):
            train(wrong, output)
        self.assertFalse(output.exists())
        cache = self.root/'cache.zip'
        cache.write_bytes(self.archive.read_bytes())
        with self.assertRaises(ValueError):
            save_selected_results(wrong.read_bytes(), cache)
        self.assertEqual(cache.read_bytes(), self.archive.read_bytes())

    def test_acceptance_resume_keeps_mean_frozen_and_is_bitwise_equal(self):
        whole, resumed = self.root/'whole', self.root/'resumed'
        a = train(self.archive, whole, device='cpu')
        train(self.archive, resumed, device='cpu', stop_after_epochs=1)
        b = train(self.archive, resumed, device='cpu', resume=True)
        sa, sb = load_checkpoint(whole/'latest.pt'), load_checkpoint(resumed/'latest.pt')
        for key, value in sa['model_state'].items():
            self.assertTrue(torch.equal(value, sb['model_state'][key]), key)
        self.assertEqual(a['best_selection_normalized_nll'], b['best_selection_normalized_nll'])
        self.assertFalse(a['mean_parameters_changed'])
        self.assertEqual(a['checkpoint_reload_max_abs_difference'], 0.)
        with self.assertRaisesRegex(ValueError, 'acceptance cannot'):
            calibrate(whole)
        with self.assertRaises(PermissionError):
            role_data(self.package, 'test')

    def test_full_freeze_and_single_calibration_use_only_synthetic_reserved_labels(self):
        config = read_config(False)
        config.update(maximum_epochs=2, patience=2, energy_score_draws=8)
        output = self.root/'research_fixture'
        def guarded(package, role):
            self.assertEqual(role, 'selection', 'Training accessed reserved calibration labels')
            return self.selection, self.roles
        with patch('modelling.uncertainty.read_config', return_value=config), patch('modelling.uncertainty.role_data', side_effect=guarded):
            summary = train(self.archive, output, full=True, device='cpu')
            freeze = verify_frozen(output)
        before = {p.name: p.read_bytes() for p in (output/'mean').glob('*.pt')}
        fake_calibration = self.selection.iloc[:4].copy().reset_index(drop=True)
        fake_calibration['SID'] = ['synthetic-a', 'synthetic-a', 'synthetic-b', 'synthetic-b']
        fake_calibration.TARGET_WIND_KTS_6H = 30.
        fake_calibration.TARGET_PRESSURE_HPA_6H = 1000.
        with patch('modelling.uncertainty.read_config', return_value=config), patch('modelling.uncertainty.role_data', return_value=(fake_calibration, self.roles)):
            inputs_only = fake_calibration.drop(columns=self.package.targets)
            uncalibrated = predict_frozen(inputs_only, output)
            result = calibrate(output, device='cpu')
            self.assertEqual(result['status'], 'calibrated_once')
            self.assertEqual(result['rows'], 4)
            calibrated = predict_frozen(inputs_only, output, calibrated=True)
            np.testing.assert_array_equal(uncalibrated['point_forecast'], calibrated['point_forecast'])
            np.testing.assert_allclose(calibrated['std_tangent_native'],
                                       scaled_std(uncalibrated['std_tangent_native'], result['scale_factors_track_wind_pressure']))
            # Modified scale receipts cannot silently alter future predictions.
            cal_path = output/'calibration/calibration.json'
            contents = cal_path.read_bytes()
            cal_path.write_bytes(contents+b' ')
            with self.assertRaisesRegex(ValueError, 'Calibration receipt'):
                predict_frozen(inputs_only, output, calibrated=True)
            cal_path.write_bytes(contents)
            with self.assertRaises(FileExistsError):
                calibrate(output, device='cpu')
            # Freeze corruption must be rejected before a reserved-role read.
            (output/'mean/normalization.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'integrity failure'):
                verify_frozen(output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in (output/'mean').glob('*.pt')})


if __name__ == '__main__':
    unittest.main()
