"""Safety-critical scientific contracts, tested without reading test observations."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modelling.data import TrainingPackage, causal_windows, validation_roles
from modelling.geometry import log_map, exp_map, haversine_km
from modelling.model import ResidualCodec, ResidualGRU, masked_multitask_loss

ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        cls.package = TrainingPackage(ROOT / "data/packages/bob_cyclone_training_v2.zip")
        cls.bob = cls.package.load("bob_train")
        cls.normalizer = cls.package.normalizer("bob")

    def test_test_partition_is_denied(self):
        with self.assertRaises(PermissionError):
            self.package.load("bob_test")

    def test_normalization_and_input_masks(self):
        self.normalizer.verify_training_statistics(self.bob)
        x = self.normalizer.transform(self.bob)
        n = len(self.package.features)
        missing = self.bob[self.package.features].isna().to_numpy()
        self.assertEqual(x.shape, (4560, 78))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[:, n:], missing)
        np.testing.assert_array_equal(x[:, :n][missing], 0)

    def test_windows_stop_at_cyclone_and_gap(self):
        frame = pd.DataFrame({"SID": ["a", "a", "a", "b", "b"], "ISO_TIME": pd.to_datetime([
            "2000-01-01T00:00Z", "2000-01-01T06:00Z", "2000-01-01T18:00Z",
            "2000-01-01T00:00Z", "2000-01-01T06:00Z"] )})
        self.assertEqual([w.tolist() for w in causal_windows(frame)], [[0], [0, 1], [2], [3], [3, 4]])
        with self.assertRaises(ValueError):
            causal_windows(frame.iloc[::-1].reset_index(drop=True))

    def test_geometry_roundtrip_across_dateline_and_zero_motion(self):
        lat = np.array([15., 30., -10., 0.])
        lon = np.array([90., 179.8, -179.7, 0.])
        future_lat = np.array([16., 30.5, -9., 0.])
        future_lon = np.array([88., -179.8, 179.9, 0.])
        displacement = log_map(lat, lon, future_lat, future_lon)
        decoded = exp_map(lat, lon, displacement)
        np.testing.assert_allclose(decoded[:, 0], future_lat, atol=1e-10)
        np.testing.assert_allclose(decoded[:, 1], future_lon, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(displacement, axis=1),
                                   haversine_km(lat, lon, future_lat, future_lon), atol=1e-10)

    def test_missing_targets_do_not_change_loss_or_gradients(self):
        p = torch.zeros((2, 4), requires_grad=True)
        t = torch.tensor([[1., 2., float("nan"), 3.], [2., 3., float("nan"), float("nan")]])
        m = torch.isfinite(t)
        loss = masked_multitask_loss(p, t, m)
        changed = t.clone()
        changed[~m] = 999999.
        self.assertEqual(loss.item(), masked_multitask_loss(p, changed, m).item())
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue((p.grad[:, 2] == 0).all())
        with self.assertRaises(ValueError):
            masked_multitask_loss(p, t, torch.zeros_like(m))

    def test_missing_current_intensity_does_not_discard_observed_target(self):
        codec = ResidualCodec.fit(self.bob, self.normalizer)
        y, mask = codec.encode(self.bob)
        for j, col in [(2, "WIND_KTS"), (3, "PRESSURE_HPA")]:
            rows = self.bob[col].isna() & self.bob[f"TARGET_{col}_6H"].notna()
            self.assertTrue(rows.any())
            self.assertTrue(mask[rows, j].all())
            self.assertTrue(np.isfinite(y[rows, j]).all())

    def test_padding_cannot_change_forecast(self):
        model = ResidualGRU().eval()
        # Nonzero output heads ensure this tests the actual encoder, not zero initialization.
        with torch.no_grad():
            for head in model.heads:
                head[-1].weight.normal_()
            x = torch.randn(2, 4, 78)
            lengths = torch.tensor([1, 4])
            before = model(x, lengths)
            x[0, 1:] = 1e6
            after = model(x, lengths)
        torch.testing.assert_close(before, after)

    def test_normalization_rebase_preserves_nonzero_model_with_nulls(self):
        model = ResidualGRU().eval()
        old = self.package.normalizer("nio")
        sample = self.bob.iloc[:128]
        lengths = torch.ones(len(sample), dtype=torch.long)
        with torch.no_grad():
            for head in model.heads:
                head[-1].weight.normal_(std=0.1)
            before = model(torch.from_numpy(old.transform(sample))[:, None], lengths)
            model.rebase_inputs(old, self.normalizer)
            after = model(torch.from_numpy(self.normalizer.transform(sample))[:, None], lengths)
        torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)

    def test_zero_residual_matches_constant_velocity_and_intensity_anchor(self):
        codec = ResidualCodec.fit(self.bob, self.normalizer)
        sample = self.bob.iloc[:128]
        forecast = codec.decode(sample, np.zeros((len(sample), 4)))
        np.testing.assert_allclose(forecast[:, 0], sample.LAT + sample.LAT_CHANGE_6H.fillna(0), atol=1e-10)
        np.testing.assert_allclose(forecast[:, 1], sample.LON + sample.LON_CHANGE_6H.fillna(0), atol=1e-10)
        for j, col in [(2, "WIND_KTS"), (3, "PRESSURE_HPA")]:
            expected = sample[col].fillna(self.normalizer.statistics[col]["mean"])
            np.testing.assert_allclose(forecast[:, j], expected)

    def test_validation_roles_keep_whole_storms(self):
        frame = self.package.load("bob_validation")
        frame["role"] = validation_roles(frame)
        self.assertEqual(frame.groupby("SID").role.nunique().max(), 1)
        self.assertEqual(frame.groupby("role").size().to_dict(), {"calibration": 361, "selection": 493})


if __name__ == "__main__":
    unittest.main()
