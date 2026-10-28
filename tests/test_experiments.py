"""Integration evidence for exact epoch-boundary resume, pairing and isolation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from modelling.baselines import predictions
from modelling.experiment_io import load_checkpoint
from modelling.metrics import prediction_records, storm_bootstrap, summarize_records
from modelling.train import Inputs, aggregate, load_config, train_one

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.inputs = Inputs(ROOT / "data/packages/bob_cyclone_training_v2.zip")
        cls.config = load_config(acceptance=True)

    def test_transfer_resume_is_bitwise_equal_across_stage_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            continuous, resumed, mirror = root / "continuous", root / "resumed", root / "mirror"
            device = torch.device("cpu")
            whole = train_one(self.inputs, self.config, "transfer_gru", 17, continuous, device)
            paused = train_one(self.inputs, self.config, "transfer_gru", 17, resumed, device, stop_after_epochs=1, mirror_dir=mirror)
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(load_checkpoint(resumed / "latest.pt")["progress"]["stage"], 0)
            result = train_one(self.inputs, self.config, "transfer_gru", 17, resumed, device, resume=True, mirror_dir=mirror)
            a, b = load_checkpoint(continuous / "latest.pt"), load_checkpoint(resumed / "latest.pt")
            self.assertEqual(a["progress"]["global_step"], b["progress"]["global_step"])
            for key, value in a["model_state"].items():
                self.assertTrue(torch.equal(value, b["model_state"][key]), key)
            self.assertEqual(whole["selection_score"], result["selection_score"])
            self.assertEqual(result["checkpoint_reload_max_abs_difference"], 0.)
            self.assertEqual((mirror / "latest.pt").read_bytes(), (resumed / "latest.pt").read_bytes())
            # Completed resume produces no new updates.
            again = train_one(self.inputs, self.config, "transfer_gru", 17, resumed, device, resume=True)
            self.assertEqual(result["global_step"], again["global_step"])
            changed = copy.deepcopy(self.config)
            changed["dropout"] = .2
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                train_one(self.inputs, changed, "transfer_gru", 17, resumed, device, resume=True)
            with self.assertRaises(FileExistsError):
                train_one(self.inputs, self.config, "transfer_gru", 17, resumed, device)
            self.assertFalse(result["test_parquet_decoded"])
            self.assertFalse(result["calibration_used_for_selection"])

    def test_intensity_pairing_excludes_only_unavailable_baseline_cases(self):
        frame = self.inputs.selection
        forecast = predictions(frame, tendency=True)
        for j, col in [(2, "WIND_KTS"), (3, "PRESSURE_HPA")]:
            forecast[:, j] = frame[col].fillna(self.inputs.norms["bob"].statistics[col]["mean"])
        records = prediction_records(frame, forecast)
        score = summarize_records(records)
        self.assertAlmostEqual(score["selection_composite"], 1., places=12)
        for j, col, prefix in [(2, "WIND_KTS", "wind"), (3, "PRESSURE_HPA", "pressure")]:
            eligible = frame[col].notna() & frame[f"TARGET_{col}_6H"].notna()
            self.assertEqual(score[f"{prefix}_paired_labels"], int(eligible.sum()))
            self.assertEqual(score[f"{prefix}_observed_labels"], int(frame[f"TARGET_{col}_6H"].notna().sum()))
            missing_current = frame[col].isna() & frame[f"TARGET_{col}_6H"].notna()
            changed = forecast.copy()
            changed[missing_current, j] += 10000.
            changed_score = summarize_records(prediction_records(frame, changed))
            self.assertEqual(score["selection_composite"], changed_score["selection_composite"])
            self.assertNotEqual(score[f"{prefix}_mae_{'kts' if j == 2 else 'hpa'}_all_observed"],
                                changed_score[f"{prefix}_mae_{'kts' if j == 2 else 'hpa'}_all_observed"])

    def test_update_budget_control_has_exact_requested_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = train_one(self.inputs, self.config, "bay_budget", 17, Path(tmp) / "run", torch.device("cpu"), budget_updates=5)
            self.assertEqual(result["global_step"], 5)

    def test_aggregation_requires_declared_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = copy.deepcopy(self.config)
            config["training_plan"]["seeds"] = [17, 29, 43]
            result = aggregate(self.inputs, config, "bay_gru", tmp)
            self.assertEqual(result["status"], "waiting_for_declared_seeds")

    def test_runtime_change_requires_explicit_flag_and_is_recorded(self):
        from unittest.mock import patch
        from modelling.experiment_io import runtime_info
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run"
            device = torch.device("cpu")
            train_one(self.inputs, self.config, "bay_budget", 17, folder, device, budget_updates=2)
            changed_runtime = dict(runtime_info(device), device_name="simulated replacement runtime")
            with patch("modelling.train.runtime_info", return_value=changed_runtime):
                with self.assertRaisesRegex(ValueError, "Resume rejected"):
                    train_one(self.inputs, self.config, "bay_budget", 17, folder, device, budget_updates=2, resume=True)
                result = train_one(self.inputs, self.config, "bay_budget", 17, folder, device,
                                   budget_updates=2, resume=True, allow_runtime_change=True)
            self.assertEqual(result["global_step"], 2)
            self.assertFalse(result["runtime_changes"][0]["bitwise_equivalence_claimed"])
            self.assertEqual(len(load_checkpoint(folder / "latest.pt")["progress"]["runtime_changes"]), 1)

    def test_bootstrap_draws_storm_clusters_and_handles_unobserved_labels(self):
        import pandas as pd
        # Storm a has one row; storm b has three. A row bootstrap could not yield
        # the exact all-a/all-b limits with the same cluster sampling mechanism.
        records = pd.DataFrame({"SID": ["a", "b", "b", "b"], "TRACK_ERROR_KM": [1., 5., 5., 5.],
                                "BASE_TRACK_ERROR_KM": [2., 6., 6., 6.]})
        for name in ["WIND_PAIRED_ERROR", "WIND_BASE_ERROR", "PRESSURE_PAIRED_ERROR", "PRESSURE_BASE_ERROR", "WIND_ABS_ERROR", "PRESSURE_ABS_ERROR"]:
            records[name] = [np.nan, 1., 1., 1.]
        result = storm_bootstrap(records, 500, seed=1701)
        interval = result["intervals"]["track_mean_km"]
        self.assertEqual((interval["low"], interval["high"]), (1., 5.))
        self.assertLess(result["intervals"]["wind_mae_kts_all_observed"]["valid_replicates"], 500)


if __name__ == "__main__":
    unittest.main()
