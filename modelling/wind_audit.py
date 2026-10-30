"""Audit a completed export and evaluate a declared wind-persistence follow-up.

Reads JSON/CSV evidence only. Model binaries are hashed, never deserialized.
All comparisons use training or selection storms; no calibration evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .data import TrainingPackage, validation_roles
from .experiment_io import atomic_write, digest, source_hashes, write_json
from .geometry import exp_map, log_map
from .metrics import FORECASTS, TARGETS, evaluate_records, prediction_records, summarize_records
from .paths import DEFAULT_PACKAGE

FAMILIES = ("bay_gru", "transfer_gru", "pooled_gru", "bay_mlp", "bay_tree")
SEEDS = (17, 29, 43)
PROTOCOL_ID = "wind_mae_followup_v1"


class ResultsArchive:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with zipfile.ZipFile(self.path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate archive members")
            if 'RUN_FILE_MANIFEST.json' not in names:
                if 'BUNDLE_MANIFEST.json' in names:
                    kind = 'a source/code bundle'
                elif 'package_file_checksums.json' in names or 'training_manifest_v2.json' in names:
                    kind = 'a training dataset bundle'
                elif any(name.endswith('/RUN_FILE_MANIFEST.json') for name in names):
                    kind = 'a repackaged results folder with a nested manifest'
                else:
                    kind = 'an archive without a results manifest'
                raise ValueError(f'This ZIP is {kind}. Upload the original deterministic_v0_3_results ZIP '
                                 'with RUN_FILE_MANIFEST.json at its root. Rerun the results-upload step '
                                 'to replace a cached incorrect original_results.zip.')
            self.manifest = json.loads(archive.read("RUN_FILE_MANIFEST.json"))
            if set(names) != set(self.manifest) | {"RUN_FILE_MANIFEST.json"}:
                raise ValueError("Unlisted or missing archive member")
            for name, expected in self.manifest.items():
                if name.startswith(("/", "\\")) or ".." in name.replace("\\", "/").split("/"):
                    raise ValueError("Unsafe archive member")
                payload = archive.read(name)
                if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                    raise ValueError(f"Result integrity failure: {name}")

    def require_research(self):
        required = {f'research/{family}/{member}/{name}'
                    for family in FAMILIES
                    for member in [*(f'seed_{s}' for s in SEEDS), 'ensemble']
                    for name in ['summary.json', 'selection_predictions.csv']}
        if not required.issubset(self.manifest):
            raise ValueError('This results ZIP does not contain the completed original five-family, '
                             'three-seed research comparison. Upload deterministic_v0_3_results.zip '
                             'from the full original run, rather than an acceptance-only or wind-follow-up export.')
        return self

    def read(self, name):
        if name not in self.manifest or not name.endswith((".json", ".csv")):
            raise ValueError("Only listed JSON/CSV evidence can be parsed")
        with zipfile.ZipFile(self.path) as archive:
            payload = archive.read(name)
        return json.loads(payload) if name.endswith(".json") else pd.read_csv(io.BytesIO(payload))


def save_original_results(payload, destination):
    """Validate an upload before replacing the cached ZIP; preserve it on failure."""
    def write(path):
        path.write_bytes(payload)
        ResultsArchive(path).require_research()
    try:
        atomic_write(destination, write)
    except zipfile.BadZipFile as exc:
        raise ValueError('The selected file is not a readable ZIP. Upload the original research results ZIP.') from exc
    return ResultsArchive(destination)


def validate_records(records, expected):
    """Bind saved outputs to immutable selection inputs, including unavailable labels."""
    if len(records) != len(expected) or records.duplicated(["SID", "ISO_TIME"]).any():
        raise ValueError("Prediction count/keys differ from selection")
    if records.SID.tolist() != expected.SID.tolist():
        raise ValueError("Prediction storm/order differs from selection")
    for col in ["ISO_TIME", "TARGET_TIME_6H"]:
        if pd.to_datetime(records[col], utc=True).tolist() != expected[col].tolist():
            raise ValueError(f"Prediction {col} differs from selection")
    for col in ["LAT", "LON", "WIND_KTS", "PRESSURE_HPA", "DIST2LAND", *TARGETS]:
        if not np.allclose(records[col], expected[col], rtol=1e-12, atol=1e-9, equal_nan=True):
            raise ValueError(f"Prediction input/target differs from selection: {col}")
    if not np.isfinite(records[FORECASTS].to_numpy(float)).all():
        raise ValueError("Non-finite forecast")


def persistence_wind(frame, forecast):
    """Causal rule: current observed wind; retain learned wind if current is missing."""
    result = np.asarray(forecast, dtype=float).copy()
    if result.shape != (len(frame), 4) or not np.isfinite(result).all():
        raise ValueError("Expected finite four-target forecasts")
    current = frame.WIND_KTS.to_numpy(float)
    available = np.isfinite(current)
    result[available, 2] = current[available]
    return result


def wind_summary(frame, forecast):
    truth = frame.TARGET_WIND_KTS_6H.to_numpy(float)
    current = frame.WIND_KTS.to_numpy(float)
    observed = np.isfinite(truth)
    paired = observed & np.isfinite(current)
    def mean(values):
        return float(np.mean(values)) if len(values) else None
    error = np.abs(np.asarray(forecast) - truth)
    base = np.abs(current-truth)
    return {"rows": len(frame), "storms": int(frame.SID.nunique()),
            "observed_labels": int(observed.sum()), "paired_labels": int(paired.sum()),
            "missing_current_observed_target": int((observed & ~np.isfinite(current)).sum()),
            "mae_all_observed_kts": mean(error[observed]), "mae_paired_kts": mean(error[paired]),
            "persistence_mae_paired_kts": mean(base[paired]),
            "paired_error_difference_kts": mean(error[paired]-base[paired]),
            "bias_all_observed_kts": mean((np.asarray(forecast)-truth)[observed])}


def wind_groups(frame, forecast):
    change = frame.TARGET_WIND_KTS_6H-frame.WIND_KTS
    groups = {"steady": change.eq(0), "strengthening": change.gt(0), "weakening": change.lt(0),
              "missing_current": frame.WIND_KTS.isna(), "missing_target": frame.TARGET_WIND_KTS_6H.isna()}
    intensity = pd.cut(frame.WIND_KTS, [-np.inf, 34, 64, np.inf], right=False,
                       labels=["under_34", "34_to_under_64", "64_or_above"])
    groups.update({"current_wind_"+str(band): intensity.eq(band) for band in intensity.cat.categories})
    groups.update({"sst_observed": frame.SST_C.notna(), "sst_missing": frame.SST_C.isna()})
    return {name: wind_summary(frame.loc[mask], np.asarray(forecast)[mask]) for name, mask in groups.items()}


def label_profile(frame, scale):
    paired = frame.WIND_KTS.notna() & frame.TARGET_WIND_KTS_6H.notna()
    changes = frame.loc[paired, "TARGET_WIND_KTS_6H"]-frame.loc[paired, "WIND_KTS"]
    return {"rows": len(frame), "paired_labels": len(changes), "unchanged": int(changes.eq(0).sum()),
            "unchanged_fraction": float(changes.eq(0).mean()), "wind_residual_scale_kts": float(scale),
            "fraction_paired_changes_in_smooth_l1_quadratic_region": float((changes.abs()/scale < 1).mean()),
            "change_counts_kts": {str(k): int(v) for k, v in changes.value_counts().sort_index().items()},
            "observed_target_missing_current": int((frame.WIND_KTS.isna() & frame.TARGET_WIND_KTS_6H.notna()).sum())}


def audit(results, output, package_path=DEFAULT_PACKAGE, bootstrap_replicates=2000):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new audit directory; completed evidence is immutable")
    source = ResultsArchive(results).require_research()
    package = TrainingPackage(package_path)
    package.verify_checksums()
    training = package.load("bob_train")
    validation = package.load("bob_validation")
    selection = validation.loc[validation_roles(validation) == "selection"].reset_index(drop=True)
    forecasts, comparison, source_map = {}, {}, None
    profiles, histories = {}, []
    for family in FAMILIES:
        aggregation = source.read(f"research/{family}/aggregation_status.json")
        if aggregation.get("status") != "completed" or aggregation.get("expected_seeds") != list(SEEDS) or not aggregation.get("complete_declared_seed_set"):
            raise ValueError("Incomplete declared research family")
        scores = []
        members = {}
        family_config = None
        for member in [*(f"seed_{s}" for s in SEEDS), "ensemble"]:
            prefix = f"research/{family}/{member}/"
            summary = source.read(prefix+"summary.json")
            if summary.get("status") != "completed" or summary.get("run_purpose") != "research_training":
                raise ValueError("Incomplete/non-research run")
            if any(summary[k] for k in ["calibration_used_for_selection", "test_parquet_decoded", "test_metrics_computed"]):
                raise ValueError("Export violated selection isolation")
            records = source.read(prefix+"selection_predictions.csv")
            validate_records(records, selection)
            forecast = records[FORECASTS].to_numpy(float)
            recomputed = prediction_records(selection, forecast)
            metrics = summarize_records(recomputed)
            if not np.isclose(metrics['selection_composite'], summary['selection_score'], atol=1e-10, rtol=0):
                raise ValueError("Saved score differs from recomputed predictions")
            members[member] = forecast
            if member != "ensemble":
                manifest = source.read(prefix+"run_manifest.json")
                if manifest["package_sha256"] != hashlib.sha256(package.path.read_bytes()).hexdigest():
                    raise ValueError("Source package mismatch")
                if source_map is None:
                    source_map = manifest["source_sha256"]
                if manifest["source_sha256"] != source_map:
                    raise ValueError("Mixed source versions in original comparison")
                if family_config is None:
                    family_config = manifest['config']
                if manifest['config'] != family_config:
                    raise ValueError('Mixed configurations within a family')
                roles = source.read(prefix+"validation_roles.csv")
                if roles.role.tolist() != validation_roles(validation).tolist() or roles.SID.tolist() != validation.SID.tolist():
                    raise ValueError("Saved validation roles differ")
                if pd.to_datetime(roles.ISO_TIME, utc=True).tolist() != validation.ISO_TIME.tolist():
                    raise ValueError('Saved validation role timestamps differ')
                if summary.get("checkpoint_reload_max_abs_difference") != 0:
                    raise ValueError("Checkpoint roundtrip was not exact")
                scores.append(metrics["selection_composite"])
                if family == "bay_mlp":
                    codec = summary["output_codec"]
                    profiles[member] = codec
                if family != "bay_tree":
                    history = source.read(prefix+"history.csv")
                    histories.append({"family": family, "member": member, "epochs": len(history),
                                      "best_epoch": summary['best_epoch'],
                                      "first_training_loss": float(history.train_loss_batch_weighted.iloc[0]),
                                      "last_training_loss": float(history.train_loss_batch_weighted.iloc[-1]),
                                      "note": "Total mixed-task training loss; original export has no per-epoch wind loss"})
        if not np.isclose(np.mean(scores), aggregation['family_selection_score_mean_of_seeds'], atol=1e-10, rtol=0):
            raise ValueError("Family ranking disagrees with individual seeds")
        # Wind ensembles must use the declared arithmetic mean in native units.
        np.testing.assert_allclose(members['ensemble'][:, 2:], np.mean([members[f'seed_{s}'][:, 2:] for s in SEEDS], axis=0), atol=1e-9, rtol=0)
        displacement = np.mean([log_map(selection.LAT.to_numpy(), selection.LON.to_numpy(),
                                        members[f'seed_{s}'][:, 0], members[f'seed_{s}'][:, 1]) for s in SEEDS], axis=0)
        expected_track = exp_map(selection.LAT.to_numpy(), selection.LON.to_numpy(), displacement)
        np.testing.assert_allclose(members['ensemble'][:, :2], expected_track, atol=1e-9, rtol=0)
        forecasts[family] = members
        comparison[family] = {"mean_seed_selection_score": float(np.mean(scores)),
                              "ensemble": summarize_records(prediction_records(selection, members['ensemble']))}
    selected = min(comparison, key=lambda f: comparison[f]['mean_seed_selection_score'])
    if selected != "bay_mlp":
        raise ValueError("This bounded follow-up was declared for the Bay MLP winner")
    # Complete validation before creating any deliverables.
    output.mkdir(parents=True)
    scale = profiles['seed_17']['residual_scales_km_km_kts_hpa'][2]
    baseline = forecasts[selected]['ensemble']
    hybrid = persistence_wind(selection, baseline)
    hybrid_records = prediction_records(selection, hybrid)
    hybrid_records.to_csv(output/'hybrid_selection_predictions.csv', index=False)
    hybrid_metrics = evaluate_records(hybrid_records, bootstrap_replicates, 1701)
    hybrid_seed_scores = {str(seed): summarize_records(prediction_records(
        selection, persistence_wind(selection, forecasts[selected][f'seed_{seed}'])))['selection_composite'] for seed in SEEDS}
    write_json(output/'hybrid_selection_metrics.json', hybrid_metrics)
    per_storm = []
    grouped = {}
    for family in FAMILIES:
        forecast = forecasts[family]['ensemble'][:, 2]
        grouped[family] = wind_groups(selection, forecast)
        for sid, storm in selection.groupby('SID'):
            per_storm.append(dict(family=family, SID=sid, **wind_summary(storm, forecast[storm.index])))
    pd.DataFrame(per_storm).to_csv(output/'wind_per_storm.csv', index=False)
    report = {"status": "passed", "original_export_sha256": source.sha256,
              "verified_files": len(source.manifest), "package_sha256": hashlib.sha256(package.path.read_bytes()).hexdigest(),
              "original_source_sha256": source_map, "audit_source_sha256": source_hashes(),
              "calibration_labels_used": False, "test_parquet_decoded": False,
              "selected_family_by_original_rule": selected, "comparison": comparison,
              "label_profiles": {"training": label_profile(training, scale), "selection": label_profile(selection, scale)},
              "wind_groups": grouped, "training_histories": histories,
              "output_codecs": profiles,
              "hybrid": {"status": "post_selection_followup_not_independent_confirmation",
                         "individual_seed_scores": hybrid_seed_scores,
                         "mean_seed_selection_score": float(np.mean(list(hybrid_seed_scores.values()))),
                         "rule": "Observed current wind, otherwise original MLP wind; original MLP track and pressure",
                         "full": hybrid_metrics['full'], "geographic_bay": hybrid_metrics['geographic_bay']},
              "inference_limit": "Outcome-change bands are diagnostic only; no forecast rule uses future wind",
              "uncertainty_limit": "Bootstrap on reused selection storms is descriptive after selection"}
    report['original_mlp_checkpoints'] = {str(seed): source.manifest[f'research/bay_mlp/seed_{seed}/best.pt'] for seed in SEEDS}
    write_json(output/'audit.json', report)
    config_path = Path(__file__).parent/'configs/wind_mae_followup_v1.json'
    config = json.loads(config_path.read_text())
    protocol = {"protocol_id": PROTOCOL_ID, "stage": "declared_before_new_training",
                "audit_sha256": digest(report),
                "original_results_sha256": source.sha256, "config_sha256": digest(config),
                "source_sha256": source_hashes(), "package_sha256": report['package_sha256'],
                "candidate_set": ["original_bay_mlp", "bay_mlp_wind_persistence", "bay_mlp_wind_absolute_loss"],
                "seeds": list(SEEDS), "new_training_runs": 3,
                "only_training_change": "Wind uses absolute residual error; track/pressure remain SmoothL1",
                "selection_rule": "Lowest mean individual-seed composite; exact ties retain earlier candidate in this list",
                "no_further_search": True, "calibration_labels_used": False, "test_parquet_decoded": False,
                "limits": "Follow-up is informed by reused 2011-2014 selection results; no independent confirmation or wind-improvement claim from hybrid persistence"}
    write_json(output/'followup_protocol.json', protocol)
    write_json(output/'training_history_diagnostics.json', histories)
    text = ["# Wind audit and bounded follow-up", "", "Original research evidence passed integrity and selection checks.", "",
            "| MLP wind outcome band | Paired labels | MLP MAE (kt) | Persistence MAE (kt) |",
            "|---|---:|---:|---:|"]
    for band in ['steady', 'strengthening', 'weakening']:
        g = grouped['bay_mlp'][band]
        text.append(f"| {band} | {g['paired_labels']} | {g['mae_paired_kts']:.3f} | {g['persistence_mae_paired_kts']:.3f} |")
    text += ["", f"Training unchanged wind: {report['label_profiles']['training']['unchanged_fraction']:.1%}; selection: {report['label_profiles']['selection']['unchanged_fraction']:.1%}.",
             "", f"Hybrid composite: {hybrid_metrics['full']['selection_composite']:.6f}; wind persistence is an explicit rule, not learned wind skill.",
             "", "The wind loss/metric mismatch is a hypothesis, not an established cause. The original logs do not contain per-epoch wind loss.",
             "", "One new three-seed MLP experiment changes only the wind loss. See followup_protocol.json and docs/wind-followup.md.",
             "", "Selection storms have been reused. Calibration and final-test evaluation remain pending."]
    (output/'README.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--package', default=str(DEFAULT_PACKAGE))
    args = parser.parse_args()
    result = audit(args.results, args.output, args.package)
    print(json.dumps({"status": result['status'], "verified_files": result['verified_files'],
                      "selected_family": result['selected_family_by_original_rule'], "output": args.output}, indent=2))


if __name__ == '__main__':
    main()
