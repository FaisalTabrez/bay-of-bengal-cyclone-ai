"""Paired metrics, selection scoring and whole-storm bootstrap uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import predictions
from .geometry import haversine_km

TARGETS = ["TARGET_LAT_6H", "TARGET_LON_6H", "TARGET_WIND_KTS_6H", "TARGET_PRESSURE_HPA_6H"]
FORECASTS = ["PRED_LAT_6H", "PRED_LON_6H", "PRED_WIND_KTS_6H", "PRED_PRESSURE_HPA_6H"]


def prediction_records(frame, forecast):
    forecast = np.asarray(forecast, dtype=float)
    if forecast.shape != (len(frame), 4) or not np.isfinite(forecast).all():
        raise ValueError("Expected one finite four-target prediction per input row")
    result = frame[["SID", "ISO_TIME", "TARGET_TIME_6H", "LAT", "LON", "WIND_KTS", "PRESSURE_HPA", "DIST2LAND"] + TARGETS].copy()
    result[FORECASTS] = forecast
    result["INSIDE_BAY"] = frame.LAT.between(5, 25) & frame.LON.between(78, 100)
    result["TRACK_ERROR_KM"] = haversine_km(frame.TARGET_LAT_6H, frame.TARGET_LON_6H, forecast[:, 0], forecast[:, 1])
    cv = predictions(frame, tendency=True)
    result["BASE_TRACK_ERROR_KM"] = haversine_km(frame.TARGET_LAT_6H, frame.TARGET_LON_6H, cv[:, 0], cv[:, 1])
    result["LAT_ABS_ERROR_DEG"] = np.abs(forecast[:, 0] - frame.TARGET_LAT_6H)
    result["LON_ABS_ERROR_DEG"] = np.abs((forecast[:, 1] - frame.TARGET_LON_6H + 180) % 360 - 180)
    for j, variable, unit in [(2, "WIND", "KTS"), (3, "PRESSURE", "HPA")]:
        actual = frame[TARGETS[j]].to_numpy(float)
        current = frame[f"{variable}_{unit}"].to_numpy(float)
        result[f"{variable}_ABS_ERROR"] = np.abs(actual - forecast[:, j])
        paired = np.isfinite(actual) & np.isfinite(current)
        result[f"{variable}_PAIRED_ERROR"] = np.where(paired, np.abs(actual - forecast[:, j]), np.nan)
        result[f"{variable}_BASE_ERROR"] = np.where(paired, np.abs(actual - current), np.nan)
    return result


def mean_or_none(series):
    return float(series.mean()) if series.notna().any() else None


def summarize_records(records):
    if records.empty:
        return {"rows": 0, "storms": 0}
    track = records.TRACK_ERROR_KM
    result = {"rows": len(records), "storms": int(records.SID.nunique()),
              "track_mean_km": float(track.mean()), "track_median_km": float(track.median()),
              "track_p90_km": float(track.quantile(.9)),
              "latitude_mae_deg": float(records.LAT_ABS_ERROR_DEG.mean()),
              "longitude_mae_deg": float(records.LON_ABS_ERROR_DEG.mean()),
              "track_storm_macro_mean_km": float(records.groupby("SID").TRACK_ERROR_KM.mean().mean())}
    ratios = [track.mean() / records.BASE_TRACK_ERROR_KM.mean()]
    result["track_skill_vs_constant_velocity"] = float(1 - ratios[0])
    for variable, unit in [("WIND", "kts"), ("PRESSURE", "hpa")]:
        prefix = variable.lower()
        errors = records[f"{variable}_ABS_ERROR"]
        paired = records[f"{variable}_PAIRED_ERROR"]
        baseline = records[f"{variable}_BASE_ERROR"]
        result.update({f"{prefix}_mae_{unit}_all_observed": mean_or_none(errors),
                       f"{prefix}_observed_labels": int(errors.notna().sum()),
                       f"{prefix}_mae_{unit}_paired": mean_or_none(paired),
                       f"{prefix}_paired_labels": int(paired.notna().sum()),
                       f"{prefix}_persistence_mae_{unit}_paired": mean_or_none(baseline),
                       f"{prefix}_storm_macro_mae_{unit}": mean_or_none(records.groupby("SID")[f"{variable}_ABS_ERROR"].mean())})
        if paired.notna().any() and baseline.mean() > 0:
            ratio = float(paired.mean() / baseline.mean())
            result[f"{prefix}_skill_vs_persistence"] = 1 - ratio
            ratios.append(ratio)
        else:
            result[f"{prefix}_skill_vs_persistence"] = None
    result["selection_composite"] = float(np.mean(ratios)) if len(ratios) == 3 else None
    return result


def storm_bootstrap(records, replicates=2000, seed=1701):
    if replicates < 1:
        return {"replicates": 0, "status": "not_requested"}
    # Group sums and counts preserve all rows when a storm is drawn multiple times.
    columns = ["TRACK_ERROR_KM", "BASE_TRACK_ERROR_KM", "WIND_PAIRED_ERROR", "WIND_BASE_ERROR",
               "PRESSURE_PAIRED_ERROR", "PRESSURE_BASE_ERROR", "WIND_ABS_ERROR", "PRESSURE_ABS_ERROR"]
    group = records.groupby("SID")[columns]
    sums, counts = group.sum().to_numpy(), group.count().to_numpy()
    if len(sums) < 2:
        return {"replicates": 0, "status": "fewer_than_two_storms"}
    rng = np.random.default_rng(seed)
    sampled = []
    for _ in range(replicates):
        indices = rng.integers(0, len(sums), size=len(sums))
        denominator = counts[indices].sum(axis=0)
        means = np.divide(sums[indices].sum(axis=0), denominator,
                          out=np.full(len(columns), np.nan), where=denominator > 0)
        sampled.append([means[0], means[6], means[7],
                        means[0] - means[1], means[2] - means[3], means[4] - means[5]])
    values = np.asarray(sampled)
    names = ["track_mean_km", "wind_mae_kts_all_observed", "pressure_mae_hpa_all_observed",
             "paired_track_error_difference_km", "paired_wind_error_difference_kts", "paired_pressure_error_difference_hpa"]
    intervals = {}
    for j, name in enumerate(names):
        valid = values[np.isfinite(values[:, j]), j]
        intervals[name] = {"low": float(np.quantile(valid, .025)) if len(valid) else None,
                           "high": float(np.quantile(valid, .975)) if len(valid) else None,
                           "valid_replicates": len(valid)}
    return {"replicates": replicates, "seed": seed, "unit": "whole_cyclone",
            "interval": "95% percentile, descriptive after model selection", "intervals": intervals}


def evaluate_records(records, bootstrap_replicates=2000, bootstrap_seed=1701):
    result = {}
    for label, subset in [("full", records), ("geographic_bay", records.loc[records.INSIDE_BAY])]:
        result[label] = summarize_records(subset)
        result[label]["bootstrap"] = storm_bootstrap(subset, bootstrap_replicates, bootstrap_seed)
    # Diagnostics use input-time conditions; unknown intensity is preserved.
    distance = pd.cut(records.DIST2LAND, [-np.inf, 50, 200, np.inf], labels=["0_to_50_km", "50_to_200_km", "over_200_km"])
    intensity = pd.cut(records.WIND_KTS, [-np.inf, 34, 64, np.inf], right=False,
                       labels=["under_34_kt", "34_to_under_64_kt", "64_kt_or_above"]).astype(object).fillna("unknown")
    result["land_proximity"] = {str(label): summarize_records(records.loc[distance == label]) for label in distance.cat.categories}
    result["current_intensity_diagnostic"] = {str(label): summarize_records(records.loc[intensity == label])
                                              for label in ["under_34_kt", "34_to_under_64_kt", "64_kt_or_above", "unknown"]}
    result["current_intensity_note"] = "Diagnostic bands, not harmonized IMD cyclone categories"
    return result
