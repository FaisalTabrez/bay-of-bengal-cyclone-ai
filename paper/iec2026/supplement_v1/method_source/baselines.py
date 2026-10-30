"""Handoff-compatible baseline definitions and explicit evaluation denominators."""
import numpy as np

from .geometry import haversine_km


def predictions(frame, tendency=False):
    cols = ["LAT", "LON", "WIND_KTS", "PRESSURE_HPA"]
    out = frame[cols].to_numpy(float).copy()
    if tendency:
        changes = ["LAT_CHANGE_6H", "LON_CHANGE_6H", "WIND_CHANGE_6H_KTS", "PRESSURE_CHANGE_6H_HPA"]
        out += frame[changes].fillna(0).to_numpy(float)
    return out


def evaluate(frame, predicted):
    truth = frame[["TARGET_LAT_6H", "TARGET_LON_6H", "TARGET_WIND_KTS_6H", "TARGET_PRESSURE_HPA_6H"]].to_numpy(float)
    bay = (frame.LAT.between(5, 25) & frame.LON.between(78, 100)).to_numpy()
    result = {}
    for suffix, subset in [("all", np.ones(len(frame), bool)), ("bay", bay)]:
        valid = subset & np.isfinite(predicted[:, :2]).all(1) & np.isfinite(truth[:, :2]).all(1)
        errors = haversine_km(truth[valid, 0], truth[valid, 1], predicted[valid, 0], predicted[valid, 1])
        result[f"track_{suffix}"] = {
            "mean_km": float(errors.mean()) if len(errors) else None,
            "median_km": float(np.median(errors)) if len(errors) else None,
            "p90_km": float(np.percentile(errors, 90)) if len(errors) else None,
            "samples": int(valid.sum()),
        }
        for j, variable, unit in [(2, "wind", "kts"), (3, "pressure", "hpa")]:
            valid = subset & np.isfinite(truth[:, j]) & np.isfinite(predicted[:, j])
            prefix = "bay_" if suffix == "bay" else ""
            result[f"{prefix}{variable}_mae_{unit}"] = (
                float(np.abs(truth[valid, j] - predicted[valid, j]).mean()) if valid.any() else None
            )
            result[f"{prefix}{variable}_samples"] = int(valid.sum())
    return result
