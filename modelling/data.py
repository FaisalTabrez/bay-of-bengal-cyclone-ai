"""Read immutable training/validation data; never decode the frozen test set."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_SHA256 = "3df324f20f373a8c868bc8f4b29449fb79226725e9e3705efbe898f83e67cd42"
ALLOWED_PARTITIONS = {"nio_pretrain_train", "bob_train", "bob_validation"}
COUNTS = {"nio_pretrain_train": 5557, "bob_train": 4560, "bob_validation": 854}


class TrainingPackage:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != PACKAGE_SHA256:
            raise ValueError("Package SHA-256 differs from the immutable v2 handoff")
        self.manifest = self.read_json("training_manifest_v2.json")
        self.features = self.manifest["feature_columns"]
        self.targets = self.manifest["target_columns"]

    def read_json(self, name: str):
        if "/" in name or not name.endswith(".json"):
            raise ValueError("Only root metadata JSON is accessible here")
        with zipfile.ZipFile(self.path) as archive:
            return json.loads(archive.read(name))

    def verify_checksums(self):
        checksums = self.read_json("package_file_checksums.json")
        with zipfile.ZipFile(self.path) as archive:
            for name, expected in checksums.items():
                # Raw byte integrity only, including test bytes; no test decoding.
                payload = archive.read(name)
                if len(payload) != expected["size_bytes"]:
                    raise ValueError(f"Size mismatch: {name}")
                if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                    raise ValueError(f"Hash mismatch: {name}")
        return len(checksums)

    def load(self, partition: str) -> pd.DataFrame:
        if partition not in ALLOWED_PARTITIONS:
            raise PermissionError("Design-stage loader permits training and validation only")
        with zipfile.ZipFile(self.path) as archive:
            names = [n for n in archive.namelist()
                     if n.startswith(partition + "/") and n.endswith(".parquet")]
            if len(names) != 1:
                raise ValueError(f"Expected one Parquet part for {partition}")
            frame = pd.read_parquet(io.BytesIO(archive.read(names[0])))
        expected = self.manifest["identifier_columns"] + self.features + self.targets
        if list(frame.columns) != expected or len(frame) != COUNTS[partition]:
            raise ValueError(f"Unexpected schema/count for {partition}")
        frame = frame.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
        if frame.duplicated(["SID", "ISO_TIME"]).any():
            raise ValueError("Duplicate observation keys")
        if not frame.TARGET_TIME_6H.sub(frame.ISO_TIME).eq(pd.Timedelta(hours=6)).all():
            raise ValueError("Non-six-hour target")
        if frame[self.targets[:2]].isna().any().any():
            raise ValueError("Missing track target")
        if not frame.SST_AVAILABLE.eq(frame.SST_C.notna().astype(int)).all():
            raise ValueError("SST flag disagrees with observed SST")
        if not np.isfinite(frame[self.features].dropna(axis=1).to_numpy(float)).all():
            raise ValueError("Non-finite complete feature")
        for col in self.features:
            values = frame[col].to_numpy(float)
            if np.isinf(values).any():
                raise ValueError(f"Infinite feature: {col}")
        years = frame.ISO_TIME.dt.year
        lo, hi = (2011, 2017) if partition == "bob_validation" else (1982, 2010)
        if not years.between(lo, hi).all() or frame.SPLIT.nunique() != 1:
            raise ValueError("Partition chronology or split contract failed")
        return frame

    def normalizer(self, stage: str):
        if stage not in {"bob", "nio"}:
            raise ValueError("stage must be bob or nio")
        return Normalizer(self.features, self.read_json(f"normalization_{stage}_train.json"))


class Normalizer:
    """39 standardized features followed by 39 binary missingness indicators."""
    def __init__(self, features, statistics):
        self.features = list(features)
        self.statistics = statistics
        self.mean = np.array([statistics[c]["mean"] for c in features], dtype=float)
        self.std = np.array([statistics[c]["std"] for c in features], dtype=float)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Non-finite normalization statistics")
        self.std = np.where(self.std > 1e-8, self.std, 1.0)

    def transform(self, frame):
        values = frame[self.features].to_numpy(float)
        missing = ~np.isfinite(values)
        standardized = np.where(missing, 0.0, (values - self.mean) / self.std)
        return np.concatenate([standardized, missing.astype(float)], axis=1).astype(np.float32)

    def verify_training_statistics(self, frame):
        for col in self.features:
            s = frame[col]
            expected = self.statistics[col]
            if int(s.notna().sum()) != expected["valid_count"]:
                raise ValueError(f"Training count mismatch: {col}")
            if not np.isclose(s.mean(), expected["mean"], atol=1e-8, rtol=1e-10):
                raise ValueError(f"Training mean mismatch: {col}")
            if not np.isclose(s.std(ddof=1), expected["std"], atol=1e-8, rtol=1e-10):
                raise ValueError(f"Training std mismatch: {col}")


def validation_roles(frame):
    """Assign whole storms to later calibration if any input is in 2015 onward."""
    last = frame.groupby("SID").ISO_TIME.max()
    late = set(last.index[last >= pd.Timestamp("2015-01-01", tz="UTC")])
    return np.where(frame.SID.isin(late), "calibration", "selection")


def causal_windows(frame, context=4):
    """Index only current/past input rows in the same contiguous cyclone segment."""
    if context < 1:
        raise ValueError("context must be positive")
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError("Reset frame index before building windows")
    if not frame.equals(frame.sort_values(["SID", "ISO_TIME"])):
        raise ValueError("Sort frame by SID and ISO_TIME first")
    windows = []
    start = 0
    for i in range(len(frame)):
        if i == 0 or frame.SID.iloc[i] != frame.SID.iloc[i - 1] or (
            frame.ISO_TIME.iloc[i] - frame.ISO_TIME.iloc[i - 1] != pd.Timedelta(hours=6)
        ):
            start = i
        windows.append(np.arange(max(start, i - context + 1), i + 1))
    return windows
