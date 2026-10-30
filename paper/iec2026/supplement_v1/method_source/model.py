"""Prototype residual GRU, causal sequence dataset, and masked deterministic loss."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset

from .data import causal_windows
from .geometry import log_map, exp_map


class ResidualCodec:
    """Keep scales and anchor fallback frozen through transfer; fit on training only."""
    def __init__(self, fallback, scales=None):
        self.fallback = np.asarray(fallback, dtype=float)
        self.scales = None if scales is None else np.asarray(scales, dtype=float)

    @classmethod
    def fit(cls, training, normalizer):
        codec = cls([normalizer.statistics[c]["mean"] for c in ["WIND_KTS", "PRESSURE_HPA"]])
        residuals = codec.raw_residuals(training)
        track_scale = max(float(np.sqrt(np.mean(residuals[:, :2] ** 2))), 1.0)
        codec.scales = np.array([track_scale, track_scale] + [
            max(float(np.sqrt(np.nanmean(residuals[:, j] ** 2))), 1.0) for j in [2, 3]
        ])
        return codec

    def anchors(self, frame):
        lat, lon = frame.LAT.to_numpy(float), frame.LON.to_numpy(float)
        base_lat = lat + frame.LAT_CHANGE_6H.fillna(0).to_numpy(float)
        base_lon = lon + frame.LON_CHANGE_6H.fillna(0).to_numpy(float)
        track = log_map(lat, lon, base_lat, base_lon)
        intensity = frame[["WIND_KTS", "PRESSURE_HPA"]].to_numpy(float)
        intensity = np.where(np.isfinite(intensity), intensity, self.fallback)
        return np.concatenate([track, intensity], axis=1)

    def raw_residuals(self, frame):
        lat, lon = frame.LAT.to_numpy(float), frame.LON.to_numpy(float)
        future = log_map(lat, lon, frame.TARGET_LAT_6H.to_numpy(float), frame.TARGET_LON_6H.to_numpy(float))
        future = np.concatenate([future, frame[["TARGET_WIND_KTS_6H", "TARGET_PRESSURE_HPA_6H"]].to_numpy(float)], axis=1)
        return future - self.anchors(frame)

    def encode(self, frame):
        y = (self.raw_residuals(frame) / self.scales).astype(np.float32)
        return y, np.isfinite(y)

    def decode(self, frame, predicted):
        physical = np.asarray(predicted) * self.scales + self.anchors(frame)
        track = exp_map(frame.LAT.to_numpy(float), frame.LON.to_numpy(float), physical[:, :2])
        return np.concatenate([track, physical[:, 2:]], axis=1)

    def as_dict(self):
        return {"fallback_wind_kts_pressure_hpa": self.fallback.tolist(),
                "residual_scales_km_km_kts_hpa": self.scales.tolist(),
                "transfer_policy": "Keep this codec fixed during fine-tuning"}

    @classmethod
    def from_dict(cls, state):
        return cls(state["fallback_wind_kts_pressure_hpa"], state["residual_scales_km_km_kts_hpa"])


class CycloneSequences(Dataset):
    def __init__(self, frame, normalizer, codec, context=4):
        self.x = normalizer.transform(frame)
        self.y, self.mask = codec.encode(frame)
        self.windows = causal_windows(frame, context)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return (torch.from_numpy(self.x[self.windows[i]]),
                torch.from_numpy(self.y[i]), torch.from_numpy(self.mask[i]))


def collate_sequences(batch):
    sequences, targets, masks = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True), lengths, torch.stack(targets), torch.stack(masks)


class ResidualGRU(nn.Module):
    def __init__(self, input_size=78, projection_size=32, hidden_size=64, head_size=32, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_size, projection_size)
        self.gru = nn.GRU(projection_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size + projection_size, head_size), nn.SiLU(), nn.Linear(head_size, n))
            for n in [2, 1, 1]
        ])
        # The first prediction reproduces the physical anchor, rather than random absolute positions.
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x, lengths):
        projected = nn.functional.silu(self.input_linear(x))
        packed = pack_padded_sequence(projected, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        current = projected[torch.arange(len(x), device=x.device), lengths.to(x.device) - 1]
        encoded = self.dropout(torch.cat([hidden[-1], current], dim=-1))
        return torch.cat([head(encoded) for head in self.heads], dim=-1)

    def rebase_inputs(self, old, new):
        """Preserve predictions when switching training means/stds, including nulls."""
        if old.features != new.features or self.input_linear.in_features != 2 * len(old.features):
            raise ValueError("Input contract mismatch for normalization transfer")
        with torch.no_grad():
            weight = self.input_linear.weight
            n = len(old.features)
            original = weight[:, :n].clone()
            a = torch.as_tensor(new.std / old.std, dtype=weight.dtype, device=weight.device)
            b = torch.as_tensor((new.mean - old.mean) / old.std, dtype=weight.dtype, device=weight.device)
            weight[:, :n] = original * a
            weight[:, n:] -= original * b  # Cancel the mean shift for imputed values.
            self.input_linear.bias += original @ b


class ResidualMLP(nn.Module):
    """Snapshot neural control; the shared four-output layer retains task masking."""
    def __init__(self, input_size=78, widths=(64, 32), dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, widths[0]), nn.SiLU(),
                                     nn.Dropout(dropout), nn.Linear(widths[0], widths[1]),
                                     nn.SiLU(), nn.Linear(widths[1], 4))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x, lengths):
        current = x[torch.arange(len(x), device=x.device), lengths.to(x.device) - 1]
        return self.network(current)


def masked_multitask_loss(predicted, target, mask, wind_loss="smooth_l1"):
    if wind_loss not in {"smooth_l1", "absolute"}:
        raise ValueError("wind_loss must be smooth_l1 or absolute")
    losses = []
    for columns in [(0, 1), (2,), (3,)]:
        p, t, m = predicted[:, columns], target[:, columns], mask[:, columns]
        # Index before loss computation: NaN labels are never converted into zeros.
        if m.any():
            loss = nn.functional.l1_loss if columns == (2,) and wind_loss == "absolute" else nn.functional.smooth_l1_loss
            losses.append(loss(p[m], t[m]))
    if not losses:
        raise ValueError("No observed targets in batch")
    return torch.stack(losses).mean()
