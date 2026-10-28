"""Verify the modelling design against train/validation; write inspectable receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .baselines import evaluate, predictions
from .data import TrainingPackage, causal_windows, validation_roles
from .model import CycloneSequences, ResidualCodec, ResidualGRU, collate_sequences, masked_multitask_loss
from .paths import DEFAULT_ACCEPTANCE, DEFAULT_PACKAGE, REPORTS_DIR


def compare_nested(actual, expected, path=""):
    for key, value in expected.items():
        here = f"{path}/{key}"
        if isinstance(value, dict):
            compare_nested(actual[key], value, here)
        elif value is None:
            if actual[key] is not None:
                raise ValueError(f"Baseline mismatch: {here}")
        elif not np.isclose(actual[key], value, atol=1e-8, rtol=1e-10):
            raise ValueError(f"Baseline mismatch: {here}: {actual[key]} vs {value}")


def summarize_frame(frame):
    windows = causal_windows(frame)
    gaps = frame.groupby("SID").ISO_TIME.diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=6)).all():
        raise ValueError("Unexpected source gap in handoff")
    for i, window in enumerate(windows):
        if not frame.SID.iloc[window].eq(frame.SID.iloc[i]).all() or (window > i).any():
            raise ValueError("Non-causal sequence")
    return {
        "rows": len(frame), "storms": int(frame.SID.nunique()),
        "first_input": str(frame.ISO_TIME.min()), "last_input": str(frame.ISO_TIME.max()),
        "context_length_counts": pd.Series([len(w) for w in windows]).value_counts().sort_index().to_dict(),
        "missing_columns": {c: int(n) for c, n in frame.isna().sum().items() if n},
        "observed_target_missing_current": {
            c: int((frame[f"TARGET_{c}_6H"].notna() & frame[c].isna()).sum())
            for c in ["WIND_KTS", "PRESSURE_HPA"]
        },
        "rh700_above_100_count_informational": int((frame.ERA5_RH700_PERCENT > 100).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--acceptance", default=str(DEFAULT_ACCEPTANCE))
    parser.add_argument("--output", default=str(REPORTS_DIR))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    np.random.seed(17)
    package = TrainingPackage(args.package)
    checksums = package.verify_checksums()
    frames = {s: package.load(s) for s in ["nio_pretrain_train", "bob_train", "bob_validation"]}
    nio, bob, validation = [frames[s] for s in frames]
    if set(nio.SID) & set(validation.SID) or set(bob.SID) & set(validation.SID):
        raise ValueError("Training-validation cyclone overlap")
    columns = ["SID", "ISO_TIME"]
    shared = nio.set_index(columns).loc[bob.set_index(columns).index].reset_index()
    pd.testing.assert_frame_equal(shared[bob.columns], bob)
    normalizers = {stage: package.normalizer(stage) for stage in ["nio", "bob"]}
    for stage, training in [("nio", nio), ("bob", bob)]:
        normalizers[stage].verify_training_statistics(training)
    baseline = {"persistence": evaluate(validation, predictions(validation)),
                "constant_velocity": evaluate(validation, predictions(validation, tendency=True))}
    with zipfile.ZipFile(args.acceptance) as archive:
        expected = json.loads(archive.read("validation_baselines.json"))
    compare_nested(baseline, expected)
    (output / "validation_baselines_reproduced.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    roles = validation_roles(validation)
    roles_frame = validation[["SID", "ISO_TIME"]].copy()
    roles_frame["role"] = roles
    roles_frame.to_csv(output / "validation_roles.csv", index=False)
    role_summary = {}
    for role in ["selection", "calibration"]:
        subset = validation.loc[roles == role].reset_index(drop=True)
        role_summary[role] = {"rows": len(subset), "storms": int(subset.SID.nunique()),
                              "first_input": str(subset.ISO_TIME.min()), "last_input": str(subset.ISO_TIME.max())}
    # Pre-register the comparison denominators on selection storms only.
    selection = validation.loc[roles == "selection"].reset_index(drop=True)
    selection_reference = {
        "persistence": evaluate(selection, predictions(selection)),
        "constant_velocity": evaluate(selection, predictions(selection, tendency=True)),
    }
    codec = ResidualCodec.fit(nio, normalizers["nio"])
    dataset = CycloneSequences(nio, normalizers["nio"], codec)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_sequences)
    x, lengths, y, mask = next(iter(loader))
    config = json.loads((Path(__file__).parent / "configs/residual_gru_v0.json").read_text(encoding="utf-8"))
    model = ResidualGRU(input_size=2 * len(package.features), **{
        k: config[k] for k in ["projection_size", "hidden_size", "head_size", "dropout"]
    })
    model.eval()
    with torch.no_grad():
        initial = model(x, lengths)
    if not torch.equal(initial, torch.zeros_like(initial)):
        raise ValueError("Zero-head initialization did not reproduce the anchor")
    # One optimizer step is an execution check; no validation training/tuning or checkpoint.
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss = masked_multitask_loss(model(x, lengths), y, mask)
    loss.backward()
    finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    if not torch.isfinite(loss).item() or not finite:
        raise ValueError("Non-finite smoke loss/gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    # Check transfer on the same physical Bay inputs with actual missing values.
    old_dataset = CycloneSequences(bob, normalizers["nio"], codec)
    new_dataset = CycloneSequences(bob, normalizers["bob"], codec)
    old_x, lens, _, _ = next(iter(DataLoader(old_dataset, batch_size=256, collate_fn=collate_sequences)))
    new_x, new_lens, _, _ = next(iter(DataLoader(new_dataset, batch_size=256, collate_fn=collate_sequences)))
    model.eval()
    with torch.no_grad():
        before = model(old_x, lens)
        model.rebase_inputs(normalizers["nio"], normalizers["bob"])
        after = model(new_x, new_lens)
    max_difference = float((before - after).abs().max())
    if max_difference > 1e-5:
        raise ValueError("Normalization transfer changed the forecast")
    try:
        package.load("bob_test")
    except PermissionError:
        test_guard = True
    else:
        raise ValueError("Frozen-test guard failed")
    source_hashes = {p.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.rglob("*.py"))}
    report = {
        "status": "design_contract_checks_passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_sha256": hashlib.sha256(Path(args.package).read_bytes()).hexdigest(),
        "internal_file_hashes_verified": checksums,
        "test_parquet_decoded": False, "test_metrics_computed": False, "test_loader_guard_passed": test_guard,
        "partitions_checked": {s: summarize_frame(f) for s, f in frames.items()},
        "nio_bob_shared_rows_identical": len(shared),
        "nio_extra_training_rows": len(nio) - len(bob),
        "nio_extra_training_storms": int(nio.SID.nunique() - bob.SID.nunique()),
        "training_normalization_matches_supplied_statistics": True,
        "validation_baselines_match_acceptance_at_absolute_tolerance": 1e-8,
        "validation_roles": role_summary,
        "selection_baseline_reference": selection_reference,
        "prototype": {"parameters": sum(p.numel() for p in model.parameters()),
                      "input_size": 2 * len(package.features), "context_observations": 4,
                      "zero_head_reproduces_anchor": True, "smoke_optimizer_steps": 1,
                      "smoke_batch_rows": len(x), "smoke_loss_not_performance": float(loss.detach()),
                      "finite_gradients": finite, "normalization_transfer_max_abs_output_difference": max_difference,
                      "output_codec": codec.as_dict(), "trained_checkpoint_saved": False},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "torch": torch.__version__, "device": "cpu", "seed": 17, "torch_threads": 2},
        "source_sha256": source_hashes,
    }
    (output / "design_checks.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output / "design_checks.json"),
                      "parameters": report["prototype"]["parameters"], "test_parquet_decoded": False}, indent=2))


if __name__ == "__main__":
    main()
