"""Deterministic experiment CLI. Calibration is excluded; frozen test is denied."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import TrainingPackage, validation_roles
from .experiment_io import (atomic_write, capture_rng, digest, load_checkpoint, mirror_run,
                            restore_rng, runtime_info, save_checkpoint, seed_everything,
                            source_hashes, write_json)
from .geometry import exp_map, log_map
from .metrics import FORECASTS, evaluate_records, prediction_records, summarize_records
from .model import (CycloneSequences, ResidualCodec, ResidualGRU, ResidualMLP,
                    collate_sequences, masked_multitask_loss)
from .paths import DEFAULT_PACKAGE, RUNS_DIR

REGIMES = ["bay_gru", "transfer_gru", "pooled_gru", "bay_mlp", "bay_tree", "bay_budget"]
DEFAULT_CONFIG = Path(__file__).parent / "configs/residual_gru_v0.json"


def load_config(path=DEFAULT_CONFIG, acceptance=False):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config["run_purpose"] = "acceptance_not_research" if acceptance else "research_training"
    if config.get("wind_loss", "smooth_l1") not in {"smooth_l1", "absolute"}:
        raise ValueError("Unsupported wind_loss")
    if acceptance:
        plan = config["training_plan"]
        plan.update(seeds=[17], pretrain_epochs_fixed=1, finetune_max_epochs=2,
                    bay_only_max_epochs=2, early_stopping_patience=2)
        config["bootstrap_replicates"] = 40
        config["tree_iterations"] = 3
    if config["validation_selection_input_years"] != [2011, 2014] or config["validation_calibration_input_years"] != [2015, 2017]:
        raise ValueError("This release locks the preregistered validation roles")
    for key in ["pretrain_epochs_fixed", "finetune_max_epochs", "bay_only_max_epochs", "batch_size", "early_stopping_patience"]:
        if config["training_plan"][key] < 1:
            raise ValueError(f"Invalid positive setting: {key}")
    if config["context_observations"] < 1 or config["input_features"] != 39 or config["missing_indicators"] != 39:
        raise ValueError("Unsupported input contract")
    return config


class Inputs:
    def __init__(self, path):
        self.package = TrainingPackage(path)
        self.package.verify_checksums()
        self.frames = {name: self.package.load(name) for name in ["nio_pretrain_train", "bob_train"]}
        validation = self.package.load("bob_validation")
        roles = validation_roles(validation)
        self.roles = validation[["SID", "ISO_TIME"]].copy()
        self.roles["role"] = roles
        self.role_hash = hashlib.sha256(self.roles.to_csv(index=False).encode()).hexdigest()
        # No calibration label/frame is retained for the experiment API.
        self.selection = validation.loc[roles == "selection"].reset_index(drop=True)
        self.norms = {stage: self.package.normalizer(stage) for stage in ["nio", "bob"]}
        for stage, partition in [("nio", "nio_pretrain_train"), ("bob", "bob_train")]:
            training = self.frames[partition]
            self.norms[stage].verify_training_statistics(training)
            if set(training.SID) & set(validation.SID):
                raise ValueError("Training/validation storm overlap")


def build_model(regime, config):
    if regime == "bay_mlp":
        return ResidualMLP(input_size=78, widths=config.get("mlp_widths", [64, 32]), dropout=config["dropout"])
    return ResidualGRU(input_size=78, **{k: config[k] for k in ["projection_size", "hidden_size", "head_size", "dropout"]})


def stage_plan(regime, config, budget_updates=None):
    p = config["training_plan"]
    pre = {"name": "nio_pretrain", "partition": "nio_pretrain_train", "norm": "nio",
           "epochs": p["pretrain_epochs_fixed"], "lr": p["pretrain_learning_rate"], "select": False}
    bay = {"name": "bay_train", "partition": "bob_train", "norm": "bob",
           "epochs": p["bay_only_max_epochs"], "lr": p["bay_only_learning_rate"], "select": True}
    if regime == "transfer_gru":
        fine = dict(bay, name="bay_finetune", epochs=p["finetune_max_epochs"], lr=p["finetune_learning_rate"])
        return [pre, fine]
    if regime == "pooled_gru":
        return [dict(pre, name="nio_pooled", select=True, fixed=True)]
    if regime == "bay_budget":
        if budget_updates is None or budget_updates < 1:
            raise ValueError("bay_budget requires --budget-updates from the transfer run's global_step")
        epochs = math.ceil(budget_updates / math.ceil(4560 / p["batch_size"]))
        return [dict(bay, name="bay_budget", epochs=epochs, update_cap=budget_updates, fixed=True)]
    return [bay]


def predict(model, frame, norm, codec, config, device):
    dataset = CycloneSequences(frame, norm, codec, config["context_observations"])
    loader = DataLoader(dataset, batch_size=config["training_plan"]["batch_size"],
                        collate_fn=collate_sequences, generator=torch.Generator().manual_seed(0))
    values = []
    model.eval()
    with torch.no_grad():
        for x, lengths, _, _ in loader:
            values.append(model(x.to(device), lengths).cpu().numpy())
    return codec.decode(frame, np.concatenate(values))


def report_forecast(run_dir, frame, forecast, config, metadata):
    records = prediction_records(frame, forecast)
    atomic_write(Path(run_dir) / "selection_predictions.csv", lambda p: records.to_csv(p, index=False))
    columns = ["TRACK_ERROR_KM", "WIND_ABS_ERROR", "PRESSURE_ABS_ERROR", "WIND_PAIRED_ERROR", "PRESSURE_PAIRED_ERROR"]
    per_storm = records.groupby("SID")[columns].agg(["mean", "count"])
    per_storm.columns = ["_".join(c) for c in per_storm.columns]
    atomic_write(Path(run_dir) / "selection_per_storm.csv", lambda p: per_storm.to_csv(p))
    result = evaluate_records(records, config.get("bootstrap_replicates", 2000), config.get("bootstrap_seed", 1701))
    write_json(Path(run_dir) / "selection_metrics.json", result)
    summary = dict(metadata, selection_score=result["full"]["selection_composite"],
                   selection_metrics=result, selection_rows=len(frame), calibration_used_for_selection=False,
                   test_parquet_decoded=False, test_metrics_computed=False)
    write_json(Path(run_dir) / "summary.json", summary)
    return summary


def train_one(inputs, config, regime, seed, run_dir, device, resume=False,
              stop_after_epochs=None, mirror_dir=None, budget_updates=None, allow_runtime_change=False):
    if regime == "bay_tree":
        return train_tree(inputs, config, seed, run_dir, resume, mirror_dir)
    run_dir = Path(run_dir)
    seed_everything(seed)
    stages = stage_plan(regime, config, budget_updates)
    runtime = runtime_info(device)
    context = {"format_version": 1, "regime": regime, "seed": seed, "config": config,
               "package_sha256": hashlib.sha256(inputs.package.path.read_bytes()).hexdigest(),
               "source_sha256": source_hashes(), "validation_roles_sha256": inputs.role_hash,
               "feature_columns": inputs.package.features,
               "model_input_columns": inputs.package.features + [f"{c}_MISSING" for c in inputs.package.features],
               "stages": stages, "runtime": runtime}
    signature = digest(context)
    model = build_model(regime, config).to(device)
    generator = torch.Generator().manual_seed(seed)
    if resume:
        saved = load_checkpoint(run_dir / "latest.pt")
        if saved["signature"] != signature:
            old_recipe = {k: v for k, v in saved["context"].items() if k != "runtime"}
            new_recipe = {k: v for k, v in context.items() if k != "runtime"}
            if not allow_runtime_change or digest(old_recipe) != digest(new_recipe):
                raise ValueError("Resume rejected: source/config/data/roles/runtime changed. Restore the original bundle/environment; --allow-runtime-change permits only an explicitly recorded environment change.")
            saved["progress"].setdefault("runtime_changes", []).append({"from": saved["context"]["runtime"],
                "to": runtime, "at_global_step": saved["progress"]["global_step"], "bitwise_equivalence_claimed": False})
            write_json(run_dir / "run_manifest.json", dict(context, signature=signature,
                       runtime_changes=saved["progress"]["runtime_changes"]))
        state = saved["progress"]
        model.load_state_dict(saved["model_state"])
        codec = ResidualCodec.from_dict(saved["output_codec"])
        restore_rng(saved["rng"], generator, allow_device_change=allow_runtime_change)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Run already exists: {run_dir}; use --resume or a new output directory")
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {"stage": 0, "epoch": 0, "global_step": 0, "epochs_total": 0, "bad_epochs": 0,
                 "best_score": None, "stop_score": None, "best_epoch": None, "history": [], "status": "running"}
        codec = ResidualCodec.fit(inputs.frames[stages[0]["partition"]], inputs.norms[stages[0]["norm"]])
        saved = None
        write_json(run_dir / "run_manifest.json", dict(context, signature=signature))
        atomic_write(run_dir / "validation_roles.csv", lambda p: inputs.roles.to_csv(p, index=False))

    def optimizer_for(stage):
        return torch.optim.AdamW(model.parameters(), lr=stage["lr"], weight_decay=config["training_plan"]["weight_decay"])

    optimizer = optimizer_for(stages[state["stage"]])
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer_state"])

    def checkpoint():
        norm = inputs.norms[stages[state["stage"]]["norm"]]
        return {"signature": signature, "context": context, "progress": copy.deepcopy(state),
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "input_normalization": norm.statistics, "normalizer_stage": stages[state["stage"]]["norm"],
                "output_codec": codec.as_dict(), "rng": capture_rng(generator)}

    def save_latest():
        save_checkpoint(run_dir / "latest.pt", checkpoint())
        atomic_write(run_dir / "history.csv", lambda p: pd.DataFrame(state["history"]).to_csv(p, index=False))
        mirror_run(run_dir, mirror_dir)

    if saved is not None and saved["signature"] != signature and state["status"] == "completed":
        save_latest()
    if state["status"] != "completed":
        state["status"] = "running"
        save_latest()
        while True:
            stage = stages[state["stage"]]
            training = inputs.frames[stage["partition"]]
            norm = inputs.norms[stage["norm"]]
            dataset = CycloneSequences(training, norm, codec, config["context_observations"])
            loader = DataLoader(dataset, batch_size=config["training_plan"]["batch_size"], shuffle=True,
                                generator=generator, collate_fn=collate_sequences, num_workers=0)
            while state["epoch"] < stage["epochs"]:
                started = time.perf_counter()
                model.train()
                total_loss, rows, batches = 0., 0, 0
                for x, lengths, target, mask in loader:
                    x, target, mask = x.to(device), target.to(device), mask.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = masked_multitask_loss(model(x, lengths), target, mask,
                                                 wind_loss=config.get("wind_loss", "smooth_l1"))
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    loss.backward()
                    norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training_plan"]["gradient_norm_clip"])
                    if not torch.isfinite(norm_value):
                        raise ValueError("Non-finite training gradients")
                    optimizer.step()
                    total_loss += loss.item() * len(x)
                    rows += len(x)
                    batches += 1
                    state["global_step"] += 1
                    if stage.get("update_cap") and state["global_step"] >= stage["update_cap"]:
                        break
                state["epoch"] += 1
                state["epochs_total"] += 1
                score = None
                # Fixed pooled/budget controls use the final epoch, never validation early stopping.
                selected = stage["select"] and (not stage.get("fixed") or state["epoch"] == stage["epochs"])
                improved = False
                if selected:
                    forecast = predict(model, inputs.selection, norm, codec, config, device)
                    score = summarize_records(prediction_records(inputs.selection, forecast))["selection_composite"]
                    if score is None or not np.isfinite(score):
                        raise ValueError("Selection score is unavailable")
                    improved = state["best_score"] is None or score < state["best_score"]
                    if improved:
                        state["best_score"], state["best_epoch"] = score, state["epoch"]
                    if state["stop_score"] is None or score < state["stop_score"] - config["training_plan"]["early_stopping_min_score_improvement"]:
                        state["stop_score"], state["bad_epochs"] = score, 0
                    else:
                        state["bad_epochs"] += 1
                record = {"stage": stage["name"], "epoch": state["epoch"], "global_step": state["global_step"],
                          "training_rows": rows, "batches": batches, "train_loss_batch_weighted": total_loss / rows,
                          "selection_score": score, "seconds": time.perf_counter() - started}
                state["history"].append(record)
                if improved:
                    save_checkpoint(run_dir / "best.pt", checkpoint())
                save_latest()
                print(json.dumps(dict(regime=regime, seed=seed, **record)), flush=True)
                if stop_after_epochs is not None and state["epochs_total"] >= stop_after_epochs:
                    state["status"] = "paused"
                    save_latest()
                    return {"status": "paused", "global_step": state["global_step"]}
                if stage["select"] and not stage.get("fixed") and state["bad_epochs"] >= config["training_plan"]["early_stopping_patience"]:
                    break
            if state["stage"] + 1 == len(stages):
                break
            # Save the completed NIO stage before rebasing; resume at this boundary is safe.
            save_checkpoint(run_dir / "pretrain.pt", checkpoint())
            model.rebase_inputs(inputs.norms[stage["norm"]], inputs.norms[stages[state["stage"] + 1]["norm"]])
            state.update(stage=state["stage"] + 1, epoch=0, bad_epochs=0,
                         best_score=None, stop_score=None, best_epoch=None)
            optimizer = optimizer_for(stages[state["stage"]])
            save_latest()
        state["status"] = "completed"
        save_latest()
    best = load_checkpoint(run_dir / "best.pt")
    model.load_state_dict(best["model_state"])
    norm = inputs.norms[best["normalizer_stage"]]
    forecast = predict(model, inputs.selection, norm, codec, config, device)
    # Independently reconstruct from serialized preprocessing to check artifact usability.
    from .data import Normalizer
    restored_model = build_model(regime, config).to(device)
    restored_model.load_state_dict(best["model_state"])
    restored_norm = Normalizer(best["context"]["feature_columns"], best["input_normalization"])
    restored_codec = ResidualCodec.from_dict(best["output_codec"])
    reloaded = predict(restored_model, inputs.selection, restored_norm, restored_codec, config, device)
    delta = float(np.max(np.abs(forecast - reloaded)))
    if delta > 1e-6:
        raise ValueError("Checkpoint reload changed predictions")
    summary = report_forecast(run_dir, inputs.selection, forecast, config, {
        "status": "completed", "run_purpose": config["run_purpose"], "regime": regime, "seed": seed,
        "signature": signature, "config_sha256": digest(config), "global_step": state["global_step"],
        "epochs_completed": state["epochs_total"], "best_epoch": best["progress"]["best_epoch"],
        "best_checkpoint_sha256": hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest(),
        "checkpoint_reload_max_abs_difference": delta, "parameters": sum(p.numel() for p in model.parameters()),
        "runtime": runtime, "runtime_changes": state.get("runtime_changes", []), "output_codec": codec.as_dict()})
    mirror_run(run_dir, mirror_dir)
    return summary


def train_tree(inputs, config, seed, run_dir, resume=False, mirror_dir=None):
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    run_dir = Path(run_dir)
    context = {"regime": "bay_tree", "seed": seed, "config": config, "source_sha256": source_hashes(),
               "validation_roles_sha256": inputs.role_hash, "package_sha256": hashlib.sha256(inputs.package.path.read_bytes()).hexdigest(),
               "runtime": runtime_info(torch.device("cpu"))}
    signature = digest(context)
    if run_dir.exists() and any(run_dir.iterdir()):
        if not resume:
            raise FileExistsError("Existing tree run; use a new output or --resume")
        old = json.loads((run_dir / "run_manifest.json").read_text())
        if old["signature"] != signature:
            raise ValueError("Tree recipe changed")
        if (run_dir / "summary.json").exists():
            return json.loads((run_dir / "summary.json").read_text())
        # Tree fitting is short and deterministic; interrupted fits restart, never fake resume.
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_manifest.json", dict(context, signature=signature))
    atomic_write(run_dir / "validation_roles.csv", lambda p: inputs.roles.to_csv(p, index=False))
    training, norm = inputs.frames["bob_train"], inputs.norms["bob"]
    codec = ResidualCodec.fit(training, norm)
    x, y = norm.transform(training), codec.encode(training)[0]
    started = time.perf_counter()
    models = []
    with threadpool_limits(limits=2):
        for j in range(4):
            valid = np.isfinite(y[:, j])
            model = HistGradientBoostingRegressor(learning_rate=.05, max_iter=config.get("tree_iterations", 100),
                    max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=1., early_stopping=False, random_state=seed)
            model.fit(x[valid], y[valid, j])
            models.append(model)
        scaled = np.column_stack([model.predict(norm.transform(inputs.selection)) for model in models])
        artifact = {"models": models, "input_normalization": norm.statistics,
                    "features": norm.features, "output_codec": codec.as_dict(), "context": context}
        atomic_write(run_dir / "model.joblib", lambda p: joblib.dump(artifact, p))
        # Reload only the artifact just created in this process.
        restored = joblib.load(run_dir / "model.joblib")
        again = np.column_stack([m.predict(norm.transform(inputs.selection)) for m in restored["models"]])
    if not np.array_equal(scaled, again):
        raise ValueError("Tree round trip failed")
    summary = report_forecast(run_dir, inputs.selection, codec.decode(inputs.selection, scaled), config, {
        "status": "completed", "run_purpose": config["run_purpose"], "regime": "bay_tree", "seed": seed,
        "signature": signature, "config_sha256": digest(config), "runtime": context["runtime"],
        "iterations_per_target": config.get("tree_iterations", 100), "seconds": time.perf_counter() - started,
        "checkpoint_reload_max_abs_difference": 0., "resume_policy": "completed_fit_reuse_or_restart_interrupted_fit"})
    mirror_run(run_dir, mirror_dir)
    return summary


def aggregate(inputs, config, regime, root):
    root = Path(root)
    seeds = config["training_plan"]["seeds"]
    folders = [root / regime / f"seed_{s}" for s in seeds]
    if not all((p / "summary.json").exists() for p in folders):
        return {"status": "waiting_for_declared_seeds", "regime": regime, "expected_seeds": seeds}
    summaries = [json.loads((p / "summary.json").read_text()) for p in folders]
    manifests = [json.loads((p / "run_manifest.json").read_text()) for p in folders]
    for manifest in manifests:
        if (manifest["source_sha256"] != source_hashes() or manifest["validation_roles_sha256"] != inputs.role_hash
                or manifest["package_sha256"] != hashlib.sha256(inputs.package.path.read_bytes()).hexdigest()):
            raise ValueError("Aggregation rejected: source/data/role provenance mismatch")
    for s, seed in zip(summaries, seeds):
        if s["config_sha256"] != digest(config) or s["seed"] != seed or s["regime"] != regime:
            raise ValueError("Cannot combine different experiment recipes")
    records = [pd.read_csv(p / "selection_predictions.csv") for p in folders]
    ids = ["SID", "ISO_TIME", "TARGET_TIME_6H"]
    for r in records[1:]:
        pd.testing.assert_frame_equal(r[ids], records[0][ids])
    forecasts = [r[FORECASTS].to_numpy(float) for r in records]
    # Average physical displacements, not longitude degrees or differently scaled residuals.
    frame = inputs.selection
    track = np.mean([log_map(frame.LAT.to_numpy(), frame.LON.to_numpy(), p[:, 0], p[:, 1]) for p in forecasts], axis=0)
    positions = exp_map(frame.LAT.to_numpy(), frame.LON.to_numpy(), track)
    intensity = np.mean([p[:, 2:] for p in forecasts], axis=0)
    summary = report_forecast(root / regime / "ensemble", frame, np.column_stack([positions, intensity]), config, {
        "status": "completed", "run_purpose": config["run_purpose"], "regime": regime,
        "expected_seeds": seeds, "complete_declared_seed_set": True,
        "family_selection_score_mean_of_seeds": float(np.mean([s["selection_score"] for s in summaries])),
        "individual_seed_scores": {str(s["seed"]): s["selection_score"] for s in summaries},
        "ensemble_rule": "mean_current_centred_spherical_displacements_and_native_intensity",
        "selection_uncertainty_note": "Bootstrap is descriptive on selection storms, not independent test evidence"})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=["bay_gru", "transfer_gru"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", default=str(RUNS_DIR / "experiments"))
    parser.add_argument("--mirror-dir")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-runtime-change", action="store_true", help="Record changed hardware/software on resume; exact equivalence is no longer claimed")
    parser.add_argument("--stop-after-epochs", type=int)
    parser.add_argument("--budget-updates", type=int)
    parser.add_argument("--acceptance", action="store_true", help="Shortened schedule, labelled non-research")
    args = parser.parse_args()
    if args.threads < 1 or (args.stop_after_epochs is not None and args.stop_after_epochs < 1):
        parser.error("threads and stop-after-epochs must be positive")
    config = load_config(args.config, args.acceptance)
    seeds = args.seeds or config["training_plan"]["seeds"]
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset(config["training_plan"]["seeds"]):
        parser.error("Seeds must be a distinct subset of the declared configuration seeds")
    torch.set_num_threads(args.threads)
    seed_everything(seeds[0])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu")
    inputs = Inputs(args.package)
    for regime in args.regimes:
        for seed in seeds:
            folder = Path(args.output) / regime / f"seed_{seed}"
            mirror = Path(args.mirror_dir) / regime / f"seed_{seed}" if args.mirror_dir else None
            result = train_one(inputs, config, regime, seed, folder, device,
                               resume=args.resume and (folder / "run_manifest.json").exists(),
                               stop_after_epochs=args.stop_after_epochs, mirror_dir=mirror, budget_updates=args.budget_updates,
                               allow_runtime_change=args.allow_runtime_change)
            print(json.dumps({"regime": regime, "seed": seed, "status": result["status"]}), flush=True)
        result = aggregate(inputs, config, regime, args.output)
        write_json(Path(args.output) / regime / "aggregation_status.json", {k: v for k, v in result.items() if k != "selection_metrics"})
        if args.mirror_dir:
            mirror_run(Path(args.output) / regime, Path(args.mirror_dir) / regime)


if __name__ == "__main__":
    main()
