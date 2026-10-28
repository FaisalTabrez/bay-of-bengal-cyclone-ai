# Deterministic cyclone modelling runner Â· v0.3

Read the root `docs/model-design.md` for scientific definitions and `docs/colab.md` for the notebook workflow. The original data ZIPs are read in place and never modified. This release implements deterministic training; full research comparisons, probabilistic modelling/calibration and final-test evaluation remain pending.

## Setup and acceptance

Use Python 3.12+. `requirements/checked.txt` records the local checked versions. `requirements/colab.txt` accepts compatible preinstalled packages, including Colab's CUDA-enabled torch, and installs missing requirements. Actual versions are recorded with every run; compatible version ranges are not a claim that every combination has been tested.

```powershell
python -m venv .venv-model
.\.venv-model\Scripts\python.exe -m pip install -r requirements/checked.txt
```

The commands below use `python`; substitute the virtual environment's interpreter if created. A fresh isolated installation and live CUDA execution have not been verified locally.

```powershell
python -m modelling.preflight
python -m modelling.run_design_checks
python -m unittest tests.test_contract tests.test_experiments tests.test_artifacts -v
python -m modelling.train --acceptance --regimes bay_gru transfer_gru bay_mlp bay_tree pooled_gru --device cpu --output runs/acceptance
```

Acceptance is labelled `acceptance_not_research`: seed 17, one NIO epoch, two Bay epochs, three tree iterations and 40 bootstrap draws. The full configuration defaults to seeds 17/29/43, 30 NIO epochs, at most 100 Bay epochs, 100 tree iterations and 2,000 bootstrap draws. Do not present acceptance scores as model skill or use them to retune the full comparison.

## Full experiments

```powershell
python -m modelling.train --regimes bay_gru transfer_gru bay_mlp bay_tree pooled_gru --output runs/research --device auto
```

`bay_gru` is Bay-only, `transfer_gru` pretrains on NIO then rebases inputs and fine-tunes on Bay, `pooled_gru` uses a fixed NIO schedule with no fine-tuning, `bay_mlp` is the snapshot neural control, and `bay_tree` fits four residual histogram-boosting regressors. Training uses complete cyclone-safe windows, shuffling within the training partition. Pretraining receives no validation score. Only the 493 selection rows/25 storms are passed to learned-model selection/evaluation. Calibration rows/labels are not used for fitting, stopping or comparison.

Use `--seeds 17` to schedule one declared seed at a time. Ensemble/family summaries wait until all declared seeds are complete; the CLI never picks only the best seed. The family selection statistic is mean individual-seed composite score. The ensemble averages physical tangent-plane track displacement and native-unit intensities, with paired metrics and observed-label counts.

The `bay_budget` control accepts an exact update count:

```powershell
python -m modelling.train --regimes bay_budget --seeds 17 --budget-updates 116 --output runs/budget_example
```

Here 116 is an illustrative acceptance count, **not the full research budget**. For research, read each transfer seed's `global_step` in its completed summary and match that count for the corresponding Bay control. Its checkpoint is terminal after exactly that many updates, with no validation early stopping. Record the source transfer signature when interpreting the comparison. No claim of a causal transfer benefit follows from N1/N2 alone.

## Interruption, resume and persistence

```powershell
python -m modelling.train --acceptance --regimes transfer_gru --output runs/resume_check --stop-after-epochs 1
python -m modelling.train --acceptance --regimes transfer_gru --output runs/resume_check --resume
```

The first command pauses at the end of NIO pretraining; the second safely performs the Bay transition once. Resume is at the last saved epoch boundary. An interruption halfway through an epoch replays that epoch from the previous checkpoint. A tree fit reuses a completed fit or restarts an interrupted fit from scratch; it does not claim to restore internal boosting progress.

`latest.pt` contains model/optimizer states, stage, epoch/update counters, patience state, history, Python/NumPy/torch/CUDA random states, loader shuffle state, feature order, input statistics, output codec, source/config/data/role hashes and runtime. `best.pt` is the selection checkpoint; `pretrain.pt` preserves the NIO state before rebasing. Torch checkpoints use restricted `weights_only=True` loading. Tree `.joblib` files are Python serialization artifacts; load only artifacts produced by your trusted run.

The runner refuses accidental overwrite. `--resume` checks the exact recipe and environment before continuing; the same-environment CPU test demonstrates identical interrupted/uninterrupted weights. For an intentionally changed Colab runtime/hardware, add `--allow-runtime-change`. Only environment differences are allowed; code/config/data/role changes remain errors. The transition is recorded with `bitwise_equivalence_claimed=false`; resumed scores are not promised to match a continuous run on the original device.

Add `--mirror-dir <persistent-directory>` to copy checkpoint snapshots and run artifacts after each epoch and final report. Train from a local VM copy; mirror to Drive. On a new VM restore the mirror to the same local run layout, then resume. Do not run two writers against the same run directory.

## Outputs

Each `<output>/<regime>/seed_<seed>/` contains a run manifest, validation-role assignment, checkpoints or tree model, epoch history, selection predictions, per-storm CSV, metrics JSON and summary. Metrics include full/geographic-Bay track errors, observed-label and paired intensity MAEs, baseline skill, storm macro means, input-condition subgroups, and descriptive whole-storm bootstrap intervals. Selection-stage intervals are not independent final-test evidence.

`<regime>/ensemble/` appears only when all declared seeds are complete. It records every seed score, family mean and ensemble metrics. All summaries label acceptance/research purpose and explicitly record calibration/test exclusion.

```powershell
python -m modelling.artifacts export --source runs/research --output research_results.zip
python -m modelling.artifacts bundle
```

Exports include checksum manifests. The bundle contains source, notebook, documentation, v2 data ZIP and original acceptance ZIP; it excludes local run/checkpoint caches. Verification:

```powershell
python -m modelling.artifacts verify --root <extracted-bundle-directory>
```

The old v1 smoke checkpoint has a different input/target contract and is never loaded by this runner. The design-stage loader still refuses `bob_test`; there is no final-test command. `run_design_checks` verifies historical full-validation baselines for handoff acceptance; training uses only the selection role. No probabilities or calibration results are produced by this release.

## Notebook/build dependencies

The supplied notebook is ready to open; Colab already provides notebook execution. To rebuild/execute it locally, install the optional tools in `requirements/notebook.txt`, then use `python -m modelling.make_notebook`. Release validation executes a copy from a clean extracted bundle with `nbclient`; the upload/Drive/CUDA branches still require a live Colab session.

Implementation references: [PyTorch checkpoint recipes](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [scikit-learn histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
