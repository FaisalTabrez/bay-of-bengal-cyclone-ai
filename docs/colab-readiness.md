# Colab readiness review

## Implementation update Â· 28 October 2026

The deterministic runner, simpler controls, paired metrics, storm bootstrap, checkpoint/resume, persistent mirroring, notebook and transfer/result packaging listed below are now implemented. Start with `docs/colab.md` and `docs/modelling.md`. Short local CPU runs and a clean extracted-bundle notebook execution are the release acceptance scope; detailed outcomes are recorded in `reports/release_verification.json` after verification completes.

**Current remaining gate:** a live Colab environment/device acceptance run. Full research experiments, probabilistic calibration and final-test evaluation remain pending. The following sections preserve the original pre-implementation audit and its original findings; statements about missing software describe that earlier state.

## Original audit, before implementation

Reviewed 28 October 2026 against the local source, package archives, saved design receipts and official runtime/package documentation.

**Verdict: ready for a Colab acceptance/smoke session; not yet ready for the planned training experiments.** The data needs no new acquisition for the first retrospective experiment. The remaining blockers are experiment software and runtime verification.

## What is ready

- v1 and v2 ZIP SHA-256 values match the immutable handoff.
- All 12 v2 internal file hashes and sizes pass. Test bytes were hashed only; test Parquet was not decoded.
- Source hashes in `reports/design_checks.json` match the current Python sources.
- Supplied training means/stds, trainingâ€“validation storm separation, causal windows and original validation baselines have verification receipts.
- The residual GRU, masked losses, spherical decoding and normalization transfer are implemented.
- Selection and calibration roles are recorded: 493 rows/25 storms for selection, 361 rows/25 storms for calibration.
- All ten scientific contract tests pass locally on CPU, including a fresh run for this review.

## Needed before an actual training session

| Priority | Missing item | Acceptance condition |
|---|---|---|
| Required | Experiment runner | Implement N1 Bay-only GRU and N2 NIOâ†’Bay GRU first, with declared seeds, complete epoch loops, shuffled training windows, selection-only checkpoint choice, early stopping and physical-unit predictions. Honour the configuration rather than hard-coding settings. |
| Required | Colab modelling notebook | Add a new v2 modelling notebook with upload/setup, environment inspection, checks, a short training acceptance run, explicit experiment launch and export cells. The existing v1 notebook is an acceptance notebook for a different model. |
| Required | Checkpoint and resume | Save latest and best checkpoints, optimizer state, epoch/global step, early-stop state, random-number states, feature order, input normalizer, output codec, config, validation role definition and package/code hashes. Reload and resume successfully on a small training-only run. |
| Required | Evaluation implementation | Compute the declared selection score with matched intensity denominators, all-observed-label metrics, full/geographic-Bay metrics and per-storm records. Aggregate seeds without selecting only the best seed. Keep calibration storms out of early stopping and model choice. |
| Required | Output persistence and transfer bundle | Supply one code archive plus a new notebook. Persist checkpoints/metrics outside the ephemeral Colab VM and provide a downloadable results archive. Verify the transferred source hashes. |
| Runtime gate | Linux/CUDA acceptance | Check the actual Python/dependency versions, run contract checks, perform a forward/backward update on the chosen device, and test checkpoint reload before full experiments. CPU fallback should work. |
| Before comparative claims | Simpler controls and uncertainty | Implement the tree/snapshot/pooled controls and storm-bootstrap uncertainty described in `docs/model-design.md`; add a compute-budget control if attributing a gain specifically to transfer. |

There is no full training or evaluation CLI today. `python -m modelling.run_design_checks` performs exactly one optimizer step on 128 NIO rows, saves no weights and discards the model. Re-running this command in Colab does not train the proposed research model.

## Dependency findings

The checked file pins NumPy 2.5.3, pandas 3.0.6, PyArrow 25.0.1 and PyTorch 2.14.0. Their public release metadata contains Linux x86-64 CPython 3.12 wheels; these versions are not merely local version strings. NumPy 2.5.3 requires Python >=3.12. Sources: [NumPy release metadata](https://pypi.org/pypi/numpy/2.5.3/json), [pandas release metadata](https://pypi.org/pypi/pandas/3.0.6/json), [PyArrow release metadata](https://pypi.org/pypi/pyarrow/25.0.1/json), [PyTorch release metadata](https://pypi.org/pypi/torch/2.14.0/json).

Wheel availability is not a clean-install, transitive-dependency or CUDA execution test. The local evidence uses Python 3.12.10 and CPU PyTorch. Inspect the Colab runtime first, then choose and verify an environment; do not assume its preinstalled versions match this lock file. Google's documented 2026.07 runtime, for example, lists Python 3.12.13, NumPy 2.0.2 and PyTorch 2.11.0. [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html).

The future tree baseline also needs scikit-learn, which is not in `requirements/checked.txt`. That is not a dependency of the current GRU checks; add and validate it with the tree implementation. Plotting dependencies can be added with the reporting implementation. Record the resolved environment for every experiment.

Colab virtual machines have limited lifetimes and can be deleted after inactivity. Keep working copies locally on the VM for training, but save checkpoint snapshots and final artifacts to persistent storage or download them. No paid Colab plan is established as necessary by the current model size, and GPU availability is not guaranteed. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Transfer contents

For an acceptance-only session, the current minimum input set is:

1. `bob_cyclone_training_v2.zip`.
2. `bob_cyclone_v1_acceptance_results.zip`, used only for its baseline JSON.
3. The complete `modelling/` source folder, preserving package structure and `configs/`.

Also preserve `docs/model-design.md`, the original handoff and this review. The old v1 data ZIP, old smoke weights and old notebook are reproducibility references rather than inputs to the new GRU. Avoid replacing or loading the old smoke checkpoint into this architecture.

For training, additionally supply the new notebook/runner and a declared persistent run-output location. Do not upload unrelated caches or assume that saving a notebook saves its VM files.

## Items that can wait

- Gaussian/other probabilistic heads and calibration: after deterministic comparisons are established.
- Final-test evaluation command: after model and preprocessing freeze; keep the existing test-loader denial in place now.
- New SST coverage, spatial ERA5 patches, rainfall or climate projections: separate data releases, not prerequisites for the current experiment.
- Source-construction notebooks and an as-of availability audit: needed to strengthen reproducibility and before operational-forecast claims, but not a blocker to explicitly labelled retrospective experimentation.

Recommended order: implement the deterministic runner and checkpoint round-trip locally; package a new Colab notebook; perform a fresh Colab environment/device acceptance run; then launch the declared experiments. Do not spend a Colab training session on the current smoke-check command alone.
