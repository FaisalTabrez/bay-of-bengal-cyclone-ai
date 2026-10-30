# Colab guide · code release 0.3.0

For the calibrated model's final evaluation, use [the final-evaluation workflow](final-evaluation.md) and `notebooks/bob_cyclone_final_evaluation_colab.ipynb` with its own source bundle and the calibrated uncertainty export. The default pass prepares a plan while keeping the real test sealed.

For the selected model's uncertainty stage, use [the uncertainty workflow](uncertainty.md) and `notebooks/bob_cyclone_uncertainty_colab.ipynb` with its own code bundle and completed wind research export.

For the next wind diagnostic/experiment round, use [the wind follow-up guide](wind-followup.md) and `notebooks/bob_cyclone_wind_followup_colab.ipynb` with its new source bundle. The instructions below reproduce the original v0.3.0 comparison.

Open the notebook from the repository's **Open in Colab** badge. The default `USE_GITHUB_SOURCE=True` retrieves the public repository at tag `v0.3.0`, including the immutable data packages. You do not need Fabric credentials or a separate dataset download.

1. Choose a CPU or GPU runtime. Local release verification uses CPU; the first Colab session must verify its actual device and Drive integration.
2. Leave `RUN_FULL_EXPERIMENTS=False` for the first pass and run the notebook from the beginning. It checks source/data integrity, runs the tests, exercises Bay and transfer pause/resume, and exports execution evidence.
3. Mount Drive when prompted. Work is local under `/content/bob_cyclone_ai`; snapshots are mirrored to `MyDrive/BayOfBengalCycloneAI/runs/deterministic_v0_3`. If persistence is disabled, download the result archive before ending the session.
4. Confirm that every acceptance run completes and checkpoint reload differences are zero. The table and plot use shortened schedules and are not research results.
5. Set `RUN_FULL_EXPERIMENTS=True` when ready. The full comparison has separate output paths and declared seeds 17/29/43. Calibration and test labels do not select checkpoints.
6. Retain the source tag, exported results ZIP and notebook together.

## Offline bundle

Build with `python -m modelling.artifacts bundle`. Open `notebooks/bob_cyclone_modelling_v2_colab.ipynb` in Colab using **File → Upload notebook**, set `USE_GITHUB_SOURCE=False`, and upload `dist/bob_cyclone_colab_bundle_v0_3.zip` when prompted. The bundle uses the same organized folder layout and contains the data archives without unpacking their Parquet files.

## Resume

Rerun setup with the same source tag and run ID. A persistent mirror is restored when the local run directory is absent. An intentional runtime/hardware change requires `ALLOW_RUNTIME_CHANGE=True`; it is recorded and does not claim bitwise equivalence. Source/config/data changes require a new run. Keep the kernel/source consistent; restart after changing already imported libraries.

See [the modelling guide](modelling.md) for CLI equivalents and exact-update controls, [the model design](model-design.md) for methodology, and [the data handoff](data-handoff.md) for provenance and missingness policies. Full scientific skill, probabilistic calibration and final-test evaluation remain pending.
