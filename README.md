# Bay of Bengal Cyclone AI

[![CPU verification](https://github.com/FaisalTabrez/bay-of-bengal-cyclone-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/FaisalTabrez/bay-of-bengal-cyclone-ai/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FaisalTabrez/bay-of-bengal-cyclone-ai/blob/v0.3.0/notebooks/bob_cyclone_modelling_v2_colab.ipynb)

Research code for **six-hour-ahead tropical cyclone track, wind and pressure prediction**, using versioned IBTrACS, OISST and ERA5-derived training packages.

The supplied Colab exports contain completed deterministic comparisons and the bounded wind-loss follow-up, selecting the three-seed Bay MLP with absolute wind loss. Full frozen-mean Gaussian uncertainty training and one-time reserved-role calibration are verified. The separately gated final evaluator is implemented; independent final-test results remain pending plan review. The current experiment is retrospective; operational forecast skill has not been established.

## Start here

The calibrated model is verified. For the next phase, use the [reviewed final-evaluation guide](docs/final-evaluation.md) and `notebooks/bob_cyclone_final_evaluation_colab.ipynb`. Its first pass prepares the locked protocol and leaves the real test sealed. Final metrics require the exact reviewed plan digest and a persistent one-time ledger claim.

For uncertainty estimation, use the [uncertainty guide](docs/uncertainty.md) and `notebooks/bob_cyclone_uncertainty_colab.ipynb`. It requires the uncertainty code bundle and the completed wind follow-up research results. Acceptance and full training are separate, and calibration is disabled by default.

For the completed v0.3 research results, use the [wind audit and follow-up guide](docs/wind-followup.md) and the new `notebooks/bob_cyclone_wind_followup_colab.ipynb`. It requires a newly built follow-up code bundle; the pinned v0.3.0 notebook below reproduces the original experiment.

Open the Colab badge above and run the notebook with `RUN_FULL_EXPERIMENTS=False` first. It retrieves the pinned repository tag, verifies the data packages and exercises training, checkpoint reload and resume. The offline upload-bundle workflow is also supported. See the [Colab guide](docs/colab.md).

For local development, use Python 3.12+ from the repository root:

```bash
git clone https://github.com/FaisalTabrez/bay-of-bengal-cyclone-ai.git
cd bay-of-bengal-cyclone-ai
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -e ".[notebook]"
python -m modelling.run_design_checks
python -m unittest discover -s tests -v
python -m modelling.train --acceptance --device cpu --output runs/acceptance
```

On Windows PowerShell activate with `.\.venv\Scripts\Activate.ps1`; on Linux/macOS use `source .venv/bin/activate`. The [modelling guide](docs/modelling.md) covers full schedules, seed ensembles, exact-update controls and interrupted runs. Development uses an editable checkout so the versioned data, configuration and notebook remain together.

## Approach

- A compact residual GRU uses up to four causal observations, spanning 18 hours, to correct constant-velocity track and persistence intensity forecasts.
- Explicit input masks and training-only imputation preserve missingness. Masked losses retain observed targets even when current intensity is missing.
- Bay-only and NIO-to-Bay training are compared with snapshot MLP, tree and pooled-training controls. NIO includes the Bay training rows; it is not an independent additional dataset.
- Checkpoint selection uses 2011–2014 validation storms. The 2015–2017 role is reserved for later calibration. The 2018–2025 test partition remains sealed by the modelling loader.
- Evaluation uses paired baseline denominators, all-observed-label metrics, three declared seeds and whole-cyclone bootstrap intervals. Acceptance scores are execution diagnostics, not scientific results.

See the [model design](docs/model-design.md) for the full protocol and claim boundaries.

## Versioned data

| Package partition | Samples | Use |
|---|---:|---|
| `nio_pretrain_train` | 5,557 | Broader NIO pretraining |
| `bob_train` | 4,560 | Bay learning/fine-tuning |
| `bob_validation` | 854 | 493 selection rows; 361 calibration rows |
| `bob_test` | 1,018 | Frozen final evaluation, not opened by this runner |

Archives are preserved byte-for-byte in [`data/packages/`](data/packages). Their hashes and provider citations are documented in the [data README](data/README.md) and [original handoff](docs/data-handoff.md). The v1 archive is retained for reproducibility. Source acquisition/Fabric construction notebooks are not supplied by this repository; the legacy notebook is an acceptance reference.

## Repository layout

```text
modelling/             Python package, model/metrics/runner and configuration
tests/                 Scientific contract, resume and artifact regression tests
notebooks/             Output-free Colab entry point and legacy acceptance reference
docs/                  Model design, data handoff, Colab and modelling guides
data/packages/         Immutable v1/v2 data and v1 acceptance archives
requirements/          Checked local versions and Colab/notebook requirements
scripts/               Publication hygiene checks
.github/workflows/     Linux CPU and notebook verification
```

`runs/`, `reports/`, `dist/`, `output/` and `local_artifacts/` are generated/local storage and are ignored by Git. No local training checkpoints or executed-notebook outputs are committed.

## Verification and reproducibility

```bash
python scripts/check_repository.py --tracked
python -m modelling.verify_release
python -m modelling.artifacts bundle
```

Release verification runs the tests and all notebook cells from a clean transfer bundle, checks checkpoint round-trips and source hashes, and writes evidence under `reports/`. Build output is `dist/bob_cyclone_colab_bundle_v0_3.zip` with a SHA-256 sidecar. CI uploads its verification reports and bundle as workflow artifacts.

Code release **0.3.0** organizes the repository; the underlying dataset remains **2.0.0**, with unchanged archive checksums. Older local checkpoints retain their original provenance and should not be silently resumed with reorganized source.

GPU execution and Google Drive/upload integration require acceptance in a live Colab session. Probabilistic heads and a final-test command are intentionally deferred.

## Attribution

Data are derived from NOAA IBTrACS v04r01, NOAA OISST v2.1 and ECMWF ERA5 accessed through the public Earthmover Icechunk archive. Provider attribution, source-specific terms and modification notes are in [NOTICE.md](NOTICE.md) and [data/README.md](data/README.md). No provider endorses the models or results. A repository-wide code licence has not been selected; upstream data terms are separate.
