# Wind audit and bounded follow-up

This round follows the completed v0.3 research comparison. Its selection set has
already been inspected. New results are a disclosed follow-up, not independent
confirmation. Preserve the original results ZIP and the v0.3.0 tag.

## What the audit checks

Run from the repository root, substituting your results ZIP path:

```bash
python -m modelling.wind_audit --results "original_results.zip" --output runs/wind_audit_v1
```

The reader verifies every exported file's size and SHA-256 without loading model
binaries. It binds all saved predictions to the immutable selection observations,
checks missing targets and roles, recomputes paired scores, checks the declared
three-seed ranking, and writes per-storm and wind-outcome diagnostics.
It refuses an existing output directory.

`audit.json` contains full and geographic-Bay comparisons, wind missingness and
change counts, input-time wind/SST bands, per-seed codec scales and mixed-task
training-loss histories. `wind_per_storm.csv` contains model and persistence MAE
with explicit observed and paired denominators. Outcome-change bands use future
wind only to diagnose errors; they are never forecast inputs or selection gates.
Missing-current and missing-target groups overlap other diagnostic groups and
must not be summed as exclusive categories. Below-34/64-kt bands are diagnostics,
not harmonized cyclone categories.

Original histories lack per-epoch wind loss. They cannot establish that wind
overfitting caused the observed results. Rounded/unchanged six-hour best-track
wind makes persistence competitive; loss/MAE mismatch is a testable hypothesis.

## The fixed candidate set

1. Original Bay MLP, selected by the original mean-seed composite rule.
2. Original Bay MLP with observed current wind persisted. If current wind is
   unavailable, retain its learned wind prediction. Track and pressure are
   unchanged. This causal rule has no tuned threshold. Persistence-equivalent
   wind MAE is not evidence of learned wind improvement.
3. One new Bay MLP experiment, changing only the wind task from normalized
   SmoothL1 to normalized absolute residual error. Track and pressure retain
   SmoothL1. Architecture, normalizers, codec, masking, optimizer, schedules,
   seeds and checkpoint rule are otherwise unchanged.

Absolute error on a fixed scaled wind residual has the same per-task minimizer
as native-unit wind MAE. Its gradient weighting relative to other tasks changes;
this is part of the declared intervention, not proof that it will improve skill.
Do not add loss weights, shrinkage grids, architectures or intensity thresholds
after seeing this comparison.

## Run and resume

The audit writes `followup_protocol.json` before any new training. It binds the
original export, data, audit contents, configuration, seeds and current source.
Changing those requires a fresh audit/output, not resuming an old experiment.

```bash
# Acceptance only: seed 17, two epochs; no candidate decision.
python -m modelling.wind_followup --audit runs/wind_audit_v1 --output runs/wind_acceptance --device cpu

# Research: exactly seeds 17/29/43, at most 100 epochs each, existing early stopping.
python -m modelling.wind_followup --audit runs/wind_audit_v1 --output runs/wind_research --device auto --full

# Resume the same source/config/data/runtime; optional persistent mirror.
python -m modelling.wind_followup --audit runs/wind_audit_v1 --output runs/wind_research --device auto --full --resume --mirror-dir /path/to/persistent/wind_research
```

Use the new `notebooks/bob_cyclone_wind_followup_colab.ipynb` with the
`bob_cyclone_wind_followup_code.zip` bundle. The old v0.3.0 notebook/tag does not
contain these commands. The follow-up notebook defaults to acceptance, verifies
both upload bundles and optionally mirrors all evidence to Drive. Full research
uses a separate output directory. A changed runtime requires the explicit
`--allow-runtime-change` flag and does not claim bitwise equivalence.

## Upload recovery

There are two different uploads: first `bob_cyclone_wind_followup_code.zip`, then
the completed `deterministic_v0_3_results (1).zip`. The results ZIP must contain
`RUN_FILE_MANIFEST.json` at its root and the original `research/` comparison.
The updated notebook rejects a cached code/dataset/acceptance ZIP and prompts
again. An invalid new upload never replaces the existing cached file.

For an already open notebook with the older upload cell, run this small recovery
cell and choose the original results ZIP, then rerun the audit cell onward:

```python
from google.colab import files
import io, zipfile

uploaded = files.upload()  # Choose deterministic_v0_3_results (1).zip
if len(uploaded) != 1:
    raise ValueError("Choose exactly one original results ZIP")
name, payload = next(iter(uploaded.items()))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    required = {"RUN_FILE_MANIFEST.json", "research/bay_mlp/seed_17/summary.json"}
    if not required.issubset(archive.namelist()):
        raise ValueError("Choose the original full research results ZIP")
RESULTS_PATH.write_bytes(payload)
print("Original results selected:", name)
```

This recovery cell changes only the cached results upload. It does not require a
source change or retraining. Keep an already started experiment on its original
source bundle; a fresh run using updated source needs a fresh audit/run ID.

## Decision and claim boundaries

Select the smallest mean individual-seed composite among the three declared
candidates, using the original eligible track/wind/pressure denominators. Exact
ties retain the earlier candidate in the list. Do not select on ensemble score
or one attractive target. Report ensemble, seeds and all-observed-label errors
alongside paired scores; keep the original physical reference visible.

The full runner writes `candidate_decision.json` with checkpoint hashes, data,
source, original export and the forecast rule. Acceptance cannot create a
research selection. This is a deterministic decision receipt, not the complete
probabilistic/final-evaluation freeze record. Future probabilistic work must
freeze uncertainty architecture, transformations, validation roles and density
conventions before fitting calibration parameters on reserved 2015–2017 storms.
The 2018–2025 test decoder remains denied. Neither command calibrates or evaluates
the final test; selection-bootstrap intervals remain descriptive.
