# Reviewed one-time final-test evaluation

The calibrated model and inference implementation are fixed. This phase prepares
the command and scoring plan before any real `bob_test` Parquet is decoded.
The default notebook performs software checks and selection-input reproduction
only. It produces a reviewable plan, not final-test metrics.

## Exact inputs

- Source: `bob_cyclone_final_evaluation_code.zip`.
- Calibrated results: `uncertainty_v1_results (3).zip`, SHA-256
  `666244e9474f6c4398b54ca35ebc233cf2f4a7fb5d28b466c18d0874ae7f939e`.
- Frozen model canonical digest:
  `6f4535600f4eeffd8ceeb361456fe96166752d8dae54486f413340f584542d84`.
- Calibration **file** SHA-256:
  `19a3bd974ce3500f7b1f5ccc7626e8ff40c506ed41745b6ab45ac5a85b4dad30`.
- Dataset remains the byte-identical v2 archive. Its handoff metadata declares
  1,018 test observations from 65 cyclones, with input years 2018–2025.

The calibrated-export checksum and calibration-file checksum are different
artifacts. They are checked separately. Neither the selected means nor the
learned spreads nor their three calibration factors are estimated again.

## Preserving the original source freeze

The earlier model freezes include a complete inference source map. Adding this
evaluator changes the current package's source map, so reusing that package as
the old inference implementation would violate its freeze.

`data/frozen_runtime/uncertainty_v1_source.zip` contains the exact 34 original
source files named in that freeze. The evaluator verifies their hashes, copies
them into an isolated runtime and runs `scripts/final_inference_worker.py` in a
separate process. That worker imports only the original model code, verifies its
unchanged source/model/calibration receipts, and receives target-free inputs.
The existing training and calibration loaders continue to reject `bob_test`.

The new evaluator and scoring source are separately bound by the reviewed plan.
Inference uses CPU, two threads and seed 71. Preparation records the actual
software/runtime and checks reproduction on 493 already-used selection inputs.
The original T4 forecasts must match within the declared cross-runtime tolerances.

## Colab procedure

Rebuild the supplied notebook and source bundle locally, when needed:

```bash
python -m modelling.make_final_notebook
python -m modelling.artifacts bundle --purpose reviewed_final_evaluation --output dist/bob_cyclone_final_evaluation_code.zip
```

1. Upload `notebooks/bob_cyclone_final_evaluation_colab.ipynb` to Colab.
2. Upload the final-evaluation source ZIP at the first prompt.
3. Upload the calibrated uncertainty results ZIP at the second prompt.
4. Keep `RUN_FINAL_TEST=False` and the review digest empty. Run the notebook.
5. Inspect the software-check log, displayed `REVIEW.md`, protocol JSON and
   `reviewed_plan.json`. This first pass leaves the real test sealed.
6. After reviewing the complete plan, copy its printed canonical SHA-256 into
   `REVIEWED_PLAN_SHA256` and set `RUN_FINAL_TEST=True`. Rerun the parameters cell,
   execution cell, result display and export cells. Use the same prepared plan.

The digest is specific to the prepared plan/runtime. Do not copy a local Windows
digest into a newly prepared Colab plan. The notebook never fills the review
field automatically. A changed scoring source, configuration, model artifact,
calibration receipt or runtime blocks execution.

Drive persistence is required for an actual Colab test attempt. Its global ledger
lives at `MyDrive/BayOfBengalCycloneAI/final_test_ledger`, independently of the
run ID. Keep it and the mirrored run. Local CLI runs use an explicitly supplied
persistent filesystem ledger. These are research-use controls within that ledger,
not protection against a user deliberately deleting or replacing it.

## Locked scoring protocol

The point forecast is the selected three-seed absolute-wind-loss MLP ensemble.
Compare it with the existing persistence and constant-velocity/tendency references.
Report all-observed-label intensity MAE as well as paired baseline comparisons
with explicit counts. Missing targets remain missing. When a reference has zero
error, normalized error/skill is undefined and is reported as `null`.

Report three fixed probabilistic forecasts with the same frozen mean:

1. A diagonal Gaussian using the frozen in-sample training-RMS initialization.
   This is a reference; it is not an out-of-fold uncertainty estimate.
2. The frozen learned Gaussian before spread calibration.
3. The frozen learned Gaussian with the three existing calibration factors.

For full test and fixed current-position geographic Bay (5–25°N, 78–100°E),
report native-unit Gaussian NLL, marginal CRPS, spatial energy score, 50/80/90%
coverage and widths, track ellipse area, and unclipped tail-mass diagnostics.
Track density/area remain current-centered tangent-plane quantities, not a
latitude/longitude density. There is no wind–pressure noise dependence.

Bootstrap intervals use 2,000 whole-cyclone replicates, preserving all six-hour
observations within sampled storms. Paired score differences compare calibrated
minus each Gaussian reference. Negative differences indicate lower proper scores.
The energy score uses 512 fixed common-random-number draws; geographic subgroups
reuse those per-observation scores. Report the predefined point land-proximity
and current-intensity diagnostics and storm-macro summaries with counts.

The final test must not select models, thresholds, scale factors, epochs, features,
reporting cohorts or additional experiments. Retain the known central-coverage
mismatch and the retrospective best-track/reanalysis availability limitations.

## One-time execution and technical recovery

```bash
python -m modelling.final_evaluation prepare --results "uncertainty_v1_results (3).zip" --output runs/final_v1/prepared

# Only after reviewing REVIEW.md and the exact plan digest:
python -m modelling.final_evaluation evaluate --prepared runs/final_v1/prepared --output runs/final_v1/execution --ledger /persistent/final_test_ledger --reviewed-plan-sha256 <printed_digest>

# A technical retry of the identical attempt, or read-only access when completed:
python -m modelling.final_evaluation evaluate --prepared runs/final_v1/prepared --output runs/final_v1/execution --ledger /persistent/final_test_ledger --reviewed-plan-sha256 <same_digest> --resume

# Inspect the checksum-verified completed metrics without invoking inference:
python -m modelling.final_evaluation inspect --output runs/final_v1/execution
```

The dataset-level ledger is exclusively claimed **before** decoding. Changing the
run ID, model or result folder does not reset this study's dataset claim.
The dedicated decoder verifies the immutable bytes, exact schema, expected counts,
chronology, six-hour horizons, finite observations and cyclone isolation. It does
not silently discard rows to repair a contract failure.

After the first decode, the verified frame is cached and its checksum recorded in
the persistent ledger. A technical resume can compute the identical frozen
analysis from that cache; it cannot re-decode the original ZIP. If access started
but no verified cache survived, preserve the attempt for review rather than
starting another run. A completed resume verifies its result receipt and returns
saved metrics without inference or scoring. Keep all outputs and ledger receipts.

Preparation exports are named `bob_cyclone_final_preflight_results.zip` and contain
no real test labels. Actual-attempt exports are `bob_cyclone_final_results.zip`;
they include the test cache for technical recovery of that same frozen analysis.
Generated results remain outside Git. No final-test results have been generated
as part of implementing or validating this command.
