# Uncertainty estimation and the reserved calibration stage

This workflow consumes `wind_mae_followup_v1_results (1).zip`, the completed
three-seed follow-up that selected `bay_mlp_wind_absolute_loss`. It does not
replace or retrain those point models. Original archives and earlier source tags
remain reproducibility artifacts.

## Fixed scientific contract

The deterministic forecast remains the mean of seeds 17/29/43 in the common
current-centered tangent plane and native intensity units. A separate 78→32→5
MLP predicts four positive standard deviations and one track correlation.
The track is a correlated bivariate Gaussian; wind and pressure are independent
univariate Gaussians. The resulting distribution is one Gaussian centered on the
frozen ensemble, **not** the mixture of three member Gaussians. It does not model
wind–pressure dependence or cross-task dependence.

Track density is defined in east/north kilometres at the current cyclone centre.
It is not a Gaussian density in latitude/longitude; no coordinate-density or
spherical-area likelihood is reported. Ellipse areas are tangent-plane km².
Wind uses knots and pressure hPa. The spherical exponential map remains the point
forecast decoder.

Spread uses softplus plus declared native floors of 1 km per track axis, 0.5 kt
and 0.5 hPa. Correlation is bounded by `0.95*tanh(raw)`. Initial spread is the RMS
of **in-sample training errors** of the fixed ensemble. This is an initialization,
not an out-of-fold estimate or a calibrated uncertainty claim.

Only the spread network is optimized. Training uses equal active-task mean NLL:
one joint track task and one marginal intensity task each. Targets remain masked
before loss arithmetic. Errors/stds are normalized by the selected training-only
codec scales. Native-unit NLL is reported separately; it differs by fixed log
scale constants, with two track dimensions counted jointly.

There is one declared spread model and seed 71, batch 128, AdamW learning rate
0.0003, weight decay 0.0001, gradient clipping 1, at most 50 epochs, patience 10
and minimum stopping improvement 0.001. The smallest selection NLL chooses the
checkpoint. The 2011–2014 storms are reused selection data, not independent
confirmation. No architecture grid or calibration-label selection is performed.

## Colab

1. Upload `notebooks/bob_cyclone_uncertainty_colab.ipynb` to Colab.
2. First upload **`bob_cyclone_uncertainty_code.zip`** when asked for source.
3. At the second prompt upload **`wind_mae_followup_v1_results (1).zip`**.
   The original deterministic ZIP and acceptance-only follow-up are insufficient.
4. Keep `RUN_FULL_UNCERTAINTY=False` and `RUN_CALIBRATION=False` initially.
5. Inspect acceptance: frozen means, two spread epochs, exact checkpoint reload,
   selection diagnostics and an exported `uncertainty_v1_results.zip`.
6. Set `RUN_FULL_UNCERTAINTY=True`; rerun parameters, training, diagnostics and
   export. Research and acceptance use separate directories. Keep calibration
   disabled until the full freeze is reviewed.

Drive mirroring is optional. Checkpoints and the entire frozen-mean contract are
copied each epoch. Restore the same source/configuration and run ID after a
disconnect. Changed runtime requires `ALLOW_RUNTIME_CHANGE=True` and is recorded
without a bitwise-equivalence claim. Source/config/data changes require a new run.

## CLI and freeze artifacts

```bash
# Acceptance only; calibration is unavailable for this run.
python -m modelling.uncertainty train --results "wind_mae_followup_v1_results (1).zip" --output runs/uncertainty_v1/acceptance --device cpu

# Full spread training, with a separate research freeze.
python -m modelling.uncertainty train --results "wind_mae_followup_v1_results (1).zip" --output runs/uncertainty_v1/research --full --device auto

# Resume exactly the same experiment and mirror epoch snapshots.
python -m modelling.uncertainty train --results "wind_mae_followup_v1_results (1).zip" --output runs/uncertainty_v1/research --full --resume --mirror-dir /persistent/uncertainty_v1/research
```

Before training, all uploaded file hashes and selected checkpoint hashes are
verified. Original weights load with `weights_only=True`. Each member must
reproduce its saved selection forecasts within 0.00001 degrees for latitude and
longitude and 0.001 native units for intensity across runtimes. The saved training
normalization, feature order, masks, codec and whole-storm roles must agree with
the immutable data package. The mean parameters are set non-trainable.

`mean/mean_freeze.json` records the three original checkpoint hashes, inference
contract, data/roles, source results and reproduction differences. A full spread
run adds `frozen_model.json`, binding mean weights, spread checkpoint, config,
source, feature order, normalization, codec, validation roles and scoring
conventions. Acceptance cannot create this research freeze. Artifact corruption
or source/config/data changes reject calibration.

Reports include native Gaussian NLL, analytic marginal CRPS, fixed-draw spatial
energy score, 50/80/90% interval coverage and width, joint track ellipse coverage
and area, and the unchanged point metrics. Counts use observed labels; they are
not silently reduced to persistence-eligible labels. Selection diagnostics are
row weighted over correlated six-hour observations, not coverage guarantees.

For inference on input features without any future target columns, the Python
API is `modelling.uncertainty.predict_frozen(frame, research_directory,
calibrated=False)`. It returns point forecasts, tangent-plane/native means and
standard deviations, and track correlation. `calibrated=True` requires the
completed calibration receipt to match the frozen model and its file checksum.
It changes only spread, and cannot be used with an acceptance run.

Gaussian wind has negative-tail probability. We report it without clipping.
Pressure mass outside the predeclared **diagnostic** bounds 850–1100 hPa is also
reported; these are not universal physical limits. These checks must inform any
later operational suitability assessment. Changing the distribution after
examining reserved calibration results would require a new experimental protocol,
not a second tuning pass on those same storms.

## Separate one-time calibration

After reviewing and fixing the full research model:

```bash
python -m modelling.uncertainty calibrate --frozen runs/uncertainty_v1/research --device auto
```

This is the only command that uses reserved 2015–2017 calibration labels. Means,
correlation, network weights and all modelling choices stay fixed. For the single
Gaussian, exact row-weighted observed-label NLL minimization yields three factors:
track `sqrt(mean(Mahalanobis²)/2)`, and intensity
`sqrt(mean(standardized_error²))` each. Factors are constrained to the declared
range [0.25, 4]. Correlated track covariance is multiplied by its factor squared;
intensity standard deviations by their factors. The resulting effective floors
can be below the pre-calibration training floors by the declared factor.

The command creates the fixed `research/calibration/` directory and refuses a
second fit there. A started-but-failed calibration is preserved for review; do not
delete it to try another model or objective. `calibration.json` binds the frozen
record, factors, counts, bounds and before/after diagnostics. Calibration fit-set
coverage is not independent validation and has no exact guarantee with 25 storms.

The notebook keeps `RUN_CALIBRATION=False` by default. Enable it only for the
reviewed research freeze. Rerunning a successful notebook calibration cell reads
the existing result and does not refit. Preserve source, original selected-model
results, frozen model, calibration receipt and all exports together.

The uncertainty runner has no final-test command or loader override. The next
stage is the [separate reviewed final-evaluation workflow](final-evaluation.md),
which binds this exact freeze/calibration receipt and keeps the test sealed
until its plan digest and persistent one-time gate are supplied.
