# Frozen research model card

Status: final retrospective evaluation complete, 30 October 2026. This card describes the submitted frozen model, not the initial GRU prototype. Full results and uncertainty limitations are in [final-results.md](final-results.md).

## Purpose and outputs

Six-hour cyclone latitude, longitude, maximum sustained wind in knots and central pressure in hPa, with one joint Gaussian for current-centered east/north track displacement and Gaussian marginal intensity uncertainty. Intended use is reproducible research on the supplied Bay/NIO data packages. No real-time operational or warning-use validation is claimed.

## Model and preprocessing

- Selected mean: Bay-only snapshot MLP ensemble with absolute wind loss; seeds 17, 29, 43.
- Each member: 78 inputs (39 predictors plus 39 missingness indicators), hidden widths 64 and 32, four residual outputs, 7,268 parameters.
- Track anchor: constant-velocity latitude/longitude extrapolation converted to spherical tangent-plane displacement. Intensity anchor: current intensity, with frozen training-mean fallback when missing.
- Input normalization, imputation, residual scales, feature order and fallback values use training data and remain frozen. Missing target labels are masked.
- The ensemble mean is frozen before spread training. A 78→32→5 network predicts four standard deviations and track correlation. Training selected epoch 48 within 50 epochs; this is a single Gaussian, not a three-member Gaussian mixture.
- Before calibration, standard-deviation floors are 1 km for each track component and 0.5 kt/hPa for intensity; track correlation is bounded by ±0.95. Calibration then applies the three fixed scale multipliers.
- Calibration multiplies standard deviations by 0.9489812800977547 (track), 1.0572971499610047 (wind), 0.8083077712600791 (pressure). It changes no means or correlation. Wind–pressure noise dependence is not modelled.

## Data roles

| Role | Observations | Cyclones | Period |
|---|---:|---:|---|
| Bay training | 4,560 | 235 | Historical training partition |
| Selection | 493 | 25 | 2011–2014 |
| Reserved calibration | 361 | 25 | 2015–2017 |
| Frozen final test | 1,018 | 65 | 2018–2025 |

The NIO pretraining comparison used 5,557 observations from 289 cyclones and includes the complete Bay training subset. It is not independent additional Bay data. It was not the selected mean model. Provider provenance and terms are documented in [the data README](../data/README.md) and [NOTICE.md](../NOTICE.md).

## Final-test performance

Mean track error is 35.915 km versus 40.753 km for constant velocity (1,018 observations). Paired wind MAE is 2.727 kt versus 3.113 kt for persistence (779 labels); paired pressure MAE is 1.385 hPa versus 1.625 hPa (823 labels). Whole-cyclone paired nominal 95% intervals favor the model for all three outcomes. Full counts, geographic results, subgroup diagnostics and uncertainty scores appear in the results report.

Calibrated 90% coverage is 89.49% track, 89.11% wind and 92.02% pressure. Central 50% coverage is 64.44%, 67.20% and 60.92%: the model is not calibrated across all nominal levels. Calibration worsens wind CRPS slightly. Track probabilistic scores do not clearly improve over the fixed training-RMS Gaussian reference. Below 34 kt, wind and pressure MAE are slightly worse than persistence.

## Limits and maintenance

The study uses retrospective best-track and reanalysis/ocean predictors. Their availability does not demonstrate operational forecast-time availability. The data do not establish performance in other basins, longer lead times or a changing climate. Small subgroup samples limit regime-specific conclusions. Selection data were used repeatedly during development; spread initialization uses in-sample RMS. A single Gaussian cannot represent multimodal forecast outcomes, and no physical tail clipping is applied.

The final test is consumed. Do not use it to change architectures, features, epochs, calibration, thresholds or the chosen model. A subsequent model version requires a separately designed study with untouched evaluation data.

## Artifact identity

- Frozen model canonical digest: `6f4535600f4eeffd8ceeb361456fe96166752d8dae54486f413340f584542d84`.
- Calibration file SHA-256: `19a3bd974ce3500f7b1f5ccc7626e8ff40c506ed41745b6ab45ac5a85b4dad30`.
- Reviewed final plan: `457eda3ec98c67c905728e8bacd02e0a3150b1e0e15096bfc0ccdab51e954a04`.
- Archive contains frozen weights and preprocessing, original exports, exact final-evaluation transfer bundle, source snapshot, raw metrics, ledger receipts and verification instructions.

This local research handoff is separate from the previously published v0.3.0 code release. No new release identifier, DOI or repository-wide code licence is asserted by this card.
