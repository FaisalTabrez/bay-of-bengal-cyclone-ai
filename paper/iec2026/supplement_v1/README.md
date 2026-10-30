# Supplement S1 Methods and data contract

Version 1.0, prepared 30 October 2026 for **Machine Learning for Cyclone Prediction in the Bay of Bengal**.

This supplement documents the supplied derived dataset and the implemented method. It does not supply new model results or complete the held experiments. Original source acquisition and extraction remain partly unresolved. Current source snapshots are documentation of the working implementation, not a replacement for historical training-run receipts or a new experiment freeze.

## Data identity and files

- Dataset version: 2.0.0, `bob_cyclone_training_v2.zip`.
- SHA-256: `3df324f20f373a8c868bc8f4b29449fb79226725e9e3705efbe898f83e67cd42`.
- Package creation time from metadata: 27 October 2026, 12:58:23 UTC. This is not a source acquisition timestamp.
- `dataset_manifest.json`: original package metadata, including ordered predictors and targets.
- `feature_contract.csv`: all 39 predictors, units, meanings and provenance limitations, in model order. Numerical native units are retained for fidelity. One knot equals 1852/3600 metres per second; one hPa equals 100 Pa. Metadata suffixes and the data handoff support the listed units; raw-source conversion implementations remain to be recovered.
- `label_populations.csv`: input and target availability counts, computed only from training and validation partitions.
- `validation_roles.csv`: complete original storm-time role assignments, copied without modification.
- `configs/`: current configuration snapshots, retained byte-for-byte, including the separate final-evaluation plan.
- `method_source/`: source snapshots for the inspected methods. These files were not executed to train, infer, calibrate or evaluate a model during manuscript revision.
- `provenance.json`: data, role and current-source identities and the explicitly unresolved construction fields.

All 39 standardised feature values precede all 39 missingness indicators. `SST_AVAILABLE` is a supplied feature in addition to the explicit `SST_C_MISSING` indicator. Identifiers and future targets are not predictors. Four targets are future latitude, longitude, maximum sustained wind in knots and minimum central pressure in hPa. `TARGET_TIME_6H` is an identifier, not an output.

The source snapshots are not a standalone executable training bundle. `provenance.json` maps them to their repository paths; use the project environment and the actual archived experiment source when reproducing a run. Final model weights and research-result archives remain outside this drafting supplement.

## Observed populations

| Population | Pairs | Storms | Wind targets | Wind persistence pairs | Pressure targets | Pressure persistence pairs |
|---|---:|---:|---:|---:|---:|---:|
| Bay training | 4560 | 235 | 2334 | 2090 | 2267 | 2122 |
| NIO training including Bay | 5557 | 289 | 2848 | 2553 | 2742 | 2562 |
| Full Bay validation | 854 | 50 | 647 | 627 | 650 | 631 |
| Selection role | 493 | 25 | 349 | 336 | 347 | 334 |
| Calibration role | 361 | 25 | 298 | 291 | 303 | 297 |

Persistence eligibility means that both current and future intensity are observed. Thus Bay training retains 244 wind and 145 pressure targets whose current values are missing. Selection has 13 such cases for each intensity; calibration has seven wind and six pressure cases. Counts across populations overlap: NIO contains Bay training, and validation contains both roles. They must not be summed as independent datasets.

The final test is omitted from this availability audit. Its 1018 pairs and 65 storms in the paper are metadata counts. No final-test Parquet was decoded for this supplement. Unknown intensity values are not reconstructed, and no assumption that labels are missing at random is made. Descriptive coverage increases from approximately half the Bay training rows to roughly three quarters of validation rows; that does not identify the intensity distribution of unreported cases.

## Temporal and geographical construction

The data handoff reports main-track filtering, valid positions, unique `SID + ISO_TIME` keys, and synoptic observations at 00, 06, 12 and 18 UTC. Targets are the next observation of the same cyclone only when the gap is exactly six hours. Current-minus-previous tendencies require a continuous six-hour predecessor. Model histories stop at a cyclone boundary, partition boundary or missing six-hour interval. Histories contain one to four exported input observations and never reconstruct features from earlier target columns.

Bay-affecting storm lifecycles extend beyond the geographic reporting box of 5-25 degrees north and 78-100 degrees east. The original event-index definition used to select Bay-affecting storms is not fully reconstructible from the supplied modelling package; recover it from the source construction notebooks. The geographic subset uses current input position, not future position.

The regional OISST crop has daily 0.25-degree cells. Arabian Sea additions outside the crop retain null SST with `SST_AVAILABLE=0`. ERA5 feature names identify 850, 500 and 200 hPa winds, 700 hPa relative humidity, 500 hPa geopotential height and 850/200 hPa temperatures, together with surface fields. Centre-linked deep-layer wind values are proxies, not annular environmental steering estimates.

The package does not record source download timestamps, exact spatial interpolation/nearest-neighbour policy, daily SST alignment, a forecast-origin cutoff for daily products, or every derived-feature formula. No interpolation or latency policy is inferred here. Exact raw-source reproduction and operational availability remain unresolved for those fields. `feature_contract.csv` marks uncertain definitions rather than silently inventing them.

## Transformations and residual coordinates

Training feature means and sample standard deviations come from the supplied normalisation files. For an observed value, subtract its training mean and divide by its training standard deviation. Replace standard deviations at most 1e-8 by one. Missing standardised values become zero and their masks become one. No target imputation occurs.

The track reference adds current six-hour latitude/longitude tendencies, replacing an unavailable tendency by zero. Both the reference and target are converted to east-north vectors using the spherical logarithmic map at the current cyclone centre, with Earth radius 6371.0088 km. The difference is the track residual. The output adds the learned residual vector to the anchor and uses the spherical exponential map to decode position. Native intensity anchors are observed current measurements or training current-feature means when absent.

For residual coordinates j, the scale is the training RMS with a floor of one native unit. The two track axes share the RMS calculated across both coordinates. Wind and pressure have separate scales. Residual means are not subtracted, so zero network output reproduces the anchor. The training-only codec stores these scales and fallback means. See `method_equations.md` and the copied source for the precise definitions.

## Model and optimisation settings

| Component | Fixed method description |
|---|---|
| Snapshot MLP | 78-64-32-4; SiLU after the first two hidden linear layers; dropout 0.1 after the first activation; final weights/bias start at zero |
| GRU | Linear 78-32 and SiLU; one forward GRU layer, hidden 64; concatenate final hidden and current projected vector to 96; dropout 0.1; three heads 96-32-2, 96-32-1 and 96-32-1 with SiLU and zero final layers |
| Trees | Four independent residual histogram-boosting regressors; learning rate 0.05, 100 iterations, seven leaves, minimum 30 samples per leaf, L2 regularisation 1; no random internal validation holdout |
| Deterministic seeds | 17, 29, 43; family selected by mean individual-seed composite, never best seed |
| Bay neural training | Batch 128; AdamW; learning rate 0.001; weight decay 0.0001; gradient norm clip 1; maximum 100 epochs; patience 12; minimum composite improvement 0.001 |
| Transfer | NIO pretraining: 30 fixed epochs at 0.001; Bay fine-tuning: maximum 100 epochs at 0.0003; rebase input/value-mask weights; reset optimiser; retain NIO output codec |
| Pooled GRU | NIO preprocessing/codec; fixed 30-epoch schedule; no Bay fine-tuning |
| Original loss | PyTorch SmoothL1 with beta 1 on scaled residuals; two track axes form one task; average each task over its observed coordinates and then average active tasks |
| Bounded wind follow-up | Original MLP, observed-current wind persistence replacement, or absolute scaled wind residual loss; other task losses unchanged; defined after initial selection inspection |
| Spread MLP | 78-32-5 with SiLU; four softplus standard deviations plus floors; correlation 0.95*tanh; single Gaussian around the frozen ensemble |
| Spread schedule | Seed 71; batch 128; AdamW learning rate 0.0003, weight decay 0.0001; clip 1; maximum 50 epochs, patience 10, minimum NLL improvement 0.001 |
| Calibration | One common track factor and separate wind/pressure factors; bounds 0.25-4; row-weighted observed-label Gaussian likelihood; fixed means/correlation; no repeated tuning |

SiLU denotes the sigmoid linear unit. The original deterministic configuration retains old descriptive status strings. They are historical metadata and must not be used to infer the completion status of the current experiments. Current source snapshots and archived training-source hashes have different purposes; final research reporting must identify the actual training and inference receipts used.

## Probability references and score conventions

The reference standard deviations are the frozen spread-initialisation values: in-sample deterministic training-error RMS, floored at 1.1 times the declared native floors in the implementation. This is a fitted reference and initialisation, not an out-of-fold estimate.

All three probability forecasts have identical frozen ensemble means:

1. Constant diagonal Gaussian: those frozen training-error scales, correlation zero.
2. Uncalibrated learned Gaussian: selected input-dependent scales and correlation, no scale factors.
3. Calibrated learned Gaussian: same means and correlation, with the frozen factors applied to standard deviations.

The same observations and denominators are used for paired score differences. Native-unit Gaussian NLL is different from scale-normalised training NLL by fixed log-scale constants. Track NLL is joint in east/north kilometres. Marginal CRPS is not a joint track score; the spatial energy score uses 512 draws and diagnostic seed 7101. Track covariance scaling is by the square of the track factor. Wind-pressure and cross-task dependence are not modelled.

Intervals and ellipses use nominal 50%, 80% and 90% levels. Coverage is reported alongside width/area. Whole-cyclone percentile intervals use 2000 draws and seed 1701. Nominal coverage on reused selection data or the calibration fit set is not independent evidence or a finite-sample guarantee. Gaussian wind can be negative; pressure mass outside the diagnostic 850-1100 hPa range is reported without clipping. Those pressure bounds are diagnostics, not universal physical limits.

## Focused related-work comparison

| Study | Relevant distinction from the present protocol |
|---|---|
| Giffard-Roisin et al. 2020, DOI 10.3389/fdata.2020.00001 | Spatial reanalysis and trajectories for 24-hour track prediction; a different input representation and lead time |
| Mawatwal and Das 2024, DOI 10.1007/s12524-024-01929-8 | NIO satellite intensity estimation; explicitly contrasts chronological independent testing with stratified splitting |
| Fernandez et al. 2025, DOI 10.1175/AIES-D-24-0066.1 | Learned bivariate Gaussian track-forecast uncertainty from forecast guidance; establishes relevant uncertainty prior work |

The present study combines six-hour tabular track/intensity residual comparisons with explicit missing-label populations and frozen-mean probability references. This is an application and evaluation contribution to be assessed by the held evidence, not a claim that these model classes or distributions are new.

## Items still on hold

Model-performance tables, outstanding scientific tests, claim-dependent ablations, final-test findings and performance conclusions remain on hold. This supplement does not change the existing model, calibration or evaluation protocol. Adding source documentation or describing declared references does not authorise a new fit or a test evaluation.
