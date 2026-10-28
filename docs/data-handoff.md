# Bay of Bengal Cyclone AI — Complete Data Handoff

## 1. Handoff status

The data engineering and preprocessing scope is complete. The final deliverable is ready for model development in Google Colab or another Python training environment.

Primary package:

- File: `bob_cyclone_training_v2.zip`
- Size: approximately 2.30 MB
- SHA-256: `3df324f20f373a8c868bc8f4b29449fb79226725e9e3705efbe898f83e67cd42`
- Dataset version: `2.0.0`
- Historical period: 1982–2025
- Temporal resolution: 6 hours

Frozen reference package:

- File: `bob_cyclone_training_v1.zip`
- SHA-256: `469a6da0b9cb76c4e794feded1cf511799bd9ed7593436b0845b373772e6b0cb`

The v1 package has already passed an independent Colab acceptance test. The v2 package has passed all Fabric-side integrity, schema, linkage, completeness and physical-range checks.

## 2. Research objective

The immediate modelling task is six-hour-ahead probabilistic prediction of tropical cyclone position and intensity using learned historical cyclone states and physically meaningful environmental variables.

The scientific focus is the Bay of Bengal and the eastern Indian coast, especially:

- Andhra Pradesh
- Odisha
- West Bengal
- Tamil Nadu
- Puducherry

The recommended modelling strategy is:

1. Pretrain on the expanded North Indian Ocean training data.
2. Fine-tune on the frozen Bay of Bengal training split.
3. Select the model and hyperparameters using the frozen Bay validation split.
4. Open the frozen Bay test split only once after the complete model configuration is fixed.

The current package supports historical track and intensity prediction. It does not claim to predict the exact date or path of an individual cyclone decades in advance.

## 3. Scope completed by the data team

The completed work includes:

- Acquisition and preservation of source data in Microsoft Fabric OneLake.
- Standardization of IBTrACS cyclone observations.
- Identification of Bay of Bengal and broader North Indian Ocean cyclone records.
- Six-hour temporal alignment.
- ERA5 pressure-level feature extraction.
- ERA5 surface feature extraction.
- OISST processing for the Bay domain.
- Feature engineering and six-hour target construction.
- Chronological, cyclone-safe dataset partitioning.
- Missing-value and missing-target policies.
- Training-only normalization statistics.
- Bronze, Silver and Gold Lakehouse organization.
- Portable Parquet exports for Colab.
- Manifests, quality reports and file checksums.
- Independent v1 Colab loading and one-epoch training verification.
- Persistence and constant-velocity validation baselines.

The modelling team owns the remaining model architecture, training, tuning, calibration, experiments and final evaluation.

## 4. Source datasets

### 4.1 NOAA IBTrACS v04r01

Source file:

`Files/bronze/tracks/ibtracs/ibtracs.NI.list.v04r01.csv`

Source archive properties:

- Raw observations: 62,864
- Raw columns: 174
- Modern observations during 1982–2025: 17,629
- Standard six-hour observations: 8,917
- Finalized main-track observations: 8,726
- Finalized storms: 450

Filtering rules:

- Historical period restricted to 1982–2025.
- Retained times at 00, 06, 12 and 18 UTC.
- Retained `TRACK_TYPE = MAIN`.
- Excluded 158 `US-PROVISIONAL` observations.
- Excluded 33 `SPUR-MERGE` observations.
- Removed malformed positions and duplicate `SID + ISO_TIME` keys.
- Used New Delhi wind and pressure fields when available, with WMO values as fallback.
- Converted wind from knots to metres per second where required.

The final North Indian Ocean observations include lifecycle segments classified as:

| Subbasin | Observations | Storms represented |
|---|---:|---:|
| Bay of Bengal (`BB`) | 4,893 | 352 |
| Arabian Sea (`AS`) | 2,626 | 166 |
| Maritime/WP lifecycle segments (`MM`) | 1,207 | 41 |

Storm counts by subbasin are not additive because a storm can cross subbasins. Complete storm lifecycles were preserved.

### 4.2 NOAA OISST v2.1

Coverage used:

- Daily 0.25-degree sea-surface temperature.
- Latitude: 5°N to 25°N.
- Longitude: 78°E to 100°E.
- Period: 1982–2025.
- Annual regional files: 44.
- Monthly summary rows: 528.
- Mean regional SST: approximately 28.56°C.

The regional files were created by downloading annual NOAA OISST data, cropping to the Bay domain, validating the requested dates and coordinates, and removing temporary global staging files.

Important limitation: the Arabian Sea additions lie west of 78°E and therefore fall outside the v1 OISST crop. In v2.0 these rows intentionally contain:

```text
SST_C = null
SST_AVAILABLE = 0
```

This is intentional missingness, not a failed join. The model must use the availability flag and a train-derived imputation value or a mask-aware input layer.

### 4.3 ECMWF ERA5 pressure levels

ERA5 was accessed through the public Earthmover Icechunk archive backed by ECMWF/Copernicus and NSF NCAR data.

Pressure-level inputs:

- U and V wind at 850 hPa.
- U and V wind at 500 hPa.
- U and V wind at 200 hPa.
- Relative humidity at 700 hPa.
- Geopotential at 500 hPa, converted to geopotential height in metres.
- Temperature at 850 hPa.
- Temperature at 200 hPa.

Derived pressure-level features:

- 200–850 hPa wind shear magnitude.
- Deep-layer U wind proxy.
- Deep-layer V wind proxy.
- 850–200 hPa temperature difference.

The deep-layer values are centre-sampled proxies. They must not be described as a final environmental steering flow until a surrounding annulus or spatial-field calculation is implemented.

### 4.4 ECMWF ERA5 single levels

Surface inputs:

- Mean sea-level pressure.
- Surface pressure.
- 10 m U wind.
- 10 m V wind.
- 2 m temperature.
- 2 m dew point.
- Total-column water vapour.
- CAPE.

Derived surface features:

- Mean sea-level pressure in hPa.
- Surface pressure in hPa.
- 10 m wind-speed magnitude.
- Temperature and dew point in Celsius where retained in Silver tables.
- 2 m relative humidity.

All 1,942 new Arabian Sea observations received complete ERA5 pressure and surface features. The full v2 atmospheric tables contain 8,726 rows with zero missing ERA5 feature rows.

## 5. OneLake architecture

The Lakehouse follows a Bronze–Silver–Gold design.

### Bronze

Preserves source-aligned files and provenance.

Important path:

`Files/bronze/tracks/ibtracs/ibtracs.NI.list.v04r01.csv`

### Silver

Contains standardized tracks and aligned environmental features.

Core v1 tables:

| Table | Rows | Purpose |
|---|---:|---|
| `dbo.silver_ibtracs_bob_tracks` | 6,784 | Bay-affecting cyclone observations |
| `dbo.silver_cyclone_event_index` | 352 | One row per v1 cyclone |
| `dbo.silver_track_sst_features` | 6,784 | Track-linked OISST |
| `dbo.silver_era5_track_features` | 6,784 | ERA5 pressure features |
| `dbo.silver_era5_surface_track_features` | 6,784 | ERA5 surface features |

Core v2 tables:

| Table | Rows | Purpose |
|---|---:|---|
| `dbo.silver_ibtracs_nio_tracks_v2` | 8,726 | Expanded NIO standardized tracks |
| `dbo.silver_nio_cyclone_event_index_v2` | 450 | One row per v2 cyclone |
| `dbo.silver_ibtracs_nio_additions_v2` | 1,942 | New Arabian Sea extraction workload |
| `dbo.silver_era5_nio_addition_features_v2` | 1,942 | Combined new ERA5 features |
| `dbo.silver_era5_nio_track_features_v2` | 8,726 | Full v2 pressure features |
| `dbo.silver_era5_nio_surface_track_features_v2` | 8,726 | Full v2 surface features |
| `dbo.silver_nio_track_sst_features_v2` | 8,726 | Full v2 SST compatibility table |

Annual resumable ERA5 addition files are stored under:

`Files/silver/atmosphere/era5/nio_v2_additions`

### Gold

Contains model-ready feature and target tables.

| Table | Rows | Purpose |
|---|---:|---|
| `dbo.gold_cyclone_feature_master` | 6,784 | Frozen v1 feature master |
| `dbo.gold_track_intensity_6h` | 6,432 | Frozen v1 supervised pairs |
| `dbo.gold_nio_cyclone_feature_master_v2` | 8,726 | Expanded v2 feature master |
| `dbo.gold_nio_track_intensity_6h_v2` | 8,276 | Expanded v2 supervised pairs |

Export location:

`Files/exports/gold_snapshot`

## 6. Dataset versions and intended use

### v1: frozen Bay reference

The v1 package contains full lifecycles of Bay-affecting storms. It is not limited to points geographically inside the Bay.

| Partition | Years | Samples | Cyclones |
|---|---:|---:|---:|
| Train | 1982–2010 | 4,560 | 235 |
| Validation | 2011–2017 | 854 | 50 |
| Test | 2018–2025 | 1,018 | 65 |
| Total | 1982–2025 | 6,432 | 350 with valid pairs |

There are 352 storms in the v1 event archive. Two storms do not contribute a valid continuous six-hour input-target pair.

Geographic point coverage within the working Bay box of 5–25°N and 78–100°E:

- Train: 3,005 rows, 65.90%.
- Validation: 673 rows, 78.81%.
- Test: 789 rows, 77.50%.

Results should be reported for the complete frozen test partition and for the Bay-domain subset.

### v2: NIO pretraining plus frozen Bay evaluation

The v2 Gold tables contain:

- 450 cyclones.
- 8,726 track observations.
- 8,276 valid continuous six-hour samples.
- No internal non-six-hour gaps.
- No duplicate keys.
- Maximum one split per cyclone.

The v2 portable package intentionally contains four modelling partitions:

| Package directory | Samples | Intended use |
|---|---:|---|
| `nio_pretrain_train` | 5,557 | NIO pretraining |
| `bob_train` | 4,560 | Bay-specific fine-tuning |
| `bob_validation` | 854 | Model selection and tuning |
| `bob_test` | 1,018 | One-time final evaluation only |

The broader NIO validation and test rows remain in the Lakehouse for research audits but are not used as the official paper evaluation. The official validation and test sets remain the frozen Bay v1 partitions.

## 7. Feature contract

Each exported modelling partition has 49 columns:

- 6 identifier columns.
- 39 feature columns.
- 4 numerical target columns.

Identifier columns:

```text
SID
NAME
SEASON
ISO_TIME
TARGET_TIME_6H
SPLIT
```

Feature groups:

- Current storm position and intensity.
- Distance to land.
- SST and SST availability.
- ERA5 pressure-level winds.
- Atmospheric humidity, geopotential and temperature structure.
- Vertical wind shear and deep-layer wind proxies.
- ERA5 surface pressure, wind, temperature, humidity, moisture and CAPE.
- Month and day-of-year cyclical encodings.
- Previous six-hour changes in position, wind and pressure.

Prediction targets:

```text
TARGET_LAT_6H
TARGET_LON_6H
TARGET_WIND_KTS_6H
TARGET_PRESSURE_HPA_6H
```

`TARGET_TIME_6H` is an identifier, not a numerical prediction target.

## 8. Feature engineering formulas

Calendar encodings:

```text
MONTH_SIN = sin(2π × MONTH_NUMBER / 12)
MONTH_COS = cos(2π × MONTH_NUMBER / 12)
DAY_OF_YEAR_SIN = sin(2π × DAY_OF_YEAR / 365.25)
DAY_OF_YEAR_COS = cos(2π × DAY_OF_YEAR / 365.25)
```

Recent tendencies:

```text
LAT_CHANGE_6H = LAT - PREVIOUS_LAT
LON_CHANGE_6H = LON - PREVIOUS_LON
WIND_CHANGE_6H_KTS = WIND_KTS - PREVIOUS_WIND_KTS
PRESSURE_CHANGE_6H_HPA = PRESSURE_HPA - PREVIOUS_PRESSURE_HPA
```

Targets are constructed using the next chronologically ordered observation within the same `SID`. Only rows where `TARGET_GAP_HOURS = 6.0` enter supervised datasets.

## 9. Missing-data policy

### Inputs

The following v1 training features contain missing values and require masks:

| Feature | v1 training coverage |
|---|---:|
| `WIND_KTS` | 51.36% |
| `PRESSURE_HPA` | 49.93% |
| `SST_C` | 48.11% |
| `LAT_CHANGE_6H` | 94.85% |
| `LON_CHANGE_6H` | 94.85% |
| `WIND_CHANGE_6H_KTS` | 43.99% |
| `PRESSURE_CHANGE_6H_HPA` | 45.22% |

Required procedure:

1. Create one binary missingness indicator for every feature containing nulls.
2. Use the appropriate training-only mean for imputation.
3. Normalize using only the corresponding training statistics.
4. Never calculate normalization from validation or test data.
5. Do not interpret an imputed value as an observed measurement.

### Targets

Track targets are complete. Historical wind and pressure targets are partially observed.

v1 target coverage:

| Target | Train | Validation | Test |
|---|---:|---:|---:|
| Latitude | 100% | 100% | 100% |
| Longitude | 100% | 100% | 100% |
| Wind | 51.18% | 75.76% | 79.37% |
| Pressure | 49.71% | 76.11% | 83.69% |

Required procedure:

- Use masked multitask losses.
- Use every sample for track prediction.
- Apply wind loss only where the wind target exists.
- Apply pressure loss only where the pressure target exists.
- Never fill a missing label with zero.

The increase in intensity-label availability over time is a reporting-distribution shift and must be mentioned as a limitation.

## 10. Normalization files

The v2 ZIP contains:

- `normalization_nio_train.json` for NIO pretraining.
- `normalization_bob_train.json` for Bay fine-tuning and Bay-only experiments.

Do not use NIO normalization automatically for the Bay fine-tuning stage. The training pipeline must state clearly whether preprocessing is frozen from pretraining or reconfigured using the provided Bay training statistics.

Recommended first experiment:

1. Pretrain with NIO normalization.
2. Initialize the Bay model from the pretrained weights.
3. Use Bay normalization for fine-tuning inputs.
4. Use the identical Bay normalization for validation and final testing.

## 11. Package contents

`bob_cyclone_training_v2.zip` contains:

```text
bob_test/
bob_train/
bob_validation/
nio_pretrain_train/
normalization_bob_train.json
normalization_nio_train.json
package_file_checksums.json
quality_report_v2.json
training_manifest_v2.json
```

Each data directory contains one Snappy-compressed Parquet part file and a Spark `_SUCCESS` marker.

The receiving team should first verify the ZIP SHA-256 and then verify the internal files using `package_file_checksums.json`.

## 12. Validation evidence

Completed integrity checks include:

- Exact expected row counts.
- Matching 49-column schemas across exported datasets.
- Zero duplicate `SID + ISO_TIME` keys.
- Zero cyclone overlap across frozen Bay partitions.
- Chronological ordering within each cyclone.
- Maximum one split per cyclone.
- Exact six-hour target continuity.
- Zero missing ERA5 feature rows.
- Zero invalid physical-range rows in the new ERA5 extraction.
- Complete one-to-one linkage between new tracks and ERA5 features.
- Complete v1-key preservation in v2.
- Correct SST availability flags.
- Train-only normalization.
- Successful independent Colab loading.
- Successful one-epoch masked multitask training and checkpoint save.

v1 acceptance ZIP:

`bob_cyclone_v1_acceptance_results.zip`

Expected contents:

```text
colab_preprocessing_contract.json
normalization_train_only.json
smoke_model_checkpoint.pt
smoke_test_metrics.json
training_manifest.json
v1_acceptance_summary.json
validation_baselines.csv
validation_baselines.json
```

## 13. Baseline results

Validation baselines were calculated without using the test partition.

| Model | Mean track error, all | Mean track error, Bay | Wind MAE | Pressure MAE |
|---|---:|---:|---:|---:|
| Persistence | 85.251 km | 77.358 km | 2.431 kt | 1.494 hPa |
| Constant velocity/tendency | 39.040 km | 37.932 km | 2.992 kt | 1.677 hPa |

Best baseline thresholds:

- Track: constant velocity, 37.932 km mean Bay error.
- Wind: persistence, 2.353 kt Bay MAE.
- Pressure: persistence, 1.447 hPa Bay MAE.

The one-epoch neural smoke model was only an execution test and is not a scientific result. It produced approximately 495 km mean Bay track error after one CPU epoch and should not be compared as a trained model.

## 14. Instructions for the modelling team

Required workflow:

1. Verify `bob_cyclone_training_v2.zip` SHA-256.
2. Validate the four row counts and internal checksums.
3. Build missingness indicators before imputation.
4. Reproduce the persistence and constant-velocity baselines.
5. Pretrain on `nio_pretrain_train`.
6. Fine-tune on `bob_train`.
7. Select hyperparameters using `bob_validation` only.
8. Report deterministic and probabilistic validation metrics.
9. Freeze preprocessing, model architecture, weights and thresholds.
10. Evaluate `bob_test` once.
11. Report results for the full Bay test partition and the geographic Bay subset.

Recommended initial architecture:

- A compact temporal convolutional network, GRU or small state-space encoder.
- Separate output heads for position, wind and pressure.
- Masked wind and pressure losses.
- Residual prediction relative to persistence or constant velocity rather than unconstrained absolute position prediction.
- A probabilistic output such as Gaussian or mixture-density heads after the deterministic pipeline is stable.

Recommended physics guidance:

- Spherical displacement consistency.
- Track smoothness.
- Pressure–wind consistency.
- Plausible intensity-change limits.
- Land-decay penalty where appropriate.
- Environmental steering based on a surrounding field or annulus in a later spatial extension.

## 15. Required evaluation metrics

Track:

- Great-circle position error in kilometres.
- Mean, median and 90th-percentile error.
- Latitude and longitude error.
- Results by intensity, basin sector and land proximity.

Intensity:

- Wind MAE in knots.
- Pressure MAE in hPa.
- Cyclone-category macro F1 if category labels are constructed.
- Rapid-intensification and landfall-subset diagnostics.

Probabilistic outputs:

- Negative log-likelihood.
- CRPS.
- Prediction-interval coverage.
- Reliability diagrams.
- Sharpness.
- Calibration by intensity and coastal proximity.

## 16. Known limitations

- The dataset is still small for a large deep neural network. Use compact models and strong regularization.
- The NIO expansion increases training samples from 4,560 to 5,557, a 21.9% increase rather than a multi-fold increase.
- Arabian Sea SST is intentionally missing in v2.0.
- ERA5 values are sampled at the cyclone centre, not extracted as spatial image patches.
- The deep-layer wind feature is a centre-sampled proxy, not a full environmental steering calculation.
- Historical IBTrACS intensity reporting is incomplete and improves over time.
- Agency wind-averaging conventions differ; New Delhi values were preferred with WMO fallback.
- Complete lifecycles of Bay-affecting storms include positions outside the geographic Bay box.
- The dataset does not yet contain validated administrative-state landfall labels.
- The current package does not support village-level inundation, river-level, landslide or loss claims.

## 17. Deferred work

The following were intentionally excluded from the core handoff:

- Rainfall labels and rainfall heatmaps.
- CMIP6 future-climate scenarios.
- Long-horizon monthly cyclone occurrence modelling.
- Synthetic future cyclone generation.
- River-level prediction.
- Inundation modelling.
- Landslide prediction.
- Population and monetary loss estimation.

Rainfall was deferred because an attempted CHIRPS pilot returned an incorrect time period and other remote access routes were impractical during the fast-track build. The invalid rainfall pilot was removed and no rainfall file is represented as valid training data.

If rainfall is added later, preferred sources and checks are:

- NASA GPM IMERG Final V07 for 2000 onward.
- CHIRPS only as a land-focused fallback with its ocean limitation stated.
- Mandatory verification of requested and returned dates before accepting a file.
- Separate semantic dataset version and manifest.

If climate projection work is added later:

- Use multiple CMIP6 models rather than one model.
- Use SSP2-4.5 and SSP5-8.5 as contrasting scenarios.
- Bias-correct against historical ERA5 and OISST.
- Model monthly or seasonal occurrence probabilistically.
- Report future windows such as 2026–2040, 2041–2060 and 2061–2075.
- Do not claim exact cyclone dates or exact tracks decades ahead.

## 18. Important technical issues already resolved

### Parquet nanosecond timestamps

Fabric Spark rejected `INT64 (TIMESTAMP(NANOS,false))`. The solution was to convert timestamps through text or Spark timestamp types before writing Delta/Parquet.

### OISST grid dimensions

The regional grid contained 80 latitude cells and 88 longitude cells because coordinates are cell centres from 5.125 to 24.875 and 78.125 to 99.875. Assertions expecting 81 × 89 were incorrect and were replaced with coordinate/range validation.

### OISST staging files

Temporary global annual files were removed after regional crops were validated. Code was made tolerant of already-removed staging files.

### ERA5 certificate error

The Icechunk Rust client initially reported `UnknownIssuer`. The working session sets `SSL_CERT_FILE`, `AWS_CA_BUNDLE`, `REQUESTS_CA_BUNDLE` and `CURL_CA_BUNDLE` to the Certifi bundle before importing and opening Icechunk.

### PCodec dependency

The ERA5 archive requires PCodec. The environment must install and import `numcodecs.pcodec` before reading arrays.

### Spark versus Python OneLake paths

- Spark writes use relative paths such as `Files/exports/...`.
- Python file operations use mounted paths such as `/lakehouse/default/Files/exports/...`.

Passing the mounted absolute path to Spark produced a OneLake HTTP 400 error.

### Interactive authentication widgets

Fabric notebooks did not reliably render `ipywidgets` credential boxes. Public authenticated-free data routes were used where possible. Credentials should not be embedded in notebooks or packages.

## 19. Reproducibility and governance rules

- Never overwrite v1 with v2 data.
- Never silently change a package after recording its checksum.
- Any SST expansion, rainfall addition or feature correction requires a new semantic version.
- Keep source versions, access dates, extraction code and checksums with each release.
- Treat `SID` and timestamps as identifiers, not ordinary numerical predictors.
- Keep cyclones, not individual rows, as the unit of partitioning.
- Do not randomly repartition the supplied data.
- Keep the test split sealed until final evaluation.
- Record random seeds, software versions, configurations and checkpoint hashes for every reported experiment.

## 20. Handoff checklist

- [ ] Download `bob_cyclone_training_v2.zip` from OneLake.
- [ ] Verify its SHA-256 checksum.
- [ ] Copy v1, v2 and acceptance artifacts to two independent storage locations.
- [ ] Download and preserve the Fabric notebooks used for data construction.
- [ ] Preserve the v1 Colab acceptance notebook.
- [ ] Confirm the recipient can load all four v2 Parquet partitions.
- [ ] Confirm the recipient understands the missing-input and masked-target policies.
- [ ] Confirm the recipient will not tune against `bob_test`.
- [ ] Confirm the baseline thresholds are reproduced.
- [ ] Record the handoff date, recipient and storage locations.

## 21. Formal completion statement

The data acquisition, cleaning, preprocessing, feature engineering, quality assurance, versioning and portable packaging work for the core Bay of Bengal Cyclone AI project is complete.

The receiving team can begin North Indian Ocean pretraining and Bay-specific fine-tuning without continued access to Microsoft Fabric. Subsequent model design, training, probabilistic calibration, experimental evaluation and paper-result generation are the responsibility of the modelling team.
