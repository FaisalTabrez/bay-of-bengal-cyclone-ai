# Frozen final-test review

Reviewed 30 October 2026. The submitted final export records a completed evaluation under the same reviewed plan as the submitted preflight export. Independent artifact and numerical checks passed. The model improves aggregate six-hour track, wind and pressure prediction over the declared physical references. Uncertainty performance is mixed, with excessive central coverage and no clear spatial-score improvement over the fixed training-RMS Gaussian reference.

## Frozen model and study design

Data preparation used NOAA IBTrACS v04r01, NOAA OISST v2.1 and ECMWF ERA5; provider attribution and derived-data terms are retained in [NOTICE.md](../NOTICE.md). The immutable v2 package supplies 5,557 NIO training observations, including all 4,560 Bay training observations. Selection uses 493 observations from 25 cyclones in 2011–2014. Calibration uses a disjoint 361 observations from 25 cyclones in 2015–2017. Whole cyclones remain within their declared chronological roles.

The deterministic comparison and bounded wind-loss follow-up selected a Bay-only snapshot MLP ensemble, with seeds 17, 29 and 43. Each member receives 39 standardized predictors and 39 missingness indicators, with hidden widths 64 and 32, and predicts four physical residuals. The selected model is the snapshot MLP; the initial GRU and NIO transfer models remain comparison experiments. Training-only imputation, normalization, residual scaling and fallback intensity values are frozen. Track residuals correct constant-velocity displacement in a current-centered spherical tangent plane; intensity residuals correct persistence. Missing intensity targets are masked rather than filled.

The three member forecasts define a fixed ensemble mean. A separate 78→32→5 network predicts four positive Gaussian standard deviations and a bounded east/north correlation, with means frozen. Its training ran for 50 epochs, selecting epoch 48 on selection data. The distribution is one Gaussian about the frozen mean, not a mixture. Wind and pressure noise dependence is not represented.

Calibration fitted exactly three standard-deviation multipliers on the reserved role: track 0.9489812801, wind 1.0572971500 and pressure 0.8083077713. Calibration leaves the means and correlation fixed and multiplies track covariance by the squared track factor. The final test estimates none of these quantities. The initial spread RMS uses in-sample training errors, not out-of-fold errors. Repeated selection-stage development should be disclosed; final-test evidence applies to the model frozen before that test.

## Evaluation and provenance

- Frozen test: 1,018 observations from 65 cyclones, input years 2018–2025.
- Final archive: `bob_cyclone_final_results.zip`; SHA-256 `207480715d7a53351c6b0803c6205db45a6876550f8beee0deb0eeda8d40b44c`.
- Preflight archive: `bob_cyclone_final_preflight_results.zip`; SHA-256 `0cf0bca30901d10da8f5331c2bfc21b84d26462df9d5da4bdcf61f08bf9192b9`.
- Reviewed plan digest: `457eda3ec98c67c905728e8bacd02e0a3150b1e0e15096bfc0ccdab51e954a04`.
- Completed attempt: `d3661f8f-d99b-4a2c-9547-b10fab7879d5`.
- All 91 final and 71 preflight manifested files passed exact file-set, size and SHA-256 checks. Execution receipts and ledger agree.
- Frozen model and calibration files are byte-identical to the original calibrated export. Prepared artifacts match preflight; the current evaluator fingerprint matches the plan.
- No training or calibration refit is recorded. Inference input files exclude future target labels; their future timestamp specifies the fixed six-hour horizon.

The independent review used only submitted predictions, distribution parameters and the exported cached test frame. It did not decode the original dataset, rerun inference, train a model or refit calibration. It reproduced point errors, paired whole-cyclone bootstrap intervals, Gaussian likelihoods, marginal CRPS and coverage. Archive integrity also covers the supplied energy scores and other summaries; energy-score Monte Carlo calculations were not independently repeated. These checks validate consistency of the submitted record, rather than independently observing every action in the remote Colab session.

## Six-hour point predictions

Intensity comparisons use the same observations for model and persistence. Counts differ because missing labels and missing current intensity exclude observations from paired comparisons.

| Outcome | Model | Declared reference | Relative error reduction | Paired observations |
|---|---:|---:|---:|---:|
| Mean track error | 35.915 km | 40.753 km, constant velocity | 11.87% | 1,018 |
| Wind MAE | 2.727 kt | 3.113 kt, persistence | 12.41% | 779 |
| Pressure MAE | 1.385 hPa | 1.625 hPa, persistence | 14.74% | 823 |

The 95% paired whole-cyclone percentile bootstrap intervals for model-minus-reference error are:

| Outcome | Difference interval |
|---|---:|
| Track | −6.213 to −3.585 km |
| Wind | −0.628 to −0.162 kt |
| Pressure | −0.450 to −0.046 hPa |

All three nominal intervals favor the model. The protocol used 2,000 resamples of complete cyclones, seed 1701, with row-weighted means. These are nominal intervals; no multiplicity adjustment is claimed.

On all available target labels, wind MAE is 2.719 kt across 808 labels and pressure MAE is 1.404 hPa across 852 labels. These values must not replace the paired model values when comparing with persistence. Track median and 90th-percentile error are 28.802 and 69.897 km.

The predeclared geographic Bay cohort (current latitude 5–25°N, longitude 78–100°E) contains 789 observations from 62 cyclones. Relative reductions are 11.95% for track, 13.19% for paired wind and 17.58% for paired pressure; the intensity paired counts are both 656. These are a subset of the full test, not additional independent evidence.

## Predeclared diagnostic subgroups

| Current wind category | Rows | Cyclones | Track error reduction | Wind error reduction | Pressure error reduction |
|---|---:|---:|---:|---:|---:|
| Below 34 kt | 479 | 62 | 15.01% | −1.27% | −4.47% |
| 34 to below 64 kt | 246 | 26 | 1.71% | 14.87% | 19.70% |
| At least 64 kt | 68 | 10 | 8.29% | 24.59% | 27.04% |
| Unknown wind | 225 | 29 | 14.14% | Undefined, no paired wind labels | −41.23%, 41 paired labels |

Subgroups use current intensity, not future outcomes. Cyclones may contribute to multiple subgroups. These descriptive estimates have small and unequal denominators and should not be interpreted as confirmed differences between groups. Under 34 kt, wind MAE is 1.536 versus 1.517 kt and pressure MAE is 0.714 versus 0.684 hPa: aggregate improvement does not mean improvement in every regime.

Near land (0–50 km), reductions are 12.81% track, 24.93% wind and 19.65% pressure. At 50–200 km, wind is approximately tied (+0.34%) and pressure is worse (−3.56%). Beyond 200 km, reductions are 12.03%, 4.99% and 17.42%. These are distance-to-land diagnostics, not verified landfall-event forecasts.

## Frozen uncertainty forecasts

All three distributions use the same frozen point mean. Lower negative log-likelihood (NLL), CRPS and energy score are better. The training-RMS reference is a diagonal Gaussian initialized from frozen in-sample training residuals; it is not an out-of-fold estimate.

| Score, full cohort | Training-RMS reference | Uncalibrated | Calibrated |
|---|---:|---:|---:|
| Track joint NLL | 9.7453 | 9.7349 | 9.7313 |
| Track energy score, km | 26.7897 | 26.9296 | 26.7842 |
| Wind NLL | 3.0958 | 2.8918 | 2.8803 |
| Wind CRPS, kt | 2.6311 | 2.3008 | 2.3221 |
| Pressure NLL | 2.4435 | 1.9398 | 1.8979 |
| Pressure CRPS, hPa | 1.3322 | 1.0688 | 1.0371 |

Whole-cyclone paired intervals favor calibrated intensity NLL and CRPS over the training-RMS reference. Track NLL and energy-score difference intervals include zero against that reference. This does not support a clear gain in spatial probabilistic skill over the simple reference.

Calibration has mixed effects relative to the frozen uncalibrated Gaussian: track energy and pressure CRPS improve, but wind CRPS worsens by 0.02124 kt (95% paired interval +0.01441 to +0.02842 kt). The three calibrated-minus-uncalibrated NLL intervals each include zero. Report the frozen calibrated model as declared; do not select a different version using these test results.

| Nominal coverage | Track ellipse, N=1,018 | Wind interval, N=808 | Pressure interval, N=852 |
|---|---:|---:|---:|
| 50% | 64.44% | 67.20% | 60.92% |
| 80% | 84.68% | 82.55% | 86.15% |
| 90% | 89.49% | 89.11% | 92.02% |

The 90% coverage is close to nominal, but this does not establish calibration across the distribution. Central 50% regions overcover for all three outcomes. The supplied coverage figure was visually checked against these values. A single Gaussian remains a limited uncertainty representation. No tail clipping was applied.

## Claim boundary and next phase

This is retrospective six-hour prediction using best-track observations and reanalysis/ocean predictors. It does not establish real-time operational forecast performance, warning skill, or performance against an operational numerical forecasting system.

The final holdout has been consumed. Preserve the original exports, completed ledger, plan, frozen checkpoints and calibration alongside the analysis receipt. The next phase is to prepare the research results, methods, limitations and reproducibility handoff. Any future model improvement requires a separate study and an untouched evaluation set; this test must not guide further tuning or calibration.

See the [model card](model-card.md) and [reproducibility archive guide](reproducibility-archive.md). The archive includes the machine-readable verification record, full exported metrics, the reviewed plan and the portable audit script `scripts/verify_final_exports.py`. All predeclared cohorts, scores and intervals remain available in the exported metrics.
