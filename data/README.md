# Versioned data packages

`packages/` contains the original handoff archives, preserved byte-for-byte:

- `bob_cyclone_training_v2.zip`: dataset 2.0.0; default modelling input.
- `bob_cyclone_training_v1.zip`: frozen Bay reference, retained for reproducibility.
- `bob_cyclone_v1_acceptance_results.zip`: original v1 acceptance evidence, baseline JSON and a smoke checkpoint. That checkpoint is an execution reference, not trained research weights for the new GRU.

[`checksums.json`](checksums.json) records archive SHA-256 values and sizes. Inside v2, `package_file_checksums.json` also covers individual files. The runner verifies those bytes but refuses to decode `bob_test`. No raw global NOAA/ERA5 source archive is included.

Sources are NOAA IBTrACS v04r01, NOAA OISST v2.1 and ECMWF ERA5. Citation, licence references and modification notes are in [NOTICE.md](../NOTICE.md); detailed coverage and preprocessing are in the [data handoff](../docs/data-handoff.md). These are derived research tables, with upstream terms retained, not a new claim of ownership over provider data.

The v2 partition counts are 5,557 NIO pretraining, 4,560 Bay training, 854 Bay validation and 1,018 frozen Bay test samples. Bay training is included in NIO pretraining. Historical wind/pressure labels and regional SST are partially observed; the documented masks and train-only preprocessing are mandatory.

Data preparation is complete. Model skill and operational suitability have not been established. Test bytes are public as part of the reproducibility handoff, but the experimental protocol still keeps that partition sealed until final model freeze.
