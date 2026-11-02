# Frozen final model weights

`frozen_v1/` contains the exact model artifacts used for the completed final evaluation:

- `mean/seed_17.pt`, `mean/seed_29.pt`, `mean/seed_43.pt`: the three selected deterministic MLP members.
- `best.pt`: the Gaussian uncertainty network checkpoint around the frozen ensemble mean.
- `mean/normalization.json` and `mean/codec.json`: frozen preprocessing and residual decoding.
- `calibration/calibration.json`: the three fixed spread-calibration scales.
- The original freeze records, selection decision, training summary and validation roles preserve the inference contract and provenance.

All 14 files are byte-identical copies from the verified local research archive. Sizes and SHA-256 hashes are recorded in `frozen_v1_manifest.json`. The checkpoints are small enough to be stored directly in Git; cloning this branch retrieves all four checkpoints without Git LFS.

Use these weights with the matching architecture and inference contract in the [model card](../docs/model-card.md). The exact original inference source is preserved in [`data/frozen_runtime/uncertainty_v1_source.zip`](../data/frozen_runtime/uncertainty_v1_source.zip). See the [final results](../docs/final-results.md) for evaluation and limitations.

The final test has been consumed. These files publish the existing frozen model; they do not constitute new training, inference, calibration or evaluation.
