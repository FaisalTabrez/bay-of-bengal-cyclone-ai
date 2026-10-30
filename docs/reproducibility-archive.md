# Final research reproducibility archive

The final holdout was evaluated on 30 October 2026. Use the saved artifacts for inspection and numerical reproduction. Do not start a new final-test attempt or change the frozen model using these outcomes.

## Contents

`dist/bob_cyclone_research_archive_v1.zip` is a local handoff, separate from the published v0.3.0 code release. Its adjacent `.sha256` file checks the ZIP itself. The internal `ARCHIVE_MANIFEST.json` records the size and SHA-256 of every payload file, the reviewed plan, frozen model identity and test-consumed status.

| Directory | Contents |
|---|---|
| `source/` | Current code, tests, documentation, immutable original data packages and frozen inference source snapshot |
| `inputs/` | Byte-identical final, preflight and calibrated uncertainty exports, plus the exact final-evaluation transfer ZIP |
| `frozen_model/` | Extracted unchanged weights, preprocessing, declarations and calibration for inspection |
| `final_run/` | Completed execution artifacts, reviewed plan, test cache, CSV predictions, metrics and ledger receipts |
| `evidence/` | Independent score verification, original final coverage figure and fresh synthetic/software test logs |

Original exports may retain their recorded Colab paths and runtime details. The archive preserves provenance; it is not a newly sanitized public release. Supporting deterministic-selection records are embedded in the calibrated export. Full earlier training-run histories are retained separately in the original project artifacts and are not all duplicated in this archive. No raw global IBTrACS, OISST or ERA5 acquisition archive is included.

## Verify integrity first

Extract the archive into a new directory. From that directory, with Python 3.12 or later:

```bash
python verify_archive.py .
```

This standard-library check validates every manifested file and rejects missing or extra payload files. Alternatively, verify the ZIP without extracting it:

```bash
python scripts/verify_research_archive.py dist/bob_cyclone_research_archive_v1.zip
```

The SHA-256 sidecar and manifest detect changes; they do not independently authenticate the author. Preserve a trusted copy of the sidecar alongside the original archives.

## Reproduce the submitted-score audit

From the extracted archive root, install the project requirements in a fresh environment and run:

```bash
python -m venv .venv
# Activate the environment using the command for your shell.
python -m pip install -e "./source"
python source/scripts/verify_final_exports.py --inputs-dir inputs --output-dir reproduced_review
```

Create the environment and audit output outside the archive root if you intend to repeat the strict no-extra-files integrity check afterward. That check deliberately rejects generated files. The code requirements allow compatible versions; `source/requirements/checked.txt` records earlier verified local versions, while the reviewed plan and inference records preserve the actual final runtime. The audit records its own runtime separately.

The audit checks both submitted manifests, prepared-plan equality, frozen-weight/calibration identity and execution receipts. It uses only the submitted cache and predictions to reproduce point errors, paired whole-cyclone bootstrap intervals, Gaussian likelihoods, marginal CRPS and coverage. It does not invoke original test decoding, model inference, training or calibration. Track energy scores remain in the original exports; this independent audit does not repeat their Monte Carlo computation.

To run the synthetic software regression tests from `source/`:

```bash
python -m unittest discover -s tests -v
```

These tests demonstrate software behavior and do not constitute additional scientific test samples. The default historical notebooks remain available as provenance; they are not instructions to repeat the consumed final evaluation.

## Rebuild the archive from the project workspace

With the three original result exports present in a chosen input directory and the exact final-evaluation code ZIP retained under `dist/`:

```bash
python scripts/verify_final_exports.py --inputs-dir /path/to/original/exports --output-dir reports/final_archive_review
python scripts/build_research_archive.py --inputs-dir /path/to/original/exports
```

The builder requires a passing audit for the exact inputs and matching source fingerprint, includes the retained final-evaluation bundle unchanged, and verifies the completed archive before writing its checksum sidecar. It does not rebuild or overwrite the reviewed inference bundle, rerun the model or create a new test ledger.
