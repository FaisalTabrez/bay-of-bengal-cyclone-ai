"""Generate the final-evaluation handoff; real test access defaults to disabled."""
from copy import deepcopy
from pathlib import Path
import tempfile

import nbformat as nbf

from .make_wind_notebook import build as build_wind
from .paths import NOTEBOOKS_DIR


def build(path):
    with tempfile.TemporaryDirectory() as tmp:
        template = Path(tmp)/'template.ipynb'
        build_wind(template)
        original = nbf.read(template, as_version=4)
    source = deepcopy(original.cells[3])
    source.source = source.source.replace('bob_cyclone_wind_followup_code.zip', 'bob_cyclone_final_evaluation_code.zip').replace('modelling/wind_followup.py', 'modelling/final_evaluation.py').replace('follow-up source', 'final-evaluation source').replace('follow-up SOURCE', 'final-evaluation SOURCE')
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# Bay of Bengal: reviewed one-time final evaluation

## Goal

Evaluate the frozen calibrated predictor once on the independent 2018–2025 test
partition. Model weights, spread factors, input transformations and scoring rules
are fixed. The original inference source runs in an isolated, verified worker.

## Setup

Upload `bob_cyclone_final_evaluation_code.zip`, then the calibrated
`uncertainty_v1_results (3).zip`. This notebook initially performs software checks
and prepares a plan using selection inputs only. **The real test stays sealed**
until you review that plan, enter its exact digest and enable `RUN_FINAL_TEST`.
In Colab, Drive persistence is required for the real attempt's global ledger.
'''), code('''from pathlib import Path
import os
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
WORKDIR = Path("/content/bob_cyclone_final_evaluation") if IN_COLAB else Path.cwd()
RUN_FINAL_TEST = False
REVIEWED_PLAN_SHA256 = ""
PERSIST_TO_DRIVE = IN_COLAB
RUN_ID = "final_evaluation_v1"
RESULTS_PATH = WORKDIR / "calibrated_uncertainty_results.zip"
RUN_ROOT = WORKDIR / "runs" / RUN_ID
PREPARED_DIR = RUN_ROOT / "prepared"
RESULT_DIR = RUN_ROOT / "execution"
'''), md('''### 1. Verify the final-evaluation source

The source bundle includes an immutable snapshot of the original inference code.
The worker checks that exact source map and the original model/calibration hashes;
the training and calibration loaders continue to deny the test partition.
'''), source, md('''### 2. Restore persistence and validate the calibrated export

At this second prompt choose `uncertainty_v1_results (3).zip`. Earlier uncalibrated
or acceptance ZIPs are rejected. The ledger lives outside the run ID, so changing
the run folder cannot silently reset the same dataset holdout.
'''), code('''import shutil
from modelling.final_evaluation import calibrated_results, ledger_key, protocol_config
from modelling.experiment_io import atomic_write
MIRROR = None
if PERSIST_TO_DRIVE:
    if not IN_COLAB:
        raise RuntimeError("Drive persistence is available in Colab; local CLI uses a persistent filesystem ledger")
    from google.colab import drive
    drive.mount("/content/drive")
    persistent_root = Path("/content/drive/MyDrive/BayOfBengalCycloneAI")
    MIRROR = persistent_root / "runs" / RUN_ID
    LEDGER_DIR = persistent_root / "final_test_ledger"
    if MIRROR.exists() and not RUN_ROOT.exists():
        shutil.copytree(MIRROR, RUN_ROOT)
else:
    LEDGER_DIR = WORKDIR / "runs/final_test_ledger"
reader = None
if RESULTS_PATH.exists():
    try:
        reader, frozen, calibration = calibrated_results(RESULTS_PATH)
    except (ValueError, zipfile.BadZipFile) as exc:
        print("Cached calibrated export rejected:", str(exc))
if reader is None:
    if not IN_COLAB:
        raise FileNotFoundError("Set RESULTS_PATH to uncertainty_v1_results (3).zip")
    from google.colab import files
    print("Upload CALIBRATED RESEARCH RESULTS: uncertainty_v1_results (3).zip")
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Select exactly one calibrated results ZIP")
    payload = next(iter(uploaded.values()))
    def save_checked(path):
        path.write_bytes(payload)
        calibrated_results(path)
    atomic_write(RESULTS_PATH, save_checked)
    reader, frozen, calibration = calibrated_results(RESULTS_PATH)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
print("Calibrated export verified:", reader.sha256)
print("Fixed spread factors:", calibration["scale_factors_track_wind_pressure"])
'''), md('''## Steps

### 3. Check the gate using synthetic fixtures

The software tests use temporary synthetic test ZIPs. They check review/hash
guards, target-free inference inputs, whole-storm scoring, ledger exclusivity and
cache-only recovery. They do not decode the project's real test observations.
'''), code('''test_process = subprocess.run([sys.executable, "-m", "unittest", "tests.test_final_evaluation", "-v"],
                              cwd=WORKDIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(RUN_ROOT / "software_checks.log").write_text(test_process.stdout, encoding="utf-8")
if test_process.returncode:
    raise RuntimeError("Synthetic software checks failed; inspect software_checks.log")
print("Synthetic final-gate checks passed. Real test decoder not exercised.")
'''), md('''### 4. Prepare and review the exact plan

Preparation verifies all frozen artifacts and reproduces archived forecasts using
selection inputs. It never decodes the real test Parquet. Review the model,
calibration, references, scoring rules, runtime and one-time policy below.

The printed digest belongs to this prepared plan and runtime. After review, copy
it into `REVIEWED_PLAN_SHA256` and enable `RUN_FINAL_TEST` in the parameter cell.
The notebook deliberately does not populate your review field automatically.
'''), code('''from IPython.display import Markdown, display
from modelling.final_evaluation import prepare
from modelling.experiment_io import digest, mirror_run
plan = prepare(RESULTS_PATH, PREPARED_DIR)
display(Markdown((PREPARED_DIR / "REVIEW.md").read_text(encoding="utf-8")))
print("REVIEWED_PLAN_SHA256 =", repr(digest(plan)))
print("Test Parquet decoded during preparation:", plan["test_parquet_decoded"])
if MIRROR:
    mirror_run(RUN_ROOT, MIRROR)
'''), md('''### 5. Execute only the reviewed final attempt

The exact digest and a persistent exclusive dataset-level claim are required
before decoding. The frozen worker receives features and identifiers only.
After a technical interruption, resume requires the same plan and verified
cached frame; it cannot decode the original test ZIP again. Preserve the ledger
and output together. Completed reruns read the saved receipt instead of rescoring.
'''), code('''from modelling.final_evaluation import evaluate
metrics = None
if RUN_FINAL_TEST:
    if IN_COLAB and MIRROR is None:
        raise RuntimeError("Enable Drive persistence before the one-time final test")
    if not REVIEWED_PLAN_SHA256 or REVIEWED_PLAN_SHA256 != digest(plan):
        raise PermissionError("Review the plan and enter its exact digest before enabling the test")
    metrics = evaluate(PREPARED_DIR, RESULT_DIR, LEDGER_DIR, REVIEWED_PLAN_SHA256,
                       resume=RESULT_DIR.exists(), mirror_directory=MIRROR / "execution" if MIRROR else None)
    print("Final attempt status:", metrics["status"])
    print("Final observations/cyclones:", metrics["rows"], metrics["storms"])
else:
    existing_claim = LEDGER_DIR / (ledger_key(protocol_config()) + ".json")
    if existing_claim.exists():
        print("No new test access requested. A persistent holdout claim already exists; preserve its original execution.")
    else:
        print("RUN_FINAL_TEST=False. The real test remains sealed in this run.")
'''), md('''### 6. Inspect saved final results

All declared comparisons are reported. Bootstrap intervals resample whole
cyclones, preserving six-hour observations within each sampled storm. Gaussian
shape limitations and retrospective data-availability limits remain part of the
interpretation; final results must not select another model or calibration.
'''), code('''from modelling.final_evaluation import inspect_result
if metrics is None and (RESULT_DIR / "result_receipt.json").exists():
    metrics = inspect_result(RESULT_DIR)
    print("Displaying previously saved final results; no new evaluation was performed.")
if metrics is None:
    print("No final-test metrics generated. Expected 1,018 rows / 65 storms are handoff metadata only.")
else:
    import pandas as pd
    import matplotlib.pyplot as plt
    point = metrics["point_metrics"]["full"]
    display(pd.DataFrame([{"Target": "Track", "Mean error": point["track_mean_km"], "Unit": "km"},
                          {"Target": "Wind (paired)", "Mean error": point["wind_mae_kts_paired"], "Unit": "kt"},
                          {"Target": "Pressure (paired)", "Mean error": point["pressure_mae_hpa_paired"], "Unit": "hPa"}]))
    probability = metrics["probability_metrics"]["frozen_calibrated_gaussian"]["full"]
    levels = plan["protocol"]["coverage_levels"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, target, label in zip(axes, ["track", "wind_kts", "pressure_hpa"], ["Track ellipse", "Wind interval", "Pressure interval"]):
        names = [target + "_coverage_" + str(round(p*100)) for p in levels]
        values = [probability["means"][name] for name in names]
        ax.plot([0, 1], [0, 1], "--", color="#777777", label="Nominal")
        if all(value is not None for value in values):
            intervals = probability["bootstrap"]["intervals"]
            ax.vlines(levels, [intervals[name]["low"] for name in names], [intervals[name]["high"] for name in names], color="#2266aa")
            ax.plot(levels, values, "o-", color="#2266aa", label="Observed final test")
        ax.set(title=label, xlabel="Nominal coverage", xlim=(0, 1), ylim=(0, 1.02))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Observed coverage")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Frozen final-test coverage: 95% whole-cyclone intervals")
    fig.tight_layout()
    fig.savefig(RUN_ROOT / "final_test_coverage.png", dpi=150)
    plt.show()
'''), md('''## Checks and handoff

Keep the calibrated model export, final code bundle, reviewed plan, persistent
ledger and final result ZIP. Preparation exports contain no real test labels.
Actual-attempt exports include the verified test cache to support technical
recovery of this same frozen analysis. Preserve them as research evidence.
'''), code('''from modelling.artifacts import export_runs
from modelling.experiment_io import write_json
claim_path = LEDGER_DIR / (ledger_key(protocol_config()) + ".json")
claim = json.loads(claim_path.read_text()) if claim_path.exists() else None
write_json(RUN_ROOT / "execution_status.json", {
    "run_final_test_requested": RUN_FINAL_TEST,
    "plan_sha256": digest(plan), "ledger_claim": claim,
    "execution_artifacts_present": (RESULT_DIR / "attempt.json").exists()})
if MIRROR:
    mirror_run(RUN_ROOT, MIRROR)
has_attempt = (RESULT_DIR / "attempt.json").exists()
export_path = WORKDIR / ("bob_cyclone_final_results.zip" if has_attempt else "bob_cyclone_final_preflight_results.zip")
receipt = export_runs(RUN_ROOT, export_path)
print("Exported:", export_path.name)
print("SHA-256:", receipt["sha256"])
if IN_COLAB:
    from google.colab import files
    files.download(str(export_path))
''')]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'colab': {'name': Path(path).name}})
    nbf.validate(notebook)
    Path(path).write_text(nbf.writes(notebook), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    build(NOTEBOOKS_DIR/'bob_cyclone_final_evaluation_colab.ipynb')
