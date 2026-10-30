"""Generate the uncertainty workflow, reusing the verified source-upload cell."""
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
        wind = nbf.read(template, as_version=4)
    source = deepcopy(wind.cells[3])
    source.source = source.source.replace('bob_cyclone_wind_followup_code.zip', 'bob_cyclone_uncertainty_code.zip').replace('modelling/wind_followup.py', 'modelling/uncertainty.py')
    source.source = source.source.replace('follow-up source', 'uncertainty source').replace('follow-up SOURCE', 'uncertainty SOURCE')
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# Bay of Bengal: uncertainty around the frozen MLP ensemble

## Goal

Keep the selected three-seed point predictor fixed and learn Gaussian spread.
The track distribution is correlated in current-centered east/north kilometres;
wind and pressure have independent marginal Gaussian distributions. This is one
Gaussian around the ensemble mean, not a three-component mixture.

## Setup

First upload `bob_cyclone_uncertainty_code.zip`. At the second prompt choose
**`wind_mae_followup_v1_results (1).zip`**, containing the completed research
comparison and selected checkpoints. Start with acceptance. Full uncertainty
training and reserved-role calibration have separate controls; calibration is
disabled by default. The frozen test loader remains denied.
'''), code('''from pathlib import Path
import os
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
WORKDIR = Path("/content/bob_cyclone_uncertainty") if IN_COLAB else Path.cwd()
RUN_FULL_UNCERTAINTY = False
RUN_CALIBRATION = False
DEVICE = "auto"
PERSIST_TO_DRIVE = IN_COLAB
ALLOW_RUNTIME_CHANGE = False
RUN_ID = "uncertainty_v1"
RESULTS_PATH = WORKDIR / "selected_wind_results.zip"
RUN_ROOT = WORKDIR / "runs" / RUN_ID
'''), md('''### 1. Verify the uncertainty source bundle

The entire source bundle is checked before importing project code. Missing or
incompatible dependencies are installed; restart the kernel if an imported core
library was replaced. Keep the same source/configuration when resuming.
'''), source, md('''### 2. Restore persistence and validate the selected-model upload

The second ZIP must be the completed wind research export. A code, dataset or
acceptance-only ZIP is rejected before it replaces the cache. Drive persistence
is optional. Restore the same run ID/source bundle after a disconnection.
'''), code('''import shutil
from modelling.uncertainty_io import selected_results, save_selected_results
MIRROR = None
if PERSIST_TO_DRIVE:
    if not IN_COLAB:
        raise RuntimeError("Drive persistence is available in Colab")
    from google.colab import drive
    drive.mount("/content/drive")
    MIRROR = Path("/content/drive/MyDrive/BayOfBengalCycloneAI/runs") / RUN_ID
    if MIRROR.exists() and not RUN_ROOT.exists():
        shutil.copytree(MIRROR, RUN_ROOT)
reader = None
if RESULTS_PATH.exists():
    try:
        reader = selected_results(RESULTS_PATH)
    except (ValueError, zipfile.BadZipFile) as exc:
        print("Cached selected-model ZIP rejected:", str(exc))
if reader is None:
    if not IN_COLAB:
        raise FileNotFoundError("Set RESULTS_PATH to the completed wind_mae_followup_v1_results ZIP")
    from google.colab import files
    print("Upload SELECTED MODEL RESULTS: wind_mae_followup_v1_results (1).zip")
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Select exactly one completed wind research ZIP")
    reader = save_selected_results(next(iter(uploaded.values())), RESULTS_PATH)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
print("Selected model export verified; SHA-256:", reader.sha256)
'''), md('''## Steps

### 3. Verify and freeze the means, then train spread only

Selected checkpoint hashes, normalization, feature order, validation roles and
saved forecast reproduction are checked. Mean parameters are excluded from the
optimizer. Spread initialization uses in-sample training errors only; it is not
an out-of-fold uncertainty estimate. Selection NLL chooses the spread epoch.

Acceptance uses two epochs and cannot create a calibratable research freeze.
Full research uses the fixed schedule and writes `research/frozen_model.json`.
'''), code('''import contextlib
from modelling.uncertainty import train
TRAIN_DIR = RUN_ROOT / ("research" if RUN_FULL_UNCERTAINTY else "acceptance")
log_path = RUN_ROOT / ("research.log" if RUN_FULL_UNCERTAINTY else "acceptance.log")
with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
    summary = train(RESULTS_PATH, TRAIN_DIR, full=RUN_FULL_UNCERTAINTY, device=DEVICE,
                    resume=TRAIN_DIR.exists(), mirror_dir=MIRROR / TRAIN_DIR.name if MIRROR else None,
                    allow_runtime_change=ALLOW_RUNTIME_CHANGE)
print("Status:", summary["status"])
print("Mean parameters changed:", summary["mean_parameters_changed"])
print("Spread checkpoint reload difference:", summary["checkpoint_reload_max_abs_difference"])
print("Calibration labels used:", summary["calibration_labels_used"])
'''), md('''### 4. Inspect selection uncertainty diagnostics

These are reused 2011–2014 selection storms, not independent confirmation or
calibration results. CRPS for the two track axes is marginal; the joint spatial
score is the energy score. Track ellipse areas are in the tangent plane, not
latitude/longitude probability densities. Tail masses are reported without clipping.
'''), code('''import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
metrics = json.loads((TRAIN_DIR / "selection_metrics.json").read_text())
levels = summary["config"]["coverage_levels"]
rows = []
for name in ["wind_kts", "pressure_hpa"]:
    m = metrics["marginals"][name]
    rows.append({"Target": name, "Observed labels": m["labels"], "CRPS (native units)": m["crps"],
                 "90% coverage": m["intervals"]["0.9"]["coverage"],
                 "90% mean width": m["intervals"]["0.9"]["mean_width"]})
display(pd.DataFrame(rows).round(4))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
for ax, target in zip(axes, ["track", "wind_kts", "pressure_hpa"]):
    if target == "track":
        observed = [metrics["track_ellipses"][str(p)]["coverage"] for p in levels]
        count = metrics["track_labels"]
    else:
        observed = [metrics["marginals"][target]["intervals"][str(p)]["coverage"] for p in levels]
        count = metrics["marginals"][target]["labels"]
    ax.plot([0, 1], [0, 1], "--", color="#777777", label="Nominal")
    ax.plot(levels, observed, "o-", color="#2266aa", label="Observed selection")
    ax.set(title=f"{target} (n={count})", xlabel="Nominal coverage", xlim=(0, 1), ylim=(0, 1.02))
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Observed coverage")
axes[-1].legend(fontsize=8, loc="lower right")
fig.suptitle("Selection uncertainty diagnostic — " + ("research" if RUN_FULL_UNCERTAINTY else "short acceptance"))
fig.tight_layout()
fig.savefig(RUN_ROOT / "selection_uncertainty.png", dpi=150)
plt.show()
print("Gaussian tail-mass diagnostics:", metrics["gaussian_tail_mass"])
'''), md('''### 5. Optional one-time calibration after the research freeze

Leave `RUN_CALIBRATION=False` until the full uncertainty run and freeze are
reviewed. With the research freeze fixed, the separate command fits three
positive spread factors on the reserved 2015–2017 storms. It changes no means,
architecture or learned weights. It refuses to fit again when its fixed output
directory exists. Diagnostics on that fit set do not prove independent coverage.
'''), code('''if RUN_CALIBRATION:
    if not RUN_FULL_UNCERTAINTY:
        raise ValueError("Calibration requires the completed full uncertainty research freeze")
    from modelling.uncertainty import calibrate, verify_frozen
    frozen = verify_frozen(TRAIN_DIR)
    cal_path = TRAIN_DIR / "calibration/calibration.json"
    if cal_path.exists():
        calibration = json.loads(cal_path.read_text())
        from modelling.experiment_io import digest
        if calibration["frozen_model_sha256"] != digest(frozen):
            raise ValueError("Calibration belongs to a different freeze")
        print("Existing calibration retained; no refit performed.")
    else:
        calibration = calibrate(TRAIN_DIR, DEVICE, MIRROR / TRAIN_DIR.name if MIRROR else None)
    print("Calibration status:", calibration["status"])
else:
    print("Calibration disabled. Reserved labels have not been used by this workflow.")
'''), md('''## Checks and next steps

Keep the original selected-model ZIP, code bundle and exported results together.
First inspect acceptance, then enable full uncertainty training and rerun the
training, diagnostics and export cells. Calibration is a later explicit step.
The workflow has no final-test command or loader override.
'''), code('''from modelling.artifacts import export_runs
from modelling.experiment_io import mirror_run, write_json
write_json(RUN_ROOT / "execution_status.json", {
    "full_uncertainty_requested": RUN_FULL_UNCERTAINTY, "calibration_requested": RUN_CALIBRATION,
    "test_parquet_decoded": False, "source_results_sha256": reader.sha256})
if MIRROR:
    mirror_run(RUN_ROOT, MIRROR)
export_path = WORKDIR / (RUN_ID + "_results.zip")
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
    build(NOTEBOOKS_DIR/'bob_cyclone_uncertainty_colab.ipynb')
