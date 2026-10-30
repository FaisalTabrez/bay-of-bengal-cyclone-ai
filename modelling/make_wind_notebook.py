"""Generate the portable follow-up notebook without saved outputs."""
from pathlib import Path
import nbformat as nbf
from .paths import NOTEBOOKS_DIR


def build(path):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# Bay of Bengal: wind audit and bounded follow-up

## Goal

Audit completed v0.3 results, evaluate an explicit wind-persistence candidate,
and optionally run one new three-seed MLP experiment with absolute wind loss.
Selection storms have already been inspected; these are disclosed follow-ups.
Calibration and final-test evaluation remain pending.

## Setup

Upload `bob_cyclone_wind_followup_code.zip`, then your original results ZIP.
The old v0.3.0 code bundle does not contain this runner. Start with acceptance.
Full research uses a separate directory and seeds 17/29/43.
'''), code('''from pathlib import Path
import os
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
WORKDIR = Path("/content/bob_cyclone_wind_followup") if IN_COLAB else Path.cwd()
RUN_FULL_EXPERIMENTS = False
DEVICE = "auto"
PERSIST_TO_DRIVE = IN_COLAB
ALLOW_RUNTIME_CHANGE = False
RUN_ID = "wind_mae_followup_v1"
RESULTS_PATH = WORKDIR / "original_results.zip"
RUN_ROOT = WORKDIR / "runs" / RUN_ID
'''), md('''### 1. Verify source and dependencies

Source files are verified before importing project code. Result model binaries
are hashed, never loaded. Dependency changes may require restarting the runtime.
'''), code('''import hashlib, io, json, subprocess, sys, zipfile
import importlib.metadata as metadata
if not (WORKDIR / "modelling/wind_followup.py").exists():
    if not IN_COLAB:
        raise FileNotFoundError("Start from an extracted follow-up code bundle")
    from google.colab import files
    print("Upload the SOURCE ZIP: bob_cyclone_wind_followup_code.zip")
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Upload exactly one follow-up source ZIP")
    with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
        if not {'BUNDLE_MANIFEST.json', 'modelling/wind_followup.py'}.issubset(archive.namelist()):
            raise ValueError("This is not the follow-up SOURCE bundle. Choose bob_cyclone_wind_followup_code.zip")
        for name in archive.namelist():
            if not (WORKDIR / name).resolve().is_relative_to(WORKDIR.resolve()):
                raise ValueError("Unsafe source archive member")
        archive.extractall(WORKDIR)
manifest_path = WORKDIR / "BUNDLE_MANIFEST.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        member = (WORKDIR / name).resolve()
        if not member.is_relative_to(WORKDIR.resolve()):
            raise ValueError("Unsafe manifest member")
        payload = member.read_bytes()
        if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
            raise ValueError("Source integrity failure: " + name)
    print("Verified source files:", len(manifest["files"]))
elif not (WORKDIR / ".git").exists():
    raise RuntimeError("Use a verified bundle or repository checkout")
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR))
import tomllib
from packaging.requirements import Requirement
missing = []
for line in tomllib.loads((WORKDIR / "pyproject.toml").read_text())["project"]["dependencies"]:
    req = Requirement(line)
    try:
        compatible = req.specifier.contains(metadata.version(req.name), prereleases=True)
    except metadata.PackageNotFoundError:
        compatible = False
    if not compatible:
        missing.append(line)
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
print("Dependencies checked")
'''), md('''### 2. Preserve evidence and supply original results

Drive persistence is optional. Resume restores the entire run before opening the
audit. A source/configuration change requires a new run ID. Keep the original ZIP.
The second upload must be `deterministic_v0_3_results (1).zip`, the completed
research export. An incorrect cached ZIP is rejected and can be replaced here.
'''), code('''import shutil
MIRROR = None
if PERSIST_TO_DRIVE:
    if not IN_COLAB:
        raise RuntimeError("Drive persistence is available in Colab")
    from google.colab import drive
    drive.mount("/content/drive")
    MIRROR = Path("/content/drive/MyDrive/BayOfBengalCycloneAI/runs") / RUN_ID
    if MIRROR.exists() and not RUN_ROOT.exists():
        shutil.copytree(MIRROR, RUN_ROOT)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
from modelling.wind_audit import ResultsArchive, save_original_results
results_reader = None
if RESULTS_PATH.exists():
    try:
        results_reader = ResultsArchive(RESULTS_PATH).require_research()
    except (ValueError, zipfile.BadZipFile) as exc:
        print("Cached results ZIP rejected:", str(exc))
if results_reader is None:
    if not IN_COLAB:
        raise FileNotFoundError("Set RESULTS_PATH to your completed v0.3 results ZIP")
    from google.colab import files
    print("Upload ORIGINAL RESULTS: deterministic_v0_3_results (1).zip")
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Upload exactly one original results ZIP")
    results_reader = save_original_results(next(iter(uploaded.values())), RESULTS_PATH)
print("Original results verified; SHA-256:", results_reader.sha256)
'''), md('''## Steps

### 3. Audit and declare the comparison

This verifies all file hashes and binds predictions to immutable selection rows.
The protocol is written before any new training.
'''), code('''from modelling.wind_audit import audit
from modelling.wind_followup import verify_protocol
from modelling.experiment_io import mirror_run
AUDIT_DIR = RUN_ROOT / "audit"
if not AUDIT_DIR.exists():
    audit(RESULTS_PATH, AUDIT_DIR)
protocol, report = verify_protocol(AUDIT_DIR)
if hashlib.sha256(RESULTS_PATH.read_bytes()).hexdigest() != report["original_export_sha256"]:
    raise ValueError("Original results archive changed")
if MIRROR:
    mirror_run(AUDIT_DIR, MIRROR / "audit")
print("Verified result files:", report["verified_files"])
print("Original selected family:", report["selected_family_by_original_rule"])
'''), md('''### 4. Inspect the wind trade-off

Paired labels from 2011–2014 selection storms. Future wind defines these
retrospective diagnostic groups; it is never an input to the forecast rule.
'''), code('''import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
rows = []
for band in ["steady", "strengthening", "weakening"]:
    g = report["wind_groups"]["bay_mlp"][band]
    rows.append({"Outcome band": band, "Paired labels": g["paired_labels"],
                 "MLP MAE (kt)": g["mae_paired_kts"],
                 "Persistence MAE (kt)": g["persistence_mae_paired_kts"]})
table = pd.DataFrame(rows)
display(table.round(3))
plt.rcParams.update({"font.size": 11, "figure.figsize": (8, 4)})
ax = table.plot.bar(x="Outcome band", y=["MLP MAE (kt)", "Persistence MAE (kt)"],
                    color=["#2266aa", "#c05a00"], rot=0, width=.75)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=10)
ax.set(title="Six-hour wind error: selection 2011–2014", ylabel="Paired MAE (kt)", xlabel="")
ax.set_ylim(0, max(table["MLP MAE (kt)"].max(), table["Persistence MAE (kt)"].max())*1.22)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(title=None)
plt.tight_layout()
plt.savefig(RUN_ROOT / "wind_outcome_diagnostic.png", dpi=150)
plt.show()
print("Hybrid ensemble composite:", round(report["hybrid"]["full"]["selection_composite"], 6))
print("Persistence wind is an explicit rule, not learned wind improvement.")
'''), md('''### 5. Run acceptance, then optionally the declared research experiment

Acceptance runs seed 17 for two epochs and cannot select a research candidate.
Full research runs the fixed three seeds; only wind loss changes. Epoch snapshots
are mirrored when enabled. Resume keeps the same source/configuration/runtime.
'''), code('''import contextlib
from modelling.wind_followup import run
TRAIN_DIR = RUN_ROOT / ("research" if RUN_FULL_EXPERIMENTS else "acceptance")
log_path = RUN_ROOT / ("research.log" if RUN_FULL_EXPERIMENTS else "acceptance.log")
with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
    decision = run(AUDIT_DIR, TRAIN_DIR, full=RUN_FULL_EXPERIMENTS, device=DEVICE,
                   resume=TRAIN_DIR.exists(), mirror_dir=MIRROR / TRAIN_DIR.name if MIRROR else None,
                   allow_runtime_change=ALLOW_RUNTIME_CHANGE)
print("Run status:", decision["status"])
if RUN_FULL_EXPERIMENTS:
    print("Selected candidate:", decision["selected_candidate"])
else:
    print("Acceptance only; no research candidate selected.")
'''), md('''## Checks and next steps

The export includes audit, protocol, diagnostics, checkpoints and decision receipt.
Full selection still reuses selection storms. A deterministic decision does not
complete probabilistic design, calibration or final testing. Read
`docs/wind-followup.md` before the next scientific stage.
'''), code('''from modelling.artifacts import export_runs
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(nbf.writes(notebook), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    build(NOTEBOOKS_DIR/'bob_cyclone_wind_followup_colab.ipynb')
