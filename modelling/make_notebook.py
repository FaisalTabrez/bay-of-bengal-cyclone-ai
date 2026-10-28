"""Build the portable Colab notebook using nbformat, with no saved research claims."""
from pathlib import Path

import nbformat as nbf
from .paths import NOTEBOOKS_DIR


def build(path):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("""# Bay of Bengal Cyclone AI · deterministic modelling v0.3

This notebook starts with an **acceptance run**, using a deliberately short schedule. Its numbers check execution and artifact handling; they are not research results or evidence that a model beats the baselines.

The default path keeps the frozen test decoder disabled. Only 2011–2014 validation storms select checkpoints; 2015–2017 storms remain reserved for later calibration. Probabilistic heads and final-test evaluation are outside this notebook.

## Goal

Verify the transferred code/data, exercise training and resume, then optionally run the declared deterministic comparison. Open `docs/model-design.md` for task definitions and scientific limits. Inputs are the v2 training ZIP and v1 acceptance ZIP supplied inside the transfer bundle."""),
        md("""## Setup

### 1. Parameters

In Colab, optionally choose a GPU runtime before starting. CPU is supported. Upload **`bob_cyclone_colab_bundle_v0_3.zip`** when prompted. Keep `RUN_FULL_EXPERIMENTS=False` for the first pass. Enable it only when the acceptance cells complete successfully.

The default source is the public GitHub repository pinned to `v0.3.0`. Set `USE_GITHUB_SOURCE=False` to use the offline upload bundle instead.

The exact local dependency lock is a reference. This notebook preserves compatible packages already installed in Colab, including its GPU-enabled PyTorch, and records the actual environment. Set `PERSIST_TO_DRIVE=False` only if you will download artifacts manually."""),
        code("""import importlib.util
from pathlib import Path
import os

try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False
WORKDIR = Path("/content/bob_cyclone_ai") if IN_COLAB else Path.cwd()
RUN_ID = "deterministic_v0_3"  # Change for a deliberately new experiment recipe.
RUN_FULL_EXPERIMENTS = False
PERSIST_TO_DRIVE = IN_COLAB
DEVICE = "auto"  # auto, cpu, cuda
USE_GITHUB_SOURCE = True
GITHUB_URL = "https://github.com/FaisalTabrez/bay-of-bengal-cyclone-ai.git"
GITHUB_REF = "v0.3.0"
ALLOW_RUNTIME_CHANGE = False  # Explicitly record a changed resume environment; no bitwise claim.
FULL_REGIMES = ["bay_gru", "transfer_gru", "bay_mlp", "bay_tree", "pooled_gru"]
RUN_ROOT = WORKDIR / "runs" / RUN_ID
print({"workdir": str(WORKDIR), "full_experiments": RUN_FULL_EXPERIMENTS, "device": DEVICE})"""),
        md("""### 2. Obtain and verify the source

Use the pinned GitHub tag or extract the offline bundle. Versioned data ZIPs remain intact in `data/packages/`. Bundle mode verifies every transferred file; repository mode records the checkout. The next contract check verifies the immutable data hashes without decoding test observations."""),
        code("""import hashlib
import json
import zipfile
import subprocess

WORKDIR.mkdir(parents=True, exist_ok=True)
if IN_COLAB and USE_GITHUB_SOURCE and not (WORKDIR / "modelling/train.py").exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", GITHUB_REF, GITHUB_URL, str(WORKDIR)], check=True)
if not (WORKDIR / "BUNDLE_MANIFEST.json").exists() and not (WORKDIR / "modelling/train.py").exists():
    if not IN_COLAB:
        raise FileNotFoundError("Execute this notebook from an extracted transfer bundle directory.")
    from google.colab import files
    uploaded = files.upload()
    bundle_names = [n for n in uploaded if n.endswith(".zip")]
    if len(bundle_names) != 1:
        raise ValueError("Upload exactly one outer Colab bundle ZIP.")
    import io
    with zipfile.ZipFile(io.BytesIO(uploaded[bundle_names[0]])) as archive:
        for item in archive.infolist():
            destination = (WORKDIR / item.filename).resolve()
            if not destination.is_relative_to(WORKDIR.resolve()):
                raise ValueError("Unsafe archive member path")
        archive.extractall(WORKDIR)
if (WORKDIR / "BUNDLE_MANIFEST.json").exists():
    manifest = json.loads((WORKDIR / "BUNDLE_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        member = (WORKDIR / name).resolve()
        if not member.is_relative_to(WORKDIR.resolve()):
            raise ValueError("Invalid manifest path")
        payload = member.read_bytes()
        assert len(payload) == expected["bytes"] and hashlib.sha256(payload).hexdigest() == expected["sha256"], name
    print(f"Verified {len(manifest['files'])} transferred files.")
elif (WORKDIR / ".git").exists():
    commit = subprocess.check_output(["git", "-C", str(WORKDIR), "rev-parse", "HEAD"], text=True).strip()
    print("Repository checkout:", commit)
else:
    raise RuntimeError("Use a verified bundle or a Git checkout.")
os.chdir(WORKDIR)
print("Test Parquet has not been decoded.")"""),
        md("""### 3. Check dependencies and preserve results

Only missing or incompatible packages are installed. The notebook invokes the same Python entry points as the CLI in its own kernel, avoiding duplicate framework processes. If a core package changed after being imported, restart the kernel before continuing.

Checkpoints are written locally each epoch, then copied to the persistent mirror. If Colab disconnects, rerun setup to restore that mirror and use the same run ID/configuration. Resume rejects changed source, data, roles or runtime."""),
        code("""import sys
import subprocess
import importlib.metadata as metadata
from packaging.requirements import Requirement

if sys.version_info < (3, 12):
    raise RuntimeError("Select a Python 3.12+ Colab runtime.")
requirements = WORKDIR / "requirements/colab.txt"
needed = []
already_loaded_changes = []
for line in requirements.read_text().splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    requirement = Requirement(line)
    try:
        installed = metadata.version(requirement.name)
        if installed not in requirement.specifier:
            needed.append(str(requirement))
            module_name = {"scikit-learn": "sklearn"}.get(requirement.name, requirement.name)
            if module_name in sys.modules:
                already_loaded_changes.append(module_name)
    except metadata.PackageNotFoundError:
        needed.append(str(requirement))
if needed:
    subprocess.run([sys.executable, "-m", "pip", "install", *needed], check=True)
if already_loaded_changes:
    raise RuntimeError(f"Restart the kernel after upgrading imported libraries: {already_loaded_changes}. Then rerun setup.")
print("Dependency check complete; packages installed:", needed)

MIRROR_ROOT = None
if PERSIST_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    MIRROR_ROOT = Path("/content/drive/MyDrive/BayOfBengalCycloneAI/runs") / RUN_ID
    import shutil
    if MIRROR_ROOT.exists() and not RUN_ROOT.exists():
        shutil.copytree(MIRROR_ROOT, RUN_ROOT)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
print("Persistent mirror:", str(MIRROR_ROOT) if MIRROR_ROOT else "manual download; no Drive mirror")"""),
        md("""## Steps

### 4. Record the runtime and run contract checks

The full process output is retained in log files. The notebook shows a bounded tail. The tests check target masks, cyclone boundaries, padding, normalization transfer, matched baseline denominators and exact CPU resume. Device-specific training is checked in the next cells."""),
        code("""import contextlib
import importlib
import traceback

def run(label, arguments):
    log_dir = RUN_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    arguments = list(map(str, arguments))
    if arguments[0] != "-m" or not (arguments[1].startswith("modelling.") or arguments[1] == "unittest"):
        raise ValueError("Only the release's Python module entry points are supported.")
    previous_argv = sys.argv[:]
    status = 0
    print(f"Starting {label}; full output is saved under {log_dir}", flush=True)
    with (log_dir / f"{label}.log").open("w", encoding="utf-8") as log:
        try:
            sys.argv = [arguments[1], *arguments[2:]]
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                if arguments[1] == "unittest":
                    import unittest
                    result = unittest.main(module=None, argv=["unittest", *arguments[2:]], exit=False)
                    status = 0 if result.result.wasSuccessful() else 1
                else:
                    importlib.import_module(arguments[1]).main()
        except Exception:
            traceback.print_exc(file=log)
            status = 1
        finally:
            sys.argv = previous_argv
    tail = (log_dir / f"{label}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-10:]
    print(f"{label}: exit {status}")
    print("\\n".join(tail))
    if status:
        raise RuntimeError(f"{label} failed; see {log_dir / (label + '.log')}")

EVIDENCE = RUN_ROOT / "evidence"
run("environment", ["-m", "modelling.preflight", "--output", EVIDENCE / "environment.json"])
run("design_contracts", ["-m", "modelling.run_design_checks", "--output", EVIDENCE / "design"])
run("tests", ["-m", "unittest", "tests.test_contract", "tests.test_experiments", "tests.test_artifacts", "-v"])"""),
        md("""### 5. Exercise Bay training, interruption and resume

Acceptance uses seed 17 and two Bay epochs. On the first execution, pause after one epoch and then resume from `latest.pt`. A rerun reuses the completed checkpoint after validating its recipe. No checkpoint is chosen using calibration or test labels."""),
        code("""ACCEPTANCE = RUN_ROOT / "acceptance"
mirror_arguments = ["--mirror-dir", MIRROR_ROOT / "acceptance"] if MIRROR_ROOT else []
runtime_arguments = ["--allow-runtime-change"] if ALLOW_RUNTIME_CHANGE else []
common = ["-m", "modelling.train", "--acceptance", "--device", DEVICE, "--output", ACCEPTANCE] + runtime_arguments
bay_latest = ACCEPTANCE / "bay_gru/seed_17/latest.pt"
if not bay_latest.exists():
    run("bay_pause", common + ["--regimes", "bay_gru", "--stop-after-epochs", "1"] + mirror_arguments)
run("bay_resume", common + ["--regimes", "bay_gru", "--resume"] + mirror_arguments)"""),
        md("""### 6. Exercise transfer and simpler controls

Pause transfer immediately after its single NIO acceptance epoch. Resume must switch to Bay preprocessing exactly once, keep the NIO output codec and use a fresh Bay optimizer. The pooled GRU, snapshot MLP and three-iteration tree are execution controls here, not fitted research baselines."""),
        code("""transfer_latest = ACCEPTANCE / "transfer_gru/seed_17/latest.pt"
if not transfer_latest.exists():
    run("transfer_pause", common + ["--regimes", "transfer_gru", "--stop-after-epochs", "1"] + mirror_arguments)
run("transfer_resume", common + ["--regimes", "transfer_gru", "--resume"] + mirror_arguments)
run("acceptance_controls", common + ["--regimes", "bay_mlp", "bay_tree", "pooled_gru", "--resume"] + mirror_arguments)"""),
        md("""### 7. Optional full deterministic comparison

This cell skips by default. Set `RUN_FULL_EXPERIMENTS=True` in the parameters to run the configured 17/29/43 seeds and full schedules. It resumes existing runs and does not silently overwrite them. This may take materially longer than acceptance, depending on the allocated runtime.

The three seed scores select a family; the ensemble is also reported. Results on selection storms are not final generalization evidence. If claiming a transfer benefit, run the documented `bay_budget` control with each transfer seed's actual global update count. Probabilistic calibration and test evaluation remain later work."""),
        code("""RESEARCH = RUN_ROOT / "research"
if RUN_FULL_EXPERIMENTS:
    full_mirror = ["--mirror-dir", MIRROR_ROOT / "research"] if MIRROR_ROOT else []
    run("research_comparison", ["-m", "modelling.train", "--device", DEVICE,
        "--output", RESEARCH, "--regimes", *FULL_REGIMES, "--resume"] + full_mirror + runtime_arguments)
else:
    print("Full experiments skipped. Acceptance results below are execution diagnostics only.")"""),
        md("""## Checks

### 8. Inspect saved artifacts and selection diagnostics

The table reports matched intensity MAEs alongside track error. Each seed also has all-observed-label metrics, sample counts, per-storm records and descriptive whole-storm bootstrap intervals in its JSON/CSV files.

Acceptance comparisons are deliberately undertrained and do not identify a scientifically preferred architecture."""),
        code("""import pandas as pd
from IPython.display import display

ACTIVE = RESEARCH if RUN_FULL_EXPERIMENTS else ACCEPTANCE
rows = []
for summary_path in sorted(ACTIVE.glob("*/ensemble/summary.json")):
    summary = json.loads(summary_path.read_text())
    metrics = summary["selection_metrics"]["full"]
    assert summary["test_parquet_decoded"] is False
    assert summary["calibration_used_for_selection"] is False
    rows.append({"Regime": summary["regime"], "Purpose": summary["run_purpose"],
        "Seeds": len(summary["expected_seeds"]), "Mean seed selection score": summary["family_selection_score_mean_of_seeds"],
        "Ensemble track error (km)": metrics["track_mean_km"],
        "Paired wind MAE (kt)": metrics["wind_mae_kts_paired"],
        "Paired pressure MAE (hPa)": metrics["pressure_mae_hpa_paired"]})
if not rows:
    raise RuntimeError("No complete declared-seed ensemble found.")
display(pd.DataFrame(rows).round(4))
print("Selection population: 493 observations from 25 Bay-affecting cyclones, 2011–2014.")
print("Calibration and frozen test labels have not selected any checkpoint.")"""),
        md("""### 9. Inspect training progress

The score uses the same selection rows and paired intensity denominators across regimes. A score of one matches the hybrid reference of constant-velocity track and persistence intensity. Short-run fluctuations are diagnostic and do not establish skill."""),
        code("""get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
colors = {"bay_gru": "#2563A6", "transfer_gru": "#CB6A20", "bay_mlp": "#707A32", "pooled_gru": "#9C477D"}
styles = {"bay_gru": "-", "transfer_gru": "--", "bay_mlp": "-.", "pooled_gru": ":"}
fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
for regime, color in colors.items():
    histories = sorted((ACTIVE / regime).glob("seed_*/history.csv"))
    for history_path in histories:
        history = pd.read_csv(history_path).dropna(subset=["selection_score"])
        if not history.empty:
            ax.plot(history.global_step, history.selection_score, styles[regime], color=color,
                    marker="o", markersize=4, label=f"{regime} · {history_path.parent.name}")
ax.axhline(1, color="#42464D", linestyle="--", linewidth=1, label="Hybrid reference = 1")
ax.set(xlabel="Optimizer updates", ylabel="Selection composite score · lower is better",
       title="Selection progress" if RUN_FULL_EXPERIMENTS else "Acceptance diagnostics · shortened schedules")
ax.grid(axis="y", alpha=.18)
ax.legend(fontsize=8, loc="best")
EVIDENCE.mkdir(parents=True, exist_ok=True)
fig.savefig(EVIDENCE / "selection_progress.png", dpi=150)
display(fig)
plt.close(fig)"""),
        md("""## Next steps

### 10. Export results

The archive includes checkpoints, serialized preprocessing, manifests, predictions, metrics, histories and logs, with a file-checksum manifest. Keep this result archive together with the original code/data bundle. A saved notebook alone does not preserve the VM's files.

After full deterministic experiments, inspect whether any learned candidate improves reliably on the declared baselines. Then design/run the probabilistic stage, calibrate on its reserved role, freeze the complete experiment and evaluate the test set once. This release intentionally provides no test-evaluation command."""),
        code("""from modelling.artifacts import export_runs
from modelling.experiment_io import mirror_run

if MIRROR_ROOT:
    mirror_run(RUN_ROOT, MIRROR_ROOT)
archive = WORKDIR / f"{RUN_ID}_results.zip"
receipt = export_runs(RUN_ROOT, archive)
print(json.dumps(receipt, indent=2))
if IN_COLAB:
    from google.colab import files
    files.download(str(archive))
else:
    print("Local verification export:", archive)"""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"},
                         "colab": {"name": "bob_cyclone_modelling_v2_colab.ipynb"}}
    nbf.validate(notebook)
    Path(path).write_text(nbf.writes(notebook) + "\n", encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":
    print(build(NOTEBOOKS_DIR / "bob_cyclone_modelling_v2_colab.ipynb"))
