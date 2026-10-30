"""Execute the release notebook from a clean bundle; retain local CPU receipts."""
from __future__ import annotations

import hashlib
import asyncio
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from .artifacts import ROOT, build_bundle, verify_bundle
from .experiment_io import source_hashes, write_json
from .paths import REPORTS_DIR, RUNS_DIR


def command(arguments, log):
    result = subprocess.run([sys.executable, *arguments], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Release command failed; inspect {log}\n{result.stdout[-5000:]}")
    return result.stdout


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    started = time.perf_counter()
    reports = REPORTS_DIR
    reports.mkdir(exist_ok=True)
    command(["-m", "modelling.run_design_checks"], reports / "release_design_checks.log")
    output = command(["-m", "unittest", "discover", "-s", "tests", "-v"], reports / "release_unit_tests.log")
    match = re.search(r"Ran (\d+) tests", output)
    if match is None or not output.rstrip().endswith("OK"):
        raise ValueError("Missing successful unit-test receipt")
    bundle = ROOT / "dist/bob_cyclone_colab_bundle_v0_3.zip"
    build_bundle(bundle)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    workspace = RUNS_DIR / ("release_qa_" + stamp)
    workspace.mkdir(parents=True)
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(workspace)
    integrity = verify_bundle(workspace)
    notebook_path = workspace / "notebooks/bob_cyclone_modelling_v2_colab.ipynb"
    source_notebook_hash = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    print("Executing clean-bundle notebook on local CPU", flush=True)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(workspace)}},
                            on_cell_start=lambda **event: print(f"Notebook cell {event['cell_index'] + 1}", flush=True))
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, reports / "colab_notebook_failed_local_cpu.ipynb")
        raise
    errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    if errors:
        raise ValueError("Notebook has cell errors")
    image_outputs = sum("image/png" in o.get("data", {}) for c in notebook.cells
                        if c.cell_type == "code" for o in c.outputs)
    if image_outputs < 1:
        raise ValueError("Executed notebook is missing its rendered diagnostic figure")
    executed_path = reports / "colab_notebook_executed_local_cpu.ipynb"
    nbformat.write(notebook, executed_path)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (reports / "colab_notebook_executed_local_cpu.html").write_text(html, encoding="utf-8")
    run_root = workspace / "runs/deterministic_v0_3"
    plot = run_root / "evidence/selection_progress.png"
    shutil.copyfile(plot, reports / "selection_progress_acceptance.png")
    result_zip = workspace / "deterministic_v0_3_results.zip"
    export_copy = ROOT / "dist/local_cpu_acceptance_results.zip"
    shutil.copyfile(result_zip, export_copy)
    with zipfile.ZipFile(export_copy) as archive:
        manifest = json.loads(archive.read("RUN_FILE_MANIFEST.json"))
        for name, expected in manifest.items():
            payload = archive.read(name)
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError("Result export integrity failure")
    summaries = {p.parent.parent.name: json.loads(p.read_text())
                 for p in (run_root / "acceptance").glob("*/seed_17/summary.json")}
    if len(summaries) != 5 or any(s["status"] != "completed" or s["checkpoint_reload_max_abs_difference"] != 0
                                    or s["calibration_used_for_selection"] or s["test_parquet_decoded"] for s in summaries.values()):
        raise ValueError("Acceptance output completeness/isolation check failed")
    # Resolve modules in the extracted copy, not the original working tree.
    notebook_sources = json.loads((run_root / "acceptance/bay_gru/seed_17/run_manifest.json").read_text())["source_sha256"]
    if notebook_sources != source_hashes():
        raise ValueError("Executed bundle source differs from release source")
    report = {
        "status": "local_cpu_release_acceptance_passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "unit_tests_passed": int(match.group(1)),
        "notebook_code_cells_executed": sum(c.cell_type == "code" for c in notebook.cells),
        "notebook_cell_errors": len(errors), "source_notebook_sha256": source_notebook_hash,
        "notebook_png_outputs": image_outputs,
        "clean_bundle_integrity": integrity, "source_sha256": source_hashes(),
        "regimes_completed": {r: {"purpose": s["run_purpose"], "seed": s["seed"],
                "global_step": s.get("global_step"), "checkpoint_reload_max_abs_difference": s["checkpoint_reload_max_abs_difference"]}
                for r, s in summaries.items()},
        "resume_evidence": "unit test: uninterrupted and interrupted transfer weights bitwise equal across NIO/Bay boundary",
        "result_archive_files_verified": len(manifest), "result_archive_sha256": hashlib.sha256(export_copy.read_bytes()).hexdigest(),
        "executed_notebook": str(executed_path.relative_to(ROOT)), "qa_workspace": str(workspace.relative_to(ROOT)),
        "figure_visual_inspection": "pending", "html_visual_inspection": "pending",
        "gpu_verified": False, "live_colab_upload_drive_verified": False, "fresh_dependency_install_verified": False,
        "full_research_training_run": False, "calibration_used_for_model_selection": False,
        "test_parquet_decoded": False, "test_metrics_computed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(reports / "release_verification.json", report)
    print(json.dumps({"status": report["status"], "tests": report["unit_tests_passed"],
                      "cells_executed": report["notebook_code_cells_executed"], "reports": str(reports)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
