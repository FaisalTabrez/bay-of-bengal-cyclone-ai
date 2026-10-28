"""Create/verify the Colab transfer bundle and export completed run artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .experiment_io import atomic_write
from .paths import BUNDLE_NAME, CODE_VERSION, ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_files(root):
    root = Path(root)
    names = ["README.md", "pyproject.toml", "NOTICE.md", "CITATION.cff",
             "data/packages/bob_cyclone_training_v1.zip", "data/packages/bob_cyclone_training_v2.zip",
             "data/packages/bob_cyclone_v1_acceptance_results.zip",
             "notebooks/bob_cyclone_modelling_v2_colab.ipynb"]
    files = [root / name for name in names]
    files += [p for p in (root / "modelling").rglob("*") if p.is_file()
              and not any(part in {"__pycache__", "runs", "reports"} for part in p.relative_to(root / "modelling").parts)
              and p.suffix in {".py", ".json", ".txt", ".md"}]
    for folder, suffixes in [("tests", {".py"}), ("requirements", {".txt"}),
                             ("docs", {".md"}), ("data", {".md", ".json"}), ("scripts", {".py"})]:
        files += [p for p in (root / folder).rglob("*") if p.is_file() and p.suffix in suffixes
                  and "__pycache__" not in p.parts]
    if not all(p.exists() for p in files):
        raise FileNotFoundError("A required release source is missing")
    return sorted(set(files))


def build_bundle(output, root=ROOT):
    root, output = Path(root), Path(output)
    files = release_files(root)
    if output.resolve() in {p.resolve() for p in files}:
        raise ValueError("Bundle output cannot overwrite a source or immutable data archive")
    manifest = {"release": CODE_VERSION, "purpose": "deterministic_training_and_acceptance",
                "test_policy": "frozen_test_decoder_denied",
                "files": {p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in files}}
    def write(path):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in files:
                archive.write(p, p.relative_to(root).as_posix())
            archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))
    atomic_write(output, write)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    atomic_write(sidecar, lambda p: p.write_text(f"{sha(output)}  {output.name}\n", encoding="utf-8"))
    return {"bundle": str(output), "sha256": sha(output), "files": len(files)}


def verify_bundle(root):
    root = Path(root)
    manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Bundle path escaped the workspace")
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError(f"Bundle integrity failure: {name}")
    return {"status": "bundle_verified", "files": len(manifest["files"])}


def export_runs(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output.is_relative_to(source):
        raise ValueError("Export path must be outside the existing run directory")
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix != ".tmp")
    if not files:
        raise ValueError("No run artifacts found")
    receipt = {p.relative_to(source).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in files}
    def write(path):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in files:
                archive.write(p, p.relative_to(source).as_posix())
            archive.writestr("RUN_FILE_MANIFEST.json", json.dumps(receipt, indent=2))
    atomic_write(output, write)
    return {"archive": str(output), "sha256": sha(output), "files": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--output", default=str(ROOT / "dist" / BUNDLE_NAME))
    verify = sub.add_parser("verify")
    verify.add_argument("--root", default=".")
    export = sub.add_parser("export")
    export.add_argument("--source", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_bundle(args.output) if args.command == "bundle" else verify_bundle(args.root) if args.command == "verify" else export_runs(args.source, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
