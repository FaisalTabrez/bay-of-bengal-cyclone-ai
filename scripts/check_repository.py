"""Review publication inputs without printing potential secret values."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", "__pycache__", ".venv", ".venv-model", "runs", "reports", "dist", "output", "local_artifacts", "build"}
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"),
    "provider_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{32,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "credential_url": re.compile(r"https?://[^/\s:]+:[^/@\s]+@"),
    "private_windows_path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\s\\/\"']+", re.I),
}


def review_text(name, text, findings):
    for label, pattern in PATTERNS.items():
        if pattern.search(text):
            findings.append({"path": name, "check": label})


def inspect(root=ROOT, tracked=False):
    root = Path(root)
    if tracked:
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        paths = [root / n for n in names if n]
    else:
        paths = [p for p in root.rglob("*") if p.is_file()
                 and not any(part in IGNORED or part.endswith(".egg-info") for part in p.relative_to(root).parts)]
    findings = []
    archives = 0
    for path in sorted(paths):
        name = path.relative_to(root).as_posix()
        if tracked and any(part in IGNORED for part in path.relative_to(root).parts) and name != "reports/README.md":
            findings.append({"path": name, "check": "generated_or_private_content_tracked"})
        payload = path.read_bytes()
        if path.suffix == ".zip":
            archives += 1
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                # Metadata review only; do not decode Parquet/test observations or checkpoints.
                for entry in archive.namelist():
                    if entry.endswith((".json", ".csv", ".md", ".txt", ".py")):
                        review_text(name + "::" + entry, archive.read(entry).decode("utf-8", errors="replace"), findings)
        elif path.suffix == ".ipynb":
            notebook = json.loads(payload)
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    findings.append({"path": name, "check": f"notebook_output_cell_{index}"})
            review_text(name, payload.decode(), findings)
        else:
            review_text(name, payload.decode("utf-8", errors="replace"), findings)
    checksums = json.loads((root / "data/checksums.json").read_text())
    for name, expected in checksums["archives"].items():
        path = root / "data/packages" / name
        if path.stat().st_size != expected["size_bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            findings.append({"path": str(path.relative_to(root)), "check": "immutable_archive_changed"})
    return {"status": "passed" if not findings else "review_required", "files_reviewed": len(paths),
            "metadata_archives_reviewed": archives, "scope": "pattern-based publication review, not a guarantee of absence of all secrets",
            "test_parquet_decoded": False, "findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true")
    args = parser.parse_args()
    result = inspect(tracked=args.tracked)
    print(json.dumps(result, indent=2))
    if result["findings"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
