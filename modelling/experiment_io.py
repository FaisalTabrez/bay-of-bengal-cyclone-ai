"""Atomic run artifacts, provenance and safe, epoch-boundary resume state."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import tempfile
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).parent
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".py", ".json", ".txt"}
             and "reports" not in p.relative_to(root).parts and "runs" not in p.relative_to(root).parts}
    for folder in ["tests", "requirements"]:
        for p in sorted((root.parent / folder).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".txt"}:
                files[p.relative_to(root.parent).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    project = root.parent / "pyproject.toml"
    if project.exists():
        files["pyproject.toml"] = hashlib.sha256(project.read_bytes()).hexdigest()
    return files


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(path, lambda p: p.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"))


def save_checkpoint(path, state):
    import torch
    atomic_write(path, lambda p: torch.save(state, p))


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def mirror_run(run_dir, mirror_dir):
    if mirror_dir is None:
        return
    run_dir, mirror_dir = Path(run_dir), Path(mirror_dir)
    if run_dir.resolve() == mirror_dir.resolve() or run_dir.resolve() in mirror_dir.resolve().parents:
        raise ValueError("Mirror must be outside the run directory")
    for source in run_dir.rglob("*"):
        if source.is_file() and source.suffix != ".tmp":
            destination = mirror_dir / source.relative_to(run_dir)
            atomic_write(destination, lambda p, s=source: shutil.copyfile(s, p))


def seed_everything(seed):
    import numpy as np
    import torch
    # Set before the first CUDA operation; required for deterministic CUDA GEMM.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng(generator):
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(),
            int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator, allow_device_change=False):
    import numpy as np
    import torch
    random.setstate(state["python"])
    s = state["numpy"]
    np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() and not allow_device_change:
            raise ValueError("CUDA checkpoint requires a CUDA runtime for resume")
        if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])
        elif not allow_device_change:
            raise ValueError("CUDA device count differs from checkpoint")
    generator.set_state(state["loader"])


def runtime_info(device):
    import torch
    packages = {}
    for name in ["numpy", "pandas", "pyarrow", "torch", "scikit-learn"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "device": str(device), "cuda_runtime": torch.version.cuda, "cuda_device_count": torch.cuda.device_count(),
            "cudnn": torch.backends.cudnn.version(), "threads": torch.get_num_threads(),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
