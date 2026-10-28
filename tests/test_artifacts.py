import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from modelling.artifacts import build_bundle, export_runs, verify_bundle


class ArtifactTests(unittest.TestCase):
    def test_bundle_roundtrip_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_bundle(root / "transfer.zip")
            destination = root / "extracted"
            with zipfile.ZipFile(root / "transfer.zip") as archive:
                archive.extractall(destination)
            self.assertEqual(verify_bundle(destination)["status"], "bundle_verified")
            # Parent folder names must not accidentally exclude every source.
            from modelling.experiment_io import source_hashes
            self.assertIn("model.py", source_hashes())
            source = destination / "modelling/model.py"
            source.write_bytes(source.read_bytes() + b"\n# corruption fixture\n")
            with self.assertRaisesRegex(ValueError, "integrity failure"):
                verify_bundle(destination)

    def test_results_export_hashes_and_recursive_path_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "run"
            source.mkdir()
            (source / "metrics.json").write_text('{"fixture":true}', encoding="utf-8")
            receipt = export_runs(source, root / "results.zip")
            self.assertEqual(receipt["files"], 1)
            with zipfile.ZipFile(root / "results.zip") as archive:
                manifest = json.loads(archive.read("RUN_FILE_MANIFEST.json"))
                for name, expected in manifest.items():
                    payload = archive.read(name)
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), expected["sha256"])
                    self.assertEqual(len(payload), expected["bytes"])
            with self.assertRaisesRegex(ValueError, "outside"):
                export_runs(source, source / "recursive.zip")


if __name__ == "__main__":
    unittest.main()
