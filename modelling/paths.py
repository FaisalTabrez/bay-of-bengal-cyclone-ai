"""Canonical repository paths; also valid inside a verified Colab bundle."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "packages"
DOCS_DIR = ROOT / "docs"
NOTEBOOKS_DIR = ROOT / "notebooks"
REPORTS_DIR = ROOT / "reports"
RUNS_DIR = ROOT / "runs"
DEFAULT_PACKAGE = DATA_DIR / "bob_cyclone_training_v2.zip"
DEFAULT_ACCEPTANCE = DATA_DIR / "bob_cyclone_v1_acceptance_results.zip"
CODE_VERSION = "0.3.0"
BUNDLE_NAME = "bob_cyclone_colab_bundle_v0_3.zip"
