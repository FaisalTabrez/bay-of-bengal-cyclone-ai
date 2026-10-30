# Generated verification evidence

Run `python -m modelling.verify_release` from the repository root to generate contract/test logs, an executed local-CPU notebook, HTML preview, diagnostic figure and verification JSON here. Generated outputs are intentionally ignored by Git because they may contain machine-specific paths and acceptance predictions.

GitHub Actions uploads verification evidence as workflow artifacts. Acceptance results demonstrate implementation execution, not research accuracy. The later supplied Colab research and final-evaluation exports are separately verified. See [the final results](../docs/final-results.md) and [archive guide](../docs/reproducibility-archive.md). The local `final_archive_review/` directory contains the current export audit, software test log and archive-build receipt; generated evidence remains outside Git.
