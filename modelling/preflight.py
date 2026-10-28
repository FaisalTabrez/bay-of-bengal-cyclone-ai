"""Record the actual runtime without presuming GPU execution or installing packages."""
import argparse
import json
import sys

import numpy
import pandas
import pyarrow
import sklearn
import torch

from .experiment_io import runtime_info, seed_everything, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/environment.json")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        raise RuntimeError("Use Python 3.12+ for the checked release")
    torch.set_num_threads(2)
    seed_everything(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = dict(runtime_info(device), core_imports_passed=True, cuda_available=torch.cuda.is_available(),
                  device_training_verified=False,
                  next_gate="Run the acceptance training and checkpoint-resume cells on this device")
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
