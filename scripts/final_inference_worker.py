"""Run only the verified original inference implementation on target-free inputs.

This process never reads the dataset ZIP, computes metrics or fits anything.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

request = json.loads(Path(sys.argv[1]).read_text())
runtime_root = Path(request['runtime_root']).resolve()
sys.path.insert(0, str(runtime_root))
os.environ['OMP_NUM_THREADS'] = '2'

import numpy as np
import pandas as pd
import torch
from modelling.experiment_io import digest, runtime_info, seed_everything, source_hashes
from modelling.uncertainty import predict_frozen, verify_frozen

if not Path(sys.modules['modelling'].__file__).resolve().is_relative_to(runtime_root):
    raise RuntimeError('Frozen inference runtime was not isolated')
if source_hashes() != request['expected_source_hashes']:
    raise ValueError('Original inference source hash mismatch')
torch.set_num_threads(2)
seed_everything(71)
frozen = verify_frozen(request['frozen_directory'])
if digest(frozen) != request['expected_frozen_digest']:
    raise ValueError('Original model freeze differs')
input_path = Path(request['inputs'])
frame = pd.read_parquet(input_path)
for column in ['TARGET_LAT_6H', 'TARGET_LON_6H', 'TARGET_WIND_KTS_6H', 'TARGET_PRESSURE_HPA_6H']:
    if column in frame:
        raise ValueError('Frozen inference worker must never receive future target labels')
result = predict_frozen(frame, request['frozen_directory'], calibrated=True, device='cpu')
calibration = json.loads((Path(request['frozen_directory'])/'calibration/calibration.json').read_text())
factors = np.asarray(calibration['scale_factors_track_wind_pressure'], dtype=float)[[0, 0, 1, 2]]
target = Path(request['output'])
temporary = target.with_suffix('.tmp')
with temporary.open('wb') as stream:
    np.savez_compressed(stream, point=result['point_forecast'], mean=result['mean_tangent_native'],
                        std_calibrated=result['std_tangent_native'],
                        std_uncalibrated=result['std_tangent_native']/factors,
                        rho=result['track_correlation'])
os.replace(temporary, target)
metadata = {'input_rows': len(frame), 'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
            'runtime': runtime_info(torch.device('cpu')), 'frozen_inference_source_digest': digest(source_hashes()),
            'target_columns_received': False, 'training_performed': False, 'calibration_refit': False}
Path(request['metadata']).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
