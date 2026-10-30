"""Verify selected deterministic artifacts and freeze their inference contract."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch

from .data import Normalizer, TrainingPackage, validation_roles
from .experiment_io import atomic_write, digest, source_hashes, write_json
from .geometry import log_map
from .metrics import FORECASTS
from .model import ResidualCodec, ResidualMLP
from .wind_audit import ResultsArchive, validate_records

CONFIG_PATH = Path(__file__).parent/'configs/uncertainty_v1.json'


def read_config(acceptance=False):
    config = json.loads(CONFIG_PATH.read_text())
    if acceptance:
        config.update(maximum_epochs=2, patience=2, energy_score_draws=64)
    config['purpose'] = 'acceptance_not_research' if acceptance else 'uncertainty_research'
    return config


def role_data(package, role):
    if role not in {'selection', 'calibration'}:
        raise PermissionError('Only selection or explicit calibration roles are available; test decoding is denied')
    validation = package.load('bob_validation')
    roles = validation[['SID', 'ISO_TIME']].copy()
    roles['role'] = validation_roles(validation)
    frame = validation.loc[roles.role.eq(role)].reset_index(drop=True)
    return frame, roles


def file_receipt(path):
    p = Path(path)
    return {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def verify_files(root, receipt):
    root = Path(root).resolve()
    for name, expected in receipt.items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or file_receipt(p) != expected:
            raise ValueError(f'Frozen artifact integrity failure: {name}')


def selected_results(path):
    """Distinguish the completed follow-up export from source/acceptance ZIPs."""
    message = ('Upload wind_mae_followup_v1_results (1).zip from the completed three-seed '
               'research run, rather than a code bundle, original deterministic export or acceptance-only ZIP.')
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            required = {'RUN_FILE_MANIFEST.json', 'research/candidate_decision.json',
                        'audit/audit.json', 'audit/followup_protocol.json', 'research/followup_protocol.json',
                        'research/declaration.json'}
            required |= {f'research/bay_mlp/seed_{s}/{n}' for s in [17, 29, 43]
                         for n in ['best.pt', 'summary.json', 'run_manifest.json', 'selection_predictions.csv', 'validation_roles.csv']}
            if not required.issubset(names):
                raise ValueError(message)
    except zipfile.BadZipFile as exc:
        raise ValueError(message) from exc
    archive = ResultsArchive(path)
    decision = archive.read('research/candidate_decision.json')
    if decision.get('status') != 'bounded_deterministic_selection_complete' or decision.get('selected_candidate') != 'bay_mlp_wind_absolute_loss':
        raise ValueError(message)
    return archive


def save_selected_results(payload, path):
    def write(destination):
        destination.write_bytes(payload)
        selected_results(destination)
    atomic_write(path, write)
    return selected_results(path)


class FrozenMean:
    """Only the validated three-seed Bay MLP ensemble; parameters never optimized."""
    def __init__(self, directory, device='cpu'):
        self.directory, self.device = Path(directory), torch.device(device)
        self.record = json.loads((self.directory/'mean_freeze.json').read_text())
        verify_files(self.directory, self.record['files'])
        self.features = self.record['feature_columns']
        self.norm = Normalizer(self.features, json.loads((self.directory/'normalization.json').read_text()))
        self.codec = ResidualCodec.from_dict(json.loads((self.directory/'codec.json').read_text()))
        self.models = []
        for seed in self.record['seeds']:
            saved = torch.load(self.directory/f'seed_{seed}.pt', map_location='cpu', weights_only=True)
            config = saved['context']['config']
            model = ResidualMLP(input_size=78, widths=config['mlp_widths'], dropout=config['dropout'])
            model.load_state_dict(saved['model_state'], strict=True)
            model.to(self.device).eval().requires_grad_(False)
            self.models.append(model)

    def scaled(self, frame, batch_size=512):
        x = self.norm.transform(frame)
        predictions = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                tensor = torch.from_numpy(x[start:start+batch_size]).to(self.device)
                lengths = torch.ones(len(tensor), dtype=torch.long)
                predictions.append(torch.stack([model(tensor[:, None, :], lengths) for model in self.models]).mean(0).cpu().numpy())
        return np.concatenate(predictions)

    def forecast(self, frame):
        return self.codec.decode(frame, self.scaled(frame))

    def native_mean(self, frame, scaled):
        physical = scaled*self.codec.scales+self.codec.anchors(frame)
        return physical  # east/north km in common current-centered tangent plane, wind kt, pressure hPa.


def prepare_mean(results, directory, package, selection, roles, config):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError('Mean freeze already exists; resume the same experiment instead')
    archive = selected_results(results)
    decision = archive.read('research/candidate_decision.json')
    audit = archive.read('audit/audit.json')
    protocol = archive.read('audit/followup_protocol.json')
    if (decision['status'] != 'bounded_deterministic_selection_complete'
            or decision['selected_candidate'] != config['mean_candidate']
            or decision['seeds'] != config['mean_seeds']
            or min(decision['candidate_mean_seed_scores'], key=decision['candidate_mean_seed_scores'].get) != decision['selected_candidate']
            or decision['protocol_sha256'] != digest(protocol) or protocol['audit_sha256'] != digest(audit)
            or archive.read('research/followup_protocol.json') != protocol
            or archive.read('research/declaration.json')['protocol_sha256'] != digest(protocol)):
        raise ValueError('Upload the completed wind follow-up research export selecting the absolute-loss MLP')
    if decision['calibration_labels_used'] or decision['test_parquet_decoded']:
        raise ValueError('Deterministic source violated reserved-role isolation')
    package_hash = hashlib.sha256(package.path.read_bytes()).hexdigest()
    if decision['package_sha256'] != package_hash:
        raise ValueError('Selected model dataset differs')
    # New freezes use canonical LF. Legacy hashes remain tied to original bytes.
    role_hash = hashlib.sha256(roles.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    payloads, states, norm, codec = {}, [], None, None
    with zipfile.ZipFile(results) as source:
        for seed in config['mean_seeds']:
            prefix = f'research/bay_mlp/seed_{seed}/'
            summary, manifest = archive.read(prefix+'summary.json'), archive.read(prefix+'run_manifest.json')
            payload = source.read(prefix+'best.pt')
            checkpoint_hash = hashlib.sha256(payload).hexdigest()
            if checkpoint_hash != decision['checkpoint_hashes'][str(seed)]['sha256'] or checkpoint_hash != summary['best_checkpoint_sha256']:
                raise ValueError('Selected checkpoint hash differs')
            if summary['run_purpose'] != 'research_training' or summary['status'] != 'completed' or summary['checkpoint_reload_max_abs_difference'] != 0:
                raise ValueError('Unfinished deterministic source')
            saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
            context = saved['context']
            archived_role_bytes = source.read(prefix+'validation_roles.csv')
            archived_roles = archive.read(prefix+'validation_roles.csv')
            if (archived_roles.SID.tolist() != roles.SID.tolist()
                    or archived_roles.role.tolist() != roles.role.tolist()
                    or pd.to_datetime(archived_roles.ISO_TIME, utc=True).tolist() != roles.ISO_TIME.tolist()):
                raise ValueError('Archived whole-storm roles differ from the immutable package')
            if (context != {k:v for k,v in manifest.items() if k != 'signature'}
                    or saved['signature'] != manifest['signature'] or digest(context) != saved['signature']
                    or context['source_sha256'] != decision['source_sha256']
                    or context['package_sha256'] != package_hash
                    or context['validation_roles_sha256'] != hashlib.sha256(archived_role_bytes).hexdigest()
                    or context['feature_columns'] != package.features
                    or context['model_input_columns'] != package.features+[f'{c}_MISSING' for c in package.features]
                    or context['config'].get('wind_loss') != 'absolute'
                    or saved['normalizer_stage'] != 'bob'):
                raise ValueError('Selected checkpoint contract/provenance differs')
            if saved['input_normalization'] != package.normalizer('bob').statistics:
                raise ValueError('Selected preprocessing differs from train-only statistics')
            if norm is None:
                norm, codec = saved['input_normalization'], saved['output_codec']
            if norm != saved['input_normalization'] or codec != saved['output_codec']:
                raise ValueError('Ensemble members use different codecs or normalizers')
            reference = archive.read(prefix+'selection_predictions.csv')
            validate_records(reference, selection)
            model = ResidualMLP(widths=context['config']['mlp_widths'], dropout=context['config']['dropout']).eval()
            model.load_state_dict(saved['model_state'])
            x = torch.from_numpy(Normalizer(package.features, norm).transform(selection))
            with torch.no_grad():
                raw = model(x[:, None, :], torch.ones(len(x), dtype=torch.long)).numpy()
            reproduced = ResidualCodec.from_dict(codec).decode(selection, raw)
            differences = np.abs(reproduced-reference[FORECASTS].to_numpy(float)).max(0)
            if (differences > [1e-5, 1e-5, 1e-3, 1e-3]).any():
                raise ValueError('Checkpoint does not reproduce archived deterministic forecasts within declared cross-runtime tolerances')
            states.append({'seed': seed, 'checkpoint_sha256': checkpoint_hash,
                           'max_reproduction_difference_lat_lon_wind_pressure': differences.tolist(),
                           'source_runtime': context['runtime']})
            payloads[f'seed_{seed}.pt'] = payload
    directory.mkdir(parents=True)
    for name, payload in payloads.items():
        atomic_write(directory/name, lambda p, b=payload: p.write_bytes(b))
    write_json(directory/'normalization.json', norm)
    write_json(directory/'codec.json', codec)
    roles.to_csv(directory/'validation_roles.csv', index=False, lineterminator='\n')
    write_json(directory/'deterministic_decision.json', decision)
    record = {'format_version': 1, 'status': 'deterministic_mean_frozen', 'seeds': config['mean_seeds'],
              'feature_columns': package.features, 'model_input_columns': package.features+[f'{c}_MISSING' for c in package.features],
              'package_sha256': package_hash, 'original_export_sha256': archive.sha256,
              'validation_roles_sha256': role_hash, 'source_sha256': source_hashes(),
              'ensemble_rule': 'mean_common_current_centered_displacements_and_native_intensity',
              'mean_parameters_frozen': True, 'member_checks': states,
              'files': {p.name: file_receipt(p) for p in sorted(directory.iterdir()) if p.is_file()},
              'calibration_labels_used': False, 'test_parquet_decoded': False}
    write_json(directory/'mean_freeze.json', record)
    return FrozenMean(directory)


def native_truth(frame):
    track = log_map(frame.LAT.to_numpy(), frame.LON.to_numpy(), frame.TARGET_LAT_6H.to_numpy(), frame.TARGET_LON_6H.to_numpy())
    return np.column_stack([track, frame[['TARGET_WIND_KTS_6H', 'TARGET_PRESSURE_HPA_6H']].to_numpy(float)])
