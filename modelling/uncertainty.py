"""Frozen-mean spread training and separately invoked reserved-role calibration.

There is deliberately no final-test command or test-loader override.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import TrainingPackage
from .experiment_io import (capture_rng, digest, load_checkpoint, mirror_run, restore_rng,
                            runtime_info, save_checkpoint, seed_everything, source_hashes, write_json)
from .metrics import FORECASTS, prediction_records, summarize_records
from .paths import DEFAULT_PACKAGE
from .uncertainty_io import (FrozenMean, file_receipt, native_truth, prepare_mean, read_config,
                             role_data, selected_results, verify_files)
from .uncertainty_math import (SpreadMLP, distribution_scores, fit_scale_factors,
                               gaussian_nll, scaled_std)


def choose_device(name):
    device = torch.device('cuda' if name == 'auto' and torch.cuda.is_available() else 'cpu' if name == 'auto' else name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return device


def make_spread(config, scales, initial_std):
    return SpreadMLP(np.asarray(config['native_std_floors'])/scales, initial_std,
                     config['spread_network']['hidden_size'], config['maximum_absolute_track_correlation'])


def predict_spread(model, mean, frame, device):
    values, correlations = [], []
    x = mean.norm.transform(frame)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 512):
            s, r = model(torch.from_numpy(x[start:start+512]).to(device))
            values.append(s.cpu().numpy())
            correlations.append(r.cpu().numpy())
    return np.concatenate(values)*mean.codec.scales, np.concatenate(correlations)


def report_distribution(output, role, mean, frame, std, rho, config, factors=None):
    output = Path(output)
    raw_mean = mean.scaled(frame)
    native_mean = mean.native_mean(frame, raw_mean)
    if factors is not None:
        std = scaled_std(std, factors)
    point = mean.codec.decode(frame, raw_mean)
    scores = distribution_scores(native_truth(frame), native_mean, std, rho, config)
    scores.update(storms=int(frame.SID.nunique()), role=role,
                  calibration_labels_used=role == 'calibration', test_parquet_decoded=False,
                  point_metrics=summarize_records(prediction_records(frame, point)))
    records = frame[['SID', 'ISO_TIME', 'LAT', 'LON']].copy()
    for j, axis in enumerate(['EAST_KM', 'NORTH_KM', 'WIND_KTS', 'PRESSURE_HPA']):
        records['MEAN_'+axis] = native_mean[:, j]
        records['STD_'+axis] = std[:, j]
    for j, col in enumerate(FORECASTS):
        records[col] = point[:, j]
    records['TRACK_CORRELATION'] = rho
    records.to_csv(output/f'{role}_distribution.csv', index=False)
    write_json(output/f'{role}_metrics.json', scores)
    return scores


def verify_frozen(directory):
    directory = Path(directory)
    if not (directory/'frozen_model.json').is_file():
        raise ValueError('A completed full uncertainty research freeze is required; acceptance cannot be calibrated')
    record = json.loads((directory/'frozen_model.json').read_text())
    if record['status'] != 'research_uncertainty_frozen' or record['source_sha256'] != source_hashes():
        raise ValueError('Only a research freeze with unchanged source can be calibrated')
    if record['calibration_labels_used'] or record['test_parquet_decoded']:
        raise ValueError('Freeze contains reserved-role use')
    verify_files(directory, record['files'])
    if record['config'] != read_config(False):
        raise ValueError('Frozen uncertainty configuration differs')
    return record


def predict_frozen(frame, directory, calibrated=False, device='cpu'):
    """Inference on current input features only; target columns are not required."""
    directory = Path(directory)
    frozen = verify_frozen(directory)
    device = choose_device(device)
    mean = FrozenMean(directory/'mean', device)
    saved = load_checkpoint(directory/'best.pt')
    model = make_spread(frozen['config'], mean.codec.scales, saved['initial_std']).to(device)
    model.load_state_dict(saved['model_state'])
    std, rho = predict_spread(model, mean, frame, device)
    if calibrated:
        status = json.loads((directory/'calibration/status.json').read_text())
        path = directory/'calibration/calibration.json'
        calibration = json.loads(path.read_text())
        if (status['status'] != 'completed' or calibration['status'] != 'calibrated_once'
                or status['frozen_model_sha256'] != digest(frozen)
                or calibration['frozen_model_sha256'] != digest(frozen)
                or status['calibration_sha256'] != file_receipt(path)['sha256']):
            raise ValueError('Calibration receipt is incomplete or differs from the frozen model')
        std = scaled_std(std, calibration['scale_factors_track_wind_pressure'])
    raw = mean.scaled(frame)
    return {'point_forecast': mean.codec.decode(frame, raw),
            'mean_tangent_native': mean.native_mean(frame, raw), 'std_tangent_native': std,
            'track_correlation': rho, 'calibration_applied': calibrated}


def train(results, output, full=False, device='auto', resume=False, mirror_dir=None,
          allow_runtime_change=False, stop_after_epochs=None):
    output = Path(output)
    config = read_config(acceptance=not full)
    selected_results(results)  # Reject a wrong upload before creating any output.
    if stop_after_epochs is not None and stop_after_epochs < 1:
        raise ValueError('stop_after_epochs must be positive')
    torch.set_num_threads(2)
    seed_everything(config['seed'])
    device = choose_device(device)
    package = TrainingPackage(DEFAULT_PACKAGE)
    package.verify_checksums()
    selection, roles = role_data(package, 'selection')
    training = package.load('bob_train')
    if set(training.SID) & set(selection.SID):
        raise ValueError('Training and selection storms overlap')
    original_hash = hashlib.sha256(Path(results).read_bytes()).hexdigest()
    if output.exists():
        if not resume or not (output/'latest.pt').exists():
            raise FileExistsError('Use a fresh uncertainty directory or resume the exact saved run')
        mean = FrozenMean(output/'mean', device)
    else:
        output.mkdir(parents=True)
        prepare_mean(results, output/'mean', package, selection, roles, config)
        mean = FrozenMean(output/'mean', device)
    if mean.record['original_export_sha256'] != original_hash or mean.record['source_sha256'] != source_hashes():
        raise ValueError('Selected archive/source changed; preserve the old run and start a fresh run')
    mean.norm.verify_training_statistics(training)
    initial_mean = mean.scaled(training)
    encoded, mask = mean.codec.encode(training)
    error = encoded-initial_mean
    initial_std = np.sqrt(np.nanmean(error.astype(float)**2, axis=0))
    initial_std = np.maximum(initial_std, np.asarray(config['native_std_floors'])/mean.codec.scales*1.1)
    model = make_spread(config, mean.codec.scales, initial_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(config['seed'])
    context = {'config': config, 'source_sha256': source_hashes(), 'runtime': runtime_info(device),
               'mean_freeze_sha256': file_receipt(output/'mean/mean_freeze.json')['sha256'],
               'package_sha256': mean.record['package_sha256'], 'original_export_sha256': original_hash}
    if resume:
        saved = load_checkpoint(output/'latest.pt')
        if saved['context'] != context:
            old = {k:v for k,v in saved['context'].items() if k != 'runtime'}
            new = {k:v for k,v in context.items() if k != 'runtime'}
            if not allow_runtime_change or old != new:
                raise ValueError('Uncertainty resume rejected: source/config/data/mean/runtime changed')
        state = saved['progress']
        if saved['context']['runtime'] != context['runtime']:
            state.setdefault('runtime_changes', []).append({'from': saved['context']['runtime'], 'to': context['runtime'], 'bitwise_equivalence_claimed': False})
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        restore_rng(saved['rng'], generator, allow_device_change=allow_runtime_change)
        initial_std = np.asarray(saved['initial_std'])
        if state['status'] == 'completed':
            if full:
                verify_frozen(output)
            return json.loads((output/'training_summary.json').read_text())
    else:
        state = {'status': 'running', 'epoch': 0, 'global_step': 0, 'best_nll': None,
                 'stop_nll': None, 'bad_epochs': 0, 'best_epoch': None, 'history': [], 'runtime_changes': []}
    write_json(output/'declaration.json', context)
    def checkpoint():
        return {'context': context, 'progress': copy.deepcopy(state), 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(), 'rng': capture_rng(generator), 'initial_std': initial_std.tolist()}
    def save():
        save_checkpoint(output/'latest.pt', checkpoint())
        pd.DataFrame(state['history']).to_csv(output/'history.csv', index=False)
        mirror_run(output, mirror_dir)
    save()
    x = torch.from_numpy(mean.norm.transform(training))
    dataset = TensorDataset(x, torch.from_numpy(error.astype(np.float32)), torch.from_numpy(mask))
    selection_encoded, selection_mask = mean.codec.encode(selection)
    selection_error = torch.from_numpy((selection_encoded-mean.scaled(selection)).astype(np.float32)).to(device)
    selection_mask = torch.from_numpy(selection_mask).to(device)
    state['status'] = 'running'
    while state['epoch'] < config['maximum_epochs'] and state['bad_epochs'] < config['patience']:
        model.train()
        total, rows = 0., 0
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, generator=generator)
        for batch_x, batch_error, batch_mask in loader:
            optimizer.zero_grad(set_to_none=True)
            std, rho = model(batch_x.to(device))
            loss = gaussian_nll(batch_error.to(device), std, rho, batch_mask.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite Gaussian training loss')
            loss.backward()
            clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            if not torch.isfinite(clipped):
                raise ValueError('Non-finite Gaussian gradients')
            optimizer.step()
            total += float(loss.item())*len(batch_x)
            rows += len(batch_x)
            state['global_step'] += 1
        model.eval()
        with torch.no_grad():
            s, r = model(torch.from_numpy(mean.norm.transform(selection)).to(device))
            score = float(gaussian_nll(selection_error, s, r, selection_mask).item())
        if not np.isfinite(score):
            raise ValueError('Non-finite selection NLL')
        state['epoch'] += 1
        improved = state['best_nll'] is None or score < state['best_nll']
        if improved:
            state['best_nll'], state['best_epoch'] = score, state['epoch']
        if state['stop_nll'] is None or score < state['stop_nll']-config['minimum_nll_improvement']:
            state['stop_nll'], state['bad_epochs'] = score, 0
        else:
            state['bad_epochs'] += 1
        record = {'epoch': state['epoch'], 'global_step': state['global_step'],
                  'training_normalized_nll': total/rows, 'selection_normalized_nll': score}
        state['history'].append(record)
        if improved:
            save_checkpoint(output/'best.pt', checkpoint())
        save()
        print(json.dumps(record), flush=True)
        if stop_after_epochs is not None and state['epoch'] >= stop_after_epochs:
            state['status'] = 'paused'
            save()
            return {'status': 'paused', 'global_step': state['global_step']}
    state['status'] = 'completed'
    save()
    best = load_checkpoint(output/'best.pt')
    model.load_state_dict(best['model_state'])
    std, rho = predict_spread(model, mean, selection, device)
    restored = make_spread(config, mean.codec.scales, best['initial_std']).to(device)
    restored.load_state_dict(best['model_state'])
    std_again, rho_again = predict_spread(restored, mean, selection, device)
    if not np.array_equal(std, std_again) or not np.array_equal(rho, rho_again):
        raise ValueError('Spread checkpoint reload changed predictions')
    # Re-hash all original mean artifacts after training: no mean weights changed.
    verify_files(output/'mean', mean.record['files'])
    metrics = report_distribution(output, 'selection', mean, selection, std, rho, config)
    summary = {'status': 'uncertainty_research_completed' if full else 'acceptance_passed_not_research',
               'mean_parameters_changed': False, 'checkpoint_reload_max_abs_difference': 0.,
               'epochs_completed': state['epoch'], 'best_epoch': state['best_epoch'],
               'global_step': state['global_step'], 'best_selection_normalized_nll': state['best_nll'],
               'config': config, 'runtime': context['runtime'], 'runtime_changes': state['runtime_changes'],
               'initial_std_normalized': initial_std.tolist(), 'initialization_is_not_OOF': True,
               'selection_rows': len(selection), 'selection_storms': int(selection.SID.nunique()),
               'best_checkpoint_sha256': file_receipt(output/'best.pt')['sha256'],
               'calibration_labels_used': False, 'test_parquet_decoded': False,
               'selection_use_limit': 'Reused selection storms; uncertainty scores are not independent final-test evidence'}
    write_json(output/'training_summary.json', summary)
    if full:
        files = [p for p in (output/'mean').iterdir() if p.is_file()]+[output/'best.pt', output/'declaration.json', output/'training_summary.json']
        record = {'status': 'research_uncertainty_frozen', 'config': config, 'source_sha256': source_hashes(),
                  'package_sha256': mean.record['package_sha256'], 'feature_columns': mean.features,
                  'validation_roles_sha256': mean.record['validation_roles_sha256'],
                  'mean_freeze_sha256': context['mean_freeze_sha256'], 'mean_parameters_frozen': True,
                  'original_export_sha256': original_hash,
                  'distribution': config['distribution'], 'metrics_definition': {'marginal_CRPS': 'analytic Gaussian native units',
                  'track_energy_score': 'fixed Monte Carlo in current-centered east/north km',
                  'ellipse_threshold': '-2*log(1-coverage) for chi-square(2)', 'clipping': False},
                  'files': {p.relative_to(output).as_posix(): file_receipt(p) for p in files},
                  'calibration_labels_used': False, 'test_parquet_decoded': False}
        write_json(output/'frozen_model.json', record)
        verify_frozen(output)
    mirror_run(output, mirror_dir)
    return summary


def calibrate(frozen_directory, device='auto', mirror_dir=None):
    directory = Path(frozen_directory)
    frozen = verify_frozen(directory)  # All scientific choices and weights fixed first.
    torch.set_num_threads(2)
    config, device = frozen['config'], choose_device(device)
    seed_everything(config['seed'])
    package = TrainingPackage(DEFAULT_PACKAGE)
    if hashlib.sha256(package.path.read_bytes()).hexdigest() != frozen['package_sha256']:
        raise ValueError('Frozen data hash differs')
    # Preflight device and data before claiming the single calibration output.
    destination = directory/'calibration'
    destination.mkdir()
    write_json(destination/'status.json', {'status': 'started', 'frozen_model_sha256': digest(frozen)})
    frame, roles = role_data(package, 'calibration')
    if hashlib.sha256(roles.to_csv(index=False, lineterminator='\n').encode()).hexdigest() != frozen['validation_roles_sha256']:
        raise ValueError('Calibration role assignment changed')
    mean = FrozenMean(directory/'mean', device)
    saved = load_checkpoint(directory/'best.pt')
    model = make_spread(config, mean.codec.scales, saved['initial_std']).to(device)
    model.load_state_dict(saved['model_state'])
    std, rho = predict_spread(model, mean, frame, device)
    native_mean = mean.native_mean(frame, mean.scaled(frame))
    factors, details = fit_scale_factors(native_truth(frame), native_mean, std, rho, config['calibration_scale_bounds'])
    before = distribution_scores(native_truth(frame), native_mean, std, rho, config)
    after = report_distribution(destination, 'calibration', mean, frame, std, rho, config, factors)
    record = {'status': 'calibrated_once', 'frozen_model_sha256': digest(frozen),
              'scale_factors_track_wind_pressure': factors, 'fit_details': details,
              'covariance_rule': 'Track covariance multiplied by track_factor squared; correlation and means unchanged',
              'means_changed': False, 'rows': len(frame), 'storms': int(frame.SID.nunique()),
              'before': before, 'after': after, 'calibration_labels_used': True, 'test_parquet_decoded': False,
              'evaluation_limit': 'Calibration fit-set diagnostics, not independent validation or a coverage guarantee',
              'runtime': runtime_info(device),
              'source_sha256': source_hashes()}
    write_json(destination/'calibration.json', record)
    write_json(destination/'status.json', {'status': 'completed', 'frozen_model_sha256': digest(frozen),
                                         'calibration_sha256': file_receipt(destination/'calibration.json')['sha256']})
    mirror_run(directory, mirror_dir)
    return {k:record[k] for k in ['status', 'scale_factors_track_wind_pressure', 'rows', 'storms', 'test_parquet_decoded']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--results', required=True)
    fit.add_argument('--output', required=True)
    fit.add_argument('--full', action='store_true')
    fit.add_argument('--resume', action='store_true')
    fit.add_argument('--stop-after-epochs', type=int)
    fit.add_argument('--allow-runtime-change', action='store_true')
    calibration = sub.add_parser('calibrate')
    calibration.add_argument('--frozen', required=True)
    for command in [fit, calibration]:
        command.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        command.add_argument('--mirror-dir')
    args = parser.parse_args()
    if args.command == 'train':
        result = train(args.results, args.output, args.full, args.device, args.resume, args.mirror_dir,
                       args.allow_runtime_change, args.stop_after_epochs)
        print(json.dumps({k:v for k,v in result.items() if k not in ['config', 'runtime']}, indent=2))
    else:
        print(json.dumps(calibrate(args.frozen, args.device, args.mirror_dir), indent=2))


if __name__ == '__main__':
    main()
