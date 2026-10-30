"""Run the one declared wind-loss experiment, then record the bounded decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiment_io import digest, mirror_run, source_hashes, write_json
from .paths import DEFAULT_PACKAGE
from .wind_audit import PROTOCOL_ID, SEEDS

CONFIG = Path(__file__).parent/'configs/wind_mae_followup_v1.json'


def verify_protocol(audit_dir):
    audit_dir = Path(audit_dir)
    protocol = json.loads((audit_dir/'followup_protocol.json').read_text())
    audit = json.loads((audit_dir/'audit.json').read_text())
    config = json.loads(CONFIG.read_text())
    if (protocol['protocol_id'] != PROTOCOL_ID or protocol['config_sha256'] != digest(config)
            or protocol['source_sha256'] != source_hashes()
            or protocol['original_results_sha256'] != audit['original_export_sha256']
            or protocol['audit_sha256'] != digest(audit)
            or protocol['package_sha256'] != audit['package_sha256']
            or protocol['seeds'] != list(SEEDS) or protocol['new_training_runs'] != 3):
        raise ValueError('Audit/protocol/source/config differs; produce a new audit before training')
    if audit['status'] != 'passed' or audit['calibration_labels_used'] or audit['test_parquet_decoded']:
        raise ValueError('Audit is not eligible')
    return protocol, audit


def run(audit_dir, output, full=False, device='auto', resume=False, mirror_dir=None, allow_runtime_change=False):
    import torch
    from .train import Inputs, aggregate, load_config, train_one

    protocol, audit = verify_protocol(audit_dir)
    output = Path(output)
    if output.resolve().is_relative_to(Path(audit_dir).resolve()) or Path(audit_dir).resolve().is_relative_to(output.resolve()):
        raise ValueError('Training and audit need separate directories')
    purpose = 'research_training' if full else 'acceptance_not_research'
    receipt = {'protocol_sha256': digest(protocol), 'purpose': purpose}
    if output.exists():
        if not resume or not (output/'declaration.json').exists():
            raise FileExistsError('Use a new output directory or resume this exact declaration')
        if json.loads((output/'declaration.json').read_text()) != receipt:
            raise ValueError('Resume declaration differs')
    else:
        output.mkdir(parents=True)
        write_json(output/'declaration.json', receipt)
        write_json(output/'followup_protocol.json', protocol)
    if mirror_dir:
        # Persist the resume declaration before the first epoch snapshot.
        mirror_run(output, mirror_dir)
    config = load_config(CONFIG, acceptance=not full)
    torch.set_num_threads(2)
    selected_device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device)
    if selected_device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    inputs = Inputs(DEFAULT_PACKAGE)
    for seed in config['training_plan']['seeds']:
        folder = output/'bay_mlp'/f'seed_{seed}'
        mirror = Path(mirror_dir)/'bay_mlp'/f'seed_{seed}' if mirror_dir else None
        train_one(inputs, config, 'bay_mlp', seed, folder, selected_device,
                  resume=resume and (folder/'run_manifest.json').exists(), mirror_dir=mirror,
                  allow_runtime_change=allow_runtime_change)
    result = aggregate(inputs, config, 'bay_mlp', output)
    write_json(output/'bay_mlp/aggregation_status.json', {k:v for k,v in result.items() if k!='selection_metrics'})
    if not full:
        decision = {'status': 'acceptance_passed_not_research', 'candidate_selection_performed': False,
                    'calibration_labels_used': False, 'test_parquet_decoded': False}
    else:
        candidates = {'original_bay_mlp': audit['comparison']['bay_mlp']['mean_seed_selection_score'],
                      'bay_mlp_wind_persistence': audit['hybrid']['mean_seed_selection_score'],
                      'bay_mlp_wind_absolute_loss': result['family_selection_score_mean_of_seeds']}
        if not np.isfinite(list(candidates.values())).all():
            raise ValueError('Non-finite candidate score')
        chosen = min(candidates, key=candidates.get)
        checkpoints = audit['original_mlp_checkpoints']
        if chosen == 'bay_mlp_wind_absolute_loss':
            checkpoints = {str(seed): {'sha256': json.loads((output/'bay_mlp'/f'seed_{seed}'/'summary.json').read_text())['best_checkpoint_sha256']}
                           for seed in SEEDS}
        decision = {'status': 'bounded_deterministic_selection_complete', 'protocol_sha256': digest(protocol),
                    'selected_candidate': chosen, 'candidate_mean_seed_scores': candidates, 'seeds': list(SEEDS),
                    'checkpoint_hashes': checkpoints, 'original_export_sha256': audit['original_export_sha256'],
                    'package_sha256': audit['package_sha256'], 'source_sha256': source_hashes(),
                    'ensemble_rule': 'mean_current_centred_spherical_displacements_and_native_intensity',
                    'wind_rule': 'observed current wind, otherwise learned wind' if chosen == 'bay_mlp_wind_persistence' else 'learned wind',
                    'calibration_labels_used': False, 'test_parquet_decoded': False,
                    'probabilistic_calibration': 'not_implemented', 'final_test_evaluation': 'not_authorized_by_this_command',
                    'claim_limit': 'Selection storms reused; this decision is not independent performance confirmation'}
    write_json(output/'candidate_decision.json', decision)
    if mirror_dir:
        mirror_run(output, mirror_dir)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--full', action='store_true', help='Run exactly the declared three research seeds; default is a two-epoch acceptance run')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--mirror-dir')
    parser.add_argument('--allow-runtime-change', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.audit, args.output, args.full, args.device, args.resume, args.mirror_dir, args.allow_runtime_change), indent=2))


if __name__ == '__main__':
    main()
