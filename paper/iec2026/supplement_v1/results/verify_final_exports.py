"""Independent review of submitted final artifacts; never decodes original holdout."""
import argparse
import hashlib
import io
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
from statistics import NormalDist
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modelling.final_evaluation import evaluator_fingerprint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inputs-dir', type=Path, default=ROOT.parent/'inputs' if (ROOT.parent/'inputs').is_dir() else Path.home()/'Downloads')
parser.add_argument('--output-dir', type=Path, default=ROOT/'reports/final_export_review')
args = parser.parse_args()
DOWNLOADS = args.inputs_dir
OUTPUT = args.output_dir
OUTPUT.mkdir(parents=True, exist_ok=True)
sha = lambda b: hashlib.sha256(b).hexdigest()
canonical = lambda v: sha(json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
checks = {}

def close(label, a, b, atol=1e-9):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=atol, equal_nan=True, err_msg=label)
    checks[label] = True

def files(z, receipts, prefix=''):
    for name, receipt in receipts.items():
        payload = z.read(prefix + name)
        assert len(payload) == receipt['bytes'] and sha(payload) == receipt['sha256'], name
    return len(receipts)

def distance(lat1, lon1, lat2, lon2):
    a, b = np.deg2rad(lat1), np.deg2rad(lat2)
    h = np.sin((b-a)/2)**2 + np.cos(a)*np.cos(b)*np.sin(np.deg2rad(lon2-lon1)/2)**2
    return 12742.0176*np.arctan2(np.sqrt(np.clip(h, 0, 1)), np.sqrt(np.clip(1-h, 0, 1)))

archives = {}
with zipfile.ZipFile(DOWNLOADS/'bob_cyclone_final_results.zip') as z, zipfile.ZipFile(DOWNLOADS/'bob_cyclone_final_preflight_results.zip') as pre:
    for label, archive in [('final', z), ('preflight', pre)]:
        manifest = json.loads(archive.read('RUN_FILE_MANIFEST.json'))
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == set(manifest) | {'RUN_FILE_MANIFEST.json'}
        archives[label] = dict(sha256=sha(Path(archive.filename).read_bytes()), verified_files=files(archive, manifest))
    plan = json.loads(z.read('prepared/reviewed_plan.json'))
    assert plan == json.loads(pre.read('prepared/reviewed_plan.json')) == json.loads(z.read('execution/reviewed_plan.json'))
    assert plan['evaluator_fingerprint'] == evaluator_fingerprint()
    files(z, plan['files'], 'prepared/')
    for name in plan['files']:
        assert z.read('prepared/'+name) == pre.read('prepared/'+name), name
    checks['prepared_files_identical_to_preflight'] = True
    with zipfile.ZipFile(DOWNLOADS/'uncertainty_v1_results (3).zip') as original:
        assert sha(Path(original.filename).read_bytes()) == plan['protocol']['calibrated_export_sha256']
        for name in z.namelist():
            if name.startswith('prepared/model/'):
                assert z.read(name) == original.read(name.replace('prepared/model/', 'research/')), name
    checks['frozen_model_and_calibration_identical_to_original_export'] = True
    receipt = json.loads(z.read('execution/result_receipt.json'))
    receipt_files = files(z, receipt['files'], 'execution/')
    status = json.loads(z.read('execution_status.json'))
    ledger = status['ledger_claim']
    m = json.loads(z.read('execution/final_metrics.json'))
    assert ledger['status'] == receipt['status'] == 'completed'
    assert ledger['result_receipt_sha256'] == sha(z.read('execution/result_receipt.json'))
    assert canonical(plan) == ledger['reviewed_plan_sha256'] == receipt['reviewed_plan_sha256'] == m['reviewed_plan_sha256']
    assert ledger['attempt_id'] == receipt['attempt_id'] == m['attempt_id']
    assert not m['model_training'] and not m['calibration_refit']
    assert json.loads(pre.read('execution_status.json'))['ledger_claim'] is None
    checks['reviewed_plan_execution_and_ledger_consistent'] = True
    f = pd.read_parquet(io.BytesIO(z.read('execution/cache/test_frame.parquet')))
    p = pd.read_csv(io.BytesIO(z.read('execution/test_point_predictions.csv')))
    d = pd.read_csv(io.BytesIO(z.read('execution/test_distribution.csv')))
    parameters = np.load(io.BytesIO(z.read('execution/inference/parameters.npz')), allow_pickle=False)
    close('distribution_csv_means_match_binary_parameters', d[['MEAN_'+n for n in ['EAST_KM','NORTH_KM','WIND_KTS','PRESSURE_HPA']]], parameters['mean'])
    close('distribution_csv_correlation_matches_binary_parameters', d.TRACK_CORRELATION, parameters['rho'].astype(float), atol=1e-7)
    assert len(f) == len(p) == len(d) == 1018 and f.SID.nunique() == 65
    assert not f.duplicated(['SID', 'ISO_TIME']).any()
    assert set(pd.to_datetime(f.ISO_TIME).dt.year).issubset(range(2018, 2026))
    assert ((pd.to_datetime(f.TARGET_TIME_6H)-pd.to_datetime(f.ISO_TIME)) == pd.Timedelta(hours=6)).all()
    for other in [p, d]:
        assert f.SID.tolist() == other.SID.tolist()
        assert (pd.to_datetime(f.ISO_TIME).to_numpy() == pd.to_datetime(other.ISO_TIME).to_numpy()).all()
    for col in f.columns.intersection(p.columns).difference(['SID', 'ISO_TIME', 'TARGET_TIME_6H']):
        close('cached_frame_matches_predictions_'+col, f[col], p[col])
    inputs = pd.read_parquet(io.BytesIO(z.read('execution/inference/inputs.parquet')))
    assert not any(c.startswith('TARGET_') and c != 'TARGET_TIME_6H' for c in inputs)
    checks['cohort_horizon_unique_keys_and_target_label_free_inference_inputs'] = True

    track = distance(f.TARGET_LAT_6H.to_numpy(), f.TARGET_LON_6H.to_numpy(), p.PRED_LAT_6H.to_numpy(), p.PRED_LON_6H.to_numpy())
    cv = distance(f.TARGET_LAT_6H.to_numpy(), f.TARGET_LON_6H.to_numpy(), (f.LAT+f.LAT_CHANGE_6H.fillna(0)).to_numpy(), (f.LON+f.LON_CHANGE_6H.fillna(0)).to_numpy())
    close('independent_track_errors', track, p.TRACK_ERROR_KM)
    close('independent_constant_velocity_errors', cv, p.BASE_TRACK_ERROR_KM)
    error_columns = {'track': track, 'cv': cv}
    for label, var in [('wind', 'WIND_KTS'), ('pressure', 'PRESSURE_HPA')]:
        actual, current = f['TARGET_'+var+'_6H'].to_numpy(), f[var].to_numpy()
        error = np.abs(actual-p['PRED_'+var+'_6H'].to_numpy())
        paired = np.isfinite(actual) & np.isfinite(current)
        error_columns[label] = np.where(paired, error, np.nan)
        error_columns[label+'_base'] = np.where(paired, np.abs(actual-current), np.nan)
        close('independent_'+label+'_all_errors', error, p[label.upper()+'_ABS_ERROR'])
        close('independent_'+label+'_paired_errors', error_columns[label], p[label.upper()+'_PAIRED_ERROR'])
        close('independent_'+label+'_persistence_errors', error_columns[label+'_base'], p[label.upper()+'_BASE_ERROR'])
    errors = pd.DataFrame(error_columns).assign(SID=f.SID)
    grouped = errors.groupby('SID')
    sums, counts = grouped.sum().to_numpy(), grouped.count().to_numpy()
    rng = np.random.default_rng(1701)
    picks = rng.integers(0, 65, size=(2000, 65))
    means = sums[picks].sum(1)/counts[picks].sum(1)
    differences = means[:, [0, 2, 4]]-means[:, [1, 3, 5]]
    intervals = {}
    for j, name in enumerate(['paired_track_error_difference_km', 'paired_wind_error_difference_kts', 'paired_pressure_error_difference_hpa']):
        ci = np.quantile(differences[:, j], [.025, .975])
        declared = m['point_metrics']['full']['bootstrap']['intervals'][name]
        close('independent_bootstrap_'+name, ci, [declared['low'], declared['high']])
        intervals[name] = ci.tolist()

    # Independent spherical inverse and native-unit Gaussian score calculations.
    lat, lon, tlat, tlon = [np.deg2rad(f[c].to_numpy()) for c in ['LAT', 'LON', 'TARGET_LAT_6H', 'TARGET_LON_6H']]
    bearing = np.arctan2(np.sin(tlon-lon)*np.cos(tlat), np.cos(lat)*np.sin(tlat)-np.sin(lat)*np.cos(tlat)*np.cos(tlon-lon))
    radial = distance(f.LAT.to_numpy(), f.LON.to_numpy(), f.TARGET_LAT_6H.to_numpy(), f.TARGET_LON_6H.to_numpy())
    truth = np.column_stack([radial*np.sin(bearing), radial*np.cos(bearing), f.TARGET_WIND_KTS_6H, f.TARGET_PRESSURE_HPA_6H])
    names = ['EAST_KM', 'NORTH_KM', 'WIND_KTS', 'PRESSURE_HPA']
    mu = d[['MEAN_'+n for n in names]].to_numpy()
    cdf = np.vectorize(NormalDist().cdf)
    coverage = {}
    for model in m['probability_metrics']:
        std = (np.broadcast_to(plan['training_RMS_reference_std_native'], (len(f), 4)) if model.startswith('training')
               else d[['STD_'+('UNCALIBRATED' if 'uncalibrated' in model else 'CALIBRATED')+'_'+n for n in names]].to_numpy())
        # CSV renders float32 rho at its source precision; use the supplied binary
        # values for strict score checks and verify the CSV separately above.
        rho = np.zeros(len(f)) if model.startswith('training') else parameters['rho'].astype(float)
        residual = (truth-mu)/std
        q = (residual[:,0]**2-2*rho*residual[:,0]*residual[:,1]+residual[:,1]**2)/(1-rho**2)
        values = {'track_joint_nll': np.log(2*np.pi*std[:,0]*std[:,1]*np.sqrt(1-rho*rho))+.5*q}
        for level in [.5, .8, .9]:
            values['track_coverage_'+str(round(100*level))] = (q <= -2*np.log(1-level)).astype(float)
        for j, label in [(2,'wind_kts'), (3,'pressure_hpa')]:
            valid = np.isfinite(truth[:,j])
            r = residual[:,j]
            values[label+'_nll'] = np.where(valid, np.log(std[:,j]*np.sqrt(2*np.pi))+.5*r*r, np.nan)
            values[label+'_crps'] = np.where(valid, std[:,j]*(r*(2*cdf(r)-1)+np.sqrt(2/np.pi)*np.exp(-r*r/2)-1/np.sqrt(np.pi)), np.nan)
            for level in [.5,.8,.9]:
                values[label+'_coverage_'+str(round(100*level))] = np.where(valid, (np.abs(r)<=NormalDist().inv_cdf((1+level)/2)).astype(float), np.nan)
        scores = pd.read_csv(io.BytesIO(z.read('execution/'+model+'_row_scores.csv')))
        for name, values_array in values.items():
            close('independent_'+model+'_'+name, values_array, scores[name])
            close('summary_'+model+'_'+name, np.nanmean(values_array), m['probability_metrics'][model]['full']['means'][name])
        if model == 'frozen_calibrated_gaussian':
            coverage = {n: float(np.nanmean(v)) for n, v in values.items() if 'coverage' in n}
    (OUTPUT/'final_test_coverage.png').write_bytes(z.read('final_test_coverage.png'))

report = dict(status='passed', review_scope='Submitted exports and cached test rows only; no model fitting, calibration, inference or original dataset decoding',
              audit_runtime={'python': platform.python_version(), 'platform': platform.system(),
                             'packages': {name: version(name) for name in ['numpy', 'pandas', 'pyarrow', 'torch']}},
              archives=archives, reviewed_plan_sha256=canonical(plan), attempt_id=ledger['attempt_id'], result_receipt_files_verified=receipt_files,
              checks=checks, rows=1018, storms=65, input_years=[2018,2025], point_metrics=m['point_metrics'],
              independently_reproduced_paired_bootstrap_intervals=intervals, calibrated_coverage=coverage,
              probability_metrics=m['probability_metrics'], paired_probability_differences=m['paired_probability_differences_calibrated_minus_reference'],
              limitations=['Retrospective reanalysis and best-track evaluation, not operational forecast validation.',
                           'Central 50% intervals overcover; calibration changes some proper scores adversely.',
                           'Intensity skill is not uniform across predeclared subgroups.',
                           'Final holdout consumed; no further model or calibration changes using these outcomes.'])
(OUTPUT/'final_test_export_verification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps(dict(status=report['status'], checks=len(checks), archives=archives, paired_bootstrap_intervals=intervals, coverage=coverage), indent=2))
