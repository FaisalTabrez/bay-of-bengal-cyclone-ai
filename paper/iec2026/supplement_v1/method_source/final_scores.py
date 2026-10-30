"""Predeclared final-test scores; no model, scale or threshold estimation."""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from .baselines import evaluate as evaluate_baseline, predictions
from .metrics import evaluate_records, prediction_records
from .geometry import log_map


def native_truth(frame):
    track = log_map(frame.LAT.to_numpy(), frame.LON.to_numpy(), frame.TARGET_LAT_6H.to_numpy(), frame.TARGET_LON_6H.to_numpy())
    return np.column_stack([track, frame[['TARGET_WIND_KTS_6H', 'TARGET_PRESSURE_HPA_6H']].to_numpy(float)])


def normal_cdf(value):
    return np.vectorize(lambda x: .5*math.erfc(-x/math.sqrt(2)), otypes=[float])(value)


def marginal_crps(truth, mean, std):
    z = (truth-mean)/std
    return std*(z*(2*normal_cdf(z)-1)+2*np.exp(-.5*z*z)/math.sqrt(2*math.pi)-1/math.sqrt(math.pi))


def storm_intervals(records, columns, replicates, seed):
    grouped = records.groupby('SID')[columns]
    sums, counts = grouped.sum().to_numpy(), grouped.count().to_numpy()
    if len(sums) < 2:
        return {'unit': 'whole_cyclone', 'status': 'fewer_than_two_storms', 'intervals': {}}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(sums), size=len(sums))
        numerator, denominator = sums[sampled].sum(0), counts[sampled].sum(0)
        draws.append(np.divide(numerator, denominator, out=np.full(len(columns), np.nan), where=denominator > 0))
    values = np.asarray(draws)
    intervals = {}
    for j, name in enumerate(columns):
        valid = values[np.isfinite(values[:, j]), j]
        intervals[name] = {'low': float(np.quantile(valid, .025)) if len(valid) else None,
                           'high': float(np.quantile(valid, .975)) if len(valid) else None,
                           'valid_replicates': len(valid)}
    return {'unit': 'whole_cyclone', 'replicates': replicates, 'seed': seed,
            'interval': '95% percentile, row-weighted means resampling whole cyclones', 'intervals': intervals}


def probability_records(frame, mean, std, rho, protocol, pressure_bounds):
    """Per-row scores permit paired storm bootstrap and stable subgroup summaries."""
    truth = native_truth(frame)
    mean, std, rho = np.asarray(mean, float), np.asarray(std, float), np.asarray(rho, float)
    if mean.shape != (len(frame), 4) or std.shape != mean.shape or rho.shape != (len(frame),):
        raise ValueError('Distribution shape differs from frozen test rows')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any() or not np.isfinite(rho).all() or (np.abs(rho) >= 1).any():
        raise ValueError('Invalid Gaussian output')
    result = frame[['SID', 'ISO_TIME']].copy()
    error = truth-mean
    valid_track = np.isfinite(truth[:, :2]).all(1)
    z = error[:, :2]/std[:, :2]
    q = z[:, 0]**2+((z[:, 1]-rho*z[:, 0])/np.sqrt(1-rho*rho))**2
    result['track_joint_nll'] = np.where(valid_track, math.log(2*math.pi)+np.log(std[:, :2]).sum(1)+.5*np.log1p(-rho*rho)+.5*q, np.nan)
    for j, name in enumerate(['east_km', 'north_km', 'wind_kts', 'pressure_hpa']):
        valid = np.isfinite(truth[:, j])
        result[name+'_crps'] = np.where(valid, marginal_crps(truth[:, j], mean[:, j], std[:, j]), np.nan)
        if j >= 2:
            result[name+'_nll'] = np.where(valid, .5*math.log(2*math.pi)+np.log(std[:, j])+.5*(error[:, j]/std[:, j])**2, np.nan)
        for level in protocol['coverage_levels']:
            tag = str(round(level*100))
            if j >= 2:
                width = NormalDist().inv_cdf((1+level)/2)*std[:, j]
                result[f'{name}_coverage_{tag}'] = np.where(valid, (np.abs(error[:, j]) <= width).astype(float), np.nan)
                result[f'{name}_width_{tag}'] = np.where(valid, 2*width, np.nan)
    for level in protocol['coverage_levels']:
        tag, threshold = str(round(level*100)), -2*math.log1p(-level)
        result['track_coverage_'+tag] = np.where(valid_track, (q <= threshold).astype(float), np.nan)
        result['track_area_km2_'+tag] = np.where(valid_track, math.pi*threshold*np.prod(std[:, :2], axis=1)*np.sqrt(1-rho*rho), np.nan)
    # Identical standard-normal draws across the three fixed Gaussian forecasts.
    rng = np.random.default_rng(protocol['energy_score_seed'])
    shape = (len(frame), protocol['energy_score_draws'], 2)
    def sample():
        noise = rng.normal(size=shape)
        noise[:, :, 1] = rho[:, None]*noise[:, :, 0]+np.sqrt(1-rho[:, None]**2)*noise[:, :, 1]
        return mean[:, None, :2]+std[:, None, :2]*noise
    a, b = sample(), sample()
    result['track_energy_km'] = np.where(valid_track, np.linalg.norm(a-truth[:, None, :2], axis=2).mean(1)-.5*np.linalg.norm(a-b, axis=2).mean(1), np.nan)
    result['probability_wind_below_zero'] = normal_cdf(-mean[:, 2]/std[:, 2])
    lo, hi = pressure_bounds
    result['probability_pressure_outside_diagnostic_bounds'] = normal_cdf((lo-mean[:, 3])/std[:, 3])+normal_cdf((mean[:, 3]-hi)/std[:, 3])
    return result


def probability_summary(records, protocol):
    numeric = [c for c in records if c not in {'SID', 'ISO_TIME'}]
    means = {c: float(records[c].mean()) if records[c].notna().any() else None for c in numeric}
    counts = {c: int(records[c].notna().sum()) for c in numeric}
    return {'rows': len(records), 'storms': int(records.SID.nunique()), 'means': means,
            'observed_counts': counts,
            'bootstrap': storm_intervals(records, numeric, protocol['bootstrap_replicates'], protocol['bootstrap_seed'])}


def _final_names(value):
    if isinstance(value, dict):
        return {('mean_reference_normalized_error' if k == 'selection_composite' else k):
                ('95% whole-cyclone percentile interval on frozen final test' if k == 'interval' and isinstance(v, str)
                 else _final_names(v)) for k, v in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def score_all(frame, parameters, protocol, frozen_config, reference_std, output):
    from pathlib import Path
    output = Path(output)
    point = prediction_records(frame, parameters['point'])
    point.to_csv(output/'test_point_predictions.csv', index=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        point_metrics = _final_names(evaluate_records(point, protocol['bootstrap_replicates'], protocol['bootstrap_seed']))
    point.groupby('SID')[['TRACK_ERROR_KM', 'WIND_ABS_ERROR', 'PRESSURE_ABS_ERROR']].agg(['mean', 'count']).to_csv(output/'point_per_storm.csv')
    bay = (frame.LAT.between(*protocol['geographic_bay']['latitude']) & frame.LON.between(*protocol['geographic_bay']['longitude'])).to_numpy()
    cohorts = {'full': np.ones(len(frame), dtype=bool), 'geographic_bay': bay}
    models = {'training_RMS_diagonal_gaussian': (np.broadcast_to(reference_std, (len(frame), 4)), np.zeros(len(frame))),
              'frozen_uncalibrated_gaussian': (parameters['std_uncalibrated'], parameters['rho']),
              'frozen_calibrated_gaussian': (parameters['std_calibrated'], parameters['rho'])}
    probability, records_by_model, storm_rows = {}, {}, []
    for model, (std, rho) in models.items():
        records = probability_records(frame, parameters['mean'], std, rho, protocol, frozen_config['pressure_diagnostic_bounds_hpa'])
        records.to_csv(output/(model+'_row_scores.csv'), index=False)
        records_by_model[model] = records
        probability[model] = {}
        for cohort, mask in cohorts.items():
            subset = records.loc[mask]
            probability[model][cohort] = probability_summary(subset, protocol)
            numeric = [c for c in subset if c not in {'SID', 'ISO_TIME'}]
            for sid, group in subset.groupby('SID'):
                storm_rows.append(dict(model=model, cohort=cohort, SID=sid, rows=len(group), **{c: float(group[c].mean()) if group[c].notna().any() else None for c in numeric}))
    pd.DataFrame(storm_rows).to_csv(output/'probability_per_storm.csv', index=False)
    paired = {}
    paired_columns = ['track_joint_nll', 'wind_kts_nll', 'pressure_hpa_nll', 'wind_kts_crps', 'pressure_hpa_crps', 'track_energy_km']
    calibrated = records_by_model['frozen_calibrated_gaussian']
    for reference in ['training_RMS_diagonal_gaussian', 'frozen_uncalibrated_gaussian']:
        differences = calibrated[['SID', 'ISO_TIME']].copy()
        for column in paired_columns:
            differences[column] = calibrated[column]-records_by_model[reference][column]
        paired[reference] = {label: probability_summary(differences.loc[mask], protocol) for label, mask in cohorts.items()}
    distribution = frame[['SID', 'ISO_TIME', 'LAT', 'LON']].copy()
    for j, name in enumerate(['EAST_KM', 'NORTH_KM', 'WIND_KTS', 'PRESSURE_HPA']):
        distribution['MEAN_'+name] = parameters['mean'][:, j]
        distribution['STD_CALIBRATED_'+name] = parameters['std_calibrated'][:, j]
        distribution['STD_UNCALIBRATED_'+name] = parameters['std_uncalibrated'][:, j]
    distribution['TRACK_CORRELATION'] = parameters['rho']
    distribution.to_csv(output/'test_distribution.csv', index=False)
    return {'status': 'final_test_scored', 'rows': len(frame), 'storms': int(frame.SID.nunique()),
            'input_years': [int(frame.ISO_TIME.dt.year.min()), int(frame.ISO_TIME.dt.year.max())],
            'point_metrics': point_metrics,
            'physical_references': {'persistence': evaluate_baseline(frame, predictions(frame)),
                                    'constant_velocity_and_tendency': evaluate_baseline(frame, predictions(frame, tendency=True))},
            'probability_metrics': probability, 'paired_probability_differences_calibrated_minus_reference': paired,
            'density_convention': 'Gaussian current-centered east/north km and native intensity; no latitude/longitude density',
            'bootstrap_unit': 'whole_cyclone', 'tail_clipping_applied': False,
            'probability_reference_note': 'Training-RMS reference uses frozen in-sample initialization; no test-fitted parameters',
            'zero_reference_error_policy': 'Normalized error and skill are undefined when the reference error is zero; report null',
            'post_test_policy': 'Report all declared results; no adaptation or new calibration on this test set'}
