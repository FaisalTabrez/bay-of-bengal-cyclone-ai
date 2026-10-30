"""Gaussian spread model and scores in declared tangent-plane/native units."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import torch
from torch import nn


class SpreadMLP(nn.Module):
    def __init__(self, floors, initial_std, hidden_size=32, correlation_limit=.95):
        super().__init__()
        if not 0 < correlation_limit < 1:
            raise ValueError('Correlation limit must be strictly between zero and one')
        floors, initial_std = np.asarray(floors, float), np.asarray(initial_std, float)
        if floors.shape != (4,) or initial_std.shape != (4,) or not np.isfinite([floors, initial_std]).all() or (floors <= 0).any():
            raise ValueError('Expected four finite positive standard-deviation floors')
        self.register_buffer('floors', torch.tensor(floors, dtype=torch.float32))
        self.correlation_limit = correlation_limit
        self.network = nn.Sequential(nn.Linear(78, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 5))
        nn.init.zeros_(self.network[-1].weight)
        delta = np.maximum(initial_std - floors, 1e-4)
        with torch.no_grad():
            self.network[-1].bias[:4].copy_(torch.tensor(delta + np.log(-np.expm1(-delta)), dtype=torch.float32))
            self.network[-1].bias[4] = 0

    def forward(self, x):
        raw = self.network(x)
        return nn.functional.softplus(raw[:, :4]) + self.floors, self.correlation_limit * torch.tanh(raw[:, 4])


def gaussian_nll(error, std, rho, mask):
    """Index unavailable labels before arithmetic. Joint track is one task."""
    tasks = []
    observed_track = mask[:, :2].all(dim=1)
    if observed_track.any():
        z = error[observed_track, :2] / std[observed_track, :2]
        r = rho[observed_track]
        q = (z[:, 0]**2 - 2*r*z[:, 0]*z[:, 1] + z[:, 1]**2)/(1-r*r)
        tasks.append((math.log(2*math.pi) + torch.log(std[observed_track, :2]).sum(1)
                      + .5*torch.log1p(-r*r) + .5*q).mean())
    for j in [2, 3]:
        valid = mask[:, j]
        if valid.any():
            e, s = error[valid, j], std[valid, j]
            tasks.append((.5*math.log(2*math.pi) + torch.log(s) + .5*(e/s)**2).mean())
    if not tasks:
        raise ValueError('No observed targets')
    return torch.stack(tasks).mean()


def check_distribution(mean, std, rho):
    mean, std, rho = np.asarray(mean, float), np.asarray(std, float), np.asarray(rho, float)
    if mean.ndim != 2 or mean.shape[1] != 4 or std.shape != mean.shape or rho.shape != (len(mean),):
        raise ValueError('Expected N by 4 means/stds and N correlations')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not np.isfinite(rho).all() or (std <= 0).any() or (np.abs(rho) >= 1).any():
        raise ValueError('Invalid Gaussian parameters')
    return mean, std, rho


def normal_cdf(x):
    return np.vectorize(lambda z: .5*math.erfc(-z/math.sqrt(2)), otypes=[float])(x)


def marginal_crps(truth, mean, std):
    z = (np.asarray(truth)-mean)/std
    return std*(z*(2*normal_cdf(z)-1) + 2*np.exp(-.5*z*z)/math.sqrt(2*math.pi) - 1/math.sqrt(math.pi))


def mahalanobis_track(error, std, rho):
    z = error[:, :2]/std[:, :2]
    return (z[:, 0]**2-2*rho*z[:, 0]*z[:, 1]+z[:, 1]**2)/(1-rho*rho)


def fit_scale_factors(truth, mean, std, rho, bounds=(.25, 4.)):
    """Exact constrained MLE for one Gaussian, not a Gaussian mixture."""
    mean, std, rho = check_distribution(mean, std, rho)
    truth = np.asarray(truth, float)
    if truth.shape != mean.shape or not 0 < bounds[0] < bounds[1]:
        raise ValueError('Invalid calibration input/bounds')
    factors, details = [], {}
    for label, cols in [('track', [0, 1]), ('wind', [2]), ('pressure', [3])]:
        valid = np.isfinite(truth[:, cols]).all(1)
        if valid.sum() < 2:
            raise ValueError(f'At least two observed labels required for {label} calibration')
        if label == 'track':
            q = mahalanobis_track(truth[valid]-mean[valid], std[valid], rho[valid])
            raw = float(np.sqrt(q.mean()/2))
        else:
            j = cols[0]
            raw = float(np.sqrt(np.mean(((truth[valid, j]-mean[valid, j])/std[valid, j])**2)))
        factor = float(np.clip(raw, *bounds))
        factors.append(factor)
        details[label] = {'labels': int(valid.sum()), 'unconstrained_factor': raw, 'factor': factor,
                          'at_declared_bound': factor in bounds}
    return factors, details


def scaled_std(std, factors):
    f = np.asarray(factors, float)
    if f.shape != (3,) or not np.isfinite(f).all() or (f <= 0).any():
        raise ValueError('Expected positive track/wind/pressure scale factors')
    return np.asarray(std)*f[[0, 0, 1, 2]]


def distribution_scores(truth, mean, std, rho, config):
    mean, std, rho = check_distribution(mean, std, rho)
    truth = np.asarray(truth, float)
    if truth.shape != mean.shape:
        raise ValueError('Truth shape differs')
    levels = config['coverage_levels']
    result = {'density_convention': 'Gaussian density in current-centered east/north km and native intensity; not latitude/longitude density',
              'distribution': 'single Gaussian around fixed ensemble; not a three-component mixture',
              'rows': len(mean), 'marginals': {}}
    error = truth-mean
    track = np.isfinite(truth[:, :2]).all(1)
    q = mahalanobis_track(error[track], std[track], rho[track])
    track_nll = math.log(2*math.pi)+np.log(std[track, :2]).sum(1)+.5*np.log1p(-rho[track]**2)+.5*q
    result['track_joint_nll'] = float(track_nll.mean()) if len(q) else None
    result['track_labels'] = int(track.sum())
    result['track_ellipses'] = {}
    for level in levels:
        threshold = -2*math.log1p(-level)  # Exact chi-square(2) quantile.
        area = math.pi*threshold*np.prod(std[track, :2], axis=1)*np.sqrt(1-rho[track]**2)
        result['track_ellipses'][str(level)] = {'coverage': float(np.mean(q <= threshold)) if len(q) else None,
                                             'mean_area_km2': float(area.mean()) if len(area) else None}
    for j, name in enumerate(['east_km', 'north_km', 'wind_kts', 'pressure_hpa']):
        observed = np.isfinite(truth[:, j])
        y, m, s = truth[observed, j], mean[observed, j], std[observed, j]
        entry = {'labels': int(observed.sum()),
                 'gaussian_nll': float(np.mean(.5*math.log(2*math.pi)+np.log(s)+.5*((y-m)/s)**2)) if len(y) else None,
                 'crps': float(marginal_crps(y, m, s).mean()) if len(y) else None, 'intervals': {}}
        for level in levels:
            width = NormalDist().inv_cdf((1+level)/2)*s
            entry['intervals'][str(level)] = {'coverage': float(np.mean(np.abs(y-m) <= width)) if len(y) else None,
                                            'mean_width': float((2*width).mean()) if len(y) else None}
        result['marginals'][name] = entry
    # Fixed Monte Carlo draws for the joint spatial energy score, in km.
    draws = config['energy_score_draws']
    rng = np.random.default_rng(config['diagnostic_seed'])
    if len(q) and draws >= 2:
        def sample():
            z = rng.normal(size=(len(q), draws, 2))
            z[:, :, 1] = rho[track, None]*z[:, :, 0]+np.sqrt(1-rho[track, None]**2)*z[:, :, 1]
            return mean[track, None, :2]+std[track, None, :2]*z
        a, b = sample(), sample()
        score = np.linalg.norm(a-truth[track, None, :2], axis=2).mean(1)-.5*np.linalg.norm(a-b, axis=2).mean(1)
        result['track_energy_score_km'] = float(score.mean())
    else:
        result['track_energy_score_km'] = None
    result['energy_score_draws'] = draws
    lo, hi = config['pressure_diagnostic_bounds_hpa']
    result['gaussian_tail_mass'] = {
        'mean_probability_wind_below_zero': float(normal_cdf(-mean[:, 2]/std[:, 2]).mean()),
        'pressure_diagnostic_bounds_hpa': [lo, hi],
        'mean_probability_pressure_outside_bounds': float((normal_cdf((lo-mean[:, 3])/std[:, 3])+normal_cdf((mean[:, 3]-hi)/std[:, 3])).mean()),
        'clipping_applied': False, 'pressure_bounds_are_declared_diagnostics_not_universal_physical_limits': True}
    return result
