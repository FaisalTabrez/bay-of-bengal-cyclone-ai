"""Spherical log/exp maps: east/north displacement in km at the current centre."""
import numpy as np

EARTH_KM = 6371.0088


def haversine_km(lat, lon, target_lat, target_lon):
    p, q = np.radians(lat), np.radians(target_lat)
    dl = np.radians(np.asarray(target_lon) - np.asarray(lon))
    a = np.sin((q - p) / 2) ** 2 + np.cos(p) * np.cos(q) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def log_map(lat, lon, target_lat, target_lon):
    p, q = np.radians(lat), np.radians(target_lat)
    dl = np.radians(np.asarray(target_lon) - np.asarray(lon))
    bearing = np.arctan2(np.sin(dl) * np.cos(q),
                        np.cos(p) * np.sin(q) - np.sin(p) * np.cos(q) * np.cos(dl))
    distance = haversine_km(lat, lon, target_lat, target_lon)
    return np.stack([distance * np.sin(bearing), distance * np.cos(bearing)], axis=-1)


def exp_map(lat, lon, east_north):
    p, l = np.radians(lat), np.radians(lon)
    east, north = np.asarray(east_north).T
    distance = np.hypot(east, north) / EARTH_KM
    bearing = np.arctan2(east, north)
    q = np.arcsin(np.clip(np.sin(p) * np.cos(distance) +
                         np.cos(p) * np.sin(distance) * np.cos(bearing), -1, 1))
    dl = np.arctan2(np.sin(bearing) * np.sin(distance) * np.cos(p),
                    np.cos(distance) - np.sin(p) * np.sin(q))
    return np.stack([np.degrees(q), (np.degrees(l + dl) + 180) % 360 - 180], axis=-1)
