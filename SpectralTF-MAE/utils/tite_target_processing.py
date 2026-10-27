"""Ti/Te target parsing and 32-point profile construction.

Target construction follows these steps:
- Ti_out is parsed as [rho, Ti, error], then Ti/error are converted from keV to eV.
- Te_out is parsed as [rho, Te] and receives a synthetic rho=0 center point.
- Te is smoothed/interpolated to 32 points on rho in [0, 1].
- Ti is linearly interpolated where measured and its outer tail follows a scaled Te tail.
- Invalid, nonphysical, or strongly non-monotonic profiles are rejected.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

try:
    from smoothing_spline import Smoothing_spline
except (ImportError, ModuleNotFoundError):
    Smoothing_spline = None

INTERP_POINTS = 32
TARGET_GRID = np.linspace(0.0, 1.0, INTERP_POINTS, dtype=np.float32)


def parse_array_string(value) -> np.ndarray:
    """Parse bracketed/space/comma separated numeric text into a flat array."""
    if not isinstance(value, str):
        return np.array([], dtype=np.float32)
    cleaned = value.replace("[", " ").replace("]", " ").replace(",", " ").strip()
    if not cleaned:
        return np.array([], dtype=np.float32)
    return np.fromstring(cleaned, sep=" ", dtype=np.float32)


def parse_profile_matrix(value, preferred_columns: int, keep_error: bool = False) -> np.ndarray:
    """Parse Ti_out/Te_out into a 2-D matrix.

    Ti normally has 3 columns and Te normally has 2.  A two-column Ti input is
    accepted and receives a zero error column.
    """
    values = parse_array_string(value)
    if values.size == 0:
        return np.zeros((0, 3 if keep_error else 2), dtype=np.float32)

    if preferred_columns > 0 and values.size % preferred_columns == 0:
        data = values.reshape(-1, preferred_columns)
    elif values.size % 2 == 0:
        data = values.reshape(-1, 2)
    elif values.size % 3 == 0:
        data = values.reshape(-1, 3)
    else:
        return np.zeros((0, 3 if keep_error else 2), dtype=np.float32)

    if keep_error:
        if data.shape[1] >= 3:
            return data[:, :3].astype(np.float32, copy=False)
        error = np.zeros((data.shape[0], 1), dtype=np.float32)
        return np.concatenate([data[:, :2], error], axis=1)
    return data[:, :2].astype(np.float32, copy=False)


def _safe_savgol(values: np.ndarray) -> np.ndarray:
    if values.size >= 5:
        return savgol_filter(values, window_length=5, polyorder=2).astype(np.float32)
    return values.astype(np.float32, copy=False)


def interpolate_te_profile(data: np.ndarray, target_grid: np.ndarray = TARGET_GRID) -> np.ndarray:
    """Smooth/interpolate a Te [rho, value] profile to 32 radial points."""
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        return np.zeros_like(target_grid, dtype=np.float32)

    rho = np.asarray(data[:, 0], dtype=np.float64)
    val = np.asarray(data[:, 1], dtype=np.float64)
    mask = np.isfinite(rho) & np.isfinite(val)
    rho, val = rho[mask], val[mask]
    if rho.size < 2:
        return np.zeros_like(target_grid, dtype=np.float32)

    order = np.argsort(rho)
    rho, val = rho[order], val[order]
    _, unique_idx = np.unique(rho, return_index=True)
    rho, val = rho[unique_idx], val[unique_idx]

    # Insert intermediate samples in the densely weighted core.
    core = rho < 0.4
    if np.count_nonzero(core) >= 2:
        core_rho, core_val = rho[core], val[core]
        rho = np.concatenate([rho, (core_rho[:-1] + core_rho[1:]) / 2.0])
        val = np.concatenate([val, (core_val[:-1] + core_val[1:]) / 2.0])
        order = np.argsort(rho)
        rho, val = rho[order], val[order]

    # Densify the edge until there are at least five points in [0.8, 1.0].
    while True:
        edge = (rho >= 0.8) & (rho <= 1.0)
        n_edge = int(np.count_nonzero(edge))
        if n_edge < 2 or n_edge >= 5:
            break
        edge_rho, edge_val = rho[edge], val[edge]
        rho = np.concatenate([rho, (edge_rho[:-1] + edge_rho[1:]) / 2.0])
        val = np.concatenate([val, (edge_val[:-1] + edge_val[1:]) / 2.0])
        order = np.argsort(rho)
        rho, val = rho[order], val[order]

    new_val: Optional[np.ndarray] = None
    if Smoothing_spline is not None and rho.size >= 4:
        try:
            mask_core = rho < 0.5
            mask_edge = rho >= 0.3
            rho_core, val_core = rho[mask_core], val[mask_core]
            rho_edge, val_edge = rho[mask_edge], val[mask_edge]

            if rho_core.size >= 4 and rho_edge.size >= 4:
                w_core = np.ones_like(rho_core)
                w_core[rho_core < 0.4] = 10.0
                if rho_core[0] == 0.0:
                    w_core[0] = 1e6
                spline_core = Smoothing_spline(rho_core, val_core, w_core, 0.01)
                spline_core.fit()

                w_edge = np.ones_like(rho_edge)
                w_edge[rho_edge >= 0.8] = 10.0
                spline_edge = Smoothing_spline(rho_edge, val_edge, w_edge, 0.01)
                spline_edge.fit()

                grid = np.asarray(target_grid, dtype=np.float64)
                new_val = np.zeros_like(grid)
                core_grid = grid < 0.30
                edge_grid = grid > 0.50
                middle_grid = ~(core_grid | edge_grid)
                if np.any(core_grid):
                    new_val[core_grid] = spline_core.eval(grid[core_grid])
                if np.any(edge_grid):
                    new_val[edge_grid] = spline_edge.eval(grid[edge_grid])
                if np.any(middle_grid):
                    middle = grid[middle_grid]
                    core_pred = spline_core.eval(middle)
                    edge_pred = spline_edge.eval(middle)
                    t = (middle - 0.30) / 0.20
                    edge_weight = (1.0 - np.cos(t * np.pi)) / 2.0
                    new_val[middle_grid] = (1.0 - edge_weight) * core_pred + edge_weight * edge_pred
            else:
                weights = np.ones_like(rho)
                if rho[0] == 0.0:
                    weights[0] = 1e6
                spline = Smoothing_spline(rho, val, weights, 0.01)
                spline.fit()
                new_val = np.asarray(spline.eval(target_grid), dtype=np.float64)
        except Exception:
            new_val = None

    if new_val is None:
        kind = "cubic" if rho.size >= 4 else "linear"
        fn = interp1d(rho, val, kind=kind, bounds_error=False, fill_value=(val[0], val[-1]))
        new_val = np.nan_to_num(fn(target_grid), nan=0.0)
        new_val = _safe_savgol(new_val)

    new_val = np.asarray(new_val, dtype=np.float32)
    new_val[new_val < 0.0] = 0.0

    # Suppress local temperature increases along the radial grid.
    for _ in range(2):
        for i in range(1, len(new_val) - 1):
            if new_val[i] >= new_val[i - 1]:
                average = (new_val[i - 1] + new_val[i + 1]) / 2.0
                new_val[i] = min(average, new_val[i - 1])
        if new_val[-1] >= new_val[-2]:
            new_val[-1] = new_val[-2] - 1.0

    new_val = _safe_savgol(new_val)
    new_val[new_val < 0.0] = 0.0
    if rho[0] == 0.0 and target_grid[0] == 0.0:
        new_val[0] = float(val[0])

    if np.std(new_val) < 1.0 and np.std(val) > 10.0:
        fn = interp1d(rho, val, kind="linear", bounds_error=False, fill_value=(val[0], val[-1]))
        new_val = np.nan_to_num(fn(target_grid), nan=0.0).astype(np.float32)
        new_val[new_val < 0.0] = 0.0

    return new_val.astype(np.float32, copy=False)


def process_ti_profile(
    ti_data: np.ndarray,
    te_processed: np.ndarray,
    target_grid: np.ndarray = TARGET_GRID,
) -> Optional[np.ndarray]:
    """Construct a 32-point Ti target using measured Ti and a scaled Te tail."""
    if ti_data.ndim != 2 or ti_data.shape[0] < 2 or ti_data.shape[1] < 3:
        return None

    value = ti_data[:, 1]
    error = ti_data[:, 2]
    valid = (
        np.isfinite(ti_data[:, 0])
        & np.isfinite(value)
        & np.isfinite(error)
        & (value > 0.0)
        & ((error / np.maximum(value, 1e-12)) <= 0.5)
    )
    rho = ti_data[valid, 0]
    val = ti_data[valid, 1]
    if rho.size < 2:
        return None

    order = np.argsort(rho)
    rho, val = rho[order], val[order]
    _, unique_idx = np.unique(rho, return_index=True)
    rho, val = rho[unique_idx], val[unique_idx]
    if rho.size < 2:
        return None

    fn = interp1d(rho, val, kind="linear", bounds_error=False, fill_value=np.nan)
    interpolated = np.asarray(fn(target_grid), dtype=np.float32)
    valid_indices = np.where(np.isfinite(interpolated))[0]
    if valid_indices.size == 0:
        return None

    last_idx = int(valid_indices[-1])
    if last_idx == len(target_grid) - 1:
        combined = interpolated
    else:
        ti_last = float(interpolated[last_idx])
        te_join = float(te_processed[last_idx])
        factor = ti_last / te_join if te_join > 1.0 else 0.0
        combined = np.concatenate(
            [interpolated[: last_idx + 1], np.asarray(te_processed[last_idx + 1:]) * factor]
        )

    missing = ~np.isfinite(combined)
    if np.any(missing):
        finite = combined[np.isfinite(combined)]
        if finite.size == 0:
            return None
        combined[missing] = finite[0]

    final = np.asarray(combined, dtype=np.float32)
    if Smoothing_spline is not None:
        try:
            spline = Smoothing_spline(target_grid, final, np.ones_like(target_grid), 0.0021)
            spline.fit()
            final = np.asarray(spline.eval(target_grid), dtype=np.float32)
        except Exception:
            pass

    final[final < 0.0] = 0.0
    if final.size == 0 or not np.all(np.isfinite(final)):
        return None

    max_idx = int(np.argmax(final))
    if float(target_grid[max_idx]) >= 0.1:
        return None

    diffs = np.diff(final)
    max_increase = float(np.max(diffs)) if diffs.size else 0.0
    max_val = float(np.max(final))
    if max_increase > max_val * 0.15 or max_increase > 100.0:
        return None

    return final.astype(np.float32, copy=False)


def is_monotonic_with_tolerance(profile: np.ndarray, tolerance: float = 10.0) -> bool:
    if profile is None or len(profile) < 2:
        return False
    return not np.any(np.diff(profile) > tolerance)


def build_tite_targets(
    ti_raw,
    te_raw,
    target_grid: np.ndarray = TARGET_GRID,
    strict_quality: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(ti_32, te_32)`` or ``None`` when the profile is unusable."""
    ti = parse_profile_matrix(ti_raw, preferred_columns=3, keep_error=True)
    te = parse_profile_matrix(te_raw, preferred_columns=2, keep_error=False)
    if ti.shape[0] < 2 or te.shape[0] < 2:
        return None

    # Convert Ti measurements from keV to eV.
    ti = ti.copy()
    ti[:, 1] *= 1000.0
    ti[:, 2] *= 1000.0

    te = te[np.isfinite(te[:, 0]) & np.isfinite(te[:, 1])]
    if te.shape[0] < 2:
        return None
    te = te[np.argsort(te[:, 0])]
    synthetic_center = np.array([[0.0, float(te[0, 1]) * 1.01]], dtype=np.float32)
    te_with_center = np.vstack([synthetic_center, te])
    te_32 = interpolate_te_profile(te_with_center, target_grid)

    if strict_quality:
        core = target_grid <= 0.2
        if np.any(core) and float(np.max(te_32[core]) - np.min(te_32[core])) < 50.0:
            return None

    ti_32 = process_ti_profile(ti, te_32, target_grid)
    if ti_32 is None:
        return None

    if strict_quality:
        if not is_monotonic_with_tolerance(ti_32, tolerance=10.0):
            return None
        if not is_monotonic_with_tolerance(te_32, tolerance=10.0):
            return None

    ti_32 = np.asarray(ti_32, dtype=np.float32)
    te_32 = np.asarray(te_32, dtype=np.float32)
    ti_32[(ti_32 < 0.0) | (ti_32 > 10000.0)] = 0.0
    te_32[(te_32 < 0.0) | (te_32 > 10000.0)] = 0.0

    if not np.all(np.isfinite(ti_32)) or not np.all(np.isfinite(te_32)):
        return None
    if np.all(ti_32 == 0.0) or np.all(te_32 == 0.0):
        return None
    if ti_32.shape != (32,) or te_32.shape != (32,):
        return None
    return ti_32, te_32
