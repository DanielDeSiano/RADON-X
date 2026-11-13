from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radon_x.core.boundaries import (
    BoundaryCondition,
    base_face_flux,
    surface_face_flux,
)
from radon_x.core.decay import LAMBDA_RN222
from radon_x.core.solver import (
    DEFAULT_BASE_CONDITION,
    DEFAULT_SURFACE_CONDITION,
    MINIMUM_CELL_COUNT,
    cell_centres,
)

__all__ = [
    "PLATEAU_RELATIVE_SPAN",
    "PLATEAU_TAU_MULTIPLE",
    "REFERENCE_DEPTH_M",
    "column_balance",
    "column_inventory",
    "column_mean_concentration",
    "concentration_at_depth",
    "exhalation_plateau",
    "surface_exhalation_flux",
]

PLATEAU_TAU_MULTIPLE: float = 3.0

PLATEAU_RELATIVE_SPAN: float = math.exp(-PLATEAU_TAU_MULTIPLE)

REFERENCE_DEPTH_M: float = 0.8


def _require_finite(name: str, value: float) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _require_positive(name: str, value: float, unit: str) -> float:
    _require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0 {unit}, got {value!r}")
    return float(value)


def _as_cell_array(
    name: str,
    values: ArrayLike,
    unit: str,
    *,
    strictly_positive: bool,
    expected_size: int | None = None,
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.ndim} dimensions")
    if expected_size is not None and array.size != expected_size:
        raise ValueError(
            f"{name} must have {expected_size} entries, one per cell, got {array.size}"
        )
    if array.size < MINIMUM_CELL_COUNT:
        raise ValueError(
            f"{name} must have at least {MINIMUM_CELL_COUNT} cells, got {array.size}"
        )
    if not np.all(np.isfinite(array)):
        bad = int(np.argmin(np.isfinite(array)))
        raise ValueError(f"{name} must be finite everywhere, entry {bad} is {array[bad]!r}")
    if strictly_positive:
        if not np.all(array > 0.0):
            bad = int(np.argmin(array))
            raise ValueError(f"{name} must be > 0 {unit} everywhere, entry {bad} is {array[bad]!r}")
    elif not np.all(array >= 0.0):
        bad = int(np.argmin(array))
        raise ValueError(f"{name} must be >= 0 {unit} everywhere, entry {bad} is {array[bad]!r}")
    return array


def _as_concentration_array(
    values: ArrayLike,
    *,
    expected_size: int | None = None,
) -> NDArray[np.float64]:
    return _as_cell_array(
        "concentration_bq_m3",
        values,
        "Bq m-3",
        strictly_positive=False,
        expected_size=expected_size,
    )


def _require_finite_result(name: str, value: float) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{name} overflowed to {value!r}; check the magnitudes of the inputs")
    return float(value)


def surface_exhalation_flux(
    d_bulk_surface_m2_s: float,
    c_first_cell_bq_m3: float,
    dz_m: float,
    surface: BoundaryCondition | None = None,
) -> float:
    condition = DEFAULT_SURFACE_CONDITION if surface is None else surface
    return surface_face_flux(condition, d_bulk_surface_m2_s, c_first_cell_bq_m3, dz_m)


def column_mean_concentration(concentration_bq_m3: ArrayLike) -> float:
    profile = _as_concentration_array(concentration_bq_m3)
    return _require_finite_result("column mean concentration", float(np.mean(profile)))


def concentration_at_depth(
    depth_m: float,
    concentration_bq_m3: ArrayLike,
    dz_m: float,
) -> float:
    _require_finite("depth_m", depth_m)
    _require_positive("dz_m", dz_m, "m")
    profile = _as_concentration_array(concentration_bq_m3)
    depths = cell_centres(profile.size, float(dz_m))
    if depth_m < depths[0] or depth_m > depths[-1]:
        raise ValueError(
            f"depth_m must lie between the first and the last cell centre, "
            f"[{depths[0]!r}, {depths[-1]!r}] m, got {depth_m!r}; the outer half cells are "
            "fixed by the boundary condition and are not interpolated here"
        )
    return _require_finite_result(
        "concentration at depth", float(np.interp(float(depth_m), depths, profile))
    )


def column_inventory(
    concentration_bq_m3: ArrayLike,
    partition_porosity: ArrayLike,
    dz_m: float,
) -> float:
    profile = _as_concentration_array(concentration_bq_m3)
    beta = _as_cell_array(
        "partition_porosity",
        partition_porosity,
        "m3 m-3",
        strictly_positive=True,
        expected_size=profile.size,
    )
    _require_positive("dz_m", dz_m, "m")
    return _require_finite_result(
        "column inventory", float(np.sum(beta * profile) * float(dz_m))
    )


def column_balance(
    concentration_bq_m3: ArrayLike,
    partition_porosity: ArrayLike,
    source_bulk_bq_m3_s: ArrayLike,
    d_bulk_m2_s: ArrayLike,
    dz_m: float,
    lambda_s: float = LAMBDA_RN222,
    surface: BoundaryCondition | None = None,
    base: BoundaryCondition | None = None,
) -> dict[str, float]:
    profile = _as_concentration_array(concentration_bq_m3)
    n = profile.size
    beta = _as_cell_array(
        "partition_porosity", partition_porosity, "m3 m-3",
        strictly_positive=True, expected_size=n,
    )
    source = _as_cell_array(
        "source_bulk_bq_m3_s", source_bulk_bq_m3_s, "Bq m-3 s-1",
        strictly_positive=False, expected_size=n,
    )
    d_bulk = _as_cell_array(
        "d_bulk_m2_s", d_bulk_m2_s, "m2 s-1", strictly_positive=True, expected_size=n
    )
    dz = _require_positive("dz_m", dz_m, "m")
    lam = _require_positive("lambda_s", lambda_s, "s-1")
    surface_condition = DEFAULT_SURFACE_CONDITION if surface is None else surface
    base_condition = DEFAULT_BASE_CONDITION if base is None else base

    production = _require_finite_result("production", float(np.sum(source) * dz))
    decay = _require_finite_result("decay", float(np.sum(lam * beta * profile) * dz))
    surface_flux = surface_face_flux(
        surface_condition, float(d_bulk[0]), float(profile[0]), dz
    )
    base_flux = base_face_flux(base_condition, float(d_bulk[-1]), float(profile[-1]), dz)

    residual = _require_finite_result(
        "balance residual", production + base_flux - decay - surface_flux
    )
    scale = max(abs(production), abs(decay), abs(surface_flux), abs(base_flux))
    relative = 0.0 if scale == 0.0 else residual / scale
    return {
        "production_bq_m2_s": production,
        "decay_bq_m2_s": decay,
        "surface_flux_bq_m2_s": float(surface_flux),
        "base_flux_bq_m2_s": float(base_flux),
        "residual_bq_m2_s": residual,
        "relative_residual": float(relative),
    }


def _as_series(
    name: str, values: ArrayLike, expected_size: int | None = None
) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.ndim} dimensions")
    if array.size < 2:
        raise ValueError(f"{name} must have at least 2 samples, got {array.size}")
    if expected_size is not None and array.size != expected_size:
        raise ValueError(f"{name} must have {expected_size} samples, got {array.size}")
    if not np.all(np.isfinite(array)):
        bad = int(np.argmin(np.isfinite(array)))
        raise ValueError(f"{name} must be finite everywhere, entry {bad} is {array[bad]!r}")
    return array


def _time_weighted_mean(times_s: NDArray[np.float64], values: NDArray[np.float64]) -> float:
    widths = times_s[1:] - times_s[:-1]
    integral = float(np.sum(widths * (values[1:] + values[:-1]) / 2.0))
    return integral / float(times_s[-1] - times_s[0])


def exhalation_plateau(
    times_s: ArrayLike,
    flux_bq_m2_s: ArrayLike,
    window_s: float,
    relative_span: float = PLATEAU_RELATIVE_SPAN,
) -> float:
    times = _as_series("times_s", times_s)
    flux = _as_series("flux_bq_m2_s", flux_bq_m2_s, expected_size=times.size)
    if not np.all(times[1:] > times[:-1]):
        bad = int(np.argmin(times[1:] > times[:-1]))
        raise ValueError(
            f"times_s must increase strictly, entry {bad + 1} is {times[bad + 1]!r} "
            f"after {times[bad]!r}"
        )
    _require_positive("window_s", window_s, "s")
    _require_finite("relative_span", relative_span)
    if relative_span < 0.0:
        raise ValueError(f"relative_span must be >= 0, got {relative_span!r}")

    duration = float(times[-1] - times[0])
    if window_s > duration:
        raise ValueError(
            f"window_s must not exceed the span of times_s, {duration!r} s, "
            f"got {window_s!r}"
        )

    inside = times >= times[-1] - float(window_s)
    held = int(np.count_nonzero(inside))
    if held < 2:
        raise ValueError(
            f"the trailing window of {float(window_s)!r} s holds {held} of the "
            f"{times.size} samples; a plateau is read on at least 2"
        )
    window_times = times[inside]
    window_flux = flux[inside]

    plateau = _require_finite_result(
        "exhalation plateau", _time_weighted_mean(window_times, window_flux)
    )
    span = float(np.max(window_flux) - np.min(window_flux))
    limit = float(relative_span) * abs(plateau)
    if span > limit:
        ratio = span / abs(plateau) if plateau != 0.0 else math.inf
        raise ValueError(
            f"no plateau on the trailing {float(window_s)!r} s: the flux spans "
            f"{span!r} Bq m-2 s-1 about a mean of {plateau!r}, that is {ratio!r} in "
            f"relative terms, above the declared relative_span "
            f"{float(relative_span)!r}; the series is still transient over that window"
        )
    return plateau
