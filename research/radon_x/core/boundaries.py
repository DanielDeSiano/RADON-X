from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

__all__ = [
    "BoundaryCoefficients",
    "BoundaryCondition",
    "Dirichlet",
    "ZeroFlux",
    "base_dirichlet_ghost_value",
    "base_face_flux",
    "base_zero_flux_ghost_value",
    "boundary_coefficients",
    "ghost_value",
    "surface_dirichlet_ghost_value",
    "surface_face_flux",
]


def _require_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _require_positive(name: str, value: float, unit: str) -> float:
    _require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0 {unit}, got {value!r}")
    return value


def _require_non_negative(name: str, value: float, unit: str) -> float:
    _require_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0 {unit}, got {value!r}")
    return value


def _require_finite_result(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} overflowed to {value!r}; check the magnitudes of the inputs")
    return value


@dataclass(frozen=True)
class Dirichlet:

    value_bq_m3: float

    def __post_init__(self) -> None:
        _require_non_negative("value_bq_m3", self.value_bq_m3, "Bq m-3")


@dataclass(frozen=True)
class ZeroFlux:
    pass


BoundaryCondition = Union[Dirichlet, ZeroFlux]


@dataclass(frozen=True)
class BoundaryCoefficients:

    diagonal_m2_s: float
    source_bq_m_s: float


def surface_dirichlet_ghost_value(
    c_boundary_bq_m3: float,
    c_first_cell_bq_m3: float,
) -> float:
    _require_non_negative("c_boundary_bq_m3", c_boundary_bq_m3, "Bq m-3")
    _require_non_negative("c_first_cell_bq_m3", c_first_cell_bq_m3, "Bq m-3")
    return 2.0 * c_boundary_bq_m3 - c_first_cell_bq_m3


def base_dirichlet_ghost_value(
    c_boundary_bq_m3: float,
    c_last_cell_bq_m3: float,
) -> float:
    _require_non_negative("c_boundary_bq_m3", c_boundary_bq_m3, "Bq m-3")
    _require_non_negative("c_last_cell_bq_m3", c_last_cell_bq_m3, "Bq m-3")
    return 2.0 * c_boundary_bq_m3 - c_last_cell_bq_m3


def base_zero_flux_ghost_value(c_last_cell_bq_m3: float) -> float:
    _require_non_negative("c_last_cell_bq_m3", c_last_cell_bq_m3, "Bq m-3")
    return c_last_cell_bq_m3


def ghost_value(condition: BoundaryCondition, c_adjacent_cell_bq_m3: float) -> float:
    if isinstance(condition, Dirichlet):
        _require_non_negative("c_adjacent_cell_bq_m3", c_adjacent_cell_bq_m3, "Bq m-3")
        return 2.0 * condition.value_bq_m3 - c_adjacent_cell_bq_m3
    if isinstance(condition, ZeroFlux):
        return base_zero_flux_ghost_value(c_adjacent_cell_bq_m3)
    raise TypeError(f"condition must be Dirichlet or ZeroFlux, got {type(condition).__name__}")


def _half_cell_flux(
    d_bulk_face_m2_s: float,
    c_deeper_bq_m3: float,
    c_shallower_bq_m3: float,
    dz_m: float,
) -> float:
    _require_positive("d_bulk_face_m2_s", d_bulk_face_m2_s, "m2 s-1")
    _require_non_negative("c_deeper_bq_m3", c_deeper_bq_m3, "Bq m-3")
    _require_non_negative("c_shallower_bq_m3", c_shallower_bq_m3, "Bq m-3")
    _require_positive("dz_m", dz_m, "m")
    flux = d_bulk_face_m2_s * (c_deeper_bq_m3 - c_shallower_bq_m3) / (0.5 * dz_m)
    return _require_finite_result("boundary face flux", flux)


def surface_face_flux(
    condition: BoundaryCondition,
    d_bulk_face_m2_s: float,
    c_first_cell_bq_m3: float,
    dz_m: float,
) -> float:
    if isinstance(condition, Dirichlet):
        return _half_cell_flux(
            d_bulk_face_m2_s,
            c_first_cell_bq_m3,
            condition.value_bq_m3,
            dz_m,
        )
    if isinstance(condition, ZeroFlux):
        _require_positive("d_bulk_face_m2_s", d_bulk_face_m2_s, "m2 s-1")
        _require_non_negative("c_first_cell_bq_m3", c_first_cell_bq_m3, "Bq m-3")
        _require_positive("dz_m", dz_m, "m")
        return 0.0
    raise TypeError(f"condition must be Dirichlet or ZeroFlux, got {type(condition).__name__}")


def base_face_flux(
    condition: BoundaryCondition,
    d_bulk_face_m2_s: float,
    c_last_cell_bq_m3: float,
    dz_m: float,
) -> float:
    if isinstance(condition, Dirichlet):
        return _half_cell_flux(
            d_bulk_face_m2_s,
            condition.value_bq_m3,
            c_last_cell_bq_m3,
            dz_m,
        )
    if isinstance(condition, ZeroFlux):
        _require_positive("d_bulk_face_m2_s", d_bulk_face_m2_s, "m2 s-1")
        _require_non_negative("c_last_cell_bq_m3", c_last_cell_bq_m3, "Bq m-3")
        _require_positive("dz_m", dz_m, "m")
        return 0.0
    raise TypeError(f"condition must be Dirichlet or ZeroFlux, got {type(condition).__name__}")


def boundary_coefficients(
    condition: BoundaryCondition,
    d_bulk_face_m2_s: float,
) -> BoundaryCoefficients:
    _require_positive("d_bulk_face_m2_s", d_bulk_face_m2_s, "m2 s-1")
    if isinstance(condition, Dirichlet):
        return BoundaryCoefficients(
            diagonal_m2_s=2.0 * d_bulk_face_m2_s,
            source_bq_m_s=2.0 * d_bulk_face_m2_s * condition.value_bq_m3,
        )
    if isinstance(condition, ZeroFlux):
        return BoundaryCoefficients(diagonal_m2_s=0.0, source_bq_m_s=0.0)
    raise TypeError(f"condition must be Dirichlet or ZeroFlux, got {type(condition).__name__}")
