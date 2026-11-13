from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radon_x.core.boundaries import (
    BoundaryCondition,
    Dirichlet,
    ZeroFlux,
)
from radon_x.core.decay import LAMBDA_RN222
from radon_x.errors import NumericalError
from radon_x.numerics.tridiagonal import (
    assemble_block,
    solve_banded_system,
    tridiagonal_matvec,
)

__all__ = [
    "DEFAULT_BASE_CONDITION",
    "DEFAULT_SURFACE_CONDITION",
    "MAX_STEP_HALVINGS",
    "MINIMUM_CELL_COUNT",
    "NEGATIVE_TOLERANCE_ABSOLUTE_BQ_M3",
    "NEGATIVE_TOLERANCE_RELATIVE",
    "ColumnSolution",
    "MassBalance",
    "NegativityWarning",
    "cell_centres",
    "negative_tolerance_bq_m3",
    "solve_steady",
    "solve_transient",
]

MINIMUM_CELL_COUNT: int = 3

MAX_STEP_HALVINGS: int = 12

NEGATIVE_TOLERANCE_RELATIVE: float = 1e-12

NEGATIVE_TOLERANCE_ABSOLUTE_BQ_M3: float = 1e-9

DEFAULT_SURFACE_CONDITION: BoundaryCondition = Dirichlet(0.0)

DEFAULT_BASE_CONDITION: BoundaryCondition = ZeroFlux()


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


def _require_finite_array(name: str, array: NDArray[np.float64]) -> NDArray[np.float64]:
    if not np.all(np.isfinite(array)):
        bad = int(np.argmin(np.isfinite(array)))
        raise NumericalError(
            f"{name} is not finite: entry {bad} is {array[bad]!r}; "
            "check the magnitudes of the coefficients and of the time step"
        )
    return array


def negative_tolerance_bq_m3(scale_bq_m3: float) -> float:
    scale = abs(float(scale_bq_m3))
    if not np.isfinite(scale):
        return NEGATIVE_TOLERANCE_ABSOLUTE_BQ_M3
    return max(NEGATIVE_TOLERANCE_RELATIVE * scale, NEGATIVE_TOLERANCE_ABSOLUTE_BQ_M3)


@dataclass(frozen=True)
class NegativityWarning:

    step_index: int
    halvings: int
    time_step_s: float
    minimum_bq_m3: float
    tolerance_bq_m3: float


@dataclass(frozen=True)
class MassBalance:

    production_bq_m2_s: float
    decay_bq_m2_s: float
    surface_flux_bq_m2_s: float
    base_flux_bq_m2_s: float
    storage_rate_bq_m2_s: float
    residual_bq_m2_s: float
    relative_residual: float


@dataclass(frozen=True)
class ColumnSolution:

    concentration_bq_m3: NDArray[np.float64]
    depth_m: NDArray[np.float64]
    cell_count: int
    cell_size_m: float
    column_thickness_m: float
    step_count: int
    time_step_s: float | None
    elapsed_time_s: float | None
    theta: float | None
    max_cell_fourier_number: float | None
    minimum_concentration_bq_m3: float
    surface_flux_bq_m2_s: float
    base_flux_bq_m2_s: float
    balance: MassBalance
    step_halvings: int = 0
    halved_step_count: int = 0
    warnings: tuple[NegativityWarning, ...] = ()


@dataclass(frozen=True)
class _Assembly:

    lower: NDArray[np.float64]
    diag: NDArray[np.float64]
    upper: NDArray[np.float64]
    rhs: NDArray[np.float64]
    beta: NDArray[np.float64]
    source_bulk_bq_m3_s: NDArray[np.float64]
    d_bulk_m2_s: NDArray[np.float64]
    d_face_interior_m2_s: NDArray[np.float64]
    surface_diagonal_m2_s: float
    surface_source_bq_m_s: float
    base_diagonal_m2_s: float
    base_source_bq_m_s: float
    dz_m: float
    lambda_s: float


def cell_centres(cell_count: int, dz_m: float) -> NDArray[np.float64]:
    if not isinstance(cell_count, (int, np.integer)) or isinstance(cell_count, bool):
        raise TypeError(f"cell_count must be an integer, got {cell_count!r}")
    if cell_count < MINIMUM_CELL_COUNT:
        raise ValueError(f"cell_count must be >= {MINIMUM_CELL_COUNT}, got {cell_count!r}")
    _require_positive("dz_m", dz_m, "m")
    return (np.arange(int(cell_count), dtype=float) + 0.5) * float(dz_m)


def _assemble(
    d_bulk_m2_s: NDArray[np.float64],
    beta: NDArray[np.float64],
    source_bulk_bq_m3_s: NDArray[np.float64],
    dz_m: float,
    lambda_s: float,
    surface: BoundaryCondition,
    base: BoundaryCondition,
) -> _Assembly:
    bands = assemble_block(
        d_bulk_m2_s=d_bulk_m2_s,
        beta=beta,
        dz_m=dz_m,
        lambda_s=lambda_s,
        surface=surface,
        base=base,
    )
    return _Assembly(
        lower=bands.lower,
        diag=bands.diag,
        upper=bands.upper,
        rhs=bands.load_vector(source_bulk_bq_m3_s),
        beta=beta,
        source_bulk_bq_m3_s=source_bulk_bq_m3_s,
        d_bulk_m2_s=d_bulk_m2_s,
        d_face_interior_m2_s=bands.d_face_interior_m2_s,
        surface_diagonal_m2_s=bands.surface_diagonal_m2_s,
        surface_source_bq_m_s=bands.surface_source_bq_m_s,
        base_diagonal_m2_s=bands.base_diagonal_m2_s,
        base_source_bq_m_s=bands.base_source_bq_m_s,
        dz_m=dz_m,
        lambda_s=lambda_s,
    )


def _solve_tridiagonal(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    return solve_banded_system(lower, diag, upper, rhs)


def _tridiagonal_matvec(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    return tridiagonal_matvec(lower, diag, upper, vector)


def _surface_flux(assembly: _Assembly, profile: NDArray[np.float64]) -> float:
    return float(
        (assembly.surface_diagonal_m2_s * profile[0] - assembly.surface_source_bq_m_s)
        / assembly.dz_m
    )


def _base_flux(assembly: _Assembly, profile: NDArray[np.float64]) -> float:
    return float(
        (assembly.base_source_bq_m_s - assembly.base_diagonal_m2_s * profile[-1]) / assembly.dz_m
    )


def _mass_balance(
    assembly: _Assembly,
    profile_for_fluxes: NDArray[np.float64],
    storage_rate_bq_m2_s: float,
) -> MassBalance:
    dz = assembly.dz_m
    production = float(np.sum(assembly.source_bulk_bq_m3_s) * dz)
    decay = float(np.sum(assembly.lambda_s * assembly.beta * profile_for_fluxes) * dz)
    surface_flux = _surface_flux(assembly, profile_for_fluxes)
    base_flux = _base_flux(assembly, profile_for_fluxes)
    residual = production + base_flux - decay - surface_flux - storage_rate_bq_m2_s
    scale = max(
        abs(production), abs(decay), abs(surface_flux), abs(base_flux), abs(storage_rate_bq_m2_s)
    )
    relative = 0.0 if scale == 0.0 else residual / scale
    return MassBalance(
        production_bq_m2_s=production,
        decay_bq_m2_s=decay,
        surface_flux_bq_m2_s=surface_flux,
        base_flux_bq_m2_s=base_flux,
        storage_rate_bq_m2_s=storage_rate_bq_m2_s,
        residual_bq_m2_s=residual,
        relative_residual=float(relative),
    )


def _validated_inputs(
    d_bulk_m2_s: ArrayLike,
    beta: ArrayLike,
    source_bulk_bq_m3_s: ArrayLike,
    dz_m: float,
    lambda_s: float,
    surface: BoundaryCondition | None,
    base: BoundaryCondition | None,
) -> _Assembly:
    d_bulk = _as_cell_array("d_bulk_m2_s", d_bulk_m2_s, "m2 s-1", strictly_positive=True)
    n = d_bulk.size
    beta_array = _as_cell_array(
        "beta", beta, "m3 m-3", strictly_positive=True, expected_size=n
    )
    source = _as_cell_array(
        "source_bulk_bq_m3_s",
        source_bulk_bq_m3_s,
        "Bq m-3 s-1",
        strictly_positive=False,
        expected_size=n,
    )
    _require_positive("dz_m", dz_m, "m")
    _require_positive("lambda_s", lambda_s, "s-1")
    resolved_surface = DEFAULT_SURFACE_CONDITION if surface is None else surface
    resolved_base = DEFAULT_BASE_CONDITION if base is None else base
    return _assemble(
        d_bulk_m2_s=d_bulk,
        beta=beta_array,
        source_bulk_bq_m3_s=source,
        dz_m=float(dz_m),
        lambda_s=float(lambda_s),
        surface=resolved_surface,
        base=resolved_base,
    )


def solve_steady(
    d_bulk_m2_s: ArrayLike,
    beta: ArrayLike,
    source_bulk_bq_m3_s: ArrayLike,
    dz_m: float,
    lambda_s: float = LAMBDA_RN222,
    surface: BoundaryCondition | None = None,
    base: BoundaryCondition | None = None,
) -> ColumnSolution:
    assembly = _validated_inputs(
        d_bulk_m2_s, beta, source_bulk_bq_m3_s, dz_m, lambda_s, surface, base
    )
    profile = _solve_tridiagonal(assembly.lower, assembly.diag, assembly.upper, assembly.rhs)
    n = profile.size
    return ColumnSolution(
        concentration_bq_m3=profile,
        depth_m=cell_centres(n, assembly.dz_m),
        cell_count=n,
        cell_size_m=assembly.dz_m,
        column_thickness_m=n * assembly.dz_m,
        step_count=0,
        time_step_s=None,
        elapsed_time_s=None,
        theta=None,
        max_cell_fourier_number=None,
        minimum_concentration_bq_m3=float(np.min(profile)),
        surface_flux_bq_m2_s=_surface_flux(assembly, profile),
        base_flux_bq_m2_s=_base_flux(assembly, profile),
        balance=_mass_balance(assembly, profile, 0.0),
    )


def _theta_step(
    assembly: _Assembly,
    previous: NDArray[np.float64],
    capacity: NDArray[np.float64],
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    theta_value: float,
) -> NDArray[np.float64]:
    explicit = _tridiagonal_matvec(assembly.lower, assembly.diag, assembly.upper, previous.copy())
    rhs = capacity * previous - (1.0 - theta_value) * explicit + assembly.rhs
    return _solve_tridiagonal(lower, diag, upper, rhs)


def _halved_retry(
    assembly: _Assembly,
    start: NDArray[np.float64],
    candidate: NDArray[np.float64],
    candidate_minimum: float,
    dt: float,
    theta_value: float,
    tolerance: float,
    halvings_allowed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int, float]:
    best = candidate
    best_start = start
    best_sub_dt = dt
    best_minimum = candidate_minimum
    for level in range(1, halvings_allowed + 1):
        substeps = 2**level
        sub_dt = dt / substeps
        sub_capacity = assembly.beta / sub_dt
        sub_lower = theta_value * assembly.lower
        sub_diag = sub_capacity + theta_value * assembly.diag
        sub_upper = theta_value * assembly.upper
        trial = start
        trial_previous = start
        trial_minimum = float("inf")
        for _ in range(substeps):
            trial_previous = trial
            trial = _theta_step(
                assembly, trial_previous, sub_capacity, sub_lower, sub_diag, sub_upper, theta_value
            )
            trial_minimum = min(trial_minimum, float(np.min(trial)))
        best, best_start, best_sub_dt, best_minimum = (
            trial,
            trial_previous,
            sub_dt,
            trial_minimum,
        )
        if trial_minimum >= -tolerance:
            return trial, trial_previous, sub_dt, level, trial_minimum
    return best, best_start, best_sub_dt, halvings_allowed, best_minimum


def solve_transient(
    d_bulk_m2_s: ArrayLike,
    beta: ArrayLike,
    source_bulk_bq_m3_s: ArrayLike,
    dz_m: float,
    initial_bq_m3: ArrayLike,
    time_step_s: float,
    step_count: int,
    lambda_s: float = LAMBDA_RN222,
    surface: BoundaryCondition | None = None,
    base: BoundaryCondition | None = None,
    theta: float = 1.0,
    max_step_halvings: int = MAX_STEP_HALVINGS,
) -> ColumnSolution:
    assembly = _validated_inputs(
        d_bulk_m2_s, beta, source_bulk_bq_m3_s, dz_m, lambda_s, surface, base
    )
    n = assembly.beta.size
    profile = _as_cell_array(
        "initial_bq_m3", initial_bq_m3, "Bq m-3", strictly_positive=False, expected_size=n
    )
    _require_positive("time_step_s", time_step_s, "s")
    _require_finite("theta", theta)
    if not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta must lie in [0, 1], got {theta!r}")
    if not isinstance(step_count, (int, np.integer)) or isinstance(step_count, bool):
        raise TypeError(f"step_count must be an integer, got {step_count!r}")
    if step_count < 1:
        raise ValueError(f"step_count must be >= 1, got {step_count!r}")
    if not isinstance(max_step_halvings, (int, np.integer)) or isinstance(
        max_step_halvings, bool
    ):
        raise TypeError(f"max_step_halvings must be an integer, got {max_step_halvings!r}")
    if not 0 <= max_step_halvings <= MAX_STEP_HALVINGS:
        raise ValueError(
            f"max_step_halvings must lie in [0, {MAX_STEP_HALVINGS}], got {max_step_halvings!r}"
        )

    dt = float(time_step_s)
    theta_value = float(theta)
    halvings_allowed = int(max_step_halvings)
    capacity = assembly.beta / dt
    step_lower = theta_value * assembly.lower
    step_diag = capacity + theta_value * assembly.diag
    step_upper = theta_value * assembly.upper

    deepest_halving = 0
    halved_steps = 0
    emitted: list[NegativityWarning] = []

    current = profile
    balance_start = profile
    balance_dt = dt
    for index in range(int(step_count)):
        previous = current
        current = _theta_step(
            assembly, previous, capacity, step_lower, step_diag, step_upper, theta_value
        )
        balance_start = previous
        balance_dt = dt
        if theta_value >= 1.0:
            continue
        minimum = float(np.min(current))
        tolerance = negative_tolerance_bq_m3(float(np.max(np.abs(previous))))
        if minimum >= -tolerance:
            continue
        current, balance_start, balance_dt, level, trajectory_minimum = _halved_retry(
            assembly,
            previous,
            current,
            minimum,
            dt,
            theta_value,
            tolerance,
            halvings_allowed,
        )
        if level > 0:
            halved_steps += 1
            deepest_halving = max(deepest_halving, level)
        if trajectory_minimum < -tolerance:
            emitted.append(
                NegativityWarning(
                    step_index=index + 1,
                    halvings=level,
                    time_step_s=dt / 2**level,
                    minimum_bq_m3=trajectory_minimum,
                    tolerance_bq_m3=tolerance,
                )
            )

    final = current
    weighted = theta_value * final + (1.0 - theta_value) * balance_start
    storage_rate = float(
        np.sum(assembly.beta * (final - balance_start)) * assembly.dz_m / balance_dt
    )

    d_face_max = float(
        max(
            np.max(assembly.d_face_interior_m2_s),
            assembly.d_bulk_m2_s[0],
            assembly.d_bulk_m2_s[-1],
        )
    )
    fourier = d_face_max * dt / (float(np.min(assembly.beta)) * assembly.dz_m * assembly.dz_m)

    return ColumnSolution(
        concentration_bq_m3=final,
        depth_m=cell_centres(n, assembly.dz_m),
        cell_count=n,
        cell_size_m=assembly.dz_m,
        column_thickness_m=n * assembly.dz_m,
        step_count=int(step_count),
        time_step_s=dt,
        elapsed_time_s=dt * int(step_count),
        theta=theta_value,
        max_cell_fourier_number=fourier,
        minimum_concentration_bq_m3=float(np.min(final)),
        surface_flux_bq_m2_s=_surface_flux(assembly, final),
        base_flux_bq_m2_s=_base_flux(assembly, final),
        balance=_mass_balance(assembly, weighted, storage_rate),
        step_halvings=deepest_halving,
        halved_step_count=halved_steps,
        warnings=tuple(emitted),
    )
