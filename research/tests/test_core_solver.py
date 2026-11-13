from __future__ import annotations

import importlib.util
import os
import math
import sys
from typing import Any

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.special import erf, erfc

from radon_x.core.analytic import (
    steady_profile_finite_slab,
    steady_profile_semi_infinite,
    surface_flux_finite_slab,
    surface_flux_semi_infinite,
)
from radon_x.core.boundaries import Dirichlet, ZeroFlux, base_face_flux, surface_face_flux
from radon_x.core.decay import LAMBDA_RN222, diffusion_length
from radon_x.core.solver import (
    DEFAULT_BASE_CONDITION,
    DEFAULT_SURFACE_CONDITION,
    MINIMUM_CELL_COUNT,
    ColumnSolution,
    MassBalance,
    cell_centres,
    solve_steady,
    solve_transient,
)
from radon_x.core.sources import (
    equilibrium_concentration,
    production_ceiling_sealed_bed,
    radon_production_bulk,
)

from radon_x.legacy_adapter import legacy_source_path

LEGACY_MODEL_PATH = str(legacy_source_path())

RA226_BQ_KG = 40.0
RHO_B_KG_M3 = 1600.0
EPS = 0.25
D_A_M2_S = 4.7213e-06
BETA = 0.30
D_E_M2_S = BETA * D_A_M2_S
S_BULK_BQ_M3_S = radon_production_bulk(RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222)
C_EQ_BQ_M3 = equilibrium_concentration(S_BULK_BQ_M3_S, LAMBDA_RN222, BETA)
L_D_M = diffusion_length(D_A_M2_S, LAMBDA_RN222)
THICKNESS_M = 3.0

RB1_THICKNESS_M = 0.060
RB1_RA226_BQ_KG = 451.0
RB1_RHO_B_KG_M3 = 1080.0
RB1_EPS = 0.20
RB1_BETA_ASSUMED = 0.55
RB1_D_A_ASSUMED_M2_S = 3.0e-06
RB1_D_E_ASSUMED_M2_S = RB1_BETA_ASSUMED * RB1_D_A_ASSUMED_M2_S
RB1_S_BULK_BQ_M3_S = radon_production_bulk(
    RB1_RA226_BQ_KG, RB1_RHO_B_KG_M3, RB1_EPS, LAMBDA_RN222
)


def _load_legacy() -> Any:
    if not os.path.isfile(LEGACY_MODEL_PATH):
        pytest.skip(f"legacy model not found at {LEGACY_MODEL_PATH}")
    spec = importlib.util.spec_from_file_location("legacy_for_solver_tests", LEGACY_MODEL_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"legacy model not importable from {LEGACY_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    saved_argv = sys.argv
    sys.argv = ["x"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved_argv
    return module


@pytest.fixture(scope="module")
def legacy() -> Any:
    return _load_legacy()


def _uniform(value: float, cell_count: int) -> np.ndarray:
    return np.full(cell_count, value, dtype=float)


def _reference_operator(
    d_bulk: np.ndarray,
    beta: np.ndarray,
    source: np.ndarray,
    dz_m: float,
    lambda_s: float,
    c_top_bq_m3: float | None,
    c_bot_bq_m3: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    n = d_bulk.size
    inv_dz2 = 1.0 / (dz_m * dz_m)
    operator = np.zeros((n, n), dtype=float)
    load = np.array(source, dtype=float)
    for face in range(n - 1):
        left = d_bulk[face]
        right = d_bulk[face + 1]
        conductance = 2.0 * left * right / (left + right) * inv_dz2
        operator[face, face] += conductance
        operator[face, face + 1] -= conductance
        operator[face + 1, face + 1] += conductance
        operator[face + 1, face] -= conductance
    if c_top_bq_m3 is not None:
        operator[0, 0] += 2.0 * d_bulk[0] * inv_dz2
        load[0] += 2.0 * d_bulk[0] * c_top_bq_m3 * inv_dz2
    if c_bot_bq_m3 is not None:
        operator[n - 1, n - 1] += 2.0 * d_bulk[-1] * inv_dz2
        load[n - 1] += 2.0 * d_bulk[-1] * c_bot_bq_m3 * inv_dz2
    operator += np.diag(lambda_s * beta)
    return operator, load


def _analytic_slab_profile(depths_m: np.ndarray) -> np.ndarray:
    return np.array(
        [steady_profile_finite_slab(z, C_EQ_BQ_M3, L_D_M, THICKNESS_M) for z in depths_m]
    )


STEP_THICKNESS_M = 6.0
STEP_HORIZON_S = 86400.0
STEP_SURFACE_BQ_M3 = 1.0e5


def _step_response_semi_infinite(depth_m: np.ndarray, time_s: float) -> np.ndarray:
    rate = math.sqrt(LAMBDA_RN222 * time_s)
    spread = 2.0 * math.sqrt(D_A_M2_S * time_s)
    return 0.5 * STEP_SURFACE_BQ_M3 * (
        np.exp(-depth_m / L_D_M) * erfc(depth_m / spread - rate)
        + np.exp(depth_m / L_D_M) * erfc(depth_m / spread + rate)
    )


def _step_response_surface_flux(time_s: float) -> float:
    rate = math.sqrt(LAMBDA_RN222 * time_s)
    return -STEP_SURFACE_BQ_M3 * D_E_M2_S * (
        erf(rate) / L_D_M + math.exp(-LAMBDA_RN222 * time_s) / math.sqrt(math.pi * D_A_M2_S * time_s)
    )


def _observed_orders(errors: list[float]) -> list[float]:
    return [errors[k] / errors[k + 1] for k in range(len(errors) - 1)]


def _format_table(header: str, rows: list[tuple[Any, ...]]) -> str:
    return header + "\n" + "\n".join("  ".join(str(cell) for cell in row) for row in rows)


def test_steady_solution_satisfies_the_declared_discrete_operator() -> None:
    cell_count = 40
    dz = THICKNESS_M / cell_count
    generator = np.random.default_rng(7)
    d_bulk = D_E_M2_S * np.exp(generator.normal(0.0, 0.8, cell_count))
    beta = BETA * np.exp(generator.normal(0.0, 0.3, cell_count))
    source = S_BULK_BQ_M3_S * (1.0 + 0.5 * generator.random(cell_count))

    solution = solve_steady(d_bulk, beta, source, dz, LAMBDA_RN222)
    operator, load = _reference_operator(
        d_bulk, beta, source, dz, LAMBDA_RN222, c_top_bq_m3=0.0, c_bot_bq_m3=None
    )
    expected = np.linalg.solve(operator, load)

    assert np.allclose(solution.concentration_bq_m3, expected, rtol=1e-11, atol=0.0)


def test_dirichlet_base_enters_the_assembly_with_the_half_cell_factor() -> None:
    cell_count = 25
    dz = THICKNESS_M / cell_count
    d_bulk = _uniform(D_E_M2_S, cell_count)
    beta = _uniform(BETA, cell_count)
    source = _uniform(S_BULK_BQ_M3_S, cell_count)

    solution = solve_steady(
        d_bulk, beta, source, dz, LAMBDA_RN222, base=Dirichlet(C_EQ_BQ_M3)
    )
    operator, load = _reference_operator(
        d_bulk, beta, source, dz, LAMBDA_RN222, c_top_bq_m3=0.0, c_bot_bq_m3=C_EQ_BQ_M3
    )
    expected = np.linalg.solve(operator, load)

    assert np.allclose(solution.concentration_bq_m3, expected, rtol=1e-11, atol=0.0)


def test_cell_centres_are_the_declared_grid() -> None:
    centres = cell_centres(6, 0.5)
    assert np.allclose(centres, [0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    assert centres[0] > 0.0
    assert centres[-1] < 6 * 0.5


def test_default_boundary_conditions_are_the_minimal_core_ones() -> None:
    assert DEFAULT_SURFACE_CONDITION == Dirichlet(0.0)
    assert DEFAULT_BASE_CONDITION == ZeroFlux()
    cell_count = 20
    dz = THICKNESS_M / cell_count
    explicit = solve_steady(
        _uniform(D_E_M2_S, cell_count),
        _uniform(BETA, cell_count),
        _uniform(S_BULK_BQ_M3_S, cell_count),
        dz,
        LAMBDA_RN222,
        surface=Dirichlet(0.0),
        base=ZeroFlux(),
    )
    defaulted = solve_steady(
        _uniform(D_E_M2_S, cell_count),
        _uniform(BETA, cell_count),
        _uniform(S_BULK_BQ_M3_S, cell_count),
        dz,
        LAMBDA_RN222,
    )
    assert np.array_equal(explicit.concentration_bq_m3, defaulted.concentration_bq_m3)


def test_inputs_are_copied_not_aliased() -> None:
    cell_count = 20
    dz = THICKNESS_M / cell_count
    d_bulk = _uniform(D_E_M2_S, cell_count)
    solution = solve_steady(
        d_bulk,
        _uniform(BETA, cell_count),
        _uniform(S_BULK_BQ_M3_S, cell_count),
        dz,
        LAMBDA_RN222,
    )
    before = solution.concentration_bq_m3.copy()
    d_bulk *= 1000.0
    assert np.array_equal(solution.concentration_bq_m3, before)


def test_steady_reproduces_analytic_finite_slab_on_a_fine_grid() -> None:
    cell_count = 600
    dz = THICKNESS_M / cell_count
    solution = solve_steady(
        _uniform(D_E_M2_S, cell_count),
        _uniform(BETA, cell_count),
        _uniform(S_BULK_BQ_M3_S, cell_count),
        dz,
        LAMBDA_RN222,
    )
    expected = _analytic_slab_profile(solution.depth_m)
    relative = np.max(np.abs(solution.concentration_bq_m3 - expected)) / C_EQ_BQ_M3
    assert relative < 1e-4


def test_steady_surface_flux_reproduces_analytic_finite_slab() -> None:
    cell_count = 600
    dz = THICKNESS_M / cell_count
    solution = solve_steady(
        _uniform(D_E_M2_S, cell_count),
        _uniform(BETA, cell_count),
        _uniform(S_BULK_BQ_M3_S, cell_count),
        dz,
        LAMBDA_RN222,
    )
    expected = surface_flux_finite_slab(
        RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S, THICKNESS_M
    )
    assert solution.surface_flux_bq_m2_s == pytest.approx(expected, rel=1e-4)
    assert solution.base_flux_bq_m2_s == 0.0


def test_deep_column_with_equilibrium_base_reproduces_the_semi_infinite_solution() -> None:
    thickness = 12.0 * L_D_M
    cell_count = 800
    dz = thickness / cell_count
    solution = solve_steady(
        _uniform(D_E_M2_S, cell_count),
        _uniform(BETA, cell_count),
        _uniform(S_BULK_BQ_M3_S, cell_count),
        dz,
        LAMBDA_RN222,
        base=Dirichlet(C_EQ_BQ_M3),
    )
    expected = np.array(
        [steady_profile_semi_infinite(z, C_EQ_BQ_M3, L_D_M) for z in solution.depth_m]
    )
    relative = np.max(np.abs(solution.concentration_bq_m3 - expected)) / C_EQ_BQ_M3
    assert relative < 1e-4

    expected_flux = surface_flux_semi_infinite(
        RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S
    )
    assert solution.surface_flux_bq_m2_s == pytest.approx(expected_flux, rel=1e-4)

    expected_base = D_E_M2_S * C_EQ_BQ_M3 / (L_D_M * math.sinh(thickness / L_D_M))
    assert solution.base_flux_bq_m2_s == pytest.approx(expected_base, rel=1e-3)
    assert solution.base_flux_bq_m2_s < 1e-4 * solution.surface_flux_bq_m2_s


def test_boundary_fluxes_agree_with_the_boundaries_module() -> None:
    cell_count = 40
    dz = THICKNESS_M / cell_count
    for base in (ZeroFlux(), Dirichlet(C_EQ_BQ_M3)):
        solution = solve_steady(
            _uniform(D_E_M2_S, cell_count),
            _uniform(BETA, cell_count),
            _uniform(S_BULK_BQ_M3_S, cell_count),
            dz,
            LAMBDA_RN222,
            surface=Dirichlet(0.0),
            base=base,
        )
        expected_surface = surface_face_flux(
            Dirichlet(0.0), D_E_M2_S, float(solution.concentration_bq_m3[0]), dz
        )
        expected_base = base_face_flux(
            base, D_E_M2_S, float(solution.concentration_bq_m3[-1]), dz
        )
        assert solution.surface_flux_bq_m2_s == pytest.approx(expected_surface, rel=1e-13)
        assert solution.base_flux_bq_m2_s == pytest.approx(expected_base, rel=1e-13, abs=1e-30)


def test_transient_reproduces_the_erf_step_response_of_the_half_space() -> None:
    horizon = STEP_HORIZON_S
    errors: list[float] = []
    rows: list[tuple[Any, ...]] = []
    finest: ColumnSolution | None = None
    for cell_count, steps in [(150, 240), (300, 480), (600, 960), (1200, 1920)]:
        dz = STEP_THICKNESS_M / cell_count
        solution = solve_transient(
            _uniform(D_E_M2_S, cell_count),
            _uniform(BETA, cell_count),
            np.zeros(cell_count),
            dz,
            np.zeros(cell_count),
            horizon / steps,
            steps,
            LAMBDA_RN222,
            surface=Dirichlet(STEP_SURFACE_BQ_M3),
            base=ZeroFlux(),
        )
        exact = _step_response_semi_infinite(solution.depth_m, horizon)
        errors.append(float(np.max(np.abs(solution.concentration_bq_m3 - exact))))
        rows.append((cell_count, steps, f"{errors[-1]:.6e}"))
        finest = solution

    assert finest is not None
    orders = _observed_orders(errors)
    table = _format_table("cells steps max_error_bq_m3", rows) + "\n ratios " + str(
        [f"{value:.4f}" for value in orders]
    )
    assert all(errors[k] > errors[k + 1] for k in range(len(errors) - 1)), table
    assert 2.0 < orders[-1] < 2.1, table
    assert max(orders) < 2.3, table
    assert errors[-1] / STEP_SURFACE_BQ_M3 < 7.08e-5, table

    assert float(finest.concentration_bq_m3[-1]) < 1e-5
    assert finest.base_flux_bq_m2_s == 0.0
    assert finest.surface_flux_bq_m2_s < 0.0
    assert np.all(np.diff(finest.concentration_bq_m3) < 0.0)


def test_transient_surface_flux_reproduces_the_erf_step_response() -> None:
    horizon = STEP_HORIZON_S
    expected = _step_response_surface_flux(horizon)
    gaps: list[float] = []
    rows: list[tuple[Any, ...]] = []
    for cell_count, steps in [(150, 240), (300, 480), (600, 960), (1200, 1920)]:
        dz = STEP_THICKNESS_M / cell_count
        solution = solve_transient(
            _uniform(D_E_M2_S, cell_count),
            _uniform(BETA, cell_count),
            np.zeros(cell_count),
            dz,
            np.zeros(cell_count),
            horizon / steps,
            steps,
            LAMBDA_RN222,
            surface=Dirichlet(STEP_SURFACE_BQ_M3),
            base=ZeroFlux(),
        )
        gaps.append(abs(solution.surface_flux_bq_m2_s - expected) / abs(expected))
        rows.append((cell_count, steps, f"{solution.surface_flux_bq_m2_s:.8f}", f"{gaps[-1]:.6e}"))

    table = _format_table("cells steps flux_bq_m2_s relative_gap", rows)
    assert all(gaps[k] > gaps[k + 1] for k in range(len(gaps) - 1)), table
    assert gaps[-1] < 1.57e-4, table
    assert expected == pytest.approx(-0.14714072, rel=1e-7)


def test_steady_solution_is_non_negative_with_non_negative_data() -> None:
    cell_count = 60
    dz = THICKNESS_M / cell_count
    generator = np.random.default_rng(11)
    d_bulk = D_E_M2_S * np.exp(generator.normal(0.0, 1.0, cell_count))
    beta = BETA * np.exp(generator.normal(0.0, 0.4, cell_count))
    source = S_BULK_BQ_M3_S * generator.random(cell_count)
    source[cell_count // 3 : cell_count // 2] = 0.0

    for base in (ZeroFlux(), Dirichlet(C_EQ_BQ_M3)):
        for surface in (Dirichlet(0.0), Dirichlet(15.0)):
            solution = solve_steady(
                d_bulk, beta, source, dz, LAMBDA_RN222, surface=surface, base=base
            )
            assert solution.minimum_concentration_bq_m3 >= 0.0
            assert np.all(solution.concentration_bq_m3 >= 0.0)


def test_transient_from_a_non_negative_start_stays_non_negative_with_backward_euler() -> None:
    cell_count = 30
    dz = RB1_THICKNESS_M / cell_count
    solution = solve_transient(
        _uniform(RB1_D_E_ASSUMED_M2_S, cell_count),
        _uniform(RB1_BETA_ASSUMED, cell_count),
        _uniform(RB1_S_BULK_BQ_M3_S, cell_count),
        dz,
        np.zeros(cell_count),
        3600.0,
        240,
        LAMBDA_RN222,
    )
    assert solution.max_cell_fourier_number > 1.0e3
    assert solution.minimum_concentration_bq_m3 >= 0.0


def test_no_nan_is_produced_or_hidden() -> None:
    cell_count = 30
    dz = THICKNESS_M / cell_count
    steady = solve_steady(
        _uniform(D_E_M2_S, cell_count),
        _uniform(BETA, cell_count),
        _uniform(S_BULK_BQ_M3_S, cell_count),
        dz,
        LAMBDA_RN222,
    )
    transient = solve_transient(
        _uniform(D_E_M2_S, cell_count),
        _uniform(BETA, cell_count),
        _uniform(S_BULK_BQ_M3_S, cell_count),
        dz,
        np.zeros(cell_count),
        3600.0,
        24,
        LAMBDA_RN222,
    )
    for solution in (steady, transient):
        assert np.all(np.isfinite(solution.concentration_bq_m3))
        assert np.all(np.isfinite(solution.depth_m))
        assert math.isfinite(solution.surface_flux_bq_m2_s)
        assert math.isfinite(solution.base_flux_bq_m2_s)
        assert math.isfinite(solution.balance.residual_bq_m2_s)
        assert math.isfinite(solution.balance.relative_residual)
        assert math.isfinite(solution.minimum_concentration_bq_m3)


def test_time_step_convergence_order_is_one_for_backward_euler() -> None:
    cell_count = 20
    thickness = 2.0
    dz = thickness / cell_count
    d_bulk = _uniform(D_E_M2_S, cell_count)
    beta = _uniform(BETA, cell_count)
    source = _uniform(S_BULK_BQ_M3_S, cell_count)
    initial = np.zeros(cell_count)
    horizon_s = 5.0 * 86400.0

    operator, load = _reference_operator(
        d_bulk, beta, source, dz, LAMBDA_RN222, c_top_bq_m3=0.0, c_bot_bq_m3=None
    )
    system = -operator / beta[:, None]
    forcing = load / beta
    steady_state = -np.linalg.solve(system, forcing)
    exact = expm(system * horizon_s) @ (initial - steady_state) + steady_state

    step_counts = [10, 20, 40, 80, 160, 320, 640]
    errors: list[float] = []
    rows: list[tuple[Any, ...]] = []
    for steps in step_counts:
        dt = horizon_s / steps
        solution = solve_transient(
            d_bulk, beta, source, dz, initial, dt, steps, LAMBDA_RN222
        )
        error = float(np.max(np.abs(solution.concentration_bq_m3 - exact)))
        errors.append(error)
        rows.append((steps, f"{dt:.1f}", f"{error:.6e}"))

    orders = _observed_orders(errors)
    table = _format_table("steps dt_s max_error_bq_m3", rows) + "\n ratios " + str(
        [f"{value:.4f}" for value in orders]
    )
    assert all(errors[k] > errors[k + 1] for k in range(len(errors) - 1)), table
    assert 1.95 < orders[-1] < 2.05, table
    assert 1.90 < orders[-2] < 2.05, table
    assert max(orders) < 2.5, table


def test_time_step_convergence_order_is_two_for_crank_nicolson_below_threshold() -> None:
    cell_count = 20
    thickness = 6.0
    dz = thickness / cell_count
    d_bulk = _uniform(D_E_M2_S, cell_count)
    beta = _uniform(BETA, cell_count)
    source = _uniform(S_BULK_BQ_M3_S, cell_count)
    initial = np.zeros(cell_count)
    horizon_s = 2.0 * 86400.0

    operator, load = _reference_operator(
        d_bulk, beta, source, dz, LAMBDA_RN222, c_top_bq_m3=0.0, c_bot_bq_m3=None
    )
    system = -operator / beta[:, None]
    forcing = load / beta
    steady_state = -np.linalg.solve(system, forcing)
    exact = expm(system * horizon_s) @ (initial - steady_state) + steady_state

    errors: list[float] = []
    rows: list[tuple[Any, ...]] = []
    for steps in [40, 80, 160, 320, 640]:
        dt = horizon_s / steps
        solution = solve_transient(
            d_bulk, beta, source, dz, initial, dt, steps, LAMBDA_RN222, theta=0.5
        )
        assert solution.max_cell_fourier_number < 1.0
        errors.append(float(np.max(np.abs(solution.concentration_bq_m3 - exact))))
        rows.append((steps, f"{dt:.1f}", f"{errors[-1]:.6e}"))

    orders = _observed_orders(errors)
    table = _format_table("steps dt_s max_error_bq_m3", rows) + "\n ratios " + str(
        [f"{value:.4f}" for value in orders]
    )
    assert 3.6 < orders[-1] < 4.4, table


def test_cell_count_convergence_order_is_two_for_the_profile() -> None:
    errors: list[float] = []
    rows: list[tuple[Any, ...]] = []
    cell_counts = [30, 60, 120, 240, 480]
    for cell_count in cell_counts:
        dz = THICKNESS_M / cell_count
        solution = solve_steady(
            _uniform(D_E_M2_S, cell_count),
            _uniform(BETA, cell_count),
            _uniform(S_BULK_BQ_M3_S, cell_count),
            dz,
            LAMBDA_RN222,
        )
        expected = _analytic_slab_profile(solution.depth_m)
        error = float(np.max(np.abs(solution.concentration_bq_m3 - expected)))
        errors.append(error)
        rows.append((cell_count, f"{dz:.5f}", f"{error:.6e}", f"{error / C_EQ_BQ_M3:.3e}"))

    orders = _observed_orders(errors)
    table = _format_table("cells dz_m max_error_bq_m3 relative", rows) + "\n ratios " + str(
        [f"{value:.4f}" for value in orders]
    )
    assert 3.8 < orders[-1] < 4.2, table
    assert all(3.5 < value < 4.2 for value in orders), table


def test_cell_count_convergence_order_is_two_for_the_surface_flux() -> None:
    expected = surface_flux_finite_slab(
        RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S, THICKNESS_M
    )
    ratio = THICKNESS_M / L_D_M
    errors: list[float] = []
    rows: list[tuple[Any, ...]] = []
    for cell_count in [30, 60, 120, 240, 480]:
        dz = THICKNESS_M / cell_count
        solution = solve_steady(
            _uniform(D_E_M2_S, cell_count),
            _uniform(BETA, cell_count),
            _uniform(S_BULK_BQ_M3_S, cell_count),
            dz,
            LAMBDA_RN222,
        )
        deficit = 1.0 - solution.surface_flux_bq_m2_s / expected
        predicted = (dz / L_D_M) ** 2 * (
            0.125 + ratio / (12.0 * math.sinh(2.0 * ratio))
        )
        errors.append(abs(deficit))
        rows.append(
            (cell_count, f"{dz:.5f}", f"{deficit:.6e}", f"{predicted:.6e}",
             f"{deficit / predicted:.4f}")
        )
        assert deficit == pytest.approx(predicted, rel=2e-2)

    orders = _observed_orders(errors)
    table = _format_table("cells dz_m deficit predicted ratio", rows) + "\n orders " + str(
        [f"{value:.4f}" for value in orders]
    )
    assert 3.9 < orders[-1] < 4.1, table


def test_thin_bed_surface_flux_does_not_degrade_when_the_grid_is_refined() -> None:
    expected = surface_flux_finite_slab(
        RB1_RA226_BQ_KG,
        RB1_RHO_B_KG_M3,
        RB1_EPS,
        LAMBDA_RN222,
        RB1_D_A_ASSUMED_M2_S,
        RB1_THICKNESS_M,
    )
    ceiling = production_ceiling_sealed_bed(
        RB1_RA226_BQ_KG, RB1_RHO_B_KG_M3, RB1_EPS, LAMBDA_RN222, RB1_THICKNESS_M
    )
    assert expected < ceiling

    steady_fluxes: list[float] = []
    transient_fluxes: list[float] = []
    rows: list[tuple[Any, ...]] = []
    for cell_count in [10, 15, 20, 30, 45]:
        dz = RB1_THICKNESS_M / cell_count
        d_bulk = _uniform(RB1_D_E_ASSUMED_M2_S, cell_count)
        beta = _uniform(RB1_BETA_ASSUMED, cell_count)
        source = _uniform(RB1_S_BULK_BQ_M3_S, cell_count)

        steady = solve_steady(d_bulk, beta, source, dz, LAMBDA_RN222)
        transient = solve_transient(
            d_bulk, beta, source, dz, np.zeros(cell_count), 3600.0, 720, LAMBDA_RN222
        )
        steady_fluxes.append(steady.surface_flux_bq_m2_s)
        transient_fluxes.append(transient.surface_flux_bq_m2_s)
        rows.append(
            (
                cell_count,
                f"{dz * 1e3:.2f}mm",
                f"{steady.surface_flux_bq_m2_s * 1e3:.4f}",
                f"{transient.surface_flux_bq_m2_s * 1e3:.4f}",
                f"{transient.max_cell_fourier_number:.2e}",
            )
        )
        assert transient.max_cell_fourier_number > 2.0e2

    table = _format_table("cells dz J_steady_mBq J_transient_mBq Fourier", rows)
    assert max(steady_fluxes) / min(steady_fluxes) - 1.0 < 1e-5, table
    assert max(transient_fluxes) / min(transient_fluxes) - 1.0 < 1e-5, table
    for flux in steady_fluxes + transient_fluxes:
        assert flux == pytest.approx(expected, rel=1e-4), table
        assert 0.0 < flux < ceiling


def test_crank_nicolson_rings_at_a_high_fourier_number_and_backward_euler_does_not() -> None:
    cell_count = 30
    dz = RB1_THICKNESS_M / cell_count
    d_bulk = _uniform(RB1_D_E_ASSUMED_M2_S, cell_count)
    beta = _uniform(RB1_BETA_ASSUMED, cell_count)
    source = _uniform(RB1_S_BULK_BQ_M3_S, cell_count)
    c_eq = equilibrium_concentration(RB1_S_BULK_BQ_M3_S, LAMBDA_RN222, RB1_BETA_ASSUMED)
    initial = _uniform(c_eq, cell_count)

    ringing = [
        solve_transient(
            d_bulk,
            beta,
            source,
            dz,
            initial,
            3600.0,
            steps,
            LAMBDA_RN222,
            theta=0.5,
            max_step_halvings=0,
        )
        for steps in (1, 2, 3)
    ]
    assert all(solution.max_cell_fourier_number > 1.0e3 for solution in ringing)
    surface_cells = [float(solution.concentration_bq_m3[0]) for solution in ringing]
    assert surface_cells[0] < 0.0
    assert surface_cells[1] > 0.0
    assert surface_cells[2] < 0.0
    assert min(abs(value) for value in surface_cells) > 0.5 * c_eq

    damping = [
        solve_transient(
            d_bulk, beta, source, dz, initial, 3600.0, steps, LAMBDA_RN222, theta=1.0
        )
        for steps in (1, 2, 3)
    ]
    for solution in damping:
        assert solution.minimum_concentration_bq_m3 >= 0.0
    surface_cells_be = [float(solution.concentration_bq_m3[0]) for solution in damping]
    assert surface_cells_be[0] > surface_cells_be[1] > surface_cells_be[2] > 0.0


def test_crank_nicolson_ringing_does_not_itself_destroy_mass() -> None:
    cell_count = 30
    dz = RB1_THICKNESS_M / cell_count
    c_eq = equilibrium_concentration(RB1_S_BULK_BQ_M3_S, LAMBDA_RN222, RB1_BETA_ASSUMED)
    solution = solve_transient(
        _uniform(RB1_D_E_ASSUMED_M2_S, cell_count),
        _uniform(RB1_BETA_ASSUMED, cell_count),
        _uniform(RB1_S_BULK_BQ_M3_S, cell_count),
        dz,
        _uniform(c_eq, cell_count),
        3600.0,
        3,
        LAMBDA_RN222,
        theta=0.5,
        max_step_halvings=0,
    )
    assert solution.minimum_concentration_bq_m3 < 0.0
    assert abs(solution.balance.relative_residual) < 1e-11


def _assert_balance_closes(balance: MassBalance, tolerance: float) -> None:
    reconstructed = (
        balance.production_bq_m2_s
        + balance.base_flux_bq_m2_s
        - balance.decay_bq_m2_s
        - balance.surface_flux_bq_m2_s
        - balance.storage_rate_bq_m2_s
    )
    assert reconstructed == pytest.approx(balance.residual_bq_m2_s, rel=1e-12, abs=1e-30)
    assert abs(balance.relative_residual) < tolerance


def test_steady_balance_closes_with_a_sealed_base() -> None:
    for cell_count in (10, 40, 160):
        dz = THICKNESS_M / cell_count
        solution = solve_steady(
            _uniform(D_E_M2_S, cell_count),
            _uniform(BETA, cell_count),
            _uniform(S_BULK_BQ_M3_S, cell_count),
            dz,
            LAMBDA_RN222,
        )
        balance = solution.balance
        assert balance.base_flux_bq_m2_s == 0.0
        assert balance.storage_rate_bq_m2_s == 0.0
        assert balance.production_bq_m2_s == pytest.approx(
            S_BULK_BQ_M3_S * THICKNESS_M, rel=1e-13
        )
        _assert_balance_closes(balance, 1e-11)
        assert balance.decay_bq_m2_s > 0.0
        assert balance.surface_flux_bq_m2_s > 0.0


def test_steady_balance_closes_with_a_dirichlet_base() -> None:
    thickness = L_D_M
    for cell_count in (10, 40, 160):
        dz = thickness / cell_count
        solution = solve_steady(
            _uniform(D_E_M2_S, cell_count),
            _uniform(BETA, cell_count),
            _uniform(S_BULK_BQ_M3_S, cell_count),
            dz,
            LAMBDA_RN222,
            base=Dirichlet(C_EQ_BQ_M3),
        )
        balance = solution.balance
        assert balance.base_flux_bq_m2_s > 0.05 * balance.surface_flux_bq_m2_s
        assert balance.storage_rate_bq_m2_s == 0.0
        _assert_balance_closes(balance, 1e-11)


def test_steady_balance_closes_on_a_heterogeneous_column() -> None:
    cell_count = 80
    dz = THICKNESS_M / cell_count
    generator = np.random.default_rng(3)
    d_bulk = D_E_M2_S * np.exp(generator.normal(0.0, 1.2, cell_count))
    beta = BETA * np.exp(generator.normal(0.0, 0.5, cell_count))
    source = S_BULK_BQ_M3_S * generator.random(cell_count)
    for base in (ZeroFlux(), Dirichlet(C_EQ_BQ_M3)):
        solution = solve_steady(d_bulk, beta, source, dz, LAMBDA_RN222, base=base)
        _assert_balance_closes(solution.balance, 1e-11)


def test_balance_residual_is_round_off_and_not_discretisation() -> None:
    residuals: list[float] = []
    rows: list[tuple[Any, ...]] = []
    cell_counts = [10, 40, 160, 640, 2560]
    for cell_count in cell_counts:
        dz = THICKNESS_M / cell_count
        solution = solve_steady(
            _uniform(D_E_M2_S, cell_count),
            _uniform(BETA, cell_count),
            _uniform(S_BULK_BQ_M3_S, cell_count),
            dz,
            LAMBDA_RN222,
        )
        relative = solution.balance.relative_residual
        operator_norm = 4.0 * D_E_M2_S / (dz * dz) + LAMBDA_RN222 * BETA
        estimate = (
            np.finfo(float).eps
            * operator_norm
            * float(np.max(solution.concentration_bq_m3))
            * math.sqrt(cell_count)
            * dz
            / solution.balance.production_bq_m2_s
        )
        residuals.append(relative)
        rows.append((cell_count, f"{dz:.6f}", f"{relative:+.3e}", f"{estimate:.3e}"))
        assert abs(relative) < 200.0 * estimate
        assert abs(relative) < 1e-9

    table = _format_table("cells dz_m relative_residual roundoff_estimate", rows)
    assert abs(residuals[-1]) > abs(residuals[0]), table
    signs = {math.copysign(1.0, value) for value in residuals}
    assert len(signs) == 2, table


def test_transient_balance_includes_storage() -> None:
    cell_count = 40
    dz = THICKNESS_M / cell_count
    d_bulk = _uniform(D_E_M2_S, cell_count)
    beta = _uniform(BETA, cell_count)
    source = _uniform(S_BULK_BQ_M3_S, cell_count)
    for base in (ZeroFlux(), Dirichlet(C_EQ_BQ_M3)):
        for steps in (1, 12, 240):
            solution = solve_transient(
                d_bulk,
                beta,
                source,
                dz,
                np.zeros(cell_count),
                3600.0,
                steps,
                LAMBDA_RN222,
                base=base,
            )
            _assert_balance_closes(solution.balance, 1e-11)
            assert solution.balance.storage_rate_bq_m2_s > 0.0
        early = solve_transient(
            d_bulk, beta, source, dz, np.zeros(cell_count), 3600.0, 1, LAMBDA_RN222, base=base
        )
        assert early.balance.storage_rate_bq_m2_s > 0.9 * early.balance.production_bq_m2_s


def test_transient_storage_vanishes_once_the_steady_state_is_reached() -> None:
    cell_count = 40
    dz = THICKNESS_M / cell_count
    solution = solve_transient(
        _uniform(D_E_M2_S, cell_count),
        _uniform(BETA, cell_count),
        _uniform(S_BULK_BQ_M3_S, cell_count),
        dz,
        np.zeros(cell_count),
        3600.0,
        24 * 90,
        LAMBDA_RN222,
    )
    balance = solution.balance
    assert abs(balance.storage_rate_bq_m2_s) < 1e-6 * balance.production_bq_m2_s
    _assert_balance_closes(balance, 1e-11)


def test_relative_residual_is_zero_for_an_entirely_inert_column() -> None:
    cell_count = 10
    dz = THICKNESS_M / cell_count
    solution = solve_steady(
        _uniform(D_E_M2_S, cell_count),
        _uniform(BETA, cell_count),
        np.zeros(cell_count),
        dz,
        LAMBDA_RN222,
        surface=Dirichlet(0.0),
        base=ZeroFlux(),
    )
    assert np.all(solution.concentration_bq_m3 == 0.0)
    assert solution.balance.relative_residual == 0.0
    assert solution.balance.residual_bq_m2_s == 0.0


def test_transient_converges_to_the_steady_solution() -> None:
    cell_count = 40
    dz = THICKNESS_M / cell_count
    d_bulk = _uniform(D_E_M2_S, cell_count)
    beta = _uniform(BETA, cell_count)
    source = _uniform(S_BULK_BQ_M3_S, cell_count)
    steady = solve_steady(d_bulk, beta, source, dz, LAMBDA_RN222)
    transient = solve_transient(
        d_bulk, beta, source, dz, np.zeros(cell_count), 3600.0, 24 * 90, LAMBDA_RN222
    )
    relative = np.max(
        np.abs(transient.concentration_bq_m3 - steady.concentration_bq_m3)
    ) / np.max(steady.concentration_bq_m3)
    assert relative < 1e-6
    assert transient.surface_flux_bq_m2_s == pytest.approx(
        steady.surface_flux_bq_m2_s, rel=1e-6
    )
    assert transient.elapsed_time_s == 24 * 90 * 3600.0
    assert transient.step_count == 24 * 90
    assert transient.theta == 1.0


def test_transient_relaxes_from_above_as_well_as_from_below() -> None:
    cell_count = 40
    dz = THICKNESS_M / cell_count
    d_bulk = _uniform(D_E_M2_S, cell_count)
    beta = _uniform(BETA, cell_count)
    source = _uniform(S_BULK_BQ_M3_S, cell_count)
    steady = solve_steady(d_bulk, beta, source, dz, LAMBDA_RN222)
    transient = solve_transient(
        d_bulk,
        beta,
        source,
        dz,
        _uniform(3.0 * C_EQ_BQ_M3, cell_count),
        3600.0,
        24 * 90,
        LAMBDA_RN222,
    )
    relative = np.max(
        np.abs(transient.concentration_bq_m3 - steady.concentration_bq_m3)
    ) / np.max(steady.concentration_bq_m3)
    assert relative < 1e-6


def test_steady_result_reports_no_transient_metadata() -> None:
    cell_count = 12
    dz = THICKNESS_M / cell_count
    solution = solve_steady(
        _uniform(D_E_M2_S, cell_count),
        _uniform(BETA, cell_count),
        _uniform(S_BULK_BQ_M3_S, cell_count),
        dz,
        LAMBDA_RN222,
    )
    assert isinstance(solution, ColumnSolution)
    assert solution.step_count == 0
    assert solution.time_step_s is None
    assert solution.elapsed_time_s is None
    assert solution.theta is None
    assert solution.max_cell_fourier_number is None
    assert solution.cell_count == cell_count
    assert solution.column_thickness_m == pytest.approx(THICKNESS_M)


def test_steady_solution_satisfies_the_legacy_divergence_operator(legacy: Any) -> None:
    assert legacy._DIFF_FACE_HARMONIC is True
    cell_count = 40
    dz = THICKNESS_M / cell_count
    generator = np.random.default_rng(7)
    d_bulk = D_E_M2_S * np.exp(generator.normal(0.0, 0.8, cell_count))
    beta = BETA * np.exp(generator.normal(0.0, 0.3, cell_count))
    source = S_BULK_BQ_M3_S * (1.0 + 0.5 * generator.random(cell_count))

    solution = solve_steady(d_bulk, beta, source, dz, LAMBDA_RN222)
    divergence = legacy._div_diff_flux(
        solution.concentration_bq_m3, d_bulk, dz, C_top=0.0, C_bot=None
    )
    residual = divergence + source - LAMBDA_RN222 * beta * solution.concentration_bq_m3
    assert np.max(np.abs(residual)) < 1e-10 * np.max(source)


def test_steady_profile_matches_legacy_soil_column_steady_at_second_order(
    legacy: Any,
) -> None:
    source_pore_gas = S_BULK_BQ_M3_S / BETA
    differences: list[float] = []
    rows: list[tuple[Any, ...]] = []
    for cell_count in (60, 120, 240, 480):
        dz = THICKNESS_M / cell_count
        ours = solve_steady(
            _uniform(D_E_M2_S, cell_count),
            _uniform(BETA, cell_count),
            _uniform(S_BULK_BQ_M3_S, cell_count),
            dz,
            LAMBDA_RN222,
        )
        node_count = cell_count + 1
        legacy_profile = legacy._soil_column_steady(
            D_A_M2_S, LAMBDA_RN222, source_pore_gas, THICKNESS_M, node_count, C_top=0.0
        )
        legacy_depths = np.linspace(0.0, THICKNESS_M, node_count)
        interpolated = np.interp(ours.depth_m, legacy_depths, legacy_profile)
        difference = float(np.max(np.abs(ours.concentration_bq_m3 - interpolated)))
        differences.append(difference)
        rows.append((cell_count, f"{dz:.5f}", f"{difference:.4e}",
                     f"{difference / C_EQ_BQ_M3:.3e}"))

    orders = _observed_orders(differences)
    table = _format_table("cells dz_m max_diff_bq_m3 relative", rows) + "\n ratios " + str(
        [f"{value:.4f}" for value in orders]
    )
    assert 3.7 < orders[-1] < 4.3, table
    assert differences[-1] / C_EQ_BQ_M3 < 1e-4, table


def test_legacy_two_node_entry_flux_is_only_first_order(legacy: Any) -> None:
    expected = surface_flux_finite_slab(
        RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S, THICKNESS_M
    )
    source_pore_gas = S_BULK_BQ_M3_S / BETA
    legacy_deficits: list[float] = []
    ours_deficits: list[float] = []
    rows: list[tuple[Any, ...]] = []
    for cell_count in (60, 120, 240, 480):
        dz = THICKNESS_M / cell_count
        node_count = cell_count + 1
        legacy_profile = legacy._soil_column_steady(
            D_A_M2_S, LAMBDA_RN222, source_pore_gas, THICKNESS_M, node_count, C_top=0.0
        )
        node_spacing = THICKNESS_M / (node_count - 1)
        legacy_flux = D_E_M2_S * (legacy_profile[1] - legacy_profile[0]) / node_spacing
        ours = solve_steady(
            _uniform(D_E_M2_S, cell_count),
            _uniform(BETA, cell_count),
            _uniform(S_BULK_BQ_M3_S, cell_count),
            dz,
            LAMBDA_RN222,
        )
        legacy_deficits.append(abs(1.0 - legacy_flux / expected))
        ours_deficits.append(abs(1.0 - ours.surface_flux_bq_m2_s / expected))
        rows.append(
            (cell_count, f"{dz:.5f}", f"{legacy_deficits[-1]:.4e}",
             f"{ours_deficits[-1]:.4e}")
        )

    legacy_orders = _observed_orders(legacy_deficits)
    ours_orders = _observed_orders(ours_deficits)
    table = _format_table("cells dz_m legacy_deficit our_deficit", rows) + (
        "\n legacy ratios " + str([f"{value:.4f}" for value in legacy_orders])
        + "\n our ratios " + str([f"{value:.4f}" for value in ours_orders])
    )
    assert 1.9 < legacy_orders[-1] < 2.1, table
    assert 3.9 < ours_orders[-1] < 4.1, table
    assert legacy_deficits[0] > 0.015, table
    assert ours_deficits[0] < 0.001, table
    assert legacy_deficits[-1] > 50.0 * ours_deficits[-1], table


def _valid_kwargs(cell_count: int = 10) -> dict:
    return {
        "d_bulk_m2_s": _uniform(D_E_M2_S, cell_count),
        "beta": _uniform(BETA, cell_count),
        "source_bulk_bq_m3_s": _uniform(S_BULK_BQ_M3_S, cell_count),
        "dz_m": THICKNESS_M / cell_count,
        "lambda_s": LAMBDA_RN222,
    }


def test_too_few_cells_is_rejected() -> None:
    assert MINIMUM_CELL_COUNT == 3
    for cell_count in (1, 2):
        with pytest.raises(ValueError, match="at least 3 cells"):
            solve_steady(**_valid_kwargs(cell_count))
    solve_steady(**_valid_kwargs(3))


@pytest.mark.parametrize("bad_dz", [0.0, -0.1, float("nan"), float("inf")])
def test_non_positive_or_non_finite_cell_size_is_rejected(bad_dz: float) -> None:
    kwargs = _valid_kwargs()
    kwargs["dz_m"] = bad_dz
    with pytest.raises(ValueError, match="dz_m"):
        solve_steady(**kwargs)


@pytest.mark.parametrize("bad_lambda", [0.0, -1.0, float("nan")])
def test_non_positive_decay_constant_is_rejected(bad_lambda: float) -> None:
    kwargs = _valid_kwargs()
    kwargs["lambda_s"] = bad_lambda
    with pytest.raises(ValueError, match="lambda_s"):
        solve_steady(**kwargs)


@pytest.mark.parametrize("bad_value", [0.0, -0.2])
def test_non_positive_partition_porosity_is_rejected(bad_value: float) -> None:
    kwargs = _valid_kwargs()
    kwargs["beta"][4] = bad_value
    with pytest.raises(ValueError, match="beta must be > 0"):
        solve_steady(**kwargs)


@pytest.mark.parametrize("bad_value", [0.0, -1e-6])
def test_non_positive_diffusion_coefficient_is_rejected(bad_value: float) -> None:
    kwargs = _valid_kwargs()
    kwargs["d_bulk_m2_s"][2] = bad_value
    with pytest.raises(ValueError, match="d_bulk_m2_s must be > 0"):
        solve_steady(**kwargs)


def test_negative_production_is_rejected() -> None:
    kwargs = _valid_kwargs()
    kwargs["source_bulk_bq_m3_s"][1] = -1.0
    with pytest.raises(ValueError, match="source_bulk_bq_m3_s must be >= 0"):
        solve_steady(**kwargs)


@pytest.mark.parametrize("field", ["d_bulk_m2_s", "beta", "source_bulk_bq_m3_s"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_entries_are_rejected(field: str, bad_value: float) -> None:
    kwargs = _valid_kwargs()
    kwargs[field][3] = bad_value
    with pytest.raises(ValueError, match=field):
        solve_steady(**kwargs)


def test_mismatched_array_lengths_are_rejected() -> None:
    kwargs = _valid_kwargs(10)
    kwargs["beta"] = _uniform(BETA, 9)
    with pytest.raises(ValueError, match="beta must have 10 entries"):
        solve_steady(**kwargs)


def test_two_dimensional_input_is_rejected() -> None:
    kwargs = _valid_kwargs()
    kwargs["d_bulk_m2_s"] = np.full((4, 4), D_E_M2_S)
    with pytest.raises(ValueError, match="one-dimensional"):
        solve_steady(**kwargs)


def test_unknown_boundary_condition_type_is_rejected() -> None:
    kwargs = _valid_kwargs()
    with pytest.raises(TypeError, match="Dirichlet or ZeroFlux"):
        solve_steady(**kwargs, base="sealed")


@pytest.mark.parametrize("bad_dt", [0.0, -60.0, float("nan")])
def test_non_positive_time_step_is_rejected(bad_dt: float) -> None:
    kwargs = _valid_kwargs()
    with pytest.raises(ValueError, match="time_step_s"):
        solve_transient(
            **kwargs, initial_bq_m3=np.zeros(10), time_step_s=bad_dt, step_count=1
        )


@pytest.mark.parametrize("bad_count", [0, -1])
def test_non_positive_step_count_is_rejected(bad_count: int) -> None:
    kwargs = _valid_kwargs()
    with pytest.raises(ValueError, match="step_count must be >= 1"):
        solve_transient(
            **kwargs, initial_bq_m3=np.zeros(10), time_step_s=60.0, step_count=bad_count
        )


def test_non_integer_step_count_is_rejected() -> None:
    kwargs = _valid_kwargs()
    with pytest.raises(TypeError, match="step_count must be an integer"):
        solve_transient(
            **kwargs, initial_bq_m3=np.zeros(10), time_step_s=60.0, step_count=2.5
        )


def test_non_integer_cell_count_is_rejected_by_the_grid_helper() -> None:
    with pytest.raises(TypeError, match="cell_count must be an integer"):
        cell_centres(5.5, 0.1)


@pytest.mark.parametrize("bad_theta", [-0.1, 1.1, float("nan")])
def test_theta_outside_the_unit_interval_is_rejected(bad_theta: float) -> None:
    kwargs = _valid_kwargs()
    with pytest.raises(ValueError, match="theta"):
        solve_transient(
            **kwargs,
            initial_bq_m3=np.zeros(10),
            time_step_s=60.0,
            step_count=1,
            theta=bad_theta,
        )


def test_negative_initial_condition_is_rejected() -> None:
    kwargs = _valid_kwargs()
    initial = np.zeros(10)
    initial[5] = -1.0
    with pytest.raises(ValueError, match="initial_bq_m3 must be >= 0"):
        solve_transient(
            **kwargs, initial_bq_m3=initial, time_step_s=60.0, step_count=1
        )


def test_initial_condition_length_must_match_the_grid() -> None:
    kwargs = _valid_kwargs(10)
    with pytest.raises(ValueError, match="initial_bq_m3 must have 10 entries"):
        solve_transient(
            **kwargs, initial_bq_m3=np.zeros(11), time_step_s=60.0, step_count=1
        )


@pytest.mark.parametrize("bad_count", [2, 0, -3])
def test_cell_centres_rejects_a_grid_that_is_too_small(bad_count: int) -> None:
    with pytest.raises(ValueError, match="cell_count"):
        cell_centres(bad_count, 0.1)


def test_cell_centres_rejects_a_non_positive_spacing() -> None:
    with pytest.raises(ValueError, match="dz_m"):
        cell_centres(5, 0.0)
