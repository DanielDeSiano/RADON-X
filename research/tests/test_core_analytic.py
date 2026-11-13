from __future__ import annotations

import importlib.util
import os
import math
import sys
from typing import Any

import pytest

from radon_x.core.analytic import (
    steady_profile_finite_slab,
    steady_profile_semi_infinite,
    surface_flux_finite_slab,
    surface_flux_semi_infinite,
)
from radon_x.core.decay import LAMBDA_RN222, diffusion_length
from radon_x.core.diffusion import apparent_diffusivity_rogers_nielson
from radon_x.core.sources import production_ceiling_sealed_bed

from radon_x.legacy_adapter import legacy_source_path

LEGACY_MODEL_PATH = str(legacy_source_path())

RA226_BQ_KG = 40.0
RHO_B_KG_M3 = 1600.0
EPS = 0.25
D_A_M2_S = 4.7213e-06


def _load_legacy() -> Any:
    if not os.path.isfile(LEGACY_MODEL_PATH):
        pytest.skip(f"legacy model not found at {LEGACY_MODEL_PATH}")
    spec = importlib.util.spec_from_file_location("legacy_for_analytic_tests", LEGACY_MODEL_PATH)
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


def _legacy_apparent_diffusivity(porosity: float, saturation: float, d0: float = 1.1e-5) -> float:
    return (
        porosity
        * d0
        * math.exp(-6.0 * saturation * porosity - 6.0 * saturation ** (14.0 * porosity))
    )


def test_thick_slab_flux_tends_to_semi_infinite() -> None:
    l_d = diffusion_length(D_A_M2_S, LAMBDA_RN222)
    finite = surface_flux_finite_slab(
        RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S, 20.0 * l_d
    )
    semi_infinite = surface_flux_semi_infinite(
        RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S
    )
    assert math.isclose(finite, semi_infinite, rel_tol=1e-6)


@pytest.mark.parametrize("ratio", [5.0, 10.0, 20.0, 50.0])
def test_thick_slab_flux_approaches_semi_infinite_monotonically(ratio: float) -> None:
    l_d = diffusion_length(D_A_M2_S, LAMBDA_RN222)
    finite = surface_flux_finite_slab(
        RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S, ratio * l_d
    )
    semi_infinite = surface_flux_semi_infinite(
        RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S
    )
    assert finite <= semi_infinite
    assert math.isclose(finite / semi_infinite, math.tanh(ratio), rel_tol=1e-14)


def test_thin_slab_flux_tends_to_production_ceiling() -> None:
    l_d = diffusion_length(D_A_M2_S, LAMBDA_RN222)
    thickness = l_d / 1000.0
    flux = surface_flux_finite_slab(
        RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S, thickness
    )
    ceiling = production_ceiling_sealed_bed(
        ra226_bq_kg=RA226_BQ_KG,
        dry_bulk_density_kg_m3=RHO_B_KG_M3,
        emanation_fraction=EPS,
        lambda_s=LAMBDA_RN222,
        thickness_m=thickness,
    )
    assert flux < ceiling
    assert math.isclose(flux, ceiling, rel_tol=1e-6)


def test_thin_slab_flux_is_independent_of_the_diffusion_coefficient() -> None:
    thickness = 0.001
    fluxes = [
        surface_flux_finite_slab(RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, d_a, thickness)
        for d_a in (1.0e-7, 1.0e-6, 1.0e-5, 1.1e-5)
    ]
    assert max(fluxes) / min(fluxes) - 1.0 < 1e-4


def test_semi_infinite_profile_is_zero_at_the_surface() -> None:
    assert steady_profile_semi_infinite(0.0, 5.0e4, 1.5) == 0.0


def test_semi_infinite_profile_tends_to_c_eq_at_depth() -> None:
    c_eq = 5.0e4
    deep = steady_profile_semi_infinite(20.0 * 1.5, c_eq, 1.5)
    assert deep < c_eq
    assert math.isclose(deep, c_eq, rel_tol=1e-8)


@pytest.mark.parametrize("z_over_l", [0.25, 0.5, 1.0, 2.0, 5.0])
def test_semi_infinite_profile_matches_closed_form(z_over_l: float) -> None:
    c_eq = 3.0e4
    l_d = 1.5
    value = steady_profile_semi_infinite(z_over_l * l_d, c_eq, l_d)
    assert math.isclose(value / c_eq, 1.0 - math.exp(-z_over_l), rel_tol=1e-14)


def test_finite_slab_profile_is_zero_at_the_surface() -> None:
    for thickness in (0.01, 0.06, 1.0, 3.0, 100.0):
        assert steady_profile_finite_slab(0.0, 5.0e4, 1.5, thickness) == 0.0


@pytest.mark.parametrize("thickness_m", [0.06, 0.5, 1.5, 4.5])
def test_finite_slab_profile_has_zero_gradient_at_the_base(thickness_m: float) -> None:
    c_eq = 5.0e4
    l_d = 1.5
    h = 1.0e-4 * thickness_m
    c0 = steady_profile_finite_slab(thickness_m, c_eq, l_d, thickness_m)
    c1 = steady_profile_finite_slab(thickness_m - h, c_eq, l_d, thickness_m)
    c2 = steady_profile_finite_slab(thickness_m - 2.0 * h, c_eq, l_d, thickness_m)
    gradient = (3.0 * c0 - 4.0 * c1 + c2) / (2.0 * h)
    assert abs(gradient) / (c_eq / l_d) < 1e-6


def test_finite_slab_profile_tends_to_semi_infinite_when_thick() -> None:
    c_eq = 5.0e4
    l_d = 1.5
    thickness = 30.0 * l_d
    for z_over_l in (0.0, 0.5, 1.0, 3.0, 10.0, 30.0):
        z = z_over_l * l_d
        assert math.isclose(
            steady_profile_finite_slab(z, c_eq, l_d, thickness),
            steady_profile_semi_infinite(z, c_eq, l_d),
            abs_tol=1e-9 * c_eq,
        )


def test_finite_slab_profile_is_monotonic_and_bounded() -> None:
    c_eq = 5.0e4
    l_d = 1.5
    thickness = 2.0
    values = [
        steady_profile_finite_slab(thickness * i / 200.0, c_eq, l_d, thickness) for i in range(201)
    ]
    assert values[0] == 0.0
    assert all(0.0 <= v <= c_eq for v in values)
    assert all(values[i + 1] > values[i] for i in range(len(values) - 1))


def test_finite_slab_profile_does_not_overflow_for_a_very_thick_slab() -> None:
    c_eq = 1.0e4
    l_d = 1.0e-3
    value = steady_profile_finite_slab(10.0, c_eq, l_d, 10.0)
    assert math.isfinite(value)
    assert math.isclose(value, c_eq, rel_tol=1e-12)


@pytest.mark.parametrize("thickness_m", [0.06, 0.5, 1.5, 10.0])
def test_flux_equals_bulk_gradient_of_the_profile(thickness_m: float) -> None:
    beta = 0.35
    l_d = diffusion_length(D_A_M2_S, LAMBDA_RN222)
    c_eq = RHO_B_KG_M3 * RA226_BQ_KG * EPS / beta
    surface_gradient = c_eq * math.tanh(thickness_m / l_d) / l_d
    flux_from_profile = beta * D_A_M2_S * surface_gradient
    flux = surface_flux_finite_slab(
        RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S, thickness_m
    )
    assert math.isclose(flux_from_profile, flux, rel_tol=1e-12)


def test_semi_infinite_flux_equals_bulk_gradient_of_the_profile() -> None:
    beta = 0.35
    l_d = diffusion_length(D_A_M2_S, LAMBDA_RN222)
    c_eq = RHO_B_KG_M3 * RA226_BQ_KG * EPS / beta
    flux_from_profile = beta * D_A_M2_S * c_eq / l_d
    flux = surface_flux_semi_infinite(RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S)
    assert math.isclose(flux_from_profile, flux, rel_tol=1e-12)


@pytest.mark.parametrize(
    ("depth_m", "c_eq", "l_d", "message"),
    [
        (-0.1, 1.0e4, 1.5, "depth_m must be >= 0"),
        (float("nan"), 1.0e4, 1.5, "depth_m must be finite"),
        (1.0, -1.0, 1.5, "c_eq_bq_m3 must be >= 0"),
        (1.0, float("inf"), 1.5, "c_eq_bq_m3 must be finite"),
        (1.0, 1.0e4, 0.0, "diffusion_length_m must be > 0"),
        (1.0, 1.0e4, -1.5, "diffusion_length_m must be > 0"),
        (1.0, 1.0e4, float("nan"), "diffusion_length_m must be finite"),
    ],
)
def test_steady_profile_semi_infinite_rejects_bad_input(
    depth_m: float, c_eq: float, l_d: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        steady_profile_semi_infinite(depth_m, c_eq, l_d)


@pytest.mark.parametrize(
    ("depth_m", "c_eq", "l_d", "thickness_m", "message"),
    [
        (-0.1, 1.0e4, 1.5, 2.0, "depth_m must be >= 0"),
        (1.0, -1.0, 1.5, 2.0, "c_eq_bq_m3 must be >= 0"),
        (1.0, 1.0e4, 0.0, 2.0, "diffusion_length_m must be > 0"),
        (1.0, 1.0e4, 1.5, 0.0, "thickness_m must be > 0"),
        (1.0, 1.0e4, 1.5, -2.0, "thickness_m must be > 0"),
        (1.0, 1.0e4, 1.5, float("nan"), "thickness_m must be finite"),
        (2.5, 1.0e4, 1.5, 2.0, "depth_m must be <= thickness_m"),
    ],
)
def test_steady_profile_finite_slab_rejects_bad_input(
    depth_m: float, c_eq: float, l_d: float, thickness_m: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        steady_profile_finite_slab(depth_m, c_eq, l_d, thickness_m)


@pytest.mark.parametrize(
    ("ra226", "rho_b", "eps", "lam", "d_a", "message"),
    [
        (-1.0, 1600.0, 0.25, LAMBDA_RN222, D_A_M2_S, "ra226_bq_kg must be >= 0"),
        (40.0, 0.0, 0.25, LAMBDA_RN222, D_A_M2_S, "dry_bulk_density_kg_m3 must be > 0"),
        (40.0, -1600.0, 0.25, LAMBDA_RN222, D_A_M2_S, "dry_bulk_density_kg_m3 must be > 0"),
        (40.0, 1600.0, -0.1, LAMBDA_RN222, D_A_M2_S, r"emanation_fraction must be in \[0, 1\]"),
        (40.0, 1600.0, 1.5, LAMBDA_RN222, D_A_M2_S, r"emanation_fraction must be in \[0, 1\]"),
        (40.0, 1600.0, float("nan"), LAMBDA_RN222, D_A_M2_S, "emanation_fraction must be finite"),
        (40.0, 1600.0, 0.25, 0.0, D_A_M2_S, "lambda_s must be > 0"),
        (40.0, 1600.0, 0.25, LAMBDA_RN222, 0.0, "d_apparent_m2_s must be > 0"),
        (40.0, 1600.0, 0.25, LAMBDA_RN222, -1e-6, "d_apparent_m2_s must be > 0"),
        (40.0, 1600.0, 0.25, LAMBDA_RN222, float("inf"), "d_apparent_m2_s must be finite"),
    ],
)
def test_surface_flux_semi_infinite_rejects_bad_input(
    ra226: float, rho_b: float, eps: float, lam: float, d_a: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        surface_flux_semi_infinite(ra226, rho_b, eps, lam, d_a)


@pytest.mark.parametrize(
    ("thickness_m", "message"),
    [
        (0.0, "thickness_m must be > 0"),
        (-1.0, "thickness_m must be > 0"),
        (float("nan"), "thickness_m must be finite"),
    ],
)
def test_surface_flux_finite_slab_rejects_bad_thickness(thickness_m: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        surface_flux_finite_slab(RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S, thickness_m)


def test_surface_flux_finite_slab_propagates_the_other_checks() -> None:
    with pytest.raises(ValueError, match="d_apparent_m2_s must be > 0"):
        surface_flux_finite_slab(RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, -1.0, 1.0)


def test_no_nan_over_the_declared_domain() -> None:
    thicknesses = (1.0e-4, 1.0e-2, 0.06, 1.0, 3.0, 100.0)
    diffusivities = (1.0e-12, 1.0e-9, 1.0e-7, 1.0e-6, 1.1e-5)
    for d_a in diffusivities:
        l_d = diffusion_length(d_a, LAMBDA_RN222)
        semi_infinite = surface_flux_semi_infinite(RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, d_a)
        assert math.isfinite(semi_infinite)
        for thickness in thicknesses:
            flux = surface_flux_finite_slab(
                RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, d_a, thickness
            )
            assert math.isfinite(flux)
            for fraction in (0.0, 0.25, 0.5, 1.0):
                z = fraction * thickness
                slab = steady_profile_finite_slab(z, 5.0e4, l_d, thickness)
                half_space = steady_profile_semi_infinite(z, 5.0e4, l_d)
                assert math.isfinite(slab)
                assert math.isfinite(half_space)


def test_zero_valued_but_valid_input_returns_zero() -> None:
    assert surface_flux_semi_infinite(0.0, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S) == 0.0
    assert surface_flux_semi_infinite(RA226_BQ_KG, RHO_B_KG_M3, 0.0, LAMBDA_RN222, D_A_M2_S) == 0.0
    assert surface_flux_finite_slab(0.0, RHO_B_KG_M3, EPS, LAMBDA_RN222, D_A_M2_S, 1.0) == 0.0
    assert steady_profile_semi_infinite(1.0, 0.0, 1.5) == 0.0
    assert steady_profile_finite_slab(1.0, 0.0, 1.5, 2.0) == 0.0


@pytest.mark.parametrize(
    ("porosity", "saturation", "thickness_m"),
    [(0.42, 0.20, 3.0), (0.30, 0.05, 1.0), (0.48, 0.50, 5.0), (0.55, 0.02, 0.06)],
)
def test_regression_against_legacy_flux_uncertainty_analytic(
    legacy: Any, porosity: float, saturation: float, thickness_m: float
) -> None:
    result = legacy.flux_uncertainty_analytic(
        Ra226=RA226_BQ_KG,
        eps_rn=EPS,
        rho_b=RHO_B_KG_M3,
        phi=porosity,
        Sw=saturation,
        H=thickness_m,
    )
    d_a = _legacy_apparent_diffusivity(porosity, saturation)
    mine = surface_flux_finite_slab(RA226_BQ_KG, RHO_B_KG_M3, EPS, LAMBDA_RN222, d_a, thickness_m)
    assert math.isclose(mine, result["J_Bq_m2_s"], rel_tol=1e-12)


def test_regression_against_legacy_jobbagy_exhalation(legacy: Any) -> None:
    porosity, saturation, ach = 0.40, 0.30, 0.5
    indoor = legacy.jobbagy_2017(RA226_BQ_KG, EPS, porosity, saturation, 3.0, ach)
    water_content = porosity * saturation
    gas_porosity = porosity * (1.0 - saturation)
    beta = max(gas_porosity + 0.25 * water_content, 1e-6)
    d_a = (1.1e-5 * gas_porosity ** (10.0 / 3.0) / porosity**2) / beta
    exhalation_legacy = indoor * 2.5 * (ach / 3600.0 + legacy._LAM_RN)
    mine = surface_flux_semi_infinite(RA226_BQ_KG, RHO_B_KG_M3, EPS, legacy._LAM_RN, d_a)
    assert math.isclose(mine, exhalation_legacy, rel_tol=1e-12)


@pytest.mark.parametrize("n_nodes", [201, 401, 801, 1601])
def test_regression_against_legacy_soil_column_steady(legacy: Any, n_nodes: int) -> None:
    import numpy as np

    model = legacy.build_model("tuff_unwelded")
    thickness = 2.0
    l_d = thickness / 1.44
    d_a = l_d**2 * LAMBDA_RN222
    source = LAMBDA_RN222 * model.p["rho_bulk"] * model.p["Ra226_Bqkg"] * 0.25
    c_eq = source / LAMBDA_RN222

    discrete = legacy._soil_column_steady(d_a, LAMBDA_RN222, source, thickness, n_nodes)
    depths = np.linspace(0.0, thickness, n_nodes)
    analytic = np.array(
        [steady_profile_finite_slab(float(z), c_eq, l_d, thickness) for z in depths]
    )
    max_relative_error = float(np.max(np.abs(discrete - analytic))) / c_eq
    dz = thickness / (n_nodes - 1)
    assert max_relative_error < (dz / l_d) ** 2


def test_legacy_soil_column_converges_at_second_order(legacy: Any) -> None:
    import numpy as np

    thickness = 2.0
    l_d = thickness / 1.44
    d_a = l_d**2 * LAMBDA_RN222
    source = 1.0e-3
    c_eq = source / LAMBDA_RN222

    errors = []
    for n_nodes in (201, 401, 801):
        discrete = legacy._soil_column_steady(d_a, LAMBDA_RN222, source, thickness, n_nodes)
        depths = np.linspace(0.0, thickness, n_nodes)
        analytic = np.array(
            [steady_profile_finite_slab(float(z), c_eq, l_d, thickness) for z in depths]
        )
        errors.append(float(np.max(np.abs(discrete - analytic))) / c_eq)

    for i in range(len(errors) - 1):
        assert 3.7 < errors[i] / errors[i + 1] < 4.3


def test_depth_averaged_fraction_reproduces_the_audit_value(legacy: Any) -> None:
    import numpy as np

    thickness = 2.0
    b = 1.44
    l_d = thickness / b
    d_a = l_d**2 * LAMBDA_RN222
    source = 1.0e-3
    c_eq = source / LAMBDA_RN222

    closed_form = 1.0 - math.tanh(b) / b
    assert math.isclose(closed_form, 0.3793766, rel_tol=1e-6)

    n_nodes = 10001
    depths = np.linspace(0.0, thickness, n_nodes)
    analytic = np.array(
        [steady_profile_finite_slab(float(z), c_eq, l_d, thickness) for z in depths]
    )
    analytic_mean = float(np.trapezoid(analytic, depths)) / thickness / c_eq
    assert math.isclose(analytic_mean, closed_form, rel_tol=1e-8)

    discrete = legacy._soil_column_steady(d_a, LAMBDA_RN222, source, thickness, n_nodes)
    discrete_mean = float(np.trapezoid(discrete, depths)) / thickness / c_eq
    assert math.isclose(discrete_mean, closed_form, rel_tol=1e-6)
    assert round(discrete_mean, 3) == 0.379


def test_huelva_rb1_bench_flux() -> None:
    ra226 = 451.0
    rho_b = 1080.0
    emanation = 0.20
    thickness = 0.060
    porosity = 1.0 - rho_b / 2420.0
    saturation = 0.05

    d_a = apparent_diffusivity_rogers_nielson(porosity, saturation)
    flux_mbq = 1.0e3 * surface_flux_finite_slab(
        ra226, rho_b, emanation, LAMBDA_RN222, d_a, thickness
    )

    assert math.isclose(flux_mbq, 12.26, rel_tol=1e-3)
    assert math.isclose(flux_mbq, 13.25, rel_tol=0.20)
    ceiling_mbq = 1.0e3 * production_ceiling_sealed_bed(
        ra226_bq_kg=ra226,
        dry_bulk_density_kg_m3=rho_b,
        emanation_fraction=emanation,
        lambda_s=LAMBDA_RN222,
        thickness_m=thickness,
    )
    assert flux_mbq < ceiling_mbq
    assert math.isclose(flux_mbq, ceiling_mbq, rel_tol=1e-3)


def test_huelva_rb2_bench_flux() -> None:
    ra226 = 420.0
    rho_b = 1011.0
    emanation = 0.20
    thickness = 0.130
    porosity = 1.0 - rho_b / 2420.0
    saturation = 0.05

    d_a = apparent_diffusivity_rogers_nielson(porosity, saturation)
    flux_mbq = 1.0e3 * surface_flux_finite_slab(
        ra226, rho_b, emanation, LAMBDA_RN222, d_a, thickness
    )
    assert math.isclose(flux_mbq, 23.1, rel_tol=0.02)
    assert math.isclose(flux_mbq, 23.47, rel_tol=0.20)
