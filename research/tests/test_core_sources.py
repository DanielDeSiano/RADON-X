from __future__ import annotations

import importlib.util
import os
import math
import sys
from typing import Any

import pytest

from radon_x.core.decay import LAMBDA_RN222
from radon_x.core.sources import (
    equilibrium_concentration,
    production_ceiling_sealed_bed,
    radon_production_bulk,
    radon_production_pore_gas,
)

from radon_x.legacy_adapter import legacy_source_path

LEGACY_MODEL_PATH = str(legacy_source_path())


def _load_legacy() -> Any:
    if not os.path.isfile(LEGACY_MODEL_PATH):
        pytest.skip(f"legacy model not found at {LEGACY_MODEL_PATH}")
    spec = importlib.util.spec_from_file_location("legacy_for_sources_tests", LEGACY_MODEL_PATH)
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


def test_bulk_source_has_units_bq_per_cubic_metre_per_second() -> None:
    ra226_bq_kg = 1.0e3
    dry_bulk_density_kg_m3 = 1.0e3
    emanation_fraction = 0.5
    lambda_s = 2.0e-6

    source_bq_m3_s = radon_production_bulk(
        ra226_bq_kg, dry_bulk_density_kg_m3, emanation_fraction, lambda_s
    )

    assert math.isclose(source_bq_m3_s, 1.0, rel_tol=1e-15)


def test_bulk_source_scales_linearly_with_every_factor() -> None:
    base = {
        "ra226_bq_kg": 451.0,
        "dry_bulk_density_kg_m3": 1080.0,
        "emanation_fraction": 0.20,
        "lambda_s": LAMBDA_RN222,
    }
    reference = radon_production_bulk(**base)
    scale = 3.0
    for factor in ("ra226_bq_kg", "dry_bulk_density_kg_m3", "emanation_fraction", "lambda_s"):
        scaled = dict(base)
        scaled[factor] = base[factor] * scale
        if factor == "emanation_fraction":
            scaled[factor] = base[factor] / scale
            assert math.isclose(
                radon_production_bulk(**scaled), reference / scale, rel_tol=1e-14
            )
        else:
            assert math.isclose(
                radon_production_bulk(**scaled), reference * scale, rel_tol=1e-14
            )


def test_ceiling_has_units_bq_per_square_metre_per_second() -> None:
    ceiling_bq_m2_s = production_ceiling_sealed_bed(
        ra226_bq_kg=1.0e3,
        dry_bulk_density_kg_m3=1.0e3,
        emanation_fraction=0.5,
        lambda_s=2.0e-6,
        thickness_m=0.5,
    )
    assert math.isclose(ceiling_bq_m2_s, 0.5, rel_tol=1e-15)


def test_pore_gas_source_is_bulk_source_divided_by_beta() -> None:
    args = {
        "ra226_bq_kg": 35.0,
        "dry_bulk_density_kg_m3": 1600.0,
        "emanation_fraction": 0.20,
    }
    beta = 0.36
    bulk = radon_production_bulk(lambda_s=LAMBDA_RN222, **args)
    gas = radon_production_pore_gas(
        partition_porosity=beta, lambda_s=LAMBDA_RN222, **args
    )
    assert math.isclose(gas, bulk / beta, rel_tol=1e-15)


def test_convention_confusion_costs_a_factor_of_about_three_in_dry_soil() -> None:
    args = {
        "ra226_bq_kg": 35.0,
        "dry_bulk_density_kg_m3": 1600.0,
        "emanation_fraction": 0.20,
    }
    beta_dry = 0.34
    ratio = radon_production_pore_gas(
        partition_porosity=beta_dry, lambda_s=LAMBDA_RN222, **args
    ) / radon_production_bulk(lambda_s=LAMBDA_RN222, **args)
    assert math.isclose(ratio, 1.0 / beta_dry, rel_tol=1e-15)
    assert 2.5 < ratio < 3.5


def test_equilibrium_concentration_matches_closed_form() -> None:
    source_bulk = 1.5e-3
    beta = 0.36
    c_eq = equilibrium_concentration(source_bulk, LAMBDA_RN222, beta)
    assert math.isclose(c_eq, source_bulk / (LAMBDA_RN222 * beta), rel_tol=1e-15)


@pytest.mark.parametrize("ra226_bq_kg", [10.0, 35.0, 100.0, 451.0])
def test_equilibrium_concentration_is_linear_in_ra226(ra226_bq_kg: float) -> None:
    beta = 0.36
    unit_source = radon_production_bulk(1.0, 1600.0, 0.20, LAMBDA_RN222)
    unit_c_eq = equilibrium_concentration(unit_source, LAMBDA_RN222, beta)
    source = radon_production_bulk(ra226_bq_kg, 1600.0, 0.20, LAMBDA_RN222)
    c_eq = equilibrium_concentration(source, LAMBDA_RN222, beta)
    assert math.isclose(c_eq, unit_c_eq * ra226_bq_kg, rel_tol=1e-13)


def test_equilibrium_concentration_decreases_with_beta() -> None:
    source_bulk = 1.5e-3
    betas = [0.05, 0.10, 0.20, 0.34, 0.45]
    values = [equilibrium_concentration(source_bulk, LAMBDA_RN222, b) for b in betas]
    assert all(values[i + 1] < values[i] for i in range(len(values) - 1))
    products = [values[i] * betas[i] for i in range(len(betas))]
    assert all(math.isclose(p, products[0], rel_tol=1e-13) for p in products)


def test_equilibrium_concentration_typical_soil_order_of_magnitude() -> None:
    source_bulk = radon_production_bulk(35.0, 1600.0, 0.20, LAMBDA_RN222)
    c_eq = equilibrium_concentration(source_bulk, LAMBDA_RN222, 0.36)
    assert 1.0e3 < c_eq < 1.0e5
    assert math.isclose(c_eq, 35.0 * 1600.0 * 0.20 / 0.36, rel_tol=1e-13)


HUELVA_RB1 = {
    "ra226_bq_kg": 451.0,
    "dry_bulk_density_kg_m3": 1080.0,
    "thickness_m": 0.060,
    "emanation_fraction": 0.20,
    "measured_flux_bq_m2_s": 13.25e-3,
}
HUELVA_RB2 = {
    "ra226_bq_kg": 420.0,
    "dry_bulk_density_kg_m3": 1011.0,
    "thickness_m": 0.130,
    "emanation_fraction": 0.20,
    "measured_flux_bq_m2_s": 23.47e-3,
}


def _ceiling(bed: dict) -> float:
    return production_ceiling_sealed_bed(
        ra226_bq_kg=bed["ra226_bq_kg"],
        dry_bulk_density_kg_m3=bed["dry_bulk_density_kg_m3"],
        emanation_fraction=bed["emanation_fraction"],
        lambda_s=LAMBDA_RN222,
        thickness_m=bed["thickness_m"],
    )


def test_huelva_rb1_production_ceiling() -> None:
    ceiling = _ceiling(HUELVA_RB1)
    assert math.isclose(ceiling, 12.3e-3, rel_tol=0.05)
    assert math.isclose(ceiling, 12.265e-3, rel_tol=1e-3)


def test_huelva_rb2_production_ceiling() -> None:
    ceiling = _ceiling(HUELVA_RB2)
    assert math.isclose(ceiling, 23.2e-3, rel_tol=0.05)
    assert math.isclose(ceiling, 23.166e-3, rel_tol=1e-3)
    assert 23.1e-3 < ceiling


@pytest.mark.parametrize("bed", [HUELVA_RB1, HUELVA_RB2], ids=["RB1", "RB2"])
def test_huelva_measured_flux_exceeds_the_ceiling_at_the_measured_emanation(
    bed: dict,
) -> None:
    ceiling = _ceiling(bed)
    ratio = bed["measured_flux_bq_m2_s"] / ceiling
    assert ratio > 1.0
    implied_emanation = bed["emanation_fraction"] * ratio
    assert abs(implied_emanation - 0.20) <= 0.04


def test_ceiling_is_proportional_to_thickness() -> None:
    thin = production_ceiling_sealed_bed(451.0, 1080.0, 0.20, LAMBDA_RN222, 0.060)
    thick = production_ceiling_sealed_bed(451.0, 1080.0, 0.20, LAMBDA_RN222, 0.120)
    assert math.isclose(thick, 2.0 * thin, rel_tol=1e-14)


def test_ceiling_is_zero_for_zero_thickness() -> None:
    assert production_ceiling_sealed_bed(451.0, 1080.0, 0.20, LAMBDA_RN222, 0.0) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "expected_message_fragment"),
    [
        ({"ra226_bq_kg": -1.0}, "ra226_bq_kg"),
        ({"ra226_bq_kg": float("nan")}, "ra226_bq_kg"),
        ({"dry_bulk_density_kg_m3": -1.0}, "dry_bulk_density_kg_m3"),
        ({"dry_bulk_density_kg_m3": float("inf")}, "dry_bulk_density_kg_m3"),
        ({"emanation_fraction": -0.01}, "emanation_fraction"),
        ({"emanation_fraction": 1.01}, "emanation_fraction"),
        ({"emanation_fraction": float("nan")}, "emanation_fraction"),
        ({"lambda_s": 0.0}, "lambda_s"),
        ({"lambda_s": -1.0}, "lambda_s"),
    ],
)
def test_radon_production_bulk_rejects_out_of_domain_input(
    kwargs: dict, expected_message_fragment: str
) -> None:
    valid = {
        "ra226_bq_kg": 35.0,
        "dry_bulk_density_kg_m3": 1600.0,
        "emanation_fraction": 0.20,
        "lambda_s": LAMBDA_RN222,
    }
    valid.update(kwargs)
    with pytest.raises(ValueError, match=expected_message_fragment):
        radon_production_bulk(**valid)


@pytest.mark.parametrize("beta", [0.0, -0.1, float("nan"), float("inf")])
def test_radon_production_pore_gas_rejects_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="partition_porosity"):
        radon_production_pore_gas(35.0, 1600.0, 0.20, beta, LAMBDA_RN222)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"production_bulk_bq_m3_s": -1e-6},
        {"production_bulk_bq_m3_s": float("nan")},
        {"lambda_s": 0.0},
        {"lambda_s": float("nan")},
        {"partition_porosity": 0.0},
        {"partition_porosity": -0.2},
    ],
)
def test_equilibrium_concentration_rejects_out_of_domain_input(kwargs: dict) -> None:
    valid = {
        "production_bulk_bq_m3_s": 1.5e-3,
        "lambda_s": LAMBDA_RN222,
        "partition_porosity": 0.36,
    }
    valid.update(kwargs)
    with pytest.raises(ValueError):
        equilibrium_concentration(**valid)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thickness_m": -0.01},
        {"thickness_m": float("nan")},
        {"lambda_s": 0.0},
        {"emanation_fraction": 1.5},
        {"ra226_bq_kg": float("nan")},
    ],
)
def test_production_ceiling_rejects_out_of_domain_input(kwargs: dict) -> None:
    valid = {
        "ra226_bq_kg": 451.0,
        "dry_bulk_density_kg_m3": 1080.0,
        "emanation_fraction": 0.20,
        "lambda_s": LAMBDA_RN222,
        "thickness_m": 0.060,
    }
    valid.update(kwargs)
    with pytest.raises(ValueError):
        production_ceiling_sealed_bed(**valid)


def test_zero_emanation_gives_zero_source_and_is_not_clipped() -> None:
    assert radon_production_bulk(451.0, 1080.0, 0.0, LAMBDA_RN222) == 0.0
    assert radon_production_pore_gas(451.0, 1080.0, 0.0, 0.36, LAMBDA_RN222) == 0.0
    assert equilibrium_concentration(0.0, LAMBDA_RN222, 0.36) == 0.0


def test_no_nan_over_the_declared_domain() -> None:
    ra226_values = [0.0, 1.0, 35.0, 451.0, 1.0e4]
    density_values = [0.0, 1.0, 1011.0, 2650.0]
    emanation_values = [0.0, 0.01, 0.20, 0.80, 1.0]
    beta_values = [1.0e-4, 0.05, 0.36, 1.0]
    thickness_values = [0.0, 0.06, 3.0]

    for ra226 in ra226_values:
        for density in density_values:
            for emanation in emanation_values:
                bulk = radon_production_bulk(ra226, density, emanation, LAMBDA_RN222)
                assert math.isfinite(bulk) and bulk >= 0.0
                for beta in beta_values:
                    gas = radon_production_pore_gas(
                        ra226, density, emanation, beta, LAMBDA_RN222
                    )
                    assert math.isfinite(gas) and gas >= 0.0
                    c_eq = equilibrium_concentration(bulk, LAMBDA_RN222, beta)
                    assert math.isfinite(c_eq) and c_eq >= 0.0
                for thickness in thickness_values:
                    ceiling = production_ceiling_sealed_bed(
                        ra226, density, emanation, LAMBDA_RN222, thickness
                    )
                    assert math.isfinite(ceiling) and ceiling >= 0.0


def _legacy_beta(model: Any, theta: float, temperature_k: float) -> float:
    phi = model.p["phi_m"]
    henry = float(model._henry_rn([temperature_k])[0])
    return (phi - theta) + henry * theta


def test_regression_pore_gas_source_against_legacy_source_rn222() -> None:
    legacy = _load_legacy()
    theta = 0.12
    temperature_k = 288.15
    emanation = 0.20
    model = legacy.build_model(
        "tuff_unwelded",
        override={"rho_bulk": 1600.0, "Ra226_Bqkg": 35.0},
    )
    legacy_value = float(
        model.source_rn222([theta], [temperature_k], [1.0], [emanation])[0]
    )
    beta = _legacy_beta(model, theta, temperature_k)
    core_value = radon_production_pore_gas(
        ra226_bq_kg=model.p["Ra226_Bqkg"],
        dry_bulk_density_kg_m3=model.p["rho_bulk"],
        emanation_fraction=emanation,
        partition_porosity=beta,
        lambda_s=model.lam_rn,
    )
    assert math.isclose(core_value, legacy_value, rel_tol=1e-10)
    assert abs(core_value - legacy_value) / legacy_value < 1e-15


@pytest.mark.parametrize("theta", [0.02, 0.12, 0.25])
@pytest.mark.parametrize("emanation", [0.05, 0.20, 0.50])
def test_regression_pore_gas_source_over_a_grid(theta: float, emanation: float) -> None:
    legacy = _load_legacy()
    temperature_k = 283.15
    model = legacy.build_model(
        "tuff_unwelded",
        override={"rho_bulk": 1450.0, "Ra226_Bqkg": 60.0},
    )
    legacy_value = float(
        model.source_rn222([theta], [temperature_k], [1.0], [emanation])[0]
    )
    core_value = radon_production_pore_gas(
        ra226_bq_kg=model.p["Ra226_Bqkg"],
        dry_bulk_density_kg_m3=model.p["rho_bulk"],
        emanation_fraction=emanation,
        partition_porosity=_legacy_beta(model, theta, temperature_k),
        lambda_s=model.lam_rn,
    )
    assert math.isclose(core_value, legacy_value, rel_tol=1e-10)


def test_regression_bulk_source_is_the_legacy_numerator() -> None:
    legacy = _load_legacy()
    theta = 0.12
    temperature_k = 288.15
    emanation = 0.20
    model = legacy.build_model(
        "tuff_unwelded",
        override={"rho_bulk": 1600.0, "Ra226_Bqkg": 35.0},
    )
    beta = _legacy_beta(model, theta, temperature_k)
    legacy_gas = float(
        model.source_rn222([theta], [temperature_k], [1.0], [emanation])[0]
    )
    core_bulk = radon_production_bulk(
        ra226_bq_kg=model.p["Ra226_Bqkg"],
        dry_bulk_density_kg_m3=model.p["rho_bulk"],
        emanation_fraction=emanation,
        lambda_s=model.lam_rn,
    )
    assert math.isclose(core_bulk, legacy_gas * beta, rel_tol=1e-10)


def test_regression_equilibrium_concentration_against_legacy_deep_asymptote() -> None:
    legacy = _load_legacy()
    theta = 0.18
    temperature_k = 290.0
    emanation = 0.25
    model = legacy.build_model(
        "tuff_unwelded",
        override={"rho_bulk": 1600.0, "Ra226_Bqkg": 45.0},
    )
    beta = _legacy_beta(model, theta, temperature_k)
    legacy_c_eq = (
        float(model.source_rn222([theta], [temperature_k], [1.0], [emanation])[0])
        / model.lam_rn
    )
    core_c_eq = equilibrium_concentration(
        production_bulk_bq_m3_s=radon_production_bulk(
            model.p["Ra226_Bqkg"], model.p["rho_bulk"], emanation, model.lam_rn
        ),
        lambda_s=model.lam_rn,
        partition_porosity=beta,
    )
    assert math.isclose(core_c_eq, legacy_c_eq, rel_tol=1e-10)


def test_legacy_decay_constant_is_the_one_used_by_the_core() -> None:
    legacy = _load_legacy()
    model = legacy.build_model("tuff_unwelded")
    assert math.isclose(model.lam_rn, LAMBDA_RN222, rel_tol=1e-15)


def test_documented_difference_legacy_clips_emanation_low() -> None:
    legacy = _load_legacy()
    theta = 0.12
    temperature_k = 288.15
    model = legacy.build_model(
        "tuff_unwelded",
        override={"rho_bulk": 1600.0, "Ra226_Bqkg": 35.0},
    )
    legacy_at_zero = float(model.source_rn222([theta], [temperature_k], [1.0], [0.0])[0])
    legacy_at_floor = float(
        model.source_rn222([theta], [temperature_k], [1.0], [0.01])[0]
    )
    assert legacy_at_zero > 0.0
    assert math.isclose(legacy_at_zero, legacy_at_floor, rel_tol=1e-15)
    assert radon_production_pore_gas(35.0, 1600.0, 0.0, 0.36, model.lam_rn) == 0.0


def test_documented_difference_legacy_clips_emanation_high() -> None:
    legacy = _load_legacy()
    theta = 0.12
    temperature_k = 288.15
    model = legacy.build_model(
        "tuff_unwelded",
        override={"rho_bulk": 1600.0, "Ra226_Bqkg": 35.0},
    )
    beta = _legacy_beta(model, theta, temperature_k)
    legacy_at_one = float(model.source_rn222([theta], [temperature_k], [1.0], [1.0])[0])
    core_at_one = radon_production_pore_gas(
        35.0, 1600.0, 1.0, beta, model.lam_rn
    )
    assert math.isclose(core_at_one / legacy_at_one, 1.0 / 0.80, rel_tol=1e-12)


def test_documented_difference_accessibility_factor_is_not_in_the_core() -> None:
    legacy = _load_legacy()
    theta = 0.12
    temperature_k = 288.15
    emanation = 0.20
    accessibility = 0.6
    model = legacy.build_model(
        "tuff_unwelded",
        override={"rho_bulk": 1600.0, "Ra226_Bqkg": 35.0},
    )
    beta = _legacy_beta(model, theta, temperature_k)
    legacy_value = float(
        model.source_rn222(
            [theta], [temperature_k], [accessibility], [emanation]
        )[0]
    )
    core_value = radon_production_pore_gas(
        ra226_bq_kg=35.0,
        dry_bulk_density_kg_m3=1600.0,
        emanation_fraction=emanation * accessibility,
        partition_porosity=beta,
        lambda_s=model.lam_rn,
    )
    assert math.isclose(core_value, legacy_value, rel_tol=1e-10)
