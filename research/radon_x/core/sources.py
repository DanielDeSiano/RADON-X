from __future__ import annotations

import math

from radon_x.core.decay import LAMBDA_RN222

__all__ = [
    "equilibrium_concentration",
    "production_ceiling_sealed_bed",
    "radon_production_bulk",
    "radon_production_pore_gas",
]


def _check_non_negative(name: str, value: float, unit: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite [{unit}], got {value!r}")
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0 [{unit}], got {value!r}")


def _check_positive(name: str, value: float, unit: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite [{unit}], got {value!r}")
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0 [{unit}], got {value!r}")


def _check_fraction(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite [-], got {value!r}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be within [0, 1] [-], got {value!r}")


def radon_production_bulk(
    ra226_bq_kg: float,
    dry_bulk_density_kg_m3: float,
    emanation_fraction: float,
    lambda_s: float = LAMBDA_RN222,
) -> float:
    _check_non_negative("ra226_bq_kg", ra226_bq_kg, "Bq kg-1")
    _check_non_negative("dry_bulk_density_kg_m3", dry_bulk_density_kg_m3, "kg m-3")
    _check_fraction("emanation_fraction", emanation_fraction)
    _check_positive("lambda_s", lambda_s, "s-1")
    return lambda_s * dry_bulk_density_kg_m3 * ra226_bq_kg * emanation_fraction


def radon_production_pore_gas(
    ra226_bq_kg: float,
    dry_bulk_density_kg_m3: float,
    emanation_fraction: float,
    partition_porosity: float,
    lambda_s: float = LAMBDA_RN222,
) -> float:
    _check_positive("partition_porosity", partition_porosity, "m3_gas m-3_bulk")
    production_bulk = radon_production_bulk(
        ra226_bq_kg=ra226_bq_kg,
        dry_bulk_density_kg_m3=dry_bulk_density_kg_m3,
        emanation_fraction=emanation_fraction,
        lambda_s=lambda_s,
    )
    return production_bulk / partition_porosity


def equilibrium_concentration(
    production_bulk_bq_m3_s: float,
    lambda_s: float,
    partition_porosity: float,
) -> float:
    _check_non_negative("production_bulk_bq_m3_s", production_bulk_bq_m3_s, "Bq m-3 s-1")
    _check_positive("lambda_s", lambda_s, "s-1")
    _check_positive("partition_porosity", partition_porosity, "m3_gas m-3_bulk")
    denominator = lambda_s * partition_porosity
    if denominator <= 0.0:
        raise ValueError(
            f"lambda_s * partition_porosity underflowed to {denominator!r}: "
            f"lambda_s={lambda_s!r} s-1 and partition_porosity={partition_porosity!r} "
            "are individually positive but their product is not representable."
        )
    return production_bulk_bq_m3_s / denominator


def production_ceiling_sealed_bed(
    ra226_bq_kg: float,
    dry_bulk_density_kg_m3: float,
    emanation_fraction: float,
    lambda_s: float,
    thickness_m: float,
) -> float:
    _check_positive("lambda_s", lambda_s, "s-1")
    _check_non_negative("thickness_m", thickness_m, "m")
    production_bulk = radon_production_bulk(
        ra226_bq_kg=ra226_bq_kg,
        dry_bulk_density_kg_m3=dry_bulk_density_kg_m3,
        emanation_fraction=emanation_fraction,
        lambda_s=lambda_s,
    )
    return production_bulk * thickness_m
