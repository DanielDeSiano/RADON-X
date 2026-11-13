from __future__ import annotations

import math

from radon_x.core.decay import LAMBDA_RN222, diffusion_length

__all__ = [
    "D0_RN222_AIR_M2_S",
    "apparent_diffusivity_from_bulk",
    "apparent_diffusivity_rogers_nielson",
    "bulk_diffusivity",
    "radon_diffusion_length",
]


D0_RN222_AIR_M2_S: float = 1.1e-5


def _check_unit_interval(name: str, value: float, *, open_low: bool, open_high: bool) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if open_low:
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0 (dimensionless), got {value!r}")
    elif value < 0.0:
        raise ValueError(f"{name} must be >= 0 (dimensionless), got {value!r}")
    if open_high:
        if value >= 1.0:
            raise ValueError(f"{name} must be < 1 (dimensionless), got {value!r}")
    elif value > 1.0:
        raise ValueError(f"{name} must be <= 1 (dimensionless), got {value!r}")


def apparent_diffusivity_rogers_nielson(
    porosity: float,
    saturation: float,
    d0_m2_s: float = D0_RN222_AIR_M2_S,
) -> float:
    _check_unit_interval("porosity", porosity, open_low=True, open_high=True)
    _check_unit_interval("saturation", saturation, open_low=False, open_high=False)
    if not math.isfinite(d0_m2_s):
        raise ValueError(f"d0_m2_s must be finite, got {d0_m2_s!r}")
    if d0_m2_s <= 0.0:
        raise ValueError(f"d0_m2_s must be > 0 m2 s-1, got {d0_m2_s!r}")

    exponent = -6.0 * saturation * porosity - 6.0 * saturation ** (14.0 * porosity)
    return porosity * d0_m2_s * math.exp(exponent)


def bulk_diffusivity(apparent_d_m2_s: float, partition_porosity: float) -> float:
    if not math.isfinite(apparent_d_m2_s):
        raise ValueError(f"apparent_d_m2_s must be finite, got {apparent_d_m2_s!r}")
    if apparent_d_m2_s <= 0.0:
        raise ValueError(f"apparent_d_m2_s must be > 0 m2 s-1, got {apparent_d_m2_s!r}")
    if not math.isfinite(partition_porosity):
        raise ValueError(f"partition_porosity must be finite, got {partition_porosity!r}")
    if partition_porosity <= 0.0:
        raise ValueError(
            f"partition_porosity must be > 0 (dimensionless), got {partition_porosity!r}"
        )
    return partition_porosity * apparent_d_m2_s


def apparent_diffusivity_from_bulk(bulk_d_m2_s: float, partition_porosity: float) -> float:
    if not math.isfinite(bulk_d_m2_s):
        raise ValueError(f"bulk_d_m2_s must be finite, got {bulk_d_m2_s!r}")
    if bulk_d_m2_s <= 0.0:
        raise ValueError(f"bulk_d_m2_s must be > 0 m2 s-1, got {bulk_d_m2_s!r}")
    if not math.isfinite(partition_porosity):
        raise ValueError(f"partition_porosity must be finite, got {partition_porosity!r}")
    if partition_porosity <= 0.0:
        raise ValueError(
            f"partition_porosity must be > 0 (dimensionless), got {partition_porosity!r}"
        )
    return bulk_d_m2_s / partition_porosity


def radon_diffusion_length(
    apparent_d_m2_s: float,
    lambda_s: float = LAMBDA_RN222,
) -> float:
    return diffusion_length(apparent_d_m2_s, lambda_s)
