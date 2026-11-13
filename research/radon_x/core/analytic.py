from __future__ import annotations

import math

from radon_x.core.decay import diffusion_length

__all__ = [
    "steady_profile_finite_slab",
    "steady_profile_semi_infinite",
    "surface_flux_finite_slab",
    "surface_flux_semi_infinite",
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


def _cosh_ratio(numerator_arg: float, denominator_arg: float) -> float:
    a = numerator_arg
    b = denominator_arg
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(
            "cosh ratio arguments must be finite; got "
            f"numerator_arg={a!r}, denominator_arg={b!r}. This happens when "
            "thickness_m / diffusion_length_m exceeds the float range."
        )
    return math.exp(a - b) * (1.0 + math.exp(-2.0 * a)) / (1.0 + math.exp(-2.0 * b))


def steady_profile_semi_infinite(
    depth_m: float,
    c_eq_bq_m3: float,
    diffusion_length_m: float,
) -> float:
    _require_non_negative("depth_m", depth_m, "m")
    _require_non_negative("c_eq_bq_m3", c_eq_bq_m3, "Bq m-3")
    _require_positive("diffusion_length_m", diffusion_length_m, "m")
    return c_eq_bq_m3 * (1.0 - math.exp(-depth_m / diffusion_length_m))


def steady_profile_finite_slab(
    depth_m: float,
    c_eq_bq_m3: float,
    diffusion_length_m: float,
    thickness_m: float,
) -> float:
    _require_non_negative("depth_m", depth_m, "m")
    _require_non_negative("c_eq_bq_m3", c_eq_bq_m3, "Bq m-3")
    _require_positive("diffusion_length_m", diffusion_length_m, "m")
    _require_positive("thickness_m", thickness_m, "m")
    if depth_m > thickness_m:
        raise ValueError(
            f"depth_m must be <= thickness_m ({thickness_m!r} m) inside the slab, got {depth_m!r}"
        )
    ratio = _cosh_ratio(
        (thickness_m - depth_m) / diffusion_length_m,
        thickness_m / diffusion_length_m,
    )
    return c_eq_bq_m3 * (1.0 - ratio)


def surface_flux_semi_infinite(
    ra226_bq_kg: float,
    dry_bulk_density_kg_m3: float,
    emanation_fraction: float,
    lambda_s: float,
    d_apparent_m2_s: float,
) -> float:
    _require_non_negative("ra226_bq_kg", ra226_bq_kg, "Bq kg-1")
    _require_positive("dry_bulk_density_kg_m3", dry_bulk_density_kg_m3, "kg m-3")
    _require_finite("emanation_fraction", emanation_fraction)
    if not 0.0 <= emanation_fraction <= 1.0:
        raise ValueError(f"emanation_fraction must be in [0, 1], got {emanation_fraction!r}")
    _require_positive("lambda_s", lambda_s, "s-1")
    _require_positive("d_apparent_m2_s", d_apparent_m2_s, "m2 s-1")
    return (
        dry_bulk_density_kg_m3
        * ra226_bq_kg
        * emanation_fraction
        * math.sqrt(lambda_s * d_apparent_m2_s)
    )


def surface_flux_finite_slab(
    ra226_bq_kg: float,
    dry_bulk_density_kg_m3: float,
    emanation_fraction: float,
    lambda_s: float,
    d_apparent_m2_s: float,
    thickness_m: float,
) -> float:
    _require_positive("thickness_m", thickness_m, "m")
    semi_infinite = surface_flux_semi_infinite(
        ra226_bq_kg=ra226_bq_kg,
        dry_bulk_density_kg_m3=dry_bulk_density_kg_m3,
        emanation_fraction=emanation_fraction,
        lambda_s=lambda_s,
        d_apparent_m2_s=d_apparent_m2_s,
    )
    l_d = diffusion_length(d_apparent_m2_s, lambda_s)
    return semi_infinite * math.tanh(thickness_m / l_d)
