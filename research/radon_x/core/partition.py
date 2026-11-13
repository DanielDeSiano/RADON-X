from __future__ import annotations

import math

__all__ = [
    "OSTWALD_ENTHALPY_K",
    "OSTWALD_L0",
    "OSTWALD_REFERENCE_TEMPERATURE_K",
    "TEMPERATURE_MAX_K",
    "TEMPERATURE_MIN_K",
    "gas_filled_porosity",
    "ostwald_solubility",
    "partition_porosity",
    "water_content",
    "water_saturation",
]


OSTWALD_L0: float = 0.25

OSTWALD_ENTHALPY_K: float = 2790.0

OSTWALD_REFERENCE_TEMPERATURE_K: float = 293.15

TEMPERATURE_MIN_K: float = 250.0

TEMPERATURE_MAX_K: float = 350.0


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _require_porosity(porosity: float) -> float:
    porosity = _require_finite("porosity", porosity)
    if not 0.0 < porosity < 1.0:
        raise ValueError(f"porosity must be in (0, 1) m3 m-3, got {porosity!r}")
    return porosity


def _require_water_content(porosity: float, water_content_value: float) -> float:
    water_content_value = _require_finite("water_content", water_content_value)
    if water_content_value < 0.0:
        raise ValueError(f"water_content must be >= 0 m3 m-3, got {water_content_value!r}")
    if water_content_value > porosity:
        raise ValueError(
            "water_content must be <= porosity m3 m-3, got "
            f"water_content={water_content_value!r}, porosity={porosity!r}"
        )
    return water_content_value


def _require_saturation(saturation: float) -> float:
    saturation = _require_finite("saturation", saturation)
    if not 0.0 <= saturation <= 1.0:
        raise ValueError(f"saturation must be in [0, 1], got {saturation!r}")
    return saturation


def _require_temperature(temperature_k: float) -> float:
    temperature_k = _require_finite("temperature_k", temperature_k)
    if not TEMPERATURE_MIN_K <= temperature_k <= TEMPERATURE_MAX_K:
        raise ValueError(
            f"temperature_k must be in [{TEMPERATURE_MIN_K}, {TEMPERATURE_MAX_K}] K, "
            f"got {temperature_k!r}"
        )
    return temperature_k


def ostwald_solubility(temperature_k: float) -> float:
    temperature_k = _require_temperature(temperature_k)
    exponent = OSTWALD_ENTHALPY_K * (1.0 / temperature_k - 1.0 / OSTWALD_REFERENCE_TEMPERATURE_K)
    return OSTWALD_L0 * math.exp(exponent)


def gas_filled_porosity(porosity: float, water_content: float) -> float:
    porosity = _require_porosity(porosity)
    water_content = _require_water_content(porosity, water_content)
    return porosity - water_content


def partition_porosity(porosity: float, water_content: float, temperature_k: float) -> float:
    porosity = _require_porosity(porosity)
    water_content = _require_water_content(porosity, water_content)
    temperature_k = _require_temperature(temperature_k)
    return (porosity - water_content) + ostwald_solubility(temperature_k) * water_content


def water_saturation(porosity: float, water_content: float) -> float:
    porosity = _require_porosity(porosity)
    water_content = _require_water_content(porosity, water_content)
    return water_content / porosity


def water_content(porosity: float, saturation: float) -> float:
    porosity = _require_porosity(porosity)
    saturation = _require_saturation(saturation)
    return saturation * porosity
