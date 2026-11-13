"""Adapter between the browser worker and the deterministic scientific core."""

from __future__ import annotations

import json
import math
from typing import Any

from radon_x.config import ProjectConfig
from radon_x.unione_delle_parti import (
    CoreEngineOptions,
    core_flux_series,
    run_core_simulation,
)


def _finite(value: Any, fallback: float | None = None) -> float | None:
    """Return JSON-safe finite floats and discard NaN/Infinity."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy/scientific values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _build_options(payload: dict[str, Any]) -> CoreEngineOptions:
    """Translate optional UI controls into the existing core options."""
    options = payload.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    allowed = {
        "soil_temperature_k",
        "d0_m2_s",
        "reference_depth_m",
        "base_condition",
        "initial_condition",
        "compute_indoor",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key in allowed:
        if key in options:
            kwargs[key] = options[key]
    # Keep the UI/API wording friendly while matching the core's typed enum.
    if kwargs.get("base_condition") == "equilibrium":
        kwargs["base_condition"] = "half_space"
    return CoreEngineOptions(**kwargs)


def simulate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the deterministic core and return a compact UI-oriented response."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    config_data = payload.get("config")
    if not isinstance(config_data, dict):
        raise ValueError("config must be an object")
    config = ProjectConfig.from_dict(config_data)
    if config.simulation.cells > 240:
        raise ValueError("cells must be <= 240 for the local interactive lab")
    if config.simulation.duration_days > 31:
        raise ValueError("duration_days must be <= 31 for the local interactive lab")
    if config.simulation.duration_days * 86400 / config.simulation.output_step_seconds > 50000:
        raise ValueError("Increase the output step to keep the simulation below 50,000 intervals.")
    options = _build_options(payload)
    result = run_core_simulation(config, options, record_column_profile=True)
    times, fluxes = core_flux_series(config, options)
    diagnostics = result.diagnostics
    profile = []
    for index in range(config.simulation.cells):
        key = f"column_profile_bq_m3_{index:02d}"
        value = _finite(diagnostics.get(key))
        if value is not None:
            profile.append(
                {
                    "depth_m": (index + 0.5) * config.site.column_depth_m / config.simulation.cells,
                    "concentration_bq_m3": value,
                }
            )
    series = [
        {"day": float(time_s) / 86400.0, "flux_bq_m2_s": _finite(flux)}
        for time_s, flux in zip(times, fluxes)
    ]
    response = result.to_dict()
    response["diagnostics"] = {
        key: _json_safe(value)
        for key, value in diagnostics.items()
        if not key.startswith("column_profile_bq_m3_")
    }
    response["series"] = series
    response["column_profile"] = profile
    response["request"] = {"config": config.to_dict(), "options": options.__dict__}
    return _json_safe(response)

