from __future__ import annotations

import math

import numpy as np

from radon_x.config import ProjectConfig
from radon_x.dynamics.forcing import SourceTerm, TimeSeriesProfile
from radon_x.unione_delle_parti import CoreEngineOptions, core_flux_series


def soil_flux_profile(
    config: ProjectConfig,
    options: CoreEngineOptions | None = None,
) -> TimeSeriesProfile:
    times, fluxes = core_flux_series(config, options)
    valori = np.empty(times.size)
    valori[:-1] = 0.5 * (fluxes[:-1] + fluxes[1:])
    valori[-1] = fluxes[-1]
    return TimeSeriesProfile(
        times_s=tuple(float(t) for t in times),
        values=tuple(float(v) for v in valori),
    )


def soil_flux_source(
    zone: str,
    config: ProjectConfig,
    entry_area_m2: float,
    options: CoreEngineOptions | None = None,
) -> SourceTerm:
    if not math.isfinite(entry_area_m2) or entry_area_m2 <= 0.0:
        raise ValueError(f"area d'ingresso non fisica: {entry_area_m2!r}")
    profilo = soil_flux_profile(config, options)
    resolved = CoreEngineOptions() if options is None else options
    return SourceTerm(
        zone=zone,
        rate_bq_s=TimeSeriesProfile(
            times_s=profilo.times_s,
            values=tuple(v * entry_area_m2 for v in profilo.values),
        ),
        origin=(
            f"colonna del core sul sito {config.site.name!r}, "
            f"partenza {resolved.initial_condition!r}, area d'ingresso "
            f"{entry_area_m2:g} m2 (impronta, per conservazione); modalita' "
            f"disaccoppiata: la retroazione dell'edificio sul suolo non e' "
            f"rappresentata"
        ),
    )
