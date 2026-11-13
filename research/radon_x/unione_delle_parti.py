from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import numpy as np

from radon_x.config import ProjectConfig, SiteConfig
from radon_x.core.boundaries import BoundaryCondition, Dirichlet, ZeroFlux
from radon_x.core.decay import LAMBDA_RN222
from radon_x.core.diffusion import (
    D0_RN222_AIR_M2_S,
    apparent_diffusivity_rogers_nielson,
    bulk_diffusivity,
    radon_diffusion_length,
)
from radon_x.core.dual_porosity import exchange_rate_s, solve_steady_dual
from radon_x.core.indoor import (
    deep_pore_concentration,
    indoor_coupled_bq_m3,
    soil_transfer_velocity,
)
from radon_x.core.outputs import (
    REFERENCE_DEPTH_M,
    column_balance,
    column_mean_concentration,
    concentration_at_depth,
    surface_exhalation_flux,
)
from radon_x.core.partition import partition_porosity
from radon_x.core.solver import ColumnSolution, solve_steady, solve_transient
from radon_x.core.sources import equilibrium_concentration, radon_production_bulk
from radon_x.errors import ConfigError, NumericalError
from radon_x.platform_config import CHIAVE_PROVENIENZA, PlatformConfig
from radon_x.results import SimulationResult

__all__ = [
    "CELL_FOURIER_WARNING_THRESHOLD",
    "COLUMN_PROFILE_PREFIX",
    "COLUMN_SNAPSHOT_SCHEMA",
    "CORE_MODEL_VERSION",
    "DEFAULT_SOIL_TEMPERATURE_K",
    "ENGINES",
    "LEGACY_MODEL_VERSION",
    "CoreEngineError",
    "CoreEngineOptions",
    "EngineName",
    "UnknownEngineError",
    "aligned_with_legacy",
    "column_snapshot",
    "core_flux_series",
    "run_core_simulation",
    "run_simulation",
    "with_core_option",
]

ENGINES: tuple[str, ...] = ("legacy", "core")

EngineName = Literal["legacy", "core"]

CORE_MODEL_VERSION: str = "core-minimal-1d-diffusive"

LEGACY_MODEL_VERSION: str = "legacy-volcanic-radon-monolith"

DEFAULT_SOIL_TEMPERATURE_K: float = 288.0

CELL_FOURIER_WARNING_THRESHOLD: float = 1.0

COLUMN_PROFILE_PREFIX: str = "column_profile_bq_m3_"

COLUMN_SNAPSHOT_SCHEMA: str = "core-column-snapshot-1"

_SECONDS_PER_DAY: float = 86400.0

_SITE_FIELD_DEFAULTS: dict[str, Any] = {
    campo.name: campo.default for campo in fields(SiteConfig)
}


class UnknownEngineError(ConfigError):
    pass


class CoreEngineError(NumericalError, RuntimeError):
    pass


@dataclass(frozen=True)
class CoreEngineOptions:

    soil_temperature_k: float = DEFAULT_SOIL_TEMPERATURE_K
    d0_m2_s: float = D0_RN222_AIR_M2_S
    reference_depth_m: float = REFERENCE_DEPTH_M
    base_condition: Literal["sealed", "half_space"] = "sealed"
    initial_condition: Literal["steady", "zero", "given"] = "steady"
    building_volume_m3: float = 50.0
    building_footprint_m2: float = 50.0
    air_exchange_h: float = 0.10
    deposition_h: float = 0.0
    slab_conductance_m3_s: float | None = None
    compute_indoor: bool = True
    fracture_porosity: float | None = None
    fracture_water_content: float = 0.01
    matrix_fracture_area_m: float = 1.0
    matrix_fracture_distance_m: float = 0.02

    def validate(self) -> None:
        if not math.isfinite(self.soil_temperature_k):
            raise ValueError(
                f"soil_temperature_k must be finite [K], got {self.soil_temperature_k!r}"
            )
        if not math.isfinite(self.d0_m2_s) or self.d0_m2_s <= 0.0:
            raise ValueError(f"d0_m2_s must be > 0 [m2 s-1], got {self.d0_m2_s!r}")
        if not math.isfinite(self.reference_depth_m) or self.reference_depth_m <= 0.0:
            raise ValueError(
                f"reference_depth_m must be > 0 [m], got {self.reference_depth_m!r}"
            )
        if self.base_condition not in ("sealed", "half_space"):
            raise ValueError(
                "base_condition must be 'sealed' or 'half_space', "
                f"got {self.base_condition!r}"
            )
        if self.initial_condition not in ("steady", "zero", "given"):
            raise ValueError(
                "initial_condition must be 'steady', 'zero' or 'given', "
                f"got {self.initial_condition!r}"
            )
        for nome, valore in (
            ("building_volume_m3", self.building_volume_m3),
            ("building_footprint_m2", self.building_footprint_m2),
        ):
            if not math.isfinite(valore) or valore <= 0.0:
                raise ValueError(f"{nome} must be > 0, got {valore!r}")
        for nome, valore in (
            ("air_exchange_h", self.air_exchange_h),
            ("deposition_h", self.deposition_h),
        ):
            if not math.isfinite(valore) or valore < 0.0:
                raise ValueError(f"{nome} must be >= 0 [1/h], got {valore!r}")
        if self.fracture_porosity is not None:
            if not 0.0 < self.fracture_porosity <= 1.0:
                raise ValueError(
                    f"fracture_porosity fuori dominio: {self.fracture_porosity!r}"
                )
            if not 0.0 <= self.fracture_water_content < self.fracture_porosity:
                raise ValueError(
                    "fracture_water_content deve stare fra 0 e fracture_porosity, "
                    f"ricevuto {self.fracture_water_content!r}"
                )
        for nome, valore in (
            ("matrix_fracture_area_m", self.matrix_fracture_area_m),
            ("matrix_fracture_distance_m", self.matrix_fracture_distance_m),
        ):
            if not math.isfinite(valore) or valore <= 0.0:
                raise ValueError(f"{nome} must be > 0, got {valore!r}")
        if self.slab_conductance_m3_s is not None:
            k = self.slab_conductance_m3_s
            if not math.isfinite(k) or k < 0.0:
                raise ValueError(
                    f"slab_conductance_m3_s must be finite and >= 0 [m3 s-1], got {k!r}"
                )


def aligned_with_legacy(
    soil_temperature_k: float = 288.0375,
    fracture_porosity: float = 0.08,
    entry_area_m2: float = 0.5,
) -> CoreEngineOptions:
    return CoreEngineOptions(
        d0_m2_s=D0_RN222_AIR_M2_S * (soil_temperature_k / 293.15) ** 1.5,
        fracture_porosity=fracture_porosity,
        building_footprint_m2=entry_area_m2,
    )


@dataclass(frozen=True)
class _CoreColumn:

    cell_count: int
    dz_m: float
    water_content_m3_m3: float
    partition_porosity: float
    apparent_diffusivity_m2_s: float
    bulk_diffusivity_m2_s: float
    source_bulk_bq_m3_s: float
    equilibrium_concentration_bq_m3: float
    diffusion_length_m: float
    surface: BoundaryCondition
    base: BoundaryCondition


def _build_column(config: ProjectConfig, options: CoreEngineOptions) -> _CoreColumn:
    site = config.site
    cells = config.simulation.cells
    dz_m = site.column_depth_m / cells
    water_content = site.water_saturation * site.porosity
    beta = partition_porosity(site.porosity, water_content, options.soil_temperature_k)
    apparent = apparent_diffusivity_rogers_nielson(
        site.porosity, site.water_saturation, options.d0_m2_s
    )
    bulk = bulk_diffusivity(apparent, beta)
    source = radon_production_bulk(
        ra226_bq_kg=site.ra226_bq_kg,
        dry_bulk_density_kg_m3=site.dry_bulk_density_kg_m3,
        emanation_fraction=site.emanation_fraction,
    )
    c_eq = equilibrium_concentration(source, LAMBDA_RN222, beta)
    base: BoundaryCondition
    base = ZeroFlux() if options.base_condition == "sealed" else Dirichlet(c_eq)
    return _CoreColumn(
        cell_count=cells,
        dz_m=dz_m,
        water_content_m3_m3=water_content,
        partition_porosity=beta,
        apparent_diffusivity_m2_s=apparent,
        bulk_diffusivity_m2_s=bulk,
        source_bulk_bq_m3_s=source,
        equilibrium_concentration_bq_m3=c_eq,
        diffusion_length_m=radon_diffusion_length(apparent),
        surface=Dirichlet(0.0),
        base=base,
    )


def _output_times(config: ProjectConfig) -> tuple[float, int]:
    total_s = config.simulation.duration_days * _SECONDS_PER_DAY
    frames = round(total_s / config.simulation.output_step_seconds) + 1
    step_count = max(frames - 1, 1)
    return total_s / step_count, step_count


def _checked_initial_profile(
    initial_condition: str,
    profile: Sequence[float] | None,
    cells: int,
) -> tuple[float, ...] | None:
    if initial_condition != "given":
        if profile is not None:
            raise ValueError(
                "initial_profile_bq_m3 entra solo con initial_condition='given', "
                f"ricevuto initial_condition={initial_condition!r}"
            )
        return None
    if profile is None:
        raise ValueError(
            "initial_condition='given' richiede initial_profile_bq_m3, "
            "un valore per cella"
        )
    valori = tuple(float(valore) for valore in profile)
    if not valori:
        raise ValueError("initial_profile_bq_m3 non puo' essere vuoto")
    if len(valori) != cells:
        raise ValueError(
            f"initial_profile_bq_m3 ha {len(valori)} valori e la colonna ha "
            f"{cells} celle"
        )
    for indice, valore in enumerate(valori):
        if not math.isfinite(valore) or valore < 0.0:
            raise ValueError(
                f"initial_profile_bq_m3[{indice}] deve essere finito e >= 0 "
                f"[Bq m-3], ricevuto {valore!r}"
            )
    return valori


def _initial_profile(
    column: _CoreColumn,
    options: CoreEngineOptions,
    d_bulk: np.ndarray,
    beta: np.ndarray,
    source: np.ndarray,
    given: tuple[float, ...] | None,
) -> np.ndarray:
    if options.initial_condition == "zero":
        return np.zeros(column.cell_count, dtype=float)
    if options.initial_condition == "given":
        return np.asarray(given, dtype=float)
    steady = solve_steady(
        d_bulk_m2_s=d_bulk,
        beta=beta,
        source_bulk_bq_m3_s=source,
        dz_m=column.dz_m,
        lambda_s=LAMBDA_RN222,
        surface=column.surface,
        base=column.base,
    )
    return steady.concentration_bq_m3


def run_core_simulation(
    config: ProjectConfig,
    options: CoreEngineOptions | None = None,
    *,
    initial_profile_bq_m3: Sequence[float] | None = None,
    record_column_profile: bool = False,
) -> SimulationResult:
    config.validate()
    resolved = CoreEngineOptions() if options is None else options
    resolved.validate()
    profilo_dato = _checked_initial_profile(
        resolved.initial_condition, initial_profile_bq_m3, config.simulation.cells
    )

    warnings: list[str] = []
    column = _build_column(config, resolved)
    cells = column.cell_count

    d_bulk = np.full(cells, column.bulk_diffusivity_m2_s, dtype=float)
    beta = np.full(cells, column.partition_porosity, dtype=float)
    source = np.full(cells, column.source_bulk_bq_m3_s, dtype=float)

    step_s, step_count = _output_times(config)
    window_start_s = config.simulation.average_start_day * _SECONDS_PER_DAY
    window_end_s = config.simulation.average_end_day * _SECONDS_PER_DAY

    started = time.perf_counter()
    profile = _initial_profile(column, resolved, d_bulk, beta, source, profilo_dato)
    solution: ColumnSolution | None = None
    flux_samples: list[float] = []
    for index in range(step_count + 1):
        if index > 0:
            solution = solve_transient(
                d_bulk_m2_s=d_bulk,
                beta=beta,
                source_bulk_bq_m3_s=source,
                dz_m=column.dz_m,
                initial_bq_m3=profile,
                time_step_s=step_s,
                step_count=1,
                lambda_s=LAMBDA_RN222,
                surface=column.surface,
                base=column.base,
            )
            profile = solution.concentration_bq_m3
        sample_time_s = index * step_s
        if window_start_s <= sample_time_s <= window_end_s:
            if index == 0:
                flux_samples.append(
                    surface_exhalation_flux(
                        column.bulk_diffusivity_m2_s,
                        float(profile[0]),
                        column.dz_m,
                        column.surface,
                    )
                )
            else:
                flux_samples.append(float(solution.surface_flux_bq_m2_s))
    runtime_s = time.perf_counter() - started

    if solution is None:
        raise CoreEngineError("the core integration produced no time step")
    if not flux_samples:
        raise CoreEngineError(
            "averaging interval contains no output frame: "
            f"[{window_start_s}, {window_end_s}] s does not contain any multiple of "
            f"{step_s} s up to {step_count * step_s} s"
        )
    minimum = float(np.min(profile))
    if minimum < 0.0:
        raise CoreEngineError(
            f"the core solver returned a negative concentration, minimum {minimum!r} Bq m-3; "
            "the profile is reported as computed and is not clipped"
        )

    mean_flux = float(np.mean(flux_samples))
    flux_single_continuum = mean_flux
    dual = None
    if resolved.fracture_porosity is not None:
        theta_f = resolved.fracture_water_content
        d_apparent_f = apparent_diffusivity_rogers_nielson(
            resolved.fracture_porosity,
            theta_f / resolved.fracture_porosity,
            resolved.d0_m2_s,
        )
        beta_f = partition_porosity(
            resolved.fracture_porosity, theta_f, resolved.soil_temperature_k
        )
        alpha = exchange_rate_s(
            resolved.matrix_fracture_area_m,
            float(column.bulk_diffusivity_m2_s),
            beta_f * d_apparent_f,
            resolved.matrix_fracture_distance_m,
        )
        dual = solve_steady_dual(
            cell_count=column.cell_count,
            thickness_m=column.cell_count * column.dz_m,
            matrix_bulk_diffusivity_m2_s=float(column.bulk_diffusivity_m2_s),
            fracture_bulk_diffusivity_m2_s=beta_f * d_apparent_f,
            matrix_partition_porosity=float(column.partition_porosity),
            fracture_partition_porosity=beta_f,
            source_bulk_bq_m3_s=float(column.source_bulk_bq_m3_s),
            exchange_s=alpha,
            lambda_s=LAMBDA_RN222,
        )
        mean_flux = dual.surface_flux_bq_m2_s
        profile = dual.matrix_bq_m3

    balance = column_balance(
        concentration_bq_m3=profile,
        partition_porosity=beta,
        source_bulk_bq_m3_s=source,
        d_bulk_m2_s=d_bulk,
        dz_m=column.dz_m,
        lambda_s=LAMBDA_RN222,
        surface=column.surface,
        base=column.base,
    )
    storage_rate = solution.balance.storage_rate_bq_m2_s
    fourier = solution.max_cell_fourier_number
    if fourier is None:
        raise CoreEngineError("the transient solver returned no Fourier diagnostic")
    fourier_number = float(fourier)
    budget_flux = (
        balance["production_bq_m2_s"]
        + balance["base_flux_bq_m2_s"]
        - balance["decay_bq_m2_s"]
        - storage_rate
    )
    face_flux = balance["surface_flux_bq_m2_s"]
    if dual is not None:
        decay_fracture = float(
            LAMBDA_RN222
            * beta_f
            * float(np.sum(dual.fracture_bq_m3))
            * column.dz_m
        )
        budget_flux = (
            balance["production_bq_m2_s"]
            + balance["base_flux_bq_m2_s"]
            - balance["decay_bq_m2_s"]
            - decay_fracture
            - storage_rate
        )
        face_flux = mean_flux
    budget_gap_ratio = abs(budget_flux - face_flux) / max(abs(face_flux), 1e-30)

    depths_min = 0.5 * column.dz_m
    depths_max = column.dz_m * (cells - 0.5)
    if depths_min <= resolved.reference_depth_m <= depths_max:
        at_reference = concentration_at_depth(
            resolved.reference_depth_m, profile, column.dz_m
        )
    else:
        at_reference = float("nan")
        warnings.append(
            f"profondita' di riferimento {resolved.reference_depth_m} m fuori dalla "
            f"colonna: i centri cella vanno da {depths_min} m a {depths_max} m, "
            "soil_at_reference_depth_bq_m3 = NaN"
        )
    if config.site.lithotype != _SITE_FIELD_DEFAULTS["lithotype"]:
        warnings.append(
            f"litotipo dichiarato {config.site.lithotype!r}: il motore core non legge "
            "il campo, i coefficienti della colonna vengono dai campi espliciti della "
            "configurazione"
        )
    accessibilita = config.site.initial_pore_accessibility
    if accessibilita != _SITE_FIELD_DEFAULTS["initial_pore_accessibility"]:
        warnings.append(
            f"initial_pore_accessibility {accessibilita}: il motore core non applica "
            "l'accessibilita' dei pori, la frazione di emanazione entra intera al "
            f"valore {config.site.emanation_fraction}"
        )
    if (
        fourier_number > CELL_FOURIER_WARNING_THRESHOLD
        and resolved.initial_condition != "steady"
    ):
        warnings.append(
            f"numero di Fourier di cella {fourier_number} sopra "
            f"{CELL_FOURIER_WARNING_THRESHOLD}: il passo d'uscita {step_s} s supera il "
            f"tempo diffusivo di cella {step_s / fourier_number} s, condizione "
            f"iniziale {resolved.initial_condition!r}"
        )

    indoor = upper_bound = sub_slab = deep_pore = float("nan")
    if resolved.compute_indoor:
        velocity = soil_transfer_velocity(
            config.site.porosity, config.site.water_saturation, resolved.d0_m2_s
        )
        deep_pore = deep_pore_concentration(mean_flux, velocity)
        upper_bound = indoor_coupled_bq_m3(
            mean_flux,
            config.site.porosity,
            config.site.water_saturation,
            resolved.building_footprint_m2,
            resolved.building_volume_m3,
            resolved.air_exchange_h,
            math.inf,
            resolved.deposition_h,
        )
        conductance = (
            math.inf
            if resolved.slab_conductance_m3_s is None
            else resolved.slab_conductance_m3_s
        )
        indoor = (
            upper_bound
            if resolved.slab_conductance_m3_s is None
            else indoor_coupled_bq_m3(
                mean_flux,
                config.site.porosity,
                config.site.water_saturation,
                resolved.building_footprint_m2,
                resolved.building_volume_m3,
                resolved.air_exchange_h,
                conductance,
                resolved.deposition_h,
            )
        )
        if math.isfinite(conductance) and conductance > 0.0:
            lam_eff = LAMBDA_RN222 + (
                resolved.air_exchange_h + resolved.deposition_h
            ) / 3600.0
            sub_slab = indoor * (
                1.0 + resolved.building_volume_m3 * lam_eff / conductance
            )
        else:
            sub_slab = indoor

    profilo_registrato: dict[str, float] = {}
    if record_column_profile:
        cifre = len(str(max(cells - 1, 0)))
        profilo_registrato = {
            f"{COLUMN_PROFILE_PREFIX}{indice:0{cifre}d}": float(valore)
            for indice, valore in enumerate(profile)
        }

    diagnostica_doppia: dict[str, float] = {}
    if dual is not None:
        diagnostica_doppia = {
            "dual_porosity_matrix_flux_bq_m2_s": dual.matrix_flux_bq_m2_s,
            "dual_porosity_fracture_flux_bq_m2_s": dual.fracture_flux_bq_m2_s,
            "dual_porosity_exchange_rate_s": dual.exchange_rate_s,
        }

    return SimulationResult(
        schema_version="1.0",
        model_version=CORE_MODEL_VERSION,
        site_name=config.site.name,
        runtime_seconds=runtime_s,
        surface_flux_bq_m2_s=mean_flux,
        surface_flux_bq_m2_day=mean_flux * _SECONDS_PER_DAY,
        soil_mean_bq_m3=column_mean_concentration(profile),
        soil_at_reference_depth_bq_m3=at_reference,
        reference_depth_m=resolved.reference_depth_m,
        thoron_flux_bq_m2_s=float("nan"),
        indoor_bq_m3=indoor,
        indoor_upper_bound_bq_m3=upper_bound,
        sub_slab_bq_m3=sub_slab,
        deep_pore_bq_m3=deep_pore,
        diagnostics={
            "final_surface_flux_bq_m2_s": face_flux,
            "budget_residual_bq_m2_s": budget_flux,
            "budget_gap_ratio": budget_gap_ratio,
            "time_points": float(step_count + 1),
            "flux_single_continuum_bq_m2_s": flux_single_continuum,
            **diagnostica_doppia,
            **profilo_registrato,
            "state_variables": float(cells),
            "production_bq_m2_s": balance["production_bq_m2_s"],
            "decay_bq_m2_s": balance["decay_bq_m2_s"],
            "base_flux_bq_m2_s": balance["base_flux_bq_m2_s"],
            "storage_rate_bq_m2_s": storage_rate,
            "balance_residual_bq_m2_s": solution.balance.residual_bq_m2_s,
            "relative_balance_residual": solution.balance.relative_residual,
            "partition_porosity": column.partition_porosity,
            "water_content_m3_m3": column.water_content_m3_m3,
            "apparent_diffusivity_m2_s": column.apparent_diffusivity_m2_s,
            "bulk_diffusivity_m2_s": column.bulk_diffusivity_m2_s,
            "source_bulk_bq_m3_s": column.source_bulk_bq_m3_s,
            "equilibrium_concentration_bq_m3": column.equilibrium_concentration_bq_m3,
            "diffusion_length_m": column.diffusion_length_m,
            "cell_size_m": column.dz_m,
            "time_step_s": step_s,
            "max_cell_fourier_number": fourier_number,
            "minimum_concentration_bq_m3": minimum,
            "soil_temperature_k": resolved.soil_temperature_k,
        },
        warnings=tuple(warnings),
        input_config=config.to_dict(),
    )


def column_snapshot(result: SimulationResult) -> dict[str, Any]:
    celle: list[tuple[int, float]] = []
    for chiave, valore in result.diagnostics.items():
        if not chiave.startswith(COLUMN_PROFILE_PREFIX):
            continue
        coda = chiave[len(COLUMN_PROFILE_PREFIX) :]
        if not coda.isdigit():
            raise ValueError(f"chiave di profilo non numerata: {chiave!r}")
        celle.append((int(coda), float(valore)))
    if not celle:
        raise ValueError(
            "il risultato non porta il profilo della colonna: la corsa che lo produce "
            "deve dichiarare run_core_simulation(record_column_profile=True); "
            f"model_version {result.model_version!r}"
        )
    celle.sort()
    if [indice for indice, _ in celle] != list(range(len(celle))):
        raise ValueError(
            "le celle del profilo non sono numerate da 0 senza salti: "
            f"{[indice for indice, _ in celle]}"
        )
    concentrazioni = [valore for _, valore in celle]
    dz_m = float(result.diagnostics["cell_size_m"])
    return {
        "schema": COLUMN_SNAPSHOT_SCHEMA,
        "site_name": result.site_name,
        "model_version": result.model_version,
        "cell_count": len(concentrazioni),
        "cell_size_m": dz_m,
        "column_depth_m": dz_m * len(concentrazioni),
        "concentration_bq_m3": concentrazioni,
    }


def core_flux_series(
    config: ProjectConfig,
    options: CoreEngineOptions | None = None,
    *,
    initial_profile_bq_m3: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    config.validate()
    resolved = CoreEngineOptions() if options is None else options
    resolved.validate()
    profilo_dato = _checked_initial_profile(
        resolved.initial_condition, initial_profile_bq_m3, config.simulation.cells
    )

    column = _build_column(config, resolved)
    cells = column.cell_count
    d_bulk = np.full(cells, column.bulk_diffusivity_m2_s, dtype=float)
    beta = np.full(cells, column.partition_porosity, dtype=float)
    source = np.full(cells, column.source_bulk_bq_m3_s, dtype=float)

    step_s, step_count = _output_times(config)
    profile = _initial_profile(column, resolved, d_bulk, beta, source, profilo_dato)

    times = np.empty(step_count + 1)
    fluxes = np.empty(step_count + 1)
    times[0] = 0.0
    fluxes[0] = surface_exhalation_flux(
        column.bulk_diffusivity_m2_s, float(profile[0]), column.dz_m, column.surface
    )
    for index in range(1, step_count + 1):
        solution = solve_transient(
            d_bulk_m2_s=d_bulk,
            beta=beta,
            source_bulk_bq_m3_s=source,
            dz_m=column.dz_m,
            initial_bq_m3=profile,
            time_step_s=step_s,
            step_count=1,
            lambda_s=LAMBDA_RN222,
            surface=column.surface,
            base=column.base,
        )
        profile = solution.concentration_bq_m3
        times[index] = index * step_s
        fluxes[index] = float(solution.surface_flux_bq_m2_s)

    minimum = float(np.min(profile))
    if minimum < 0.0:
        raise CoreEngineError(
            f"the core solver returned a negative concentration, minimum {minimum!r} Bq m-3; "
            "the profile is reported as computed and is not clipped"
        )
    return times, fluxes


def run_simulation(
    config: ProjectConfig,
    engine: EngineName = "legacy",
    core_options: CoreEngineOptions | None = None,
    platform: PlatformConfig | None = None,
) -> SimulationResult:
    if engine not in ENGINES:
        raise UnknownEngineError(
            f"unknown engine {engine!r}; expected one of {', '.join(ENGINES)}"
        )
    piattaforma = None if platform is None else PlatformConfig.risolta(platform)
    if engine == "core":
        risultato = run_core_simulation(config, core_options)
    elif core_options is not None:
        raise ValueError(
            "core_options applies to the core engine only; the legacy engine takes its "
            "parameters from its own lithotype tables and would ignore them"
        )
    else:
        from radon_x.legacy_adapter import run_legacy_simulation

        risultato = run_legacy_simulation(config)
    if piattaforma is None:
        return risultato
    return replace(
        risultato,
        input_config={
            **risultato.input_config,
            CHIAVE_PROVENIENZA: piattaforma.echo(),
        },
    )


def with_core_option(options: CoreEngineOptions | None, **changes: object) -> CoreEngineOptions:
    base = CoreEngineOptions() if options is None else options
    updated = replace(base, **changes)
    updated.validate()
    return updated
