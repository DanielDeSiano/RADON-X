from __future__ import annotations

import importlib.util
import os
import time
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from .config import ProjectConfig
from .errors import DataError
from .legacy_layout import LegacyStateLayout
from .results import SimulationResult

LEGACY_REFERENCE_DEPTH_M: float = 0.8


class LegacyModelError(DataError, RuntimeError):
    pass


def legacy_source_path() -> Path:
    configured = os.environ.get("RADON_X_LEGACY_MODEL")
    if configured:
        return Path(configured).expanduser().resolve()
    workspace_root = Path(__file__).resolve().parents[1]
    return workspace_root / "radon_suolo_parte1_finale.py"


@lru_cache(maxsize=1)
def load_legacy_module() -> ModuleType:
    path = legacy_source_path()
    if not path.is_file():
        raise LegacyModelError(
            f"legacy model not found at {path}; set RADON_X_LEGACY_MODEL if it was moved"
        )
    spec = importlib.util.spec_from_file_location("radon_x_legacy_model", path)
    if spec is None or spec.loader is None:
        raise LegacyModelError(f"cannot create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise LegacyModelError(f"legacy model import failed: {exc}") from exc
    return module


def _install_initial_accessibility(model: object, value: float) -> None:
    layout = LegacyStateLayout.from_model(model)
    original = model.build_y0

    def build_y0_with_explicit_accessibility() -> np.ndarray:
        y0 = original()
        y0[layout.slice_of("Ar")] = value
        return y0

    model.build_y0 = build_y0_with_explicit_accessibility


def run_legacy_simulation(config: ProjectConfig) -> SimulationResult:
    config.validate()
    legacy = load_legacy_module()
    simulation = config.simulation
    site = config.site

    model = legacy.build_model(
        site.lithotype,
        override=site.legacy_overrides(simulation.cells),
    )
    _install_initial_accessibility(model, site.initial_pore_accessibility)

    started = time.perf_counter()
    try:
        solution = model.simulate_fast(
            days=simulation.duration_days,
            dt_out=simulation.output_step_seconds,
        )
    except Exception as exc:
        raise LegacyModelError(f"legacy simulation failed: {exc}") from exc
    runtime = time.perf_counter() - started

    if not getattr(solution, "success", False):
        raise LegacyModelError(
            f"legacy solver reported failure: {getattr(solution, 'message', 'no message')}"
        )
    if solution.y.size == 0 or solution.t.size == 0:
        raise LegacyModelError("legacy solver returned an empty solution")
    if not np.isfinite(solution.y).all() or not np.isfinite(solution.t).all():
        raise LegacyModelError("legacy solver returned NaN or infinite values")

    start_s = simulation.average_start_day * 86400.0
    end_s = simulation.average_end_day * 86400.0
    indices = np.where((solution.t >= start_s) & (solution.t <= end_s))[0]
    if indices.size == 0:
        raise LegacyModelError("averaging interval contains no solver output")

    flux_samples: list[float] = []
    for index in indices:
        partial = SimpleNamespace(
            t=solution.t[: index + 1],
            y=solution.y[:, : index + 1],
        )
        flux = legacy.soil_surface_flux(model, partial, verbose=False)
        flux_samples.append(float(flux["J_surface"]))

    mean_surface_flux = float(np.mean(flux_samples))
    if not np.isfinite(mean_surface_flux) or mean_surface_flux < 0:
        raise LegacyModelError("legacy model returned a non-physical mean surface flux")

    outputs = legacy.soil_outputs(model, solution, z_ref=LEGACY_REFERENCE_DEPTH_M, verbose=False)
    flux_terms = outputs.get("_flux_terms", {})
    final_surface_flux = float(flux_terms.get("J_surface", np.nan))
    budget_residual = float(flux_terms.get("budget_residual", np.nan))
    budget_gap_ratio = float("nan")
    if np.isfinite(final_surface_flux) and np.isfinite(budget_residual):
        budget_gap_ratio = abs(budget_residual - final_surface_flux) / max(
            abs(final_surface_flux), 1e-30
        )


    raw_thoron = outputs.get("J_thoron_Bq_m2_s")
    if raw_thoron is None:
        thoron_flux = float("nan")
    else:
        thoron_flux = float(raw_thoron)

    warnings: list[str] = []
    nodi = np.asarray(model.z, dtype=float)
    if nodi.size and not (
        float(nodi.min()) <= LEGACY_REFERENCE_DEPTH_M <= float(nodi.max())
    ):
        letto_a = float(nodi[int(np.argmin(np.abs(nodi - LEGACY_REFERENCE_DEPTH_M)))])
        warnings.append(
            f"profondita' di riferimento {LEGACY_REFERENCE_DEPTH_M} m fuori dalla "
            f"colonna: i nodi della griglia legacy vanno da {float(nodi.min())} m a "
            f"{float(nodi.max())} m, soil_at_reference_depth_bq_m3 e' il valore a "
            f"{letto_a} m, riportato con reference_depth_m = "
            f"{LEGACY_REFERENCE_DEPTH_M} m"
        )

    return SimulationResult(
        schema_version="1.0",
        model_version="legacy-volcanic-radon-monolith",
        site_name=site.name,
        runtime_seconds=runtime,
        surface_flux_bq_m2_s=mean_surface_flux,
        surface_flux_bq_m2_day=mean_surface_flux * 86400.0,
        soil_mean_bq_m3=float(outputs["C_mean_kBq_m3"]) * 1000.0,
        soil_at_reference_depth_bq_m3=float(outputs["C_at_zref_kBq_m3"]) * 1000.0,
        reference_depth_m=float(outputs["z_ref_m"]),
        thoron_flux_bq_m2_s=thoron_flux,
        diagnostics={
            "final_surface_flux_bq_m2_s": final_surface_flux,
            "budget_residual_bq_m2_s": budget_residual,
            "budget_gap_ratio": budget_gap_ratio,
            "time_points": float(solution.t.size),
            "state_variables": float(solution.y.shape[0]),
        },
        warnings=tuple(warnings),
        input_config=config.to_dict(),
    )
