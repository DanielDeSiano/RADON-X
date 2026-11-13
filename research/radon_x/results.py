from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SimulationResult:
    schema_version: str
    model_version: str
    site_name: str
    runtime_seconds: float
    surface_flux_bq_m2_s: float
    surface_flux_bq_m2_day: float
    soil_mean_bq_m3: float
    soil_at_reference_depth_bq_m3: float
    reference_depth_m: float
    thoron_flux_bq_m2_s: float
    indoor_bq_m3: float = float("nan")
    indoor_upper_bound_bq_m3: float = float("nan")
    sub_slab_bq_m3: float = float("nan")
    deep_pore_bq_m3: float = float("nan")
    diagnostics: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    input_config: dict[str, Any] = field(default_factory=dict)
    solver_report: dict[str, Any] = field(default_factory=dict)
    indoor_effective_series: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data
