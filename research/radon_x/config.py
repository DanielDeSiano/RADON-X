from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class SiteConfig:
    name: str
    lithotype: str = "tuff_unwelded"
    ra226_bq_kg: float = 185.0
    emanation_fraction: float = 0.31
    porosity: float = 0.42
    dry_bulk_density_kg_m3: float = 1612.0
    water_saturation: float = 0.11
    column_depth_m: float = 3.0
    initial_pore_accessibility: float = 1.0
    provenance: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("site.name must not be empty")
        if self.ra226_bq_kg < 0:
            raise ConfigurationError("ra226_bq_kg must be non-negative")
        if not 0.0 <= self.emanation_fraction <= 1.0:
            raise ConfigurationError("emanation_fraction must be between 0 and 1")
        if not 0.0 < self.porosity < 1.0:
            raise ConfigurationError("porosity must be between 0 and 1")
        if self.dry_bulk_density_kg_m3 <= 0:
            raise ConfigurationError("dry_bulk_density_kg_m3 must be positive")
        if not 0.0 <= self.water_saturation < 1.0:
            raise ConfigurationError("water_saturation must be in [0, 1)")
        if self.column_depth_m <= 0:
            raise ConfigurationError("column_depth_m must be positive")
        if not 0.0 <= self.initial_pore_accessibility <= 1.0:
            raise ConfigurationError("initial_pore_accessibility must be between 0 and 1")

    def legacy_overrides(self, cells: int) -> dict[str, float | int | str]:
        theta0 = self.water_saturation * self.porosity
        return {
            "nz": cells,
            "H": self.column_depth_m,
            "Ra226_Bqkg": self.ra226_bq_kg,
            "phi_m": self.porosity,
            "rho_bulk": self.dry_bulk_density_kg_m3,
            "theta0": theta0,
            "theta_r": min(0.02, 0.5 * theta0),
            "eps_max": self.emanation_fraction,
            "eps_dry": self.emanation_fraction,
            "grain_exp_eman": 0.0,
            "beta_T_eman": 0.0,
            "rain0": 0.0,
            "evap0": 1e-7,
            "seismic_site": "N/A",
            "A_k_close": 0.0,
            "dP_bottom": 0.0,
            "k_frac": 1e-13,
        }


@dataclass(frozen=True)
class SimulationConfig:
    cells: int = 30
    duration_days: float = 6.0
    output_step_seconds: float = 3600.0
    average_start_day: float = 4.5
    average_end_day: float = 5.5

    def validate(self) -> None:
        if self.cells < 3:
            raise ConfigurationError("simulation.cells must be at least 3")
        if self.duration_days <= 0:
            raise ConfigurationError("simulation.duration_days must be positive")
        if self.output_step_seconds <= 0:
            raise ConfigurationError("simulation.output_step_seconds must be positive")
        if not 0.0 <= self.average_start_day < self.average_end_day:
            raise ConfigurationError("averaging interval must have start < end")
        if self.average_end_day > self.duration_days:
            raise ConfigurationError("averaging interval must be inside the simulation")


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: str
    site: SiteConfig
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ConfigurationError(
                f"unsupported schema_version {self.schema_version!r}; expected '1.0'"
            )
        self.site.validate()
        self.simulation.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    CHIAVI_AMMESSE = frozenset({"schema_version", "site", "simulation"})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        ignote = set(data) - cls.CHIAVI_AMMESSE
        if ignote:
            raise ConfigurationError(
                f"unknown top-level keys: {sorted(ignote)}; "
                f"expected {sorted(cls.CHIAVI_AMMESSE)}"
            )
        try:
            config = cls(
                schema_version=str(data["schema_version"]),
                site=SiteConfig(**data["site"]),
                simulation=SimulationConfig(**data.get("simulation", {})),
            )
        except (KeyError, TypeError) as exc:
            raise ConfigurationError(f"invalid configuration structure: {exc}") from exc
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> ProjectConfig:
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot read configuration {source}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigurationError("configuration root must be a JSON object")
        return cls.from_dict(data)
