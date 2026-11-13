from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from radon_x.core.decay import LAMBDA_RN222
from radon_x.core.diffusion import (
    D0_RN222_AIR_M2_S,
    apparent_diffusivity_rogers_nielson,
)
from radon_x.data_manifest import assert_usable_for_fitting, data_directory

PREDICTORS = (
    "ra226_bq_kg",
    "emanation",
    "porosity",
    "saturation",
    "dry_bulk_density",
)


ZHUO_PLATEAU_A = 1.5


@dataclass(frozen=True)
class Site:

    source_file: str
    site_id: str
    ra226_bq_kg: float
    emanation: float
    porosity: float
    saturation: float
    dry_bulk_density: float
    measured_flux_bq_m2_s: float
    emanation_state: str = "undeclared"

    def features(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in PREDICTORS)

    def field_emanation(self, plateau_a: float = ZHUO_PLATEAU_A) -> float:
        if self.emanation_state == "dry":
            return self.emanation * (1.0 + plateau_a)
        return self.emanation

    def physical_flux_bq_m2_s(
        self,
        d0_m2_s: float = D0_RN222_AIR_M2_S,
        plateau_a: float = ZHUO_PLATEAU_A,
        use_emanation_state: bool = True,
    ) -> float:
        d_apparent = apparent_diffusivity_rogers_nielson(
            self.porosity, self.saturation, d0_m2_s
        )
        emanation = self.field_emanation(plateau_a) if use_emanation_state else self.emanation
        return (
            self.dry_bulk_density
            * self.ra226_bq_kg
            * emanation
            * (LAMBDA_RN222 * d_apparent) ** 0.5
        )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def _load_voltaggio(root: Path) -> list[Site]:
    name = "voltaggio2006_lazio_6siti.csv"
    assert_usable_for_fitting(name)
    sites = []
    for row in _rows(root / name):
        sites.append(
            Site(
                source_file=name,
                site_id=f"voltaggio-{row['site_id']}",
                ra226_bq_kg=float(row["Ra226_Bq_kg"]),
                emanation=float(row["g_emanation"]),
                porosity=float(row["porosity_phi"]),
                saturation=float(row["Rs_saturation_at_sampling"]),
                dry_bulk_density=float(row["rho_bulk_kg_m3"]),
                measured_flux_bq_m2_s=float(row["J_Rn_measured_Bq_m2_day"]) / 86400.0,
                emanation_state="undeclared",
            )
        )
    return sites


def _load_zhuo(root: Path) -> list[Site]:
    name = "zhuo2006_10sites.csv"
    assert_usable_for_fitting(name)
    sites = []
    for row in _rows(root / name):
        if row.get("footnote", "").strip():
            continue
        sites.append(
            Site(
                source_file=name,
                site_id=f"zhuo-{row['site']}",
                ra226_bq_kg=float(row["Ra226_Bq_kg"]),
                emanation=float(row["eps0_at25C"]),
                porosity=float(row["total_porosity"]),
                saturation=float(row["S_avg"]),
                dry_bulk_density=float(row["rho_b_g_cm3"]) * 1000.0,
                measured_flux_bq_m2_s=float(row["J_meas_avg"]) / 1000.0,
                emanation_state="dry",
            )
        )
    return sites


def load_sites(root: Path | None = None) -> list[Site]:
    root = data_directory() if root is None else root
    sites = _load_voltaggio(root) + _load_zhuo(root)
    if not sites:
        raise RuntimeError("no site loaded: the dataset cannot be empty")
    return sites


def groups(sites: list[Site]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for index, site in enumerate(sites):
        out.setdefault(site.source_file, []).append(index)
    return out
