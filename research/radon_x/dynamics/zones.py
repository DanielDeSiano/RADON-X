from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from radon_x.core.units import SECONDI_PER_ORA

ESTERNO = "outdoor"

TOLLERANZA_BILANCIO_ARIA = 1e-9


@dataclass(frozen=True)
class ZoneNetwork:

    volumes_m3: dict[str, float]
    flows_m3_s: dict[tuple[str, str], float]

    def __post_init__(self) -> None:
        if not self.volumes_m3:
            raise ValueError("servono una o piu' zone")
        for zona, volume in self.volumes_m3.items():
            if not math.isfinite(volume) or volume <= 0.0:
                raise ValueError(f"volume di {zona!r} non fisico: {volume!r}")
        nodi = set(self.volumes_m3) | {ESTERNO}
        for (da, a), portata in self.flows_m3_s.items():
            if da not in nodi or a not in nodi:
                raise ValueError(f"flusso {da!r}->{a!r} su un nodo ignoto")
            if da == a:
                raise ValueError(f"flusso di {da!r} su se stessa")
            if not math.isfinite(portata) or portata < 0.0:
                raise ValueError(f"portata {da!r}->{a!r} non fisica: {portata!r}")
        for zona in self.volumes_m3:
            entrata = sum(q for (da, a), q in self.flows_m3_s.items() if a == zona)
            uscita = sum(q for (da, a), q in self.flows_m3_s.items() if da == zona)
            scala = max(entrata, uscita, 1e-30)
            if abs(entrata - uscita) / scala > TOLLERANZA_BILANCIO_ARIA:
                raise ValueError(
                    f"il bilancio d'aria di {zona!r} non chiude: entra "
                    f"{entrata:.6g} m3/s, esce {uscita:.6g}"
                )

    @classmethod
    def single_zone(cls, volume_m3: float, ach_h: float) -> ZoneNetwork:
        if not math.isfinite(ach_h) or ach_h < 0.0:
            raise ValueError(f"ricambio d'aria non fisico: {ach_h!r}")
        portata = ach_h * volume_m3 / SECONDI_PER_ORA
        flussi = (
            {(ESTERNO, "zona"): portata, ("zona", ESTERNO): portata}
            if portata > 0.0
            else {}
        )
        return cls(volumes_m3={"zona": volume_m3}, flows_m3_s=flussi)

    @property
    def zones(self) -> tuple[str, ...]:
        return tuple(self.volumes_m3)

    def air_changes_h(self, zone: str) -> float:
        uscita = sum(q for (da, _a), q in self.flows_m3_s.items() if da == zone)
        return uscita * SECONDI_PER_ORA / self.volumes_m3[zone]


def exchange_matrix_from_flows(
    network: ZoneNetwork,
) -> tuple[np.ndarray, np.ndarray]:
    zone = network.zones
    n = len(zone)
    a = np.zeros((n, n))
    b = np.zeros(n)
    for (da, verso), portata in network.flows_m3_s.items():
        if verso in zone:
            i = zone.index(verso)
            volume = network.volumes_m3[verso]
            if da == ESTERNO:
                b[i] += portata / volume
            else:
                a[i, zone.index(da)] += portata / volume
        if da in zone:
            i = zone.index(da)
            a[i, i] -= portata / network.volumes_m3[da]
    return a, b
