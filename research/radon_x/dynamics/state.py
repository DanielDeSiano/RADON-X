from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SPECIE_RN = "Rn222"
SPECIE_PROGENIE = ("Po218", "Pb214", "Bi214")


@dataclass(frozen=True)
class StateLayout:

    zones: tuple[str, ...]
    with_progeny: bool = False

    def __post_init__(self) -> None:
        if not self.zones:
            raise ValueError("servono una o piu' zone")
        if len(set(self.zones)) != len(self.zones):
            raise ValueError(f"nomi di zona duplicati: {self.zones}")

    @property
    def species(self) -> tuple[str, ...]:
        return (SPECIE_RN, *SPECIE_PROGENIE) if self.with_progeny else (SPECIE_RN,)

    @property
    def size(self) -> int:
        return len(self.zones) * len(self.species)

    def index(self, zone: str, species: str = SPECIE_RN) -> int:
        if zone not in self.zones:
            raise KeyError(f"zona {zone!r} non nel layout {self.zones}")
        if species not in self.species:
            raise KeyError(f"specie {species!r} non fra {self.species}")
        return self.zones.index(zone) * len(self.species) + self.species.index(species)

    def unpack(self, x: np.ndarray) -> dict[tuple[str, str], float]:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.size,):
            raise ValueError(f"atteso un vettore di {self.size}, forma {x.shape}")
        return {
            (zona, specie): float(x[self.index(zona, specie)])
            for zona in self.zones
            for specie in self.species
        }


@dataclass(frozen=True)
class InitialState:

    layout: StateLayout
    values: np.ndarray
    origin: str

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=float)
        if v.shape != (self.layout.size,):
            raise ValueError(
                f"stato di {v.shape} contro un layout di {self.layout.size}"
            )
        if not np.all(np.isfinite(v)) or np.any(v < 0.0):
            raise ValueError("lo stato iniziale deve essere finito e non negativo")

    @classmethod
    def zero(cls, layout: StateLayout) -> InitialState:
        return cls(layout=layout, values=np.zeros(layout.size), origin="zero")

    @classmethod
    def from_snapshot(
        cls, layout: StateLayout, snapshot: dict[tuple[str, str] | str, float]
    ) -> InitialState:
        valori = np.zeros(layout.size)
        for chiave, valore in snapshot.items():
            zona, specie = (chiave, SPECIE_RN) if isinstance(chiave, str) else chiave
            if not math.isfinite(valore) or valore < 0.0:
                raise ValueError(f"{chiave!r}: valore non fisico {valore!r}")
            valori[layout.index(zona, specie)] = valore
        return cls(layout=layout, values=valori, origin="snapshot")
