from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CAMPI_DI_COLONNA",
    "CAMPI_INDOOR",
    "SCALARI_INDOOR",
    "LegacyStateLayout",
]

CAMPI_DI_COLONNA: tuple[str, ...] = (
    "theta",
    "Pg",
    "Cm",
    "Cf",
    "M",
    "Cs",
    "Ar",
    "Cm220",
    "Cf220",
    "eps_mem",
)

CAMPI_INDOOR: tuple[str, ...] = (
    "C_bs",
    "C_gt",
    "P_in",
    "C_Po218",
    "C_Pb214",
    "C_Bi214",
    "C_Po214",
)

SCALARI_INDOOR: int = len(CAMPI_INDOOR)


@dataclass(frozen=True)
class LegacyStateLayout:

    nz: int

    def __post_init__(self) -> None:
        if isinstance(self.nz, bool) or not isinstance(self.nz, int):
            raise TypeError(f"nz deve essere un intero, ricevuto {self.nz!r}")
        if self.nz < 1:
            raise ValueError(f"nz deve essere >= 1, ricevuto {self.nz!r}")

    @property
    def size(self) -> int:
        return len(CAMPI_DI_COLONNA) * self.nz + SCALARI_INDOOR

    @property
    def names(self) -> tuple[str, ...]:
        return CAMPI_DI_COLONNA + CAMPI_INDOOR

    def slice_of(self, nome: str) -> slice:
        if nome in CAMPI_DI_COLONNA:
            posizione = CAMPI_DI_COLONNA.index(nome)
            return slice(posizione * self.nz, (posizione + 1) * self.nz)
        indice = self.index_of(nome)
        return slice(indice, indice + 1)

    def index_of(self, nome: str) -> int:
        if nome in CAMPI_DI_COLONNA:
            return CAMPI_DI_COLONNA.index(nome) * self.nz
        if nome in CAMPI_INDOOR:
            return len(CAMPI_DI_COLONNA) * self.nz + CAMPI_INDOOR.index(nome)
        raise ValueError(
            f"campo di stato non dichiarato: {nome!r}; i campi di colonna sono "
            f"{', '.join(CAMPI_DI_COLONNA)}, quelli d'ambiente sono "
            f"{', '.join(CAMPI_INDOOR)}"
        )

    @classmethod
    def from_model(cls, model: object) -> LegacyStateLayout:
        nz = getattr(model, "nz", None)
        if not isinstance(nz, int) or isinstance(nz, bool):
            raise TypeError(
                f"il modello legacy non espone un nz intero, ricevuto {nz!r}"
            )
        tracciato = cls(nz)
        dichiarata = getattr(model, "nstate", None)
        if dichiarata is not None and int(dichiarata) != tracciato.size:
            raise ValueError(
                f"il modello legacy dichiara nstate={int(dichiarata)}, il tracciato "
                f"{tracciato.size} = 10 * {nz} + {SCALARI_INDOOR}: l'ordine dei campi "
                "non e' piu' quello dichiarato in radon_x.legacy_layout"
            )
        return tracciato
