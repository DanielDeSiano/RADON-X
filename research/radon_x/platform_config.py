from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from radon_x.seeds import SEME_RADICE, USI, derive, semi

__all__ = [
    "CHIAVE_PROVENIENZA",
    "LIVELLI",
    "LIVELLO_PREDEFINITO",
    "PlatformConfig",
    "PlatformConfigError",
    "provenance_of",
]

CHIAVE_PROVENIENZA: str = "platform"

LIVELLI: dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

LIVELLO_PREDEFINITO: str = "WARNING"

_SEME_MASSIMO: int = 2**32


class PlatformConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PlatformConfig:

    seed: int = SEME_RADICE
    output_dir: Path | None = None
    log_level: str = LIVELLO_PREDEFINITO

    def validate(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise PlatformConfigError(
                f"seed deve essere un intero, ricevuto {self.seed!r}"
            )
        if not 0 <= self.seed < _SEME_MASSIMO:
            raise PlatformConfigError(
                f"seed deve stare in [0, {_SEME_MASSIMO}), ricevuto {self.seed!r}"
            )
        if self.log_level not in LIVELLI:
            raise PlatformConfigError(
                f"log_level {self.log_level!r} non riconosciuto; "
                f"i livelli sono {', '.join(LIVELLI)}"
            )
        if self.output_dir is not None:
            if not isinstance(self.output_dir, Path):
                raise PlatformConfigError(
                    f"output_dir deve essere un Path o None, ricevuto {self.output_dir!r}"
                )
            if self.output_dir.exists() and not self.output_dir.is_dir():
                raise PlatformConfigError(
                    f"output_dir {self.output_dir} esiste e non e' una cartella"
                )

    def seed_for(self, uso: str) -> int:
        return derive(self.seed, uso)

    def semi(self) -> dict[str, int]:
        return semi(self.seed)

    def usi(self) -> tuple[str, ...]:
        return USI

    def mostra(self, livello: str) -> bool:
        if livello not in LIVELLI:
            raise PlatformConfigError(
                f"livello {livello!r} non riconosciuto; i livelli sono {', '.join(LIVELLI)}"
            )
        return LIVELLI[livello] >= LIVELLI[self.log_level]

    def prepara_uscita(self) -> Path | None:
        self.validate()
        if self.output_dir is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def con(self, **cambi: Any) -> PlatformConfig:
        ignoti = sorted(set(cambi) - _CAMPI)
        if ignoti:
            raise PlatformConfigError(
                f"campi non riconosciuti: {', '.join(ignoti)}; "
                f"i campi sono {', '.join(sorted(_CAMPI))}"
            )
        copia = replace(self, **cambi)
        copia.validate()
        return copia

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "output_dir": None if self.output_dir is None else str(self.output_dir),
            "log_level": self.log_level,
        }

    def echo(self) -> dict[str, Any]:
        corpo = self.to_dict()
        corpo["seeds"] = self.semi()
        return corpo

    def provenance(self) -> Mapping[str, Any]:
        corpo = self.echo()
        corpo["seeds"] = MappingProxyType(corpo["seeds"])
        return MappingProxyType(corpo)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlatformConfig:
        if not isinstance(data, dict):
            raise PlatformConfigError(
                f"la configurazione di piattaforma deve essere un dizionario, "
                f"ricevuto {type(data).__name__}"
            )
        ignote = sorted(set(data) - _CAMPI)
        if ignote:
            raise PlatformConfigError(
                f"chiavi non riconosciute: {', '.join(ignote)}; "
                f"le chiavi sono {', '.join(sorted(_CAMPI))}"
            )
        cartella = data.get("output_dir")
        istanza = cls(
            seed=data.get("seed", SEME_RADICE),
            output_dir=None if cartella is None else Path(cartella),
            log_level=str(data.get("log_level", LIVELLO_PREDEFINITO)),
        )
        istanza.validate()
        return istanza

    @classmethod
    def risolta(cls, platform: PlatformConfig | None) -> PlatformConfig:
        if platform is None:
            return cls()
        if not isinstance(platform, cls):
            raise PlatformConfigError(
                f"platform deve essere un PlatformConfig o None, ricevuto {platform!r}"
            )
        platform.validate()
        return platform


_CAMPI: frozenset = frozenset(campo.name for campo in fields(PlatformConfig))


def provenance_of(result: Any) -> Mapping[str, Any] | None:
    corpo = result if isinstance(result, dict) else getattr(result, "input_config", None)
    if not isinstance(corpo, dict):
        return None
    voce = corpo.get(CHIAVE_PROVENIENZA)
    if voce is None:
        return None
    if not isinstance(voce, dict):
        raise PlatformConfigError(
            f"la voce {CHIAVE_PROVENIENZA!r} del risultato deve essere un dizionario, "
            f"ricevuto {type(voce).__name__}"
        )
    letta = dict(voce)
    semi_letti = letta.get("seeds")
    if isinstance(semi_letti, dict):
        letta["seeds"] = MappingProxyType(dict(semi_letti))
    return MappingProxyType(letta)
