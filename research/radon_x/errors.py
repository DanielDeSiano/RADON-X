from __future__ import annotations

import enum
from dataclasses import dataclass

__all__ = [
    "REGISTRO_AVVISI",
    "Avviso",
    "CodiceAvviso",
    "ConfigError",
    "DataError",
    "LeakageError",
    "NumericalError",
    "RadonXError",
    "VoceDiRegistro",
    "codici_noti",
    "rendi_avviso",
]


class RadonXError(Exception):
    pass


class ConfigError(RadonXError, ValueError):
    pass


class NumericalError(RadonXError, ValueError):
    pass


class DataError(RadonXError, ValueError):
    pass


class LeakageError(RadonXError, ValueError):
    pass


class CodiceAvviso(enum.Enum):

    ZREF_FUORI_COLONNA = "zref-fuori-colonna"
    ZREF_FUORI_GRIGLIA_LEGACY = "zref-fuori-griglia-legacy"
    LITOTIPO_NON_LETTO = "litotipo-non-letto"
    ACCESSIBILITA_PORI_NON_APPLICATA = "accessibilita-pori-non-applicata"
    FOURIER_DI_CELLA_SOPRA_SOGLIA = "fourier-di-cella-sopra-soglia"
    RESIDUO_DI_COMPOSIZIONE = "residuo-di-composizione"
    GRIGLIA_DI_USCITA_NON_UNIFORME = "griglia-di-uscita-non-uniforme"
    TRAIETTORIA_NEGATIVA = "traiettoria-negativa"


@dataclass(frozen=True)
class VoceDiRegistro:

    campo: str
    motivo: str
    modulo: str
    modello: str


REGISTRO_AVVISI: dict[CodiceAvviso, VoceDiRegistro] = {
    CodiceAvviso.ZREF_FUORI_COLONNA: VoceDiRegistro(
        campo="reference_depth_m",
        motivo=(
            "la profondita' di riferimento cade fuori dai centri cella della "
            "colonna: il valore riportato e' NaN, mai un'estrapolazione"
        ),
        modulo="unione_delle_parti.py",
        modello=(
            "profondita' di riferimento {valore} m fuori dalla colonna: i "
            "centri cella vanno da {centro_primo_m} m a {centro_ultimo_m} m, "
            "soil_at_reference_depth_bq_m3 = NaN"
        ),
    ),
    CodiceAvviso.ZREF_FUORI_GRIGLIA_LEGACY: VoceDiRegistro(
        campo="reference_depth_m",
        motivo=(
            "il motore legacy cerca la profondita' di riferimento con un "
            "argmin sulla propria griglia e non controlla di essere dentro la "
            "colonna: su un letto piu' corto rende il nodo piu' profondo. Il "
            "numero non viene corretto, perche' correggerlo sarebbe cambiare "
            "il modello legacy"
        ),
        modulo="legacy_adapter.py",
        modello=(
            "profondita' di riferimento {valore} m fuori dalla colonna: i nodi "
            "della griglia legacy vanno da {nodo_primo_m} m a {nodo_ultimo_m} "
            "m, soil_at_reference_depth_bq_m3 e' il valore a {nodo_letto_m} m, "
            "riportato con reference_depth_m = {valore} m"
        ),
    ),
    CodiceAvviso.LITOTIPO_NON_LETTO: VoceDiRegistro(
        campo="lithotype",
        motivo=(
            "il motore core non ha tabelle per litotipo: prende i coefficienti "
            "dai campi espliciti della configurazione e il litotipo dichiarato "
            "non entra nel conto"
        ),
        modulo="unione_delle_parti.py",
        modello=(
            "litotipo dichiarato {valore!r}: il motore core non legge il campo, "
            "i coefficienti della colonna vengono dai campi espliciti della "
            "configurazione"
        ),
    ),
    CodiceAvviso.ACCESSIBILITA_PORI_NON_APPLICATA: VoceDiRegistro(
        campo="initial_pore_accessibility",
        motivo=(
            "il motore core non rappresenta l'accessibilita' iniziale dei "
            "pori: la frazione di emanazione entra intera"
        ),
        modulo="unione_delle_parti.py",
        modello=(
            "initial_pore_accessibility {valore}: il motore core non applica "
            "l'accessibilita' dei pori, la frazione di emanazione entra intera "
            "al valore {emanazione}"
        ),
    ),
    CodiceAvviso.FOURIER_DI_CELLA_SOPRA_SOGLIA: VoceDiRegistro(
        campo="max_cell_fourier_number",
        motivo=(
            "il passo d'uscita supera il tempo diffusivo di cella e la "
            "condizione iniziale non e' lo stazionario: il transitorio "
            "campionato su quella griglia non e' risolto"
        ),
        modulo="unione_delle_parti.py",
        modello=(
            "numero di Fourier di cella {valore} sopra {soglia}: il passo "
            "d'uscita {passo_s} s supera il tempo diffusivo di cella "
            "{tempo_diffusivo_s} s, condizione iniziale {condizione_iniziale!r}"
        ),
    ),
    CodiceAvviso.RESIDUO_DI_COMPOSIZIONE: VoceDiRegistro(
        campo="composition_residual",
        motivo=(
            "l'esponenziale di matrice su un colpo solo e quello a tratti non "
            "coincidono entro la soglia: la propagazione non e' composta"
        ),
        modulo="dynamics/solver.py",
        modello="residuo di composizione {valore:.3e} sopra {soglia:.0e}",
    ),
    CodiceAvviso.GRIGLIA_DI_USCITA_NON_UNIFORME: VoceDiRegistro(
        campo="step_min_s",
        motivo=(
            "un evento cade fra due frame d'uscita e spezza il passo: la "
            "griglia consegnata non ha il passo richiesto"
        ),
        modulo="dynamics/solver.py",
        modello=(
            "griglia di uscita non uniforme: passo minimo {valore:.6g} s contro "
            "il passo richiesto {passo_richiesto_s:.6g} s"
        ),
    ),
    CodiceAvviso.TRAIETTORIA_NEGATIVA: VoceDiRegistro(
        campo="values",
        motivo=(
            "la traiettoria scende sotto zero: il solutore non tronca il "
            "proprio risultato e il valore esce come e' stato calcolato"
        ),
        modulo="dynamics/solver.py",
        modello="minimo della traiettoria negativo: {valore:.3e} Bq m-3",
    ),
}


@dataclass(frozen=True)
class Avviso:

    codice: CodiceAvviso
    campo: str
    valore_usato: float | str
    motivo: str
    dati: tuple[tuple[str, float | str], ...] = ()

    @classmethod
    def per(cls, codice: CodiceAvviso, valore_usato: float | str, **dati: float | str) -> Avviso:
        try:
            voce = REGISTRO_AVVISI[codice]
        except KeyError as exc:
            raise ConfigError(f"codice di avviso non registrato: {codice!r}") from exc
        return cls(
            codice=codice,
            campo=voce.campo,
            valore_usato=valore_usato,
            motivo=voce.motivo,
            dati=tuple(sorted(dati.items())),
        )


def rendi_avviso(avviso: Avviso) -> str:
    try:
        voce = REGISTRO_AVVISI[avviso.codice]
    except KeyError as exc:
        raise ConfigError(f"codice di avviso non registrato: {avviso.codice!r}") from exc
    try:
        return voce.modello.format(valore=avviso.valore_usato, **dict(avviso.dati))
    except KeyError as exc:
        raise ConfigError(
            f"avviso {avviso.codice.value}: manca il dato {exc.args[0]!r} richiesto dal modello"
        ) from exc


def codici_noti() -> tuple[CodiceAvviso, ...]:
    return tuple(REGISTRO_AVVISI)
