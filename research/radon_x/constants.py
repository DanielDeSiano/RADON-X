from __future__ import annotations

from dataclasses import dataclass
from math import log

__all__ = [
    "CITAZIONI",
    "DECAY_CONSTANTS_PER_SECOND",
    "DOCUMENTO_DI_VERIFICA",
    "HALF_LIVES_SECONDS",
    "STATI_DI_VERIFICA",
    "Citazione",
]

HALF_LIVES_SECONDS = {
    "Rn222": 3.8232 * 86400.0,
    "Po218": 185.88,
    "Pb214": 1608.0,
    "Bi214": 1182.0,
    "Po214": 1.643e-4,
    "Rn220": 55.8,
    "Po216": 0.148,
    "Pb212": 10.64 * 3600.0,
}

DECAY_CONSTANTS_PER_SECOND = {
    isotope: log(2.0) / half_life
    for isotope, half_life in HALF_LIVES_SECONDS.items()
}

DOCUMENTO_DI_VERIFICA = "docs/decay-constants-verification.md"

STATI_DI_VERIFICA = ("needs-source", "verified")


@dataclass(frozen=True)
class Citazione:

    valutazione: str
    tabella: str
    valore_dichiarato: str
    incertezza_s: float | None
    stato: str
    nota: str = ""


_DDEP = "DDEP (Decay Data Evaluation Project)"
_TABELLA_DDEP = "Be et al., Table de Radionucleides; monolite riga 32"
_NON_DICHIARATA = "non dichiarata"

CITAZIONI = {
    "Rn222": Citazione(
        valutazione=_DDEP,
        tabella=_TABELLA_DDEP + ", riga 34",
        valore_dichiarato="3.8232 d",
        incertezza_s=None,
        stato="needs-source",
        nota=(
            "governa il trasporto: lambda entra in L_d, nel Damkoehler "
            "diffusivo e nel flusso di esalazione. Il monolite porta alla riga "
            "5930 un secondo valore scritto a mano, _LAM_RN = 2.098e-6 s^-1, "
            "che scarta dello 0.018 per cento da quello derivato qui"
        ),
    ),
    "Po218": Citazione(
        valutazione=_DDEP,
        tabella=_TABELLA_DDEP + ", riga 35",
        valore_dichiarato="3.098 min",
        incertezza_s=None,
        stato="needs-source",
        nota=(
            "l'audit locale dichiara un valore precedente di 183 s, cioe' una "
            "revisione dell'1.574 per cento: due valutazioni diverse sono "
            "circolate nel progetto e la tabella non dice quale sia questa"
        ),
    ),
    "Pb214": Citazione(
        valutazione=_DDEP,
        tabella=_TABELLA_DDEP + ", riga 36",
        valore_dichiarato="26.8 min",
        incertezza_s=None,
        stato="needs-source",
        nota="collo di bottiglia della catena a vita breve",
    ),
    "Bi214": Citazione(
        valutazione=_DDEP,
        tabella=_TABELLA_DDEP + ", riga 37",
        valore_dichiarato="19.7 min",
        incertezza_s=None,
        stato="needs-source",
        nota="",
    ),
    "Po214": Citazione(
        valutazione=_NON_DICHIARATA,
        tabella="monolite riga 38; intestazione collettiva alla riga 32",
        valore_dichiarato="164.3 us",
        incertezza_s=None,
        stato="needs-source",
        nota=(
            "il commento del monolite non attribuisce la riga a DDEP; "
            "l'attribuzione resterebbe quella collettiva della riga 32"
        ),
    ),
    "Rn220": Citazione(
        valutazione=_DDEP,
        tabella=_TABELLA_DDEP + ", riga 39",
        valore_dichiarato="55.8 s",
        incertezza_s=None,
        stato="needs-source",
        nota="toron; il ramo Rn-220 e' in quarantena per la sezione 9 del mandato",
    ),
    "Po216": Citazione(
        valutazione=_NON_DICHIARATA,
        tabella="monolite riga 40; intestazione collettiva alla riga 32",
        valore_dichiarato="148 ms",
        incertezza_s=None,
        stato="needs-source",
        nota="il commento del monolite non attribuisce la riga a DDEP",
    ),
    "Pb212": Citazione(
        valutazione=_NON_DICHIARATA,
        tabella="monolite riga 41; nessuna tabella citata",
        valore_dichiarato="10.64 h",
        incertezza_s=None,
        stato="needs-source",
        nota=(
            "unico isotopo assente dall'audit locale: non esiste un valore di "
            "riferimento contro cui confrontare il codice"
        ),
    ),
}
