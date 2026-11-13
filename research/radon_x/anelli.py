from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass


class StatoAnello(enum.Enum):

    VALIDATO = "validato"
    PARZIALE = "parzialmente validato"
    NON_VALIDATO = "non validato"


class FamigliaMetrica(enum.Enum):

    DECADI = "decadi di log10"
    RAPPORTO = "rapporto predetto/misurato"
    Z_RIR = "z su punti percentuali di RIR"
    NESSUNA = "nessuna"


@dataclass(frozen=True)
class VoceIncertezza:

    nome: str
    basso: float
    alto: float
    origine: str
    misurata: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.basso) or not math.isfinite(self.alto):
            raise ValueError(f"voce {self.nome!r}: estremi non finiti")
        if self.basso > 0.0:
            raise ValueError(f"voce {self.nome!r}: basso deve essere <= 0, e' {self.basso!r}")
        if self.alto < 0.0:
            raise ValueError(f"voce {self.nome!r}: alto deve essere >= 0, e' {self.alto!r}")


VOCI_ANELLO_B: tuple[VoceIncertezza, ...] = (
    VoceIncertezza(
        nome="altezza equivalente h = V/A_f, da 1.0 a 3.0 m",
        basso=-0.4771212547196624,
        alto=0.0,
        origine="nessuna misura di edificio sui tredici siti",
    ),
    VoceIncertezza(
        nome="ricambio d'aria, da 0.05 a 3.0 1/h rispetto a 0.10",
        basso=-1.4465861748,
        alto=0.2715504306,
        origine="ACH_closed del modello originale, non misurato",
    ),
    VoceIncertezza(
        nome="conduttanza della soletta, da assente a fessura di 0.05 m2",
        basso=-1.3209642265,
        alto=0.0,
        origine="fattore 20.94 fra soletta senza resistenza e fessura vera",
    ),
    VoceIncertezza(
        nome="aria di reintegro sotto depressione",
        basso=0.0,
        alto=0.8538719643217618,
        origine="Gallese 2022, esperimenti 6 e 1 a |ACH| = 0.49 1/h",
        misurata=True,
    ),
)


def banda_da_voci(
    voci: Sequence[VoceIncertezza] = VOCI_ANELLO_B,
) -> tuple[float, float]:
    return (
        float(sum(v.basso for v in voci)),
        float(sum(v.alto for v in voci)),
    )


@dataclass(frozen=True)
class Anello:

    codice: str
    ingresso: str
    uscita: str
    unita_uscita: str
    stato: StatoAnello
    file_di_validazione: tuple[str, ...]
    n_punti: int
    famiglia_metrica: FamigliaMetrica
    banda_sistematica_dex: tuple[float, float] = (0.0, 0.0)
    nota: str = ""

    def __post_init__(self) -> None:
        if self.n_punti < 0:
            raise ValueError(f"anello {self.codice}: n_punti negativo")
        if self.stato is StatoAnello.NON_VALIDATO and self.n_punti != 0:
            raise ValueError(
                f"anello {self.codice}: dichiarato non validato ma con {self.n_punti} punti"
            )
        if self.stato is not StatoAnello.NON_VALIDATO and self.n_punti == 0:
            raise ValueError(
                f"anello {self.codice}: dichiarato {self.stato.value} senza nessun punto"
            )
        basso, alto = self.banda_sistematica_dex
        if basso > 0.0 or alto < 0.0:
            raise ValueError(f"anello {self.codice}: banda con segni invertiti")


ANELLI: dict[str, Anello] = {
    "A1": Anello(
        codice="A1",
        ingresso="predittori del suolo",
        uscita="J",
        unita_uscita="Bq m-2 s-1",
        stato=StatoAnello.VALIDATO,
        file_di_validazione=(
            "voltaggio2006_lazio_6siti.csv",
            "zhuo2006_10sites.csv",
        ),
        n_punti=13,
        famiglia_metrica=FamigliaMetrica.DECADI,
        nota="due campagne indipendenti, cinque predittori misurati sito per sito",
    ),
    "B": Anello(
        codice="B",
        ingresso="J",
        uscita="sorgente dell'ambiente",
        unita_uscita="Bq s-1",
        stato=StatoAnello.NON_VALIDATO,
        file_di_validazione=(),
        n_punti=0,
        famiglia_metrica=FamigliaMetrica.NESSUNA,
        banda_sistematica_dex=banda_da_voci(),
        nota="nessuno dei parametri del trasferimento e' misurato su un sito del progetto",
    ),
    "C1": Anello(
        codice="C1",
        ingresso="sorgente dell'ambiente",
        uscita="C_indoor",
        unita_uscita="Bq m-3",
        stato=StatoAnello.PARZIALE,
        file_di_validazione=("gallese2020_equilibri.csv",),
        n_punti=1,
        famiglia_metrica=FamigliaMetrica.RAPPORTO,
        nota="una configurazione su sei pienamente confrontabile, a ventilazione nulla",
    ),
    "C2": Anello(
        codice="C2",
        ingresso="sorgente dell'ambiente",
        uscita="riduzione relativa",
        unita_uscita="punti percentuali",
        stato=StatoAnello.PARZIALE,
        file_di_validazione=("gallese2022_ach.csv",),
        n_punti=4,
        famiglia_metrica=FamigliaMetrica.Z_RIR,
        nota="quattro righe su dodici mai toccate da una stima; solo rapporti, nessun assoluto",
    ),
}

CATENA_INDOOR: tuple[str, ...] = ("A1", "B", "C1")


def catena_contigua(anelli: Sequence[str]) -> bool:
    if not anelli:
        return False
    if any(codice not in ANELLI for codice in anelli):
        return False
    coppie = zip(anelli, anelli[1:])
    return all(ANELLI[a].uscita == ANELLI[b].ingresso for a, b in coppie)


@dataclass(frozen=True)
class Misurato:

    valore: float
    unita: str
    file: str
    riga: str
    anelli_chiusi: frozenset[str]
    ruolo: str
    incertezza: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.valore):
            raise ValueError(f"{self.file}:{self.riga}: valore non finito")
        sconosciuti = set(self.anelli_chiusi) - set(ANELLI)
        if sconosciuti:
            raise ValueError(f"anelli non nel registro: {sorted(sconosciuti)}")
        if self.incertezza is not None and (
            not math.isfinite(self.incertezza) or self.incertezza < 0.0
        ):
            raise ValueError(f"{self.file}:{self.riga}: incertezza non valida")


@dataclass(frozen=True)
class Banda:

    statistica_dex: float
    sistematica_dex: tuple[float, float]
    forma_non_quantificata: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.statistica_dex < 0.0 or not math.isfinite(self.statistica_dex):
            raise ValueError("la dispersione statistica deve essere >= 0 e finita")

    def totale_dex(self) -> tuple[float, float]:
        basso, alto = self.sistematica_dex
        return (basso - self.statistica_dex, alto + self.statistica_dex)

    def span_dex(self) -> float:
        basso, alto = self.sistematica_dex
        return float(alto - basso)

    def dominante(self) -> str:
        if self.statistica_dex == 0.0:
            return "sistematica"
        return "sistematica" if self.span_dex() > 2.0 * self.statistica_dex else "statistica"


@dataclass(frozen=True)
class Predetto:

    valore: float
    unita: str
    anelli: tuple[str, ...]
    banda: Banda | None = None
    etichetta: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.valore):
            raise ValueError("valore predetto non finito")
        sconosciuti = [c for c in self.anelli if c not in ANELLI]
        if sconosciuti:
            raise ValueError(f"anelli non nel registro: {sconosciuti}")
        if not catena_contigua(self.anelli):
            raise ValueError(f"catena non contigua: {self.anelli}")
        atteso = ANELLI[self.anelli[-1]].unita_uscita
        if self.unita != atteso:
            raise ValueError(
                f"unita' {self.unita!r} non e' quella dell'anello finale {atteso!r}"
            )

    @property
    def anelli_non_validati(self) -> tuple[str, ...]:
        return tuple(
            c for c in self.anelli if ANELLI[c].stato is StatoAnello.NON_VALIDATO
        )

    @property
    def banda_sistematica_dex(self) -> tuple[float, float]:
        basso = sum(ANELLI[c].banda_sistematica_dex[0] for c in self.anelli)
        alto = sum(ANELLI[c].banda_sistematica_dex[1] for c in self.anelli)
        return (float(basso), float(alto))

    def intervallo(self) -> tuple[float, float]:
        if self.banda is None:
            basso, alto = self.banda_sistematica_dex
        else:
            basso, alto = self.banda.totale_dex()
        return (self.valore * 10.0**basso, self.valore * 10.0**alto)

    def __format__(self, spec: str) -> str:
        numero = format(self.valore, spec or ".4g")
        non_validati = self.anelli_non_validati
        marchio = f"predetto {'>'.join(self.anelli)}"
        if non_validati:
            basso, alto = self.banda_sistematica_dex
            marchio += (
                f", non validato: {','.join(non_validati)}"
                f", banda {basso:+.3f}/{alto:+.3f} dex"
            )
        return f"{numero} {self.unita} <{marchio}>"

    def __str__(self) -> str:
        return format(self)


@dataclass(frozen=True)
class Metriche:

    n: int
    mbe: float
    sd: float
    rmse: float
    f2: float
    max_abs: float


@dataclass(frozen=True)
class MetricheRapporto:

    n: int
    rapporto_mediano: float
    rapporto_min: float
    rapporto_max: float
    z_max: float
    scenario: str


@dataclass(frozen=True)
class MetricheZ:

    n: int
    residuo_medio_pp: float
    residuo_max_pp: float
    z_mediano: float
    z_max: float
    entro_1sigma: float
    entro_2sigma: float


class ConfrontoNonAmmesso(Exception):
    pass


def _mediana(valori: Sequence[float]) -> float:
    ordinati = sorted(valori)
    n = len(ordinati)
    meta = n // 2
    return ordinati[meta] if n % 2 else 0.5 * (ordinati[meta - 1] + ordinati[meta])


def confronta(
    previsioni: Sequence[Predetto],
    osservazioni: Sequence[Misurato],
) -> Metriche | MetricheRapporto | MetricheZ:
    if not previsioni or not osservazioni:
        raise ConfrontoNonAmmesso("confronto vuoto")
    if len(previsioni) != len(osservazioni):
        raise ConfrontoNonAmmesso(
            f"{len(previsioni)} predizioni contro {len(osservazioni)} misure"
        )

    attraversati = {frozenset(p.anelli) for p in previsioni}
    if len(attraversati) != 1:
        raise ConfrontoNonAmmesso("le predizioni non attraversano tutte gli stessi anelli")
    chiusi = {o.anelli_chiusi for o in osservazioni}
    if len(chiusi) != 1:
        raise ConfrontoNonAmmesso("le misure non chiudono tutte gli stessi anelli")

    (percorsi,) = attraversati
    (coperti,) = chiusi
    if percorsi != coperti:
        raise ConfrontoNonAmmesso(
            f"la predizione attraversa {sorted(percorsi)} ma la misura chiude "
            f"{sorted(coperti)}: nessun dato del progetto chiude questa catena"
        )

    unita = {p.unita for p in previsioni} | {o.unita for o in osservazioni}
    if len(unita) != 1:
        raise ConfrontoNonAmmesso(f"unita' diverse nello stesso confronto: {sorted(unita)}")

    famiglia = ANELLI[previsioni[0].anelli[-1]].famiglia_metrica
    if famiglia is FamigliaMetrica.NESSUNA:
        raise ConfrontoNonAmmesso("l'anello finale non ha una famiglia di metrica")

    if famiglia is FamigliaMetrica.DECADI:
        residui = [
            math.log10(p.valore) - math.log10(o.valore)
            for p, o in zip(previsioni, osservazioni)
            if p.valore > 0.0 and o.valore > 0.0
        ]
        n = len(residui)
        if n == 0:
            raise ConfrontoNonAmmesso("nessuna coppia positiva: il log10 non e' definito")
        mbe = sum(residui) / n
        sd = (sum((r - mbe) ** 2 for r in residui) / (n - 1)) ** 0.5 if n > 1 else 0.0
        return Metriche(
            n=n,
            mbe=mbe,
            sd=sd,
            rmse=(sum(r * r for r in residui) / n) ** 0.5,
            f2=sum(1 for r in residui if abs(r) <= math.log10(2.0)) / n,
            max_abs=max(abs(r) for r in residui),
        )

    if famiglia is FamigliaMetrica.RAPPORTO:
        rapporti = [p.valore / o.valore for p, o in zip(previsioni, osservazioni)]
        z = [
            abs(p.valore - o.valore) / o.incertezza
            for p, o in zip(previsioni, osservazioni)
            if o.incertezza
        ]
        return MetricheRapporto(
            n=len(rapporti),
            rapporto_mediano=_mediana(rapporti),
            rapporto_min=min(rapporti),
            rapporto_max=max(rapporti),
            z_max=max(z) if z else float("nan"),
            scenario=previsioni[0].etichetta or "non dichiarato",
        )

    residui = [p.valore - o.valore for p, o in zip(previsioni, osservazioni)]
    z = [
        abs(r) / o.incertezza
        for r, o in zip(residui, osservazioni)
        if o.incertezza
    ]
    n = len(residui)
    return MetricheZ(
        n=n,
        residuo_medio_pp=sum(residui) / n,
        residuo_max_pp=max(residui, key=abs),
        z_mediano=_mediana(z) if z else float("nan"),
        z_max=max(z) if z else float("nan"),
        entro_1sigma=sum(1 for v in z if v <= 1.0) / len(z) if z else float("nan"),
        entro_2sigma=sum(1 for v in z if v <= 2.0) / len(z) if z else float("nan"),
    )
