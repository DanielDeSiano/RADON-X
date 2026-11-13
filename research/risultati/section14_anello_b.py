from __future__ import annotations

import csv
import math
from dataclasses import dataclass

from section14_dataset import Site, load_sites

from radon_x.anelli import ANELLI, Banda, VoceIncertezza
from radon_x.core.decay import LAMBDA_RN222
from radon_x.core.indoor import (
    crack_conductance_m3_s,
    indoor_coupled_bq_m3,
    soil_transfer_velocity,
)
from radon_x.data_manifest import data_directory

VOLUME_M3 = 50.0
IMPRONTA_M2 = 50.0
ACH_H = 0.10
DEPOSIZIONE_H = 0.0

ALTEZZA_ALTA_M = 3.0
ACH_BASSO_H = 0.05
ACH_ALTO_H = 3.0
AREA_FESSURA_M2 = 0.05


def lambda_efficace(ach_h: float, deposizione_h: float = DEPOSIZIONE_H) -> float:
    return LAMBDA_RN222 + (ach_h + deposizione_h) / 3600.0


def altezza_equivalente_m(volume_m3: float = VOLUME_M3, impronta_m2: float = IMPRONTA_M2) -> float:
    if impronta_m2 <= 0.0:
        raise ValueError(f"impronta deve essere > 0, ricevuto {impronta_m2!r}")
    return volume_m3 / impronta_m2


def fattore_di_trasferimento_s_m(
    porosita: float,
    saturazione: float,
    impronta_m2: float = IMPRONTA_M2,
    volume_m3: float = VOLUME_M3,
    ach_h: float = ACH_H,
    conduttanza_m3_s: float = math.inf,
    deposizione_h: float = DEPOSIZIONE_H,
) -> float:
    v = soil_transfer_velocity(porosita, saturazione)
    beta = v * impronta_m2
    lam = lambda_efficace(ach_h, deposizione_h)
    if math.isinf(conduttanza_m3_s):
        return impronta_m2 / (beta + volume_m3 * lam)
    return impronta_m2 / (beta + volume_m3 * lam * (beta / conduttanza_m3_s + 1.0))


def fattore_impronta_s_m(
    volume_m3: float = VOLUME_M3,
    impronta_m2: float = IMPRONTA_M2,
    ach_h: float = ACH_H,
    deposizione_h: float = DEPOSIZIONE_H,
) -> float:
    return impronta_m2 / (volume_m3 * lambda_efficace(ach_h, deposizione_h))


def sito_di_riferimento(siti: list[Site] | None = None) -> Site:
    return max(siti if siti is not None else load_sites(), key=lambda s: s.physical_flux_bq_m2_s())


def _concentrazione(
    sito: Site,
    ach_h: float = ACH_H,
    volume_m3: float = VOLUME_M3,
    impronta_m2: float = IMPRONTA_M2,
    conduttanza_m3_s: float = math.inf,
) -> float:
    return indoor_coupled_bq_m3(
        sito.physical_flux_bq_m2_s(),
        sito.porosity,
        sito.saturation,
        impronta_m2,
        volume_m3,
        ach_h,
        conduttanza_m3_s,
    )


def voce_altezza() -> VoceIncertezza:
    riferimento = altezza_equivalente_m()
    return VoceIncertezza(
        nome=f"altezza equivalente h = V/A_f, da {riferimento:.1f} a {ALTEZZA_ALTA_M:.1f} m",
        basso=math.log10(riferimento / ALTEZZA_ALTA_M),
        alto=0.0,
        origine="nessuna misura di edificio sui tredici siti",
    )


def voce_ricambio() -> VoceIncertezza:
    riferimento = lambda_efficace(ACH_H)
    return VoceIncertezza(
        nome=f"ricambio d'aria, da {ACH_BASSO_H} a {ACH_ALTO_H} 1/h rispetto a {ACH_H}",
        basso=math.log10(riferimento / lambda_efficace(ACH_ALTO_H)),
        alto=math.log10(riferimento / lambda_efficace(ACH_BASSO_H)),
        origine="ACH_closed del modello originale, non misurato",
    )


def voce_soletta(sito: Site | None = None) -> VoceIncertezza:
    sito = sito if sito is not None else sito_di_riferimento()
    senza = _concentrazione(sito, conduttanza_m3_s=math.inf)
    con = _concentrazione(sito, conduttanza_m3_s=crack_conductance_m3_s(AREA_FESSURA_M2))
    return VoceIncertezza(
        nome=f"conduttanza della soletta, da assente a fessura di {AREA_FESSURA_M2} m2",
        basso=math.log10(con / senza),
        alto=0.0,
        origine=f"fattore {senza / con:.2f} fra soletta senza resistenza e fessura vera",
    )


def voce_reintegro() -> VoceIncertezza:
    percorso = data_directory() / "gallese2022_ach.csv"
    righe = {
        r["experiment"]: r
        for r in csv.DictReader(
            l for l in percorso.read_text(encoding="utf-8").splitlines()
            if not l.startswith("#")
        )
    }
    estratta = 1.0 + float(righe["6"]["RIR_pct"]) / 100.0
    immessa = 1.0 + float(righe["1"]["RIR_pct"]) / 100.0
    return VoceIncertezza(
        nome="aria di reintegro sotto depressione",
        basso=0.0,
        alto=math.log10(estratta / immessa),
        origine="Gallese 2022, esperimenti 6 e 1 a |ACH| = 0.49 1/h",
        misurata=True,
    )


def voci_calcolate(sito: Site | None = None) -> tuple[VoceIncertezza, ...]:
    return (voce_altezza(), voce_ricambio(), voce_soletta(sito), voce_reintegro())


def bilancio_anello_b(sito: Site | None = None, dispersione_a1_dex: float = 0.0) -> Banda:
    voci = voci_calcolate(sito)
    return Banda(
        statistica_dex=dispersione_a1_dex,
        sistematica_dex=(
            float(sum(v.basso for v in voci)),
            float(sum(v.alto for v in voci)),
        ),
        forma_non_quantificata=("C1",),
    )


def span_dex(sito: Site | None = None) -> float:
    basso, alto = bilancio_anello_b(sito).sistematica_dex
    return alto - basso


@dataclass(frozen=True)
class Sensibilita:

    parametro: str
    intervallo: str
    basso_dex: float
    alto_dex: float


def sensibilita_di_riferimento(sito: Site | None = None) -> tuple[Sensibilita, ...]:
    sito = sito if sito is not None else sito_di_riferimento()
    base = _concentrazione(sito)

    def dex(valore: float) -> float:
        return math.log10(valore / base)

    ricambio = Sensibilita(
        "ricambio d'aria",
        f"{ACH_BASSO_H} - {ACH_ALTO_H} 1/h",
        dex(_concentrazione(sito, ach_h=ACH_ALTO_H)),
        dex(_concentrazione(sito, ach_h=ACH_BASSO_H)),
    )
    volume = Sensibilita(
        "volume",
        "25 - 300 m3",
        dex(_concentrazione(sito, volume_m3=300.0)),
        dex(_concentrazione(sito, volume_m3=25.0)),
    )
    impronta = Sensibilita(
        "impronta",
        "20 - 200 m2",
        dex(_concentrazione(sito, impronta_m2=20.0)),
        dex(_concentrazione(sito, impronta_m2=200.0)),
    )
    soletta = Sensibilita(
        "conduttanza della soletta",
        f"assente - fessura {AREA_FESSURA_M2} m2",
        dex(_concentrazione(sito, conduttanza_m3_s=crack_conductance_m3_s(AREA_FESSURA_M2))),
        0.0,
    )
    return (ricambio, volume, impronta, soletta)


def verifica_contro_il_registro(sito: Site | None = None, tolleranza: float = 1e-6) -> None:
    calcolate = voci_calcolate(sito)
    dichiarate = ANELLI["B"].banda_sistematica_dex
    ottenute = (
        float(sum(v.basso for v in calcolate)),
        float(sum(v.alto for v in calcolate)),
    )
    for nome, atteso, ottenuto in zip(("basso", "alto"), dichiarate, ottenute):
        if abs(atteso - ottenuto) > tolleranza:
            raise ValueError(
                f"anello B, estremo {nome}: il registro dichiara {atteso!r}, "
                f"la fisica da' {ottenuto!r}"
            )


def main() -> int:
    sito = sito_di_riferimento()
    verifica_contro_il_registro(sito)
    voci = voci_calcolate(sito)
    banda = bilancio_anello_b(sito)
    basso, alto = banda.sistematica_dex

    print("ANELLO B — trasferimento verso l'ambiente — 0 punti misurati — NON VALIDATO")
    print(f"sito di riferimento: {sito.site_id}, J = {sito.physical_flux_bq_m2_s():.6f} Bq m-2 s-1")
    print()
    print(f"{'voce':<52}{'basso':>10}{'alto':>10}  {'misurata':<9}")
    for v in voci:
        print(f"{v.nome:<52}{v.basso:>+10.4f}{v.alto:>+10.4f}  {'si' if v.misurata else 'no':<9}")
    print(f"{'totale (somma lineare)':<52}{basso:>+10.4f}{alto:>+10.4f}")
    print(f"span {alto - basso:.4f} dex, cioe' un fattore {10 ** (alto - basso):.3g}")
    print()
    print("controlli indipendenti, un parametro alla volta:")
    for s in sensibilita_di_riferimento(sito):
        print(f"  {s.parametro:<28}{s.intervallo:<24}{s.basso_dex:>+9.3f}{s.alto_dex:>+9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
