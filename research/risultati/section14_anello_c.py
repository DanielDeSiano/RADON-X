from __future__ import annotations

from gallese_box_model import part_a, part_b
from indoor_sorgente_variabile import CALIBRATION_EXPERIMENTS, predict_rir_pct

from radon_x.anelli import (
    ANELLI,
    MetricheRapporto,
    MetricheZ,
    Misurato,
    Predetto,
    confronta,
)

ESPERIMENTO_DEL_FONDO = 5

RIGHE_CONSUMATE: frozenset[int] = frozenset({ESPERIMENTO_DEL_FONDO}) | CALIBRATION_EXPERIMENTS

RIGHE_LIBERE: tuple[int, ...] = (7, 8, 10, 12)

EQUILIBRIO_CONFRONTABILE = "2"


class RigaConsumata(PermissionError):
    pass


def assert_riga_libera(esperimento: int) -> None:
    if esperimento in RIGHE_CONSUMATE:
        raise RigaConsumata(
            f"esperimento {esperimento} di gallese2022_ach.csv e' stato speso per "
            f"stimare un parametro: e' calibrazione, non validazione"
        )


def misure_c1() -> list[Misurato]:
    misure = []
    for riga in part_a()["rows"]:
        misure.append(
            Misurato(
                valore=float(riga["c_meas"]),
                unita=ANELLI["C1"].unita_uscita,
                file="gallese2020_equilibri.csv",
                riga=f"test-{riga['experiment']}",
                anelli_chiusi=frozenset({"C1"}),
                ruolo="VALIDATION",
                incertezza=float(riga["c_unc"]),
            )
        )
    return misure


def misura_c1_confrontabile() -> Misurato:
    for m in misure_c1():
        if m.riga == f"test-{EQUILIBRIO_CONFRONTABILE}":
            return m
    raise LookupError(f"test {EQUILIBRIO_CONFRONTABILE} assente da gallese2020_equilibri.csv")


def predizione_c1(scenario: str = "2E") -> Predetto:
    a = part_a()
    if scenario == "E":
        valore = float(a["c_eq0"])
    elif scenario == "2E":
        valore = float(a["c_eq0_sealed_2e"])
    else:
        raise ValueError(f"scenario deve essere 'E' o '2E', ricevuto {scenario!r}")
    return Predetto(
        valore=valore,
        unita=ANELLI["C1"].unita_uscita,
        anelli=("C1",),
        etichetta=scenario,
    )


def metriche_c1(scenario: str = "2E") -> MetricheRapporto:
    risultato = confronta([predizione_c1(scenario)], [misura_c1_confrontabile()])
    assert isinstance(risultato, MetricheRapporto)
    return risultato


def _righe_2022() -> dict[int, dict]:
    return {int(r["experiment"]): r for r in part_b()["rows"]}


def misure_c2_libere() -> list[Misurato]:
    righe = _righe_2022()
    misure = []
    for esperimento in RIGHE_LIBERE:
        assert_riga_libera(esperimento)
        r = righe[esperimento]
        misure.append(
            Misurato(
                valore=float(r["rir_meas"]),
                unita=ANELLI["C2"].unita_uscita,
                file="gallese2022_ach.csv",
                riga=f"esp-{esperimento}",
                anelli_chiusi=frozenset({"C2"}),
                ruolo="CALIBRATION (riga libera)",
                incertezza=float(r["rir_unc"]),
            )
        )
    return misure


def predizioni_c2_libere() -> list[Predetto]:
    righe = _righe_2022()
    previsioni = []
    for esperimento in RIGHE_LIBERE:
        r = righe[esperimento]
        previsioni.append(
            Predetto(
                valore=predict_rir_pct(
                    abs(float(r["ach"])), str(r["mode"]), str(r["inner_cover"])
                ),
                unita=ANELLI["C2"].unita_uscita,
                anelli=("C2",),
                etichetta=f"esp-{esperimento}",
            )
        )
    return previsioni


def metriche_c2() -> MetricheZ:
    risultato = confronta(predizioni_c2_libere(), misure_c2_libere())
    assert isinstance(risultato, MetricheZ)
    return risultato


def falsificazione_sorgente_costante() -> tuple[int, int, float, float]:
    z_mancate = [
        abs(float(r["z"]))
        for r in part_b()["rows"]
        if int(r["experiment"]) != ESPERIMENTO_DEL_FONDO and abs(float(r["z"])) > 3.0
    ]
    validazione = [
        r for r in part_b()["rows"] if int(r["experiment"]) != ESPERIMENTO_DEL_FONDO
    ]
    return (
        len(z_mancate),
        len(validazione),
        min(z_mancate) if z_mancate else float("nan"),
        max(z_mancate) if z_mancate else float("nan"),
    )


def main() -> int:
    print("ANELLO C1 — concentrazione di equilibrio [Bq m-3] — 1 punto pieno — rapporti")
    misurato = misura_c1_confrontabile()
    print(f"  misura: {misurato.valore:.0f} +/- {misurato.incertezza:.0f} Bq m-3 "
          f"({misurato.file}, {misurato.riga})")
    for scenario in ("E", "2E"):
        m = metriche_c1(scenario)
        print(f"  scenario {scenario:<3} predetto/misurato = {m.rapporto_mediano:.4f}"
              f"   z = {m.z_max:.1f}")
    print("  le altre cinque configurazioni non sono pienamente confrontabili")
    print()

    print("ANELLO C2 — riduzione relativa [punti percentuali] — 4 righe libere — z")
    m2 = metriche_c2()
    for p, o in zip(predizioni_c2_libere(), misure_c2_libere()):
        z = abs(p.valore - o.valore) / o.incertezza
        print(f"  {o.riga:<8} misurato {o.valore:>7.1f} +/- {o.incertezza:<4.1f}"
              f"  predetto {p.valore:>7.1f}   z = {z:.2f}")
    print(f"  n = {m2.n}   residuo medio {m2.residuo_medio_pp:+.2f} pp"
          f"   z mediano {m2.z_mediano:.2f}   z max {m2.z_max:.2f}")
    print(f"  entro 1 sigma {100 * m2.entro_1sigma:.0f}%   entro 2 sigma {100 * m2.entro_2sigma:.0f}%")
    print()

    mancate, totali, z_min, z_max = falsificazione_sorgente_costante()
    print(f"  il bilancio a sorgente costante manca {mancate} righe su {totali}, "
          f"z fra {z_min:.1f} e {z_max:.1f}")
    print(f"  righe consumate da una stima: {sorted(RIGHE_CONSUMATE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
