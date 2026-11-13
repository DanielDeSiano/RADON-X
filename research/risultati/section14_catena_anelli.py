from __future__ import annotations

from section14_anello_b import (
    bilancio_anello_b,
    fattore_di_trasferimento_s_m,
    fattore_impronta_s_m,
    sensibilita_di_riferimento,
    verifica_contro_il_registro,
    voci_calcolate,
)
from section14_anello_c import (
    RIGHE_CONSUMATE,
    RIGHE_LIBERE,
    falsificazione_sorgente_costante,
    metriche_c1,
    metriche_c2,
    misura_c1_confrontabile,
    misure_c2_libere,
    predizioni_c2_libere,
)
from section14_chain import (
    HEADER,
    fit_scaled_physics,
    leave_one_out,
    leave_one_source_out,
    metrics,
    paired_rmse_difference,
    physical,
    predict_scaled_physics,
)
from section14_dataset import Site, load_sites

from radon_x.anelli import (
    ANELLI,
    CATENA_INDOOR,
    Banda,
    ConfrontoNonAmmesso,
    Misurato,
    Predetto,
    confronta,
)


def misure_anello_a1(siti: list[Site]) -> list[Misurato]:
    return [
        Misurato(
            valore=s.measured_flux_bq_m2_s,
            unita=ANELLI["A1"].unita_uscita,
            file=s.source_file,
            riga=s.site_id,
            anelli_chiusi=frozenset({"A1"}),
            ruolo="VALIDATION",
        )
        for s in siti
    ]


def predizioni_anello_a1(siti: list[Site]) -> list[Predetto]:
    return [
        Predetto(valore=j, unita=ANELLI["A1"].unita_uscita, anelli=("A1",), etichetta=s.site_id)
        for j, s in zip(physical(siti), siti)
    ]


def tutte_le_misure(siti: list[Site] | None = None) -> list[Misurato]:
    siti = siti if siti is not None else load_sites()
    return misure_anello_a1(siti) + [misura_c1_confrontabile()] + misure_c2_libere()


def nessuna_misura_chiude_l_anello_b(siti: list[Site] | None = None) -> bool:
    return not any("B" in m.anelli_chiusi for m in tutte_le_misure(siti))


def dispersione_anello_a1(siti: list[Site]) -> float:
    misurati = [s.measured_flux_bq_m2_s for s in siti]
    fuori = leave_one_out(siti, lambda s: None, lambda c, s: physical(s))
    return metrics(fuori, misurati).rmse


def differenza_appaiata(siti: list[Site]) -> float:
    misurati = [s.measured_flux_bq_m2_s for s in siti]
    fisico = leave_one_out(siti, lambda s: None, lambda c, s: physical(s))
    appreso = leave_one_out(siti, fit_scaled_physics, predict_scaled_physics)
    appaiata, _ = paired_rmse_difference(fisico, appreso, misurati)
    return appaiata.point


def propaga(siti: list[Site], accoppiata: bool = False) -> list[Predetto]:
    banda = Banda(
        statistica_dex=dispersione_anello_a1(siti),
        sistematica_dex=bilancio_anello_b().sistematica_dex,
        forma_non_quantificata=("C1",),
    )
    previsioni = []
    for sito, flusso in zip(siti, physical(siti)):
        t = (
            fattore_di_trasferimento_s_m(sito.porosity, sito.saturation)
            if accoppiata
            else fattore_impronta_s_m()
        )
        previsioni.append(
            Predetto(
                valore=flusso * t,
                unita=ANELLI["C1"].unita_uscita,
                anelli=CATENA_INDOOR,
                banda=banda,
                etichetta=sito.site_id,
            )
        )
    return previsioni


def esegui_anello_a1(siti: list[Site]) -> None:
    a = ANELLI["A1"]
    print(f"ANELLO A1 — {a.uscita} [{a.unita_uscita}] — {a.n_punti} punti misurati "
          f"— {a.famiglia_metrica.value}")
    misurati = [s.measured_flux_bq_m2_s for s in siti]
    print(HEADER)
    print(metrics(physical(siti), misurati).line("fisica, in campione", "0"))
    print(
        metrics(leave_one_out(siti, lambda s: None, lambda c, s: physical(s)), misurati)
        .line("fisica, LOO", "0")
    )
    print(
        metrics(leave_one_out(siti, fit_scaled_physics, predict_scaled_physics), misurati)
        .line("fisica riscalata, LOO", "2")
    )
    print(
        metrics(
            leave_one_source_out(siti, fit_scaled_physics, predict_scaled_physics), misurati
        ).line("fisica riscalata, LOSO", "2")
    )
    print(f"differenza appaiata fra la fisica e il miglior modello appreso: "
          f"{differenza_appaiata(siti):+.6f} dex")


def esegui_anello_b(siti: list[Site]) -> None:
    b = ANELLI["B"]
    verifica_contro_il_registro()
    basso, alto = bilancio_anello_b().sistematica_dex
    print(f"ANELLO B — {b.uscita} — {b.n_punti} punti misurati — {b.stato.value.upper()}")
    print("Nessuna metrica: non esiste il lato misurato. Bilancio d'incertezza dichiarato.")
    print(f"{'voce':<52}{'basso':>10}{'alto':>10}")
    for v in voci_calcolate():
        print(f"{v.nome:<52}{v.basso:>+10.4f}{v.alto:>+10.4f}")
    print(f"{'totale (somma lineare)':<52}{basso:>+10.4f}{alto:>+10.4f}")
    print(f"span {alto - basso:.4f} dex, un fattore {10 ** (alto - basso):.3g}")
    print("controlli a un parametro alla volta:")
    for s in sensibilita_di_riferimento():
        print(f"  {s.parametro:<28}{s.intervallo:<26}{s.basso_dex:>+9.3f}{s.alto_dex:>+9.3f}")


def esegui_anello_c() -> None:
    c1, c2 = ANELLI["C1"], ANELLI["C2"]
    misurato = misura_c1_confrontabile()
    print(f"ANELLO C1 — {c1.uscita} [{c1.unita_uscita}] — {c1.n_punti} punto pieno "
          f"— {c1.famiglia_metrica.value}")
    print(f"  {misurato.riga}: misurato {misurato.valore:.0f} +/- {misurato.incertezza:.0f} "
          f"{misurato.unita}")
    for scenario in ("E", "2E"):
        m = metriche_c1(scenario)
        print(f"  scenario {scenario:<3} predetto/misurato = {m.rapporto_mediano:.4f}"
              f"   z = {m.z_max:.1f}")
    print()
    print(f"ANELLO C2 — {c2.uscita} [{c2.unita_uscita}] — {c2.n_punti} righe libere "
          f"— {c2.famiglia_metrica.value}")
    for p, o in zip(predizioni_c2_libere(), misure_c2_libere()):
        print(f"  {o.riga:<8} misurato {o.valore:>7.1f} +/- {o.incertezza:<4.1f}"
              f"  predetto {p.valore:>7.1f}   z = {abs(p.valore - o.valore) / o.incertezza:.2f}")
    m2 = metriche_c2()
    print(f"  n = {m2.n}   z mediano {m2.z_mediano:.2f}   z max {m2.z_max:.2f}"
          f"   entro 2 sigma {100 * m2.entro_2sigma:.0f}%")
    mancate, totali, z_min, z_max = falsificazione_sorgente_costante()
    print(f"  la sorgente costante manca {mancate} righe su {totali}, |z| fra {z_min:.1f} e {z_max:.1f}")
    print(f"  righe consumate da una stima: {sorted(RIGHE_CONSUMATE)}; libere: {list(RIGHE_LIBERE)}")


def esegui_propagazione(siti: list[Site]) -> None:
    previsioni = propaga(siti)
    banda = previsioni[0].banda
    basso, alto = banda.totale_dex()
    print("PROPAGAZIONE — C_indoor per i tredici siti — PREDETTA, NON MISURATA")
    print(f"{'sito':<16}{'C_indoor':>12}{'stat.':>10}{'sistematica':>16}{'intervallo totale':>30}")
    for p in previsioni:
        basso_i, alto_i = p.intervallo()
        sistematica = f"{banda.sistematica_dex[0]:+.3f}/{banda.sistematica_dex[1]:+.3f}"
        print(f"{p.etichetta:<16}{p.valore:>12.3e}{banda.statistica_dex:>9.3f}d"
              f"{sistematica:>16}{f'[{basso_i:.2e}, {alto_i:.2e}]':>30}")
    print()
    print(f"banda statistica  +/-{banda.statistica_dex:.3f} dex   misurata su {ANELLI['A1'].n_punti} punti")
    print(f"banda sistematica {banda.sistematica_dex[0]:+.3f} / {banda.sistematica_dex[1]:+.3f} dex"
          f"   dichiarata su {ANELLI['B'].n_punti} punti")
    print(f"intervallo totale {basso:+.3f} / {alto:+.3f} dex   decide la componente {banda.dominante()}")
    appaiata = differenza_appaiata(siti)
    print(f"lo span non misurato vale {banda.span_dex() / banda.statistica_dex:.1f} volte la "
          f"dispersione dell'anello A1")
    print(f"e {banda.span_dex() / appaiata:.0f} volte la differenza appaiata su cui la sezione 14 "
          f"confronta i modelli")


def esegui_barriera(siti: list[Site]) -> None:
    print("BARRIERA — perche' non esiste una metrica su C_indoor per questi siti")
    print(f"  nessuna misura del progetto chiude l'anello B: {nessuna_misura_chiude_l_anello_b(siti)}")
    previsione = propaga(siti)[0]
    try:
        confronta([previsione], [misura_c1_confrontabile()])
    except ConfrontoNonAmmesso as errore:
        print(f"  confronto rifiutato: {errore}")
    print(f"  una predizione propagata si stampa cosi': {previsione:.3e}")


def main() -> int:
    siti = load_sites()
    esegui_anello_a1(siti)
    print()
    esegui_anello_b(siti)
    print()
    esegui_anello_c()
    print()
    esegui_propagazione(siti)
    print()
    esegui_barriera(siti)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
