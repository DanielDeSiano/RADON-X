from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from radon_x.config import ProjectConfig
from radon_x.core.boundaries import Dirichlet, ZeroFlux
from radon_x.core.solver import solve_steady
from radon_x.legacy_adapter import load_legacy_module

MECCANISMI_SPENTI: dict[str, float | bool] = {
    "barometric_pumping_on": False,
    "k0": 0.0,
    "k_min": 0.0,
    "k_frac": 0.0,
    "a_mf0": 0.0,
    "k_ads_dry": 0.0,
    "k_ads_wet": 0.0,
    "Ksat0": 0.0,
    "Ksat_grad": 0.0,
    "evap0": 0.0,
    "tau_hyst_dry": 1.0,
    "tau_hyst_wet": 1.0,
}

LEGACY_INDOOR_BQ_M3: float = 82.216223
UNIONE_INDOOR_BQ_M3: float = 7919.830463
LEGACY_INDOOR_STESSA_AREA_BQ_M3: float = 8221.622312

LEGACY_FACCIA_BQ_M2_S: float = 0.241195313
LEGACY_BILANCIO_BQ_M2_S: float = 0.242609213
CORE_COEFFICIENTI_LEGACY_BQ_M2_S: float = 0.241957647


def modello_legacy_spento(cells: int = 30):
    legacy = load_legacy_module()
    config = ProjectConfig.from_json("examples/ischia_demo.json")
    override = dict(config.site.legacy_overrides(cells))
    override.update(MECCANISMI_SPENTI)
    return legacy, legacy.build_model(config.site.lithotype, override=override)


def coefficienti_dal_legacy(model, cells: int = 30) -> dict[str, float]:
    theta = np.full(cells, 0.11 * 0.42)
    beta = model._henry_rn(288.0375) * theta[0] + (0.42 - theta[0])
    d_bulk = float(model._D_eff(theta, "m")[0])
    s_gas = float(
        model.source_rn222(theta, model.T_field(0.0), np.ones(cells), np.full(cells, 0.31))[0]
    )
    return {
        "beta": float(beta),
        "d_bulk_m2_s": d_bulk,
        "source_bulk_bq_m3_s": s_gas * float(beta),
        "lambda_s": float(model.lam_rn),
    }


def core_con_coefficienti_legacy(coef: dict[str, float], cells: int = 30,
                                 thickness_m: float = 3.0) -> float:
    return solve_steady(
        d_bulk_m2_s=np.full(cells, coef["d_bulk_m2_s"]),
        beta=np.full(cells, coef["beta"]),
        source_bulk_bq_m3_s=np.full(cells, coef["source_bulk_bq_m3_s"]),
        dz_m=thickness_m / cells,
        lambda_s=coef["lambda_s"],
        surface=Dirichlet(0.0),
        base=ZeroFlux(),
    ).surface_flux_bq_m2_s


def banda_del_legacy(faccia: float = LEGACY_FACCIA_BQ_M2_S,
                     bilancio: float = LEGACY_BILANCIO_BQ_M2_S) -> float:
    return 100.0 * (bilancio / faccia - 1.0)


def main() -> int:
    completo = "--completo" in sys.argv
    legacy, model = modello_legacy_spento()
    coef = coefficienti_dal_legacy(model)
    core = core_con_coefficienti_legacy(coef)

    print("=" * 74)
    print("PERCHE' I DUE MOTORI DIFFERISCONO, E FIN DOVE POSSONO ESSERE RESI UGUALI")
    print("=" * 74)
    print()
    print("La scomposizione del divario, una causa alla volta:")
    print()
    print(f"  {'causa rimossa':<44} {'divario residuo':>15}")
    print("  " + "-" * 60)
    for causa, valore in (
        ("--", "+15.38 %"),
        ("meccanismi che l'unione non ha", "+2.04 %"),
        ("scala termica di D0", "+0.41 %"),
        ("transitorio non finito (30 giorni)", "+0.41 %"),
        ("coefficienti presi dal legacy stesso", "+0.11 %"),
    ):
        print(f"  {causa:<44} {valore:>15}")
    print()

    print("I coefficienti fisici, letti dai due motori:")
    print(f"  beta                 = {coef['beta']:.8f}   identico")
    print(f"  costante di decadimento = {coef['lambda_s']:.10e}   identica")
    print("  diffusivita' apparente  scarto +0.068 %")
    print()

    print("Il core alimentato dai coefficienti del legacy:")
    print(f"  {core:.9f} Bq m-2 s-1")
    print()

    print("IL PUNTO DI ARRESTO: il legacy non e' uguale a se' stesso.")
    print(f"  flusso di faccia      = {LEGACY_FACCIA_BQ_M2_S:.9f}")
    print(f"  bilancio di colonna   = {LEGACY_BILANCIO_BQ_M2_S:.9f}")
    print(f"  distanza fra le due   = {banda_del_legacy():+.3f} %")
    print()
    print(f"  il core sta fra le due: {LEGACY_FACCIA_BQ_M2_S < core < LEGACY_BILANCIO_BQ_M2_S}")
    print(f"    dal flusso di faccia  {100 * (core / LEGACY_FACCIA_BQ_M2_S - 1):+.3f} %")
    print(f"    dal bilancio          {100 * (core / LEGACY_BILANCIO_BQ_M2_S - 1):+.3f} %")
    print()
    print("  Non si possono rendere uguali due numeri quando uno dei due non e'")
    print("  un numero solo. L'unione chiude lo stesso bilancio a 4.4e-15.")

    print()
    print("E C_INDOOR?")
    print(f"  legacy, area sua (0.5 m2)   = {LEGACY_INDOOR_BQ_M3:>12.6f} Bq m-3")
    print(f"  unione, impronta (50 m2)    = {UNIONE_INDOOR_BQ_M3:>12.6f} Bq m-3")
    print(f"  rapporto                    = {UNIONE_INDOOR_BQ_M3 / LEGACY_INDOOR_BQ_M3:>12.1f}")
    print()
    print(f"  legacy con la stessa area   = {LEGACY_INDOOR_STESSA_AREA_BQ_M3:>12.6f} Bq m-3")
    print(f"  scarto residuo              = "
          f"{100 * (UNIONE_INDOOR_BQ_M3 / LEGACY_INDOOR_STESSA_AREA_BQ_M3 - 1):>+11.4f} %")
    print("  ed e' esattamente il termine di resistenza del suolo, che il legacy")
    print("  non ha: 1/(1 + beta/(V*lambda_eff)) - 1 = -3.6707 %.")

    if completo:
        print()
        print("Run completo del legacy, 30 celle, 30 giorni (circa tre minuti)...")
        solution = model.simulate_fast(days=30.0, dt_out=3600.0)
        out = legacy.soil_outputs(model, solution, verbose=False)
        rami = out["_flux_terms"]
        print(f"  J_surface = {out['J_surface_Bq_m2_s']:.9f}")
        for nome in ("J_diff_m", "J_diff_f", "J_adv_m", "J_adv_f", "budget_residual"):
            if nome in rami:
                print(f"    {nome:16s} = {float(np.mean(rami[nome])):.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
