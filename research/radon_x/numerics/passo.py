from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "COEFFICIENTI_LEGACY",
    "PAVIMENTO_S",
    "SOFFITTO_S",
    "VINCOLI",
    "LimitiDelPasso",
    "cfl_dt",
    "limiti_del_passo",
    "numero_di_fourier",
]

COEFFICIENTI_LEGACY: tuple[float, float, float] = (0.5, 0.4, 0.3)

PAVIMENTO_S: float = 30.0

SOFFITTO_S: float = 1800.0

VINCOLI: tuple[str, str, str] = ("fourier", "fronte", "decadimento")


def _positivo(nome: str, valore: float, unita: str) -> float:
    numero = float(valore)
    if not math.isfinite(numero):
        raise ValueError(f"{nome} deve essere finito, ricevuto {valore!r}")
    if numero <= 0.0:
        raise ValueError(f"{nome} deve essere > 0 {unita}, ricevuto {valore!r}")
    return numero


@dataclass(frozen=True)
class LimitiDelPasso:

    dt_fourier_s: float
    dt_fronte_s: float
    dt_decadimento_s: float
    dt_cfl_s: float
    dt_s: float
    vincolo: str
    pavimento_attivo: bool
    soffitto_attivo: bool
    numero_di_fourier: float
    cfl_violata: bool


def numero_di_fourier(d_e: float, beta: float, dz: float, dt: float) -> float:
    return _positivo("d_e", d_e, "m2 s-1") * _positivo("dt", dt, "s") / (
        _positivo("beta", beta, "m3 m-3") * _positivo("dz", dz, "m") ** 2
    )


def limiti_del_passo(
    d_e: float,
    beta: float,
    dz: float,
    lam: float,
    floor_s: float = PAVIMENTO_S,
    ceiling_s: float = SOFFITTO_S,
    coefficienti: tuple[float, float, float] = COEFFICIENTI_LEGACY,
) -> LimitiDelPasso:
    d_e_v = _positivo("d_e", d_e, "m2 s-1")
    beta_v = _positivo("beta", beta, "m3 m-3")
    dz_v = _positivo("dz", dz, "m")
    lam_v = _positivo("lam", lam, "s-1")
    pavimento = _positivo("floor_s", floor_s, "s")
    soffitto = _positivo("ceiling_s", ceiling_s, "s")
    if soffitto < pavimento:
        raise ValueError(
            f"ceiling_s deve essere >= floor_s, ricevuti {ceiling_s!r} e {floor_s!r}"
        )
    terna = tuple(coefficienti)
    if len(terna) != 3:
        raise ValueError(f"coefficienti deve avere tre elementi, ricevuti {len(terna)}")
    c0, c1, c2 = (
        _positivo(f"coefficienti[{indice}]", valore, "-")
        for indice, valore in enumerate(terna)
    )

    dt_fourier = c0 * beta_v * dz_v * dz_v / d_e_v
    dt_fronte = c1 * dz_v * math.sqrt(beta_v / (d_e_v * lam_v))
    dt_decadimento = c2 / lam_v

    candidati = (dt_fourier, dt_fronte, dt_decadimento)
    dt_cfl = min(candidati)
    vincolo = VINCOLI[candidati.index(dt_cfl)]
    dt = min(max(dt_cfl, pavimento), soffitto)

    return LimitiDelPasso(
        dt_fourier_s=dt_fourier,
        dt_fronte_s=dt_fronte,
        dt_decadimento_s=dt_decadimento,
        dt_cfl_s=dt_cfl,
        dt_s=dt,
        vincolo=vincolo,
        pavimento_attivo=dt > dt_cfl,
        soffitto_attivo=dt < dt_cfl,
        numero_di_fourier=numero_di_fourier(d_e_v, beta_v, dz_v, dt),
        cfl_violata=dt > dt_cfl,
    )


def cfl_dt(
    d_e: float,
    beta: float,
    dz: float,
    lam: float,
    floor_s: float = PAVIMENTO_S,
    ceiling_s: float = SOFFITTO_S,
    coefficienti: tuple[float, float, float] = COEFFICIENTI_LEGACY,
) -> float:
    return limiti_del_passo(
        d_e, beta, dz, lam, floor_s=floor_s, ceiling_s=ceiling_s, coefficienti=coefficienti
    ).dt_s
