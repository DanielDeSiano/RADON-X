from __future__ import annotations

import math

from radon_x.constants import HALF_LIVES_SECONDS
from radon_x.core.decay import decay_constant
from radon_x.core.units import ach_h_to_s

HALF_LIFE_PO218_S = HALF_LIVES_SECONDS["Po218"]
HALF_LIFE_PB214_S = HALF_LIVES_SECONDS["Pb214"]
HALF_LIFE_BI214_S = HALF_LIVES_SECONDS["Bi214"]

LAMBDA_PO218 = decay_constant(HALF_LIFE_PO218_S)
LAMBDA_PB214 = decay_constant(HALF_LIFE_PB214_S)
LAMBDA_BI214 = decay_constant(HALF_LIFE_BI214_S)

LAMBDA_PROGENIE = (LAMBDA_PO218, LAMBDA_PB214, LAMBDA_BI214)

DEPOSIZIONE_PROGENIE_S = ach_h_to_s(1.5)

PESI_PAEC = (0.105, 0.515, 0.380)


def secular_equilibrium_ratios(
    removal_s: float, deposition_s: float = DEPOSIZIONE_PROGENIE_S
) -> tuple[float, float, float]:
    if not math.isfinite(removal_s) or removal_s < 0.0:
        raise ValueError(f"rimozione non fisica: {removal_s!r}")
    if not math.isfinite(deposition_s) or deposition_s < 0.0:
        raise ValueError(f"deposizione non fisica: {deposition_s!r}")
    perdita = removal_s + deposition_s
    rapporti = []
    precedente = 1.0
    for lam in LAMBDA_PROGENIE:
        precedente = precedente * lam / (lam + perdita)
        rapporti.append(precedente)
    return tuple(rapporti)


def equilibrium_factor(
    removal_s: float, deposition_s: float = DEPOSIZIONE_PROGENIE_S
) -> float:
    f1, f2, f3 = secular_equilibrium_ratios(removal_s, deposition_s)
    p1, p2, p3 = PESI_PAEC
    return p1 * f1 + p2 * f2 + p3 * f3
