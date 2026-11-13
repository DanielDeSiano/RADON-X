from __future__ import annotations

import math

__all__ = [
    "SECONDI_PER_ORA",
    "ach_h_to_s",
]

SECONDI_PER_ORA: float = 3600.0


def ach_h_to_s(ach_h: float) -> float:
    valore = float(ach_h)
    if not math.isfinite(valore):
        raise ValueError(f"rateo non finito: {ach_h!r}")
    if valore < 0.0:
        raise ValueError(f"rateo negativo: {ach_h!r}")
    return valore / SECONDI_PER_ORA
