from __future__ import annotations

import math
from collections.abc import Mapping

from radon_x.constants import DECAY_CONSTANTS_PER_SECOND, HALF_LIVES_SECONDS

__all__ = [
    "DECAY_CHAIN_RN222",
    "HALF_LIFE_RN222_S",
    "LAMBDA_RN222",
    "activity_after",
    "decay_constant",
    "diffusion_length",
    "half_life",
]


def decay_constant(half_life_s: float) -> float:
    if not math.isfinite(half_life_s):
        raise ValueError(f"half_life_s must be finite, got {half_life_s!r}")
    if half_life_s <= 0.0:
        raise ValueError(f"half_life_s must be > 0 s, got {half_life_s!r}")
    return math.log(2.0) / half_life_s


def half_life(decay_constant_s: float) -> float:
    if not math.isfinite(decay_constant_s):
        raise ValueError(f"decay_constant_s must be finite, got {decay_constant_s!r}")
    if decay_constant_s <= 0.0:
        raise ValueError(f"decay_constant_s must be > 0 s-1, got {decay_constant_s!r}")
    return math.log(2.0) / decay_constant_s


def activity_after(a0_bq_m3: float, lambda_s: float, t_s: float) -> float:
    if not math.isfinite(a0_bq_m3):
        raise ValueError(f"a0_bq_m3 must be finite, got {a0_bq_m3!r}")
    if a0_bq_m3 < 0.0:
        raise ValueError(f"a0_bq_m3 must be >= 0 Bq m-3, got {a0_bq_m3!r}")
    if not math.isfinite(lambda_s):
        raise ValueError(f"lambda_s must be finite, got {lambda_s!r}")
    if lambda_s <= 0.0:
        raise ValueError(f"lambda_s must be > 0 s-1, got {lambda_s!r}")
    if not math.isfinite(t_s):
        raise ValueError(f"t_s must be finite, got {t_s!r}")
    if t_s < 0.0:
        raise ValueError(f"t_s must be >= 0 s, got {t_s!r}")
    return a0_bq_m3 * math.exp(-lambda_s * t_s)


def diffusion_length(d_apparent_m2_s: float, lambda_s: float) -> float:
    if not math.isfinite(d_apparent_m2_s):
        raise ValueError(f"d_apparent_m2_s must be finite, got {d_apparent_m2_s!r}")
    if d_apparent_m2_s <= 0.0:
        raise ValueError(f"d_apparent_m2_s must be > 0 m2 s-1, got {d_apparent_m2_s!r}")
    if not math.isfinite(lambda_s):
        raise ValueError(f"lambda_s must be finite, got {lambda_s!r}")
    if lambda_s <= 0.0:
        raise ValueError(f"lambda_s must be > 0 s-1, got {lambda_s!r}")
    return math.sqrt(d_apparent_m2_s / lambda_s)


HALF_LIFE_RN222_S: float = HALF_LIVES_SECONDS["Rn222"]

LAMBDA_RN222: float = DECAY_CONSTANTS_PER_SECOND["Rn222"]

_CHAIN_RN222_ORDER = ("Rn222", "Po218", "Pb214", "Bi214", "Po214")

DECAY_CHAIN_RN222: Mapping[str, float] = {
    isotope: DECAY_CONSTANTS_PER_SECOND[isotope] for isotope in _CHAIN_RN222_ORDER
}
