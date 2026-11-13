from __future__ import annotations

import numpy as np

__all__ = [
    "SEME_RADICE",
    "USI",
    "derive",
    "semi",
]

USI: tuple[str, ...] = (
    "bootstrap",
    "lhs",
    "morris",
    "sobol",
    "montecarlo_gum",
    "kfold",
)

SEME_RADICE: int = 42


def derive(seed: int, uso: str) -> int:
    if uso not in USI:
        raise ValueError(
            f"uso non dichiarato: {uso!r}; gli usi sono {', '.join(USI)}"
        )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"la radice deve essere un intero, ricevuto {seed!r}")
    if int(seed) < 0:
        raise ValueError(f"la radice deve essere >= 0, ricevuto {seed!r}")
    figli = np.random.SeedSequence(int(seed)).spawn(len(USI))
    stato = figli[USI.index(uso)].generate_state(1, dtype=np.uint32)
    return int(stato[0])


def semi(seed: int = SEME_RADICE) -> dict[str, int]:
    return {uso: derive(seed, uso) for uso in USI}
