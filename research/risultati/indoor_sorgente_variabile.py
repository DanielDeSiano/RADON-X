from __future__ import annotations

import math

LAMBDA_H: float = 2.09794e-6 * 3600.0

BACKGROUND_ACH_H: float = 0.0389

BASELINE_COVERED_BQ_M3: float = 641.0

BASELINE_BARE_BQ_M3: float = 770.0

OUTDOOR_BQ_M3: float = 30.0


def concentration_ratio(
    ach_h: float,
    reentry_rate_h: float,
    makeup_ratio: float,
    outdoor_ratio: float,
    background_ach_h: float = BACKGROUND_ACH_H,
) -> float:
    for nome, valore in (
        ("ach_h", ach_h),
        ("reentry_rate_h", reentry_rate_h),
        ("makeup_ratio", makeup_ratio),
        ("outdoor_ratio", outdoor_ratio),
        ("background_ach_h", background_ach_h),
    ):
        if not math.isfinite(valore):
            raise ValueError(f"{nome} deve essere finito, ricevuto {valore!r}")
        if valore < 0.0:
            raise ValueError(f"{nome} deve essere >= 0, ricevuto {valore!r}")

    numeratore = (
        (LAMBDA_H + background_ach_h + reentry_rate_h)
        - background_ach_h * outdoor_ratio
        + ach_h * makeup_ratio
    )
    return numeratore / (LAMBDA_H + ach_h + reentry_rate_h)


def rir_pct(
    ach_h: float,
    reentry_rate_h: float,
    makeup_ratio: float,
    outdoor_ratio: float,
    background_ach_h: float = BACKGROUND_ACH_H,
) -> float:
    return (
        concentration_ratio(
            ach_h, reentry_rate_h, makeup_ratio, outdoor_ratio, background_ach_h
        )
        - 1.0
    ) * 100.0


def rir_constant_source_pct(
    ach_h: float, background_ach_h: float = BACKGROUND_ACH_H
) -> float:
    if ach_h < 0.0:
        raise ValueError(f"ach_h deve essere >= 0 (usare il modulo), ricevuto {ach_h!r}")
    return ((LAMBDA_H + background_ach_h) / (LAMBDA_H + ach_h) - 1.0) * 100.0


REENTRY_COVERED_H: float = 0.0

REENTRY_BARE_H: float = 0.8082

MAKEUP_COVERED_RATIO: float = 0.670

MAKEUP_BARE_RATIO: float = 0.215

CALIBRATION_EXPERIMENTS: frozenset = frozenset({6, 9, 11})


def configuration(inner_cover: str) -> tuple[float, float, float]:
    if inner_cover == "yes":
        return (
            REENTRY_COVERED_H,
            MAKEUP_COVERED_RATIO,
            OUTDOOR_BQ_M3 / BASELINE_COVERED_BQ_M3,
        )
    if inner_cover == "no":
        return (
            REENTRY_BARE_H,
            MAKEUP_BARE_RATIO,
            OUTDOOR_BQ_M3 / BASELINE_BARE_BQ_M3,
        )
    raise ValueError(f"inner_cover deve essere 'yes' o 'no', ricevuto {inner_cover!r}")


def predict_rir_pct(ach_h: float, mode: str, inner_cover: str) -> float:
    reentry, makeup, outdoor = configuration(inner_cover)
    if mode == "air_in":
        aria = outdoor
    elif mode == "air_out":
        aria = makeup
    else:
        raise ValueError(f"mode deve essere 'air_in' o 'air_out', ricevuto {mode!r}")
    return rir_pct(ach_h, reentry, aria, outdoor)
