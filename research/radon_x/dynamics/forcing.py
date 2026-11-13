from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from radon_x.core.units import SECONDI_PER_ORA

SECONDI_PER_GIORNO: float = 24.0 * SECONDI_PER_ORA

ACH_PORTE_CHIUSE_H: float = 0.10

ACH_PORTE_APERTE_H: float = 1.20

FINESTRE_PORTE_S: tuple[tuple[float, float], ...] = (
    (25200.0, 32400.0),
    (68400.0, 79200.0),
)


@dataclass(frozen=True)
class TimeSeriesProfile:

    times_s: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times_s) != len(self.values) or not self.times_s:
            raise ValueError("servono tanti tempi quanti valori, almeno uno")
        if any(not math.isfinite(t) for t in self.times_s):
            raise ValueError("tempi non finiti")
        if any(b <= a for a, b in zip(self.times_s, self.times_s[1:])):
            raise ValueError(f"i tempi devono crescere strettamente: {self.times_s}")
        if any(not math.isfinite(v) for v in self.values):
            raise ValueError("valori non finiti")

    @classmethod
    def constant(cls, value: float) -> TimeSeriesProfile:
        return cls(times_s=(0.0,), values=(float(value),))

    def value(self, t_s: float) -> float:
        k = int(np.searchsorted(np.asarray(self.times_s), t_s, side="right")) - 1
        return self.values[max(k, 0)]

    def breakpoints(self) -> tuple[float, ...]:
        return self.times_s


@dataclass(frozen=True)
class PiecewiseProfile:

    baseline: float
    intervals: tuple[tuple[float, float, float], ...] = ()
    period_s: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.baseline):
            raise ValueError(f"base non finita: {self.baseline!r}")
        if self.period_s is not None and (
            not math.isfinite(self.period_s) or self.period_s <= 0.0
        ):
            raise ValueError(f"periodo non fisico: {self.period_s!r}")
        precedente = -math.inf
        for tratto in self.intervals:
            if len(tratto) != 3:
                raise ValueError(f"un tratto e' (t_inizio, t_fine, valore): {tratto!r}")
            t0, t1, valore = tratto
            if not all(math.isfinite(x) for x in (t0, t1, valore)):
                raise ValueError(f"tratto non finito: {tratto!r}")
            if t1 <= t0:
                raise ValueError(f"tratto di durata non positiva: {tratto!r}")
            if t0 < precedente:
                raise ValueError(
                    f"tratti sovrapposti o non ordinati: {self.intervals}"
                )
            if self.period_s is not None and (t0 < 0.0 or t1 > self.period_s):
                raise ValueError(
                    f"il tratto {tratto!r} esce dal periodo {self.period_s}"
                )
            precedente = t1

    def value(self, t_s: float) -> float:
        if not math.isfinite(t_s):
            raise ValueError(f"tempo non finito: {t_s!r}")
        t = t_s if self.period_s is None else t_s % self.period_s
        for t0, t1, valore in self.intervals:
            if t0 <= t < t1:
                return float(valore)
        return float(self.baseline)

    def breakpoints(self, t_start_s: float, t_end_s: float) -> tuple[float, ...]:
        if not (math.isfinite(t_start_s) and math.isfinite(t_end_s)):
            raise ValueError("orizzonte non finito")
        if not t_end_s > t_start_s:
            raise ValueError(f"orizzonte vuoto: [{t_start_s}, {t_end_s}]")
        bordi: set[float] = set()
        if self.period_s is None:
            for t0, t1, _valore in self.intervals:
                bordi.update((t0, t1))
        else:
            primo = math.floor(t_start_s / self.period_s)
            ultimo = math.floor(t_end_s / self.period_s)
            for k in range(primo, ultimo + 1):
                base = k * self.period_s
                for t0, t1, _valore in self.intervals:
                    bordi.update((base + t0, base + t1))
        dentro = sorted(t for t in bordi if t_start_s < t < t_end_s)
        return (t_start_s, *dentro)

    def to_time_series(self, t_start_s: float, t_end_s: float) -> TimeSeriesProfile:
        tempi = self.breakpoints(t_start_s, t_end_s)
        return TimeSeriesProfile(
            times_s=tempi, values=tuple(self.value(t) for t in tempi)
        )


def door_ach_profile(
    ach_closed_h: float = ACH_PORTE_CHIUSE_H,
    ach_open_h: float = ACH_PORTE_APERTE_H,
    windows_s: tuple[tuple[float, float], ...] = FINESTRE_PORTE_S,
    period_s: float = SECONDI_PER_GIORNO,
) -> PiecewiseProfile:
    if not math.isfinite(ach_closed_h) or ach_closed_h < 0.0:
        raise ValueError(f"ricambio a porte chiuse non fisico: {ach_closed_h!r}")
    if not math.isfinite(ach_open_h) or ach_open_h < 0.0:
        raise ValueError(f"ricambio a porte aperte non fisico: {ach_open_h!r}")
    return PiecewiseProfile(
        baseline=float(ach_closed_h),
        intervals=tuple(
            (float(t0), float(t1), float(ach_open_h)) for t0, t1 in windows_s
        ),
        period_s=float(period_s),
    )


@dataclass(frozen=True)
class SourceTerm:

    zone: str
    rate_bq_s: TimeSeriesProfile
    origin: str

    def __post_init__(self) -> None:
        if any(v < 0.0 for v in self.rate_bq_s.values):
            raise ValueError(f"sorgente in {self.zone!r} con rateo negativo")
        if not self.origin:
            raise ValueError("la provenienza della sorgente va dichiarata")

    @classmethod
    def constant(cls, zone: str, rate_bq_s: float, origin: str) -> SourceTerm:
        return cls(zone=zone, rate_bq_s=TimeSeriesProfile.constant(rate_bq_s), origin=origin)


@dataclass(frozen=True)
class Forcing:

    sources: tuple[SourceTerm, ...] = ()
    outdoor_bq_m3: TimeSeriesProfile = field(
        default_factory=lambda: TimeSeriesProfile.constant(0.0)
    )

    def __post_init__(self) -> None:
        zone = [s.zone for s in self.sources]
        if len(set(zone)) != len(zone):
            raise ValueError(f"due sorgenti sulla stessa zona: {zone}")
        if any(v < 0.0 for v in self.outdoor_bq_m3.values):
            raise ValueError("concentrazione esterna negativa")

    def source_rate(self, zone: str, t_s: float) -> float:
        for s in self.sources:
            if s.zone == zone:
                return s.rate_bq_s.value(t_s)
        return 0.0

    def breakpoints(self) -> tuple[float, ...]:
        punti: set[float] = set(self.outdoor_bq_m3.breakpoints())
        for s in self.sources:
            punti.update(s.rate_bq_s.breakpoints())
        return tuple(sorted(punti))
