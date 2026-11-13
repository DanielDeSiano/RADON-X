from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

from radon_x.core.units import SECONDI_PER_ORA
from radon_x.dynamics.forcing import (
    Forcing,
    PiecewiseProfile,
    SourceTerm,
    TimeSeriesProfile,
    door_ach_profile,
)
from radon_x.dynamics.zones import ZoneNetwork


@dataclass(frozen=True)
class Scenario:

    network: ZoneNetwork
    forcing: Forcing


@dataclass(frozen=True)
class StepEvent:

    time_s: float
    name: str
    apply: Callable[[Scenario], Scenario]

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s):
            raise ValueError(f"tempo dell'evento non finito: {self.time_s!r}")
        if not self.name:
            raise ValueError("un evento senza nome non e' raccontabile")

    @classmethod
    def set_flows(
        cls, time_s: float, name: str, edges: dict[tuple[str, str], float]
    ) -> StepEvent:
        if not edges:
            raise ValueError("nessun arco da cambiare")

        def trasforma(s: Scenario) -> Scenario:
            flussi = dict(s.network.flows_m3_s)
            flussi.update(edges)
            return replace(s, network=replace(s.network, flows_m3_s=flussi))

        return cls(time_s=time_s, name=name, apply=trasforma)

    @classmethod
    def set_flow(
        cls, time_s: float, name: str, edge: tuple[str, str], flow_m3_s: float
    ) -> StepEvent:
        return cls.set_flows(time_s, name, {edge: flow_m3_s})

    @classmethod
    def set_air_exchange(
        cls, time_s: float, name: str, zone: str, ach_h: float
    ) -> StepEvent:
        if not math.isfinite(ach_h) or ach_h < 0.0:
            raise ValueError(f"ricambio non fisico: {ach_h!r}")

        def trasforma(s: Scenario) -> Scenario:
            from radon_x.dynamics.zones import ESTERNO

            portata = ach_h * s.network.volumes_m3[zone] / SECONDI_PER_ORA
            flussi = dict(s.network.flows_m3_s)
            flussi[(ESTERNO, zone)] = portata
            flussi[(zone, ESTERNO)] = portata
            return replace(s, network=replace(s.network, flows_m3_s=flussi))

        return cls(time_s=time_s, name=name, apply=trasforma)

    @classmethod
    def scale_source(
        cls, time_s: float, name: str, zone: str, factor: float
    ) -> StepEvent:
        if not math.isfinite(factor) or factor < 0.0:
            raise ValueError(f"fattore non fisico: {factor!r}")

        def trasforma(s: Scenario) -> Scenario:
            sorgenti = []
            trovata = False
            for sorgente in s.forcing.sources:
                if sorgente.zone != zone:
                    sorgenti.append(sorgente)
                    continue
                trovata = True
                profilo = sorgente.rate_bq_s
                sorgenti.append(
                    SourceTerm(
                        zone=zone,
                        rate_bq_s=TimeSeriesProfile(
                            times_s=profilo.times_s,
                            values=tuple(v * factor for v in profilo.values),
                        ),
                        origin=f"{sorgente.origin}; riscalata da {name!r}",
                    )
                )
            if not trovata:
                raise ValueError(f"nessuna sorgente in {zone!r} da riscalare")
            return replace(s, forcing=replace(s.forcing, sources=tuple(sorgenti)))

        return cls(time_s=time_s, name=name, apply=trasforma)


def door_schedule(
    zone: str,
    t_end_s: float,
    t_start_s: float = 0.0,
    profile: PiecewiseProfile | None = None,
) -> tuple[StepEvent, ...]:
    orario = door_ach_profile() if profile is None else profile
    rotture = orario.breakpoints(t_start_s, t_end_s)
    eventi = []
    for t in rotture[1:]:
        ach = orario.value(t)
        aperte = ach > orario.baseline
        nome = "porte aperte" if aperte else "porte chiuse"
        eventi.append(
            StepEvent.set_air_exchange(t, f"{nome} a t={t:.0f} s", zone, ach)
        )
    return tuple(eventi)


@dataclass(frozen=True)
class Segment:

    t_start_s: float
    t_end_s: float
    scenario: Scenario
    events_applied: tuple[str, ...]


def compile_segments(
    scenario: Scenario,
    events: tuple[StepEvent, ...],
    t_start_s: float,
    t_end_s: float,
) -> list[Segment]:
    if not t_end_s > t_start_s:
        raise ValueError(f"intervallo vuoto: [{t_start_s}, {t_end_s}]")
    ordinati = sorted(
        (e for e in events if t_start_s < e.time_s < t_end_s),
        key=lambda e: e.time_s,
    )
    rotture = sorted(
        {t_start_s, t_end_s}
        | {e.time_s for e in ordinati}
        | {
            t
            for t in scenario.forcing.breakpoints()
            if t_start_s < t < t_end_s
        }
    )
    segmenti: list[Segment] = []
    corrente = scenario
    applicati: tuple[str, ...] = ()
    coppie = zip(rotture, rotture[1:])
    for inizio, fine in coppie:
        nuovi = [e for e in ordinati if e.time_s == inizio]
        for evento in nuovi:
            corrente = evento.apply(corrente)
            applicati = (*applicati, evento.name)
        segmenti.append(
            Segment(
                t_start_s=inizio,
                t_end_s=fine,
                scenario=corrente,
                events_applied=applicati,
            )
        )
    return segmenti
