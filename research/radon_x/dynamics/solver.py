from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

from radon_x.core.decay import LAMBDA_RN222
from radon_x.dynamics.events import Scenario, StepEvent, compile_segments
from radon_x.dynamics.progeny import (
    DEPOSIZIONE_PROGENIE_S,
    LAMBDA_PROGENIE,
)
from radon_x.dynamics.state import SPECIE_PROGENIE, InitialState, StateLayout
from radon_x.dynamics.zones import exchange_matrix_from_flows

LIVELLI_RIFERIMENTO_BQ_M3: tuple[tuple[float, str], ...] = (
    (300.0, "direttiva 2013/59/Euratom, art. 74: livello di riferimento abitazioni"),
    (100.0, "OMS 2009, Handbook on Indoor Radon: livello raccomandato"),
)

SOGLIA_COMPOSIZIONE = 1e-9

if TYPE_CHECKING:
    Deposizione = float | dict[str, float] | dict[str, dict[str, float]]


def _ratei_di_deposizione(
    deposizione_s: Deposizione, zone: tuple[str, ...]
) -> np.ndarray:
    n_figli = len(SPECIE_PROGENIE)
    if isinstance(deposizione_s, Mapping):
        mancanti = [z for z in zone if z not in deposizione_s]
        if mancanti:
            raise KeyError(f"deposizione non dichiarata per {mancanti}")
        estranee = [z for z in deposizione_s if z not in zone]
        if estranee:
            raise KeyError(f"deposizione su zone che il layout non ha: {estranee}")
        righe = []
        for zona in zone:
            voce = deposizione_s[zona]
            if isinstance(voce, Mapping):
                assenti = [s for s in SPECIE_PROGENIE if s not in voce]
                if assenti:
                    raise KeyError(f"deposizione di {zona!r} non dichiarata per {assenti}")
                intruse = [s for s in voce if s not in SPECIE_PROGENIE]
                if intruse:
                    raise KeyError(
                        f"deposizione di {zona!r} su specie fuori dalla catena: {intruse}"
                    )
                righe.append([float(voce[s]) for s in SPECIE_PROGENIE])
            else:
                righe.append([float(voce)] * n_figli)
        valori = np.asarray(righe, dtype=float)
    else:
        valori = np.full((len(zone), n_figli), float(deposizione_s))
    for i, zona in enumerate(zone):
        for k, specie in enumerate(SPECIE_PROGENIE):
            rateo = float(valori[i, k])
            if not math.isfinite(rateo) or rateo < 0.0:
                raise ValueError(
                    f"deposizione non fisica in {zona!r} per {specie}: {rateo!r}"
                )
    return valori


def _matrice_sistema(
    scenario: Scenario, layout: StateLayout, t_s: float,
    deposizione_s: Deposizione,
) -> tuple[np.ndarray, np.ndarray]:
    if layout.zones != scenario.network.zones:
        raise ValueError(
            f"layout {layout.zones} e rete {scenario.network.zones} non combaciano"
        )
    n_zone = len(layout.zones)
    specie = layout.species
    n_specie = len(specie)
    dimensione = layout.size

    a_zone, b_esterno = exchange_matrix_from_flows(scenario.network)
    m = np.zeros((dimensione, dimensione))
    q = np.zeros(dimensione)

    lambda_specie = (LAMBDA_RN222, *LAMBDA_PROGENIE)[:n_specie]
    deposizione = _ratei_di_deposizione(deposizione_s, layout.zones)
    c_out = scenario.forcing.outdoor_bq_m3.value(t_s)

    for k, (nome_specie, lam) in enumerate(zip(specie, lambda_specie)):
        for i in range(n_zone):
            for j in range(n_zone):
                m[i * n_specie + k, j * n_specie + k] += a_zone[i, j]
        for i, zona in enumerate(layout.zones):
            riga = i * n_specie + k
            m[riga, riga] -= lam
            if k == 0:
                volume = scenario.network.volumes_m3[zona]
                q[riga] += scenario.forcing.source_rate(zona, t_s) / volume
                q[riga] += b_esterno[i] * c_out
            else:
                m[riga, riga] -= deposizione[i, k - 1]
                m[riga, riga - 1] += lam
    return m, q


@dataclass(frozen=True)
class SolverReport:

    method: str
    t_start_s: float
    t_end_s: float
    n_segments: int
    n_outputs: int
    events_applied: tuple[str, ...]
    composition_residual: float
    state_size: int
    step_min_s: float = 0.0
    step_mean_s: float = 0.0
    step_max_s: float = 0.0
    success: bool = True
    warnings: tuple[str, ...] = field(default_factory=tuple)
    elapsed_s: float = 0.0
    n_rhs_evaluations: int = 0


@dataclass(frozen=True)
class OutputWriter:

    layout: StateLayout
    times_s: np.ndarray
    values: np.ndarray

    def series(self, zone: str, species: str = "Rn222") -> np.ndarray:
        return self.values[:, self.layout.index(zone, species)]

    def mean(self, zone: str, species: str = "Rn222") -> float:
        serie = self.series(zone, species)
        return float(np.trapz(serie, self.times_s) / (self.times_s[-1] - self.times_s[0]))

    def maximum(self, zone: str, species: str = "Rn222") -> float:
        return float(np.max(self.series(zone, species)))

    def time_above_s(self, level_bq_m3: float, zone: str) -> float:
        if not math.isfinite(level_bq_m3) or level_bq_m3 < 0.0:
            raise ValueError(f"livello non fisico: {level_bq_m3!r}")
        serie = self.series(zone)
        sopra = serie > level_bq_m3
        dt = np.diff(self.times_s)
        frazioni = (sopra[:-1].astype(float) + sopra[1:].astype(float)) / 2.0
        return float(np.sum(frazioni * dt))

    def exceedance_report(self, zone: str) -> tuple[tuple[float, str, float], ...]:
        return tuple(
            (livello, fonte, self.time_above_s(livello, zone) / 3600.0)
            for livello, fonte in LIVELLI_RIFERIMENTO_BQ_M3
        )

    def to_dict(self) -> dict:
        return {
            "times_s": self.times_s.tolist(),
            "series": {
                f"{zona}:{specie}": self.series(zona, specie).tolist()
                for zona in self.layout.zones
                for specie in self.layout.species
            },
        }


def steady_state(
    scenario: Scenario,
    layout: StateLayout,
    t_s: float = 0.0,
    deposizione_s: Deposizione = DEPOSIZIONE_PROGENIE_S,
) -> np.ndarray:
    m, q = _matrice_sistema(scenario, layout, t_s, deposizione_s)
    x = np.linalg.solve(m, -q)
    residuo = float(np.max(np.abs(m @ x + q)))
    scala = max(float(np.max(np.abs(q))), 1e-30)
    if residuo / scala > 1e-9:
        raise RuntimeError(f"stazionario non risolto: residuo relativo {residuo / scala:.2e}")
    return x


def solve_transient(
    scenario: Scenario,
    initial: InitialState,
    t_end_s: float,
    output_step_s: float = 3600.0,
    events: tuple[StepEvent, ...] = (),
    t_start_s: float = 0.0,
    deposizione_s: Deposizione = DEPOSIZIONE_PROGENIE_S,
) -> tuple[OutputWriter, SolverReport]:
    avvio = time.perf_counter()
    if output_step_s <= 0.0 or not math.isfinite(output_step_s):
        raise ValueError(f"passo di uscita non fisico: {output_step_s!r}")
    layout = initial.layout
    segmenti = compile_segments(scenario, events, t_start_s, t_end_s)

    tempi = [t_start_s]
    stati = [np.asarray(initial.values, dtype=float)]
    residuo_composizione = 0.0

    for segmento in segmenti:
        m, q = _matrice_sistema(
            segmento.scenario, layout, segmento.t_start_s, deposizione_s
        )
        n = layout.size
        aumentata = np.zeros((n + 1, n + 1))
        aumentata[:n, :n] = m
        aumentata[:n, n] = q

        x0 = stati[-1]
        durata = segmento.t_end_s - segmento.t_start_s

        passo = min(output_step_s, durata)
        p_passo = expm(aumentata * passo)
        t_locale = segmento.t_start_s
        x = x0
        while t_locale + passo <= segmento.t_end_s + 1e-9:
            y = p_passo @ np.append(x, 1.0)
            x = y[:n]
            t_locale += passo
            tempi.append(t_locale)
            stati.append(x)
        resto = segmento.t_end_s - t_locale
        if resto > 1e-9:
            y = expm(aumentata * resto) @ np.append(x, 1.0)
            x = y[:n]
            t_locale = segmento.t_end_s
            tempi.append(t_locale)
            stati.append(x)

        y_intero = expm(aumentata * durata) @ np.append(x0, 1.0)
        scala = max(float(np.max(np.abs(y_intero[:n]))), 1e-30)
        residuo_composizione = max(
            residuo_composizione,
            float(np.max(np.abs(y_intero[:n] - x))) / scala,
        )

    valori = np.vstack(stati)
    if not np.all(np.isfinite(valori)):
        raise RuntimeError("la propagazione ha prodotto valori non finiti")

    passi = np.diff(np.asarray(tempi, dtype=float))
    avvisi: list[str] = []
    riuscita = math.isfinite(residuo_composizione)
    if not riuscita or residuo_composizione > SOGLIA_COMPOSIZIONE:
        riuscita = False
        avvisi.append(
            f"residuo di composizione {residuo_composizione:.3e} sopra "
            f"{SOGLIA_COMPOSIZIONE:.0e}"
        )
    passo_minimo = float(np.min(passi)) if passi.size else 0.0
    if passi.size and passo_minimo < output_step_s - 1e-9:
        avvisi.append(
            f"griglia di uscita non uniforme: passo minimo {passo_minimo:.6g} s "
            f"contro il passo richiesto {output_step_s:.6g} s"
        )
    minimo = float(np.min(valori))
    if minimo < 0.0:
        avvisi.append(f"minimo della traiettoria negativo: {minimo:.3e} Bq m-3")

    uscita = OutputWriter(layout=layout, times_s=np.asarray(tempi), values=valori)
    rapporto = SolverReport(
        method="esponenziale di matrice sul sistema aumentato, esatto a tratti",
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        n_segments=len(segmenti),
        n_outputs=len(tempi),
        events_applied=segmenti[-1].events_applied if segmenti else (),
        composition_residual=residuo_composizione,
        state_size=layout.size,
        step_min_s=passo_minimo,
        step_mean_s=float(np.mean(passi)) if passi.size else 0.0,
        step_max_s=float(np.max(passi)) if passi.size else 0.0,
        success=riuscita,
        warnings=tuple(avvisi),
        elapsed_s=time.perf_counter() - avvio,
    )
    return uscita, rapporto
