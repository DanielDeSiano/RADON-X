from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np
from scipy.integrate import solve_ivp

from radon_x.dynamics.events import Scenario, StepEvent, compile_segments
from radon_x.dynamics.progeny import DEPOSIZIONE_PROGENIE_S
from radon_x.dynamics.solver import (
    OutputWriter,
    SolverReport,
    _matrice_sistema,
    solve_transient,
)
from radon_x.dynamics.state import InitialState

if TYPE_CHECKING:
    from radon_x.dynamics.solver import Deposizione

__all__ = [
    "INTEGRATORI",
    "BancoRiga",
    "Integratore",
    "IntegratoreBDF",
    "IntegratoreEsatto",
    "IntegratoreEulero",
    "IntegratoreRK4",
    "IntegratoreRK45",
    "banco",
    "errore_massimo",
    "ordine_osservato",
    "tabella_del_banco",
]

TOLLERANZA_CONFINE_S = 1e-9


@runtime_checkable
class Integratore(Protocol):

    @property
    def nome(self) -> str:
        pass

    def solve(
        self,
        scenario: Scenario,
        initial: InitialState,
        t_end_s: float,
        output_step_s: float = 3600.0,
        events: tuple[StepEvent, ...] = (),
        t_start_s: float = 0.0,
        deposizione_s: Deposizione = DEPOSIZIONE_PROGENIE_S,
    ) -> tuple[OutputWriter, SolverReport]:
        pass


def _griglia_del_segmento(
    t_start_s: float, t_end_s: float, output_step_s: float
) -> list[float]:
    passo = min(output_step_s, t_end_s - t_start_s)
    tempi: list[float] = []
    t = t_start_s
    while t + passo <= t_end_s + TOLLERANZA_CONFINE_S:
        t += passo
        tempi.append(t)
    if t_end_s - t > TOLLERANZA_CONFINE_S:
        tempi.append(t_end_s)
    return tempi


def _sotto_passi(intervallo_s: float, passo_s: float) -> int:
    if not math.isfinite(passo_s) or passo_s <= 0.0:
        raise ValueError(f"passo interno non fisico: {passo_s!r}")
    return max(1, math.ceil(intervallo_s / passo_s * (1.0 - 1e-12)))


def _stadio_eulero(m: np.ndarray, q: np.ndarray, x: np.ndarray, h: float) -> np.ndarray:
    return x + h * (m @ x + q)


def _stadio_rk4(m: np.ndarray, q: np.ndarray, x: np.ndarray, h: float) -> np.ndarray:
    k1 = m @ x + q
    k2 = m @ (x + 0.5 * h * k1) + q
    k3 = m @ (x + 0.5 * h * k2) + q
    k4 = m @ (x + h * k3) + q
    return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _marcia_a_passo_fisso(
    stadio: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
    valutazioni_per_passo: int,
    passo_s: float,
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, float, Sequence[float]],
    tuple[list[np.ndarray], int],
]:

    def avanza(
        m: np.ndarray,
        q: np.ndarray,
        x0: np.ndarray,
        t_start_s: float,
        griglia: Sequence[float],
    ) -> tuple[list[np.ndarray], int]:
        stati: list[np.ndarray] = []
        valutazioni = 0
        x = x0
        t_precedente = t_start_s
        with np.errstate(over="ignore", invalid="ignore"):
            for t in griglia:
                n = _sotto_passi(t - t_precedente, passo_s)
                h = (t - t_precedente) / n
                for _ in range(n):
                    x = stadio(m, q, x, h)
                    valutazioni += valutazioni_per_passo
                stati.append(x)
                t_precedente = t
        return stati, valutazioni

    return avanza


def _corsa_a_segmenti(
    metodo: str,
    avanza: Callable[
        [np.ndarray, np.ndarray, np.ndarray, float, Sequence[float]],
        tuple[list[np.ndarray], int],
    ],
    scenario: Scenario,
    initial: InitialState,
    t_end_s: float,
    output_step_s: float,
    events: tuple[StepEvent, ...],
    t_start_s: float,
    deposizione_s: Deposizione,
) -> tuple[OutputWriter, SolverReport]:
    avvio = time.perf_counter()
    if output_step_s <= 0.0 or not math.isfinite(output_step_s):
        raise ValueError(f"passo di uscita non fisico: {output_step_s!r}")
    layout = initial.layout
    segmenti = compile_segments(scenario, events, t_start_s, t_end_s)

    tempi = [t_start_s]
    stati = [np.asarray(initial.values, dtype=float)]
    valutazioni = 0

    for segmento in segmenti:
        m, q = _matrice_sistema(
            segmento.scenario, layout, segmento.t_start_s, deposizione_s
        )
        griglia = _griglia_del_segmento(
            segmento.t_start_s, segmento.t_end_s, output_step_s
        )
        nuovi, conto = avanza(m, q, stati[-1], segmento.t_start_s, griglia)
        if len(nuovi) != len(griglia):
            raise RuntimeError(
                f"{metodo}: {len(nuovi)} stati per {len(griglia)} istanti di uscita"
            )
        valutazioni += conto
        tempi.extend(griglia)
        stati.extend(nuovi)

    valori = np.vstack(stati)
    passi = np.diff(np.asarray(tempi, dtype=float))
    avvisi: list[str] = []

    finita = bool(np.all(np.isfinite(valori)))
    if not finita:
        avvisi.append(
            "traiettoria non finita: il passo interno supera il limite di "
            "stabilita' del metodo"
        )
    passo_minimo = float(np.min(passi)) if passi.size else 0.0
    if passi.size and passo_minimo < output_step_s - TOLLERANZA_CONFINE_S:
        avvisi.append(
            f"griglia di uscita non uniforme: passo minimo {passo_minimo:.6g} s "
            f"contro il passo richiesto {output_step_s:.6g} s"
        )
    if finita:
        minimo = float(np.min(valori))
        if minimo < 0.0:
            avvisi.append(f"minimo della traiettoria negativo: {minimo:.3e} Bq m-3")

    uscita = OutputWriter(layout=layout, times_s=np.asarray(tempi), values=valori)
    rapporto = SolverReport(
        method=metodo,
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        n_segments=len(segmenti),
        n_outputs=len(tempi),
        events_applied=segmenti[-1].events_applied if segmenti else (),
        composition_residual=float("nan"),
        state_size=layout.size,
        step_min_s=passo_minimo,
        step_mean_s=float(np.mean(passi)) if passi.size else 0.0,
        step_max_s=float(np.max(passi)) if passi.size else 0.0,
        success=finita,
        warnings=tuple(avvisi),
        elapsed_s=time.perf_counter() - avvio,
        n_rhs_evaluations=valutazioni,
    )
    return uscita, rapporto


@dataclass(frozen=True)
class IntegratoreEsatto:

    @property
    def nome(self) -> str:
        return "esatto (expm)"

    def solve(
        self,
        scenario: Scenario,
        initial: InitialState,
        t_end_s: float,
        output_step_s: float = 3600.0,
        events: tuple[StepEvent, ...] = (),
        t_start_s: float = 0.0,
        deposizione_s: Deposizione = DEPOSIZIONE_PROGENIE_S,
    ) -> tuple[OutputWriter, SolverReport]:
        return solve_transient(
            scenario,
            initial,
            t_end_s,
            output_step_s,
            events,
            t_start_s,
            deposizione_s,
        )


@dataclass(frozen=True)
class IntegratoreEulero:

    passo_s: float = 300.0

    @property
    def nome(self) -> str:
        return f"Eulero h={self.passo_s:g} s"

    def solve(
        self,
        scenario: Scenario,
        initial: InitialState,
        t_end_s: float,
        output_step_s: float = 3600.0,
        events: tuple[StepEvent, ...] = (),
        t_start_s: float = 0.0,
        deposizione_s: Deposizione = DEPOSIZIONE_PROGENIE_S,
    ) -> tuple[OutputWriter, SolverReport]:
        return _corsa_a_segmenti(
            f"Eulero esplicito a passo fisso {self.passo_s:g} s, ordine 1",
            _marcia_a_passo_fisso(_stadio_eulero, 1, self.passo_s),
            scenario,
            initial,
            t_end_s,
            output_step_s,
            events,
            t_start_s,
            deposizione_s,
        )


@dataclass(frozen=True)
class IntegratoreRK4:

    passo_s: float = 300.0

    @property
    def nome(self) -> str:
        return f"RK4 h={self.passo_s:g} s"

    def solve(
        self,
        scenario: Scenario,
        initial: InitialState,
        t_end_s: float,
        output_step_s: float = 3600.0,
        events: tuple[StepEvent, ...] = (),
        t_start_s: float = 0.0,
        deposizione_s: Deposizione = DEPOSIZIONE_PROGENIE_S,
    ) -> tuple[OutputWriter, SolverReport]:
        return _corsa_a_segmenti(
            f"Runge-Kutta 4 a passo fisso {self.passo_s:g} s, ordine 4",
            _marcia_a_passo_fisso(_stadio_rk4, 4, self.passo_s),
            scenario,
            initial,
            t_end_s,
            output_step_s,
            events,
            t_start_s,
            deposizione_s,
        )


@dataclass(frozen=True)
class _IntegratoreSolveIvp:

    rtol: float = 1e-8
    atol: float = 1e-10

    _metodo: ClassVar[str] = "RK45"
    _etichetta: ClassVar[str] = "adattivo"
    _con_jacobiana: ClassVar[bool] = False

    @property
    def nome(self) -> str:
        return f"{self._metodo} rtol={self.rtol:g}"

    def solve(
        self,
        scenario: Scenario,
        initial: InitialState,
        t_end_s: float,
        output_step_s: float = 3600.0,
        events: tuple[StepEvent, ...] = (),
        t_start_s: float = 0.0,
        deposizione_s: Deposizione = DEPOSIZIONE_PROGENIE_S,
    ) -> tuple[OutputWriter, SolverReport]:
        return _corsa_a_segmenti(
            (
                f"solve_ivp {self._metodo} {self._etichetta}, "
                f"rtol={self.rtol:g}, atol={self.atol:g}"
            ),
            self._avanzamento(),
            scenario,
            initial,
            t_end_s,
            output_step_s,
            events,
            t_start_s,
            deposizione_s,
        )

    def _avanzamento(
        self,
    ) -> Callable[
        [np.ndarray, np.ndarray, np.ndarray, float, Sequence[float]],
        tuple[list[np.ndarray], int],
    ]:
        metodo = self._metodo
        con_jacobiana = self._con_jacobiana
        rtol = self.rtol
        atol = self.atol

        def avanza(
            m: np.ndarray,
            q: np.ndarray,
            x0: np.ndarray,
            t_start_s: float,
            griglia: Sequence[float],
        ) -> tuple[list[np.ndarray], int]:
            if not griglia:
                return [], 0

            def secondo_membro(_t: float, y: np.ndarray) -> np.ndarray:
                return m @ y + q

            extra = {"jac": m} if con_jacobiana else {}
            soluzione = solve_ivp(
                secondo_membro,
                (t_start_s, float(griglia[-1])),
                np.asarray(x0, dtype=float),
                method=metodo,
                t_eval=np.asarray(griglia, dtype=float),
                rtol=rtol,
                atol=atol,
                **extra,
            )
            if not soluzione.success:
                raise RuntimeError(f"solve_ivp {metodo}: {soluzione.message}")
            stati = [np.asarray(soluzione.y[:, j]) for j in range(soluzione.y.shape[1])]
            return stati, int(soluzione.nfev)

        return avanza


@dataclass(frozen=True)
class IntegratoreRK45(_IntegratoreSolveIvp):

    _metodo: ClassVar[str] = "RK45"
    _etichetta: ClassVar[str] = "adattivo esplicito"
    _con_jacobiana: ClassVar[bool] = False


@dataclass(frozen=True)
class IntegratoreBDF(_IntegratoreSolveIvp):

    _metodo: ClassVar[str] = "BDF"
    _etichetta: ClassVar[str] = "implicito per sistemi rigidi"
    _con_jacobiana: ClassVar[bool] = True


INTEGRATORI: tuple[Integratore, ...] = (
    IntegratoreEsatto(),
    IntegratoreEulero(passo_s=300.0),
    IntegratoreRK4(passo_s=300.0),
    IntegratoreRK45(),
    IntegratoreBDF(),
)


def errore_massimo(provata: OutputWriter, oracolo: OutputWriter) -> float:
    if provata.times_s.shape != oracolo.times_s.shape:
        raise ValueError(
            f"griglie diverse: {provata.times_s.shape} contro {oracolo.times_s.shape}"
        )
    if not np.array_equal(provata.times_s, oracolo.times_s):
        raise ValueError("le due corse non condividono gli istanti di uscita")
    scala = float(np.max(np.abs(oracolo.values)))
    if scala <= 0.0:
        raise ValueError("oracolo identicamente nullo: nessuna scala di riferimento")
    differenza = np.abs(provata.values - oracolo.values)
    if not np.all(np.isfinite(differenza)):
        return float("inf")
    return float(np.max(differenza)) / scala


def ordine_osservato(
    costruttore: Callable[[float], Integratore],
    scenario: Scenario,
    initial: InitialState,
    t_end_s: float,
    passi_s: Sequence[float],
    output_step_s: float = 3600.0,
    events: tuple[StepEvent, ...] = (),
    deposizione_s: Deposizione = DEPOSIZIONE_PROGENIE_S,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(passi_s) < 2:
        raise ValueError("servono almeno due passi per leggere un ordine")
    coppie = zip(passi_s, passi_s[1:])
    for grande, piccolo in coppie:
        if not math.isclose(piccolo * 2.0, grande, rel_tol=1e-12):
            raise ValueError(f"passi non dimezzati: {grande} e {piccolo}")

    uscita_oracolo, _ = IntegratoreEsatto().solve(
        scenario,
        initial,
        t_end_s,
        output_step_s,
        events,
        0.0,
        deposizione_s,
    )
    errori = []
    for passo in passi_s:
        uscita, _rapporto = costruttore(passo).solve(
            scenario,
            initial,
            t_end_s,
            output_step_s,
            events,
            0.0,
            deposizione_s,
        )
        errori.append(errore_massimo(uscita, uscita_oracolo))
    consecutivi = zip(errori, errori[1:])
    ordini = tuple(math.log2(grande / piccolo) for grande, piccolo in consecutivi)
    return tuple(errori), ordini


@dataclass(frozen=True)
class BancoRiga:

    nome: str
    errore_massimo_relativo: float
    n_valutazioni: int
    secondi: float
    riuscito: bool


def banco(
    scenario: Scenario,
    initial: InitialState,
    t_end_s: float,
    integratori: Sequence[Integratore] = INTEGRATORI,
    output_step_s: float = 3600.0,
    events: tuple[StepEvent, ...] = (),
    deposizione_s: Deposizione = DEPOSIZIONE_PROGENIE_S,
) -> tuple[BancoRiga, ...]:
    uscita_oracolo, _ = IntegratoreEsatto().solve(
        scenario, initial, t_end_s, output_step_s, events, 0.0, deposizione_s
    )
    righe = []
    for integratore in integratori:
        uscita, rapporto = integratore.solve(
            scenario, initial, t_end_s, output_step_s, events, 0.0, deposizione_s
        )
        righe.append(
            BancoRiga(
                nome=integratore.nome,
                errore_massimo_relativo=errore_massimo(uscita, uscita_oracolo),
                n_valutazioni=rapporto.n_rhs_evaluations,
                secondi=rapporto.elapsed_s,
                riuscito=rapporto.success,
            )
        )
    return tuple(righe)


def tabella_del_banco(righe: Sequence[BancoRiga]) -> str:
    intestazione = f"{'membro':<26}{'errore max':>13}{'n f':>9}{'tempo [s]':>12}"
    linee = [intestazione, "-" * len(intestazione)]
    for riga in righe:
        linee.append(
            f"{riga.nome:<26}{riga.errore_massimo_relativo:>13.3e}"
            f"{riga.n_valutazioni:>9d}{riga.secondi:>12.6f}"
        )
    return "\n".join(linee)
