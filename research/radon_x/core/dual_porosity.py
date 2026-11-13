from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from radon_x.core.boundaries import Dirichlet, ZeroFlux
from radon_x.core.decay import LAMBDA_RN222
from radon_x.numerics.tridiagonal import TridiagonalBands, assemble_block


@dataclass(frozen=True)
class DualPorositySolution:

    matrix_bq_m3: np.ndarray
    fracture_bq_m3: np.ndarray
    surface_flux_bq_m2_s: float
    matrix_flux_bq_m2_s: float
    fracture_flux_bq_m2_s: float
    exchange_rate_s: float


def exchange_rate_s(
    specific_area_m: float,
    matrix_diffusivity_m2_s: float,
    fracture_diffusivity_m2_s: float,
    block_half_thickness_m: float,
) -> float:
    for nome, valore in (
        ("specific_area_m", specific_area_m),
        ("matrix_diffusivity_m2_s", matrix_diffusivity_m2_s),
        ("fracture_diffusivity_m2_s", fracture_diffusivity_m2_s),
        ("block_half_thickness_m", block_half_thickness_m),
    ):
        if not math.isfinite(valore):
            raise ValueError(f"{nome} deve essere finito, ricevuto {valore!r}")
    if specific_area_m < 0.0:
        raise ValueError(f"specific_area_m deve essere >= 0, ricevuto {specific_area_m!r}")
    if block_half_thickness_m <= 0.0:
        raise ValueError(
            f"block_half_thickness_m deve essere > 0, ricevuto {block_half_thickness_m!r}"
        )
    if matrix_diffusivity_m2_s <= 0.0 or fracture_diffusivity_m2_s <= 0.0:
        raise ValueError("le due diffusivita' devono essere > 0")
    d_rif = 0.5 * (matrix_diffusivity_m2_s + fracture_diffusivity_m2_s)
    return specific_area_m * d_rif / block_half_thickness_m


_SUPERFICIE = Dirichlet(0.0)
_BASE = ZeroFlux()


def _bande_di_un_continuo(n: int, d_bulk: float, beta: float, dz: float,
                          lam: float, alpha: float) -> TridiagonalBands:
    return assemble_block(
        d_bulk_m2_s=np.full(n, float(d_bulk)),
        beta=np.full(n, float(beta)),
        dz_m=dz,
        lambda_s=lam,
        surface=_SUPERFICIE,
        base=_BASE,
        removal_s=alpha,
    )


def _blocco_denso(bande: TridiagonalBands) -> np.ndarray:
    return (
        np.diag(bande.diag)
        + np.diag(bande.lower[1:], -1)
        + np.diag(bande.upper[:-1], 1)
    )


def solve_steady_dual(
    cell_count: int,
    thickness_m: float,
    matrix_bulk_diffusivity_m2_s: float,
    fracture_bulk_diffusivity_m2_s: float,
    matrix_partition_porosity: float,
    fracture_partition_porosity: float,
    source_bulk_bq_m3_s: float,
    exchange_s: float,
    lambda_s: float = LAMBDA_RN222,
) -> DualPorositySolution:
    if cell_count < 3:
        raise ValueError(f"cell_count deve essere >= 3, ricevuto {cell_count!r}")
    if thickness_m <= 0.0:
        raise ValueError(f"thickness_m deve essere > 0, ricevuto {thickness_m!r}")
    if exchange_s < 0.0:
        raise ValueError(f"exchange_s deve essere >= 0, ricevuto {exchange_s!r}")
    for nome, valore in (
        ("matrix_partition_porosity", matrix_partition_porosity),
        ("fracture_partition_porosity", fracture_partition_porosity),
    ):
        if not 0.0 < valore <= 1.0:
            raise ValueError(f"{nome} fuori dominio: {valore!r}")
    for nome, valore in (
        ("matrix_bulk_diffusivity_m2_s", matrix_bulk_diffusivity_m2_s),
        ("fracture_bulk_diffusivity_m2_s", fracture_bulk_diffusivity_m2_s),
    ):
        if not math.isfinite(valore) or valore <= 0.0:
            raise ValueError(f"{nome} deve essere > 0, ricevuto {valore!r}")

    n = cell_count
    dz = thickness_m / n

    bande_m = _bande_di_un_continuo(
        n, matrix_bulk_diffusivity_m2_s, matrix_partition_porosity, dz, lambda_s, exchange_s
    )
    bande_f = _bande_di_un_continuo(
        n, fracture_bulk_diffusivity_m2_s, fracture_partition_porosity, dz, lambda_s, exchange_s
    )
    sistema = np.zeros((2 * n, 2 * n))
    sistema[:n, :n] = _blocco_denso(bande_m)
    sistema[n:, n:] = _blocco_denso(bande_f)
    sistema[:n, n:] = -exchange_s * np.eye(n)
    sistema[n:, :n] = -exchange_s * np.eye(n)

    noto = np.zeros(2 * n)
    noto[:n] = bande_m.load_vector(np.full(n, float(source_bulk_bq_m3_s)))
    noto[n:] = bande_f.load_vector(np.zeros(n))

    soluzione = np.linalg.solve(sistema, noto)
    cm, cf = soluzione[:n], soluzione[n:]

    j_m = (bande_m.surface_diagonal_m2_s * cm[0] - bande_m.surface_source_bq_m_s) / dz
    j_f = (bande_f.surface_diagonal_m2_s * cf[0] - bande_f.surface_source_bq_m_s) / dz
    return DualPorositySolution(
        matrix_bq_m3=cm,
        fracture_bq_m3=cf,
        surface_flux_bq_m2_s=float(j_m + j_f),
        matrix_flux_bq_m2_s=float(j_m),
        fracture_flux_bq_m2_s=float(j_f),
        exchange_rate_s=float(exchange_s),
    )
