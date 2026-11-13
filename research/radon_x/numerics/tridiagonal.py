from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

from radon_x.errors import NumericalError

if TYPE_CHECKING:
    from radon_x.core.boundaries import BoundaryCondition

__all__ = [
    "TridiagonalBands",
    "assemble_block",
    "harmonic_face_mean",
    "solve_banded_system",
    "tridiagonal_matvec",
]


@dataclass(frozen=True)
class TridiagonalBands:

    lower: NDArray[np.float64]
    diag: NDArray[np.float64]
    upper: NDArray[np.float64]
    d_face_interior_m2_s: NDArray[np.float64]
    surface_diagonal_m2_s: float
    surface_source_bq_m_s: float
    base_diagonal_m2_s: float
    base_source_bq_m_s: float
    dz_m: float
    inv_dz2_m2: float

    def load_vector(
        self, source_bulk_bq_m3_s: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        rhs = np.array(source_bulk_bq_m3_s, dtype=float, copy=True)
        rhs[0] += self.surface_source_bq_m_s * self.inv_dz2_m2
        rhs[-1] += self.base_source_bq_m_s * self.inv_dz2_m2
        return rhs


def harmonic_face_mean(
    d_left: NDArray[np.float64],
    d_right: NDArray[np.float64],
) -> NDArray[np.float64]:
    return 2.0 * d_left * d_right / (d_left + d_right)


def assemble_block(
    d_bulk_m2_s: NDArray[np.float64],
    beta: NDArray[np.float64],
    dz_m: float,
    lambda_s: float,
    surface: BoundaryCondition,
    base: BoundaryCondition,
    removal_s: float = 0.0,
) -> TridiagonalBands:
    from radon_x.core.boundaries import boundary_coefficients

    n = d_bulk_m2_s.size
    inv_dz2 = 1.0 / (dz_m * dz_m)

    d_face_interior = harmonic_face_mean(d_bulk_m2_s[:-1], d_bulk_m2_s[1:])
    conductance_west = np.zeros(n, dtype=float)
    conductance_east = np.zeros(n, dtype=float)
    conductance_west[1:] = d_face_interior
    conductance_east[:-1] = d_face_interior

    surface_coefficients = boundary_coefficients(surface, float(d_bulk_m2_s[0]))
    base_coefficients = boundary_coefficients(base, float(d_bulk_m2_s[-1]))

    diag = (conductance_west + conductance_east) * inv_dz2 + lambda_s * beta
    diag += removal_s
    diag[0] += surface_coefficients.diagonal_m2_s * inv_dz2
    diag[-1] += base_coefficients.diagonal_m2_s * inv_dz2

    lower = -conductance_west * inv_dz2
    upper = -conductance_east * inv_dz2

    return TridiagonalBands(
        lower=lower,
        diag=diag,
        upper=upper,
        d_face_interior_m2_s=d_face_interior,
        surface_diagonal_m2_s=surface_coefficients.diagonal_m2_s,
        surface_source_bq_m_s=surface_coefficients.source_bq_m_s,
        base_diagonal_m2_s=base_coefficients.diagonal_m2_s,
        base_source_bq_m_s=base_coefficients.source_bq_m_s,
        dz_m=float(dz_m),
        inv_dz2_m2=inv_dz2,
    )


def solve_banded_system(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
    name: str = "solution profile",
) -> NDArray[np.float64]:
    n = diag.size
    banded = np.zeros((3, n), dtype=float)
    banded[0, 1:] = upper[:-1]
    banded[1, :] = diag
    banded[2, :-1] = lower[1:]
    solution = np.asarray(solve_banded((1, 1), banded, rhs, check_finite=True), dtype=float)
    if not np.all(np.isfinite(solution)):
        bad = int(np.argmin(np.isfinite(solution)))
        raise NumericalError(
            f"{name} is not finite: entry {bad} is {solution[bad]!r}; "
            "check the magnitudes of the coefficients and of the time step"
        )
    return solution


def tridiagonal_matvec(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    product = diag * vector
    product[1:] += lower[1:] * vector[:-1]
    product[:-1] += upper[:-1] * vector[1:]
    return product
