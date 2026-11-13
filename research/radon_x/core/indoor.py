from __future__ import annotations

import math

from radon_x.core.decay import LAMBDA_RN222
from radon_x.core.diffusion import D0_RN222_AIR_M2_S, apparent_diffusivity_rogers_nielson
from radon_x.core.units import ach_h_to_s

D0_AIR_M2_S: float = D0_RN222_AIR_M2_S

SPESSORE_SOLETTA_M: float = 0.20


def soil_transfer_velocity(porosity: float, saturation: float,
                           d0_m2_s: float = D0_AIR_M2_S) -> float:
    if not 0.0 < porosity <= 1.0:
        raise ValueError(f"porosita' fuori dominio: {porosity!r}")
    d_a = apparent_diffusivity_rogers_nielson(porosity, saturation, d0_m2_s)
    return porosity * math.sqrt(LAMBDA_RN222 * d_a)


def deep_pore_concentration(free_surface_flux_bq_m2_s: float,
                            transfer_velocity_m_s: float) -> float:
    if free_surface_flux_bq_m2_s < 0.0:
        raise ValueError("il flusso deve essere >= 0")
    if transfer_velocity_m_s <= 0.0:
        raise ValueError("la velocita' di trasferimento deve essere > 0")
    return free_surface_flux_bq_m2_s / transfer_velocity_m_s


def indoor_coupled_bq_m3(
    free_surface_flux_bq_m2_s: float,
    porosity: float,
    saturation: float,
    footprint_m2: float,
    volume_m3: float,
    ach_h: float,
    slab_conductance_m3_s: float,
    deposition_h: float = 0.0,
) -> float:
    if volume_m3 <= 0.0 or footprint_m2 <= 0.0:
        raise ValueError("volume e impronta devono essere > 0")
    if ach_h < 0.0 or deposition_h < 0.0:
        raise ValueError("i ratei di rimozione devono essere >= 0")
    if slab_conductance_m3_s < 0.0 or math.isnan(slab_conductance_m3_s):
        raise ValueError(f"conduttanza non fisica: {slab_conductance_m3_s!r}")
    if slab_conductance_m3_s == 0.0:
        return 0.0

    v = soil_transfer_velocity(porosity, saturation)
    c_inf = deep_pore_concentration(free_surface_flux_bq_m2_s, v)
    beta = v * footprint_m2
    lam_eff = LAMBDA_RN222 + ach_h_to_s(ach_h + deposition_h)
    resistenza = 0.0 if math.isinf(slab_conductance_m3_s) else beta / slab_conductance_m3_s
    return beta * c_inf / (beta + volume_m3 * lam_eff * (resistenza + 1.0))


def crack_conductance_m3_s(area_m2: float,
                           thickness_m: float = SPESSORE_SOLETTA_M,
                           d0_m2_s: float = D0_AIR_M2_S) -> float:
    if area_m2 <= 0.0 or thickness_m <= 0.0:
        raise ValueError("area e spessore devono essere > 0")
    return d0_m2_s * area_m2 / thickness_m


def footprint_convention_bq_m3(free_surface_flux_bq_m2_s: float, footprint_m2: float,
                               volume_m3: float, ach_h: float) -> float:
    lam_eff = LAMBDA_RN222 + ach_h_to_s(ach_h)
    return free_surface_flux_bq_m2_s * footprint_m2 / (volume_m3 * lam_eff)


def crack_convention_bq_m3(free_surface_flux_bq_m2_s: float, crack_m2: float,
                           volume_m3: float, ach_h: float) -> float:
    lam_eff = LAMBDA_RN222 + ach_h_to_s(ach_h)
    return free_surface_flux_bq_m2_s * crack_m2 / (volume_m3 * lam_eff)
