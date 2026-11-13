from __future__ import annotations

import math

__all__ = [
    "A_LEAK_M2",
    "CD_LEAK",
    "DP_STACK_MAX_PA",
    "DP_STACK_MIN_PA",
    "G_M_S2",
    "H_STACK_M",
    "M_ARIA_KG_MOL",
    "P_ATM_STANDARD_PA",
    "R_GAS_J_MOL_K",
    "TAU_CAP_PA_S",
    "TAU_PIN_MAX_S",
    "TAU_PIN_MIN_S",
    "VOLUME_INVOLUCRO_M3",
    "air_density",
    "indoor_pressure_relaxation_s",
    "indoor_pressure_rhs",
    "indoor_pressure_target",
    "stack_pressure",
]

G_M_S2 = 9.81

M_ARIA_KG_MOL = 0.02897

R_GAS_J_MOL_K = 8.314

P_ATM_STANDARD_PA = 101325.0

A_LEAK_M2 = 0.005

CD_LEAK = 0.62

H_STACK_M = 2.5

TAU_CAP_PA_S = 1800.0

VOLUME_INVOLUCRO_M3 = 170.0

DP_STACK_MIN_PA = -30.0
DP_STACK_MAX_PA = 5.0

TAU_PIN_MIN_S = 1.0
TAU_PIN_MAX_S = 600.0

DELTA_P_RIFERIMENTO_MIN_PA = 0.5


def _temperatura_valida(nome: str, valore: float) -> float:
    if not math.isfinite(valore) or valore <= 0.0:
        raise ValueError(f"{nome} non fisica: {valore!r} K")
    return float(valore)


def _pressione_valida(nome: str, valore: float) -> float:
    if not math.isfinite(valore) or valore <= 0.0:
        raise ValueError(f"{nome} non fisica: {valore!r} Pa")
    return float(valore)


def air_density(p_pa: float, T: float, m_kg_mol: float = M_ARIA_KG_MOL,
                r_j_mol_k: float = R_GAS_J_MOL_K) -> float:
    p_pa = _pressione_valida("pressione", p_pa)
    T = _temperatura_valida("temperatura", T)
    if not math.isfinite(m_kg_mol) or m_kg_mol <= 0.0:
        raise ValueError(f"massa molare non fisica: {m_kg_mol!r}")
    if not math.isfinite(r_j_mol_k) or r_j_mol_k <= 0.0:
        raise ValueError(f"costante dei gas non fisica: {r_j_mol_k!r}")
    return p_pa * m_kg_mol / (r_j_mol_k * T)


def stack_pressure(T_in: float, T_out: float, h: float,
                   p_atm_pa: float = P_ATM_STANDARD_PA,
                   g_m_s2: float = G_M_S2) -> float:
    T_in = _temperatura_valida("temperatura interna", T_in)
    T_out = _temperatura_valida("temperatura esterna", T_out)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"altezza del camino non fisica: {h!r} m")
    if not math.isfinite(g_m_s2) or g_m_s2 <= 0.0:
        raise ValueError(f"gravita' non fisica: {g_m_s2!r}")
    p_atm_pa = _pressione_valida("pressione atmosferica", p_atm_pa)
    rho_in = air_density(p_atm_pa, T_in)
    rho_out = air_density(p_atm_pa, T_out)
    return -0.5 * g_m_s2 * h * (rho_out - rho_in)


def indoor_pressure_target(T_in: float, T_out: float,
                           p_atm_pa: float = P_ATM_STANDARD_PA,
                           h: float = H_STACK_M,
                           g_m_s2: float = G_M_S2,
                           p_in_pa: float | None = None) -> float:
    p_atm_pa = _pressione_valida("pressione atmosferica", p_atm_pa)
    T_in = _temperatura_valida("temperatura interna", T_in)
    T_out = _temperatura_valida("temperatura esterna", T_out)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"altezza del camino non fisica: {h!r} m")
    if not math.isfinite(g_m_s2) or g_m_s2 <= 0.0:
        raise ValueError(f"gravita' non fisica: {g_m_s2!r}")
    p_interna = p_atm_pa if p_in_pa is None else _pressione_valida(
        "pressione dell'ambiente", p_in_pa
    )
    rho_in = air_density(p_interna, T_in)
    rho_out = air_density(p_atm_pa, T_out)
    dp = -0.5 * g_m_s2 * h * max(rho_out - rho_in, 0.0)
    dp = min(max(dp, DP_STACK_MIN_PA), DP_STACK_MAX_PA)
    return p_atm_pa + dp


def indoor_pressure_relaxation_s(delta_p_pa: float,
                                 rho_in_kg_m3: float,
                                 p_atm_pa: float = P_ATM_STANDARD_PA,
                                 volume_m3: float = VOLUME_INVOLUCRO_M3,
                                 a_leak_m2: float = A_LEAK_M2,
                                 cd_leak: float = CD_LEAK) -> float:
    if not math.isfinite(delta_p_pa):
        raise ValueError(f"scarto di pressione non finito: {delta_p_pa!r}")
    if not math.isfinite(rho_in_kg_m3) or rho_in_kg_m3 <= 0.0:
        raise ValueError(f"densita' interna non fisica: {rho_in_kg_m3!r}")
    if not math.isfinite(volume_m3) or volume_m3 <= 0.0:
        raise ValueError(f"volume non fisico: {volume_m3!r} m3")
    if not math.isfinite(a_leak_m2) or a_leak_m2 <= 0.0:
        raise ValueError(f"area di infiltrazione non fisica: {a_leak_m2!r} m2")
    if not math.isfinite(cd_leak) or cd_leak <= 0.0:
        raise ValueError(f"coefficiente di efflusso non fisico: {cd_leak!r}")
    p_atm_pa = _pressione_valida("pressione atmosferica", p_atm_pa)
    dp_rif = max(abs(delta_p_pa), DELTA_P_RIFERIMENTO_MIN_PA)
    tau = (volume_m3 / (p_atm_pa * cd_leak * a_leak_m2)) * math.sqrt(
        rho_in_kg_m3 * dp_rif / 2.0
    )
    return min(max(tau, TAU_PIN_MIN_S), TAU_PIN_MAX_S)


def indoor_pressure_rhs(P_in: float, P_atm: float, T_in: float, T_out: float,
                        volume_m3: float = VOLUME_INVOLUCRO_M3,
                        a_leak_m2: float = A_LEAK_M2,
                        cd_leak: float = CD_LEAK,
                        h: float = H_STACK_M,
                        g_m_s2: float = G_M_S2) -> float:
    P_in = _pressione_valida("pressione dell'ambiente", P_in)
    P_atm = _pressione_valida("pressione atmosferica", P_atm)
    bersaglio = indoor_pressure_target(T_in, T_out, P_atm, h, g_m_s2, p_in_pa=P_in)
    rho_in = air_density(P_in, T_in)
    tau = indoor_pressure_relaxation_s(
        bersaglio - P_in, rho_in, P_atm, volume_m3, a_leak_m2, cd_leak
    )
    return (bersaglio - P_in) / tau
