from __future__ import annotations

import csv
import math
from pathlib import Path

from radon_x.core.decay import LAMBDA_RN222

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

LAMBDA_H = LAMBDA_RN222 * 3600.0

C_OUT_BQ_M3 = 30.0


def _read_csv(name: str) -> list[dict[str, str]]:
    path = DATA_DIR / name
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [line for line in handle if not line.lstrip().startswith("#")]
    return list(csv.DictReader(rows))


def load_exhalation_tufo_gallese() -> tuple[float, float]:
    for row in _read_csv("gallese2020_materiali.csv"):
        if row["material"] == "Tufo di Gallese":
            return float(row["E222_Bq_m2_h"]), float(row["E222_unc"])
    raise LookupError("Tufo di Gallese not found in gallese2020_materiali.csv")


def load_geometry() -> tuple[float, float]:
    surface = volume = None
    for row in _read_csv("gallese2022_parametri_modello.csv"):
        if row["parameter"] == "geometry_exhaling_wall_surface":
            surface = float(row["value"])
        elif row["parameter"] == "geometry_inner_volume":
            volume = float(row["value"])
    if surface is None or volume is None:
        raise LookupError("geometry rows missing in gallese2022_parametri_modello.csv")
    return surface, volume


def load_material_diffusion() -> float:
    for row in _read_csv("gallese2022_parametri_modello.csv"):
        if row["parameter"] == "model_radon_diffusion_coefficient":
            return float(row["value"])
    raise LookupError("diffusion coefficient missing")


def load_equilibria() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in _read_csv("gallese2020_equilibri.csv"):
        out.append(
            {
                "experiment": row["experiment"],
                "description": row["condition_description"],
                "c_meas": float(row["Rn222_eq_Bq_m3"]),
                "c_unc": float(row["Rn222_unc"]),
            }
        )
    return out


def load_ach_experiments() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in _read_csv("gallese2022_ach.csv"):
        out.append(
            {
                "experiment": int(row["experiment"]),
                "group": int(row["group"]),
                "inner_cover": row["inner_cover"],
                "mode": row["mode"],
                "ach": float(row["ACH_h"]),
                "rir_meas": float(row["RIR_pct"]),
                "rir_unc": float(row["RIR_unc_pct"]),
            }
        )
    return out


def equilibrium_concentration(
    e_bq_m2_h: float,
    surface_m2: float,
    volume_m3: float,
    ach_h: float,
    c_out_bq_m3: float,
) -> float:
    if ach_h < 0.0:
        raise ValueError(f"ach_h must be >= 0 (use magnitude), got {ach_h!r}")
    if volume_m3 <= 0.0 or surface_m2 <= 0.0:
        raise ValueError("surface_m2 and volume_m3 must be > 0")
    generation = e_bq_m2_h * surface_m2 / volume_m3
    return (generation + ach_h * c_out_bq_m3) / (LAMBDA_H + ach_h)


def leakage_ach_from_equilibrium(
    c_eq_bq_m3: float,
    e_bq_m2_h: float,
    surface_m2: float,
    volume_m3: float,
    c_out_bq_m3: float,
) -> float:
    generation = e_bq_m2_h * surface_m2 / volume_m3
    if c_eq_bq_m3 <= c_out_bq_m3:
        raise ValueError("equilibrium at or below background: leakage undefined")
    return (generation - LAMBDA_H * c_eq_bq_m3) / (c_eq_bq_m3 - c_out_bq_m3)


def rir_model_pct(
    ach_magnitude_h: float,
    background_ach_h: float,
    e_bq_m2_h: float,
    surface_m2: float,
    volume_m3: float,
    c_out_bq_m3: float,
) -> float:
    c_base = equilibrium_concentration(
        e_bq_m2_h, surface_m2, volume_m3, background_ach_h, c_out_bq_m3
    )
    c_vent = equilibrium_concentration(
        e_bq_m2_h, surface_m2, volume_m3, background_ach_h + ach_magnitude_h, c_out_bq_m3
    )
    return 100.0 * (c_vent - c_base) / c_base


def fit_background_ach(
    target_rir_pct: float,
    ach_magnitude_h: float,
    e_bq_m2_h: float,
    surface_m2: float,
    volume_m3: float,
    c_out_bq_m3: float,
) -> float:
    lo, hi = 0.0, 5.0
    f_lo = rir_model_pct(ach_magnitude_h, lo, e_bq_m2_h, surface_m2, volume_m3, c_out_bq_m3)
    f_hi = rir_model_pct(ach_magnitude_h, hi, e_bq_m2_h, surface_m2, volume_m3, c_out_bq_m3)
    if not (min(f_lo, f_hi) <= target_rir_pct <= max(f_lo, f_hi)):
        raise ValueError(f"target RIR {target_rir_pct} not bracketed by n0 in [0, 5] 1/h")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = rir_model_pct(ach_magnitude_h, mid, e_bq_m2_h, surface_m2, volume_m3, c_out_bq_m3)
        if f_mid < target_rir_pct:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


COMPARABILITY_2020 = {
    "1": "NO (bare porous walls: box not airtight; used for leakage estimate)",
    "2": "YES (outer film seals the box; expect inward flux up to 2E, thin walls)",
    "3": "partial (airtight, but inner plasterboard alters the source)",
    "4": "partial (airtight, inner covers + thermal paint alter the source)",
    "5a": "partial (airtight, inner covers + waterproof paint alter the source)",
    "5b": "NO (film removed: pervious again; leakage estimate confounded by covers)",
}


def part_a() -> dict[str, object]:
    e_meas, e_unc = load_exhalation_tufo_gallese()
    surface, volume = load_geometry()
    diffusion = load_material_diffusion()

    c_eq0 = equilibrium_concentration(e_meas, surface, volume, 0.0, C_OUT_BQ_M3)
    c_eq0_unc = c_eq0 * (e_unc / e_meas)
    c_eq0_sealed = equilibrium_concentration(2.0 * e_meas, surface, volume, 0.0, C_OUT_BQ_M3)
    diff_length = math.sqrt(diffusion / LAMBDA_RN222)

    rows = []
    for eq in load_equilibria():
        ratio = float(eq["c_meas"]) / c_eq0
        rows.append(
            {
                **eq,
                "comparable": COMPARABILITY_2020[str(eq["experiment"])],
                "ratio_meas_over_pred": ratio,
                "deviation_pct": 100.0 * (ratio - 1.0),
            }
        )

    by_exp = {str(r["experiment"]): r for r in rows}
    leak_test1 = leakage_ach_from_equilibrium(
        float(by_exp["1"]["c_meas"]), e_meas, surface, volume, C_OUT_BQ_M3
    )
    leak_test5b = leakage_ach_from_equilibrium(
        float(by_exp["5b"]["c_meas"]), e_meas, surface, volume, C_OUT_BQ_M3
    )

    return {
        "e_meas": e_meas,
        "e_unc": e_unc,
        "surface": surface,
        "volume": volume,
        "c_eq0": c_eq0,
        "c_eq0_unc": c_eq0_unc,
        "c_eq0_sealed_2e": c_eq0_sealed,
        "diffusion_length_m": diff_length,
        "rows": rows,
        "leak_ach_test1": leak_test1,
        "leak_ach_test5b": leak_test5b,
        "ratio_test2_over_pred_2e": float(by_exp["2"]["c_meas"]) / c_eq0_sealed,
    }


def part_b() -> dict[str, object]:
    e_meas, _ = load_exhalation_tufo_gallese()
    surface, volume = load_geometry()
    experiments = load_ach_experiments()

    calib = next(e for e in experiments if e["experiment"] == 5)
    n0 = fit_background_ach(
        float(calib["rir_meas"]), abs(float(calib["ach"])), e_meas, surface, volume, C_OUT_BQ_M3
    )

    rows = []
    for exp in experiments:
        rir_mod = rir_model_pct(
            abs(float(exp["ach"])), n0, e_meas, surface, volume, C_OUT_BQ_M3
        )
        residual = rir_mod - float(exp["rir_meas"])
        rows.append(
            {
                **exp,
                "rir_mod": rir_mod,
                "residual": residual,
                "z": residual / float(exp["rir_unc"]),
                "is_calibration": exp["experiment"] == 5,
            }
        )
    return {"n0": n0, "rows": rows, "e_meas": e_meas, "surface": surface, "volume": volume}


def sensitivity_to_g_and_cout(n0: float) -> dict[str, float]:
    e_meas, _ = load_exhalation_tufo_gallese()
    surface, volume = load_geometry()
    base = rir_model_pct(0.49, n0, e_meas, surface, volume, C_OUT_BQ_M3)
    return {
        "base": base,
        "delta_e_doubled": rir_model_pct(0.49, n0, 2 * e_meas, surface, volume, C_OUT_BQ_M3) - base,
        "delta_cout_0": rir_model_pct(0.49, n0, e_meas, surface, volume, 0.0) - base,
        "delta_cout_60": rir_model_pct(0.49, n0, e_meas, surface, volume, 60.0) - base,
    }


def main() -> None:
    a = part_a()
    print("=" * 78)
    print("PART A -- no-ventilation equilibrium vs 2020 Table 3 (model room, Tufo di Gallese)")
    print("=" * 78)
    print(f"E222 (accumulation chamber, one face) = {a['e_meas']:.2f} +/- {a['e_unc']:.2f} Bq m-2 h-1")
    print(f"S = {a['surface']:.2f} m2, V = {a['volume']:.3f} m3 ('about'), S/V = {a['surface']/a['volume']:.3f} m-1")
    print(f"lambda(Rn-222) = {LAMBDA_RN222:.6e} s-1 = {LAMBDA_H:.6e} h-1  (paper: 2.09794e-6 s-1)")
    print(f"C_out (laboratory background, task briefing) = {C_OUT_BQ_M3:.0f} Bq m-3")
    print(f"Pore diffusion length in the tuff, sqrt(D/lambda) = {a['diffusion_length_m']:.2f} m "
          ">> wall thickness 0.10-0.15 m (thin-wall regime)")
    print()
    print(f"C_eq(n=0, E)  = {a['c_eq0']:.0f} +/- {a['c_eq0_unc']:.0f} Bq m-3   (free one-face exhalation)")
    print(f"C_eq(n=0, 2E) = {a['c_eq0_sealed_2e']:.0f} Bq m-3   (outer face sealed, thin-slab limit)")
    print()
    hdr = f"{'test':>4} | {'measured':>14} | {'meas/pred(E)':>12} | {'dev %':>7} | comparable to n=0 model?"
    print(hdr)
    print("-" * len(hdr))
    for r in a["rows"]:
        print(
            f"{r['experiment']:>4} | {r['c_meas']:>7.0f} +/- {r['c_unc']:>3.0f} | "
            f"{r['ratio_meas_over_pred']:>12.3f} | {r['deviation_pct']:>+7.1f} | {r['comparable']}"
        )
    print()
    print(f"Sealed test 2 vs 2E scenario: measured/predicted = {a['ratio_test2_over_pred_2e']:.3f} "
          f"({100 * (a['ratio_test2_over_pred_2e'] - 1):+.1f} %)")
    print(f"Equivalent leakage ACH, test 1 (bare walls):        {a['leak_ach_test1']:.4f} 1/h")
    print(f"Equivalent leakage ACH, test 5b (covers, no film):  {a['leak_ach_test5b']:.4f} 1/h (confounded)")
    print()

    b = part_b()
    print("=" * 78)
    print("PART B -- RIR vs 2022 Table 2 (twelve imposed-ACH experiments)")
    print("=" * 78)
    print(f"Background ACH n0 fitted on experiment 5 ONLY (ACH 0.09, air_in, cover): "
          f"n0 = {b['n0']:.4f} 1/h")
    print("All other eleven experiments are pure predictions (no further tuning).")
    print()
    hdr = (
        f"{'exp':>3} {'grp':>3} {'cover':>5} {'mode':>7} {'ACH':>6} | "
        f"{'RIR meas':>14} | {'RIR model':>9} | {'resid':>7} | {'z':>6} | note"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in b["rows"]:
        note = "CALIBRATION" if r["is_calibration"] else ("OK <1 sigma" if abs(r["z"]) < 1 else "DISCREPANT")
        print(
            f"{r['experiment']:>3} {r['group']:>3} {r['inner_cover']:>5} {r['mode']:>7} "
            f"{r['ach']:>+6.2f} | {r['rir_meas']:>+7.1f} +/- {r['rir_unc']:>3.1f} | "
            f"{r['rir_mod']:>+9.1f} | {r['residual']:>+7.1f} | {r['z']:>+6.1f} | {note}"
        )
    print()
    pred = [r for r in b["rows"] if not r["is_calibration"]]
    grp1 = [r for r in pred if r["group"] == 1]
    rest = [r for r in pred if r["group"] != 1]
    print(f"Group 1 (air_in, cover), 4 predictions: max |z| = {max(abs(r['z']) for r in grp1):.2f} "
          f"-> the balance + single leakage n0 reproduces them within 1 sigma")
    print(f"Groups 2-4 (air_out and/or no cover), 7 predictions: min |z| = "
          f"{min(abs(r['z']) for r in rest):.2f}, max |z| = {max(abs(r['z']) for r in rest):.2f} "
          "-> systematic failure")
    print("The model is symmetric in |ACH|; the measured air_in/air_out asymmetry "
          "(e.g. exp 1 vs 6: -89.5 vs -25.0 at the same |ACH|) is invisible to it.")
    print()
    sens = sensitivity_to_g_and_cout(float(b["n0"]))
    print(f"Sensitivity of RIR(0.49) = {sens['base']:+.1f} %: "
          f"E doubled {sens['delta_e_doubled']:+.2f}, C_out=0 {sens['delta_cout_0']:+.2f}, "
          f"C_out=60 {sens['delta_cout_60']:+.2f} (percentage points)")


if __name__ == "__main__":
    main()
