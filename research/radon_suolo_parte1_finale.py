"""
Radon transport model in soil.
Surface flux J [Bq/m^2/s] and concentration C_means [kBq/m^3] via soil_outputs(model, sol).
"""

import os
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gamma
import numpy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

PLOT_DARK  = "#0d1117" # figure background
PLOT_PANEL = "#161b22" # panel background
PLOT_TEXT  = "#e6edf3" # label/text color
PLOT_GRID  = "#30363d" # grid lines


try:
    from scipy.stats import qmc as _qmc
    _HAS_QMC = True
except ImportError:
    _HAS_QMC = False
    _qmc = None

_LN2 = np.log(2.0)

# half-lives from DDEP/ENSDF evaluations (Be et al., Table de Radionucleides)
DECAY_CONSTANTS = {
    "Rn222": _LN2 / (3.8232 * 86400.0),         # 3.8232 d (DDEP)
    "Po218": _LN2 / 185.88,                     # 3.098 min (DDEP)
    "Pb214": _LN2 / 1608.0,                     # 26.8 min (DDEP)
    "Bi214": _LN2 / 1182.0,                     # 19.7 min (DDEP)
    "Po214": _LN2 / 1.643e-4,                   # 164.3 us, effectively instantaneous
    "Rn220": _LN2 / 55.8,                       # 55.8 s, thoron (DDEP)
    "Po216": _LN2 / 0.148,                      # 148 ms
    "Pb212": _LN2 / (10.64 * 3600.0),           # 10.64 h
}

# NOTE: the historical VOLCANIC_LITHOTYPES literal was removed (it was dead code,
# silently shadowed by the filtered rebinding from GEOLITHOTYPES further below).



# Ra226 from UNSCEAR 2000 Annex B Tab.5, porosity from Freeze & Cherry 1979
# eps_max from Rogers & Nielson 1991, van Genuchten pars from ROSETTA-1

GEOLITHOTYPES = {
    "tuff_welded": {
        "vesicle_fraction": 0.05,
        "phi_m": 0.30, "phi_f": 0.05,
        "k0": 1.5e-14, "k_frac": 3e-12,
        "D0": 1.1e-5,   # free-air value; effective D via Rogers & Nielson (1991)
        "Ra226_Bqkg": 120.0, "rho_bulk": 1800.0,
        "soil_alpha": 1.5, "soil_n": 1.6,
        "Ksat0": 1e-6,
        "A_crack_m2": 0.10,
        "grain_size_mm": 0.05,
        "eps_max": 0.38, "Sw_peak_eman": 0.28,
        "label": "Tufo saldato",
        "seismic_site": "Campi Flegrei",
        "regime": "geothermal_active",
        "emanation_model": "volcanic_vesicular",
        "epsmax": 0.38, "Acrack_m2": 0.10,
    },
    "tuff_unwelded": {
        "vesicle_fraction": 0.10,
        "phi_m": 0.48, "phi_f": 0.08,
        "k0": 8e-14, "k_frac": 2e-11,
        "D0": 1.1e-5,   # free-air value; effective D via Rogers & Nielson (1991)
        "Ra226_Bqkg": 90.0, "rho_bulk": 1350.0,
        "soil_alpha": 2.0, "soil_n": 1.5,
        "Ksat0": 5e-6,
        "A_crack_m2": 0.50,
        "grain_size_mm": 0.08,
        "eps_max": 0.42, "Sw_peak_eman": 0.22,
        "label": "Tufo non saldato",
        "seismic_site": "Campi Flegrei",
        "regime": "geothermal_active",
        "emanation_model": "volcanic_vesicular",
        "epsmax": 0.42, "Acrack_m2": 0.50,
    },
    "pumice": {
        "vesicle_fraction": 0.40,
        "phi_m": 0.60, "phi_f": 0.12,
        "k0": 3e-13, "k_frac": 1e-10,
        "D0": 1.1e-5,   # free-air value; effective D via Rogers & Nielson (1991)
        "Ra226_Bqkg": 55.0, "rho_bulk": 900.0,
        "soil_alpha": 3.5, "soil_n": 1.45,
        "Ksat0": 2e-5,
        "A_crack_m2": 1.00,
        "grain_size_mm": 2.0,
        "eps_max": 0.28, "Sw_peak_eman": 0.15,
        "label": "Pomice",
        "seismic_site": "Etna",
        "regime": "geothermal_active",
        "emanation_model": "volcanic_vesicular",
        "epsmax": 0.28, "Acrack_m2": 1.00,
    },
    "scoria_basaltic": {
        "vesicle_fraction": 0.25,
        "phi_m": 0.45, "phi_f": 0.10,
        "k0": 1e-13, "k_frac": 5e-11,
        "D0": 1.1e-5,   # free-air value; effective D via Rogers & Nielson (1991)
        "Ra226_Bqkg": 40.0, "rho_bulk": 1200.0,
        "soil_alpha": 2.5, "soil_n": 1.5,
        "Ksat0": 1e-5,
        "A_crack_m2": 0.80,
        "grain_size_mm": 5.0,
        "eps_max": 0.22, "Sw_peak_eman": 0.18,
        "label": "Scoria basaltica",
        "seismic_site": "Etna",
        "regime": "geothermal_active",
        "emanation_model": "volcanic_vesicular",
        "epsmax": 0.22, "Acrack_m2": 0.80,
    },
    "ignimbrite": {
        "phi_m": 0.35, "phi_f": 0.06,
        "k0": 4e-14, "k_frac": 8e-12,
        "D0": 1.1e-5,   # free-air value; effective D via Rogers & Nielson (1991)
        "Ra226_Bqkg": 100.0, "rho_bulk": 1600.0,
        "soil_alpha": 1.8, "soil_n": 1.55,
        "Ksat0": 2e-6,
        "A_crack_m2": 0.30,
        "grain_size_mm": 0.1,
        "eps_max": 0.36, "Sw_peak_eman": 0.25,
        "vesicle_fraction": 0.08,
        "label": "Ignimbrite",
        "seismic_site": "Campi Flegrei",
        "regime": "geothermal_active",
        "emanation_model": "volcanic_vesicular",
        "epsmax": 0.36, "Acrack_m2": 0.30,
    },
    "basalt_dense": {
        "phi_m": 0.08, "phi_f": 0.02,
        "k0": 5e-17, "k_frac": 1e-13,
        "D0": 1.1e-5,   # free-air value; effective D via Rogers & Nielson (1991)
        "Ra226_Bqkg": 25.0, "rho_bulk": 2700.0,
        "soil_alpha": 0.8, "soil_n": 1.8,
        "Ksat0": 5e-8,
        "A_crack_m2": 0.05,
        "grain_size_mm": 0.001,
        "eps_max": 0.12, "Sw_peak_eman": 0.30,
        "vesicle_fraction": 0.01,
        "label": "Basalto denso",
        "seismic_site": "Stromboli",
        "regime": "fracture_dominant",
        "emanation_model": "crystalline_microfracture",
        "epsmax": 0.12, "Acrack_m2": 0.05,
    },
    "sandy_soil": {
        "phi_m": 0.38, "phi_f": 0.04,
        "k0": 2e-12, "k_frac": 5e-12,
        "D0": 1.1e-5,   # free-air value; effective D via Rogers & Nielson (1991)
        "Ra226_Bqkg": 60.0, "rho_bulk": 1600.0,
        "soil_alpha": 4.0, "soil_n": 2.0,
        "Ksat0": 5e-5,
        "A_crack_m2": 0.15,
        "grain_size_mm": 0.5,
        "eps_max": 0.22, "Sw_peak_eman": 0.15,
        "vesicle_fraction": 0.0,
        "label": "Suolo sabbioso (riferimento)",
        "seismic_site": "N/A",
        "regime": "porous_advective",
        "emanation_model": "grain_size_moisture",
        "epsmax": 0.22, "Acrack_m2": 0.15,
    },
    "granite": {
        # Przylibski 2004, Makinen & Karvonen 1999
        # UNSCEAR 2000: Ra-226 ~40-120 Bq/kg in European granites
        "phi_m": 0.02, "phi_f": 0.005,
        "k0": 1e-17, "k_frac": 5e-13,
        "D0": 1.1e-5,
        "Ra226_Bqkg": 80.0, "rho_bulk": 2650.0,
        "soil_alpha": 0.5, "soil_n": 1.80,
        "Ksat0": 5e-9,
        "A_crack_m2": 0.03,
        "grain_size_mm": 5.0,
        "eps_max": 0.10, "Sw_peak_eman": 0.35,  # Semkow 1990
        "vesicle_fraction": 0.0,
        "label": "Granito",
        "seismic_site": "N/A",
        "regime": "fracture_dominant",
        "emanation_model": "crystalline_microfracture",
        "eps_crystalline": 0.08,               # approx constant, Semkow 1990
        "D_matrix_crystalline": 1e-11,
        "k_frac_eff": 5e-13,
        "fracture_density_factor": 1.0,
        "texture": "fractured_rock",
        "epsmax": 0.10, "Acrack_m2": 0.03,
    },
    "gneiss": {
        # Spitieri et al. 2008, Appelo & Postma 2005
        # phi_m typically 0.02-0.04, Freeze & Cherry 1979 Tab. 2.4
        "phi_m": 0.03, "phi_f": 0.007,
        "k0": 5e-17, "k_frac": 3e-13,
        "D0": 1.1e-5,
        "Ra226_Bqkg": 60.0, "rho_bulk": 2700.0,
        "soil_alpha": 0.5, "soil_n": 1.80,
        "Ksat0": 3e-9,
        "A_crack_m2": 0.04,
        "grain_size_mm": 3.0,
        "eps_max": 0.12, "Sw_peak_eman": 0.32,
        "vesicle_fraction": 0.0,
        "label": "Gneiss",
        "seismic_site": "N/A",
        "regime": "fracture_dominant",
        "emanation_model": "crystalline_microfracture",
        "eps_crystalline": 0.09,
        "D_matrix_crystalline": 1e-11,
        "k_frac_eff": 3e-13,
        "fracture_density_factor": 1.1,
        "texture": "fractured_rock",
        "epsmax": 0.12, "Acrack_m2": 0.04,
    },
    "schist": {
        # phi_m ~0.04-0.07, Domenico & Schwartz 1990
        # Ra-226: UNSCEAR 2000, metamorphic rocks ~30-70 Bq/kg
        "phi_m": 0.05, "phi_f": 0.015,
        "k0": 2e-16, "k_frac": 8e-13,         # foliated schists: parallel fractures
        "D0": 1.1e-5,
        "Ra226_Bqkg": 55.0, "rho_bulk": 2750.0,
        "soil_alpha": 0.5, "soil_n": 1.80,
        "Ksat0": 5e-9,
        "A_crack_m2": 0.06,
        "grain_size_mm": 0.5,
        "eps_max": 0.15, "Sw_peak_eman": 0.30,
        "vesicle_fraction": 0.0,
        "label": "Scisto/Filladi",
        "seismic_site": "N/A",
        "regime": "fracture_dominant",
        "emanation_model": "crystalline_microfracture",
        "eps_crystalline": 0.12,
        "D_matrix_crystalline": 5e-12,
        "k_frac_eff": 8e-13,
        "fracture_density_factor": 1.3,
        "texture": "fractured_rock",
        "epsmax": 0.15, "Acrack_m2": 0.06,
    },
    "limestone_compact": {
        # phi_m ~0.05-0.12, Domenico & Schwartz 1990
        # Ra-226: Appelo & Postma 2005, carbonates 5-20 Bq/kg
        "phi_m": 0.08, "phi_f": 0.02,
        "k0": 1e-15, "k_frac": 1e-12,
        "D0": 1.1e-5,
        "Ra226_Bqkg": 15.0, "rho_bulk": 2500.0,
        "soil_alpha": 1.2, "soil_n": 1.50,
        "Ksat0": 1e-7,
        "A_crack_m2": 0.05,
        "grain_size_mm": 0.1,
        "eps_max": 0.08, "Sw_peak_eman": 0.28,
        "vesicle_fraction": 0.0,
        "label": "Calcare compatto",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "grain_size_moisture",
        "texture": "karst_matrix",
        "epsmax": 0.08, "Acrack_m2": 0.05,
    },
    "limestone_karst": {
        # karst limestone: matrix + conduits, Spitieri et al. 2008
        # Ra source in the red soil, Ababneh et al. 2009
        "phi_m": 0.15, "phi_f": 0.05,
        "k0": 5e-14, "k_frac": 1e-10,         # high k due to conduits
        "D0": 1.1e-5,
        "Ra226_Bqkg": 20.0, "rho_bulk": 2300.0,
        "soil_alpha": 1.2, "soil_n": 1.50,
        "Ksat0": 2e-6,
        "A_crack_m2": 0.15,
        "grain_size_mm": 0.1,
        "eps_max": 0.25, "Sw_peak_eman": 0.40,  # higher peak for red soil
        "vesicle_fraction": 0.0,
        "label": "Calcare carsico",
        "seismic_site": "N/A",
        "regime": "karst_conduit",
        "emanation_model": "karst_terra_rossa",
        "eps_karst": 0.22,
        "f_karst_conduit": 0.08,
        "D_conduit": 5e-6,
        "k_karst_conduit": 1e-10,
        "f_karst_active": 0.05,
        "texture": "karst_matrix",
        "epsmax": 0.25, "Acrack_m2": 0.15,
    },
    "dolomite": {
        # properties intermediate between limestone and granite, Domenico & Schwartz 1990
        "phi_m": 0.08, "phi_f": 0.015,
        "k0": 5e-15, "k_frac": 5e-13,
        "D0": 1.1e-5,
        "Ra226_Bqkg": 12.0, "rho_bulk": 2600.0,
        "soil_alpha": 1.0, "soil_n": 1.55,
        "Ksat0": 5e-8,
        "A_crack_m2": 0.04,
        "grain_size_mm": 0.2,
        "eps_max": 0.07, "Sw_peak_eman": 0.30,
        "vesicle_fraction": 0.0,
        "label": "Dolomia",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "grain_size_moisture",
        "texture": "karst_matrix",
        "epsmax": 0.07, "Acrack_m2": 0.04,
    },
    "sandstone": {
        # Domenico & Schwartz 1990, USGS OFR 2001-175
        # Ra-226: UNSCEAR 2000 ~30-70 Bq/kg
        "phi_m": 0.20, "phi_f": 0.03,
        "k0": 5e-13, "k_frac": 2e-12,
        "D0": 1.1e-5,
        "Ra226_Bqkg": 50.0, "rho_bulk": 2100.0,
        "soil_alpha": 7.5, "soil_n": 1.89,
        "Ksat0": 1e-5,
        "A_crack_m2": 0.10,
        "grain_size_mm": 0.30,
        "eps_max": 0.30, "Sw_peak_eman": 0.18,  # Rogers & Nielson 1991
        "vesicle_fraction": 0.0,
        "label": "Arenaria",
        "seismic_site": "N/A",
        "regime": "porous_advective",
        "emanation_model": "grain_size_moisture",
        "texture": "sandy_loam",
        "epsmax": 0.30, "Acrack_m2": 0.10,
    },
    "clay": {
        # strong hydraulic sealing, Carsel & Parrish
        # Ra-226: UNSCEAR 2000 ~30-80 Bq/kg, clays often enriched
        "phi_m": 0.45, "phi_f": 0.02,
        "k0": 5e-16, "k_frac": 1e-15,         # nearly impermeable to gas
        "D0": 1.1e-5,
        "Ra226_Bqkg": 55.0, "rho_bulk": 1400.0,
        "soil_alpha": 0.8, "soil_n": 1.09,
        "Ksat0": 5e-7,
        "A_crack_m2": 0.02,
        "grain_size_mm": 0.002,                # clay: <0.002 mm
        "eps_max": 0.45, "Sw_peak_eman": 0.12,  # Rogers & Nielson 1991, fine grains
        "vesicle_fraction": 0.0,
        "label": "Argilla",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "grain_size_moisture",
        "texture": "clay",
        "epsmax": 0.45, "Acrack_m2": 0.02,
    },
    "silt": {
        # Carsel & Parrish, silt loam
        "phi_m": 0.40, "phi_f": 0.02,
        "k0": 1e-14, "k_frac": 5e-15,
        "D0": 1.1e-5,
        "Ra226_Bqkg": 45.0, "rho_bulk": 1500.0,
        "soil_alpha": 2.0, "soil_n": 1.41,
        "Ksat0": 1e-6,
        "A_crack_m2": 0.05,
        "grain_size_mm": 0.02,                 # silt: 0.002-0.063 mm
        "eps_max": 0.40, "Sw_peak_eman": 0.14,
        "vesicle_fraction": 0.0,
        "label": "Limo/Silt",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "grain_size_moisture",
        "texture": "silt_loam",
        "epsmax": 0.40, "Acrack_m2": 0.05,
    },
    "alluvium_mixed": {
        # Freeze & Cherry 1979, Table 2.4
        "phi_m": 0.35, "phi_f": 0.04,
        "k0": 5e-13, "k_frac": 2e-12,
        "D0": 1.1e-5,
        "Ra226_Bqkg": 45.0, "rho_bulk": 1700.0,
        "soil_alpha": 3.6, "soil_n": 1.56,
        "Ksat0": 3e-6,
        "A_crack_m2": 0.12,
        "grain_size_mm": 0.5,
        "eps_max": 0.28, "Sw_peak_eman": 0.16,
        "vesicle_fraction": 0.0,
        "label": "Alluvione mista",
        "seismic_site": "N/A",
        "regime": "porous_advective",
        "emanation_model": "grain_size_moisture",
        "texture": "loam",
        "epsmax": 0.28, "Acrack_m2": 0.12,
    },
    "glacial_till": {
        # variable porosity, low k, Freeze & Cherry 1979
        "phi_m": 0.30, "phi_f": 0.03,
        "k0": 1e-14, "k_frac": 5e-14,
        "D0": 1.1e-5,
        "Ra226_Bqkg": 35.0, "rho_bulk": 2000.0,
        "soil_alpha": 1.9, "soil_n": 1.31,
        "Ksat0": 7e-7,
        "A_crack_m2": 0.06,
        "grain_size_mm": 0.5,
        "eps_max": 0.24, "Sw_peak_eman": 0.18,
        "vesicle_fraction": 0.0,
        "label": "Till glaciale",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "grain_size_moisture",
        "texture": "clay_loam",
        "epsmax": 0.24, "Acrack_m2": 0.06,
    },
    "marl": {
        # clay-carbonate mix, Appelo & Postma 2005
        "phi_m": 0.25, "phi_f": 0.025,
        "k0": 5e-15, "k_frac": 2e-14,
        "D0": 1.1e-5,
        "Ra226_Bqkg": 30.0, "rho_bulk": 2100.0,
        "soil_alpha": 1.9, "soil_n": 1.31,
        "Ksat0": 8e-7,
        "A_crack_m2": 0.05,
        "grain_size_mm": 0.01,
        "eps_max": 0.22, "Sw_peak_eman": 0.16,
        "vesicle_fraction": 0.0,
        "label": "Marna",
        "seismic_site": "N/A",
        "regime": "porous_diffusive",
        "emanation_model": "grain_size_moisture",
        "texture": "clay_loam",
        "epsmax": 0.22, "Acrack_m2": 0.05,
    },
    "uraniferous_granite": {
        # high-Ra granitoid: Bohemia (Neznal et al. 2004), Sweden (Sundal et al. 2008)
        # Ra-226 can reach 500 Bq/kg in Scandinavian/Bohemian uraniferous areas
        "phi_m": 0.02, "phi_f": 0.006,
        "k0": 2e-17, "k_frac": 8e-13,         # like granite but more open microfractures
        "D0": 1.1e-5,
        "Ra226_Bqkg": 300.0, "rho_bulk": 2680.0,  # range 200-500 Bq/kg
        "soil_alpha": 0.5, "soil_n": 1.80,
        "Ksat0": 5e-9,
        "A_crack_m2": 0.04,
        "grain_size_mm": 4.0,
        "eps_max": 0.12, "Sw_peak_eman": 0.35,
        "vesicle_fraction": 0.0,
        "label": "Granitoide uranifero",
        "seismic_site": "N/A",
        "regime": "fracture_dominant",
        "emanation_model": "crystalline_microfracture",
        "eps_crystalline": 0.10,
        "D_matrix_crystalline": 1e-11,
        "k_frac_eff": 8e-13,
        "fracture_density_factor": 1.2,
        "texture": "fractured_rock",
        "epsmax": 0.12, "Acrack_m2": 0.04,
    },
}

VOLCANIC_LITHOTYPES = {
    k: v for k, v in GEOLITHOTYPES.items()
    if k in ("tuff_welded", "tuff_unwelded", "pumice", "scoria_basaltic",
             "ignimbrite", "basalt_dense", "sandy_soil")
}

# van Genuchten texture parameters from Carsel & Parrish 1988 (WRR 24:755-769)
# format: alpha [1/m], n [-], theta_r [m3/m3], theta_s [m3/m3], Ksat [m/s]
VG_TEXTURE_PARAMS = {
    "sand":           (14.5,   2.68,  0.045,   0.430,  8.25e-5),
    "loamy_sand":     (12.4,   2.28,  0.057,   0.410,  4.05e-5),
    "sandy_loam":     ( 7.5,   1.89,  0.065,   0.410,  1.23e-5),
    "loam":           ( 3.6,   1.56,  0.078,   0.430,  2.89e-6),
    "silt_loam":      ( 2.0,   1.41,  0.067,   0.450,  1.25e-6),
    "clay_loam":      ( 1.9,   1.31,  0.095,   0.410,  7.22e-7),
    "clay":           ( 0.8,   1.09,  0.068,   0.380,  5.56e-7),
    # fractured/karst values estimated from field data
    "fractured_rock": ( 0.5,   1.80,  0.010,   0.050,  5.00e-8),
    "karst_matrix":   ( 1.2,   1.50,  0.020,   0.150,  1.00e-6),
}

REGIMES = {
    "porous_diffusive": {
        "phi_m_range": (0.15, 0.65),
        "transport": "millington_quirk",
        "description": "Argille, limi, marne — diffusione dominante, k bassa",
    },
    "porous_advective": {
        "phi_m_range": (0.15, 0.65),
        "transport": "millington_quirk + darcy",
        "description": "Sabbie, alluvioni — advection + diffusion, k moderata-alta",
    },
    "fracture_dominant": {
        "phi_m_range": (0.01, 0.15),
        "transport": "cubic_law_network",
        "description": "Graniti, gneiss, scisti, basalto denso — trasporto in fratture",
    },
    "karst_conduit": {
        "phi_m_range": (0.05, 0.30),
        "transport": "conduit_diffuse_dual",
        "description": "Calcari carsici — doppia permeabilità matrice + condotti",
    },
    "geothermal_active": {
        "phi_m_range": (0.10, 0.65),
        "transport": "buoyant_advection",
        "description": "Vulcanici attivi — gradiente termico + sovrapressione magmatica",
    },
}

SEISMIC_SITES = {
    "Etna": {
        "M_ref": 3.5, "gamma_seis": 1.8,
        "A_seis": 0.015, "tau_seis": 8 * 3600.0,
        "f_seismic": 1.5 / 86400.0,
    },
    "Campi Flegrei": {
        "M_ref": 2.5, "gamma_seis": 2.2,
        "A_seis": 0.025, "tau_seis": 12 * 3600.0,
        "f_seismic": 3.0 / 86400.0,
    },
    "Stromboli": {
        "M_ref": 1.5, "gamma_seis": 1.5,
        "A_seis": 0.010, "tau_seis": 4 * 3600.0,
        "f_seismic": 8.0 / 86400.0,
    },
    "N/A": {
        "M_ref": 0.0, "gamma_seis": 0.0,
        "A_seis": 0.0, "tau_seis": 1.0,
        "f_seismic": 0.0,
    },
}

# harmonic face mean gives the correct series resistance
_DIFF_FACE_HARMONIC = True

def _clip01(x):
    return np.clip(x, 0.0, 1.0)

def _pos(x, eps=1e-30):
    return np.maximum(np.asarray(x, dtype=float), eps)

def _smooth_step(x, x0, s=0.03):
    return 0.5 * (1.0 + np.tanh((np.asarray(x) - x0) / max(s, 1e-12)))

def _face_gradient(P, dz):
    """Face-centred gradient of P on the nz+1 cell faces."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    g = np.empty(n + 1)
    if n == 1:
        g[:] = 0.0
        return g
    g[1:n] = (P[1:] - P[:-1]) / dz          # interior faces, centred difference
    g[0]   = g[1]                            # top boundary face
    g[-1]  = g[-2]                           # bottom boundary face
    return g

def _cn_diffusion_step(C, D, dz, h, C_top=0.0, C_bot=None, harmonic=True, beta=None):
    """Crank-Nicolson implicit step of beta dC/dt = d/dz(D dC/dz), unconditionally stable.

    With beta=None the classic dC/dt = d/dz(D dC/dz) is solved. Passing the
    partition-porosity array beta scales each ROW of the operator by 1/beta_i
    while keeping the physical face fluxes D_face*(C_j - C_i)/dz continuous:
    this is the flux-conservative form of the partition equation (pre-dividing
    D by beta would make the face flux (D/beta)*dC/dz, which is discontinuous
    across cells with different beta and not mass-conserving)."""
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    n = len(C)
    if n == 1:
        return C.copy()
    # face diffusivities (harmonic or arithmetic), interior faces 1..n-1
    Dl = D[:-1]; Dr = D[1:]
    if harmonic:
        D_int = 2.0 * Dl * Dr / np.maximum(Dl + Dr, 1e-300)   # (n-1,)
    else:
        D_int = 0.5 * (Dl + Dr)
    r = h / (dz * dz)
    # transmissivity per face, interior face i (between cell i-1 and i)
    Dface = np.empty(n + 1)
    Dface[1:n] = D_int
    Dface[0]   = D[0]     # top ghost face
    Dface[n]   = D[-1]    # bottom ghost face
    aW = Dface[:-1]       # west-face conductance for each cell (n,)
    aE = Dface[1:]        # east-face conductance for each cell (n,)
    if beta is not None:
        beta_arr = np.maximum(np.asarray(beta, dtype=float), 1e-12)
    else:
        beta_arr = np.ones(n)
    # Crank-Nicolson (theta=1/2) is A-stable but NOT L-stable: when the cell
    # Fourier number r*D/beta >> 1 the Dirichlet-boundary cell rings (its
    # amplification factor tends to -1), and the positivity clip applied after
    # each substep then destroys mass at every oscillation. At field grids
    # (dz ~ 0.1 m) the number is << 1 and CN is exact; at mm-scale cells
    # (thin-bed benchmarks) it reaches O(10^2-10^3), the surface cell is pinned
    # to zero and the discrete surface flux LOSES accuracy as the grid is
    # refined (inverse convergence). Above the ringing threshold the whole
    # step switches to backward Euler (theta=1, L-stable): the O(dt) accuracy
    # loss only affects steps where CN was unusable, and using one theta for
    # all rows keeps the face fluxes conservative.
    theta = 0.5
    if r * float(np.max(Dface)) / float(np.min(beta_arr)) > 1.0:
        theta = 1.0
    # tridiagonal A = I - theta*r*L, B = I + (1-theta)*r*L, ghost-node Dirichlet BCs
    lo = np.zeros(n); di = np.zeros(n); up = np.zeros(n)
    rhs = np.zeros(n)
    for i in range(n):
        imp = theta * r / beta_arr[i]
        exq = (1.0 - theta) * r / beta_arr[i]
        cW = aW[i]; cE = aE[i]
        if i == 0:
            # top ghost node C_{-1} = 2*C_top - C_0; the Dirichlet forcing is
            # theta-weighted across the two sides, total (imp+exq)*2*cW*C_top
            di[i]  = 1.0 + imp * (2.0*cW + cE)
            up[i]  = -imp * cE
            b_expl = 1.0 - exq * (2.0*cW + cE)
            rhs[i] = b_expl * C[i] + exq*cE*C[i+1] + (imp + exq)*2.0*cW*C_top
        elif i == n-1:
            if C_bot is None:
                # zero-flux (Neumann) base: C_n = C_{n-1}
                di[i] = 1.0 + imp * (cW)
                lo[i] = -imp * cW
                b_expl = 1.0 - exq * (cW)
                rhs[i] = b_expl * C[i] + exq*cW*C[i-1]
            else:
                di[i]  = 1.0 + imp * (cW + 2.0*cE)
                lo[i]  = -imp * cW
                b_expl = 1.0 - exq * (cW + 2.0*cE)
                rhs[i] = b_expl * C[i] + exq*cW*C[i-1] + (imp + exq)*2.0*cE*C_bot
        else:
            di[i] = 1.0 + imp * (cW + cE)
            lo[i] = -imp * cW
            up[i] = -imp * cE
            b_expl = 1.0 - exq * (cW + cE)
            rhs[i] = b_expl * C[i] + exq*cW*C[i-1] + exq*cE*C[i+1]
    # Thomas algorithm tridiagonal solve
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = up[0]/di[0] if abs(di[0])>1e-300 else 0.0
    dp[0] = rhs[0]/di[0] if abs(di[0])>1e-300 else C[0]
    for i in range(1, n):
        m = di[i] - lo[i]*cp[i-1]
        if abs(m) < 1e-300:
            cp[i]=0.0; dp[i]=C[i]; continue
        cp[i] = up[i]/m
        dp[i] = (rhs[i] - lo[i]*dp[i-1])/m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x


def _upwind(C, q, C_top=0.0, C_bot_inlet=None):
    """Face-centered upwind flux C*q; top Dirichlet C=0, bottom switches with flow direction."""
    C = np.asarray(C, dtype=float)
    C_ext = np.empty(len(C) + 2)
    C_ext[1:-1] = C
    C_ext[0]    = C_top    # Dirichlet: radon concentration vanishes at surface
    if C_bot_inlet is not None and q[-1] < 0.0:
        # inflow at bottom: prescribe inlet concentration
        C_ext[-1] = C_bot_inlet
    else:
        # outflow at bottom: Neumann (zero-gradient)
        C_ext[-1] = C[-1]
    q_face = np.asarray(q, dtype=float)  # length nz+1
    F = np.where(q_face >= 0, C_ext[:-1], C_ext[1:]) * q_face
    return F

def _div_diff_flux(C, D, dz, C_top=0.0, C_bot=None):
    """Finite-volume divergence of D*grad(C); Dirichlet top, Dirichlet/Neumann bottom."""
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    n = len(C)
    C_ext = np.empty(n + 2)
    C_ext[1:-1] = C
    C_ext[0]    = 2.0 * C_top - C[0]                             # Dirichlet at top
    C_ext[-1]   = C[-1] if C_bot is None else 2.0 * C_bot - C[-1]  # Dirichlet or Neumann at bottom
    D_ext = np.empty(n + 2)
    D_ext[1:-1] = D
    D_ext[0]    = D[0]
    D_ext[-1]   = D[-1]
    # harmonic face mean is the series-resistance form (Patankar eq. 4.9);
    # the arithmetic mean overestimates flux across a wet/dry interface
    if _DIFF_FACE_HARMONIC:
        Dl = D_ext[:-1]; Dr = D_ext[1:]
        D_face = 2.0 * Dl * Dr / np.maximum(Dl + Dr, 1e-300)
    else:
        D_face = 0.5 * (D_ext[:-1] + D_ext[1:])
    F = D_face * (C_ext[1:] - C_ext[:-1]) / dz
    return (F[1:] - F[:-1]) / dz


class VolcanicRadonModelV2:
    """Coupled water/gas/radon transport model with 10 state fields per depth node plus indoor coupling."""

    def __init__(self, p: dict):
        self.p = p.copy()
        self.nz   = p.get("nz", 60)
        self.H    = p["H"]
        self.dz   = self.H / self.nz
        self.z    = (np.arange(self.nz) + 0.5) * self.dz

        # Decay constants
        self.lam_rn  = DECAY_CONSTANTS["Rn222"]
        self.lam_po218 = DECAY_CONSTANTS["Po218"]
        self.lam_pb214 = DECAY_CONSTANTS["Pb214"]
        self.lam_bi214 = DECAY_CONSTANTS["Bi214"]
        self.lam_po214 = DECAY_CONSTANTS["Po214"]
        self.lam_rn220 = DECAY_CONSTANTS["Rn220"]

        self._eps_mem_init = None
        self._source_cache_rn222 = None
        self._source_cache_rn220 = None
        self.cracks = []
        self.horizontal_network = []
        self._k_seismic = np.ones(self.nz)   # seismic permeability multiplier
        self._t_seismic_last = -1e9

        self._build_dynamic_fracture_network(
            n_cracks=p.get("n_cracks", 18),
            seed=p.get("crack_seed", 42)
        )
        self._build_horizontal_fracture_layer(
            n_frac=p.get("n_frac_horiz", 20),
            seed=p.get("horiz_seed", 7)
        )

        site = p.get("seismic_site", "Campi Flegrei")
        self.seis = SEISMIC_SITES.get(site, SEISMIC_SITES["N/A"])

        # Van Genuchten hysteresis: separate drainage/wetting branches
        self._vg_alpha_d = p["soil_alpha"]
        self._vg_n_d     = p["soil_n"]
        self._vg_alpha_w = p.get("soil_alpha_wet", p["soil_alpha"] * 0.55)  # wetting scan
        self._vg_n_w     = p.get("soil_n_wet",     p["soil_n"]     * 0.92)

        self.nstate = 10 * self.nz + 7

    def _Se_drainage(self, theta):
        """Effective saturation Se = (theta - theta_r)/(theta_s - theta_r) (van Genuchten 1980).

        theta_r is capped at phi/4 so that low-porosity rock lithotypes
        (phi_m as low as 0.02) keep a well-posed effective saturation."""
        theta   = np.asarray(theta, dtype=float)
        phi     = self.p["phi_m"]
        theta_r = min(self.p.get("theta_r", 0.03), 0.25 * phi)
        Se      = np.clip((theta - theta_r) / max(phi - theta_r, 1e-12), 1e-8, 0.9999)
        return Se

    def _h_from_theta_drainage(self, theta):
        """Capillary pressure head on drainage branch [m].

        The van Genuchten inverse diverges as Se -> 0; the head is capped at
        h_dry_cap (default -1000 m, the usual HYDRUS hCritA air-dry limit) so
        that near-residual cells cannot generate astronomical gradients that
        destabilize the explicit water update."""
        Se    = self._Se_drainage(theta)
        alpha = self._vg_alpha_d
        n     = self._vg_n_d
        m     = 1.0 - 1.0 / n
        h     = -((Se**(-1.0/m) - 1.0)**(1.0/n)) / alpha
        h_cap = -abs(self.p.get("h_dry_cap", 1.0e3))
        return np.maximum(h, h_cap)

    def _kr_vg_drainage(self, Se):
        """Mualem-van Genuchten relative permeability (drainage)."""
        Se    = np.clip(Se, 1e-8, 0.9999)
        n     = self._vg_n_d
        m     = 1.0 - 1.0 / n
        kr    = np.sqrt(Se) * (1.0 - (1.0 - Se**(1.0/m))**m)**2
        return kr

    def _K_unsat(self, theta, z):
        Se   = self._Se_drainage(theta)
        Ksat = self.p["Ksat0"] * (1.0 + self.p.get("Ksat_grad", 0.2) * z / self.H)
        # no artificial lower floor: near the dry limit K must vanish, otherwise
        # the capped-but-large suction gradients drive huge spurious fluxes
        return np.clip(Ksat * self._kr_vg_drainage(Se), 1e-20, 1e-4)

    def richards_rhs(self, theta, t):
        """Richards equation rhs (drainage branch only), vectorized over depth.

        z is DEPTH (positive downward), so the Darcy-Buckingham flux in the +z
        direction is q = -K*(dh/dz - 1): with uniform h, gravity gives q = +K
        (downward drainage)."""
        theta_r = min(self.p.get("theta_r", 0.03), 0.25 * self.p["phi_m"])
        theta = np.clip(theta, theta_r + 1e-6, 0.95 * self.p["phi_m"])
        h     = self._h_from_theta_drainage(theta)
        K     = self._K_unsat(theta, self.z)
        q     = np.empty(self.nz + 1)
        rain  = self.p.get("rain0", 2e-7) * max(np.sin(2*np.pi*t / self.p.get("T_rain", 864000)), 0.0)
        evap  = self.p.get("evap0", 1e-7) * max(np.sin(2*np.pi*t / self.p.get("T_evap", 86400) + self.p.get("evap_phase", np.pi/2)), 0.0)
        # stage-2 evaporation: atmospheric demand is cut off as the surface cell
        # approaches the residual limit (soil cannot supply the demanded flux)
        Se0   = float(self._Se_drainage(theta[:1])[0])
        evap *= float(np.clip(Se0 / 0.05, 0.0, 1.0))
        q[0]  = rain - evap
        dhdz_int = (h[1:] - h[:-1]) / self.dz          # interior face gradients, shape (nz-1,)
        Kf_int   = 0.5 * (K[1:] + K[:-1])              # arithmetic-mean face conductivity
        q[1:self.nz] = -Kf_int * (dhdz_int - 1.0)      # depth coordinate: gravity term is -1
        # fixed deep head allows upward flux; free drainage would dry the
        # column monotonically over long runs
        bc_mode = self.p.get("richards_bottom_bc", "water_table")
        if bc_mode == "free_drainage":
            q[-1] = K[-1]                               # gravity-only flux (downward)
        else:
            # deep Dirichlet head h_bot [m] allows recharge or drainage consistently
            h_bot   = self.p.get("richards_h_bottom", h[-1])
            dz_half = 0.5 * self.dz
            dhdz_bot = (h_bot - h[-1]) / dz_half        # head gradient to depth
            q[-1] = -K[-1] * (dhdz_bot - 1.0)           # depth coordinate: gravity term is -1
        return -(q[1:] - q[:-1]) / self.dz

    def T_field(self, t):
        """Depth-time temperature field: geothermal gradient plus damped daily surface wave."""
        z      = self.z
        T0     = self.p.get("T0", 288.0)
        grad   = self.p.get("T_grad", 0.025)        # K/m geothermal gradient
        dT_s   = self.p.get("T_surf_amp", 8.0)      # K surface daily amplitude
        # daily damping depth sqrt(2*kappa/omega) ~ 0.12 m for kappa ~ 5e-7 m2/s
        z_damp = self.p.get("T_damp_depth", 0.12)   # m, thermal damping depth
        omega  = 2*np.pi / 86400.0
        # damped heat-conduction wave carries the phase lag -z/z_damp (Carslaw & Jaeger 1959)
        T      = T0 + grad * z + dT_s * np.exp(-z / z_damp) * np.sin(omega * t - z / z_damp)
        return np.clip(T, 260.0, 370.0)

    def T_outdoor(self, t):
        """Outdoor air temperature [K] with annual plus daily harmonics."""
        # annual term drives the seasonal stack effect (Riley et al. 1999)
        T_out0 = self.p.get("T_out0", 283.15)
        A_day  = self.p.get("T_out_amp", 5.0)
        A_year = self.p.get("T_out_year_amp", 0.0)       # disabled by default
        phi_y  = self.p.get("T_out_year_phase", -np.pi/2) # phased to winter minimum
        T_year = self.p.get("T_year", 365.25 * 86400.0)
        T_day  = self.p.get("T_day", 86400.0)
        return (T_out0
                + A_year * np.sin(2*np.pi * t / max(T_year, 1.0) + phi_y)
                + A_day  * np.sin(2*np.pi * t / max(T_day, 1.0)))

    def T_indoor(self, t):
        """Indoor air temperature [K] for a heated/thermostatted building."""
        T_in0  = self.p.get("T_in0", 293.15)
        A_day  = self.p.get("T_in_amp", 2.0)
        A_year = self.p.get("T_in_year_amp", 0.0)
        phi_y  = self.p.get("T_in_year_phase", -np.pi/2)
        T_year = self.p.get("T_year", 365.25 * 86400.0)
        T_day  = self.p.get("T_day", 86400.0)
        return (T_in0
                + A_year * np.sin(2*np.pi * t / max(T_year, 1.0) + phi_y)
                + A_day  * np.sin(2*np.pi * t / max(T_day, 1.0)))

    def stack_pressure(self, t, Tin=None, Tout=None, Patm=None):
        """Buoyancy (stack) pressure deficit across the foundation [Pa] from indoor-outdoor density contrast."""
        g    = self.p.get("g", 9.81)
        H_s  = self.p.get("H_stack", 2.5)
        Mg   = self.p.get("M_g", 0.02897)
        Rg   = self.p.get("R_g", 8.314)
        Patm = self.p.get("P_atm0", 101325.0) if Patm is None else float(Patm)
        Tin  = self.T_indoor(t)  if Tin  is None else Tin
        Tout = self.T_outdoor(t) if Tout is None else Tout
        rho_in  = Patm * Mg / (Rg * np.maximum(Tin,  1.0))
        rho_out = Patm * Mg / (Rg * np.maximum(Tout, 1.0))
        return -0.5 * g * H_s * (rho_out - rho_in)

    def stack_effect_advective_factor(self, t):
        """Stack-effect driving pressure magnitude |dP_stack| [Pa], used as the
        meteorological regressor when separating weather from seismic signal."""
        dP = np.abs(self.stack_pressure(t))
        return dP

    def _k_gas_standard(self, theta, z, seismic_mult=None):
        """Gas permeability k(theta,z): depth gradient times capillary opening times relative-permeability power law."""
        # Mualem 1976, Water Resour. Res. 12:513-522, relative permeability power law
        theta    = np.asarray(theta, dtype=float)
        phi      = self.p["phi_m"]
        Sw       = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        Sg       = 1.0 - Sw
        Sg_crit  = self.p.get("Sg_crit", 0.08)
        Sg_sharp = self.p.get("Sg_sharp", 0.02)
        nk       = self.p.get("nk", 3.5)
        k0       = self.p["k0"] * np.exp(self.p.get("kz_grad", 0.0) * z / self.H)
        open_mask= 0.5*(1.0 + np.tanh((Sg - Sg_crit) / max(Sg_sharp, 1e-12)))
        rel      = np.clip((Sg - Sg_crit) / max(1.0 - Sg_crit, 1e-12), 0.0, 1.0)
        k        = k0 * open_mask * rel**nk
        k        = np.clip(k, self.p.get("k_min", 1e-18), self.p.get("k_max", 1e-10))
        if seismic_mult is not None:
            k = k * seismic_mult
        return k

    def _k_gas(self, theta, z, seismic_mult=None):
        """Gas permeability with regime-dependent branching (fracture, karst conduit, or standard)."""
        # fracture closure with saturation, f_open = 1 - Sw^2 (Bolton et al. 1998)
        # karst conduit intermittency after Spitieri et al. 2008, Radiat. Meas. 43:1396
        regime = self.p.get("regime", "porous_diffusive")

        if regime == "fracture_dominant":
            theta    = np.asarray(theta, dtype=float)
            phi      = self.p["phi_m"]
            Sw       = np.clip(theta / max(phi, 1e-12), 0.0, 1.0)
            # crystalline matrix permeability, negligible but non-zero
            k_matrix = self.p["k0"] * np.exp(self.p.get("kz_grad", 0.0) * z / self.H)
            k_frac_eff = self.p.get("k_frac_eff", self.p.get("k_frac", 1e-13))
            f_frac     = self.p.get("fracture_density_factor", 1.0)
            frac_open  = np.clip(1.0 - Sw**2, 0.01, 1.0)
            k_eff      = k_matrix + k_frac_eff * f_frac * frac_open
            k_eff      = np.clip(k_eff, self.p.get("k_min", 1e-18),
                                         self.p.get("k_max", 1e-10))
            if seismic_mult is not None:
                k_eff = k_eff * seismic_mult
            return k_eff

        elif regime == "karst_conduit":
            k_base    = self._k_gas_standard(theta, z, seismic_mult)
            k_conduit = self.p.get("k_karst_conduit", 1e-10)
            f_active  = self.p.get("f_karst_active", 0.05)
            k_eff     = k_base + k_conduit * f_active
            return np.clip(k_eff, self.p.get("k_min", 1e-18),
                                   self.p.get("k_max", 1e-9))

        else:
            return self._k_gas_standard(theta, z, seismic_mult)

    def _barometric_transient_profile(self, theta, t, seismic_mult=None):
        """Depth-dependent barometric pumping perturbation [Pa] from a damped pressure-diffusion wave."""
        # atmospheric pressure oscillation diffuses into soil as a damped, phase-lagged
        # wave with skin depth delta = sqrt(2*D_P/omega) (Clements & Wilkening 1974)
        amp   = self.p.get("P_atm_amp", 150.0)
        T_a   = self.p.get("T_atm", 86400.0)
        omega = 2.0 * np.pi / max(T_a, 1.0)
        mu    = self.p.get("mu", 1.82e-5)
        Mg    = self.p.get("M_g", 0.02897); Rg = self.p.get("R_g", 8.314)
        k     = self._k_gas(theta, self.z, seismic_mult)
        phi_g = np.maximum(self.p["phi_m"] - np.asarray(theta, dtype=float), 1e-6)
        P0    = self.p.get("P_atm0", 101325.0)
        D_P   = np.maximum(k * P0 / np.maximum(phi_g * mu, 1e-30), 1e-12)
        # column-mean skin depth keeps the wave form monotone and numerically robust
        D_P_bar = float(np.mean(D_P))
        delta   = float(np.sqrt(2.0 * D_P_bar / max(omega, 1e-30)))
        delta   = max(delta, 1e-3)
        zeta    = self.z / delta
        return amp * np.exp(-zeta) * np.sin(omega * t - zeta)

    def _Pbot_transient(self, theta, t, seismic_mult=None):
        """Deep-boundary gas pressure [Pa]: mean overpressure plus the barometric
        wave attenuated and lagged over the column depth H with the pneumatic skin
        depth delta = sqrt(2 D_P / omega), D_P = k P / (mu phi_g)
        (Clements & Wilkening 1974; Elberling et al. 1998)."""
        P0   = self.p.get("P_atm0", 101325.0)
        dPb  = self.p.get("dP_bottom", 5.0)
        if not self.p.get("barometric_pumping_on", True):
            return self._Patm(t) + dPb
        amp   = self.p.get("P_atm_amp", 150.0)
        T_a   = self.p.get("T_atm", 86400.0)
        omega = 2.0 * np.pi / max(T_a, 1.0)
        mu    = self.p.get("mu", 1.82e-5)
        k     = self._k_gas(np.asarray(theta, dtype=float), self.z, seismic_mult)
        phi_g = np.maximum(self.p["phi_m"] - np.asarray(theta, dtype=float), 1e-6)
        D_P   = float(np.mean(k * P0 / np.maximum(phi_g * mu, 1e-30)))
        delta = max(np.sqrt(2.0 * D_P / omega), 1e-3)
        zeta  = self.H / delta
        return P0 + dPb + amp * np.exp(-zeta) * np.sin(omega * t - zeta)

    def gas_pressure_rhs(self, theta, Pg, t, seismic_mult=None):
        """Soil gas pressure relaxing toward a quasi-stationary 1-D Darcy profile."""
        # tau_P << tau_radon, so Pg is quasi-stationary for radon transport
        theta   = np.asarray(theta, dtype=float)
        Pg      = np.asarray(Pg,    dtype=float)
        nz      = self.nz
        mu      = self.p.get("mu", 1.82e-5)
        k       = self._k_gas(theta, self.z, seismic_mult)
        dz      = self.dz
        Patm    = self._Patm(t)
        # deep boundary: attenuated, phase-lagged barometric wave plus overpressure
        Pbot    = self._Pbot_transient(theta, t, seismic_mult)

        if nz == 1:
            Pg_ss = np.array([0.5 * (Patm + Pbot)])
            # relaxation time from single-cell pressure diffusivity D_P = k P/(mu phi_g)
            phi_g1 = max(self.p["phi_m"] - float(theta[0]), 1e-6)
            D_P = k[0] * max(float(Pg[0]), 1.0) / (mu * phi_g1)
            tau = max(dz**2 / (2.0 * D_P), 1.0)
            return (Pg_ss - Pg) / tau

        k_face = np.empty(nz + 1)
        k_face[0]  = k[0]
        k_face[-1] = k[-1]
        for i in range(1, nz):
            k_face[i] = 2.0 * k[i-1] * k[i] / max(k[i-1] + k[i], 1e-40)
        T_lam = k_face / (mu * dz)  # transmissivity [m^3/(Pa s m)] per unit length
        # boundary faces sit dz/2 from the first/last cell centre: transmissivity doubles
        T_lam[0]  *= 2.0
        T_lam[-1] *= 2.0

        a = np.zeros(nz); b = np.zeros(nz); c = np.zeros(nz); rhs = np.zeros(nz)
        for i in range(nz):
            tL = T_lam[i]; tR = T_lam[i+1]
            b[i] = -(tL + tR)
            if i > 0:     a[i] = tL
            if i < nz-1:  c[i] = tR
        # Dirichlet at top: row 0 incorporates the boundary-face value P = Patm
        rhs[0]  = -T_lam[0] * Patm
        # Dirichlet at bottom: row nz-1 incorporates the boundary-face value P = Pbot
        rhs[-1] = -T_lam[-1] * Pbot

        c_ = np.zeros(nz); d_ = np.zeros(nz)
        c_[0] = c[0] / b[0] if abs(b[0]) > 1e-40 else 0.0
        d_[0] = rhs[0] / b[0] if abs(b[0]) > 1e-40 else Patm
        for i in range(1, nz):
            denom = b[i] - a[i] * c_[i-1]
            if abs(denom) < 1e-40:
                c_[i] = 0.0; d_[i] = Pg[i]
            else:
                c_[i] = c[i] / denom
                d_[i] = (rhs[i] - a[i] * d_[i-1]) / denom
        Pg_ss = np.zeros(nz)
        Pg_ss[-1] = d_[-1]
        for i in range(nz-2, -1, -1):
            Pg_ss[i] = d_[i] - c_[i] * Pg_ss[i+1]

        P_lo = min(Patm, Pbot) - 50.0
        P_hi = max(Patm, Pbot) + 50.0
        Pg_ss = np.clip(Pg_ss, P_lo, P_hi)

        # relaxation toward the quasi-steady profile with the slowest pressure-diffusion
        # eigenmode tau = H^2/(pi^2 D_P), grid-independent; D_P = k P/(mu phi_g)
        phi_g = np.maximum(self.p["phi_m"] - theta, 1e-6)
        D_P = k * np.maximum(Pg, 1.0) / (mu * phi_g)
        tau = np.maximum(self.H**2 / (np.pi**2 * D_P), 1.0)

        return (Pg_ss - Pg) / tau


    def _Patm(self, t):
        P0  = self.p.get("P_atm0", 101325.0)
        amp = self.p.get("P_atm_amp", 150.0)
        T_a = self.p.get("T_atm", 86400.0)
        return P0 + amp * np.sin(2*np.pi*t / T_a)

    def _dPatm_dt(self, t):
        amp = self.p.get("P_atm_amp", 150.0)
        T_a = self.p.get("T_atm", 86400.0)
        return amp * (2*np.pi/T_a) * np.cos(2*np.pi*t / T_a)

    def seismic_permeability_multiplier(self, t):
        """Stochastic seismic permeability enhancement from a Poisson process of Gutenberg-Richter events."""
        # k_seis(t) = 1 + A_seis * sum_events exp(gamma*M_i) * exp(-(t-t_i)/tau_seis)
        gr_on = self.p.get("seismic_gr_magnitudes_on", True)
        if not hasattr(self, "_seis_events"):
            rng  = np.random.default_rng(self.p.get("seis_seed", 99))
            rate = self.seis["f_seismic"]
            T_sim= self.p.get("t_total_seis", 7*86400.0)
            n_ev = int(rng.poisson(rate * T_sim))   # 0 events is a valid outcome
            self._seis_events = sorted(rng.uniform(0, T_sim, n_ev))
            if gr_on:
                # bounded Gutenberg-Richter law N(>=M) ~ 10^(-b*M) on [M_min, M_max],
                # sampled by inverse-CDF in natural-log units (beta = b*ln10)
                b     = self.p.get("GR_b_value", 1.0)
                M_min = self.p.get("M_min_seis", 1.0)
                M_max = self.p.get("M_max_seis", 5.0)
                beta  = b * np.log(10.0)
                u     = rng.uniform(0.0, 1.0, n_ev)
                denom = 1.0 - np.exp(-beta * (M_max - M_min))
                self._seis_mags = M_min - np.log(1.0 - u * denom) / beta
                self._seis_mags = np.clip(self._seis_mags, M_min, M_max)
            else:
                self._seis_mags = rng.exponential(self.seis["M_ref"], n_ev)

        A      = self.seis["A_seis"]
        gamma  = self.seis["gamma_seis"]
        tau    = self.seis["tau_seis"]
        M_ref  = self.seis.get("M_ref", 2.5)
        # exponent referenced to M_ref so a median event does not saturate the ceiling
        M_shift = M_ref if gr_on else 0.0
        mult   = 1.0
        for t_ev, M in zip(self._seis_events, self._seis_mags):
            if t >= t_ev:
                dt   = t - t_ev
                mult += A * np.exp(gamma * (M - M_shift)) * np.exp(-dt / max(tau, 1e-12))
        return np.clip(mult, 1.0, self.p.get("k_seis_max", 50.0)) * np.ones(self.nz)

    def seismic_permeability_multiplier_recovery(self, t, t0=None):
        """Deterministic co-/post-seismic permeability recovery k(t) toward unity."""
        # k(t) = 1 + (k_peak-1)*exp(-(t-t0)/tau) (Elkhoury 2006; Manga 2012)
        if not self.p.get("seismic_recovery_on", False):
            return np.ones(self.nz)
        t0 = float(self.p.get("seis_t0_s", 0.0)) if t0 is None else float(t0)
        # peak enhancement: explicit, else derived from unrest magnitude via Biot/Heap law
        if "k_seis_peak" in self.p:
            k_peak = float(self.p["k_seis_peak"])
        else:
            M_w = float(self.p.get("M_w_unrest", 0.0))
            A   = float(self.seis.get("A_seis", 0.30))
            g   = float(self.seis.get("gamma_seis", 0.90))
            k_peak = 1.0 + A * math.exp(g * M_w)
        tau = float(self.p.get(
            "tau_seis_recovery",
            _tau_seis_recovery_seconds(site=self.p.get("seismic_site"))))
        k_floor = float(self.p.get("k_seis_floor", 1.0))
        k_t = seismic_recovery_factor(t, t0, k_peak, tau, k_floor=k_floor)
        k_t = float(np.clip(k_t, 1.0, self.p.get("k_seis_max", 50.0)))
        return k_t * np.ones(self.nz)

    def _Se_wetting(self, theta):
        """Wetting-branch effective saturation (diagnostics only).

        NOTE: retention hysteresis is NOT implemented in the transport solver;
        the wetting-branch van Genuchten parameters (_vg_alpha_w, _vg_n_w) are
        reserved for future use and all flow computations use the drainage branch."""
        return self._Se_drainage(theta)

    def hydraulic_hysteresis_indicator(self, theta, dtheta_dt):
        """Returns 1 on the drainage branch, 0 on the wetting branch."""
        on_drainage = (np.mean(dtheta_dt) <= 0.0)
        return 1.0 if on_drainage else 0.0

    def _fracture_activation_metric(self, theta, Pg):
        """Fracture activation contribution to pore accessibility, chi(z)."""
        theta  = np.asarray(theta, dtype=float)
        Pg     = np.asarray(Pg,    dtype=float)
        phi    = self.p["phi_m"]
        Sw     = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        chi    = np.zeros(self.nz)
        for crack in self.cracks:
            a     = crack.get("aperture", crack.get("aperture_0", 1e-4))
            a0    = crack.get("aperture_0", a)
            conn  = self._crack_connectivity_scalar(Sw, crack)
            # depth attenuation lives ONLY in _crack_connectivity_scalar (dep_f);
            # repeating it here squared the weighting (e-folding 0.4 -> 0.2 m)
            chi  += conn * np.clip(a / max(a0, 1e-12), 0.1, 10.0)
        return np.nan_to_num(chi / max(len(self.cracks), 1), nan=0.0)

    def radon_accessibility_rhs(self, Ar, theta, dtheta_dt, Pg):
        """Pore accessibility relaxation: opening on drying/fracturing, closing on rewetting."""
        Ar         = np.asarray(Ar,         dtype=float)
        theta      = np.asarray(theta,      dtype=float)
        dtheta_dt  = np.asarray(dtheta_dt,  dtype=float)
        Pg         = np.asarray(Pg,         dtype=float)
        phi        = self.p["phi_m"]
        Sw         = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        Sg         = np.clip(1.0 - Sw, 1e-12, 1.0)
        drying     = np.maximum(-dtheta_dt, 0.0)
        wetting    = np.maximum( dtheta_dt, 0.0)
        # gate switches 0->1 across dry_crit only if beta_d*dry_crit >> 1;
        # beta_d = 5e8 gives sigmoid arguments of +/-5 for drying = 0 or 2*dry_crit
        beta_d     = self.p.get("A_beta_dry",    5e8)
        dry_crit   = self.p.get("A_dtheta_crit", 1e-8)
        dry_gate   = 1.0 / (1.0 + np.exp(-np.clip(beta_d * (drying - dry_crit), -50.0, 50.0)))
        ng         = self.p.get("A_ng",  2.0)
        nw         = self.p.get("A_nw",  2.0)
        chi_f      = self._fracture_activation_metric(theta, Pg)
        beta_f     = self.p.get("A_beta_frac", 1.5)
        gradPg     = np.abs(np.gradient(Pg, self.dz)) if len(Pg) > 1 else np.zeros(len(Pg))
        beta_pg    = self.p.get("A_beta_gradPg", 0.0)
        pg_ref     = self.p.get("A_gradPg_ref",  100.0)
        press_gate = 1.0 + beta_pg * gradPg / max(pg_ref, 1e-12)
        k_open     = self.p.get("A_k_open",   1.0/3600.0)
        k_close    = self.p.get("A_k_close",  0.35/3600.0)
        beta_rewet = self.p.get("A_beta_rewet", 1e6)
        phi_open   = (Sg**ng) * dry_gate * (1.0 + beta_f * chi_f) * press_gate
        phi_close  = (Sw**nw) * (1.0 + beta_rewet * wetting)
        # physical floor: gas-connected pore space stays accessible regardless of
        # the drying/rewetting transients — rewetting can only close the EXTRA
        # accessibility above the gas-filled fraction Sg, never drive Ar to zero
        Ar_base    = np.clip(Sg, 0.0, 1.0)
        dAr_dt     = (k_open * phi_open * (1.0 - Ar)
                      - k_close * phi_close * np.maximum(Ar - Ar_base, 0.0)
                      + k_open * np.maximum(Ar_base - Ar, 0.0))
        return np.clip(dAr_dt, -1e3, 1e3)

    def emanation_hysteresis_rhs(self, eps_mem, theta, dtheta_dt, T):
        """Emanation-coefficient memory relaxing toward its instantaneous value with drying/wetting timescales."""
        # tau_hyst is short on drying (pore opening) and long on rewetting
        eps_inst   = self._emanation_factor_inst(theta, T)
        tau_dry    = self.p.get("tau_hyst_dry",  2 * 3600.0)     # fast on drying
        tau_wet    = self.p.get("tau_hyst_wet", 48 * 3600.0)     # slow on rewetting
        # per-cell drying/wetting decision keeps hysteresis regime depth-resolved
        if self.p.get("hysteresis_per_cell", True):
            dtheta_dt = np.asarray(dtheta_dt, dtype=float)
            tau_h = np.where(dtheta_dt <= 0.0, tau_dry, tau_wet)
            return (eps_inst - eps_mem) / np.maximum(tau_h, 1e-12)
        else:
            drying = (np.mean(dtheta_dt) <= 0.0)
            tau_h  = tau_dry if drying else tau_wet
            return (eps_inst - eps_mem) / max(tau_h, 1e-12)


    def _emanation_vesicular(self, theta, T):
        """Radon emanation coefficient for vesicular volcanic materials."""
        theta   = np.asarray(theta, dtype=float)
        T       = np.asarray(T,     dtype=float)
        phi     = self.p["phi_m"]
        theta   = np.clip(theta, 1e-12, 0.9999*phi)
        Sw      = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        eps_max = self.p.get("eps_max",       0.30)
        Sw_pk   = self.p.get("Sw_peak_eman",  0.22)
        sigma_s = self.p.get("Sw_sigma_eman", 0.14)
        eps_dry = self.p.get("eps_dry",        0.06)
        bump    = np.exp(-0.5*((Sw - Sw_pk) / max(sigma_s, 1e-6))**2)
        eps_sw  = eps_dry + (eps_max - eps_dry) * bump
        Sw_blk  = self.p.get("Sw_block_eman",  0.90)
        blk_sh  = self.p.get("Sw_block_sharp",  0.03)
        block   = 1.0 / (1.0 + np.exp((Sw - Sw_blk) / max(blk_sh, 1e-6)))
        T_ref   = self.p.get("T_ref", 293.15)
        beta_T  = self.p.get("beta_T_eman", 0.003)
        temp_f  = np.clip(np.exp(beta_T * (T - T_ref)), 0.8, 1.25)
        g_mm    = self.p.get("grain_size_mm",  0.2)
        g_ref   = self.p.get("grain_ref_mm",  0.25)
        g_exp   = self.p.get("grain_exp_eman", 0.20)
        grain_f = np.clip((g_ref / max(g_mm, 1e-6))**g_exp, 0.7, 1.6)
        return np.clip(eps_sw * block * temp_f * grain_f, 0.01, 0.80)

    def _emanation_granular(self, theta, T):
        """Emanation coefficient for granular soils (sand, clay, silt).
        Grain-size dependence per Rogers & Nielson 1991, Health Phys. 60:807."""
        theta   = np.asarray(theta, dtype=float)
        T       = np.asarray(T,     dtype=float)
        phi     = self.p["phi_m"]
        theta   = np.clip(theta, 1e-12, 0.9999*phi)
        Sw      = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        grain_mm = self.p.get("grain_size_mm", 0.25)
        eps_dry  = self.p.get("eps_dry",  0.06)
        # eps_sat scales inversely with grain size
        eps_sat  = 0.50 * (0.1 / max(grain_mm, 0.005))**0.30
        eps_sat  = float(np.clip(eps_sat, 0.15, 0.52))
        # Sw_sat: saturation threshold for the water film onset
        Sw_sat   = float(np.clip(0.04 + 0.12 * grain_mm, 0.01, 0.20))
        f_moist  = 1.0 - np.exp(-Sw / max(Sw_sat, 1e-6))
        eps_sw   = eps_dry + (eps_sat - eps_dry) * f_moist
        block    = 1.0 / (1.0 + np.exp((Sw - 0.85) / 0.03))
        T_ref    = self.p.get("T_ref", 293.15)
        beta_T   = self.p.get("beta_T_eman", 0.003)
        temp_f   = np.clip(np.exp(beta_T * (T - T_ref)), 0.8, 1.25)
        return np.clip(eps_sw * block * temp_f, 0.01, 0.60)

    def _emanation_crystalline(self, theta, T):
        """Emanation coefficient for crystalline rocks (granite, gneiss).
        Low and near-constant; microfractures dominate (Semkow 1990)."""
        theta   = np.asarray(theta, dtype=float)
        T       = np.asarray(T,     dtype=float)
        phi     = self.p["phi_m"]
        theta   = np.clip(theta, 1e-12, 0.9999*phi)
        Sw      = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        eps_base = self.p.get("eps_crystalline", 0.08)
        eps_sw   = eps_base * (1.0 + 0.05 * Sw)
        T_ref    = self.p.get("T_ref", 293.15)
        beta_T   = self.p.get("beta_T_eman", 0.003)
        temp_f   = np.clip(np.exp(beta_T * (T - T_ref)), 0.8, 1.25)
        return np.clip(eps_sw * temp_f, 0.01, 0.25)

    def _emanation_karst(self, theta, T):
        """Emanation coefficient for karst limestone, sourced from terra rossa clays.
        Ra-226 is concentrated in conduit clays, not the limestone matrix."""
        theta   = np.asarray(theta, dtype=float)
        T       = np.asarray(T,     dtype=float)
        phi     = self.p["phi_m"]
        theta   = np.clip(theta, 1e-12, 0.9999*phi)
        Sw      = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        eps_base = self.p.get("eps_karst", 0.20)
        f_wash   = 1.0 + 0.30 * np.clip(Sw - 0.50, 0.0, 1.0)
        block    = 1.0 / (1.0 + np.exp((Sw - 0.88) / 0.03))
        T_ref    = self.p.get("T_ref", 293.15)
        beta_T   = self.p.get("beta_T_eman", 0.003)
        temp_f   = np.clip(np.exp(beta_T * (T - T_ref)), 0.8, 1.25)
        return np.clip(eps_base * f_wash * block * temp_f, 0.05, 0.45)

    def _moisture_emanation_factor(self, Sw):
        """Dimensionless emanation multiplier, unity at the moisture peak.
        Peaks near Sw~0.15-0.20 then falls via capillary blocking (Rogers & Nielson 1991)."""
        Sw      = np.clip(np.asarray(Sw, dtype=float), 0.0, 1.0)
        Sw_pk   = self.p.get("Sw_peak_moist",   0.17)   # 15-20% documented peak
        sigma   = self.p.get("Sw_sigma_moist",  0.12)
        f_dry   = self.p.get("f_emanation_dry", 0.45)   # dry value relative to peak
        Sw_blk  = self.p.get("Sw_block_moist",  0.88)
        blk_sh  = self.p.get("Sw_block_sharp_moist", 0.04)
        bump    = np.exp(-((Sw - Sw_pk) / max(sigma, 1e-6))**2)
        rise    = f_dry + (1.0 - f_dry) * bump
        block   = 1.0 / (1.0 + np.exp((Sw - Sw_blk) / max(blk_sh, 1e-6)))
        # normalise so the factor equals 1 exactly at the documented peak
        norm_pk = (f_dry + (1.0 - f_dry)) / (1.0 + np.exp((Sw_pk - Sw_blk) / max(blk_sh, 1e-6)))
        f = rise * block / max(norm_pk, 1e-9)
        return np.clip(f, 0.05, 1.05)

    def _moisture_diffusivity_choke(self, Sw):
        """Diffusivity choke multiplier from gas-phase percolation cutoff near saturation.
        Refines Millington-Quirk near Sw_perc (Rogers & Nielson 1991; Moldrup et al. 2000)."""
        Sw      = np.clip(np.asarray(Sw, dtype=float), 0.0, 0.9999)
        Sw_perc = self.p.get("Sw_perc_gas", 0.95)   # gas-phase percolation cutoff
        p_exp   = self.p.get("diff_choke_exp", 3.0)
        chk_min = self.p.get("diff_choke_min", 1e-4)
        x       = np.clip(1.0 - Sw / max(Sw_perc, 1e-6), 0.0, 1.0)
        return np.clip(x**p_exp, chk_min, 1.0)

    # Zhuo et al. (2008) texture parameters as tabulated (with misprint
    # corrected) by Griffiths et al. (2010, ACP 10:8969), Table 2:
    # f = f0*{1 + a*[1 - exp(-b*m)]}*[1 + c*(T-298)], m = water saturation
    _ZHUO_PARAMS = {
        "clay": (0.18, 1.53, 21.8, 0.011),
        "silt": (0.14, 1.73, 20.5, 0.010),
        "sand": (0.10, 1.85, 18.8, 0.012),
    }

    def _emanation_zhuo(self, theta, T):
        """Field-effective emanation fraction, Zhuo et al. (2008) formulation.

        f(m,T) = f0*{1 + a*[1 - exp(-b*m)]}*[1 + c*(T - 298)] with f0 the
        DRY-soil emanation and m the water saturation. Dry soil emanates a
        factor (1+a) ~ 2.5-2.9 less than soil beyond ~10% saturation
        (Zhuo et al. 2006; Sakoda et al. 2010; Griffiths et al. 2010).

        f0 resolution order:
        1. p['eps_dry_zhuo'] if given;
        2. from a measured emanation p['eps_meas'] taken at the declared
           reference state p['eps_meas_state'] ('dry' or 'plateau') and, for
           'plateau', de-rated by the plateau factor (1+a);
        3. the Zhuo texture default f0.
        Texture from p['zhuo_texture'] ('clay'|'silt'|'sand'), else inferred
        from grain_size_mm with the grain-size cuts of Griffiths et al.
        (2010) Table 2 (clay <= 0.002 mm, silt 0.002-0.5 mm, sand 0.5-2 mm)."""
        theta = np.asarray(theta, dtype=float)
        T     = np.asarray(T,     dtype=float)
        phi   = self.p["phi_m"]
        # moisture variable convention: Griffiths et al. (2010) state m is the
        # "fraction of saturation"; the primary Zhuo/Zhang tables have also been
        # read as GRAVIMETRIC water content (kg/kg), which stretches the b
        # coefficients by rho_w/rho_b*phi (~10% effect on eps). Default follows
        # the open-access Griffiths reading; set zhuo_moisture_variable =
        # "gravimetric" to use m = theta*rho_w/rho_b instead.
        if self.p.get("zhuo_moisture_variable", "saturation") == "gravimetric":
            m = np.clip(theta * 1000.0 / max(self.p.get("rho_bulk", 1600.0), 1.0), 0.0, 1.0)
        else:
            m = np.clip(theta / max(phi, 1e-12), 0.0, 1.0)

        tex = self.p.get("zhuo_texture")
        if tex not in self._ZHUO_PARAMS:
            g_mm = float(self.p.get("grain_size_mm", 0.2))
            tex = "clay" if g_mm <= 0.002 else ("silt" if g_mm < 0.5 else "sand")
        f0_tab, a, b, c = self._ZHUO_PARAMS[tex]

        if "eps_dry_zhuo" in self.p:
            f0 = float(self.p["eps_dry_zhuo"])
        elif "eps_meas" in self.p:
            eps_meas = float(self.p["eps_meas"])
            # closed-porosity (vesicle) correction for CRUSHED laboratory samples:
            # mechanical crushing connects isolated micropores/vesicles to the
            # permeable network (Nicolas et al. 2014, GRL 41; intraparticle-
            # porosity control on emanation: Rood et al. 1996, ES&T 30), so the
            # lab value overstates the field-effective emanation of vesicular
            # volcanic clasts by the closed-pore contribution. f_closed MUST be
            # measured (crushed vs intact aliquots); default 0 = no correction.
            f_closed = float(np.clip(self.p.get("eps_closed_pore_fraction", 0.0), 0.0, 0.95))
            eps_meas = eps_meas * (1.0 - f_closed)
            state    = str(self.p.get("eps_meas_state", "plateau"))
            f0 = eps_meas if state == "dry" else eps_meas / (1.0 + a)
        else:
            f0 = f0_tab

        f = f0 * (1.0 + a * (1.0 - np.exp(-b * m))) * (1.0 + c * (T - 298.0))
        # clip to the SAME range as the eps_mem state and the source terms,
        # otherwise the hysteresis ODE chases an unreachable target forever
        return np.clip(f, 0.01, 0.80)

    def _emanation_factor_inst(self, theta, T):
        """Selects the emanation submodel based on lithology."""
        eman_model = self.p.get("emanation_model", "volcanic_vesicular")
        if eman_model == "grain_size_moisture":
            eps = self._emanation_granular(theta, T)
        elif eman_model == "crystalline_microfracture":
            eps = self._emanation_crystalline(theta, T)
        elif eman_model == "karst_terra_rossa":
            eps = self._emanation_karst(theta, T)
        elif eman_model == "zhuo_moisture":
            eps = self._emanation_zhuo(theta, T)
        else:
            eps = self._emanation_vesicular(theta, T)

        # optional unity-centred moisture-peak refinement (Rogers & Nielson 1991)
        if self.p.get("moisture_emanation_on", False):
            phi = self.p["phi_m"]
            Sw  = np.clip(np.asarray(theta, dtype=float) / max(phi, 1e-12), 1e-12, 0.9999)
            eps = eps * self._moisture_emanation_factor(Sw)
            eps = np.clip(eps, 0.005, 0.80)
        return eps

    def source_rn222(self, theta, T, Ar, eps_mem):
        """Rn-222 volumetric source term S_222 [Bq/m^3_gas/s] with gas-water partitioning."""
        theta    = np.asarray(theta,   dtype=float)
        T        = np.asarray(T,       dtype=float)
        Ar       = np.clip(np.asarray(Ar,     dtype=float), 0.0, 1.0)
        eps_mem  = np.clip(np.asarray(eps_mem, dtype=float), 0.01, 0.80)
        phi      = self.p["phi_m"]
        theta    = np.clip(theta, 1e-12, 0.9999*phi)
        Sw       = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        phi_g    = np.maximum(phi - theta, 1e-12)
        phi_w    = np.maximum(theta,        1e-12)
        H        = self._henry_rn(T)
        A_part   = np.maximum(phi_g + H * phi_w, 1e-12)
        rho_b    = self.p.get("rho_bulk",   1600.0)
        Ra226    = self.p.get("Ra226_Bqkg",   35.0)
        A_bulk   = rho_b * Ra226 * eps_mem * Ar   # emanated activity concentration [Bq/m^3_bulk]
        # production rate per unit PORE-AIR volume, partition-corrected:
        # S = lambda * rho_b * A_Ra * eps / (phi_g + L*phi_w)  [Bq/(m^3_gas s)]
        S_gas    = self.lam_rn * A_bulk / A_part
        return np.maximum(S_gas, 0.0)

    def source_rn220(self, theta, T, Ar, eps_mem):
        """Rn-220 (thoron) volumetric source term [Bq/m^3_gas/s].
        Short half-life gives diffusion length L_220 = L_222/sqrt(lam220/lam222)
        = L_222/77, i.e. ~1-2 cm for L_222 ~ 1-1.5 m."""
        theta    = np.asarray(theta,  dtype=float)
        T        = np.asarray(T,      dtype=float)
        Ar       = np.clip(np.asarray(Ar,    dtype=float), 0.0, 1.0)
        eps_mem  = np.clip(np.asarray(eps_mem,dtype=float), 0.01, 0.80)
        phi      = self.p["phi_m"]
        theta    = np.clip(theta, 1e-12, 0.9999*phi)
        phi_g    = np.maximum(phi - theta, 1e-12)
        phi_w    = np.maximum(theta,        1e-12)
        H        = self._henry_rn(T)
        A_part   = np.maximum(phi_g + H * phi_w, 1e-12)
        rho_b    = self.p.get("rho_bulk",   1600.0)
        Th232    = self.p.get("Th232_Bqkg",   25.0)
        # recoil emanation of Rn-220 is comparable to Rn-222 (similar recoil range;
        # Sakoda et al. 2011, Appl. Radiat. Isot. 69:1422); measured ratios 0.5-1
        eps_220  = eps_mem * self.p.get("eps_ratio_220_222", 1.0)
        S_gas    = self.lam_rn220 * rho_b * Th232 * eps_220 * Ar / A_part
        return np.maximum(S_gas, 0.0)

    def thoron_flux_analytic(self, theta, T):
        """Analytic Rn-220 (thoron) surface flux [Bq/m^2/s], subgrid-exact.

        The thoron diffusion length L_220 = sqrt(D_a/lambda_220) is ~1-2 cm —
        unresolvable by any practical grid (dz >= 5-20 cm), so the PDE surface
        flux for Rn-220 is systematically grid-limited. Because lambda_220 >>
        every other rate in the column, thoron is in LOCAL steady state and the
        canonical semi-infinite solution is exact to O(dz/L_220 -> infinity):

            J_220 = rho_b * A_Th232 * eps_220 * sqrt(lambda_220 * D_a)

        with D_a the apparent diffusivity evaluated from the near-surface state.
        Diagnostic use: the measurable flux ratio
        J_220/J_222 = (A_Th*eps220)/(A_Ra*eps222) * sqrt(lambda_220/lambda_222)
        (sqrt factor ~ 77) constrains the emanation state in the field —
        e.g. it discriminates the dried-sample vs plateau reading of laboratory
        emanation coefficients independently of the source terms."""
        theta = np.asarray(theta, dtype=float)
        T     = np.asarray(T,     dtype=float)
        phi   = self.p["phi_m"]
        # near-surface state controls the emitting layer (top few cm)
        th0   = float(theta[0]); T0 = float(T[0])
        L_T   = float(self._henry_rn(np.asarray([T0]))[0])
        beta0 = max(phi - th0, 1e-6) + L_T * th0
        D_e0  = float(self._D_eff(theta, "m")[0])
        D_a0  = D_e0 / beta0
        rho_b = self.p.get("rho_bulk", 1600.0)
        A_Th  = self.p.get("Th232_Bqkg", 25.0)
        eps0  = float(np.atleast_1d(self._emanation_factor_inst(theta, T))[0])
        eps220 = eps0 * self.p.get("eps_ratio_220_222", 1.0)
        return rho_b * A_Th * eps220 * float(np.sqrt(self.lam_rn220 * D_a0))

    def _henry_rn(self, T):
        """Radon water/gas partition (Ostwald) coefficient L(T) = Cw/Cg, dimensionless.

        Van 't Hoff form L = L0*exp(B*(1/T - 1/T0)) with B = -dH_sol/R > 0 because
        dissolution is exothermic: solubility INCREASES as T decreases.
        B ~ +2790 K reproduces Weigel (1978): L(0 C)=0.51, L(20 C)=0.25, L(30 C)=0.20."""
        H0  = self.p.get("H_rn0",  0.25)
        dHR = self.p.get("dH_rn", 2790.0)   # -dH_sol/R [K], positive (exothermic dissolution)
        T0  = 293.15
        return H0 * np.exp(dHR * (1.0 / np.maximum(np.asarray(T, dtype=float), 1.0) - 1.0/T0))

    def _D_eff_millington_quirk(self, theta, domain="m"):
        """BULK effective radon diffusivity D_e [m^2/s] for the given domain.

        The Rogers & Nielson (1991, Health Phys. 61:225) correlation

            D_a = phi * D0 * exp(-6*Sw*phi - 6*Sw^(14*phi))

        is the APPARENT (pore/interstitial) diffusion coefficient: it is the one
        whose diffusion length is L_d = sqrt(D_a/lambda) (~1.5 m for dry soil)
        and whose canonical flux is J = rho_b*A_Ra*eps*sqrt(lambda*D_a) with no
        porosity factor (RAECOM, NUREG/CR-3533; UNSCEAR 2000 Annex B para. 98;
        Griffiths et al. 2010, Eqs. 3-6). Its dry limit phi*D0 corresponds to a
        physical tortuosity tau = phi; reading it as bulk would imply tau = 1.

        This model's transport is written in partition form
        beta*dC/dt = d/dz(D_e dC/dz) + ... with beta = phi_g + L(T)*theta, which
        requires the BULK coefficient D_e = beta * D_a (UNSCEAR 2000 Annex B
        para. 97: bulk = (eps_a + K_T*eps_w)*De; RAECOM Eq. 21:
        a_i = p_i*D_i*[1-(1-k)*m_i]). The Dmin/Dmax clip applies to the APPARENT
        value (Dmax ~ free-air ceiling) BEFORE the beta factor. With this pairing
        the model reproduces exactly L_d = sqrt(D_a/lambda) and
        J = rho*A*eps*sqrt(lambda*D_a); typical soils give 15-25 mBq/m^2/s,
        against the UNSCEAR worldwide MEAN of 16 mBq/m^2/s (33 mBq/m^2/s is
        UNSCEAR's dry-soil worked example, NOT the world mean)."""
        theta  = np.asarray(theta, dtype=float)
        phi    = self.p["phi_m"] if domain == "m" else self.p["phi_f"]
        Sw     = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        D0     = self.p.get("D0", 1.1e-5)
        # kinetic-theory temperature scaling D0 ~ T^1.5 (Schery & Wasiolek 1998;
        # Griffiths et al. 2010, Eq. 6); D0 is referenced to T_D0_ref (20 C)
        T_key = (self.p.get("T0", 288.0), self.p.get("T_grad", 0.025))
        if getattr(self, "_T_D0_key", None) != T_key:
            self._T_D0_cache = float(np.mean(self.T_field(0.0)))
            self._T_D0_key   = T_key
        T_mean = self._T_D0_cache
        if self.p.get("D0_T_scaling_on", True):
            T_ref_D = self.p.get("T_D0_ref", 293.15)
            D0 = D0 * (max(T_mean, 200.0) / T_ref_D) ** 1.5
        Dmin   = self.p.get("Dmin", 1e-12)
        Dmax   = self.p.get("Dmax", 1.2e-5)
        D_a    = phi * D0 * np.exp(-6.0 * Sw * phi - 6.0 * Sw**(14.0 * phi))
        # optional extra near-saturation percolation cutoff (off by default: the
        # second exponential of Rogers & Nielson already collapses D_a near Sw=1)
        if self.p.get("moisture_diffusivity_on", False):
            D_a = Dmin + (D_a - Dmin) * self._moisture_diffusivity_choke(Sw)
        D_a = np.clip(D_a, Dmin, Dmax)
        # bulk coefficient for the partition-form transport equations
        L_T    = float(np.mean(self._henry_rn(np.asarray([T_mean]))))
        beta_d = np.maximum(phi - theta, 1e-6) + L_T * theta
        return beta_d * D_a

    def _D_eff(self, theta, domain="m"):
        """Effective gas diffusivity [m^2/s], branched by transport regime.
        fracture_dominant uses Semkow 1990; karst_conduit uses Spitieri et al. 2008."""
        regime = self.p.get("regime", "porous_diffusive")

        if regime == "fracture_dominant":
            if domain == "f":
                # D_fracture is the open-fracture PORE diffusivity (~free air
                # reduced by tortuosity); the fracture continuum occupies only
                # phi_f of the bulk cross-section, so the BULK coefficient is
                # beta_f * D_fracture — consistent with the beta_f storage the
                # transport equations use (an unscaled value overstated the
                # crystalline-lithotype flux by ~1/phi_f, up to ~150%)
                theta_arr = np.asarray(theta, dtype=float)
                phi_f     = self.p["phi_f"]
                if not hasattr(self, "_T_D0_cache"):
                    self._T_D0_cache = float(np.mean(self.T_field(0.0)))
                L_T    = float(self._henry_rn(np.asarray([self._T_D0_cache]))[0])
                beta_f = np.maximum(phi_f - theta_arr, 1e-6) + L_T * theta_arr
                return beta_f * self.p.get("D_fracture", 5e-6)
            # crystalline matrix: measured BULK effective value (Semkow 1990)
            D_matrix = self.p.get("D_matrix_crystalline", 1e-11)
            return np.full(self.nz, D_matrix)

        elif regime == "karst_conduit":
            f_conduit = self.p.get("f_karst_conduit", 0.08)   # conduit area fraction
            if domain == "f":
                # the conduit network IS the fracture continuum: area-weighted
                # bulk value (assigning it to one continuum only — adding it to
                # both double-counted the conduit cross-section)
                D_conduit = self.p.get("D_conduit", 5e-6)     # pore diffusivity in conduits
                return np.full(self.nz, D_conduit * f_conduit)
            # matrix continuum: R&N bulk over its (1 - f_conduit) share
            return self._D_eff_millington_quirk(theta, "m") * (1.0 - f_conduit)

        else:
            return self._D_eff_millington_quirk(theta, domain)

    def gas_water_exchange(self, Cg, Cw, theta, T, domain="m"):
        """Kinetic gas-water radon exchange rate (DIAGNOSTIC; not called by the
        solver — dissolved radon is handled implicitly via the partition
        porosity beta in the transport equations)."""
        phi  = self.p["phi_m"] if domain == "m" else self.p["phi_f"]
        Sw   = np.clip(theta / max(phi, 1e-12), 1e-12, 0.9999)
        Sg   = 1.0 - Sw
        H    = self._henry_rn(T)
        Cw_eq= H * Cg
        a_int= self.p.get("a_int0", 1200.0) * (Sw**2.1) * (Sg**1.7)
        kla  = self.p.get("k_la0",  3.2e-5) * a_int * (1.0 + 0.8*np.abs(Sw - 0.5))
        return kla * (Cw_eq - Cw)

    def memory_kernel(self, s):
        """Tempered power-law (Gamma-type) memory kernel K(s) ~ s^(nu-1) e^(-s/tau).

        NOTE: this is a truncated power-law kernel, not a Mittag-Leffler function;
        it approximates fractional-order relaxation at s << tau. Diagnostic only —
        the transport solver uses the exponential memory state M (exchange_memory_state_rhs)."""
        nu  = self.p.get("mem_nu",  0.70)
        tau = self.p.get("mem_tau", 6*3600.0)
        a   = self.p.get("mem_alpha", 1.0/3600.0)
        s   = np.maximum(np.asarray(s, dtype=float), 0.0)
        return (a**nu) * (s**(nu-1.0)) * np.exp(-s / max(tau, 1e-12)) / gamma(nu)

    def exchange_memory_state_rhs(self, M, Cm, Cf):
        """Memory-state ODE dM/dt = (Cm - Cf - M) / tau_m."""
        tau = self.p.get("mem_tau", 6*3600.0)
        return (Cm - Cf - M) / max(tau, 1e-12)

    def matrix_fracture_flux(self, Cm, Cf, M, theta_m, theta_f, Pg_m, Pg_f):
        """Matrix-to-fracture radon flux, positive when net flow is matrix to fracture."""
        phi_m     = self.p["phi_m"]
        phi_f     = self.p["phi_f"]
        Swm       = np.clip(theta_m / max(phi_m, 1e-12), 1e-12, 0.9999)
        Swf       = np.clip(theta_f / max(phi_f, 1e-12), 1e-12, 0.9999)
        sat_f     = np.clip((1.0 - 0.5*(Swm+Swf))**self.p.get("n_mf_sat", 2.0), 0.0, 1.0)
        press_f   = 1.0 + self.p.get("beta_mf_p", 1e-6) * np.abs(Pg_m - Pg_f)
        Dm        = self._D_eff(theta_m, "m")
        Df        = self._D_eff(theta_f, "f")
        Dref      = 0.5*(np.mean(np.atleast_1d(Dm)) + np.mean(np.atleast_1d(Df)))
        alpha0    = self.p.get("a_mf0", 1.0) * sat_f * press_f * Dref / max(self.p.get("d_mf", 0.02), 1e-12)
        eta_inst  = self.p.get("eta_inst", 1.0)
        eta_mem   = self.p.get("eta_mem",  0.5)
        return alpha0 * (eta_inst*(Cm-Cf) + eta_mem*M)

    def _ads_rates(self, theta):
        """Saturation-dependent adsorption/desorption rate constants [1/s].
        Shared by the Cs and Cm equations so the exchange conserves mass.

        Defaults give an equilibrium adsorbed/gas ratio Kd = k_ads/(k_des+lambda)
        ~ 0.02: radon adsorption on MINERAL soil grains is negligible at ambient
        temperature (Nazaroff 1992; significant only for organic/charcoal-rich
        media). Earlier defaults implied Kd ~ 0.87, i.e. a hidden permanent sink
        of ~0.87*lambda that suppressed the surface flux by ~27%. For organic
        soils raise k_ads_dry (Kd ~ 0.5-2) explicitly and cite accordingly."""
        phi    = self.p["phi_m"]
        Sw     = np.clip(np.asarray(theta, dtype=float) / max(phi, 1e-12), 1e-12, 0.9999)
        f_ads  = 1.0 / (1.0 + np.exp(-10.0*(Sw - 0.25)))
        k_ads  = self.p.get("k_ads_dry", 1e-7) + (self.p.get("k_ads_wet", 5e-8) - self.p.get("k_ads_dry", 1e-7)) * f_ads
        k_des  = self.p.get("k_des_dry", 2e-6) + (self.p.get("k_des_wet", 4e-6) - self.p.get("k_des_dry", 2e-6)) * f_ads
        return k_ads, k_des

    def adsorption_dynamic_rhs(self, Cs, Cm, theta):
        """Adsorption/desorption ODE for radon on solid surfaces."""
        k_ads, k_des = self._ads_rates(theta)
        return k_ads * Cm - k_des * Cs - self.lam_rn * Cs

    def _build_dynamic_fracture_network(self, n_cracks=18, seed=42):
        """Builds a stochastic fracture network with log-normal apertures.

        The median aperture (2.4e-4 m) is CALIBRATED so that the tuff_unwelded
        reference configuration (closed house, ACH = 0.1 1/h) reproduces the
        literature indoor-air/soil-gas activity ratio C_in/C_soil ~ 5e-3
        (Nazaroff 1992, Rev. Geophys. 30:137, typical range 1e-3-1e-2).
        Recalibrate against measured indoor data when site observations exist."""
        rng = np.random.default_rng(seed)
        b0_med = self.p.get("crack_b0_median", 2.4e-4)
        self.cracks = []
        for i in range(n_cracks):
            b0 = rng.lognormal(np.log(b0_med), 0.85)
            self.cracks.append({
                "id":                  i,
                "aperture_0":          float(b0),
                "aperture":            float(b0),
                "current_aperture":    float(b0),
                "length":              float(rng.uniform(0.5, 3.5)),
                "depth":               float(rng.uniform(0.02, 0.85)),
                "angle":               float(rng.uniform(0.0, np.pi)),
                "tortuosity":          float(rng.uniform(1.1, 2.6)),
                "Sw_crit":             float(rng.uniform(0.25, 0.58)),
                "memory_tau":          float(rng.uniform(2*3600, 24*3600)),
                "stress_sensitivity":  float(rng.uniform(0.8, 2.2)),
                "beta_sw":             float(rng.uniform(12.0, 24.0)),
                "conn_factor":         float(rng.uniform(0.55, 1.45)),
                "roughness":           float(rng.uniform(1.0, 2.5)),
                "Re_crit":             float(rng.uniform(1200.0, 2400.0)),
                "seismic_coupling":    float(rng.uniform(0.5, 2.0)),
                "aperture_history":    [],
            })

    def update_fracture_aperture(self, theta, Pg, t, seismic_mult=None):
        """Kelvin-Voigt viscoelastic relaxation of fracture aperture toward b_target.

        The relaxation uses the elapsed simulation time since the last update
        (relax = 1 - exp(-dt/tau)); backward/repeated solver trial times give
        dt = 0 and leave the state unchanged, which keeps this state update
        approximately consistent inside an adaptive ODE solver."""
        phi      = self.p["phi_m"]
        Sw_mean  = float(np.mean(np.clip(np.asarray(theta)/max(phi,1e-12), 1e-12, 0.9999)))
        Pg_mean  = float(np.mean(np.asarray(Pg)))
        t_last   = getattr(self, "_t_aperture_last", None)
        dt       = max(t - t_last, 0.0) if t_last is not None else 0.0
        self._t_aperture_last = max(t, t_last) if t_last is not None else t
        for crack in self.cracks:
            sat_f    = 1.0 / (1.0 + np.exp(np.clip(18.0*(Sw_mean - crack["Sw_crit"]), -50.0, 50.0)))
            dP       = abs(Pg_mean - self.p.get("P_atm0", 101325.0))
            press_f  = 1.0 + crack["stress_sensitivity"] * (dP / 5000.0)
            b_target = crack["aperture_0"] * sat_f * press_f
            # seismic events cause a transient aperture increase
            if seismic_mult is not None:
                s_mult = float(np.mean(seismic_mult))
                b_target = b_target * (1.0 + crack["seismic_coupling"] * (s_mult - 1.0))
            tau   = max(crack["memory_tau"], 1e-12)
            relax = 1.0 - np.exp(-dt / tau)
            crack["current_aperture"] = (1.0-relax)*crack["current_aperture"] + relax*b_target
            crack["aperture"] = float(np.clip(crack["current_aperture"], 1e-6, 5e-3))
            if dt > 0.0 and len(crack["aperture_history"]) < 200000:
                crack["aperture_history"].append((t, crack["aperture"]))

    def _crack_connectivity_scalar(self, Sw_scalar, crack):
        """Connectivity factor for a single crack given scalar mean Sw."""
        beta    = crack.get("beta_sw",    20.0)
        Swc     = crack.get("Sw_crit",    0.45)
        sat_f   = 1.0 / (1.0 + np.exp(beta*(Sw_scalar - Swc)))
        dep_f   = np.exp(-crack.get("depth", 0.0) / 0.4)
        ang_f   = np.clip(np.abs(np.cos(crack.get("angle", 0.0))), 0.1, 1.0)
        rou_f   = 1.0 / max(crack.get("roughness", 1.6), 1e-12)
        return crack.get("conn_factor", 1.0) * sat_f * dep_f * ang_f * rou_f

    def single_fracture_flux(self, Cf0, Cin, Pg0, Pin, theta0, Tsoil, Tin, crack):
        """Laminar-turbulent blended flux through one fracture (cubic law + orifice)."""
        phi   = self.p["phi_m"]
        Sw    = float(np.clip(np.asarray(theta0)/max(phi,1e-12), 1e-12, 0.9999))
        conn  = self._crack_connectivity_scalar(Sw, crack)
        Rg    = self.p.get("R_g", 8.314)
        Mg    = self.p.get("M_g", 0.02897)
        rho   = max(Pg0,1.0)*Mg/(Rg*Tsoil)
        mu    = self.p.get("mu", 1.8e-5)
        # driving pressure: soil-indoor difference plus wind term. The stack effect
        # is NOT added here because Pin already carries it (dPindt_rhs/_Pin_steady);
        # adding it again would double-count the buoyancy driving force.
        dP_wd = 0.5*self.p.get("rho_air",1.2)*self.p.get("C_wind",0.6)*self.p.get("U_wind",0.5)**2
        dP    = (Pg0 - Pin) + dP_wd
        a     = crack["aperture"]
        L     = crack["length"] * crack["tortuosity"]
        w     = crack["length"]              # fracture transverse width [m]
        A_c   = a * crack["length"]
        Dh    = 2.0 * a
        Cd    = self.p.get("Cd_crack", 0.61)
        if self.p.get("cubic_law_width_correct", True):
            # parallel-plate cubic law Q = (w a^3 / 12 mu) dP/L (Witherspoon 1980)
            Q_lam = (w * a**3 / (12.0*mu)) * dP / max(L, 1e-12)   # [m^3/s]
            Q_ori = Cd * A_c * np.sign(dP) * np.sqrt(2.0*np.abs(dP)/max(rho,1e-12))
            u_lam = Q_lam / max(w*a, 1e-30)                        # mean velocity [m/s]
            Re    = rho * np.abs(u_lam) * Dh / max(mu, 1e-12)
            blend = np.clip(Re / crack.get("Re_crit", 1800.0), 0.0, 1.0)
            q     = (1.0-blend)*Q_lam + blend*Q_ori               # blended [m^3/s]
        else:
            # legacy branch, kept width-consistent: Q = (w a^3 / 12 mu) dP/L [m^3/s]
            q_lam = (w * a**3 / max(L,1e-12)) * dP / (12.0*mu)
            q_ori = Cd * A_c * np.sign(dP) * np.sqrt(2.0*np.abs(dP)/max(rho,1e-12))
            u_lam = q_lam / max(w*a, 1e-30)
            Re    = rho * np.abs(u_lam) * Dh / max(mu, 1e-12)
            blend = np.clip(Re / crack.get("Re_crit", 1800.0), 0.0, 1.0)
            q     = (1.0-blend)*q_lam + blend*q_ori
        C_up  = Cf0 if q >= 0.0 else Cin
        J_adv = q * C_up
        Sg    = 1.0 - Sw
        D_e   = self.p.get("D_entry", 1e-6)
        nD_e  = self.p.get("nD_entry", 2.4)
        J_diff= -D_e * (Sg**nD_e) * A_c * (Cin-Cf0) / max(L,1e-12)
        return conn * (J_adv + J_diff)

    def crack_network_flux(self, Cf0, Cin, Pg0, Pin, theta0, Tsoil, Tin):
        """Total flux through all fractures."""
        total = 0.0
        for crack in self.cracks:
            total += self.single_fracture_flux(Cf0, Cin, Pg0, Pin, theta0, Tsoil, Tin, crack)
        return total

    def _build_horizontal_fracture_layer(self, n_frac=20, seed=7):
        """Builds a 2D horizontal fracture network at foundation level."""
        rng = np.random.default_rng(seed)
        self.horizontal_network = []
        for _ in range(n_frac):
            self.horizontal_network.append({
                "k":       float(rng.lognormal(np.log(1e-12), 1.0)),
                "width":   float(rng.uniform(0.02, 0.25)),
                "length":  float(rng.uniform(0.5, 5.0)),
                "coupling":float(rng.uniform(0.2, 1.0)),
            })

    def foundation_2d_network_flux(self, Cf0, Cin, Pg0, Pin, theta0):
        """Radon entry rate [Bq/s] through the horizontal foundation fracture network.

        Darcy volumetric flow per fracture: Q = (k/mu) * (dP/L) * A_c with
        A_c = width * b_eff the flow cross-section [m^2]."""
        phi    = self.p["phi_m"]
        mu     = self.p.get("mu", 1.82e-5)
        b_eff  = self.p.get("b_foundation_m", 1e-3)   # effective slab-gap thickness [m]
        Sw     = float(np.clip(np.mean(np.asarray(theta0))/max(phi,1e-12), 1e-12, 0.9999))
        Sw_cl  = self.p.get("Sw_close", 0.85)
        closure= np.clip(1.0 - _smooth_step(Sw, Sw_cl, 0.03), 0.0, 1.0)
        dP     = Pg0 - Pin
        total  = 0.0
        for frac in self.horizontal_network:
            k_h  = frac["k"] * closure
            # superficial Darcy velocity [m/s]; positive = soil into building
            u    = (k_h / mu) * dP / max(frac["length"], 1e-12)
            A_c  = frac["width"] * b_eff
            Q    = u * A_c                            # [m^3/s]
            C_up = Cf0 if Q >= 0.0 else Cin           # upwind concentration
            total += Q * C_up * frac["coupling"]      # [Bq/s]
        return total * self.p.get("foundation_coupling", 0.35)

    def update_self_organizing_permeability(self, q_history, theta):
        """Erosion/clogging feedback on crack conductivity (EXPERIMENTAL; not
        called by the default pipeline)."""
        phi  = self.p["phi_m"]
        Sw   = float(np.mean(np.clip(np.asarray(theta)/max(phi,1e-12), 1e-12, 0.9999)))
        q_cum= float(np.abs(np.mean(q_history[-50:])) if len(q_history) > 0 else 0.0)
        for crack in self.cracks:
            erosion  = 1.0 + 8e-4 * q_cum * (1.0-Sw)**2.5
            clogging = 1.0 - 1.2e-3 * Sw**3.5
            k_base   = crack.get("conductivity_base", crack["aperture"]**2 / 12.0)
            k_new    = np.clip(k_base * erosion * clogging, 1e-15, 5e-9)
            crack["conductivity_base"] = float(k_new)

    def _gas_density(self, P, T):
        return np.maximum(P, 1.0) * self.p.get("M_g", 0.02897) / (self.p.get("R_g", 8.314) * np.maximum(T, 1.0))

    def _entry_concentration(self, Cm_arr, Cf_arr):
        """Sub-slab gas concentration feeding building entry [Bq/m^3].

        C_entry = w_f*Cf_eff + (1-w_f)*Cm_eff, each continuum depth-weighted
        0.3 (near-surface) / 0.7 (deep). The slab seals the surface, so entry
        draws on both continua beneath the foundation; a fracture-only draw
        would sample the atmosphere-vented (depleted) open-field fracture gas.
        w_f = p['entry_frac_weight'] (default 0.5)."""
        w_f  = float(np.clip(self.p.get("entry_frac_weight", 0.5), 0.0, 1.0))
        Cf_e = 0.3 * float(Cf_arr[0]) + 0.7 * float(Cf_arr[-1])
        Cm_e = 0.3 * float(Cm_arr[0]) + 0.7 * float(Cm_arr[-1])
        return w_f * Cf_e + (1.0 - w_f) * Cm_e, w_f

    def dPindt_rhs(self, Pin, Patm, Psoil0, Tin, Tout, t):
        """Indoor pressure relaxation ODE dPin/dt [Pa/s], quasi-steady vs outdoor pressure.
        Envelope air mass equilibrates in tau_P ~ 60-600 s, much faster than radon transport."""
        V      = self.p.get("V", 170.0)
        Rg     = self.p.get("R_g", 8.314)
        Mg     = self.p.get("M_g", 0.02897)
        g      = self.p.get("g", 9.81)
        H_stk  = self.p.get("H_stack", 2.5)
        
        rho_in  = self._gas_density(float(Pin),   float(Tin))
        rho_out = self._gas_density(float(Patm),  float(Tout))

        # stack-effect pressure deficit [Pa], negative means indoor depressurised
        dP_stack = -0.5 * g * H_stk * max(rho_out - rho_in, 0.0)  # only when T_out < T_in

        # balanced mechanical/natural ventilation contributes no net static pressure
        # target indoor pressure constrained to +-30 Pa of Patm
        Pin_target = Patm + np.clip(dP_stack, -30.0, 5.0)

        Cd    = self.p.get("Cd_leak", 0.62)
        A_lk  = self.p.get("A_leak", 5e-3)        # [m^2], ~50 cm^2 for a residential envelope
        A_tot = A_lk

        # pressure relaxation time from envelope-leakage compressibility balance:
        # dm = V*rho*dP/P must leave through m_dot = rho*Cd*A*sqrt(2 dP/rho), giving
        # tau = (V/(P Cd A)) * sqrt(rho dP/2) — seconds for residential envelopes
        dP_ref = max(abs(Pin - Pin_target), 0.5)   # floor avoids division by zero
        tau_Pin = (V / max(float(Patm) * Cd * A_tot, 1e-6)) * np.sqrt(rho_in * dP_ref / 2.0)
        tau_Pin = np.clip(tau_Pin, 1.0, 600.0)

        return (Pin_target - float(Pin)) / tau_Pin

    def _air_exchange_rate(self, t):
        """Dynamic ACH with two door-opening events per day."""
        ach_cl = self.p.get("ACH_closed", 0.10)
        ach_op = self.p.get("ACH_open",   1.20)
        t1, t2 = self.p.get("t_open1s", 7*3600), self.p.get("t_open1e", 9*3600)
        t3, t4 = self.p.get("t_open2s",19*3600), self.p.get("t_open2e",22*3600)
        tday   = t % 86400.0
        sh     = self.p.get("ACH_sharp", 300.0)
        m1     = 0.5*(1+np.tanh((tday-t1)/sh))*0.5*(1-np.tanh((tday-t2)/sh))
        m2     = 0.5*(1+np.tanh((tday-t3)/sh))*0.5*(1-np.tanh((tday-t4)/sh))
        opn    = np.clip(m1+m2, 0.0, 1.0)
        return (ach_cl + (ach_op-ach_cl)*opn + self.p.get("ACH_mech",0.0)) / 3600.0

    def indoor_multizone_rhs(self, J_entry_bs, C_bs, C_gt, t):
        """Two-zone (basement + ground-floor) radon mass-balance ODEs.

        The inter-zone exchange uses one volumetric flow Q_bg = lambda_bg*V_bs
        [m^3/s] applied to BOTH zones so that the exchange conserves activity:
        V_bs*dC_bs |exch = -V_gt*dC_gt |exch. Ventilation supplies outdoor air
        at C_out (~10 Bq/m^3) rather than radon-free air."""
        V_bs    = self.p.get("V_bs",   50.0)
        V_gt    = self.p.get("V_gt",  120.0)
        lam_bg  = self.p.get("lambda_bg", 0.2/3600.0)
        Q_bg    = lam_bg * V_bs                     # inter-zone volumetric flow [m^3/s]
        lam_v   = self._air_exchange_rate(t)
        C_out   = self.p.get("C_out_Rn", 10.0)      # outdoor radon [Bq/m^3]
        lr      = self.lam_rn
        dep_bs  = self.p.get("dep_bs", 0.0)
        dep_gt  = self.p.get("dep_gt", 0.0)
        dC_bs   = (J_entry_bs/max(V_bs,1e-12) + (Q_bg/max(V_bs,1e-12))*(C_gt-C_bs)
                   + lam_v*(C_out - C_bs) - (lr+dep_bs)*C_bs)
        dC_gt   = ((Q_bg/max(V_gt,1e-12))*(C_bs-C_gt)
                   + lam_v*(C_out - C_gt) - (lr+dep_gt)*C_gt)
        return dC_bs, dC_gt

    def progeny_chain_rhs(self, C_Rn, C_Po218, C_Pb214, C_Bi214, C_Po214, t):
        """Rn-222 progeny Bateman chain in ACTIVITY concentrations [Bq/m^3].

        For activity A_i = lambda_i*N_i the production term is lambda_i*A_(i-1)
        (the DAUGHTER's decay constant), since dN_i/dt = lambda_(i-1)*N_(i-1) - ...
        multiplied through by lambda_i."""
        lv    = self._air_exchange_rate(t)
        dep   = self.p.get("dep_progeny", 1.5/3600.0)  # plate-out deposition rate [s^-1]
        l1    = self.lam_po218
        l2    = self.lam_pb214
        l3    = self.lam_bi214
        l4    = self.lam_po214
        dPo218= l1*C_Rn    - (l1+lv+dep)*C_Po218
        dPb214= l2*C_Po218 - (l2+lv+dep)*C_Pb214
        dBi214= l3*C_Pb214 - (l3+lv+dep)*C_Bi214
        dPo214= l4*C_Bi214 - (l4+lv+dep)*C_Po214
        return dPo218, dPb214, dBi214, dPo214

    def equilibrium_factor(self, C_Rn, C_Po218, C_Pb214, C_Bi214, C_Po214):
        """Equilibrium factor F = EEC/C_Rn with the UNSCEAR PAEC weights.

        F = (0.105*A_Po218 + 0.516*A_Pb214 + 0.379*A_Bi214)/A_Rn; the Po-214
        contribution to the potential alpha-energy concentration is negligible
        (UNSCEAR 2000 Annex B)."""
        num = 0.105*C_Po218 + 0.516*C_Pb214 + 0.379*C_Bi214
        return float(np.clip(num / max(C_Rn, 1e-30), 0.0, 1.0))

    def mass_balance_check(self, Cm, Cf, C_bs, C_gt, J_entry, S_m, lambda_v, t,
                           theta=None, J_exhal=0.0):
        """Relative residual of an approximate steady-state Rn-222 budget [Bq/s].

        prod - loss_decay - loss_indoor - J_entry - J_exhal, normalized by the sum
        of magnitudes. S_m is per unit PORE-AIR volume, so production and the soil
        inventory are weighted by the partition porosity beta = phi_g + L*theta.
        Storage terms d/dt are not included: the residual is only meaningful near
        steady state. Pass theta for the correct porosity weighting and J_exhal
        [Bq/s over A_fp] to close the budget at the soil surface."""
        phi_m    = self.p["phi_m"]
        phi_f    = self.p["phi_f"]
        V_bs     = self.p.get("V_bs",  50.0)
        V_gt     = self.p.get("V_gt", 120.0)
        A_fp     = self.p.get("A_footprint", self.p.get("V_bs", 50.0)/2.5)  # [m^2]
        if theta is not None:
            theta_arr = np.asarray(theta, dtype=float)
            L_T       = float(np.mean(self._henry_rn(self.T_field(t))))
            beta_m    = np.maximum(phi_m - theta_arr, 1e-12) + L_T * theta_arr
        else:
            beta_m    = np.full_like(np.asarray(Cm, dtype=float), phi_m * 0.7)
        beta_f   = phi_f
        soil_inv = (np.mean(beta_m*np.asarray(Cm)) + beta_f*np.mean(Cf)) * self.H * A_fp  # [Bq]
        prod     = np.mean(np.asarray(S_m)*beta_m) * self.H * A_fp           # [Bq/s]
        loss_d   = self.lam_rn * soil_inv                                    # [Bq/s]
        loss_i   = (self.lam_rn + lambda_v) * (V_bs*C_bs + V_gt*C_gt)        # [Bq/s]
        err      = (prod - loss_d - loss_i - J_entry - J_exhal) / \
                   max(abs(prod)+abs(loss_d)+abs(loss_i)+abs(J_exhal)+1e-30, 1e-30)
        return float(err)

    def identifiability_fim(self, J_sensitivity):
        """FIM = J^T J; condition number as identifiability proxy."""
        FIM    = J_sensitivity.T @ J_sensitivity
        evals  = np.linalg.eigvalsh(FIM)
        cond   = np.inf if np.min(np.abs(evals)) < 1e-30 else np.max(np.abs(evals))/np.min(np.abs(evals))
        rank   = np.linalg.matrix_rank(FIM)
        return {"FIM": FIM, "eigenvalues": evals, "condition_number": cond, "rank": rank}

    def buckingham_pi(self, theta, Pg, T):
        """Buckingham pi dimensionless groups for radon transport (Buckingham 1914)."""
        phi   = self.p["phi_m"]
        theta = np.asarray(theta, dtype=float)
        Pg    = np.asarray(Pg,    dtype=float)
        Sw    = np.clip(theta/max(phi,1e-12), 1e-12, 0.9999)
        # apparent diffusivity D_a = D_e/beta: the rate at which the pore
        # concentration field actually spreads (L_d = sqrt(D_a/lambda))
        if not hasattr(self, "_T_D0_cache"):
            self._T_D0_cache = float(np.mean(self.T_field(0.0)))
        L_Tb   = float(self._henry_rn(np.asarray([self._T_D0_cache]))[0])
        beta_b = np.maximum(self.p["phi_m"] - theta, 1e-6) + L_Tb * theta
        D_eff = float(np.mean(np.maximum(self._D_eff(theta, "m") / beta_b, 1e-12)))
        k     = self._k_gas(theta, self.z)
        mu    = self.p.get("mu", 1.8e-5)
        dPdz  = np.gradient(Pg, self.dz) if len(Pg) > 1 else np.zeros(len(Pg))
        u_raw = np.abs(k / mu * dPdz)
        u_eff = float(np.clip(np.mean(u_raw), 1e-15, self.p.get("vmax_gas", 5e-7)))
        H     = self.H
        Pe_H  = u_eff * H / max(D_eff, 1e-30)
        Da_H  = self.lam_rn * H / max(u_eff, 1e-30)
        ach   = self.p.get("ACH_closed", 0.1) / 3600.0
        Da_v  = ach * H / max(u_eff, 1e-30)
        eps_m = float(np.mean(self._emanation_factor_inst(theta, T)))
        rho_b = self.p.get("rho_bulk", 1600.0)
        Ra226 = self.p.get("Ra226_Bqkg", 35.0)
        C_ref = self.p.get("C_ref", 100.0)
        S_m   = self.lam_rn * rho_b * Ra226 * eps_m
        Pi_src= S_m * H / max(u_eff * C_ref, 1e-30)
        Pi_ven= ach / max(self.lam_rn, 1e-30)
        k_f   = self.p.get("k_frac", 1e-12)
        k_m   = float(np.mean(k))
        Pi_frac= k_f / max(k_m, 1e-30)
        # Rn-220/Rn-222 decay-constant ratio ~ 5.9e3; the ratio of DIFFUSION
        # LENGTHS is its square root, ~77
        Pi_iso = (self.lam_rn220 / max(self.lam_rn, 1e-30))  # ~ 5.9e3
        # Damkoehler numbers: Da_H (the one used by transport_regime and plots)
        # is the DIFFUSIVE Damkoehler lambda*H^2/D_eff = (H/L_d)^2
        Da_adv  = self.lam_rn * H / max(u_eff,  1e-30)  # lambda*H/u_eff
        Da_diff = self.lam_rn * H**2 / max(D_eff, 1e-30) # lambda*H^2/D_eff
        return {
            "Pe_H":    Pe_H,
            "Da_H":    Da_diff,
            "DaAdv":   Da_adv,      # lambda*H/u_eff
            "DaDiff":  Da_diff,     # lambda*H^2/D_eff
            "Da_v":    Da_v,
            "Pi_src":  Pi_src,
            "Pi_vent": Pi_ven,
            "Pi_frac": Pi_frac,
            "Pi_iso":  Pi_iso,
            "u_eff":   u_eff,
            "D_eff":   D_eff,
        }

    def rhs(self, t, y):
        """Full coupled RHS for the state vector y."""
        nz   = self.nz
        theta   = np.clip(y[0       :  nz],     1e-8,   0.95*self.p["phi_m"])
        Pg      = np.clip(y[  nz    :2*nz],     5e4,    2e5)
        Cm      = np.maximum(y[2*nz :3*nz],     0.0)
        Cf      = np.maximum(y[3*nz :4*nz],     0.0)
        M       =            y[4*nz :5*nz]
        Cs      = np.maximum(y[5*nz :6*nz],     0.0)
        Ar      = np.clip(   y[6*nz :7*nz],     0.0,    1.0)   # Ar ∈ [0,1] always
        Cm220   = np.maximum(y[7*nz :8*nz],     0.0)
        Cf220   = np.maximum(y[8*nz :9*nz],     0.0)
        eps_mem = np.clip(   y[9*nz :10*nz],    0.01,   0.80)
        C_bs    = np.clip(float(y[10*nz]),   0.0, 1e7)   # physical upper bound ~10 MBq/m3
        C_gt    = np.clip(float(y[10*nz+1]), 0.0, 1e7)
        Pin     = np.clip(float(y[10*nz+2]), 5e4, 2e5)
        C_Po218 = max(float(y[10*nz+3]),         0.0)
        C_Pb214 = max(float(y[10*nz+4]),         0.0)
        C_Bi214 = max(float(y[10*nz+5]),         0.0)
        C_Po214 = max(float(y[10*nz+6]),         0.0)

        dtheta_dt  = self.richards_rhs(theta, t)
        T          = self.T_field(t)
        seis_mult  = self.seismic_permeability_multiplier(t)
        # optional deterministic co/post-seismic k(t) recovery overlay, default off
        if self.p.get("seismic_recovery_on", False):
            seis_mult = seis_mult * self.seismic_permeability_multiplier_recovery(t)
            seis_mult = np.clip(seis_mult, 1.0, self.p.get("k_seis_max", 50.0))
        self.update_fracture_aperture(theta, Pg, t, seis_mult)
        dPg_dt     = self.gas_pressure_rhs(theta, Pg, t, seis_mult)

        dAr_dt     = self.radon_accessibility_rhs(Ar, theta, dtheta_dt, Pg)
        deps_mem   = self.emanation_hysteresis_rhs(eps_mem, theta, dtheta_dt, T)
        S_222      = self.source_rn222(theta, T, Ar, eps_mem)
        S_220      = self.source_rn220(theta, T, Ar, eps_mem)

        theta_f    = np.clip(theta * self.p.get("gamma_sw_f", 0.2), 1e-12, 0.9999*self.p["phi_f"])
        dM_dt      = self.exchange_memory_state_rhs(M, Cm, Cf)
        # single gas-pressure field: matrix and fracture share Pg, so the
        # pressure-difference enhancement in matrix_fracture_flux is inert (=1)
        J_mf       = self.matrix_fracture_flux(Cm, Cf, M, theta, theta_f, Pg, Pg)
        dCs_dt     = self.adsorption_dynamic_rhs(Cs, Cm, theta)
        # same saturation-dependent rates as adsorption_dynamic_rhs (mass-consistent)
        k_ads_arr, k_des_arr = self._ads_rates(theta)
        J_ads_Cm   = -(k_ads_arr*Cm - k_des_arr*Cs)

        Dm         = self._D_eff(theta,   "m")
        Df         = self._D_eff(theta_f, "f")
        k_m        = self._k_gas(theta,   self.z, seis_mult)
        k_f        = self.p.get("k_frac", 1e-11) * np.ones(nz)
        mu         = self.p.get("mu", 1.8e-5)
        if self.p.get("face_consistent_gradient", True):
            # face-centred Darcy velocity with harmonic-mean face permeability
            gPdz_f     = _face_gradient(Pg, self.dz)             # (nz+1,)
            km_ext     = np.concatenate(([k_m[0]], k_m, [k_m[-1]]))
            kf_ext     = np.concatenate(([k_f[0]], k_f, [k_f[-1]]))
            km_face    = 2.0*km_ext[:-1]*km_ext[1:]/np.maximum(km_ext[:-1]+km_ext[1:], 1e-300)
            kf_face    = 2.0*kf_ext[:-1]*kf_ext[1:]/np.maximum(kf_ext[:-1]+kf_ext[1:], 1e-300)
            q_m_face   = -(km_face / mu) * gPdz_f
            q_f_face   = -(kf_face / mu) * gPdz_f
            u_m        = -(k_m / mu) * (0.5*(gPdz_f[:-1]+gPdz_f[1:]))  # cell-centred, for diagnostics/CFL
            u_f        = -(k_f / mu) * (0.5*(gPdz_f[:-1]+gPdz_f[1:]))
        else:
            dPdz       = np.gradient(Pg, self.dz) if len(Pg) > 1 else np.zeros(len(Pg))
            u_m        = -(k_m / mu) * dPdz
            u_f        = -(k_f / mu) * dPdz
            q_m_face   = np.empty(nz+1); q_m_face[0]=u_m[0]; q_m_face[-1]=u_m[-1]; q_m_face[1:-1]=0.5*(u_m[:-1]+u_m[1:])
            q_f_face   = np.empty(nz+1); q_f_face[0]=u_f[0]; q_f_face[-1]=u_f[-1]; q_f_face[1:-1]=0.5*(u_f[:-1]+u_f[1:])
        # partition porosity beta = phi_g + L(T)*theta_w converts bulk-area transport
        # divergences into pore-air concentration tendencies:
        #   beta dC/dt = d/dz(D_e dC/dz) - d/dz(u C) + S_bulk - lambda beta C
        # (Andersen 2001; Rogers & Nielson 1991). Dividing by beta gives the
        # dC/dt form used here, with S_gas = S_bulk/beta already partition-corrected.
        L_T        = self._henry_rn(T)
        phi_g_m    = np.maximum(self.p["phi_m"] - theta, 1e-12)
        beta_m     = phi_g_m + L_T * theta
        phi_f_val  = self.p["phi_f"]
        phi_g_f    = np.maximum(phi_f_val - theta_f, 1e-12)
        beta_f     = phi_g_f + L_T * theta_f
        # bottom inlet BC: the MATRIX inflow enters at the local equilibrium
        # pore-gas concentration (production-supported below the domain). The
        # FRACTURE continuum has NO infinite deep reservoir: its deep supply is
        # the matrix-fracture exchange along the column, so inflow re-enters at
        # the zero-gradient (Neumann) value. A Dirichlet C_eq inlet would let
        # barometric pumping import unlimited activity through the fast
        # fracture channel and inflate the surface flux by an order of magnitude.
        Cm_bot_eq  = float(S_222[-1] / max(self.lam_rn, 1e-30))
        F_adv_m    = _upwind(Cm, q_m_face, C_top=0.0, C_bot_inlet=Cm_bot_eq)
        F_adv_f    = _upwind(Cf, q_f_face, C_top=0.0, C_bot_inlet=None)
        adv_m      = -(F_adv_m[1:] - F_adv_m[:-1]) / self.dz
        adv_f      = -(F_adv_f[1:] - F_adv_f[:-1]) / self.dz
        diff_m     = _div_diff_flux(Cm, Dm, self.dz, C_top=0.0)
        diff_f     = _div_diff_flux(Cf, Df, self.dz, C_top=0.0)
        dCm_dt     = (diff_m + adv_m - J_mf)/beta_m + S_222 - self.lam_rn*Cm + J_ads_Cm
        dCf_dt     = (diff_f + adv_f + J_mf)/beta_f - self.lam_rn*Cf

        # thoron: diffusion length ~ L_222/77, source confined near the surface;
        # matrix-fracture exchange included with the instantaneous term only (no memory)
        J_mf220    = self.matrix_fracture_flux(Cm220, Cf220, np.zeros(nz),
                                               theta, theta_f, Pg, Pg)
        Cm220_bot  = float(S_220[-1] / max(self.lam_rn220, 1e-30))
        F_adv_m220 = _upwind(Cm220, q_m_face, C_top=0.0, C_bot_inlet=Cm220_bot)
        F_adv_f220 = _upwind(Cf220, q_f_face, C_top=0.0, C_bot_inlet=None)
        adv_m220   = -(F_adv_m220[1:] - F_adv_m220[:-1]) / self.dz
        adv_f220   = -(F_adv_f220[1:] - F_adv_f220[:-1]) / self.dz
        diff_m220  = _div_diff_flux(Cm220, Dm, self.dz, C_top=0.0)
        diff_f220  = _div_diff_flux(Cf220, Df, self.dz, C_top=0.0)
        dCm220_dt  = (diff_m220 + adv_m220 - J_mf220)/beta_m + S_220 - self.lam_rn220*Cm220
        dCf220_dt  = (diff_f220 + adv_f220 + J_mf220)/beta_f - self.lam_rn220*Cf220

        # radon enters the building through floor cracks driven by sub-slab pressure.
        # The slab seals the surface, so entry draws on BOTH continua beneath the
        # foundation (the open-field column vents the fracture continuum, which
        # would otherwise sample an unrealistically depleted reservoir).
        Pg0        = float(Pg[0])         # surface soil pressure approx Patm
        Pg_bot     = float(Pg[-1])        # sub-slab pressure = Patm + dP_bottom
        # geometric mean along column as effective driving pressure (Mowris 1985)
        Pg_eff     = float(np.sqrt(Pg0 * Pg_bot)) if Pg_bot > Pg0 else Pg0
        C_entry, w_f_entry = self._entry_concentration(Cm, Cf)
        T0_soil    = float(T[0])
        Tin        = self.T_indoor(t)
        Tout       = self.T_outdoor(t)
        theta0     = float(theta[0])
        J_cracks   = self.crack_network_flux(C_entry, C_bs, Pg_eff, Pin, theta0, T0_soil, Tin)
        J_found    = self.foundation_2d_network_flux(C_entry, C_bs, Pg_eff, Pin, theta0)
        J_entry    = J_cracks + J_found

        # thoron entry is negligible indoors due to short half-life, kept for ratio checks
        C220_entry, _ = self._entry_concentration(Cm220, Cf220)
        J_entry220 = self.crack_network_flux(C220_entry, 0.0, Pg_eff, Pin, theta0, T0_soil, Tin) * 0.01

        # mass-conserving sink: radon entering the building is removed from Cf.
        # Each receiving cell holds the partition volume A_fp*dz*beta_f, so the
        # column-integrated removal equals J_entry exactly (previous normalization
        # by A_fp*H*phi_f removed only ~1/nz of the entry rate).
        if self.p.get("entry_mass_coupling_on", True):
            A_fp     = self.p.get("A_footprint", self.p.get("V_bs", 50.0) / 2.5)
            w_top, w_bot = 0.30, 0.70
            vol_top_f  = max(A_fp * self.dz * float(beta_f[0]),  1e-9)
            vol_bot_f  = max(A_fp * self.dz * float(beta_f[-1]), 1e-9)
            vol_top_m  = max(A_fp * self.dz * float(beta_m[0]),  1e-9)
            vol_bot_m  = max(A_fp * self.dz * float(beta_m[-1]), 1e-9)
            # removal split between the continua in the same proportion as the
            # entry draw (w_f from fractures, 1-w_f from the matrix)
            sink_cf      = np.zeros(nz)
            sink_cf[0]  += w_top * w_f_entry * J_entry / vol_top_f
            sink_cf[-1] += w_bot * w_f_entry * J_entry / vol_bot_f
            sink_cm      = np.zeros(nz)
            sink_cm[0]  += w_top * (1.0 - w_f_entry) * J_entry / vol_top_m
            sink_cm[-1] += w_bot * (1.0 - w_f_entry) * J_entry / vol_bot_m
            sink_cf220   = np.zeros(nz)
            sink_cf220[0]  += w_top * w_f_entry * J_entry220 / vol_top_f
            sink_cf220[-1] += w_bot * w_f_entry * J_entry220 / vol_bot_f
            sink_cm220   = np.zeros(nz)
            sink_cm220[0]  += w_top * (1.0 - w_f_entry) * J_entry220 / vol_top_m
            sink_cm220[-1] += w_bot * (1.0 - w_f_entry) * J_entry220 / vol_bot_m
        else:
            sink_cf    = np.zeros(nz); sink_cm    = np.zeros(nz)
            sink_cf220 = np.zeros(nz); sink_cm220 = np.zeros(nz)
        dCm_dt    = dCm_dt    - sink_cm
        dCf_dt    = dCf_dt    - sink_cf
        dCm220_dt = dCm220_dt - sink_cm220
        dCf220_dt = dCf220_dt - sink_cf220

        Patm       = self._Patm(t)
        dPin_dt    = self.dPindt_rhs(Pin, Patm, Pg_bot, Tin, Tout, t)
        dC_bs, dC_gt = self.indoor_multizone_rhs(J_entry, C_bs, C_gt, t)
        dPo218, dPb214, dBi214, dPo214 = self.progeny_chain_rhs(C_bs, C_Po218, C_Pb214, C_Bi214, C_Po214, t)

        dydt = np.empty(self.nstate)
        dydt[0       :  nz] = dtheta_dt
        dydt[  nz    :2*nz] = dPg_dt
        dydt[2*nz    :3*nz] = dCm_dt
        dydt[3*nz    :4*nz] = dCf_dt
        dydt[4*nz    :5*nz] = dM_dt
        dydt[5*nz    :6*nz] = dCs_dt
        dydt[6*nz    :7*nz] = dAr_dt
        dydt[7*nz    :8*nz] = dCm220_dt
        dydt[8*nz    :9*nz] = dCf220_dt
        dydt[9*nz    :10*nz]= deps_mem
        dydt[10*nz  ]       = dC_bs
        dydt[10*nz+1]       = dC_gt
        dydt[10*nz+2]       = dPin_dt
        dydt[10*nz+3]       = dPo218
        dydt[10*nz+4]       = dPb214
        dydt[10*nz+5]       = dBi214
        dydt[10*nz+6]       = dPo214
        return np.nan_to_num(dydt, nan=0.0, posinf=0.0, neginf=0.0)

    def build_y0(self):
        nz   = self.nz
        p    = self.p
        y0   = np.zeros(self.nstate)
        phi  = p["phi_m"]
        theta0_arr = np.full(nz, p.get("theta0", 0.35 * phi))
        y0[0:nz]      = theta0_arr
        # linear Pg profile at t=0 ensures Pg > Pin_init for positive initial radon entry
        Patm0 = p.get("P_atm0", 101325.0)
        dP_bot = p.get("dP_bottom", 5.0)
        Pbot0  = Patm0 + dP_bot
        y0[nz:2*nz] = np.linspace(Patm0, Pbot0, nz)
        # Rn-222 initialized at the steady DIFFUSIVE PROFILE (not uniform
        # equilibrium): a uniform C=S/lambda start puts the near-surface cells
        # far above the true boundary layer and inflates early flux readings by
        # 5-18% even at day 5. The profile is built with the SAME Ar and eps
        # arrays that are written into the state vector below.
        T0_arr = p.get("T0", 288.0) + p.get("T_grad", 0.025) * self.z
        Sw0_arr = np.clip(theta0_arr / max(phi, 1e-12), 0.0, 0.9999)
        Ar0_arr = np.clip(1.0 - Sw0_arr, 0.05, 1.0)          # accessibility floor = Sg
        eps0_arr = self._emanation_factor_inst(theta0_arr, T0_arr)
        S222_0 = self.source_rn222(theta0_arr, T0_arr, Ar0_arr, eps0_arr)
        Cm_eq0 = S222_0 / max(self.lam_rn, 1e-30)
        # apparent diffusivity and diffusion length for the boundary layer
        L_T0    = float(np.mean(self._henry_rn(T0_arr)))
        beta0   = np.maximum(phi - theta0_arr, 1e-12) + L_T0 * theta0_arr
        D_app0  = float(np.mean(np.maximum(self._D_eff(theta0_arr, "m") / beta0, 1e-30)))
        L_d0    = max(np.sqrt(D_app0 / max(self.lam_rn, 1e-30)), 1e-3)
        profile = 1.0 - np.exp(-self.z / L_d0)
        C0m    = p.get("C0_m", None)
        C0f    = p.get("C0_f", None)
        if C0m is None: C0m = Cm_eq0 * profile
        # pore-gas equilibrium concentration is the same in both continua
        if C0f is None: C0f = Cm_eq0 * profile
        y0[2*nz:3*nz] = C0m
        y0[3*nz:4*nz] = C0f
        y0[4*nz:5*nz] = C0m - C0f
        # adsorbed phase at its own steady state Cs = k_ads*Cm/(k_des + lambda)
        k_ads0, k_des0 = self._ads_rates(theta0_arr)
        y0[5*nz:6*nz] = k_ads0 * np.asarray(C0m) / np.maximum(k_des0 + self.lam_rn, 1e-30)
        Sw0   = np.clip(p.get("theta0", 0.35*phi) / max(phi, 1e-12), 0.0, 0.9999)
        Sg0   = 1.0 - Sw0
        # initial accessibility at its physical floor (gas-connected fraction)
        Ar0   = np.clip(Sg0, 0.05, 1.0)
        y0[6*nz:7*nz] = Ar0
        y0[7*nz:8*nz] = p.get("C0_m220", 0.0)
        y0[8*nz:9*nz] = p.get("C0_f220", 0.0)
        T0_arr       = p.get("T0", 288.0) + p.get("T_grad", 0.025) * self.z
        eps_i        = self._emanation_factor_inst(np.full(nz, p.get("theta0", 0.35*phi)), T0_arr)
        y0[9*nz:10*nz]= eps_i
        y0[10*nz]    = p.get("Cin0",   50.0)    # C_bs [Bq/m3]
        y0[10*nz+1]  = p.get("Cin_gt0",20.0)    # C_gt
        y0[10*nz+2]  = p.get("P_in0", Patm0 - 1.5)  # stack effect: typical -1 to -3 Pa
        # progeny initialized at steady state with C_bs0; activity-Bateman
        # production uses the DAUGHTER's decay constant (see progeny_chain_rhs)
        C_bs0 = p.get("Cin0", 50.0)
        lv0   = p.get("ACH_closed", 0.10) / 3600.0
        dep   = p.get("dep_progeny", 1.5/3600.0)
        l1,l2,l3,l4 = self.lam_po218, self.lam_pb214, self.lam_bi214, self.lam_po214
        loss1 = l1 + lv0 + dep
        loss2 = l2 + lv0 + dep
        loss3 = l3 + lv0 + dep
        loss4 = l4 + lv0 + dep
        CPo218_eq = l1 * C_bs0 / max(loss1, 1e-12)
        CPb214_eq = l2 * CPo218_eq / max(loss2, 1e-12)
        CBi214_eq = l3 * CPb214_eq / max(loss3, 1e-12)
        CPo214_eq = l4 * CBi214_eq / max(loss4, 1e-12)
        y0[10*nz+3]  = CPo218_eq
        y0[10*nz+4]  = CPb214_eq
        y0[10*nz+5]  = CBi214_eq
        y0[10*nz+6]  = CPo214_eq
        return y0

    def simulate(self, days=3.0, max_step=1800, rtol=1e-4, atol=1e-7, method="BDF"):
        y0  = self.build_y0()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sol = solve_ivp(
                self.rhs,
                (0.0, days*86400.0),
                y0,
                method   = method,
                rtol     = rtol,
                atol     = atol,
                max_step = max_step,
                dense_output = False,
            )
        return sol

    def _cfl_dt(self, theta, Pg):
        """Max stable explicit dt from diffusion, advection and decay CFL, clipped to [30,1800] s."""
        theta = np.asarray(theta, dtype=float)
        Pg    = np.asarray(Pg,    dtype=float)
        # stability follows the APPARENT diffusivity D_e/beta (the rate at which
        # the pore concentration actually diffuses in the partition equation)
        L_T0   = float(np.mean(self._henry_rn(self.T_field(0.0))))
        beta_c = np.maximum(self.p['phi_m'] - theta, 1e-12) + L_T0 * theta
        D_eff  = float(np.mean(np.maximum(self._D_eff(theta, 'm') / beta_c, 1e-30)))
        k_g    = self._k_gas(theta, self.z)
        mu     = self.p.get('mu', 1.8e-5)
        gradPg = np.abs(np.gradient(Pg, self.dz)) if len(Pg) > 1 else np.zeros(1)
        phi_g  = np.maximum(self.p['phi_m'] - theta, 1e-12)   # gas-filled porosity
        u_g    = k_g / mu * gradPg                            # Darcy velocity [m/s]
        u_pore = float(np.max(np.abs(u_g / phi_g)))          # pore velocity
        # fracture gas is faster than matrix gas, so it sets the CFL bound
        if self.p.get("cfl_include_fracture", True):
            k_f = self.p.get("k_frac", 1e-11)                # fracture permeability [m^2]
            phi_gf = max(self.p.get("phi_f", 0.08), 1e-12)
            u_pore_f = float(np.max(np.abs(k_f / mu * gradPg / phi_gf)))
            u_pore   = max(u_pore, u_pore_f)
        dt_diff = 0.5 * self.dz**2 / max(D_eff,  1e-30)
        dt_adv  = 0.4 * self.dz    / max(u_pore, 1e-30)
        # decay is integrated analytically elsewhere, kept only as a mild guard
        lam_Po218 = DECAY_CONSTANTS.get('Po218', 3.77e-3)    # [s^-1]
        dt_dec  = 0.3  / max(lam_Po218, 1e-30)
        h_safe  = min(dt_diff, dt_adv, dt_dec)
        return float(np.clip(h_safe, 30.0, 1800.0))

    def _rhs_no_Cm(self, t, y):
        """RHS with Cm/Cf/Cm220/Cf220 derivatives zeroed for operator splitting."""
        dydt = self.rhs(t, y)
        nz = self.nz
        dydt[2*nz:3*nz] = 0.0
        dydt[3*nz:4*nz] = 0.0
        dydt[7*nz:8*nz] = 0.0
        dydt[8*nz:9*nz] = 0.0
        return dydt

    def _advance_Cm_implicit(self, Cm, Cf, Cm220, Cf220, theta, Pg, Ar, eps_mem, t, h,
                             Cs=None):
        """Semi-implicit step: analytic decay+source, explicit upwind advection, Crank-Nicolson diffusion."""
        nz    = self.nz
        phi_m = self.p["phi_m"]
        phi_f = self.p["phi_f"]
        lam   = self.lam_rn
        lam20 = self.lam_rn220
        T     = self.T_field(t + h/2)

        S222  = self.source_rn222(theta, T, Ar, eps_mem)
        S220  = self.source_rn220(theta, T, Ar, eps_mem)
        Cm_eq = S222 / np.maximum(lam,   1e-30)
        # pore-gas equilibrium concentration is the same in both continua
        Cf_eq = Cm_eq.copy()
        Cm20_eq = S220 / np.maximum(lam20, 1e-30)
        # pore-gas equilibrium concentration is the same in both continua
        Cf20_eq = Cm20_eq.copy()

        e_rn   = np.exp(-lam   * h)
        e_rn20 = np.exp(-lam20 * h)
        # matrix: analytic decay/production toward the local equilibrium.
        # FRACTURE: no own production — it only decays; its radon supply is the
        # explicit matrix-fracture exchange applied in the substep loop below
        # (mass-consistent with rhs(); the fracture activity is debited from
        # the matrix instead of being created from an implicit source).
        Cm_new   = Cm_eq   + (Cm   - Cm_eq)   * e_rn
        Cf_new   = Cf * e_rn
        Cm20_new = Cm20_eq + (Cm220 - Cm20_eq) * e_rn20
        Cf20_new = Cf220 * e_rn20

        theta_f_arr = np.clip(theta * self.p.get("gamma_sw_f", 0.2), 1e-12, 0.9999*phi_f)
        Dm   = self._D_eff(theta, "m")
        Df   = self._D_eff(theta_f_arr, "f")
        k_m  = self._k_gas(theta, self.z)
        mu   = self.p.get("mu", 1.8e-5)
        k_f_arr = self.p.get("k_frac", 1e-11) * np.ones(nz)
        # partition porosity beta = phi_g + L*theta for both continua. The
        # transport operator is (1/beta_i)*d/dz(D_e dC/dz - u C): face fluxes
        # stay PHYSICAL (bulk D_e and Darcy u), and each cell's divergence is
        # divided by its beta — the flux-conservative form (pre-dividing D or u
        # by beta breaks flux continuity across cells of different beta)
        L_T      = self._henry_rn(T)
        beta_m   = np.maximum(phi_m - theta, 1e-12) + L_T * theta
        beta_f_a = np.maximum(phi_f - theta_f_arr, 1e-12) + L_T * theta_f_arr
        if self.p.get("face_consistent_gradient", True):
            gPdz_f  = _face_gradient(Pg, self.dz)
            km_ext  = np.concatenate(([k_m[0]], k_m, [k_m[-1]]))
            kf_ext  = np.concatenate(([k_f_arr[0]], k_f_arr, [k_f_arr[-1]]))
            km_face = 2.0*km_ext[:-1]*km_ext[1:]/np.maximum(km_ext[:-1]+km_ext[1:], 1e-300)
            kf_face = 2.0*kf_ext[:-1]*kf_ext[1:]/np.maximum(kf_ext[:-1]+kf_ext[1:], 1e-300)
            q_m_face = -(km_face / mu) * gPdz_f
            q_f_face = -(kf_face / mu) * gPdz_f
            u_m      = -(k_m / mu) * (0.5*(gPdz_f[:-1]+gPdz_f[1:]))
        else:
            dPdz = np.gradient(Pg, self.dz) if nz > 1 else np.zeros(nz)
            u_m  = -(k_m / mu) * dPdz
            u_f  = -(self.p.get("k_frac", 1e-11) / mu) * dPdz
            q_m_face = np.empty(nz+1)
            q_f_face = np.empty(nz+1)
            q_m_face[0]=u_m[0]; q_m_face[-1]=u_m[-1]
            q_f_face[0]=u_f[0]; q_f_face[-1]=u_f[-1]
            if nz > 1:
                q_m_face[1:-1] = 0.5*(u_m[:-1]+u_m[1:])
                q_f_face[1:-1] = 0.5*(u_f[:-1]+u_f[1:])

        # stability of the pore-space operator is governed by the APPARENT
        # diffusivity D_e/beta (the equation diffuses C at that rate)
        D_app_mean = float(np.mean(np.maximum(Dm / beta_m, 1e-30)))
        dt_diff    = 0.5 * self.dz**2 / max(D_app_mean, 1e-30)
        u_pore_max = float(np.max(np.abs(u_m / np.maximum(
            self.p["phi_m"] - theta, 1e-12))))
        dt_adv     = 0.4 * self.dz / max(u_pore_max, 1e-30)

        use_cn = self.p.get("cn_diffusion_on", True)
        if use_cn:
            # Crank-Nicolson diffusion is unconditionally stable, only advection limits dt
            h_tr = np.clip(dt_adv, 1.0, h)
        else:
            h_tr = np.clip(min(dt_diff, dt_adv), 1.0, h)
        n_tr       = max(1, int(np.ceil(h / h_tr)))
        h_tr       = h / n_tr
        harm       = _DIFF_FACE_HARMONIC

        for _ in range(n_tr):
            # matrix inlet at local equilibrium; fracture inlet zero-gradient
            # (no infinite deep fracture reservoir — see rhs for rationale)
            F_adv_m  = _upwind(Cm_new,  q_m_face, C_top=0.0, C_bot_inlet=float(Cm_eq[-1]))
            F_adv_f  = _upwind(Cf_new,  q_f_face, C_top=0.0, C_bot_inlet=None)
            F_adv_m2 = _upwind(Cm20_new,q_m_face, C_top=0.0, C_bot_inlet=float(Cm20_eq[-1]))
            F_adv_f2 = _upwind(Cf20_new,q_f_face, C_top=0.0, C_bot_inlet=None)
            # flux divergences divided by the cell's partition porosity
            adv_m  = -(F_adv_m[1:]  - F_adv_m[:-1])  / self.dz / beta_m
            adv_f  = -(F_adv_f[1:]  - F_adv_f[:-1])  / self.dz / beta_f_a
            adv_m2 = -(F_adv_m2[1:] - F_adv_m2[:-1]) / self.dz / beta_m
            adv_f2 = -(F_adv_f2[1:] - F_adv_f2[:-1]) / self.dz / beta_f_a

            # explicit matrix-fracture exchange (instantaneous term only),
            # debited from the matrix and credited to the fracture — the same
            # coupling used by rhs(), keeping the two integrators consistent
            J_mf_s   = self.matrix_fracture_flux(Cm_new, Cf_new, np.zeros(nz),
                                                 theta, theta_f_arr, Pg, Pg)
            J_mf_s20 = self.matrix_fracture_flux(Cm20_new, Cf20_new, np.zeros(nz),
                                                 theta, theta_f_arr, Pg, Pg)
            Cm_new   = np.maximum(Cm_new   - h_tr * J_mf_s   / beta_m,   0.0)
            Cf_new   = np.maximum(Cf_new   + h_tr * J_mf_s   / beta_f_a, 0.0)
            Cm20_new = np.maximum(Cm20_new - h_tr * J_mf_s20 / beta_m,   0.0)
            Cf20_new = np.maximum(Cf20_new + h_tr * J_mf_s20 / beta_f_a, 0.0)

            if use_cn:
                Cm_new   = np.maximum(Cm_new   + h_tr*adv_m,  0.0)
                Cf_new   = np.maximum(Cf_new   + h_tr*adv_f,  0.0)
                Cm20_new = np.maximum(Cm20_new + h_tr*adv_m2, 0.0)
                Cf20_new = np.maximum(Cf20_new + h_tr*adv_f2, 0.0)
                Cm_new   = np.maximum(_cn_diffusion_step(Cm_new,  Dm, self.dz, h_tr, C_top=0.0, C_bot=None, harmonic=harm, beta=beta_m),   0.0)
                Cf_new   = np.maximum(_cn_diffusion_step(Cf_new,  Df, self.dz, h_tr, C_top=0.0, C_bot=None, harmonic=harm, beta=beta_f_a), 0.0)
                Cm20_new = np.maximum(_cn_diffusion_step(Cm20_new,Dm, self.dz, h_tr, C_top=0.0, C_bot=None, harmonic=harm, beta=beta_m),   0.0)
                Cf20_new = np.maximum(_cn_diffusion_step(Cf20_new,Df, self.dz, h_tr, C_top=0.0, C_bot=None, harmonic=harm, beta=beta_f_a), 0.0)
            else:
                diff_m  = _div_diff_flux(Cm_new,  Dm, self.dz, C_top=0.0) / beta_m
                diff_f  = _div_diff_flux(Cf_new,  Df, self.dz, C_top=0.0) / beta_f_a
                diff_m2 = _div_diff_flux(Cm20_new,Dm, self.dz, C_top=0.0) / beta_m
                diff_f2 = _div_diff_flux(Cf20_new,Df, self.dz, C_top=0.0) / beta_f_a
                Cm_new   = np.maximum(Cm_new   + h_tr * (diff_m  + adv_m),  0.0)
                Cf_new   = np.maximum(Cf_new   + h_tr * (diff_f  + adv_f),  0.0)
                Cm20_new = np.maximum(Cm20_new + h_tr * (diff_m2 + adv_m2), 0.0)
                Cf20_new = np.maximum(Cf20_new + h_tr * (diff_f2 + adv_f2), 0.0)

        # adsorption exchange with the (frozen) adsorbed phase Cs, absent from
        # _rhs_no_Cm on this path: without it Cs adsorbs radon that is never
        # debited from Cm, creating ~12-13% spurious activity/flux at steady state
        if Cs is not None:
            k_ads_a, k_des_a = self._ads_rates(theta)
            Cm_new = np.maximum((Cm_new + h * k_des_a * np.maximum(Cs, 0.0))
                                / (1.0 + h * k_ads_a), 0.0)

        Cm_new   = np.clip(Cm_new,   0.0, 5.0 * np.maximum(Cm_eq, 1.0))
        Cf_new   = np.clip(Cf_new,   0.0, 5.0 * np.maximum(Cf_eq, 1.0))
        Cm20_new = np.clip(Cm20_new, 0.0, 5.0 * np.maximum(Cm20_eq, 1.0))
        Cf20_new = np.clip(Cf20_new, 0.0, 5.0 * np.maximum(Cf20_eq, 1.0))
        return Cm_new, Cf_new, Cm20_new, Cf20_new

    def _Pg_steady(self, theta, t, seismic_mult=None):
        """Quasi-stationary soil gas pressure profile Pg_ss [Pa], shape (nz,)."""
        nz  = self.nz
        mu  = self.p.get("mu", 1.82e-5)
        k   = self._k_gas(theta, self.z, seismic_mult)
        dz  = self.dz
        Patm= self._Patm(t)
        # deep boundary: attenuated, phase-lagged barometric wave plus overpressure
        Pbot = self._Pbot_transient(theta, t, seismic_mult)

        if nz == 1:
            return np.array([0.5*(Patm+Pbot)])

        k_face = np.empty(nz+1)
        k_face[0]  = k[0]; k_face[-1] = k[-1]
        for i in range(1, nz):
            k_face[i] = 2.0*k[i-1]*k[i]/max(k[i-1]+k[i], 1e-40)
        T_lam = k_face/(mu*dz)
        # boundary faces sit dz/2 from the first/last cell centre: transmissivity doubles
        T_lam[0]  *= 2.0
        T_lam[-1] *= 2.0

        a=np.zeros(nz); b=np.zeros(nz); c=np.zeros(nz); rhs=np.zeros(nz)
        for i in range(nz):
            tL=T_lam[i]; tR=T_lam[i+1]
            b[i] = -(tL+tR)
            if i>0:     a[i] = tL
            if i<nz-1:  c[i] = tR
        rhs[0]  = -T_lam[0]  * Patm
        rhs[-1] = -T_lam[-1] * Pbot

        c_=np.zeros(nz); d_=np.zeros(nz)
        c_[0] = c[0]/b[0] if abs(b[0])>1e-40 else 0.0
        d_[0] = rhs[0]/b[0] if abs(b[0])>1e-40 else Patm
        for i in range(1, nz):
            denom = b[i] - a[i]*c_[i-1]
            c_[i] = c[i]/denom if abs(denom)>1e-40 else 0.0
            d_[i] = (rhs[i]-a[i]*d_[i-1])/denom if abs(denom)>1e-40 else Patm
        Pg_ss = np.zeros(nz)
        Pg_ss[-1] = d_[-1]
        for i in range(nz-2,-1,-1):
            Pg_ss[i] = d_[i] - c_[i]*Pg_ss[i+1]
        P_lo = min(Patm,Pbot) - 10.0
        P_hi = max(Patm,Pbot) + 10.0
        return np.clip(Pg_ss, P_lo, P_hi)

    def _Pin_steady(self, Patm, Tin, Tout):
        """Quasi-stationary indoor pressure Pin = Patm + stack-effect term, typical range Patm +/- 5 Pa."""
        g    = self.p.get("g", 9.81)
        H_s  = self.p.get("H_stack", 2.5)
        Mg   = self.p.get("M_g", 0.02897)
        Rg   = self.p.get("R_g", 8.314)
        rho_i = Patm * Mg / (Rg * max(Tin,  1.0))
        rho_o = Patm * Mg / (Rg * max(Tout, 1.0))
        # stack-effect depressurises the building when T_in > T_out
        dP_stack = -g * H_s * (rho_o - rho_i) * 0.5
        return float(np.clip(Patm + dP_stack, Patm-20.0, Patm+5.0))

    def simulate_fast(self, days=3.0, dt_out=3600.0, n_sub=None):
        """Operator-split integrator: semi-implicit Cm/Cf transport plus RK4 for the rest."""
        nz    = self.nz
        y     = self.build_y0()
        T_end = days * 86400.0
        n_out = int(round(T_end / dt_out)) + 1
        t_arr = np.linspace(0.0, T_end, n_out)
        Y     = np.zeros((self.nstate, n_out))
        Y[:, 0] = y.copy()
        t     = 0.0
        phi_m = self.p["phi_m"]

        h = self._cfl_dt(y[:nz], y[nz:2*nz])

        for i_out in range(1, n_out):
            t_next = t_arr[i_out]
            while t < t_next - 1e-9:
                h = min(h, t_next - t)

                theta   = np.clip(y[0:nz], 1e-8, 0.95*phi_m)
                Pg      = np.clip(y[nz:2*nz], 5e4, 2e5)
                Cm      = np.maximum(y[2*nz:3*nz], 0.0)
                Cf      = np.maximum(y[3*nz:4*nz], 0.0)
                Cm220   = np.maximum(y[7*nz:8*nz], 0.0)
                Cf220   = np.maximum(y[8*nz:9*nz], 0.0)
                Ar      = np.clip(y[6*nz:7*nz], 0.0, 1.0)
                eps_mem = np.clip(y[9*nz:10*nz], 0.01, 0.80)

                Cs_now = np.maximum(y[5*nz:6*nz], 0.0)
                Cm_new, Cf_new, Cm20_new, Cf20_new = self._advance_Cm_implicit(
                    Cm, Cf, Cm220, Cf220, theta, Pg, Ar, eps_mem, t, h, Cs=Cs_now)

                # _rhs_no_Cm zeroes dCf/dt, so the entry sink is applied to Cf here
                if self.p.get("entry_mass_coupling_on", True):
                    C_bs_now = float(y[10*nz])
                    Pin_now  = float(y[10*nz+2])
                    Pg0_s    = float(Pg[0]); Pg_bot_s = float(Pg[-1])
                    Pg_eff_s = float(np.sqrt(Pg0_s*Pg_bot_s)) if Pg_bot_s > Pg0_s else Pg0_s
                    C_ent_s, wf_s = self._entry_concentration(Cm_new, Cf_new)
                    T0_s     = float(self.T_field(t)[0])
                    Tin_s    = self.T_indoor(t)
                    theta0_s = float(theta[0])
                    Jc = self.crack_network_flux(C_ent_s, C_bs_now, Pg_eff_s, Pin_now, theta0_s, T0_s, Tin_s)
                    Jf = self.foundation_2d_network_flux(C_ent_s, C_bs_now, Pg_eff_s, Pin_now, theta0_s)
                    J_entry_s  = Jc + Jf
                    # same per-cell partition-volume normalization as in rhs,
                    # removal split w_f (fractures) / 1-w_f (matrix)
                    A_fp_s     = self.p.get("A_footprint", self.p.get("V_bs", 50.0)/2.5)
                    L_Ts       = float(np.mean(self._henry_rn(self.T_field(t))))
                    th_f0      = float(np.clip(theta[0]  * self.p.get("gamma_sw_f", 0.2), 1e-12, 0.9999*self.p["phi_f"]))
                    th_fb      = float(np.clip(theta[-1] * self.p.get("gamma_sw_f", 0.2), 1e-12, 0.9999*self.p["phi_f"]))
                    beta_f0    = max(self.p["phi_f"] - th_f0, 1e-12) + L_Ts*th_f0
                    beta_fb    = max(self.p["phi_f"] - th_fb, 1e-12) + L_Ts*th_fb
                    beta_m0    = max(self.p["phi_m"] - float(theta[0]),  1e-12) + L_Ts*float(theta[0])
                    beta_mb    = max(self.p["phi_m"] - float(theta[-1]), 1e-12) + L_Ts*float(theta[-1])
                    vol_top_f  = max(A_fp_s*self.dz*beta_f0, 1e-9)
                    vol_bot_f  = max(A_fp_s*self.dz*beta_fb, 1e-9)
                    vol_top_m  = max(A_fp_s*self.dz*beta_m0, 1e-9)
                    vol_bot_m  = max(A_fp_s*self.dz*beta_mb, 1e-9)
                    Cf_new[0]  = max(Cf_new[0]  - h*0.30*wf_s*J_entry_s/vol_top_f, 0.0)
                    Cf_new[-1] = max(Cf_new[-1] - h*0.70*wf_s*J_entry_s/vol_bot_f, 0.0)
                    Cm_new[0]  = max(Cm_new[0]  - h*0.30*(1.0-wf_s)*J_entry_s/vol_top_m, 0.0)
                    Cm_new[-1] = max(Cm_new[-1] - h*0.70*(1.0-wf_s)*J_entry_s/vol_bot_m, 0.0)

                y[2*nz:3*nz] = Cm_new
                y[3*nz:4*nz] = Cf_new
                y[7*nz:8*nz] = Cm20_new
                y[8*nz:9*nz] = Cf20_new

                k1 = self._rhs_no_Cm(t, y)
                y2 = np.clip(y + (h/2)*k1, -1e12, 1e12)
                k2 = self._rhs_no_Cm(t + h/2, y2)
                y3 = np.clip(y + (h/2)*k2, -1e12, 1e12)
                k3 = self._rhs_no_Cm(t + h/2, y3)
                y4 = np.clip(y + h*k3, -1e12, 1e12)
                k4 = self._rhs_no_Cm(t + h, y4)
                y  = y + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
                t += h

                y[0:nz]            = np.clip(y[0:nz],    1e-8,  0.95*phi_m)
                # Pg/Pin set to quasi-stationary values; their ODE only feeds transport
                theta_now = np.clip(y[0:nz], 1e-8, 0.95*phi_m)
                y[nz:2*nz]  = self._Pg_steady(theta_now, t)
                Patm_now    = self._Patm(t)
                Tin_now  = self.T_indoor(t)
                Tout_now = self.T_outdoor(t)
                y[10*nz+2] = self._Pin_steady(Patm_now, Tin_now, Tout_now)
                y[2*nz:3*nz]       = np.maximum(y[2*nz:3*nz],  0.0)
                y[3*nz:4*nz]       = np.maximum(y[3*nz:4*nz],  0.0)
                y[4*nz:5*nz]       = np.maximum(y[4*nz:5*nz],  0.0)
                y[5*nz:6*nz]       = np.maximum(y[5*nz:6*nz],  0.0)
                y[6*nz:7*nz]       = np.clip(y[6*nz:7*nz],  0.0, 1.0)
                y[7*nz:8*nz]       = np.maximum(y[7*nz:8*nz],  0.0)
                y[8*nz:9*nz]       = np.maximum(y[8*nz:9*nz],  0.0)
                y[9*nz:10*nz]      = np.clip(y[9*nz:10*nz],  0.01, 0.80)
                y[10*nz]           = np.clip(y[10*nz],    0.0, 5e4)
                y[10*nz+1]         = np.clip(y[10*nz+1],  0.0, 5e4)
                y[10*nz+2]         = np.clip(y[10*nz+2],  5e4, 2e5)
                y[10*nz+3:10*nz+7] = np.maximum(y[10*nz+3:10*nz+7], 0.0)

                h = self._cfl_dt(y[:nz], y[nz:2*nz])

            Y[:, i_out] = y.copy()

        class _Sol:
            pass
        sol         = _Sol()
        sol.t       = t_arr
        sol.y       = Y
        sol.success = True
        sol.message = 'simulate_fast (operator-split semi-implicit) OK'
        return sol


    def extract_results(self, sol):
        """Extract named state arrays from the ODE solution object."""
        nz  = self.nz
        t   = sol.t
        res = {
            "t":        t,
            "theta":    sol.y[0:nz],
            "Pg":       sol.y[nz:2*nz],
            "Cm":       sol.y[2*nz:3*nz],
            "Cf":       sol.y[3*nz:4*nz],
            "M":        sol.y[4*nz:5*nz],
            "Cs":       sol.y[5*nz:6*nz],
            "Ar":       sol.y[6*nz:7*nz],
            "Cm220":    sol.y[7*nz:8*nz],
            "Cf220":    sol.y[8*nz:9*nz],
            "eps_mem":  sol.y[9*nz:10*nz],
            "C_bs":     sol.y[10*nz],
            "C_gt":     sol.y[10*nz+1],
            "Pin":      sol.y[10*nz+2],
            "C_Po218":  sol.y[10*nz+3],
            "C_Pb214":  sol.y[10*nz+4],
            "C_Bi214":  sol.y[10*nz+5],
            "C_Po214":  sol.y[10*nz+6],
        }
        try:
            res["J_exhalation"] = self.exhalation_flux_fick(res)   # [Bq m-2 s-1]
        except Exception:
            res["J_exhalation"] = None
        return res

    def exhalation_flux_fick(self, res, i=None):
        """Surface Rn-222 exhalation flux via Fick's law, J [Bq m-2 s-1], always >= 0."""
        Cm_all    = np.maximum(np.asarray(res["Cm"],    dtype=float), 0.0)
        theta_all = np.clip(np.asarray(res["theta"], dtype=float),
                            1e-8, 0.95 * self.p["phi_m"])
        dz = self.dz

        def _J_at(k):
            theta_col = theta_all[:, k]
            Cm_col    = Cm_all[:, k]
            # regime-branched BULK diffusivity — the SAME coefficient the solver
            # transports with (using the raw R&N correlation here mis-reported J
            # by orders of magnitude on fracture/karst lithotypes). This
            # estimator returns the MATRIX diffusive component only; the
            # complete flux (matrix+fracture, diffusive+advective) is
            # soil_surface_flux.
            D0 = float(self._D_eff(theta_col, "m")[0])
            # boundary-consistent gradient: the solver imposes Dirichlet C=0 at the
            # surface face, dz/2 above the first cell centre
            grad = 2.0 * Cm_col[0] / dz
            return max(D0 * grad, 0.0)

        if i is not None:
            return _J_at(int(i))
        n = Cm_all.shape[1]
        return np.array([_J_at(k) for k in range(n)])

    def exhalation_flux_plateau(self, res, t0_days=5.0, t1_days=10.0):
        """Mean and std of the exhalation flux over [t0_days, t1_days], in Bq m-2 s-1."""
        t = np.asarray(res["t"], dtype=float)
        J = self.exhalation_flux_fick(res)
        mask = (t >= t0_days * 86400.0) & (t <= t1_days * 86400.0)
        if not np.any(mask):
            mask = np.ones_like(t, dtype=bool)
        return float(np.mean(J[mask])), float(np.std(J[mask]))

    def emanation_vesicular(self, theta, T_mean=None):
        """Recoil emanation fraction for volcanic grains with internal vesicles.

        Surface-recoil term for a sphere of radius r >> R: eps = (3/4)*(R/r)
        (Fluegge & Zimens 1939; Semkow 1990), with R_s ~ 34 nm the Rn-222 recoil
        range in the MINERAL (not in air). Internal vesicle surfaces multiply the
        effective emitting surface. Pore water increases emanation by stopping the
        recoiling atom in the water film instead of the opposite grain
        (Sakoda et al. 2011, Appl. Radiat. Isot. 69:1422)."""
        theta = np.asarray(theta, dtype=float)
        phi_m = self.p["phi_m"]
        Sw    = np.clip(theta / max(phi_m, 1e-6), 0.0, 1.0)

        r_grain = self.p["grain_size_mm"] * 1e-3 / 2.0   # [m]
        R_s     = 34e-9    # Rn-222 recoil range in the mineral [m] (~30-40 nm)

        f_v   = self.p.get("vesicle_fraction", 0.0)

        # direct surface recoil from the outer grain surface
        eps_surf = min(1.0, 0.75 * R_s / max(r_grain, 1e-12))

        # internal vesicles add emitting surface: specific surface scales with
        # f_v/r_ves; effective enhancement factor ~ f_v*(r_grain/r_ves)
        if f_v > 0:
            r_ves   = max(r_grain * (f_v ** (1.0/3.0)), 10.0*R_s)
            eps_ves = min(1.0, 0.75 * R_s / r_ves) * f_v
        else:
            eps_ves = 0.0

        eps_dry = np.clip(eps_surf + eps_ves, 0.0, 1.0)
        # moisture enhancement: recoil termination in pore water raises emanation
        # from its dry value toward a saturated plateau (factor 2-4 typical)
        f_wet_max = self.p.get("eman_wet_factor", 3.0)
        Sw_c      = self.p.get("eman_Sw_c", 0.10)
        moist     = 1.0 + (f_wet_max - 1.0) * (1.0 - np.exp(-Sw / max(Sw_c, 1e-6)))
        return np.clip(eps_dry * moist, 0.0, 1.0)

    def geothermal_T_field(self, t=0.0, T_surface_t=None):
        """Soil temperature field: geothermal gradient plus damped diurnal/seasonal waves."""
        # Carslaw & Jaeger (1959) solution, skin depth z0 = sqrt(2*kappa/omega)
        import math
        if T_surface_t is None:
            T_surface_t = self.p.get("T0", 288.0)

        kappa = 5e-7    # Thermal diffusivity of volcanic rock [m^2/s]

        dTdz      = self.p.get("T_grad", 0.025)   # [K/m]
        T_geo     = T_surface_t + dTdz * self.z

        # Diurnal wave, z0_d ~ 0.15 m for rock
        T_day    = self.p.get("T_atm", 86400.0)
        A_day    = self.p.get("T_surf_amp", 8.0)
        phi_day  = 0.0
        omega_d  = 2.0 * math.pi / T_day
        z0_d     = math.sqrt(2.0 * kappa / omega_d)
        T_diurnal = (A_day * np.exp(-self.z / z0_d)
                     * np.sin(omega_d * t - self.z / z0_d + phi_day))

        # Seasonal wave, z0_s ~ 3 m for rock
        T_year   = 3.1536e7   # [s]
        A_season = 12.0
        phi_seas  = 0.0
        omega_s  = 2.0 * math.pi / T_year
        z0_s     = math.sqrt(2.0 * kappa / omega_s)
        T_seasonal = (A_season * np.exp(-self.z / z0_s)
                      * np.sin(omega_s * t - self.z / z0_s + phi_seas))

        return T_geo + T_diurnal + T_seasonal

    def geothermal_pe_enhancement(self, theta=None, Pg=None, t=0.0):
        """Peclet-number change from an elevated soil temperature.

        Physics: for a FIXED driving pressure, Pe = uH/D ~ 1/(mu(T)*D(T)) with
        mu ~ T^0.7 and D ~ T^1.75, so heating REDUCES Pe as (T/T_ref)^-2.45.
        The geothermal enhancement of advection comes instead from the buoyant
        overpressure dP_buoy ~ rho*g*H*(T - T_ref)/T_ref added to the driving
        pressure (Rayleigh-Darcy mechanism), which this function includes."""
        if theta is None:
            theta = np.full(self.nz, 0.4 * self.p["phi_m"])
        if Pg is None:
            Pg = np.full(self.nz, self.p.get("P_atm0", 101325.0))

        theta = np.asarray(theta, float)
        Pg    = np.asarray(Pg,    float)

        T_hot  = self.geothermal_T_field(t)
        T_mean = float(np.mean(T_hot))
        T_ref  = self.p.get("T_ref", 293.15)

        phi_m  = self.p["phi_m"]
        Sg     = np.maximum(phi_m - theta, 0.0) / phi_m
        phi_g  = np.maximum(phi_m * Sg, 1e-12)
        D0     = self.p.get("D0", 1.1e-5)
        # Millington-Quirk structure factor with Chapman-Enskog T^1.75 scaling
        D_ref  = D0 * phi_g**(10./3.) / phi_m**2
        D_hot  = D_ref * (T_mean / T_ref)**1.75
        D_mean     = float(np.mean(D_hot))
        D_mean_ref = float(np.mean(D_ref))

        mu_ref = self.p.get("mu", 1.82e-5)
        mu_hot = mu_ref * (T_mean / T_ref)**0.7

        k0    = self.p.get("k0", 1e-13)
        H     = self.p.get("H", 3.0)
        dP    = self.p.get("dP_init", 5.0)
        # thermal buoyancy overpressure driving upward gas flow
        rho_g   = self.p.get("P_atm0", 101325.0) * self.p.get("M_g", 0.02897) / \
                  (self.p.get("R_g", 8.314) * T_ref)
        dP_buoy = rho_g * self.p.get("g", 9.81) * H * max(T_mean - T_ref, 0.0) / T_ref

        u_hot = k0 / mu_hot * (dP + dP_buoy) / max(H, 1e-3)
        u_ref = k0 / mu_ref * dP / max(H, 1e-3)

        Pe_hot = u_hot * H / max(D_mean,     1e-12)
        Pe_ref = u_ref * H / max(D_mean_ref, 1e-12)
        Da_H   = self.lam_rn * H**2 / max(D_mean, 1e-12)

        # fixed-dP scaling: Pe ~ 1/(mu*D) ~ (T/T_ref)^-(0.7+1.75)
        Pe_enh_fixed_dP = (T_mean / T_ref)**(-2.45)
        Pe_enh_actual   = Pe_hot / max(Pe_ref, 1e-30)

        return {
            "Pe_H":                  Pe_hot,
            "Da_H":                  Da_H,
            "Pe_H_cold":             Pe_ref,
            "Pe_enhancement_actual": Pe_enh_actual,
            "Pe_enhancement_theory": Pe_enh_fixed_dP,
            "dP_buoyancy_Pa":        float(dP_buoy),
            "T_mean":                T_mean,
            "mu_hot":                mu_hot,
            "D_eff_mean":            D_mean,
            "regime": "advection" if Pe_hot > 1.0 else "diffusion",
            "scaling_law": ("fixed-dP: Pe(hot)/Pe(cold) = (T/T_ref)^-2.45 = "
                            f"{Pe_enh_fixed_dP:.3f}; with buoyancy dP_buoy = "
                            f"{dP_buoy:.2f} Pa the actual ratio is {Pe_enh_actual:.3f}"),
        }

    def barometric_pumping(self, t, theta, Pg_ambient=None):
        """Barometric pressure wave penetrating the soil, per Elberling et al. (1998)."""
        # pneumatic skin depth z_p = sqrt(2*k0*P_atm/(omega*mu*phi_g)); BE = exp(-H/z_p)
        import math
        theta = np.asarray(theta, float)
        # base state is the MEAN pressure: the oscillation is added once here,
        # with the same sin(omega t) phase convention as _Patm (no double counting)
        P_base = self.p.get("P_atm0", 101325.0) if Pg_ambient is None else float(Pg_ambient)

        phi_m = self.p["phi_m"]
        Sg    = np.maximum(phi_m - theta, 0.0) / phi_m
        phi_g = np.maximum(phi_m * Sg, 1e-3)  # Mean gas porosity

        omega   = 2.0 * math.pi / self.p.get("T_atm", 86400.0)
        k0      = self.p.get("k0", 1e-13)
        mu      = self.p.get("mu", 1.82e-5)
        phi_g_m = float(np.mean(phi_g))

        # Pneumatic skin depth [m]
        z_p = math.sqrt(2.0 * k0 * P_base
                        / max(omega * mu * phi_g_m, 1e-30))
        z_p = max(z_p, 1e-3)

        dP_amp  = self.p.get("P_atm_amp", 150.0)

        Pg_baro = (P_base
                   + dP_amp * np.exp(-self.z / z_p)
                   * np.sin(omega * t - self.z / z_p))

        H  = self.p.get("H", 3.0)
        BE = math.exp(-H / z_p)

        dPbaro_dz = np.zeros(self.nz)
        dPbaro_dz[1:-1] = (Pg_baro[2:] - Pg_baro[:-2]) / (2 * self.dz)
        dPbaro_dz[0]    = (Pg_baro[1]  - Pg_baro[0])   / self.dz
        dPbaro_dz[-1]   = (Pg_baro[-1] - Pg_baro[-2])  / self.dz

        u_baro = -(k0 / mu) * dPbaro_dz

        dP_surface = dP_amp * math.sin(omega * t)

        return {
            "Pg_field":        Pg_baro,
            "u_baro":          u_baro,
            "z_p_pneumatic":   z_p,
            "BE":              BE,
            "dP_surface":      dP_surface,
            "z_p_scaling":
                f"z_p = √(2k₀P_atm/ωμφ) = {z_p:.3f} m  (k₀={k0:.1e} m²)",
            "omega":           omega,
            "phi_g_mean":      phi_g_m,
        }

    def capillary_entry_pressure(self, theta, dtheta_dt=None):
        """Dynamic capillary entry pressure for dual-porosity two-phase flow."""
        # Barenblatt (2003) dynamic term P_c = P_c_eq + tau_cap*dSw/dt
        theta = np.asarray(theta, float)
        phi_m  = self.p["phi_m"]
        theta_r = min(self.p.get("theta_r", 0.03), 0.25 * phi_m)
        Se    = np.clip((theta - theta_r) / max(phi_m - theta_r, 1e-6),
                        0.0, 1.0)

        # Bimodal capillary pressure (macro + micro)
        P_e_mac = self.p.get("P_entry", 500.0)     # macro-pore entry pressure [Pa]
        P_e_mic = P_e_mac * 50.0                    # micro-pore entry pressure
        phi_mac = self.p.get("phi_f", 0.05)
        f_mac   = min(phi_mac / max(phi_m, 1e-6), 1.0)

        # P_c(Sw) bimodal (simplified Brooks-Corey)
        P_c_eq = (P_e_mac * np.power(np.maximum(Se, 1e-12), -1./1.5) * (1.0 - f_mac)
                + P_e_mic * np.power(np.maximum(Se, 1e-12), -1./3.0) * f_mac)
        P_c_eq = np.clip(P_c_eq, 0.0, 1e6)

        # Dynamic capillary pressure (Hassanizadeh & Gray 1993):
        # P_c_dyn = P_c_eq - tau * dSw/dt, so DRAINAGE (dSw/dt < 0) raises P_c
        tau_cap = self.p.get("tau_cap", 1800.0)    # dynamic coefficient [Pa s]
        if dtheta_dt is not None:
            dSe_dt = np.asarray(dtheta_dt, float) / max(phi_m - theta_r, 1e-6)
            P_c_dyn = P_c_eq - tau_cap * dSe_dt
        else:
            P_c_dyn = P_c_eq.copy()

        # Van Genuchten relative gas permeability at equilibrium (Parker 1987)
        Se_vg = self._Se_drainage(theta)
        n_vg  = self.p.get("soil_n", 1.5)
        m_vg  = 1.0 - 1.0/n_vg
        krg_eq = np.power(np.maximum(1.0 - Se_vg, 0.0), 0.5) * \
                 np.power(np.maximum(1.0 - Se_vg**( 1.0/m_vg), 0.0), 2.0*m_vg)
        krg_eq = np.clip(krg_eq, 0.0, 1.0)

        # Trapped gas during wetting (Land/Jerauld & Salter): the trapped gas
        # saturation is removed from the FLOWING gas fraction, i.e. krg is
        # evaluated at the effective (raised) wetting saturation Se + S_trg
        S_trg      = np.maximum(0.0, 0.25 * (1.0 - Se))
        Se_wet_eff = np.clip(Se_vg + S_trg, 0.0, 1.0)
        krg_wet = np.power(np.maximum(1.0 - Se_wet_eff, 0.0), 0.5) * \
                  np.power(np.maximum(1.0 - Se_wet_eff**(1.0/m_vg), 0.0), 2.0*m_vg)
        krg_wet = np.clip(krg_wet, 0.0, 1.0)

        return {
            "P_c_eq":  P_c_eq,
            "P_c_dyn": P_c_dyn,
            "krg_eq":  krg_eq,
            "krg_wet": krg_wet,
            "S_trg":   S_trg,
            "Se":      Se,
            "f_mac":   f_mac,
            "tau_cap": tau_cap,
        }

    def fracture_aperture_climate(self, theta_mean, T_mean, Pg_mean):
        """Effective fracture aperture as function of saturation, temperature, and stress."""
        # cubic law k_f = b_eff^2/12 (Snow 1965); Terzaghi-Biot effective stress
        phi_m  = self.p["phi_m"]
        Sw     = float(np.clip(theta_mean / max(phi_m, 1e-6), 0.0, 1.0))
        T_ref  = self.p.get("T_ref", 293.15)

        b0      = self.p.get("b0_fracture_mean", 2e-4)  # [m] reference aperture
        c_sat   = 0.6     # Closure due to saturation
        c_T     = 2e-4    # thermal expansion coefficient [1/K]
        c_sig   = 5e-8    # Compliance [1/Pa]
        b_min   = 1e-7    # Minimum physical aperture [m]

        # Effective stress (simplified Terzaghi-Biot)
        sigma_tot = self.p.get("sigma_tot_frac", 3e5)  # [Pa] - lithostatic stress
        biot      = 0.7
        sigma_eff = sigma_tot - biot * float(Pg_mean)
        sigma_eff = max(sigma_eff, 0.0)

        thermal  = max(1.0 + c_T * (T_mean - T_ref), 0.05)
        b_eff    = b0 * np.exp(-c_sat * Sw) * thermal * np.exp(-c_sig * sigma_eff)
        b_eff    = max(b_eff, b_min)

        k_f_eff = b_eff**2 / 12.0

        chi0  = 1.0
        a_sat = 2.5
        a_sig = 1e-7
        chi_eff = (chi0 * np.exp(-a_sat * Sw)
                   * np.exp(-a_sig * sigma_eff))
        chi_eff = max(chi_eff, 1e-12)

        return {
            "b_eff":      b_eff,
            "k_f_eff":    k_f_eff,
            "chi_eff":    chi_eff,
            "sigma_eff":  sigma_eff,
            "Sw":         Sw,
            "thermal_factor": thermal,
        }

    def transport_regime(self, Pe_H, Da_H, phi_tot=None):
        """Classify radon transport regime in (Pe, Da, phi) space."""
        # Bear (1972); Nazaroff (1992), with an added porosity axis
        if phi_tot is None:
            phi_tot = self.p["phi_m"] + self.p.get("phi_f", 0.05)

        phi_mac  = self.p.get("phi_f", 0.05)
        f_mac    = phi_mac / max(phi_tot, 1e-6)
        baro_num = self.p.get("P_atm_amp", 150.0) / self.p.get("P_atm0", 101325.0)

        if Pe_H > 1.0:
            regime = 1
            name   = "Avvettivo (Pe > 1)"
            L_star = "H — trasporto sull'intera colonna"
            C_scaling = "C_in ~ Pe_H · Π_src · Π_geo"
            insight = (
                "Regime Pe > 1: l'advection controlla il trasporto. "
                "C_in ∝ k₀·ΔP/(μ·ACH·V). Aumentare k₀ di 10× → C_in×10. "
                "Questo regime è RARO nei suoli non-vulcanici (Pe << 1). "
                "Spiega perché gli edifici su tufo saldato hanno C_in 10-100× "
                "superiore agli edifici su terreno sabbioso."
            )
        elif Da_H > 100.0:
            regime = 2
            name   = "Limitato da decadimento (Da > 100)"
            L_star = "√(D/λ) — lunghezza di penetrazione Rn"
            C_scaling = "C_in ~ Da_H^{-1/2} · Π_src"
            insight = (
                "Il decadimento controlla. Il Rn decade prima di raggiungere "
                "la superficie. C_in ∝ (D/λ)^0.5 · S_vol. "
                "Materiali vulcanici: D_eff > D_argilla → penetrazione maggiore "
                "→ C_in più alta a parità di sorgente."
            )
        elif Da_H < 0.01:
            regime = 3
            name   = "Limitato da sorgente (Da << 1)"
            L_star = "H — l'intera colonna contribuisce"
            C_scaling = "C_in ~ Π_src · φ_tot · H"
            insight = "C_in ∝ A_Ra · ρ_b · ε · H. Dipende quasi solo dalla sorgente."
        elif f_mac > 0.15 and Pe_H > 0.1:
            regime = 4
            name   = "Bypass di frattura (φ_mac > 0.15)"
            L_star = "b_eff³/12 · frac_density"
            C_scaling = "C_in ~ χ · f_frac · Pe_H · Π_src"
            insight = (
                "Il canalizzamento attraverso fratture domina. "
                "C_in amplificato dal fattore χ = exp(3σ²_b). "
                "Rocce vulcaniche: σ_b > 0.5 → χ > 3. "
                "Spiega i valori anomali nelle costruzioni su tufo saldato fratturato."
            )
        elif Pe_H < 0.1 and baro_num > 0.01:
            # barometric dominance requires unusually large pressure swings
            # (amp > ~1000 Pa); the previous 0.001 threshold was always met by
            # the default 150 Pa amplitude, making the diffusive branch dead code
            regime = 5
            name   = "Dominato da pompaggio barometrico"
            L_star = "z_p = √(2k₀P_atm/ωμφ)"
            C_scaling = "C_in ~ BE · Π_baro · Π_src"
            insight = (
                "Il pompaggio barometrico guida il trasporto. "
                "C_in oscilla con periodo diurno. BE = exp(-H/z_p). "
                "Suoli vulcanici: z_p >> H → BE alto → segnale barometrico forte."
            )
        elif Pe_H < 0.1 and Da_H < 10.0:
            regime = 0
            name   = "Diffusivo (Pe < 0.1)"
            L_star = "√(D/λ) — lunghezza di penetrazione Rn"
            C_scaling = "C_in ~ Da_H^{-1/2} · Π_src · Π_geo"
            insight = (
                "Regime diffusivo standard. Come i suoli non-vulcanici. "
                "D_eff maggiore nei materiali vulcanici → penetrazione maggiore "
                "→ C_in superiore a parità di sorgente."
            )
        else:
            regime = 6
            name   = "Misto / Transizione"
            L_star = "Pe_H · L_diff (scala mista)"
            C_scaling = "C_in ~ Pe_H^0.5 · Da_H^{-0.5} · Π_src"
            insight = (
                "Zona di transizione. Diffusione e advection co-dominano. "
                "Sensibile a piccole variazioni di k₀ o ΔP. "
                "Monitorare Pe_H nel tempo per rilevare transizioni di regime."
            )

        return {
            "regime_id":        regime,
            "regime_name":      name,
            "L_star":           L_star,
            "C_scaling":        C_scaling,
            "physical_insight": insight,
            "Pe_H":             Pe_H,
            "Da_H":             Da_H,
            "phi_tot":          phi_tot,
            "f_macropore":      f_mac,
            "baro_number":      baro_num,
        }

    def lithotype_comparison(self):
        """Compare steady-state radon metrics across all volcanic lithotypes."""
        results = {}

        for lit_name, lit_data in VOLCANIC_LITHOTYPES.items():
            try:
                m = build_model(lit_name, override={"nz": 15})

                # 2-day simulation to estimate C_bs
                sol = m.simulate_fast(days=2.0, dt_out=3600.0)
                out = m.extract_results(sol)

                t   = np.array(out["t"])
                Cbs = np.array(out["C_bs"])
                Cgt = np.array(out["C_gt"])
                Ar  = np.array(out["Ar"])
                eps = np.array(out["eps_mem"])
                Pg  = np.array(out["Pg"])

                # Last 6h for steady-state values
                mask = t > t[-1] - 6*3600
                def m_last(a):
                    a2 = np.asarray(a)
                    if a2.ndim == 2: return float(np.nanmean(a2[:, mask]))
                    return float(np.nanmean(a2[mask]))

                Cbs_m = m_last(Cbs)
                theta_f = np.array(out["theta"])[:, -1]
                Pg_f    = Pg[:, -1]
                T_f     = m.T_field(sol.t[-1])

                # Buckingham pi numbers
                pi = m.buckingham_pi(theta_f, Pg_f, T_f)

                # Surface flux
                Cf_eff, _ = m._entry_concentration(
                    np.array(out["Cm"])[:, -1], np.array(out["Cf"])[:, -1])
                Pg0   = float(Pg_f[0]); Pg_b = float(Pg_f[-1])
                Pg_eff= np.sqrt(Pg0 * Pg_b)
                Pin   = m._Pin_steady(m._Patm(sol.t[-1]),
                                      m.p.get("T_in0", 293.15),
                                      m.p.get("T_out0", 283.15))
                J_c = m.crack_network_flux(Cf_eff, Cbs_m, Pg_eff, Pin,
                                            float(theta_f[0]),
                                            float(T_f[0]),
                                            m.p.get("T_in0", 293.15))
                J_f = m.foundation_2d_network_flux(Cf_eff, Cbs_m, Pg_eff, Pin,
                                                    float(theta_f[0]))
                J_tot = J_c + J_f

                regime = m.transport_regime(pi["Pe_H"], pi["Da_H"])

                # Fracture aperture
                frac_props = m.fracture_aperture_climate(
                    float(np.mean(theta_f)),
                    float(np.mean(T_f)),
                    float(np.mean(Pg_f))
                )

                # Vesicular emanation
                eps_ves = m.emanation_vesicular(theta_f, float(np.mean(T_f)))

                results[lit_name] = {
                    "label":       lit_data["label"],
                    "Cin_Bq_m3":   Cbs_m,
                    "C_gt_Bq_m3":  m_last(Cgt),
                    "Pe_H":        pi["Pe_H"],
                    "Da_H":        pi["Da_H"],
                    "J_surface":   J_tot,
                    "eps_mean":    m_last(eps),
                    "eps_vesicular": float(np.mean(eps_ves)),
                    "Ar_mean":     m_last(Ar),
                    "regime":      regime["regime_name"],
                    "b_eff_um":    frac_props["b_eff"] * 1e6,  # [um]
                    "k_f_eff":     frac_props["k_f_eff"],
                    "phi_tot":     m.p["phi_m"] + m.p.get("phi_f", 0.05),
                    "k0":          m.p["k0"],
                    "A_Ra":        m.p["Ra226_Bqkg"],
                    "rho_bulk":    m.p["rho_bulk"],
                }
            except Exception as e:
                results[lit_name] = {"label": lit_data["label"], "error": str(e)}

        valid_res  = {k: v for k, v in results.items() if "Cin_Bq_m3" in v}
        sorted_res = dict(sorted(valid_res.items(),
                                  key=lambda x: x[1]["Cin_Bq_m3"],
                                  reverse=True))
        sorted_res.update({k: v for k, v in results.items() if "error" in v})
        return sorted_res



    def co2_magmatic_transport(self, Cm, x_CO2_in, Pg, theta, T_z, dt=0.0):
        """Coupled Rn-CO2 Stefan-Maxwell transport in volcanic soils, Bq/m^3/s."""
        import math

        Cm    = np.asarray(Cm,       dtype=float).copy()
        x_CO2 = np.asarray(x_CO2_in, dtype=float).copy()
        Pg    = np.asarray(Pg,        dtype=float)
        theta = np.asarray(theta,     dtype=float)
        T_z   = np.asarray(T_z,       dtype=float)

        nz    = self.nz
        dz    = self.dz
        phi_m = self.p["phi_m"]
        phi_f = self.p.get("phi_f", 0.05)
        phi_tot = phi_m + phi_f
        # Local gas saturation
        phi_g = np.maximum(phi_tot - theta, 1e-6)

        D0_CO2   = self.p.get("D0_CO2", 1.6e-5)    # [m^2/s] CO2 in air at 20C
        T_mean   = float(np.mean(T_z))
        T_ref    = self.p.get("T_ref", 293.15)
        P_mean   = float(np.mean(Pg))
        # D scales as T^1.75 / P, Chapman-Enskog kinetic theory
        D_CO2_ref = D0_CO2 * (T_mean / T_ref)**1.75 * (101325.0 / P_mean)

        # Millington-Quirk model for diffusion in partially saturated soil
        D_CO2_eff = D_CO2_ref * phi_g**(10./3.) / max(phi_tot**2, 1e-12)

        # Binary Rn-CO2 diffusivity: Chapman-Enskog mass dependence
        # D_AB ~ sqrt(1/M_A + 1/M_B), referenced to the Rn-air pair
        M_Rn  = 0.222   # [kg/mol]
        M_CO2 = 0.044   # [kg/mol]
        M_air = 0.02897 # [kg/mol]
        D0_Rn = self.p.get("D0", 1.1e-5)
        D_Rn_CO2 = D0_Rn * math.sqrt((1.0/M_Rn + 1.0/M_CO2) /
                                     (1.0/M_Rn + 1.0/M_air))

        S_CO2   = self.p.get("S_CO2", 1e-5)   # [mol/m^3/s]
        # deep magmatic source: concentrated in the LOWER third of the column
        # (grid index 0 is the SURFACE, index nz-1 the bottom)
        S_CO2_field = np.zeros(nz)
        n_src = max(1, nz // 3)
        for iz in range(n_src):
            S_CO2_field[nz - 1 - iz] = S_CO2 * math.exp(-3.0 * iz / max(n_src, 1))

        lap_CO2 = np.zeros(nz)
        lap_CO2[1:-1] = (x_CO2[2:] - 2.0*x_CO2[1:-1] + x_CO2[:-2]) / dz**2
        # atmospheric Dirichlet at the SURFACE (index 0) via the ghost node
        # x_{-1} = 2*x_surf - x_0 (boundary face at dz/2 above the first centre)
        x_surf        = self.p.get("x_CO2_surface", 4e-4)       # ~400 ppm
        lap_CO2[0]    = (x_CO2[1] - 3.0*x_CO2[0] + 2.0*x_surf) / dz**2
        # zero-flux (Neumann) at the deep boundary
        lap_CO2[-1]   = (x_CO2[-2] - x_CO2[-1]) / dz**2

        # mole-fraction balance in the gas phase: phi_g dx/dt = D_e lap(x) + S/n_tot
        # with n_tot = P/(R T) the molar gas density [mol/m^3]
        Rgas_  = self.p.get("R_g", 8.314)
        n_tot  = max(P_mean / (Rgas_ * T_mean), 1e-6)
        dx_CO2_dt = (D_CO2_eff * lap_CO2 + S_CO2_field / n_tot) / np.maximum(phi_g, 1e-6)

        dx_dz = np.zeros(nz)
        dx_dz[1:-1] = (x_CO2[2:] - x_CO2[:-2]) / (2.0 * dz)
        dx_dz[0]    = (x_CO2[1] - x_CO2[0]) / dz
        dx_dz[-1]   = (x_CO2[-1] - x_CO2[-2]) / dz

        # Stefan-Maxwell carrier flux term from CO2 gradient dragging Rn
        # J_SM = -phi_g * C_Rn * (D_Rn_CO2 / (1 - x_CO2)) * dx_dz/dz
        x_safe    = np.clip(x_CO2, 0.0, 0.95)
        factor_SM = D_Rn_CO2 / np.maximum(1.0 - x_safe, 0.05)
        J_Rn_SM   = -phi_g * Cm * factor_SM * dx_dz     # [Bq/m^2/s]

        div_J_SM = np.zeros(nz)
        div_J_SM[1:-1] = (J_Rn_SM[2:] - J_Rn_SM[:-2]) / (2.0 * dz)
        div_J_SM[0]    = (J_Rn_SM[1] - J_Rn_SM[0]) / dz
        div_J_SM[-1]   = (J_Rn_SM[-1] - J_Rn_SM[-2]) / dz
        dC_carrier = -div_J_SM   # [Bq/m^3/s]

        k0      = self.p.get("k0", 1e-13)
        mu      = self.p.get("mu", 1.82e-5)
        H       = self.p.get("H", 3.0)
        g       = self.p.get("g", 9.81)
        Rgas    = self.p.get("R_g", 8.314)
        x_max   = float(np.max(np.clip(x_CO2, 0.0, 0.95)))
        # Density difference between CO2-rich zone (bottom) and pure air (top)
        rho_CO2mix = P_mean / (Rgas * T_mean) * (
            M_CO2 * x_max + M_air * (1.0 - x_max))
        rho_air    = P_mean * M_air / (Rgas * T_mean)
        Delta_rho  = max(rho_CO2mix - rho_air, 0.0)
        # Ra_v = k * rho_diff * g * H / (mu * D_CO2)
        D_CO2_mean  = float(np.mean(D_CO2_eff))
        Ra_volcanic = (k0 * Delta_rho * g * H
                       / max(mu * D_CO2_mean, 1e-30))

        # critical Rayleigh-Darcy number 4*pi^2 ~ 39.5 (Lapwood 1948)
        Ra_crit   = 4.0 * math.pi**2
        convective = Ra_volcanic > Ra_crit

        if convective:
            u_conv   = D_CO2_mean * (Ra_volcanic - Ra_crit) / H
            chi_conv = math.sqrt(Ra_volcanic / Ra_crit)
        else:
            u_conv   = 0.0
            chi_conv = 1.0

        # Dalton's law: dilution of Rn partial pressure by CO2
        C_Rn_eff = Cm * (1.0 - x_safe)

        return {
            "x_CO2":       x_CO2,
            "dx_CO2_dt":   dx_CO2_dt,
            "J_Rn_SM":     J_Rn_SM,
            "dC_carrier":  dC_carrier,
            "C_Rn_eff":    C_Rn_eff,
            "Ra_volcanic": float(Ra_volcanic),
            "Ra_critical": Ra_crit,
            "convective":  bool(convective),
            "chi_conv":    float(chi_conv),
            "u_conv":      float(u_conv),
            "D_CO2_eff":   D_CO2_eff,
            "D_Rn_CO2":    D_Rn_CO2,
        }

    def bifurcation_analysis(self, dP_array=None, n_points=40,
                              theta_sw=None, T_mean=None,
                              return_regime_map=False):
        """C_in(dP) bifurcation diagram: diffusion-dominated vs advection regime.
        Critical dP* = mu*D_eff/(k0*H) separates the two scaling regimes."""
        import math

        if dP_array is None:
            dP_array = np.logspace(-1, 2, n_points)   # 0.1-100 Pa
        dP_array = np.asarray(dP_array, dtype=float)
        n = len(dP_array)

        if theta_sw is None:
            theta_sw = 0.35 * self.p["phi_m"]
        if T_mean is None:
            T_mean = self.p.get("T_ref", 293.15)

        phi_m   = self.p["phi_m"]
        phi_f   = self.p.get("phi_f", 0.05)
        phi_tot = phi_m + phi_f
        Sg      = max(phi_tot - theta_sw, 0.0) / max(phi_tot, 1e-12)
        phi_g   = max(phi_tot * Sg, 1e-4)
        D0      = self.p.get("D0", 1.1e-5)
        # Millington-Quirk BULK value and its APPARENT counterpart D_a = D_e/beta
        # (Pe, Da and the Pe=1 bifurcation criterion live on D_a; the diffusive
        # entry flux J_d = D_e*C/H lives on the bulk value)
        beta_bif = max(phi_g + 0.25 * theta_sw, 1e-6)
        D_bulk  = D0 * phi_g**(10./3.) / max(phi_tot**2, 1e-12)
        D_eff   = D_bulk / beta_bif
        k0      = self.p.get("k0", 1e-13)
        mu      = self.p.get("mu", 1.82e-5)
        H       = self.p.get("H", 3.0)
        lam_rn  = self.lam_rn

        # Pe_H = u*H/D_a = (k0*dP/(mu*H))*H/D_a = 1  =>  dP* = mu*D_a/k0 (H cancels)
        dP_crit_anal = mu * D_eff / max(k0, 1e-30)
        Da_H_ref     = lam_rn * H**2 / max(D_eff, 1e-30)

        nz     = self.nz
        T_arr  = np.full(nz, T_mean)
        theta  = np.full(nz, theta_sw)
        Ar0    = np.full(nz, 0.75)
        eps0   = np.full(nz, self.p.get("eps_max", 0.3) * 0.6)
        S222   = self.source_rn222(theta, T_arr, Ar0, eps0)   # [Bq/m^3/s]
        Cm_eq  = S222 / max(lam_rn, 1e-30)                    # [Bq/m^3]
        Cf_eq  = Cm_eq.copy()   # pore-gas equilibrium equal in both continua

        # Pure diffusive limit (dP = 0): entry flux J_d = D_eff*C_eq/H through
        # A_eff, diluted into V_bs against ventilation+decay [Bq/m^3]
        V_bs  = self.p.get("V_bs", 50.0)
        ach   = self.p.get("ACH_closed", 0.1) / 3600.0
        lam_v = lam_rn + ach
        A_eff = self.p.get("A_crack_m2", 0.05)
        C_diff_limit = (D_bulk * float(np.mean(Cm_eq)) / max(H, 1e-3)) * A_eff \
                       / max(V_bs * lam_v, 1e-30)

        Patm  = self.p.get("P_atm0", 101325.0)
        Tin   = self.p.get("T_in0", 293.15)
        T0s   = T_mean

        Cin_arr  = np.zeros(n)
        Pe_arr   = np.zeros(n)
        Da_arr   = np.full(n, Da_H_ref)
        J_tot_arr = np.zeros(n)
        J_diff_arr= np.zeros(n)
        J_adv_arr = np.zeros(n)

        Cf0 = float(np.mean(Cf_eq[:3]))   # mean of the TOP (near-surface) layers

        for i, dP in enumerate(dP_array):
            Pg_bot = Patm + float(dP)
            Pg_top = Patm
            Pg_eff = math.sqrt(Pg_top * Pg_bot)
            Pin    = Patm - 1.5   # Quasi-stationary indoor overpressure

            u_Darcy = k0 * float(dP) / (mu * H)
            Pe_i    = u_Darcy * H / max(D_eff, 1e-30)

            J_d  = D_bulk * float(np.mean(Cm_eq)) * A_eff / H

            # Upwind: inlet concentration set by Cf_eff. Advective entry rate
            # J_a = u * C * A [Bq/s]: u_Darcy is the superficial velocity, so no
            # extra porosity factor; no ad hoc pressure-ratio factor either.
            Cf_bot = float(np.mean(Cf_eq[-3:]))
            Cf_eff = 0.3 * Cf0 + 0.7 * Cf_bot
            J_a    = u_Darcy * Cf_eff * A_eff

            J_c = self.crack_network_flux(Cf_eff, Cin_arr[i-1] if i>0 else 0.0,
                                           Pg_eff, Pin, theta_sw, T0s, Tin)
            J_f = self.foundation_2d_network_flux(Cf_eff, Cin_arr[i-1] if i>0 else 0.0,
                                                    Pg_eff, Pin, theta_sw)
            J_net = J_c + J_f
            if J_net < 0:
                J_net = max(J_d, 0.0)

            J_diff_arr[i]  = float(J_d)
            J_adv_arr[i]   = float(max(J_a, 0.0))
            J_tot_arr[i]   = float(J_net)

            C_ss       = J_net / max(V_bs * lam_v, 1e-30)
            Cin_arr[i] = max(C_ss, 0.0)
            Pe_arr[i]  = Pe_i

        dCin_ddP = np.gradient(Cin_arr, dP_array)
        idx_star = int(np.argmax(np.abs(dCin_ddP)))
        dP_num   = float(dP_array[idx_star])

        mask_adv = Pe_arr > 1.0
        alpha_adv = np.nan
        if np.sum(mask_adv) >= 3:
            lx = np.log(dP_array[mask_adv])
            ly = np.log(np.maximum(Cin_arr[mask_adv], 1e-10))
            c  = np.polyfit(lx, ly, 1)
            alpha_adv = float(c[0])

        regime_map = None
        if return_regime_map:
            n_th = 15
            theta_grid = np.linspace(0.05*phi_m, 0.95*phi_m, n_th)
            Pe_grid    = np.zeros((n_th, n))
            for it, th in enumerate(theta_grid):
                Sg_t   = max(phi_tot - th, 0.0) / max(phi_tot, 1e-12)
                phi_gt = max(phi_tot * Sg_t, 1e-4)
                D_t    = D0 * phi_gt**(10./3.) / max(phi_tot**2, 1e-12)
                for j, dP in enumerate(dP_array):
                    u_t         = k0 * dP / (mu * H)
                    Pe_grid[it, j] = u_t * H / max(D_t, 1e-30)
            regime_map = {
                "theta_grid": theta_grid,
                "dP_array":   dP_array,
                "Pe_grid":    Pe_grid,
                "levels":     [0.1, 1.0, 10.0],
            }

        bif_str = (
            f"Bifurcation point: ΔP* = {dP_crit_anal:.3f} Pa (analytical), "
            f"{dP_num:.3f} Pa (numerical). "
            f"Advection scaling: C_in ∝ ΔP^{alpha_adv:.2f} "
            f"(theoretical: 1.0). Pe_H range: {Pe_arr.min():.3f}–{Pe_arr.max():.3f}."
        )

        return {
            "dP":              dP_array,
            "C_in":            Cin_arr,
            "Pe_H":            Pe_arr,
            "Da_H":            Da_arr,
            "J_total":         J_tot_arr,
            "J_diff":          J_diff_arr,
            "J_adv":           J_adv_arr,
            "dP_critical":     dP_crit_anal,
            "dP_numerical":    dP_num,
            "alpha_adv":       alpha_adv,
            "C_diffusion":     C_diff_limit,
            "D_eff":           D_eff,
            "regime_map":      regime_map,
            "bifurcation_str": bif_str,
        }

    def buckingham_pi_paper(self, n_samples=512, n_boot=200, seed=42,
                             return_samples=False):
        """Buckingham pi dimensional analysis with bootstrapped 95% CI on scaling exponents."""
        try:
            from scipy.stats import qmc as _qmc_mod
            _HAS_QMC = True
        except ImportError:
            _HAS_QMC = False
            _qmc_mod = None

        rng = np.random.default_rng(seed)

        param_ranges = {
            "k0":          (5e-16, 5e-10),
            "phi_m":       (0.08,  0.70),
            "phi_f":       (0.005, 0.35),
            "Ra226_Bqkg":  (5.0,   250.0),
            "rho_bulk":    (400.,  2800.),
            "H":           (0.3,   6.0),
            "dP_init":     (0.1,   100.0),
            "ACH_closed":  (0.02,  5.0),
            "A_crack_m2":  (1e-5,  0.1),
            "P_atm_amp":   (30.,   600.),
            "T_grad":      (0.005, 0.20),
            "grain_size_mm":(0.005, 2.0),
        }
        keys = list(param_ranges.keys())
        ndim = len(keys)

        if _HAS_QMC and _qmc_mod is not None:
            sampler = _qmc_mod.LatinHypercube(d=ndim, seed=int(seed))
            lhs_raw = sampler.random(n=n_samples)
        else:
            lhs_raw = rng.uniform(0.0, 1.0, (n_samples, ndim))

        lo_log = np.array([np.log10(max(v[0], 1e-30)) for v in param_ranges.values()])
        hi_log = np.array([np.log10(max(v[1], 1e-30)) for v in param_ranges.values()])
        X_log  = lhs_raw * (hi_log - lo_log) + lo_log
        X      = 10.0**X_log

        lam     = self.lam_rn
        mu_gas  = self.p.get("mu", 1.82e-5)
        Patm    = self.p.get("P_atm0", 101325.0)
        V_in    = self.p.get("V", 170.0)

        Cin_arr = np.zeros(n_samples)
        pi_arr  = np.zeros((n_samples, 8))

        for j in range(n_samples):
            pj = dict(zip(keys, X[j]))
            k0_j   = pj["k0"]
            phi_m_j= pj["phi_m"]
            phi_f_j= pj["phi_f"]
            phi_tot= phi_m_j + phi_f_j
            H_j    = pj["H"]
            dP_j   = pj["dP_init"]
            Ra_j   = pj["Ra226_Bqkg"]
            rb_j   = pj["rho_bulk"]
            ACH_j  = pj["ACH_closed"] / 3600.0
            A_j    = pj["A_crack_m2"]
            dPb_j  = pj["P_atm_amp"]
            dT_j   = pj["T_grad"]
            gs_j   = pj["grain_size_mm"]

            theta_j = 0.35 * phi_m_j
            Sg_j   = max(phi_tot - theta_j, 1e-3) / max(phi_tot, 1e-6)
            phi_g_j= max(phi_tot * Sg_j, 1e-4)
            D0_j   = 1.1e-5
            D_eff_j= D0_j * phi_g_j**(10./3.) / max(phi_tot**2, 1e-12)

            u_j  = k0_j * dP_j / (mu_gas * H_j)
            Pe_j = u_j * H_j / max(D_eff_j, 1e-30)

            # empirical grain-size emanation trend (finer grains emanate more,
            # Rogers & Nielson 1991); recoil-in-air ranges do not apply to solids
            eps_j   = float(np.clip(0.35 * (0.1 / max(gs_j, 1e-3))**0.25, 0.05, 0.5))

            # deep pore-air concentration with gas/water partition beta_p
            L_ost   = 0.25
            beta_j  = phi_g_j + L_ost * theta_j
            S_vol_j = Ra_j * rb_j * lam * eps_j          # [Bq/(m^3_bulk s)]
            Cm_j    = Ra_j * rb_j * eps_j / max(beta_j, 1e-6)   # [Bq/m^3 pore air]
            lam_eff = lam + ACH_j
            # entry rate: diffusive + advective; u_j is superficial, no extra phi
            J_j     = (D_eff_j * Cm_j / max(H_j, 1e-3) + u_j * Cm_j) * A_j
            Cin_j   = J_j / max(V_in * lam_eff, 1e-30)
            Cin_arr[j] = max(Cin_j, 1e-9)

            C_ref_soil_j = 1.0e4    # fixed reference [Bq/m^3], NOT the response
            pi_arr[j, 0] = lam * H_j**2 / max(D_eff_j, 1e-30)            # Da_H
            pi_arr[j, 1] = k0_j * dP_j / (mu_gas * lam * H_j**2)         # Pe_H/Da_H
            pi_arr[j, 2] = Cm_j / C_ref_soil_j                           # source group (non-circular)
            pi_arr[j, 3] = ACH_j / max(lam + ACH_j, 1e-30)               # ventilation efficiency
            pi_arr[j, 4] = A_j / max(V_in**(2.0/3.0), 1e-30)             # geometric coupling [-]
            pi_arr[j, 5] = k0_j / max(H_j**2, 1e-30)                     # permeability/geometry [-]
            pi_arr[j, 6] = dPb_j / max(dP_j, 1e-1)                       # barometric weight
            pi_arr[j, 7] = dT_j * H_j / 293.15                           # geothermal forcing

        valid  = (Cin_arr > 1e-6) & np.all(np.isfinite(pi_arr) & (pi_arr > 0), axis=1)
        n_valid= int(np.sum(valid))
        Cin_v  = Cin_arr[valid]
        pi_v   = pi_arr[valid]

        if n_valid < 20:
            return {"error": f"Troppo pochi campioni validi ({n_valid})."}

        C_ref_global = np.median(Cin_v)
        Cin_star = Cin_v / max(C_ref_global, 1e-6)

        log_pi = np.log10(np.maximum(pi_v, 1e-30))
        log_C  = np.log10(np.maximum(Cin_star, 1e-30))
        A_mat  = np.column_stack([np.ones(n_valid), log_pi])

        coef, _, _, _ = np.linalg.lstsq(A_mat, log_C, rcond=None)
        log_K     = float(coef[0])
        exponents = coef[1:]

        log_C_pred = A_mat @ coef
        SS_res = np.sum((log_C - log_C_pred)**2)
        SS_tot = np.sum((log_C - log_C.mean())**2)
        R2     = float(1.0 - SS_res / max(SS_tot, 1e-30))

        # Bootstrap for 95% CI on exponents
        boot_exp = np.zeros((n_boot, 8))
        boot_R2  = np.zeros(n_boot)
        boot_rng = np.random.default_rng(seed + 1)
        for b in range(n_boot):
            idx_b = boot_rng.integers(0, n_valid, size=n_valid)
            A_b   = A_mat[idx_b]
            C_b   = log_C[idx_b]
            cb, _, _, _ = np.linalg.lstsq(A_b, C_b, rcond=None)
            boot_exp[b] = cb[1:]
            pred_b = A_b @ cb
            ss_r = np.sum((C_b - pred_b)**2)
            ss_t = np.sum((C_b - C_b.mean())**2)
            boot_R2[b] = float(1.0 - ss_r / max(ss_t, 1e-30))

        ci95_lo = np.percentile(boot_exp, 2.5,  axis=0)
        ci95_hi = np.percentile(boot_exp, 97.5, axis=0)
        exponents_ci95 = np.column_stack([ci95_lo, ci95_hi])
        R2_ci95 = (float(np.percentile(boot_R2, 2.5)),
                   float(np.percentile(boot_R2, 97.5)))

        pi_std     = np.std(log_pi, axis=0)
        importance = np.abs(exponents) * pi_std
        rank_idx   = np.argsort(importance)[::-1]

        pi_names = ["Da_H", "Π_Darcy-decay", "Π_src", "Π_vent",
                    "Π_geo", "Π_perm", "Π_baro", "Π_geotherm"]

        terms_tex = " \\cdot ".join(
            f"{name}^{{{exp:+.3f}_{{{lo:+.2f}}}^{{{hi:+.2f}}}}}"
            for name, exp, lo, hi
            in zip(pi_names, exponents, ci95_lo, ci95_hi)
        )
        law_str_latex = (
            f"C_{{\\rm in}}^* \\approx "
            f"{10**log_K:.3e} \\cdot {terms_tex}"
            f"\\quad (R^2 = {R2:.3f})"
        )
        law_str_plain = (
            f"C_in* ≈ {10**log_K:.3e} · " +
            " · ".join(
                f"{n}^{e:+.3f}[{l:+.2f},{h:+.2f}]"
                for n,e,l,h in zip(pi_names, exponents, ci95_lo, ci95_hi)
            ) +
            f"  (R²={R2:.3f}, n={n_valid})"
        )

        out = {
            "exponents":       exponents,
            "exponents_ci95":  exponents_ci95,
            "log_K":           log_K,
            "R2":              R2,
            "R2_ci95":         R2_ci95,
            "pi_names":        pi_names,
            "importance":      importance,
            "rank":            [pi_names[i] for i in rank_idx],
            "law_string":      law_str_plain,
            "law_latex":       law_str_latex,
            "n_valid":         n_valid,
            "n_samples":       n_samples,
            "n_boot":          n_boot,
            "C_ref":           float(C_ref_global),
        }
        if return_samples:
            out["Cin_samples"] = Cin_v
            out["pi_samples"]  = pi_v
        return out

    def indoor_box_model_validated(self, J_surface, Cin0=50.0, t_end=86400.0,
                                    n_mc=500, seed=43, zone="basement",
                                    use_progeny=True):
        """Analytic well-mixed indoor box model for C_in(t) with MC uncertainty and progeny."""
        import math

        if zone == "basement":
            V    = self.p.get("V_bs", 50.0)
            ACH  = self.p.get("ACH_closed", 0.10)
        else:
            V    = self.p.get("V_gt", 120.0)
            ACH  = self.p.get("ACH_closed", 0.10) * 0.5

        A_c      = self.p.get("A_crack_m2", 0.05)
        lam_rn   = self.lam_rn
        lam_dep  = self.p.get("dep_bs" if zone=="basement" else "dep_gt", 0.0)
        lam_v    = ACH / 3600.0
        lam_eff  = lam_rn + lam_v + lam_dep

        J_eff  = float(J_surface) * A_c / max(V, 1e-3)   # [Bq/m^3/s]
        C_ss   = J_eff / max(lam_eff, 1e-30)
        tau    = 1.0 / max(lam_eff, 1e-30)
        t_90   = math.log(10.0) / max(lam_eff, 1e-30)

        n_t    = 200
        t_arr  = np.linspace(0.0, float(t_end), n_t)
        Cin_t  = C_ss + (float(Cin0) - C_ss) * np.exp(-lam_eff * t_arr)

        rng_mc = np.random.default_rng(seed)
        ACH_mc  = rng_mc.lognormal(math.log(ACH),  0.30, n_mc)
        V_mc    = rng_mc.normal(V,   V*0.10, n_mc)
        dep_mc  = rng_mc.lognormal(math.log(max(lam_dep, 1e-7)*3600.0), 0.40, n_mc) / 3600.0
        V_mc    = np.maximum(V_mc, 1.0)

        lam_eff_mc = lam_rn + ACH_mc/3600.0 + dep_mc
        C_ss_mc    = J_eff * V / np.maximum(V_mc * lam_eff_mc, 1e-30)
        Cin_end_mc = C_ss_mc + (float(Cin0) - C_ss_mc) * np.exp(-lam_eff_mc * t_end)

        # Progeny at secular equilibrium (Po-218, Pb-214, Bi-214, Po-214)
        lam_Po218 = DECAY_CONSTANTS["Po218"]
        lam_Pb214 = DECAY_CONSTANTS["Pb214"]
        lam_Bi214 = DECAY_CONSTANTS["Bi214"]
        lam_Po214 = DECAY_CONSTANTS.get("Po214", 4.23e3)
        lam_dep_p = self.p.get("dep_progeny", 1.5/3600.0)
        lam_eff_p = lam_v + lam_dep_p

        # activity-Bateman steady state: production term uses the DAUGHTER's lambda
        Crn_ss    = float(C_ss)
        C_Po218   = Crn_ss   * lam_Po218 / max(lam_Po218 + lam_eff_p, 1e-30)
        C_Pb214   = C_Po218  * lam_Pb214 / max(lam_Pb214 + lam_eff_p, 1e-30)
        C_Bi214   = C_Pb214  * lam_Bi214 / max(lam_Bi214 + lam_eff_p, 1e-30)
        C_Po214   = C_Bi214  * lam_Po214 / max(lam_Po214 + lam_eff_p, 1e-30)

        # Equilibrium factor F_eq, UNSCEAR 2000 weighting coefficients
        F_eq = ((0.105 * C_Po218 + 0.516 * C_Pb214 + 0.379 * C_Bi214)
                / max(Crn_ss, 1e-6))

        # Annual effective dose, WHO 2009 coefficient 9 nSv per (Bq h/m^3) of EEC:
        # dose[mSv/yr] = C_Rn * F_eq * 9e-6 [mSv/(Bq h m^-3)] * T_occ [h/yr]
        T_occ      = 7000.0   # h/yr residential occupancy, WHO 2009
        dose_mSv_yr= Crn_ss * F_eq * 9e-6 * T_occ

        return {
            "C_in_t":      Cin_t,
            "t_arr":       t_arr,
            "C_ss":        float(C_ss),
            "lambda_eff":  float(lam_eff),
            "tau":         float(tau),
            "t_90":        float(t_90),
            "C_mc_p5":     float(np.percentile(Cin_end_mc, 5)),
            "C_mc_p95":    float(np.percentile(Cin_end_mc, 95)),
            "C_mc_mean":   float(np.mean(Cin_end_mc)),
            "C_mc_std":    float(np.std(Cin_end_mc)),
            "progeny": {
                "C_Po218": float(C_Po218),
                "C_Pb214": float(C_Pb214),
                "C_Bi214": float(C_Bi214),
                "C_Po214": float(C_Po214),
            },
            "F_eq":        float(F_eq),
            "dose_mSv_yr": float(dose_mSv_yr),
            "zone":        zone,
            "V":           float(V),
            "ACH":         float(ACH),
            "A_crack":     float(A_c),
        }


    # Morris screening + centered finite-difference elasticities
    # Campolongo et al. (2007, Comput. Stat. Data Anal. 50:4021-4039)

    def sensitivity_screen(self, theta_mean=None, delta_frac=0.05,
                            n_morris=10, seed=77):
        """Two-level sensitivity analysis: elasticities and Morris screening.
        Score = 0.6*|elasticity| + 0.4*normalized mu_star (Morris 1991)."""
        import math

        if theta_mean is None:
            theta_mean = 0.35 * self.p["phi_m"]

        # steady-state response function, no time integration
        def compute_Cbs_fast(model):
            nz    = model.nz
            th    = np.full(nz, theta_mean)
            T_arr = model.T_field(0.0)
            Ar0   = np.full(nz, 0.75)
            eps0  = np.full(nz, model.p.get("eps_max", 0.3) * 0.6)
            S222  = model.source_rn222(th, T_arr, Ar0, eps0)
            Cm_eq = S222 / max(model.lam_rn, 1e-30)
            Cf_eq = Cm_eq   # pore-gas equilibrium equal in both continua
            Pg    = np.full(nz, model.p.get("P_atm0", 101325.0))
            Pg[-1]+= model.p.get("dP_init", 5.0)
            Pg_eff = math.sqrt(Pg[0] * Pg[-1])
            Pin    = model.p.get("P_atm0", 101325.0) - 1.5
            T0s    = float(model.T_field(0.0)[0])
            Tin    = model.p.get("T_in0", 293.15)
            Cf_eff = 0.3*float(Cf_eq[0]) + 0.7*float(Cf_eq[-1])
            J_c    = model.crack_network_flux(Cf_eff, 0.0, Pg_eff, Pin, theta_mean, T0s, Tin)
            J_f    = model.foundation_2d_network_flux(Cf_eff, 0.0, Pg_eff, Pin, theta_mean)
            V_bs   = model.p.get("V_bs", 50.0)
            ach    = model.p.get("ACH_closed", 0.1) / 3600.0
            lam_v  = model.lam_rn + ach
            return (J_c + J_f) / max(V_bs * lam_v, 1e-30)

        params_key = [
            "k0", "Ra226_Bqkg", "rho_bulk", "phi_m", "phi_f",
            "dP_init", "ACH_closed", "H", "T_grad",
            "eps_max", "grain_size_mm", "A_crack_m2"
        ]
        params_key = [p for p in params_key if p in self.p]

        C0 = compute_Cbs_fast(self)
        if C0 < 1e-8:
            return {"error": "C_bs nominale troppo basso", "C_nominal": float(C0)}

        n_evals = 0

        # centered finite-difference elasticity: dC/dp * p/C
        elasticities = {}
        for pk in params_key:
            p0 = float(self.p[pk])
            if abs(p0) < 1e-30:
                continue
            try:
                self.p[pk] = p0 * (1.0 + delta_frac)
                C_plus  = compute_Cbs_fast(self); n_evals += 1
                self.p[pk] = p0 * (1.0 - delta_frac)
                C_minus = compute_Cbs_fast(self); n_evals += 1
                elasticities[pk] = (C_plus - C_minus) / (2.0 * delta_frac * C0)
            except Exception:
                elasticities[pk] = 0.0
            finally:
                self.p[pk] = p0

        # Morris elementary effects, one-at-a-time trajectory
        rng_m = np.random.default_rng(seed)
        morris_EE = {pk: [] for pk in params_key}

        Delta_frac = 0.10   # Morris step = 10%

        for _ in range(n_morris):
            p_base = {pk: float(self.p[pk]) for pk in params_key}
            perm = rng_m.permutation(len(params_key)).tolist()
            p_curr = p_base.copy()

            for pk in params_key:
                self.p[pk] = p_curr[pk]
            try:
                C_curr = compute_Cbs_fast(self); n_evals += 1
            except Exception:
                for pk in params_key:
                    self.p[pk] = p_base[pk]
                continue

            for idx in perm:
                pk = params_key[idx]
                p_orig = p_curr[pk]
                direction = rng_m.choice([-1.0, 1.0])
                p_new  = p_orig * (1.0 + direction * Delta_frac)
                p_curr[pk] = p_new
                for k2 in params_key:
                    self.p[k2] = p_curr[k2]
                try:
                    C_new = compute_Cbs_fast(self); n_evals += 1
                    EE_i  = (C_new - C_curr) / (direction * Delta_frac * max(abs(p_orig), 1e-30))
                    morris_EE[pk].append(EE_i)
                    C_curr = C_new
                except Exception:
                    p_curr[pk] = p_orig
                    for k2 in params_key:
                        self.p[k2] = p_base[k2]
                    break

            for pk in params_key:
                self.p[pk] = p_base[pk]

        morris_mu_star = {}
        morris_sigma   = {}
        for pk in params_key:
            ee = np.asarray(morris_EE.get(pk, [0.0]))
            morris_mu_star[pk] = float(np.mean(np.abs(ee)))
            morris_sigma[pk]   = float(np.std(ee))

        mu_max = max(v for v in morris_mu_star.values()) or 1.0
        score  = {}
        for pk in params_key:
            eps_i  = abs(elasticities.get(pk, 0.0))
            mu_i   = morris_mu_star.get(pk, 0.0) / mu_max
            score[pk] = 0.6 * eps_i + 0.4 * mu_i

        ranking = sorted(
            [(pk, elasticities.get(pk, 0.0),
              morris_mu_star.get(pk, 0.0),
              morris_sigma.get(pk, 0.0),
              score.get(pk, 0.0))
             for pk in params_key],
            key=lambda x: x[4], reverse=True
        )

        return {
            "elasticities":   elasticities,
            "morris_mu_star": morris_mu_star,
            "morris_sigma":   morris_sigma,
            "score":          score,
            "ranking":        ranking,
            "C_nominal":      float(C0),
            "n_evals":        n_evals,
        }

    def co2_rn_cotransport(self, Cm, x_CO2_profile, Pg, theta, T_z):
        """Rn-CO2 cotransport flux via simplified binary Stefan-Maxwell coupling.
        Combines Millington-Quirk diffusion, carrier drag, and Dalton dilution."""
        Cm    = np.asarray(Cm,            dtype=float).copy()
        xCO2  = np.asarray(x_CO2_profile, dtype=float).copy()
        Pg_   = np.asarray(Pg,            dtype=float)
        theta_ = np.asarray(theta,        dtype=float)
        T_z_  = np.asarray(T_z,           dtype=float)

        nz    = self.nz
        dz    = self.dz
        phi_m = self.p["phi_m"]
        phi_f = self.p.get("phi_f", 0.05)
        phi_t = phi_m + phi_f
        D0    = self.p.get("D0", 1.1e-5)
        k0    = self.p.get("k0", 1e-13)
        mu    = self.p.get("mu", 1.82e-5)
        H     = self.p.get("H", 3.0)

        # local gas saturation from total porosity minus water content
        Sg_arr  = np.clip((phi_t - theta_) / np.maximum(phi_t, 1e-12), 0.0, 1.0)
        phi_g   = phi_t * Sg_arr

        # Millington-Quirk D_eff with Chapman-Enskog T^1.75/P scaling
        T_ref = 293.15; P_ref = 101325.0
        D_local = D0 * (T_z_ / T_ref)**1.75 * (P_ref / np.maximum(Pg_, 1.0))
        phi_g_eff = np.maximum(phi_g, 1e-4)
        D_eff = D_local * phi_g_eff**(10./3.) / np.maximum(phi_t**2, 1e-12)

        # Darcy gas velocity from pressure gradient
        v_gas = np.zeros(nz)
        for i in range(1, nz - 1):
            dPdz = (Pg_[i+1] - Pg_[i-1]) / (2.0 * dz)
            v_gas[i] = -k0 / mu * dPdz
        v_gas[0] = v_gas[1]; v_gas[nz-1] = v_gas[nz-2]

        # a trace gas advects with the FULL mixture Darcy velocity, regardless of
        # the CO2 mole fraction (the fraction enters the density/buoyancy, not u)
        u_carrier = v_gas

        # D_eff is the BULK (Millington-Quirk) effective diffusivity: no extra
        # porosity factor in the flux; u is superficial, likewise no extra phi
        J_diff = np.zeros(nz)
        for i in range(1, nz - 1):
            dCdz = (Cm[i+1] - Cm[i-1]) / (2.0 * dz)
            J_diff[i] = -D_eff[i] * dCdz
        # surface (index 0) Dirichlet C=0 at the boundary face dz/2 above centre 0:
        # dC/dz|surf = (Cm[0]-0)/(dz/2); J = -D*dC/dz < 0 (upward exhalation)
        J_diff[0] = -D_eff[0] * Cm[0] / (0.5 * dz)
        # deep boundary: zero-gradient
        J_diff[nz-1] = J_diff[nz-2]

        J_carrier = u_carrier * Cm

        # Dalton dilution by CO2 mole fraction
        C_eff = Cm * np.maximum(1.0 - xCO2, 0.0)

        u_mean = float(np.mean(np.abs(u_carrier)))
        D_mean = float(np.mean(D_eff))
        Pe_CO2 = u_mean * H / max(D_mean, 1e-30)

        J_total = J_diff + J_carrier
        enh_arr = np.abs(J_total) / np.maximum(np.abs(J_diff), 1e-30)
        enhancement = float(np.mean(enh_arr))

        return {
            "J_carrier":   J_carrier,
            "J_diffusion": J_diff,
            "C_effective": C_eff,
            "enhancement": enhancement,
            "Pe_CO2":      Pe_CO2,
            "D_eff":       D_eff,
            "u_carrier":   u_carrier,
        }

    def bifurcation_dP(self, dP_array=None, n_points=40, theta_sw=None,
                       T_mean=None):
        """C_in(dP) curve and diffusive/advective bifurcation point (Pe_H=1)."""
        bif = self.bifurcation_analysis(
            dP_array=dP_array,
            n_points=n_points,
            theta_sw=theta_sw,
            T_mean=T_mean,
            return_regime_map=False,
        )
        bif["Cin"] = bif["C_in"]
        return bif

    def buckingham_pi_extended(self, n_samples=512, seed=42):
        """Buckingham-pi analysis with 8 dimensionless groups via LHS + OLS.
        Uses a steady-state analytic C_in, not the full ODE physics."""
        from scipy.stats import qmc

        rng = np.random.default_rng(int(seed))

        # log-uniform LHS bounds: k0, dP, theta, phi_m, phi_f, eps_max, ACH, S_CO2
        log_lo = np.array([-18., -1., -2.0, -2.5, -3.5, -1.5, -3., -7.])
        log_hi = np.array([-11.,  3., -0.3, -0.7, -1.0, -0.3,  1., -3.])

        sampler  = qmc.LatinHypercube(d=8, seed=int(seed))
        unit_lhs = sampler.random(n=n_samples)
        log_samp = qmc.scale(unit_lhs, log_lo, log_hi)
        raw      = 10.0 ** log_samp

        k0_s    = raw[:, 0]; dP_s  = raw[:, 1]; theta_s = raw[:, 2]
        phi_m_s = raw[:, 3]; phi_f_s = raw[:, 4]; eps_s = raw[:, 5]
        ach_s   = raw[:, 6]; sCO2_s  = raw[:, 7]

        D0    = self.p.get("D0", 1.1e-5)
        D_CO2 = self.p.get("D0_CO2", 1.6e-5)
        mu    = self.p.get("mu", 1.82e-5)
        H     = self.p.get("H", 3.0)
        lam   = self.lam_rn
        V_bs  = self.p.get("V_bs", 50.0)
        C_ref = 100.0

        phi_t = phi_m_s + phi_f_s
        # sampled water content cannot exceed the (independently sampled) porosity
        theta_s = np.minimum(theta_s, 0.9 * phi_t)
        Sg    = np.clip((phi_t - theta_s) / np.maximum(phi_t, 1e-12), 0.0, 1.0)
        phi_g = phi_t * Sg
        D_eff = D0 * phi_g**(10./3.) / np.maximum(phi_t**2, 1e-12)
        u_adv = k0_s / mu * dP_s / H
        C_ref_soil = 1e4
        lam_v_s  = ach_s / 3600.0            # ACH sampled in 1/h -> 1/s
        rho_air  = 1.2                       # [kg/m^3]
        n_ref    = 101325.0 / (8.314 * 293.15)   # molar gas density [mol/m^3]

        pi1 = u_adv * H / np.maximum(D_eff, 1e-30)
        pi2 = lam * H**2 / np.maximum(D_eff, 1e-30)
        pi3 = theta_s / np.maximum(phi_m_s, 1e-12)
        pi4 = k0_s * dP_s / (mu * max(D0, 1e-30))
        pi5 = lam / np.maximum(lam_v_s, 1e-30)            # decay/ventilation, both [1/s]
        pi6 = np.sqrt(rho_air * dP_s * k0_s) / mu          # pore Reynolds number [-]
        pi7 = eps_s * phi_m_s / np.maximum(phi_f_s, 1e-12)
        pi8 = sCO2_s * H**2 / np.maximum(D_CO2 * n_ref, 1e-30)  # CO2 source group [-]

        pi_mat   = np.column_stack([pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8])
        pi_names = ["Pe_H", "Da_H", "theta/phi_m", "k0*dP/mu*D0",
                    "lambda/lambda_v", "Re_p", "eps/phi_f", "pi_CO2"]

        # emanation source per bulk volume: S = eps*rho_b*A_Ra*lam (no phi factor);
        # deep pore-air concentration includes the gas/water partition beta_p
        rho_b_ref = self.p.get("rho_bulk", 1600.0)
        Ra_ref    = self.p.get("Ra226_Bqkg", 50.0)
        L_ost     = 0.25
        beta_p    = phi_g + L_ost * theta_s
        Cm_eq = eps_s * rho_b_ref * Ra_ref / np.maximum(beta_p, 1e-6)
        J_diff_s = D_eff * Cm_eq / H
        J_adv_s  = u_adv * Cm_eq
        A_bs     = self.p.get("A_bs", 50.0)
        lam_tot_s = lam + lam_v_s
        C_in_s   = (J_diff_s + J_adv_s) * A_bs / np.maximum(lam_tot_s * V_bs, 1e-12)
        # wide clip: only guards against numerical blow-up, minimal censoring
        C_in_s   = np.clip(C_in_s, 1e-3, 1e8)

        eps_safe = 1e-30
        Y       = np.log10(C_in_s / C_ref)
        Pi_log  = np.log10(np.maximum(np.abs(pi_mat), eps_safe))
        X       = np.column_stack([np.ones(n_samples), Pi_log])
        beta_all = np.linalg.lstsq(X, Y, rcond=None)[0]
        beta0 = float(beta_all[0]); beta = beta_all[1:]
        Y_hat = X @ beta_all
        ss_res = np.sum((Y - Y_hat)**2)
        ss_tot = np.sum((Y - Y.mean())**2)
        R2 = 1.0 - ss_res / max(ss_tot, 1e-30)

        # bootstrap CI, 200 resamples
        n_boot = 200
        beta_boot = np.zeros((n_boot, 8))
        rng_b = np.random.default_rng(int(seed) + 1000)
        for b in range(n_boot):
            idx_b = rng_b.integers(0, n_samples, size=n_samples)
            Xb = X[idx_b]; Yb = Y[idx_b]
            bb = np.linalg.lstsq(Xb, Yb, rcond=None)[0]
            beta_boot[b] = bb[1:]
        beta_ci_lo = np.percentile(beta_boot, 2.5,  axis=0)
        beta_ci_hi = np.percentile(beta_boot, 97.5, axis=0)

        terms = [f"pi_{i+1}^{{{b:.3f}}}" for i, b in enumerate(beta)]
        law = (f"C_in/C_ref = 10^{{{beta0:.3f}}} * " + " * ".join(terms))

        pi_stats = {
            pi_names[i]: {
                "mean":   float(np.mean(pi_mat[:, i])),
                "median": float(np.median(pi_mat[:, i])),
                "p5":     float(np.percentile(pi_mat[:, i], 5)),
                "p95":    float(np.percentile(pi_mat[:, i], 95)),
            }
            for i in range(8)
        }

        return {
            "pi_names":     pi_names,
            "samples":      pi_mat,
            "C_in_samples": C_in_s,
            "beta":         beta,
            "beta0":        beta0,
            "beta_ci_lo":   beta_ci_lo,
            "beta_ci_hi":   beta_ci_hi,
            "R2":           float(R2),
            "law":          law,
            "pi_stats":     pi_stats,
            "n_samples":    n_samples,
        }

    def buckingham_pi_extended_ode(self, n_samples=40, seed=42, nz=15,
                                    days=1.5, lits=None):
        """Buckingham-pi scaling law fit using C_in from the full ODE solver.
        Samples lithotypes in GEOLITHOTYPES via LHS over dP, Ra226, ACH."""
        import warnings
        if lits is None:
            lits = list(GEOLITHOTYPES.keys())
        rng = np.random.default_rng(int(seed))

        X_ode = []; Y_ode = []; lits_run = []
        # 7th feature is the mean water SATURATION Sw (not the residual theta_r)
        pi_names = ["Pe_H", "DaAdv", "DaDiff", "Pi_src", "Pi_vent",
                    "Pi_frac", "Sw", "eps_phi"]

        for _ in range(n_samples):
            lit_name = rng.choice(lits)
            lit_data = GEOLITHOTYPES[lit_name]
            Ra_base  = lit_data.get("Ra226_Bqkg", 40.0)
            override = {
                "nz":         nz,
                "dP_init":    float(rng.uniform(0.5, 25.0)),
                "Ra226_Bqkg": float(rng.uniform(Ra_base * 0.4,
                                                 Ra_base * 2.5)),
                "ACH_closed": float(rng.uniform(0.05, 0.5)),
            }
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    m   = build_model(lit_name, override=override)
                    sol = m.simulate_fast(days=days, dt_out=3600.0)
                out   = m.extract_results(sol)
                t_arr = np.array(out["t"])
                Cbs   = np.array(out["C_bs"])
                mask  = t_arr > t_arr[-1] - 4*3600
                C_in  = float(np.nanmean(Cbs[mask]))
                if C_in > 0:
                    theta_f = np.array(out["theta"])[:, -1]
                    Pg_f    = np.array(out["Pg"])[:, -1]
                    T_f     = m.T_field(sol.t[-1])
                    pi      = m.buckingham_pi(theta_f, Pg_f, T_f)
                    eps_phi = (m.p.get("eps_max", 0.2) * m.p["phi_m"]
                               / max(m.p.get("phi_f", 1e-4), 1e-12))
                    Sw_f    = float(np.mean(theta_f / max(m.p["phi_m"], 1e-12)))
                    row = [pi["Pe_H"], pi["DaAdv"], pi["DaDiff"],
                           pi["Pi_src"], pi["Pi_vent"], pi["Pi_frac"],
                           Sw_f, eps_phi]
                    X_ode.append(row)
                    Y_ode.append(C_in)
                    lits_run.append(lit_name)
            except Exception:
                pass

        n_conv = len(Y_ode)
        if n_conv < 6:
            return {"error": f"Solo {n_conv} simulazioni convergenti (<6)"}

        X_arr  = np.array(X_ode)
        Y_arr  = np.array(Y_ode)
        eps_   = 1e-30
        logY   = np.log10(np.maximum(Y_arr, eps_))
        logX   = np.log10(np.maximum(np.abs(X_arr), eps_))
        Xmat   = np.column_stack([np.ones(n_conv), logX])
        coef, _, _, _ = np.linalg.lstsq(Xmat, logY, rcond=None)
        Yhat   = Xmat @ coef
        R2     = float(1.0 - np.sum((logY - Yhat)**2)
                       / max(np.sum((logY - logY.mean())**2), eps_))
        terms  = " * ".join([f"{pi_names[i]}^{{{coef[i+1]:.3f}}}"
                              for i in range(len(pi_names))])
        law    = f"C_in/Cref = 10^{coef[0]:.3f} * " + terms

        return {
            "pi_names":    pi_names,
            "C_in_ode":    Y_arr,
            "lits_run":    lits_run,
            "n_converged": n_conv,
            "coef":        coef,
            "R2":          R2,
            "law":         law,
        }

    def q2_leave_one_out(self, bench=None, nz=15, days=1.5, verbose=True):
        """Leave-one-lithology-out Q2 predictive validation of the scaling law.
        Golbraikh & Tropsha (2002) require Q2 > 0.5 for external predictivity;
        the stricter 0.70 threshold used here is an internal quality criterion."""
        import warnings

        lits_all = list(GEOLITHOTYPES.keys())
        N = len(lits_all)

        if bench is None:
            if verbose:
                print("  Q2_LOO: esecuzione benchmark su tutti i 20 litotipi...")
            bench = run_lithology_benchmark(nz=nz, days=days, verbose=verbose)

        Pi_all = []
        Cin_all = []
        valid_lits = []

        for lit_name in lits_all:
            r = bench.get(lit_name, {})
            if "error" in r or not np.isfinite(r.get("Cin_Bq_m3", float("nan"))):
                if verbose:
                    print(f"  {lit_name}: benchmark saltato")
                continue
            Pe   = max(r.get("Pe_H", 1e-10), 1e-10)
            Da   = max(r.get("Da_H", 1e-10), 1e-10)   # Da_H equals DaAdv here
            Cin  = max(r.get("Cin_Bq_m3", 0.0), 0.1)
            Pi_all.append([Pe, Da])
            Cin_all.append(Cin)
            valid_lits.append(lit_name)

        N_v = len(valid_lits)
        if N_v < 4:
            return {"Q2_LOO": float("nan"),
                    "error": f"Solo {N_v} litotipi validi (<4)"}

        eps_  = 1e-30
        logX  = np.log10(np.maximum(np.array(Pi_all), eps_))
        logY  = np.log10(np.array(Cin_all))

        Ypred_log = np.zeros(N_v)
        for i in range(N_v):
            mask_tr  = [j for j in range(N_v) if j != i]
            Xtr      = np.column_stack([np.ones(len(mask_tr)), logX[mask_tr]])
            Ytr      = logY[mask_tr]
            coef, _, _, _ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
            Xte          = np.hstack([[1.0], logX[i]])
            Ypred_log[i] = float(Xte @ coef)

        residuals  = logY - Ypred_log
        SS_pred    = float(np.sum(residuals**2))
        SS_tot     = float(np.sum((logY - np.mean(logY))**2))
        Q2_LOO     = float(1.0 - SS_pred / max(SS_tot, eps_))

        Cin_pred = 10.0**Ypred_log

        if Q2_LOO >= 0.70:
            interp = "ECCELLENTE: il modello predice litologie mai viste"
        elif Q2_LOO >= 0.50:
            interp = "ACCETTABILE: generalizzazione parziale"
        else:
            interp = "INSUFFICIENTE: possibile overfitting o dati insufficienti"

        summary = (
            "Q2_LOO = {:.3f} | {}\n"
            "N litotipi = {} | RMSE_log10 = {:.3f}\n"
            "Range C_in osservato: [{:.1f}, {:.1f}] Bq/m3"
        ).format(
            Q2_LOO, interp, N_v, float(np.sqrt(SS_pred / N_v)),
            min(Cin_all), max(Cin_all)
        )
        if verbose:
            for _line in summary.split("\n"):
                print("  " + _line)

        return {
            "Q2_LOO":        Q2_LOO,
            "N":             N_v,
            "lits":          valid_lits,
            "C_in_obs":      np.array(Cin_all),
            "C_in_pred":     Cin_pred,
            "residuals_log": residuals,
            "RMSE_log10":    float(np.sqrt(SS_pred / N_v)),
            "summary":       summary,
        }


    def indoor_box_model(self, J_surface, Cin0=50.0, dt=3600.0,
                         zone="basement", use_progeny=True):
        """Indoor box model: steady-state and transient radon concentration [Bq/m3]."""
        zone_params = {
            "basement": {"V": self.p.get("V_bs", 50.0),
                         "A": self.p.get("A_bs", 50.0),  "ach_mult": 1.0},
            "ground":   {"V": self.p.get("V_gt", 150.0),
                         "A": self.p.get("A_bs", 50.0)*0.3, "ach_mult": 1.5},
            "upper":    {"V": 200.0,
                         "A": self.p.get("A_bs", 50.0)*0.1, "ach_mult": 2.5},
        }
        zp      = zone_params.get(zone, zone_params["basement"])
        V       = zp["V"];   A = zp["A"]
        ACH     = self.p.get("lam_vent", 0.5) * zp["ach_mult"]
        lam_v_s = ACH / 3600.0
        lam_tot = self.lam_rn + lam_v_s

        J_eff = float(J_surface) * A
        C_ss  = J_eff / (lam_tot * V)
        C_t   = C_ss + (float(Cin0) - C_ss) * np.exp(-lam_tot * float(dt))
        tau_eq = 1.0 / lam_tot

        # Steady-state progeny chain (secular scheme)
        progeny = {"C_Po218": 0.0, "C_Pb214": 0.0,
                   "C_Bi214": 0.0, "C_Po214": 0.0}
        if use_progeny:
            # single source of decay data; activity-Bateman production uses the
            # DAUGHTER's decay constant; progeny also plate out at dep_progeny
            lam_Po218 = DECAY_CONSTANTS["Po218"]
            lam_Pb214 = DECAY_CONSTANTS["Pb214"]
            lam_Bi214 = DECAY_CONSTANTS["Bi214"]
            lam_Po214 = DECAY_CONSTANTS["Po214"]
            dep_p     = self.p.get("dep_progeny", 1.5/3600.0)
            C_Po218_ss = C_ss      * lam_Po218 / (lam_Po218 + lam_v_s + dep_p)
            C_Pb214_ss = C_Po218_ss * lam_Pb214 / (lam_Pb214 + lam_v_s + dep_p)
            C_Bi214_ss = C_Pb214_ss * lam_Bi214 / (lam_Bi214 + lam_v_s + dep_p)
            C_Po214_ss = C_Bi214_ss * lam_Po214 / (lam_Po214 + lam_v_s + dep_p)
            progeny = {"C_Po218": float(C_Po218_ss), "C_Pb214": float(C_Pb214_ss),
                       "C_Bi214": float(C_Bi214_ss), "C_Po214": float(C_Po214_ss)}

        if use_progeny and C_ss > 0:
            F_eq = self.equilibrium_factor(
                C_ss, progeny["C_Po218"], progeny["C_Pb214"],
                progeny["C_Bi214"], progeny["C_Po214"])
        else:
            F_eq = 0.4

        f_occ = 0.80; hours_y = 8760.0
        dose_mSv_y = C_ss * F_eq * 9.0e-6 * f_occ * hours_y

        return {
            "C_ss": float(C_ss), "C_t": float(C_t), "C0": float(Cin0),
            "tau_eq": float(tau_eq), "J_eff": float(J_eff), "ACH": float(ACH),
            "F_eq": float(F_eq), "dose_mSv_y": float(dose_mSv_y),
            "zone": zone, "progeny": progeny,
        }

    def sensitivity_indices_analytical(self, theta_mean=None, delta_frac=0.05):
        """Centered finite-difference elasticity of C_bs to physical parameters.
        Saltelli et al. 2008; Cukier et al. 1973, JCP 26:1."""
        if theta_mean is None:
            theta_mean = 0.35 * self.p["phi_m"]

        # Fixed set avoids stale keys that silently zeroed out sensitivities
        if self.p.get("sensitivity_param_names_fixed", True):
            params_key = ["k0", "phi_m", "eps_max", "Ra226_Bqkg", "rho_bulk",
                          "D0", "ACH_closed", "H", "phi_f", "T_ref"]
        else:
            params_key = ["k0", "phi_m", "eps_max", "Ra_soil", "rho_s",
                          "D0", "lam_vent", "H", "phi_f", "T_ref"]
        p_base = {pk: self.p[pk] for pk in params_key if pk in self.p}

        nz = self.nz
        T_arr = np.full(nz, self.p.get("T_ref", 293.15))
        theta = np.full(nz, theta_mean)
        Ar0   = np.full(nz, 0.75)
        eps0  = np.full(nz, self.p.get("eps_max", 0.3) * 0.6)

        def _C_fast():
            # Recompute eps0 from current eps_max so its perturbation propagates
            eps0_loc = (np.full(nz, self.p.get("eps_max", 0.3) * 0.6)
                        if self.p.get("sensitivity_param_names_fixed", True) else eps0)
            S222   = self.source_rn222(theta, T_arr, Ar0, eps0_loc)
            Cm_eq  = S222 / max(self.lam_rn, 1e-30)
            phi_t  = self.p["phi_m"] + self.p.get("phi_f", 0.05)
            Sg     = max((phi_t - theta_mean) / max(phi_t, 1e-12), 0.0)
            phi_g  = phi_t * Sg
            D_eff  = self.p.get("D0", 1.1e-5) * phi_g**(10./3.) / max(phi_t**2, 1e-12)
            k0n    = self.p.get("k0", 1e-13)
            mu_n   = self.p.get("mu", 1.82e-5)
            H_n    = self.p.get("H", 3.0)
            dP_nom = self.p.get("dP_bottom", 5.0)
            u_adv  = k0n / mu_n * dP_nom / H_n
            J_diff = D_eff * float(np.mean(Cm_eq)) / H_n
            # pore-gas equilibrium is the same in both continua
            J_adv  = u_adv * float(np.mean(Cm_eq))
            A_bs   = self.p.get("A_bs", 50.0); V_bs = self.p.get("V_bs", 50.0)
            # read the SAME ventilation key that the sensitivity loop perturbs
            if self.p.get("sensitivity_param_names_fixed", True):
                lam_v = self.p.get("ACH_closed", 0.10) / 3600.0
            else:
                lam_v = self.p.get("lam_vent", 0.5) / 3600.0
            return (J_diff + J_adv) * A_bs / max((self.lam_rn + lam_v) * V_bs, 1e-30)

        C0 = _C_fast(); n_evals = 1
        elasticities = {}

        for pk in params_key:
            if pk not in self.p: continue
            p_orig = self.p[pk]
            try:
                self.p[pk] = p_orig * (1.0 + delta_frac)
                C_plus  = _C_fast(); n_evals += 1
                self.p[pk] = p_orig * (1.0 - delta_frac)
                C_minus = _C_fast(); n_evals += 1
            except Exception:
                elasticities[pk] = 0.0
                self.p[pk] = p_orig
                continue
            self.p[pk] = p_orig
            denom = max(abs(p_orig), 1e-30) * max(abs(C0), 1e-30)
            elasticities[pk] = (C_plus - C_minus) / (2.0 * delta_frac * denom) * p_orig

        for pk in p_base:
            self.p[pk] = p_base[pk]

        ranking = sorted(
            [(pk, elasticities.get(pk, 0.0)) for pk in params_key],
            key=lambda x: abs(x[1]), reverse=True
        )
        return {
            "elasticities": elasticities,
            "C_nominal":    float(C0),
            "ranking":      ranking,
            "n_evals":      n_evals,
        }


def build_model(lithotype="tuff_unwelded", override=None):
    """Build a VolcanicRadonModelV2 configured for a given lithotype."""
    if lithotype in GEOLITHOTYPES:
        lit = GEOLITHOTYPES[lithotype].copy()
    elif lithotype in VOLCANIC_LITHOTYPES:
        lit = VOLCANIC_LITHOTYPES[lithotype].copy()
    else:
        raise KeyError(
            f"Litotipo '{lithotype}' non trovato. Disponibili: "
            + str(list(GEOLITHOTYPES.keys()))
        )

    texture = lit.get("texture", None)
    if texture and texture in VG_TEXTURE_PARAMS:
        vg = VG_TEXTURE_PARAMS[texture]
        lit.setdefault("soil_alpha", vg[0])
        lit.setdefault("soil_n",     vg[1])
        lit.setdefault("theta_r",    vg[2])   # residual water content [m3/m3]
        lit.setdefault("phi_m",      vg[3])   # theta_s = phi_m per ROSETTA convention
        lit.setdefault("Ksat0",      vg[4])
    p = {
        # H = soil column depth [m], must exceed radon diffusion length (~1-1.5 m in tuff)
        "H":             3.0,
        "nz":            50,
        "phi_m":         lit["phi_m"],
        "phi_f":         lit["phi_f"],
        "k0":            lit["k0"],
        "k_frac":        lit["k_frac"],
        "Ksat0":         lit["Ksat0"],
        "Ksat_grad":     0.2,
        "soil_alpha":    lit["soil_alpha"],
        "soil_n":        lit["soil_n"],
        "Sg_crit":       0.08,
        "Sg_sharp":      0.02,
        "nk":            3.5,
        "kz_grad":       0.0,
        "k_min":         1e-18,
        "k_max":         1e-10,
        "mu":            1.82e-5,
        "M_g":           0.02897,
        "R_g":           8.314,
        "g":             9.81,
        "rho_air":       1.20,
        "P_atm0":        101325.0,
        "P_atm_amp":     150.0,
        "T_atm":         86400.0,
        "P_bottom":      101330.0,
        "dP_init":       5.0,
        "dP_bottom":     5.0,         # sub-slab overpressure [Pa], barometric pumping
        "T0":            288.0,
        "T_grad":        0.025,
        "T_surf_amp":    8.0,
        "T_damp_depth":  0.12,   # daily wave damping depth sqrt(2*kappa/omega), kappa~5e-7
        "theta_r":       lit.get("theta_r", 0.03),   # van Genuchten residual water content
        "T_in0":         293.15,
        "T_in_amp":      2.0,
        "T_out0":        283.15,
        "T_out_amp":     5.0,
        "T_ref":         293.15,
        "seismic_recovery_on":  False,   # enable deterministic k(t) recovery pulse
        "M_w_unrest":           0.0,     # equivalent unrest magnitude (uplift M_w)
        "seis_t0_s":            0.0,     # unrest-peak time [s]
        "k_seis_floor":         1.0,     # asymptotic healing floor
        "moisture_emanation_on":   False,  # non-monotonic emanation-vs-saturation peak
        "Sw_peak_moist":           0.17,   # documented 15-20% saturation peak
        "Sw_sigma_moist":          0.12,
        "f_emanation_dry":         0.45,   # dry emanation relative to peak
        "Sw_block_moist":          0.88,
        "Sw_block_sharp_moist":    0.04,
        "moisture_diffusivity_on": False,  # near-saturation D collapse (Rogers & Nielson)
        "Sw_perc_gas":             0.95,   # gas-phase percolation cutoff
        "diff_choke_exp":          3.0,
        "diff_choke_min":          1e-4,
        "T_out_year_amp":   0.0,         # set >0 (e.g. 10 K) to enable seasonal cycle
        "T_out_year_phase": -np.pi/2,    # outdoor minimum in winter
        "T_in_year_amp":    0.0,
        "T_in_year_phase":  -np.pi/2,
        "T_year":           365.25*86400.0,
        "T_day":            86400.0,
        "Ra226_Bqkg":    lit["Ra226_Bqkg"],
        "Th232_Bqkg":    lit["Ra226_Bqkg"] * 0.8,
        "rho_bulk":      lit["rho_bulk"],
        "D0":            lit["D0"],
        "Dmin":          1e-12,
        "Dmax":          1.2e-5,   # cap just above the free-air value
        "D_fracture":    5e-6,     # open-fracture diffusivity (fracture_dominant regime)
        "eps_max":       lit["eps_max"],
        "Sw_peak_eman":  lit["Sw_peak_eman"],
        "Sw_sigma_eman": 0.14,
        "eps_dry":       0.06,
        "Sw_block_eman": 0.90,
        "Sw_block_sharp":0.03,
        "vesicle_fraction": lit.get("vesicle_fraction", 0.0),  # vesicle fraction in grains
        "P_entry":          500.0,   # capillary entry pressure [Pa]
        "sigma_tot_frac":   3e5,     # total stress for fracture closure [Pa]
        "tau_cap":          1800.0,  # dynamic capillary coefficient [Pa.s]
        "D0_CO2":           1.6e-5,  # CO2 diffusivity in air [m^2/s]
        "S_CO2":            1e-5,    # CO2 volumetric source [mol/m^3/s]
        "b0_fracture_mean": 2e-4,    # reference fracture aperture [m]
        "beta_T_eman":   0.003,
        "grain_size_mm": lit["grain_size_mm"],
        "grain_ref_mm":  0.25,
        "grain_exp_eman":0.20,
        "eps_ratio_220_222": 1.0,   # recoil emanation comparable for Rn-220/Rn-222
        "H_rn0":         0.25,      # Ostwald coefficient at 20 C (Weigel 1978)
        "dH_rn":         2790.0,    # -dH_sol/R [K] > 0: solubility rises as T falls
        "A_k_open":      1.0/3600.0,
        "A_k_close":     0.35/3600.0,
        "A_ng":          2.0,
        "A_nw":          2.0,
        "A_beta_dry":    5e8,       # makes the drying gate an actual 0/1 switch
        "A_dtheta_crit": 1e-8,
        "A_beta_frac":   1.5,
        "A_beta_rewet":  1e6,
        "A_beta_gradPg": 0.0,
        "A_gradPg_ref":  100.0,
        "a_mf0":         1.0,
        "d_mf":          0.02,
        "n_mf_sat":      2.0,
        "beta_mf_p":     1e-6,
        "mem_nu":        0.70,
        "mem_tau":       6*3600.0,
        "mem_alpha":     1.0/3600.0,
        "eta_inst":      1.0,
        "eta_mem":       0.5,
        "gamma_sw_f":    0.20,
        # mineral-soil adsorption: Kd = k_ads/(k_des+lambda) ~ 0.02 (negligible,
        # Nazaroff 1992); raise k_ads for organic-rich soils only
        "k_ads_dry":     1e-7,
        "k_ads_wet":     5e-8,
        "k_des_dry":     2e-6,
        "k_des_wet":     4e-6,
        "n_cracks":      18,
        "crack_seed":    42,
        "crack_b0_median": 2.4e-4,  # calibrated: C_in/C_soil ~ 5e-3 (Nazaroff 1992)
        "h_dry_cap":     1.0e3,     # air-dry suction cap [m] (HYDRUS hCritA)
        "n_frac_horiz":  20,
        "horiz_seed":    7,
        "Cd_crack":      0.61,
        "D_entry":       1e-6,
        "nD_entry":      2.4,
        "b_foundation_m":1e-3,    # effective slab-gap thickness for foundation flow
        "C_out_Rn":      10.0,    # outdoor radon supplied by ventilation [Bq/m^3]
        "foundation_coupling": 0.35,
        "Sw_close":      0.85,
        "k_seis_max":    50.0,
        "seis_seed":     77,
        "V":             170.0,
        "V_bs":           50.0,
        "V_gt":          120.0,
        "H_stack":         2.5,
        "A_crack":       lit["A_crack_m2"],
        "A_crack_m2":    lit["A_crack_m2"],
        "A_leak":        5e-3,    # [m^2] realistic envelope leakage ~50 cm^2
        "A_soil_seal":   1e-4,    # air-pressure balance area, not radon transport area
        "Cd_leak":       0.62,
        "C_wind":        0.60,
        "U_wind":        1.0,
        "ACH_closed":    0.10,
        "ACH_open":      1.20,
        "ACH_mech":      0.0,
        "lambda_bg":     0.2/3600.0,
        "dep_bs":        0.0,
        "dep_gt":        0.0,
        "dep_progeny":   1.5/3600.0,
        "rain0":         2e-7,
        "T_rain":        864000.0,
        "evap0":         1e-7,
        "T_evap":        86400.0,
        "evap_phase":    np.pi/2,
        "tau_hyst_dry":  2*3600.0,
        "tau_hyst_wet":  48*3600.0,
        "seismic_site":  lit["seismic_site"],
        "t_total_seis":  7*86400.0,
        "theta0":        0.35 * lit["phi_m"],
        "C0_m220":       0.0,
        "C0_f220":       0.0,
        "Cin0":          50.0,
        "Cin_gt0":       20.0,
        "C_ref":         100.0,
        "a_int0":        1200.0,
        "k_la0":         3.2e-5,
        "t_open1s":      7*3600,
        "t_open1e":      9*3600,
        "t_open2s":      19*3600,
        "t_open2e":      22*3600,
        "ACH_sharp":     300.0,
    }

    for geo_key in ("regime", "emanation_model", "eps_crystalline",
                    "D_matrix_crystalline", "k_frac_eff",
                    "fracture_density_factor", "eps_karst",
                    "f_karst_conduit", "D_conduit",
                    "k_karst_conduit", "f_karst_active"):
        if geo_key in lit:
            p.setdefault(geo_key, lit[geo_key])

    if override:
        p.update(override)
    # accept the alias 'theta_init' (used by the regime-aware sampler) for theta0
    if "theta_init" in p:
        p["theta0"] = p.pop("theta_init")
    return VolcanicRadonModelV2(p)


def run_validation():
    """Run all 7 historical lithotypes and compute Buckingham pi groups."""
    results = {}
    for name, lit in VOLCANIC_LITHOTYPES.items():
        print(f"  Simulating {name} ({lit['label']}) ...")
        m   = build_model(name)
        sol = m.simulate(days=3.0)
        if not sol.success:
            print(f"    Attenzione: solver fallito: {sol.message}")
        res = m.extract_results(sol)
        t_i = -1
        theta_f = res["theta"][:, t_i]
        Pg_f    = res["Pg"][:, t_i]
        T_f     = m.T_field(sol.t[t_i])
        pi      = m.buckingham_pi(theta_f, Pg_f, T_f)
        C_bs_f  = float(res["C_bs"][t_i])
        Cf220_f = float(res["Cf220"][0, t_i]) if res["Cf220"].ndim == 2 else 0.0
        ratio_iso = Cf220_f / max(C_bs_f, 1.0)
        F_eq = m.equilibrium_factor(
            C_bs_f,
            float(res["C_Po218"][t_i]),
            float(res["C_Pb214"][t_i]),
            float(res["C_Bi214"][t_i]),
            float(res["C_Po214"][t_i]),
        )
        S_m_f = m.source_rn222(theta_f, T_f,
                                res["Ar"][:,t_i], res["eps_mem"][:,t_i])
        lv    = m.p.get("ACH_closed",0.1)/3600.0
        Jent  = float(res["C_bs"][t_i] * lv * m.p["V_bs"])
        mb_err= m.mass_balance_check(
            res["Cm"][:,t_i], res["Cf"][:,t_i],
            C_bs_f, float(res["C_gt"][t_i]),
            Jent, S_m_f, lv, sol.t[t_i],
            theta=theta_f,
        )
        results[name] = {
            "sol":         sol,
            "res":         res,
            "pi":          pi,
            "F_eq":        F_eq,
            "ratio_iso":   ratio_iso,
            "mb_err":      mb_err,
            "C_bs_final":  C_bs_f,
            "Pin_final":   float(res["Pin"][t_i]),
            "eps_mem_mean":float(np.mean(res["eps_mem"][:,t_i])),
            "Ar_mean":     float(np.mean(res["Ar"][:,t_i])),
        }
        print(f"    Pe_H={pi['Pe_H']:.3f}  C_bs={C_bs_f:.1f} Bq/m³  F_eq={F_eq:.3f}  mb_err={mb_err:.2e}")
    return results


def run_lithology_benchmark(lithotypes=None, days=2.0, nz=15, verbose=True):
    """Benchmark run over all GEOLITHOTYPES, sorted by steady indoor C_in [Bq/m3]."""
    if lithotypes is None:
        lithotypes = list(GEOLITHOTYPES.keys())

    results = {}

    for lit_name in lithotypes:
        if lit_name not in GEOLITHOTYPES:
            if verbose:
                print(f"  {lit_name} non in GEOLITHOTYPES")
            continue

        lit_data = GEOLITHOTYPES[lit_name]
        if verbose:
            print(f"  Simulazione {lit_name} ({lit_data['label']}) ...")

        try:
            m = build_model(lit_name, override={"nz": nz})
            sol = m.simulate_fast(days=days, dt_out=3600.0)
            out = m.extract_results(sol)

            t   = np.array(out["t"])
            Cbs = np.array(out["C_bs"])
            Cgt = np.array(out["C_gt"])
            Ar  = np.array(out["Ar"])
            eps = np.array(out["eps_mem"])
            Pg  = np.array(out["Pg"])

            # Average over last 6h to estimate steady-state values
            mask = t > t[-1] - 6*3600
            def m_last(a):
                a2 = np.asarray(a)
                if a2.ndim == 2:
                    return float(np.nanmean(a2[:, mask]))
                return float(np.nanmean(a2[mask]))

            Cbs_m   = m_last(Cbs)
            theta_f = np.array(out["theta"])[:, -1]
            Pg_f    = Pg[:, -1]
            T_f     = m.T_field(sol.t[-1])

            pi = m.buckingham_pi(theta_f, Pg_f, T_f)

            # Pi_frac: fracture/matrix permeability ratio, fracture-dominance index
            k_mat  = float(np.mean(m._k_gas_standard(theta_f, m.z)))
            k_tot  = float(np.mean(m._k_gas(theta_f, m.z)))
            Pi_frac = k_tot / max(k_mat, 1e-30)

            Cf_eff, _ = m._entry_concentration(
                np.array(out["Cm"])[:, -1], np.array(out["Cf"])[:, -1])
            Pg0    = float(Pg_f[0]); Pg_b = float(Pg_f[-1])
            Pg_eff = np.sqrt(Pg0 * Pg_b)
            Pin    = m._Pin_steady(m._Patm(sol.t[-1]),
                                   m.p.get("T_in0", 293.15),
                                   m.p.get("T_out0", 283.15))
            J_c = m.crack_network_flux(Cf_eff, Cbs_m, Pg_eff, Pin,
                                       float(theta_f[0]), float(T_f[0]),
                                       m.p.get("T_in0", 293.15))
            J_f = m.foundation_2d_network_flux(Cf_eff, Cbs_m, Pg_eff, Pin,
                                               float(theta_f[0]))
            J_tot = J_c + J_f

            regime_label = m.transport_regime(pi["Pe_H"], pi["Da_H"])
            regime_geo   = m.p.get("regime", "porous_diffusive")

            # Rn222/Rn220 ratio as a proxy for source depth
            C_bs_f  = Cbs_m
            Cf220_s = float(np.array(out["Cf220"])[0, -1]) if "Cf220" in out else 0.0
            ratio_iso = Cf220_s / max(C_bs_f, 1.0)

            F_eq = m.equilibrium_factor(
                C_bs_f,
                float(np.array(out["C_Po218"])[-1]),
                float(np.array(out["C_Pb214"])[-1]),
                float(np.array(out["C_Bi214"])[-1]),
                float(np.array(out["C_Po214"])[-1]),
            )

            eps_mean = m_last(eps)

            results[lit_name] = {
                "label":          lit_data["label"],
                "Cin_Bq_m3":      Cbs_m,
                "C_gt_Bq_m3":     m_last(Cgt),
                "Pe_H":           pi["Pe_H"],
                "Da_H":           pi["Da_H"],
                "Pi_frac":        Pi_frac,
                "regime_pd":      regime_label,     # Pe-Da classifier
                "regime":         regime_geo,        # geological regime
                "J_surface":      J_tot,
                "eps_mean":       eps_mean,
                "Ar_mean":        m_last(Ar),
                "F_eq":           F_eq,
                "ratio_iso":      ratio_iso,
                "Ra226_Bqkg":     lit_data["Ra226_Bqkg"],
                "k0":             lit_data["k0"],
                "phi_m":          lit_data["phi_m"],
                "emanation_model":lit_data.get("emanation_model", "volcanic_vesicular"),
            }

            if verbose:
                print(f"    Pe_H={pi['Pe_H']:.3f}  C_in={Cbs_m:.1f} Bq/m3  "
                      f"regime={regime_geo}  F_eq={F_eq:.3f}")

        except Exception as exc:
            if verbose:
                print(f"    Attenzione: {lit_name}: {exc}")
            results[lit_name] = {"label": lit_data["label"], "error": str(exc),
                                  "Cin_Bq_m3": float("nan")}

    results = dict(sorted(
        results.items(),
        key=lambda kv: kv[1].get("Cin_Bq_m3", float("-inf")),
        reverse=True
    ))
    return results


def print_lithology_benchmark_table(bench):
    """Print an ASCII table of run_lithology_benchmark results."""
    print()
    print(f"  {'Litotipo':<32} {'Regime':<22} "
          f"{'C_in':>9} {'Pe_H':>7} {'Da_H':>7} "
          f"{'Pi_frac':>9} {'F_eq':>7} {'eps':>7}")
    print("  " + "-"*100)
    for name, r in bench.items():
        if "error" in r:
            print(f"  {r['label']:<32} {'ERROR':22} {r.get('error','')}")
            continue
        print(f"  {r['label']:<32} {r['regime']:<22} "
              f"{r['Cin_Bq_m3']:9.1f} {r['Pe_H']:7.3f} {r['Da_H']:7.3f} "
              f"{r['Pi_frac']:9.2f} {r['F_eq']:7.3f} {r['eps_mean']:7.4f}")
    print()


def plot_lithology_benchmark(bench, save_path=None, dpi=150):
    """Build the 4-panel lithology benchmark figure and return the Figure."""
    REGIME_COLORS = {
        "geothermal_active":  "#e74c3c", # red -- volcanic
        "fracture_dominant":  "#8e44ad", # purple -- crystalline
        "karst_conduit":      "#2980b9", # blue -- karst
        "porous_advective":   "#27ae60", # green -- sand/alluvium
        "porous_diffusive":   "#e67e22", # orange -- clay/silt/marl
    }
    REGIME_LABELS = {
        "geothermal_active":  "Vulcanici attivi",
        "fracture_dominant":  "Rocce cristalline",
        "karst_conduit":      "Carsici",
        "porous_advective":   "Sabbie/Alluvioni",
        "porous_diffusive":   "Argille/Limi/Marne",
    }

    valid = {k: v for k, v in bench.items() if "error" not in v
             and np.isfinite(v.get("Cin_Bq_m3", float("nan")))}
    names    = list(valid.keys())
    labels   = [v["label"] for v in valid.values()]
    Cin      = np.array([v["Cin_Bq_m3"]   for v in valid.values()])
    Pe       = np.array([v["Pe_H"]         for v in valid.values()])
    Da       = np.array([v["Da_H"]         for v in valid.values()])
    k0_vals  = np.array([v["k0"]           for v in valid.values()])
    Ra_vals  = np.array([v["Ra226_Bqkg"]   for v in valid.values()])
    phi_vals = np.array([v["phi_m"]        for v in valid.values()])
    regimes  = [v["regime"]               for v in valid.values()]
    colors   = [REGIME_COLORS.get(r, "#95a5a6") for r in regimes]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    ax = axes[0, 0]
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, np.maximum(Cin, 0.1), color=colors, alpha=0.88,
                   edgecolor="white", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7, color="white")
    ax.set_xscale("log")
    ax.set_xlabel("C$_{in}$ [Bq m$^{-3}$]", color="white")
    ax.set_title("A — Concentrazione indoor per litotipo", color="white",
                 fontsize=10, fontweight="bold")
    ax.axvline(75,  color="#f39c12", lw=1.2, ls="--", label="IT mean 75 Bq/m3")
    ax.axvline(300, color="#e74c3c", lw=1.2, ls="--", label="WHO 300 Bq/m3")
    ax.legend(fontsize=7, facecolor="#2c2c54", labelcolor="white", loc="lower right")
    ax.grid(axis="x", alpha=0.2, color="white")

    ax = axes[0, 1]
    seen_regimes = set()
    for i, (pe, da, c, r) in enumerate(zip(Pe, Da, colors, regimes)):
        lbl = REGIME_LABELS.get(r, r) if r not in seen_regimes else None
        ax.scatter(pe, da, color=c, s=90, alpha=0.88, edgecolors="white",
                   linewidths=0.4, label=lbl, zorder=3)
        seen_regimes.add(r)
    ax.axvline(0.1, color="#aaaaaa", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1.0, color="#aaaaaa", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Pe$_H$ [-]", color="white")
    ax.set_ylabel("Da$_H$ [-]", color="white")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("B — Mappa regime Pe-Da", color="white",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, facecolor="#2c2c54", labelcolor="white",
              loc="upper left", ncol=1)
    ax.grid(alpha=0.2, color="white")
    ax.text(0.005, 5, "Diff. puro", color="#aaaaaa", fontsize=7, alpha=0.7)
    ax.text(0.5,   5, "Adv. dom.",  color="#aaaaaa", fontsize=7, alpha=0.7)
    ax.text(0.005, 0.2,"Sorgente",  color="#aaaaaa", fontsize=7, alpha=0.7)

    ax = axes[1, 0]
    seen_regimes = set()
    for i, (k, c_val, c, r) in enumerate(zip(k0_vals, Cin, colors, regimes)):
        lbl = REGIME_LABELS.get(r, r) if r not in seen_regimes else None
        ax.scatter(k, max(c_val, 0.1), color=c, s=90, alpha=0.88,
                   edgecolors="white", linewidths=0.4, label=lbl, zorder=3)
        seen_regimes.add(r)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("k$_0$ [m$^2$]", color="white")
    ax.set_ylabel("C$_{in}$ [Bq m$^{-3}$]", color="white")
    ax.set_title("C — C$_{in}$ vs permeabilita'", color="white",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.2, color="white")
    ax.legend(fontsize=7, facecolor="#2c2c54", labelcolor="white",
              loc="upper left", ncol=1)

    ax = axes[1, 1]
    sc = ax.scatter(Ra_vals, np.maximum(Cin, 0.1), c=phi_vals,
                    s=90, cmap="plasma", alpha=0.88,
                    edgecolors="white", linewidths=0.4, zorder=3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("phi_m [-]", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Ra-226 [Bq kg$^{-1}$]", color="white")
    ax.set_ylabel("C$_{in}$ [Bq m$^{-3}$]", color="white")
    ax.set_title("D — C$_{in}$ vs Ra-226 (dim. = phi_m)", color="white",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.2, color="white")

    plt.suptitle(
        ("Benchmark predittivo multi-litotipo - VolcanicRadonModel | "
         "20 litotipi | 5 regimi"),
        color="white", fontsize=11, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="#1a1a2e")
        print(f"  Salvato: {save_path}")
    return fig


def generate_causal_chain_figure(results, outpath="/home/user/workspace/causal_chain_summary.png"):
    """8-panel figure of causal chain layers for representative lithotypes."""
    lits_plot = ["tuff_unwelded", "pumice", "sandy_soil"]
    colors    = ["#c0392b", "#2980b9", "#27ae60"]
    lw        = 2.0

    fig = plt.figure(figsize=(18, 22))
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    def t_h(sol):
        return sol.t / 3600.0

    ax1 = fig.add_subplot(gs[0, 0])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        ax1.plot(t_h(r["sol"]), r["res"]["theta"][0], color=c, lw=lw,
                 label=VOLCANIC_LITHOTYPES[name]["label"])
    ax1.set_xlabel("Tempo [h]"); ax1.set_ylabel("θ [m³/m³]")
    ax1.set_title("Layer 1 — Umidità z=0\n(Richards, drenaggio)")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        Patm0 = r["res"]["Pg"].mean()
        ax2.plot(t_h(r["sol"]), r["res"]["Pg"][0]-101325, color=c, lw=lw,
                 label=VOLCANIC_LITHOTYPES[name]["label"])
    ax2.set_xlabel("Tempo [h]"); ax2.set_ylabel("ΔPg [Pa]")
    ax2.set_title("Layer 1 — Pressione gas suolo\n(barometrico + geomeccanica)")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        ax3.plot(t_h(r["sol"]), r["res"]["Ar"].mean(axis=0), color=c, lw=lw)
    ax3.set_xlabel("Tempo [h]"); ax3.set_ylabel("Ar [-]")
    ax3.set_title("Layer 2 — Accessibilità pori\n(variabile di stato Ar)")
    ax3.set_ylim(0, 1); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[0, 3])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        ax4.plot(t_h(r["sol"]), r["res"]["eps_mem"].mean(axis=0), color=c, lw=lw)
    ax4.set_xlabel("Tempo [h]"); ax4.set_ylabel("ε_mem [-]")
    ax4.set_title("Layer 2 — Isteresi emanazione\n(memoria wetting/drying)")
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 0:2])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        lab = VOLCANIC_LITHOTYPES[name]["label"]
        ax5.plot(t_h(r["sol"]), r["res"]["Cm"].mean(axis=0), color=c, lw=lw, label=f"Cm {lab}")
        ax5.plot(t_h(r["sol"]), r["res"]["Cf"].mean(axis=0), color=c, lw=lw, ls="--")
    ax5.set_xlabel("Tempo [h]"); ax5.set_ylabel("C_Rn [Bq m⁻³]")
    ax5.set_title("Layer 3 — Radon matrice () e frattura (--)\n(memoria Mittag-Leffler + adsorbimento)")
    ax5.legend(fontsize=7); ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2:4])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        lab = VOLCANIC_LITHOTYPES[name]["label"]
        ax6.plot(t_h(r["sol"]), r["res"]["C_bs"], color=c, lw=lw,   label=f"Interrato {lab}")
        ax6.plot(t_h(r["sol"]), r["res"]["C_gt"], color=c, lw=lw, ls="--")
    ax6.set_xlabel("Tempo [h]"); ax6.set_ylabel("C_in [Bq m⁻³]")
    ax6.set_title("Layer 5 — Indoor Rn-222\n(multizona: interrato —, PT --)")
    ax6.legend(fontsize=7); ax6.grid(alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 0:2])
    name0 = "tuff_unwelded"
    r0    = results[name0]
    th0   = t_h(r0["sol"])
    ax7.plot(th0, r0["res"]["C_bs"],    color="k",        lw=lw, label="Rn-222")
    ax7.plot(th0, r0["res"]["C_Po218"], color="#e74c3c", lw=lw, label="Po-218")
    ax7.plot(th0, r0["res"]["C_Pb214"], color="#e67e22", lw=lw, label="Pb-214")
    ax7.plot(th0, r0["res"]["C_Bi214"], color="#9b59b6", lw=lw, label="Bi-214")
    ax7.plot(th0, r0["res"]["C_Po214"], color="#3498db", lw=lw, label="Po-214")
    ax7.set_xlabel("Tempo [h]"); ax7.set_ylabel("Concentrazione [Bq m⁻³]")
    ax7.set_title(f"Catena progenie Rn-222\n({VOLCANIC_LITHOTYPES[name0]['label']})")
    ax7.legend(fontsize=8); ax7.grid(alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 2:4])
    for name, c in zip(lits_plot, colors):
        r   = results[name]
        lab = VOLCANIC_LITHOTYPES[name]["label"]
        ax8.plot(t_h(r["sol"]), r["res"]["Cm220"].mean(axis=0), color=c, lw=lw, label=lab)
    ax8.set_xlabel("Tempo [h]"); ax8.set_ylabel("C_Rn220 (matrice) [Bq m⁻³]")
    ax8.set_title("Thoron (Rn-220) in matrice\n(diagnostica profondità sorgente)")
    ax8.legend(fontsize=7); ax8.grid(alpha=0.3)

    ax9 = fig.add_subplot(gs[3, 0])
    for name, c in zip(lits_plot, colors):
        r = results[name]
        ax9.plot(t_h(r["sol"]), r["res"]["Pin"]-101325, color=c, lw=lw,
                 label=VOLCANIC_LITHOTYPES[name]["label"])
    ax9.set_xlabel("Tempo [h]"); ax9.set_ylabel("ΔPin [Pa]")
    ax9.set_title("Layer 5 — Pressione indoor\n(stack + vento + barometrico)")
    ax9.legend(fontsize=7); ax9.grid(alpha=0.3)

    ax10 = fig.add_subplot(gs[3, 1])
    pi_names = ["Pe_H", "Da_H", "Pi_frac"]
    pi_vals  = {name: [results[name]["pi"][k] for k in pi_names] for name in lits_plot}
    x        = np.arange(len(pi_names))
    w        = 0.25
    for i, (name, c) in enumerate(zip(lits_plot, colors)):
        ax10.bar(x + i*w, np.log10(np.maximum(pi_vals[name], 1e-6)),
                 w, color=c, alpha=0.85, label=VOLCANIC_LITHOTYPES[name]["label"])
    ax10.set_xticks(x + w)
    ax10.set_xticklabels(pi_names, fontsize=8)
    ax10.set_ylabel("log₁₀(Π)")
    ax10.set_title("Layer 6 — Numeri di Buckingham π")
    ax10.legend(fontsize=7); ax10.grid(alpha=0.3, axis="y")

    ax11 = fig.add_subplot(gs[3, 2])
    Pe_vals = [results[n]["pi"]["Pe_H"]   for n in VOLCANIC_LITHOTYPES.keys()]
    C_vals  = [results[n]["C_bs_final"]   for n in VOLCANIC_LITHOTYPES.keys()]
    labels  = [VOLCANIC_LITHOTYPES[n]["label"] for n in VOLCANIC_LITHOTYPES.keys()]
    sc_c    = ["#c0392b","#e74c3c","#2980b9","#e67e22","#8e44ad","#27ae60","#7f8c8d"]
    for i, (pe, cv, lb, cc) in enumerate(zip(Pe_vals, C_vals, labels, sc_c)):
        ax11.scatter(pe, cv, s=80, color=cc, zorder=5, label=lb)
    Pe_fit = np.logspace(-1, 1, 50)
    valid  = [(pe, cv) for pe, cv in zip(Pe_vals, C_vals) if pe > 0 and cv > 0]
    if len(valid) >= 3:
        lpe   = np.log([v[0] for v in valid])
        lcv   = np.log([v[1] for v in valid])
        coef  = np.polyfit(lpe, lcv, 1)
        ax11.plot(Pe_fit, np.exp(coef[1]) * Pe_fit**coef[0], "k--", lw=1.5,
                  label=f"Power law: α={coef[0]:.2f}")
    ax11.set_xscale("log"); ax11.set_yscale("log")
    ax11.set_xlabel("Pe_H"); ax11.set_ylabel("C_bs final [Bq m⁻³]")
    ax11.set_title("Scaling law: C_indoor ~ Pe_H^α")
    ax11.legend(fontsize=6); ax11.grid(alpha=0.3, which="both")

    ax12 = fig.add_subplot(gs[3, 3])
    F_vals = [results[n]["F_eq"] for n in VOLCANIC_LITHOTYPES.keys()]
    ax12.barh(labels, F_vals, color=sc_c, edgecolor="k", linewidth=0.5)
    ax12.axvline(0.4, color="k", ls="--", lw=1.2, label="F=0.4 (tipico indoor)")
    ax12.set_xlabel("Fattore di equilibrio F")
    ax12.set_title("Equilibrio progenie\n(F = dose efficace / dose a eq.)")
    ax12.legend(fontsize=8); ax12.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "VolcanicRadonModel v2.0 — Catena Causale Definitiva\n"
        "moisture/T/P/fractures → source → matrix-fracture → ingress → indoor → checks",
        fontsize=13, fontweight="bold", y=0.995
    )

    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figura salvata: {outpath}")
    return outpath


def compute_scaling_laws(n_samples=48, seed=42, nz=40):
    """Monte Carlo fit of a power law over dimensionless groups.
    Samples GEOLITHOTYPES (20 types), OLS in log-log space on 4 pi groups."""
    rng  = np.random.default_rng(seed)
    X    = []
    Y    = []
    lits_sampled = []
    lits = list(GEOLITHOTYPES.keys())
    for _ in range(n_samples):
        name = rng.choice(lits)
        # Ra-226 range scaled per lithotype (10 Bq/kg limestone to 400 uraniferous)
        Ra_min = GEOLITHOTYPES[name].get("Ra226_Bqkg", 40.0) * 0.5
        Ra_max = GEOLITHOTYPES[name].get("Ra226_Bqkg", 40.0) * 3.0
        Ra_max = max(Ra_max, Ra_min + 10.0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            m = build_model(name, override={
                "dP_init":   float(rng.uniform(0.5, 30.0)),
                "Ra226_Bqkg":float(rng.uniform(Ra_min, Ra_max)),
                "ACH_closed":float(rng.uniform(0.05, 0.5)),
                "nz":        nz,
            })
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                sol = m.simulate(days=2.0)
            if sol.success and np.isfinite(sol.y[10*m.nz, -1]):
                res    = m.extract_results(sol)
                theta_f= res["theta"][:,  -1]
                Pg_f   = res["Pg"][:,    -1]
                T_f    = m.T_field(sol.t[-1])
                pi     = m.buckingham_pi(theta_f, Pg_f, T_f)
                C_in   = float(res["C_bs"][-1])
                if C_in > 0:
                    # DaAdv is the advective Damkohler number, distinct from Da_H
                    X.append([pi["Pe_H"], pi["DaAdv"], pi["Pi_src"], pi["Pi_vent"]])
                    Y.append(C_in)
                    lits_sampled.append(name)
        except Exception:
            pass
    if len(X) < 6:
        return None
    X   = np.array(X)
    Y   = np.array(Y)
    eps = 1e-30
    A   = np.column_stack([
        np.ones(len(X)),
        np.log(np.maximum(X[:,0], eps)),
        np.log(np.maximum(X[:,1], eps)),
        np.log(np.maximum(X[:,2], eps)),
        np.log(np.maximum(X[:,3], eps)),
    ])
    logY = np.log(np.maximum(Y, eps))
    coef, _, _, _ = np.linalg.lstsq(A, logY, rcond=None)
    Y_pred = np.exp(A @ coef)
    logYpred = np.log(np.maximum(Y_pred, eps))
    SS_res = np.sum((logY - logYpred)**2)
    SS_tot = np.sum((logY - np.mean(logY))**2)
    R2     = 1.0 - SS_res / max(SS_tot, 1e-30)
    return {
        "coef":    coef,
        "R2":      R2,
        "n":       len(Y),
        "lits":    lits_sampled,
        "law":     (f"C_in* ~ {np.exp(coef[0]):.3e} x "
                    f"Pe_H^{coef[1]:.3f} x DaAdv^{coef[2]:.3f} x "
                    f"Pi_src^{coef[3]:.3f} x Pi_vent^{coef[4]:.3f}"),
    }


def plot_volcanic_comparison_paper(results: dict, save_path: str = None,
                                    dpi: int = 150):
    """5-panel dark-theme comparison figure across lithotypes with WHO/IAEA thresholds."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import numpy as np

    DARK, PANEL, TEXT, GRID = PLOT_DARK, PLOT_PANEL, PLOT_TEXT, PLOT_GRID
    MUTED  = "#8b949e"
    PAL    = ["#e84393","#58a6ff","#56d364","#ffa657",
              "#f0883e","#bc8cff","#39d353","#79c0ff"]

    fig = plt.figure(figsize=(20, 16), facecolor=DARK)
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.44, wspace=0.36,
        left=0.06, right=0.97, top=0.93, bottom=0.08
    )

    def _ax(row, col, colspan=1):
        return fig.add_subplot(gs[row, col:col+colspan])

    def _style(ax, title, xl, yl, xsc="linear", ysc="linear"):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=10, pad=7, fontweight="bold",
                     wrap=True)
        ax.set_xlabel(xl, color=TEXT, fontsize=8.5)
        ax.set_ylabel(yl, color=TEXT, fontsize=8.5)
        ax.tick_params(colors=TEXT, labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, alpha=0.5, lw=0.6, ls="--")
        ax.set_xscale(xsc); ax.set_yscale(ysc)

    # Accepts output from either run_validation or lithotype_comparison
    lit_names = list(results.keys())
    colors    = {n: PAL[i % len(PAL)] for i, n in enumerate(lit_names)}

    def _get(r, *keys, default=0.0):
        for k in keys:
            if k in r:
                v = r[k]
                if hasattr(v, "__len__"):
                    arr = np.asarray(v)
                    return float(arr[-1]) if arr.ndim == 1 else float(arr.mean())
                return float(v)
        return float(default)

    def _get_pi(r, key, default=1e-3):
        if "pi" in r: return float(r["pi"].get(key, default))
        return float(r.get(key, default))

    C_in_vals = [_get(r, "Cin_Bq_m3", "C_bs_final") for r in results.values()]
    Pe_vals   = [_get_pi(r, "Pe_H") for r in results.values()]
    Da_vals   = [_get_pi(r, "Da_H") for r in results.values()]
    Feq_vals  = [_get(r, "F_eq", default=0.4) for r in results.values()]
    J_vals    = [_get(r, "J_surface", default=1e-4) for r in results.values()]
    phi_vals  = [_get(r, "phi_tot", default=0.3) for r in results.values()]
    k0_vals   = [_get(r, "k0", default=1e-14) for r in results.values()]
    labels_lit= [results[n].get("label", n) for n in lit_names]

    ax_a = _ax(0, 0)
    x_pos = np.arange(len(lit_names))
    bars  = ax_a.bar(
        x_pos, C_in_vals,
        color=[colors[n] for n in lit_names],
        edgecolor=GRID, linewidth=0.8, alpha=0.90
    )
    ax_a.axhline(100, color="#ffa657", ls="--", lw=1.0, alpha=0.85,
                 label="100 Bq m⁻³ (WHO ref. level)")
    ax_a.axhline(300, color="#e84393", ls="--", lw=1.0, alpha=0.85,
                 label="300 Bq m⁻³ (IAEA action level)")
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(labels_lit, rotation=38, ha="right", fontsize=7)
    for bar_obj, val in zip(bars, C_in_vals):
        if val > 0:
            ax_a.text(bar_obj.get_x() + bar_obj.get_width()/2,
                      val * 1.12, f"{val:.0f}",
                      ha="center", va="bottom", fontsize=6.5, color=TEXT)
    ax_a.legend(fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75,
                loc="upper right")
    _style(ax_a, "A — Radon Indoor C_in per Litotipo Vulcanico",
           "Litotipo", "C_in  [Bq m⁻³]", ysc="log")

    ax_b = _ax(0, 1)
    # Regime zones follow Pe_H = 1 (diffusion/advection) and Da_H = 100 boundaries
    xmin, xmax, ymin, ymax = 1e-4, 1e3, 1e-2, 1e4
    ax_b.fill_betweenx([ymin, 100], xmin, 1,
                       color="#58a6ff", alpha=0.08, label="Diffusive")
    ax_b.fill_betweenx([100, ymax],  xmin, 1,
                       color="#58a6ff", alpha=0.05)
    ax_b.fill_betweenx([ymin, 100], 1, xmax,
                       color="#e84393", alpha=0.08, label="Advective")
    ax_b.fill_betweenx([100, ymax],  1, xmax,
                       color="#f0883e", alpha=0.06, label="Decay-limited")
    ax_b.axvline(1.0, color="#ffa657", ls="--", lw=1.2, alpha=0.8,
                 label="Pe_H = 1")
    ax_b.axhline(100, color="#56d364", ls="--", lw=1.2, alpha=0.8,
                 label="Da_H = 100")
    for i, (n, Pe, Da) in enumerate(zip(lit_names, Pe_vals, Da_vals)):
        ax_b.scatter(Pe, Da, color=colors[n], s=90, zorder=6,
                     edgecolors=TEXT, lw=0.7)
        ax_b.annotate(labels_lit[i], (Pe, Da),
                      textcoords="offset points", xytext=(5, 3),
                      fontsize=6, color=TEXT)
    ax_b.set_xlim(xmin, xmax); ax_b.set_ylim(ymin, ymax)
    ax_b.legend(fontsize=6, labelcolor=TEXT, facecolor=PANEL,
                framealpha=0.75, loc="upper left")
    _style(ax_b, "B — Diagramma di Regime Pe_H – Da_H",
           "Pe_H  (Péclet idraulico) [-]", "Da_H  (Damköhler idraulico) [-]",
           xsc="log", ysc="log")

    ax_c = _ax(0, 2)
    k0_arr = np.array(k0_vals)
    J_arr  = np.array(J_vals)
    for i, n in enumerate(lit_names):
        ax_c.scatter(k0_arr[i], J_arr[i], color=colors[n], s=90, zorder=5,
                     edgecolors=TEXT, lw=0.7, label=labels_lit[i])
    mask_c = (k0_arr > 0) & (J_arr > 0)
    if mask_c.sum() >= 3:
        c2 = np.polyfit(np.log10(k0_arr[mask_c]), np.log10(J_arr[mask_c]), 1)
        k_l  = np.logspace(np.log10(k0_arr.min()*0.5), np.log10(k0_arr.max()*2), 60)
        J_l  = 10**c2[1] * k_l**c2[0]
        ax_c.plot(k_l, J_l, "--", color="#bc8cff", lw=1.5,
                  label=f"J ∝ k₀^{c2[0]:.2f}")
    ax_c.legend(fontsize=5.8, labelcolor=TEXT, facecolor=PANEL,
                framealpha=0.75, ncol=2)
    _style(ax_c, "C — Flusso Superficiale J vs Permeabilità k₀",
           "k₀  [m²]", "J_surface  [Bq m⁻² s⁻¹]", xsc="log", ysc="log")

    ax_d = _ax(1, 0)
    phi_arr = np.array(phi_vals)
    Cin_arr2= np.array(C_in_vals)
    for i, n in enumerate(lit_names):
        ax_d.scatter(phi_arr[i], Cin_arr2[i], color=colors[n], s=90, zorder=5,
                     edgecolors=TEXT, lw=0.7, label=labels_lit[i])
    mask_d = (phi_arr > 0) & (Cin_arr2 > 0)
    if mask_d.sum() >= 3:
        c3 = np.polyfit(np.log10(phi_arr[mask_d]), np.log10(Cin_arr2[mask_d]), 1)
        phi_l = np.logspace(np.log10(phi_arr.min()*0.8), np.log10(phi_arr.max()*1.2), 60)
        Cin_l = 10**c3[1] * phi_l**c3[0]
        ax_d.plot(phi_l, Cin_l, "--", color="#39d353", lw=1.5,
                  label=f"C_in ∝ φ^{c3[0]:.2f}")
    ax_d.legend(fontsize=5.8, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    _style(ax_d, "D — Scaling C_in vs Porosità Totale φ_tot",
           "φ_tot = φ_m + φ_f  [-]", "C_in  [Bq m⁻³]", xsc="log", ysc="log")

    ax_e = _ax(1, 1, colspan=2)
    Feq_arr = np.array(Feq_vals)
    # UNSCEAR 2000 secular-equilibrium range for the indoor progeny equilibrium factor
    ax_e.axhspan(0.4, 0.5, color="#ffa657", alpha=0.10,
                 label="Equilibrio secolare F_eq ∈ [0.4, 0.5]")
    for i, n in enumerate(lit_names):
        if Cin_arr2[i] > 0 and Feq_arr[i] > 0:
            ax_e.scatter(Cin_arr2[i], Feq_arr[i], color=colors[n], s=100,
                         zorder=5, edgecolors=TEXT, lw=0.7)
            ax_e.annotate(labels_lit[i], (Cin_arr2[i], Feq_arr[i]),
                          textcoords="offset points", xytext=(6, 3),
                          fontsize=6.5, color=TEXT)
    ax_e.axvline(100, color="#ffa657", ls=":", lw=1.0, alpha=0.7)
    ax_e.axvline(300, color="#e84393", ls=":", lw=1.0, alpha=0.7)
    ax_e.set_xlim(left=1e0); ax_e.set_ylim(0.0, 1.0)
    ax_e.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    _style(ax_e, "E — Fattore di Equilibrio F_eq vs C_in Radon\n"
                 "(F_eq basso = ventilazione alta o breve tempo di sosta progenie)",
           "C_in  [Bq m⁻³]", "F_eq  (UNSCEAR 2000) [-]", xsc="log")

    fig.suptitle(
        "VolcanicRadonModel v2.0 — Confronto Multi-Litotipo Vulcanico\n"
        "Daniel de Siano  ·  Target: peer-reviewed journal",
        color=TEXT, fontsize=13, fontweight="bold", y=0.978
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=DARK)
        print(f"[plot_volcanic_comparison_paper] Salvato: {save_path}")
    return fig


def plot_bifurcation_paper(bif: dict, save_path: str = None,
                            lithotype_label: str = "", dpi: int = 150):
    """3-panel bifurcation figure of C_in, Pe_H, and diffusive/advective flux vs dP."""
    import matplotlib.pyplot as plt
    import numpy as np

    DARK, PANEL, TEXT, GRID = PLOT_DARK, PLOT_PANEL, PLOT_TEXT, PLOT_GRID

    dP     = np.asarray(bif.get("dP",    bif.get("dP_range", [])))
    Cin    = np.asarray(bif.get("C_in",  bif.get("Cin", [])))
    Pe     = np.asarray(bif.get("Pe_H",  []))
    J_tot  = np.asarray(bif.get("J_total",   np.zeros_like(dP)))
    J_diff = np.asarray(bif.get("J_diff",    np.zeros_like(dP)))
    J_adv  = np.asarray(bif.get("J_adv",     np.zeros_like(dP)))
    dP_star_a = bif.get("dP_critical",  None)
    dP_star_n = bif.get("dP_numerical", None)
    alpha_adv = bif.get("alpha_adv",    np.nan)
    C_diff    = bif.get("C_diffusion",  None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=DARK,
                              gridspec_kw={"wspace": 0.33})

    def _style(ax, title, xl, yl, xsc="log", ysc="log"):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=10, pad=7, fontweight="bold")
        ax.set_xlabel(xl, color=TEXT, fontsize=8.5)
        ax.set_ylabel(yl, color=TEXT, fontsize=8.5)
        ax.tick_params(colors=TEXT, labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, alpha=0.45, lw=0.6, ls="--")
        ax.set_xscale(xsc); ax.set_yscale(ysc)

    ax_l = axes[0]
    if dP_star_a is not None:
        ax_l.axvspan(dP.min(), dP_star_a, color="#58a6ff", alpha=0.10,
                     label="Regime diffusivo")
        ax_l.axvspan(dP_star_a, dP.max(), color="#e84393", alpha=0.10,
                     label="Regime avvettivo")
        ax_l.axvline(dP_star_a, color="#ffa657", ls="--", lw=1.4, alpha=0.9,
                     label=f"ΔP* (anal.) = {dP_star_a:.2f} Pa")
    if dP_star_n is not None and dP_star_n != dP_star_a:
        ax_l.axvline(dP_star_n, color="#39d353", ls=":", lw=1.2, alpha=0.8,
                     label=f"ΔP* (num.) = {dP_star_n:.2f} Pa")
    if C_diff is not None and C_diff > 0:
        ax_l.axhline(C_diff, color=GRID, ls=":", lw=1.0, alpha=0.7,
                     label=f"C_diff = {C_diff:.1f} Bq m⁻³")
    ax_l.plot(dP, Cin, color="#56d364", lw=2.0, marker="o",
              markersize=4, zorder=5, label="C_in (modello)")
    if not np.isnan(alpha_adv):
        mask_a = Pe > 1.0
        if mask_a.sum() >= 2:
            dP_fit = dP[mask_a]; Cin_fit = Cin[mask_a]
            C0_fit = Cin_fit[0] / (dP_fit[0]**alpha_adv + 1e-30)
            ax_l.plot(dP_fit, C0_fit * dP_fit**alpha_adv,
                      "--", color="#bc8cff", lw=1.4,
                      label=f"C_in ∝ ΔP^{alpha_adv:.2f}")
    ax_l.legend(fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    _style(ax_l, f"Biforcazione C_in vs ΔP\n{lithotype_label}",
           "ΔP_sub-slab  [Pa]", "C_in  [Bq m⁻³]")

    ax_c = axes[1]
    ax_c.plot(dP, Pe, color="#bc8cff", lw=2.0, marker="s",
              markersize=4, zorder=5, label="Pe_H (Péclet idraulico)")
    ax_c.axhline(1.0, color="#ffa657", ls="--", lw=1.2, alpha=0.8,
                 label="Pe_H = 1 (biforcazione)")
    ax_c.axhline(0.1, color="#58a6ff", ls=":", lw=1.0, alpha=0.7,
                 label="Pe_H = 0.1 (diff. puro)")
    ax_c.axhline(10,  color="#e84393", ls=":", lw=1.0, alpha=0.7,
                 label="Pe_H = 10 (adv. forte)")
    ax_c.fill_between(dP, Pe, 1.0, where=(Pe < 1.0),
                      alpha=0.12, color="#58a6ff")
    ax_c.fill_between(dP, Pe, 1.0, where=(Pe >= 1.0),
                      alpha=0.12, color="#e84393")
    ax_c.legend(fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    _style(ax_c, "Numero di Péclet Idraulico vs ΔP",
           "ΔP_sub-slab  [Pa]", "Pe_H  [-]")

    ax_r = axes[2]
    J_tot_pos = np.maximum(J_tot, 1e-12)
    sc = ax_r.scatter(J_diff, J_adv, c=dP, cmap="plasma",
                      s=60, zorder=5, edgecolors=TEXT, lw=0.5,
                      norm=plt.matplotlib.colors.LogNorm(
                          vmin=dP.min(), vmax=dP.max()))
    J_levels = np.logspace(np.log10(J_tot_pos.min()+1e-12),
                            np.log10(J_tot_pos.max()), 4)
    j_range  = np.logspace(-6, 0, 80)
    for Jlev in J_levels:
        ax_r.plot(j_range, Jlev - j_range, ":", color=GRID, lw=0.8, alpha=0.6)
    j_eq = np.logspace(np.log10(max(min(J_diff[J_diff>0], default=1e-8), 1e-8)),
                       np.log10(max(max(J_adv[J_adv>0], default=1e-5), 1e-5)), 60)
    ax_r.plot(j_eq, j_eq, "--", color="#ffa657", lw=1.2, alpha=0.7,
              label="J_diff = J_adv")
    cbar = fig.colorbar(sc, ax=ax_r, pad=0.02)
    cbar.set_label("ΔP [Pa]", color=TEXT, fontsize=7)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=6.5)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT)
    ax_r.legend(fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    _style(ax_r, "J_diffusivo vs J_avvettivo per ogni ΔP\n(colore = ΔP [Pa])",
           "J_diff  [Bq s⁻¹]", "J_adv  [Bq s⁻¹]")

    fig.suptitle(
        f"Analisi Biforcazione Rn-222 — {lithotype_label}\n"
        "VolcanicRadonModel v2.0  ·  Daniel de Siano",
        color=TEXT, fontsize=12, fontweight="bold", y=1.01
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=DARK)
        print(f"[plot_bifurcation_paper] Salvato: {save_path}")
    return fig


def plot_buckingham_scaling(bpi: dict, save_path: str = None, dpi: int = 150):
    """2-panel figure of Buckingham π importance ranking and parity plot."""
    import matplotlib.pyplot as plt
    import numpy as np

    DARK, PANEL, TEXT, GRID = PLOT_DARK, PLOT_PANEL, PLOT_TEXT, PLOT_GRID

    pi_names = bpi.get("pi_names", [])
    exps     = np.asarray(bpi.get("exponents", []))
    ci95     = np.asarray(bpi.get("exponents_ci95", np.zeros((len(exps), 2))))
    imps     = np.asarray(bpi.get("importance", np.abs(exps)))
    R2       = bpi.get("R2", 0.0)
    R2_ci    = bpi.get("R2_ci95", (R2, R2))
    law_str  = bpi.get("law_string", "")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK,
                                      gridspec_kw={"wspace": 0.35})

    def _style(ax, title, xl, yl):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=10, pad=7, fontweight="bold")
        ax.set_xlabel(xl, color=TEXT, fontsize=8.5)
        ax.set_ylabel(yl, color=TEXT, fontsize=8.5)
        ax.tick_params(colors=TEXT, labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, alpha=0.5, lw=0.6, ls="--")

    rank_idx = np.argsort(imps)[::-1]
    names_r  = [pi_names[i] if i < len(pi_names) else f"π_{i}" for i in rank_idx]
    imps_r   = imps[rank_idx]
    exps_r   = exps[rank_idx]
    ci_lo_r  = ci95[rank_idx, 0] if ci95.shape[0] > 0 else np.zeros_like(exps_r)
    ci_hi_r  = ci95[rank_idx, 1] if ci95.shape[0] > 0 else np.zeros_like(exps_r)

    clrs     = ["#56d364" if e >= 0 else "#e84393" for e in exps_r]
    y_pos    = np.arange(len(names_r))
    ax_l.barh(y_pos, imps_r, color=clrs, edgecolor=GRID, alpha=0.85, height=0.6)
    for y_i, lo, hi in zip(y_pos, ci_lo_r, ci_hi_r):
        ax_l.errorbar(imps_r[y_i], y_i,
                      xerr=[[abs(exps_r[y_i]-lo)], [abs(hi-exps_r[y_i])]],
                      fmt="none", color=TEXT, capsize=3, lw=0.8)
    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(names_r, fontsize=7.5, color=TEXT)
    ax_l.axvline(0, color=GRID, lw=0.8)
    ax_l.invert_yaxis()
    _style(ax_l,
           f"A — Importanza Relativa Gruppi π\n|aᵢ|·σ(log πᵢ)",
           "Importanza relativa [-]", "Gruppo adimensionale")

    Cin_meas = bpi.get("Cin_samples", None)
    pi_samp  = bpi.get("pi_samples",  None)
    log_K    = bpi.get("log_K", 0.0)

    if Cin_meas is not None and pi_samp is not None and len(Cin_meas) > 0:
        C_ref = bpi.get("C_ref", np.median(Cin_meas))
        Cin_star = Cin_meas / max(C_ref, 1e-6)
        log_pi_s = np.log10(np.maximum(pi_samp, 1e-30))
        A_pred   = np.column_stack([np.ones(len(Cin_star)), log_pi_s])
        Cin_pred = 10**(log_K + log_pi_s @ exps)
        ax_r.scatter(Cin_star, Cin_pred,
                     c=np.log10(np.maximum(Cin_meas, 1e-6)),
                     cmap="viridis", s=25, alpha=0.6, edgecolors="none",
                     zorder=4)
        min_v = min(Cin_star.min(), Cin_pred.min())
        max_v = max(Cin_star.max(), Cin_pred.max())
        line_v = np.logspace(np.log10(max(min_v, 1e-6)), np.log10(max_v*1.05), 40)
        ax_r.plot(line_v, line_v, "--", color="#ffa657", lw=1.5,
                  label="Predetto = Simulato (1:1)")
        ax_r.plot(line_v, line_v*2, ":", color=GRID, lw=0.8, alpha=0.6,
                  label="Fattore 2×")
        ax_r.plot(line_v, line_v*0.5, ":", color=GRID, lw=0.8, alpha=0.6)
        ax_r.set_xscale("log"); ax_r.set_yscale("log")
        ax_r.text(0.05, 0.95, f"R² = {R2:.3f}\n[{R2_ci[0]:.3f}, {R2_ci[1]:.3f}]",
                  transform=ax_r.transAxes, color=TEXT, fontsize=8,
                  va="top", fontweight="bold")
        ax_r.legend(fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, framealpha=0.75)
    else:
        ax_r.text(0.5, 0.5, "Dati campionamento non disponibili\n"
                  "(usa return_samples=True in buckingham_pi_paper)",
                  ha="center", va="center", color=TEXT, fontsize=9,
                  transform=ax_r.transAxes)

    _style(ax_r, f"B — Parity Plot  C_in* (R²={R2:.3f})\n"
                  + law_str[:60] + ("..." if len(law_str)>60 else ""),
           "C_in* simulato [-]", "C_in* predetto (OLS) [-]")

    fig.suptitle("Buckingham π Analysis — VolcanicRadonModel v2.0",
                 color=TEXT, fontsize=12, fontweight="bold", y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=DARK)
        print(f"[plot_buckingham_scaling] Salvato: {save_path}")
    return fig


def plot_sensitivity_ranking(sa: dict, save_path: str = None, dpi: int = 150):
    """Bubble chart of elasticity vs Morris μ* sensitivity ranking.
    Quadrants follow Morris (1991): high/low μ* vs high/low elasticity."""
    import matplotlib.pyplot as plt
    import numpy as np

    DARK, PANEL, TEXT, GRID = PLOT_DARK, PLOT_PANEL, PLOT_TEXT, PLOT_GRID

    ranking   = sa.get("ranking", [])
    elast     = sa.get("elasticities", {})
    mu_star   = sa.get("morris_mu_star", {})
    sigma_m   = sa.get("morris_sigma", {})
    score_d   = sa.get("score", {})
    C0        = sa.get("C_nominal", 1.0)

    if not ranking:
        return None

    names  = [r[0] for r in ranking]
    eps_v  = np.array([abs(elast.get(n, 0.0)) for n in names])
    mu_v   = np.array([mu_star.get(n, 0.0) for n in names])
    sig_v  = np.array([sigma_m.get(n, 0.0) for n in names])
    scr_v  = np.array([score_d.get(n, 0.0) for n in names])
    sign_v = np.array([np.sign(elast.get(n, 1.0)) for n in names])

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK)
    ax.set_facecolor(PANEL)

    eps_med = np.median(eps_v[eps_v > 0]) if np.any(eps_v > 0) else 0.5
    mu_med  = np.median(mu_v[mu_v > 0])   if np.any(mu_v > 0)  else 0.5
    ax.axvline(eps_med, color=GRID, ls="--", lw=0.8, alpha=0.7)
    ax.axhline(mu_med,  color=GRID, ls="--", lw=0.8, alpha=0.7)

    for txt, xy, ha_v in [
        ("Critico\n(lineare, alto impatto)", (eps_v.max()*0.7, mu_v.max()*0.85), "center"),
        ("Non-lineare\no interazione",       (eps_v.max()*0.1, mu_v.max()*0.85), "center"),
        ("Impatto locale",                   (eps_v.max()*0.7, mu_v.max()*0.10), "center"),
        ("Trascurabile",                     (eps_v.max()*0.1, mu_v.max()*0.10), "center"),
    ]:
        ax.text(*xy, txt, color=GRID, fontsize=7, ha="center", va="center",
                fontstyle="italic", alpha=0.8)

    sizes  = 100 + 400 * (scr_v / max(scr_v.max(), 1e-6))
    colors_s = ["#56d364" if s >= 0 else "#e84393" for s in sign_v]
    scatter  = ax.scatter(eps_v, mu_v, s=sizes, c=colors_s,
                           edgecolors=TEXT, lw=0.7, zorder=5, alpha=0.85)

    for i, name in enumerate(names):
        ax.annotate(
            name.replace("_", " "), (eps_v[i], mu_v[i]),
            textcoords="offset points",
            xytext=(7, 4 if i % 2 == 0 else -10),
            fontsize=7.5, color=TEXT, fontweight="bold"
        )

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#56d364",
               ms=9, label="ε > 0 (C_in aumenta con p_i)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#e84393",
               ms=9, label="ε < 0 (C_in diminuisce con p_i)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=TEXT,
               ms=6,  label="Dimensione ∝ score combinato"),
    ]
    ax.legend(handles=legend_els, fontsize=7.5, labelcolor=TEXT,
              facecolor=PANEL, framealpha=0.8, loc="upper left")

    ax.set_xlabel("|Elasticità| = |∂ log C_in / ∂ log pᵢ| [-]",
                  color=TEXT, fontsize=9)
    ax.set_ylabel("μ* Morris = ⟨|EE_i|⟩ [Bq m⁻³ / unità_i]",
                  color=TEXT, fontsize=9)
    ax.set_title(
        f"Sensitivity Screen — Morris + Elasticità  "
        f"(C_in nominale = {C0:.0f} Bq m⁻³)\n"
        "VolcanicRadonModel v2.0  ·  Daniel de Siano",
        color=TEXT, fontsize=10, fontweight="bold", pad=9
    )
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(color=GRID, alpha=0.4, lw=0.6, ls="--")

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=DARK)
        print(f"[plot_sensitivity_ranking] Salvato: {save_path}")
    return fig


# demo/benchmark block only runs if RADON_RUN_DEMO=1 is set in the environment
if __name__ == "__main__" and os.environ.get("RADON_RUN_DEMO") == "1":
    print("=" * 70)
    print("VolcanicRadonModel  (GeoLithotypes Extension)")
    print(f"  {len(GEOLITHOTYPES)} litotipi | 5 regimi | 4 modelli di emanazione")
    print("=" * 70)

    print("\n[1] Benchmark predittivo su 20 litotipi geologici (GEOLITHOTYPES)...")
    bench = run_lithology_benchmark(nz=12, days=1.5, verbose=True)
    print_lithology_benchmark_table(bench)

    print("\n[2] Validazione litotipi vulcanici originali (retrocompatibilita')...")
    results = run_validation()

    print("\n[3] Scaling laws Monte Carlo su 20 litotipi (n=32, nz=40)...")
    sl = compute_scaling_laws(n_samples=32, seed=42, nz=40)
    if sl:
        print(f"  Scaling law: {sl['law']}")
        print(f"  R2 = {sl['R2']:.3f}  (n={sl['n']} campioni)")
        if "lits" in sl:
            from collections import Counter
            cnt = Counter(sl["lits"])
            top = sorted(cnt.items(), key=lambda x: -x[1])[:5]
            print(f"  Litotipi piu' campionati: {top}")
    else:
        print("  Attenzione: scaling law non convergente -- aumentare n_samples")

    print("\n[4] Q2_LOO (leave-one-lithology-out) ...")
    m_ref = build_model("tuff_unwelded", override={"nz": 10})
    q2_res = m_ref.q2_leave_one_out(bench=bench, verbose=True)
    if "Q2_LOO" in q2_res:
        print(f"  Q2_LOO = {q2_res['Q2_LOO']:.3f}  "
              f"RMSE_log10 = {q2_res.get('RMSE_log10',float('nan')):.3f}")

    print("\n[5] Summary fisica -- 20 litotipi:")
    print(f"  {'Litotipo':<32} {'Regime':<22} {'C_in':>9} "
          f"{'Pe_H':>7} {'F_eq':>7} {'eps':>7}")
    print("  " + "-"*90)
    for name in GEOLITHOTYPES:
        r = bench.get(name, {})
        if "error" in r or not r:
            continue
        label  = r.get("label", name)
        regime = r.get("regime", "?")
        Cin    = r.get("Cin_Bq_m3",  float("nan"))
        Pe     = r.get("Pe_H",        float("nan"))
        F_eq   = r.get("F_eq",        float("nan"))
        eps    = r.get("eps_mean",     float("nan"))
        print(f"  {label:<32} {regime:<22} {Cin:9.1f} "
              f"{Pe:7.4f} {F_eq:7.3f} {eps:7.4f}")

    print("\n[6] Generazione figura catena causale (litotipi vulcanici)...")
    generate_causal_chain_figure(
        results,
        outpath="/home/user/workspace/causal_chain_summary.png"
    )

    print("\nDone. File generati:")
    print("  volcanic_radon_model.py  (GeoLithotypes Extension)")
    print("  causal_chain_summary.png")


def plot_volcanic_comparison(results: dict, save_path: str = None,
                             dpi: int = 150):
    """Wrapper for plot_volcanic_comparison_paper, legacy signature."""
    return plot_volcanic_comparison_paper(
        results=results,
        save_path=save_path,
        dpi=dpi,
    )


def plot_bifurcation(bif: dict, save_path: str = None, dpi: int = 150):
    """Wrapper for plot_bifurcation_paper, legacy signature."""
    return plot_bifurcation_paper(
        bif=bif,
        save_path=save_path,
        dpi=dpi,
    )


import os as _os
import time as _time
import warnings as _warnings
from collections import defaultdict as _defaultdict

try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def safe_log10(x, eps=1e-30):
    """log10 with values clamped to eps to avoid log(0)."""
    arr = np.maximum(np.asarray(x, dtype=float), eps)
    return np.log10(arr)


def build_design_matrix(df, predictors, add_intercept=True, log_transform=True):
    """Builds the log10 design matrix X for OLS regression."""
    n = len(df)
    cols = []
    names = []
    if add_intercept:
        cols.append(np.ones(n))
        names.append("intercept")
    for pred in predictors:
        if pred not in df.columns:
            raise KeyError(
                "Predittore '{}' non trovato nel DataFrame. "
                "Colonne disponibili: {}".format(pred, list(df.columns))
            )
        vals = df[pred].values.astype(float)
        if log_transform:
            vals = safe_log10(vals)
        cols.append(vals)
        if log_transform:
            names.append("log10_" + pred)
        else:
            names.append(pred)
    X = np.column_stack(cols)
    return X, names


def ols_fit(X, y):
    """OLS fit beta = (X'X)^-1 X'y via numerical lstsq."""
    from scipy import stats as _stats
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat  = X @ coef
    resid = y - yhat
    sse   = float(np.dot(resid, resid))
    sigma2 = sse / max(n - k, 1)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    # propagate NaN for singular designs: a failed fit must NOT be reported as
    # "all coefficients significant-less" (t=0, p=1); it is reported as NaN
    se_ok  = np.isfinite(se) & (se > 0)
    t_stat = np.where(se_ok, coef / np.where(se_ok, se, 1.0), np.nan)
    p_val  = np.where(np.isfinite(t_stat),
                      2.0 * _stats.t.sf(np.abs(np.nan_to_num(t_stat)), df=max(n - k, 1)),
                      np.nan)
    return {
        "coef":   coef,
        "yhat":   yhat,
        "resid":  resid,
        "se":     se,
        "t_stat": t_stat,
        "p_val":  p_val,
        "n":      n,
        "k":      k,
        "SSE":    sse,
        "SST":    float(np.var(y) * n),
    }


def evaluate_scaling_model(y_true, y_pred, k):
    """Goodness-of-fit metrics for log-linear scaling models."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n   = len(y_true)
    res = y_true - y_pred
    sse = float(np.dot(res, res))
    sst = float(np.var(y_true) * n)
    r2  = 1.0 - sse / max(sst, 1e-30)
    rmse = float(np.sqrt(sse / max(n, 1)))
    mae  = float(np.mean(np.abs(res)))
    # AIC/BIC assume gaussian likelihood
    sigma2_hat = sse / max(n, 1)
    aic = n * np.log(max(sigma2_hat, 1e-30)) + 2.0 * k
    bic = n * np.log(max(sigma2_hat, 1e-30)) + k * np.log(max(n, 1))
    # Fraction within factor 2/3/5 (on a linear scale)
    ratio = 10.0 ** y_pred / np.maximum(10.0 ** y_true, 1e-30)
    f2 = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))
    f3 = float(np.mean((ratio >= 1.0/3) & (ratio <= 3.0)))
    f5 = float(np.mean((ratio >= 0.2) & (ratio <= 5.0)))
    return {
        "R2": r2, "RMSE": rmse, "MAE": mae,
        "AIC": aic, "BIC": bic,
        "SSE": sse, "SST": sst,
        "n": n, "k": k,
        "F2": f2, "F3": f3, "F5": f5,
    }


def scaling_law_string(coef, feature_names, model_name="M0"):
    """Formats the fitted scaling law as log-linear and multiplicative strings."""
    lines = ["=" * 60]
    lines.append("Scaling Law — {}".format(model_name))
    lines.append("=" * 60)
    lines.append("log10(C*) = ")
    terms = []
    for i, (c, name) in enumerate(zip(coef, feature_names)):
        if name == "intercept":
            terms.append("  {:+.4f}".format(c))
        else:
            terms.append("  {:+.4f} x {}".format(c, name))
    lines.append("\n".join(terms))
    lines.append("")
    lines.append("Forma moltiplicativa (scala lineare):")
    if feature_names[0] == "intercept":
        base = 10.0 ** coef[0]
        mult_parts = ["  C* = {:.5g}".format(base)]
        for c, name in zip(coef[1:], feature_names[1:]):
            var = name.replace("log10_", "")
            mult_parts.append(" x {}^({:+.4f})".format(var, c))
        lines.append("".join(mult_parts))
    lines.append("=" * 60)
    return "\n".join(lines)


_RA_PREDICTORS = ["PeH", "DaRn", "phi_g", "Sw", "Pi_src", "Pi_vent"]
_RA_REGIMES    = [
    "barometric_pumping",
    "fracture_bypass",
    "decay_limited",
    "advective",
    "mixed_transition",
]
_LAM_RN = 2.098e-6   # Rn-222 decay constant, s^-1
_D0_REF = 1.1e-5     # reference radon diffusivity in air, m^2/s
_MU_AIR = 1.82e-5    # dynamic viscosity of air, Pa s
_C_REF  = 1000.0     # reference concentration scale for C*, Bq/m^3


def make_regimeaware_lhs_samples(n_samples=1700, seed=42, lits=None):
    """Latin hypercube samples over lithotype and transport parameters."""
    if lits is None:
        lits = list(GEOLITHOTYPES.keys())
    n_lits = len(lits)

    # bounds: dP [Pa], Ra226 [Bq/kg], ACH [h^-1], theta fraction, H [m],
    # P_atm_amp [Pa] (sampled so the barometric_pumping regime is reachable:
    # Pi_baro > 1 requires amp > 0.01*P_atm0 ~ 1013 Pa)
    low  = np.array([0.5, 5.0, 0.02, 0.10, 2.2, 30.0])
    high = np.array([50.0, 500.0, 2.0, 0.90, 8.0, 1500.0])
    n_cont = 6

    rng = np.random.default_rng(seed)

    if _HAS_QMC:
        try:
            from scipy.stats.qmc import LatinHypercube, scale as _qmc_scale
            sampler = LatinHypercube(d=n_cont, seed=seed)
            unit    = sampler.random(n_samples)
            scaled  = _qmc_scale(unit, low, high)
        except Exception:
            scaled = rng.uniform(low, high, size=(n_samples, n_cont))
    else:
        scaled = rng.uniform(low, high, size=(n_samples, n_cont))

    # Assign lithotypes with a deterministic uniform distribution
    lit_indices = np.floor(np.linspace(0, n_lits - 1e-9, n_samples)).astype(int)
    rng.shuffle(lit_indices)

    samples = []
    for i in range(n_samples):
        samples.append({
            "dP":         float(scaled[i, 0]),
            "Ra":         float(scaled[i, 1]),
            "ACH":        float(scaled[i, 2]),
            "theta_frac": float(scaled[i, 3]),
            "H":          float(scaled[i, 4]),
            "P_amp":      float(scaled[i, 5]),
            "lit_name":   lits[lit_indices[i]],
        })
    return samples


def run_regimeaware_sample(sample, nz=15, days=3.0):
    """Runs one ODE simulation via build_model + simulate_fast, no surrogate formulas."""
    lit_name = sample["lit_name"]
    try:
        lit_data = GEOLITHOTYPES[lit_name]
    except KeyError:
        return None

    # theta_init interpolated between theta_r and theta_s (saturation fraction)
    theta_r = float(lit_data.get("theta_r", 0.02))
    theta_s = float(lit_data.get("phi_m", 0.35))
    frac    = float(sample["theta_frac"])
    theta_init = theta_r + frac * (theta_s - theta_r)
    theta_init = float(np.clip(theta_init, theta_r + 1e-4, theta_s - 1e-4))

    override = {
        "dP_init":    float(sample["dP"]),
        "dP_bottom":  float(sample["dP"]),
        "Ra226_Bqkg": float(sample["Ra"]),
        "ACH_closed": float(sample["ACH"]),
        "theta_init": theta_init,             # mapped to theta0 by build_model
        "H":          float(sample["H"]),     # sampled SOIL column depth [m]
        "P_atm_amp":  float(sample.get("P_amp", 150.0)),  # barometric amplitude
        "nz":         int(nz),
    }
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            mod = build_model(lit_name, override=override)
            sol = mod.simulate_fast(days=float(days), dt_out=7200.0)
    except Exception:
        return None

    return extract_regimeaware_features_from_solution(
        mod, sol, lit_name, sample_id=None
    )


def extract_regimeaware_features_from_solution(model, sol, lith_name,
                                                sample_id=None):
    """Extracts dimensionless groups and physical features from a completed ODE run."""
    try:
        res  = model.extract_results(sol)
        p    = model.p
        nz   = model.nz

        # final indoor (basement) concentration, Bq/m^3
        C_in = float(res["C_bs"][-1])
        if not np.isfinite(C_in) or C_in <= 0.0:
            return None

        phi_m  = float(p.get("phi_m",   0.35))
        phi_f  = float(p.get("phi_f",   0.01))
        k0     = float(p.get("k0",      1e-12))
        dP     = float(p.get("dP_init", 5.0))
        ACH    = float(p.get("ACH_closed", 0.5))
        H      = float(p.get("H", p.get("H_room", 2.5)))   # soil column depth [m]
        Ra     = float(p.get("Ra226_Bqkg", 50.0))
        rho_b  = float(p.get("rho_bulk", 1500.0))
        # actual emanation state realized by the run (dynamic eps_mem), not a fixed key
        try:
            Ef = float(np.mean(res["eps_mem"][:, -1]))
        except Exception:
            Ef = float(p.get("eps_max", 0.3)) * 0.6

        # mean water content and gas porosity at final time (direct, no Se detour)
        theta_final = res["theta"][:, -1]
        theta_mean  = float(np.mean(theta_final))
        Sw    = float(np.clip(theta_mean / max(phi_m, 1e-6), 0.0, 1.0))
        phi_g = float(np.clip(phi_m - theta_mean, 1e-4, 1.0))

        # APPARENT Rogers-Nielson diffusivity — the SAME law the solver uses,
        # so the dimensionless groups describe the simulated physics
        # (L_d = sqrt(D_a/lambda); Pe/Da built on D_a)
        Sw_rn = min(max(theta_mean / max(phi_m, 1e-12), 1e-6), 0.9999)
        D_a   = float(p.get("D0", _D0_REF)) * phi_m *                 np.exp(-6.0 * Sw_rn * phi_m - 6.0 * Sw_rn ** (14.0 * phi_m))
        D_eff = max(D_a, 1e-12)

        # gas/water partition porosity (Ostwald L ~ 0.25 at 20 C)
        beta_p = max(phi_g + 0.25 * theta_mean, 1e-6)

        v_eff  = k0 * dP / (_MU_AIR * max(H, 0.1))
        Pe_H   = k0 * dP / (_MU_AIR * D_eff)          # hydraulic Peclet number
        Da_Rn  = _LAM_RN * H ** 2 / D_eff              # diffusive Damkohler number
        Da_H   = _LAM_RN * H / max(v_eff, 1e-20)       # advective Damkohler number
        # dimensionless source: deep pore-air concentration over reference conc.
        Pi_src = rho_b * Ra * Ef / (beta_p * _C_REF)
        Pi_vent = ACH / (3600.0 * _LAM_RN)             # dimensionless ventilation rate
        kDP    = k0 * dP / (_MU_AIR * _D0_REF)         # alternate Peclet definition
        Cstar  = C_in / _C_REF

        # barometric pumping amplitude relative to 1% of the reference pressure
        # (reads the model's ACTUAL parameter keys)
        Patm_amp = float(p.get("P_atm_amp", 0.0))
        Patm_ref = float(p.get("P_atm0",    101325.0))
        Pi_baro  = Patm_amp / max(Patm_ref * 0.01, 1e-10)

        regime_declared = str(GEOLITHOTYPES.get(lith_name, {}).get("regime", "unknown"))

        row = {
            "Cin":              C_in,
            "log10_Cin":        float(safe_log10(C_in)),
            "Cstar":            Cstar,
            "log10_Cstar":      float(safe_log10(Cstar)),
            "PeH":              Pe_H,
            "DaRn":             Da_Rn,
            "DaH":              Da_H,
            "phi_g":            phi_g,
            "Sw":               Sw,
            "Pi_src":           Pi_src,
            "Pi_vent":          Pi_vent,
            "kDP":              kDP,
            "phi_mac":          phi_f,
            "Pi_baro":          Pi_baro,
            "k0":               k0,
            "dP":               dP,
            "ACH":              ACH,
            "H":                H,
            "Ra226":            Ra,
            "lithotype":        lith_name,
            "regime_declared":  regime_declared,
        }
        row["regime_inferred"] = infer_physical_regime(row)
        if sample_id is not None:
            row["sample_id"] = sample_id
        return row

    except Exception:
        return None


def infer_physical_regime(row):
    """Classifies a run into one of 5 transport regimes from dimensionless numbers."""
    PeH     = row.get("PeH",     np.nan)
    DaRn    = row.get("DaRn",    row.get("DaH", np.nan))
    phi_mac = row.get("phi_mac", row.get("phi_f", 0.0))
    Pi_baro = row.get("Pi_baro", 0.0)

    if np.isfinite(float(Pi_baro)) and float(Pi_baro) > 1.0:
        return "barometric_pumping"
    # 0.08 is reachable by the fractured/vesicular lithotypes in GEOLITHOTYPES
    # (phi_f up to 0.12); the previous 0.30 made this regime dead for all inputs
    if np.isfinite(float(phi_mac)) and float(phi_mac) >= 0.08:
        return "fracture_bypass"
    if np.isfinite(float(DaRn))   and float(DaRn) > 100.0:
        return "decay_limited"
    if np.isfinite(float(PeH))    and float(PeH)  > 1.0:
        return "advective"
    return "mixed_transition"


def generate_regimeaware_dataset(n_samples=1700, seed=42, nz=15, days=3.0,
                                  lits=None, min_Cin=0.5, min_PeH=1e-8,
                                  verbose=True):
    """Runs n_samples regime-aware ODE simulations and returns a filtered dataset."""
    samples = make_regimeaware_lhs_samples(n_samples=n_samples, seed=seed, lits=lits)
    rows    = []
    n_fail  = 0
    t0      = _time.time()

    for i, samp in enumerate(samples):
        row = run_regimeaware_sample(samp, nz=nz, days=days)
        if row is None:
            n_fail += 1
            continue
        # physical filters on Cin and PeH
        if row.get("Cin",  0.0) < min_Cin:
            n_fail += 1
            continue
        if row.get("PeH",  0.0) < min_PeH:
            n_fail += 1
            continue
        row["run_id"] = len(rows)
        rows.append(row)
        if verbose and (i + 1) % 100 == 0:
            elapsed = _time.time() - t0
            rate    = (i + 1) / max(elapsed, 1e-6)
            eta     = (n_samples - i - 1) / max(rate, 1e-6)
            print("  [{}/{}]  validi={:4d}  falliti={:3d}  "
                  "{:.0f}s elapsed  ETA {:.0f}s".format(
                      i + 1, n_samples, len(rows), n_fail, elapsed, eta))

    total_elapsed = _time.time() - t0
    if verbose:
        print("Dataset generato: {}/{} validi  ({:.0f}s)".format(
            len(rows), n_samples, total_elapsed))

    if _HAS_PANDAS:
        return _pd.DataFrame(rows)
    return rows


def dataset_regime_summary(df):
    """Summary statistics of Cin, PeH, DaRn grouped by inferred regime."""
    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        records = df.to_dict("records")
    else:
        records = list(df)

    by_regime = _defaultdict(list)
    for row in records:
        by_regime[row.get("regime_inferred", "unknown")].append(row)

    n_total = max(len(records), 1)
    summary = []
    for regime in _RA_REGIMES + ["unknown"]:
        rows_r = by_regime.get(regime, [])
        if not rows_r:
            continue
        Cin_vals   = [r["Cin"]  for r in rows_r if np.isfinite(r.get("Cin",  np.nan))]
        PeH_vals   = [r["PeH"]  for r in rows_r if np.isfinite(r.get("PeH",  np.nan))]
        DaRn_vals  = [r["DaRn"] for r in rows_r if np.isfinite(r.get("DaRn", np.nan))]
        summary.append({
            "regime":      regime,
            "n":           len(rows_r),
            "percent":     100.0 * len(rows_r) / n_total,
            "Cin_median":  float(np.median(Cin_vals))  if Cin_vals  else np.nan,
            "Cin_mean":    float(np.mean(Cin_vals))    if Cin_vals  else np.nan,
            "PeH_median":  float(np.median(PeH_vals))  if PeH_vals  else np.nan,
            "DaRn_median": float(np.median(DaRn_vals)) if DaRn_vals else np.nan,
        })
    return summary




def _df_to_arrays(df, col):
    """Extracts a float array from a DataFrame or list of dicts."""
    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        return df[col].values.astype(float)
    return np.array([r[col] for r in df], dtype=float)


def _get_regime_col(df):
    """Returns the regime column, preferring the inferred one if present."""
    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        for col in ("regime_inferred", "regime_declared", "regime"):
            if col in df.columns:
                return df[col].values
        return np.array(["unknown"] * len(df))
    for col in ("regime_inferred", "regime_declared", "regime"):
        if col in df[0]:
            return np.array([r.get(col, "unknown") for r in df])
    return np.array(["unknown"] * len(df))


def fit_scaling_M0_global(df):
    """M0: global log-linear scaling law for log10(C*), k=7, no regime split."""
    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        sub = df.copy()
    else:
        sub = df

    X, feat_names = build_design_matrix(sub, _RA_PREDICTORS,
                                         add_intercept=True, log_transform=True)
    y = safe_log10(_df_to_arrays(sub, "Cstar"))
    fit    = ols_fit(X, y)
    metrics = evaluate_scaling_model(y, fit["yhat"], k=fit["k"])
    law_str = scaling_law_string(fit["coef"], feat_names, model_name="M0")
    return {
        "model":         "M0",
        "coef":          fit["coef"],
        "feature_names": feat_names,
        "se":            fit["se"],
        "t_stat":        fit["t_stat"],
        "p_val":         fit["p_val"],
        "metrics":       metrics,
        "yhat":          fit["yhat"],
        "resid":         fit["resid"],
        "law_string":    law_str,
        "X":             X,
        "y":             y,
    }


def fit_scaling_M1_regime_dummies(df, reference_regime="advective"):
    """M1: M0 plus 4 regime dummies shifting only the intercept, k=11."""
    regimes_arr = _get_regime_col(df)
    # build dummies only for regimes actually present: an all-zero dummy makes
    # X'X exactly singular and poisons every SE/p-value with NaN
    all_regimes = [r for r in _RA_REGIMES
                   if r != reference_regime and bool(np.any(regimes_arr == r))]

    X_base, feat_names_base = build_design_matrix(
        df, _RA_PREDICTORS, add_intercept=True, log_transform=True
    )
    dummy_cols = []
    dummy_names = []
    for reg in all_regimes:
        col = (regimes_arr == reg).astype(float)
        dummy_cols.append(col)
        dummy_names.append("D_" + reg)

    X = np.column_stack([X_base] + dummy_cols)
    feat_names = feat_names_base + dummy_names
    y = safe_log10(_df_to_arrays(df, "Cstar"))

    fit     = ols_fit(X, y)
    metrics = evaluate_scaling_model(y, fit["yhat"], k=fit["k"])
    law_str = scaling_law_string(fit["coef"], feat_names, model_name="M1")

    return {
        "model":              "M1",
        "coef":               fit["coef"],
        "feature_names":      feat_names,
        "se":                 fit["se"],
        "t_stat":             fit["t_stat"],
        "p_val":              fit["p_val"],
        "metrics":            metrics,
        "yhat":               fit["yhat"],
        "resid":              fit["resid"],
        "law_string":         law_str,
        "dummy_cols":         dummy_names,
        "reference_regime":   reference_regime,
        "X":                  X,
        "y":                  y,
    }


def fit_scaling_M2_interaction(df, reference_regime="advective"):
    """M2: M1 plus a log10(PeH*Sw) interaction term, k=12."""
    res_m1 = fit_scaling_M1_regime_dummies(df, reference_regime=reference_regime)
    X_base = res_m1["X"]
    feat_names_base = res_m1["feature_names"]
    y      = res_m1["y"]

    PeH_vals  = _df_to_arrays(df, "PeH")
    Sw_vals   = _df_to_arrays(df, "Sw")
    # true interaction: PRODUCT of the log-predictors. log10(PeH*Sw) would be
    # log10(PeH)+log10(Sw), an exact linear combination of columns already in
    # X_base (rank-deficient design, zero added information).
    interaction = safe_log10(PeH_vals) * safe_log10(np.maximum(Sw_vals, 1e-6))

    X         = np.column_stack([X_base, interaction])
    feat_names = feat_names_base + ["log10PeH_x_log10Sw"]

    fit     = ols_fit(X, y)
    metrics = evaluate_scaling_model(y, fit["yhat"], k=fit["k"])
    law_str = scaling_law_string(fit["coef"], feat_names, model_name="M2")

    return {
        "model":           "M2",
        "coef":            fit["coef"],
        "feature_names":   feat_names,
        "se":              fit["se"],
        "t_stat":          fit["t_stat"],
        "p_val":           fit["p_val"],
        "metrics":         metrics,
        "yhat":            fit["yhat"],
        "resid":           fit["resid"],
        "law_string":      law_str,
        "X":               X,
        "y":               y,
    }


def fit_scaling_M3_by_regime(df):
    """M3: independent M0 fit per regime, maximizes R2 over parsimony."""
    regimes_arr  = _get_regime_col(df)
    y_global     = safe_log10(_df_to_arrays(df, "Cstar"))
    # NaN-initialize: rows whose regime is too small to fit are excluded from
    # the global metrics instead of contributing uninitialized memory
    yhat_global  = np.full_like(y_global, np.nan)

    per_regime = {}
    coef_table = []
    for reg in _RA_REGIMES:
        mask = (regimes_arr == reg)
        if mask.sum() < 5:
            continue
        if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
            sub = df[mask].copy()
        else:
            sub = [r for r, m in zip(df, mask) if m]

        X, feat_names = build_design_matrix(
            sub, _RA_PREDICTORS, add_intercept=True, log_transform=True
        )
        y_sub = safe_log10(_df_to_arrays(sub, "Cstar"))
        fit   = ols_fit(X, y_sub)
        metr  = evaluate_scaling_model(y_sub, fit["yhat"], k=fit["k"])

        yhat_global[mask] = fit["yhat"]
        per_regime[reg] = {
            "coef":         fit["coef"],
            "feature_names":feat_names,
            "se":           fit["se"],
            "t_stat":       fit["t_stat"],
            "p_val":        fit["p_val"],
            "metrics":      metr,
            "n":            int(mask.sum()),
        }
        for i, (c, fn) in enumerate(zip(fit["coef"], feat_names)):
            coef_table.append({
                "regime":   reg,
                "feature":  fn,
                "coef":     float(c),
                "se":       float(fit["se"][i]),
                "p_val":    float(fit["p_val"][i]),
            })

    covered = np.isfinite(yhat_global)
    metrics_global = evaluate_scaling_model(y_global[covered], yhat_global[covered],
                                            k=7 * len(per_regime))
    return {
        "model":          "M3",
        "per_regime":     per_regime,
        "metrics_global": metrics_global,
        "yhat_global":    yhat_global,
        "y_global":       y_global,
        "n_uncovered":    int(np.sum(~covered)),
        "coef_table":     coef_table,
    }


def fit_scaling_M4_piecewise(df, pe_break=0.5):
    """M4: piecewise fit split at PeH = pe_break, quadratic term above break."""
    PeH_arr     = _df_to_arrays(df, "PeH")
    mask_hi     = PeH_arr > pe_break
    mask_lo     = ~mask_hi
    y_global    = safe_log10(_df_to_arrays(df, "Cstar"))
    # NaN-initialize: a segment with < 5 rows stays unfitted and must not leak
    # uninitialized memory into the global metrics
    yhat_global = np.full_like(y_global, np.nan)

    # Segment A - high Pe: include log10(PeH)^2
    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        sub_hi = df[mask_hi].copy()
        sub_lo = df[mask_lo].copy()
    else:
        sub_hi = [r for r, m in zip(df, mask_hi) if m]
        sub_lo = [r for r, m in zip(df, mask_lo) if m]

    fit_hi = None
    if mask_hi.sum() >= 5:
        X_hi, fn_hi = build_design_matrix(sub_hi, _RA_PREDICTORS,
                                           add_intercept=True, log_transform=True)
        PeH_hi_log = safe_log10(_df_to_arrays(sub_hi, "PeH"))
        X_hi  = np.column_stack([X_hi, PeH_hi_log ** 2])
        fn_hi = fn_hi + ["log10_PeH_sq"]
        y_hi  = y_global[mask_hi]
        fit_hi = ols_fit(X_hi, y_hi)
        yhat_global[mask_hi] = fit_hi["yhat"]

    fit_lo = None
    if mask_lo.sum() >= 5:
        X_lo, fn_lo = build_design_matrix(sub_lo, _RA_PREDICTORS,
                                           add_intercept=True, log_transform=True)
        y_lo  = y_global[mask_lo]
        fit_lo = ols_fit(X_lo, y_lo)
        yhat_global[mask_lo] = fit_lo["yhat"]

    k_tot = ((fit_hi["k"] if fit_hi else 0) +
             (fit_lo["k"] if fit_lo else 0))
    covered = np.isfinite(yhat_global)
    if k_tot == 0 or not np.any(covered):
        metrics_global = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                          "AIC": np.nan, "BIC": np.nan, "SSE": np.nan,
                          "SST": np.nan, "n": 0, "k": 0, "F2": np.nan,
                          "F3": np.nan, "F5": np.nan}
    else:
        metrics_global = evaluate_scaling_model(y_global[covered],
                                                yhat_global[covered], k=k_tot)

    return {
        "model":          "M4",
        "fit_hi":         fit_hi,
        "fit_lo":         fit_lo,
        "n_hi":           int(mask_hi.sum()),
        "n_lo":           int(mask_lo.sum()),
        "metrics_global": metrics_global,
        "yhat_global":    yhat_global,
        "y_global":       y_global,
        "pe_break":       pe_break,
    }


def compare_scaling_models_M0_M4(df):
    """Runs M0-M4 and collects their metrics into one comparison table."""
    print("  Fitting M0 (globale)...")
    M0 = fit_scaling_M0_global(df)
    print("  Fitting M1 (+ dummy regime)...")
    M1 = fit_scaling_M1_regime_dummies(df)
    print("  Fitting M2 (+ interazione PeH*Sw)...")
    M2 = fit_scaling_M2_interaction(df)
    print("  Fitting M3 (per regime)...")
    M3 = fit_scaling_M3_by_regime(df)
    print("  Fitting M4 (piecewise PeH)...")
    M4 = fit_scaling_M4_piecewise(df)

    def _row(name, fit_result):
        if name == "M3":
            m = fit_result["metrics_global"]
            # total fitted parameters: 7 coefficients per fitted regime
            k = 7 * len(fit_result["per_regime"])
        elif name == "M4":
            m = fit_result["metrics_global"]
            k = m["k"]
        else:
            m = fit_result["metrics"]
            k = m["k"]
        return {
            "model": name,
            "n": int(m["n"]),
            "k": int(k),
            "R2":   round(m["R2"],   4),
            "RMSE": round(m["RMSE"], 4),
            "AIC":  round(m["AIC"],  1),
            "BIC":  round(m["BIC"],  1),
            "F2":   round(m["F2"],   4),
        }

    table = []
    for name, fit in [("M0", M0), ("M1", M1), ("M2", M2), ("M3", M3), ("M4", M4)]:
        table.append(_row(name, fit))
    # data-driven notes (computed from the actual metrics, not hard-coded claims)
    aic_vals = {row["model"]: row["AIC"] for row in table}
    r2_vals  = {row["model"]: row["R2"]  for row in table}
    best_aic = min(aic_vals, key=aic_vals.get)
    best_r2  = max(r2_vals,  key=r2_vals.get)
    for row in table:
        notes = []
        if row["model"] == "M0":
            notes.append("baseline globale")
        if row["model"] == best_aic:
            notes.append("AIC minimo")
        if row["model"] == best_r2:
            notes.append("R2 massimo")
        row["note"] = "; ".join(notes)

    return {"M0": M0, "M1": M1, "M2": M2, "M3": M3, "M4": M4, "table": table}


def nested_f_test(model_reduced, model_full):
    """Nested F-test: F = [(SSE_r-SSE_f)/(k_f-k_r)] / [SSE_f/(n-k_f)]."""
    from scipy.stats import f as _f_dist

    def _extract(m):
        if "metrics" in m:
            return m["metrics"]["SSE"], m["metrics"]["k"], m["metrics"]["n"]
        elif "metrics_global" in m:
            return m["metrics_global"]["SSE"], m["metrics_global"]["k"], m["metrics_global"]["n"]
        else:
            return m["SSE"], m["k"], m["n"]

    SSE_r, k_r, n = _extract(model_reduced)
    SSE_f, k_f, _ = _extract(model_full)

    delta_k   = int(k_f - k_r)
    delta_sse = float(SSE_r - SSE_f)
    df2       = int(n - k_f)

    if delta_k <= 0 or SSE_f <= 0 or df2 <= 0:
        return {
            "F_stat": np.nan, "p_val": np.nan,
            "df1": delta_k, "df2": df2,
            "SSE_r": SSE_r, "SSE_f": SSE_f,
            "k_r": k_r, "k_f": k_f, "n": n,
            "delta_k": delta_k, "delta_sse": delta_sse,
        }
    F_stat = (delta_sse / delta_k) / (SSE_f / df2)
    p_val  = float(1.0 - _f_dist.cdf(F_stat, delta_k, df2))

    return {
        "F_stat":    float(F_stat),
        "p_val":     p_val,
        "df1":       delta_k,
        "df2":       df2,
        "SSE_r":     SSE_r,
        "SSE_f":     SSE_f,
        "k_r":       k_r,
        "k_f":       k_f,
        "n":         n,
        "delta_k":   delta_k,
        "delta_sse": delta_sse,
    }


def compute_predictor_vif(df, predictors=None):
    """VIF_j = 1/(1-R2_j); VIF>5 moderate collinearity, VIF>10 critical."""
    if predictors is None:
        predictors = _RA_PREDICTORS + ["kDP"]

    # Log10 matrix of the predictors
    X_dict = {}
    for pred in predictors:
        try:
            vals = _df_to_arrays(df, pred)
            X_dict[pred] = safe_log10(vals)
        except (KeyError, Exception):
            pass
    preds_ok = list(X_dict.keys())
    X_mat = np.column_stack([X_dict[p] for p in preds_ok])

    vif = {}
    for j, pred in enumerate(preds_ok):
        y_j   = X_mat[:, j]
        X_oth = np.delete(X_mat, j, axis=1)
        Xd    = np.column_stack([np.ones(len(y_j)), X_oth])
        try:
            coef_j = np.linalg.lstsq(Xd, y_j, rcond=None)[0]
            yhat_j = Xd @ coef_j
            ss_res = np.sum((y_j - yhat_j) ** 2)
            ss_tot = np.sum((y_j - y_j.mean()) ** 2)
            r2_j   = 1.0 - ss_res / max(ss_tot, 1e-30)
            vif[pred] = float(1.0 / max(1.0 - r2_j, 1e-10))
        except Exception:
            vif[pred] = np.nan

    # Add PeH-kDP correlation if both present
    if "PeH" in X_dict and "kDP" in X_dict:
        r = float(np.corrcoef(X_dict["PeH"], X_dict["kDP"])[0, 1])
        vif["_corr_PeH_kDP"] = r

    return vif


def test_kDP_redundancy(df):
    """Compares PeH-only, kDP-only, and combined fits to test kDP redundancy."""
    others = ["DaRn", "phi_g", "Sw", "Pi_src", "Pi_vent"]

    def _fit_variant(preds):
        X, _ = build_design_matrix(df, preds, add_intercept=True, log_transform=True)
        y    = safe_log10(_df_to_arrays(df, "Cstar"))
        fit  = ols_fit(X, y)
        return evaluate_scaling_model(y, fit["yhat"], k=fit["k"])

    V1 = _fit_variant(["PeH", "kDP"] + others)
    V2 = _fit_variant(["PeH"]        + others)
    V3 = _fit_variant(["kDP"]        + others)

    dR2_V1_V2 = abs(V1["R2"] - V2["R2"])
    dR2_V2_V3 = V2["R2"] - V3["R2"]

    if dR2_V1_V2 < 0.002:
        concl = ("kDP e' ridondante rispetto a PeH (delta_R2 = {:.4f}), come "
                 "ATTESO PER COSTRUZIONE: kDP = PeH * (D_eff/D0) e' una funzione "
                 "algebrica di predittori gia' presenti. Il test e' un sanity "
                 "check, non una scoperta. kDP va rimosso dal modello finale."
                 .format(dR2_V1_V2))
    else:
        concl = ("kDP apporta informazione aggiuntiva (delta_R2 = {:.4f}) — "
                 "anomalo, dato che kDP e' algebricamente dipendente da PeH e "
                 "D_eff: verificare la costruzione del dataset.".format(dR2_V1_V2))

    return {
        "V1": V1, "V2": V2, "V3": V3,
        "delta_R2_V1_V2": dR2_V1_V2,
        "delta_R2_V2_V3": dR2_V2_V3,
        "conclusion": concl,
    }


def crossval_kfold_scaling(df, model="M0", kfold=10, seed=42):
    """K-fold cross-validation estimating out-of-sample Q2 = 1 - SSE_oot/SST."""
    y_all   = safe_log10(_df_to_arrays(df, "Cstar"))
    regimes = _get_regime_col(df)
    n       = len(y_all)

    rng        = np.random.default_rng(seed)
    fold_idx   = np.arange(n) % kfold
    fold_idx   = rng.permutation(fold_idx)

    yhat_oot = np.empty(n)
    for k in range(kfold):
        te_mask = fold_idx == k
        tr_mask = ~te_mask

        if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
            df_tr = df[tr_mask].copy()
            df_te = df[te_mask].copy()
        else:
            df_tr = [r for r, m in zip(df, tr_mask) if m]
            df_te = [r for r, m in zip(df, te_mask) if m]

        if model == "M1":
            fit_k = fit_scaling_M1_regime_dummies(df_tr)
        else:
            fit_k = fit_scaling_M0_global(df_tr)

        X_te, _ = build_design_matrix(
            df_te, _RA_PREDICTORS, add_intercept=True, log_transform=True
        ) if model == "M0" else (
            _build_M1_design(df_te, fit_k), None
        )
        if model == "M1":
            X_te = _build_M1_design(df_te, fit_k)
        yhat_oot[te_mask] = X_te @ fit_k["coef"]

    sse_oot = float(np.sum((y_all - yhat_oot) ** 2))
    sst_tot = float(np.sum((y_all - y_all.mean()) ** 2))
    Q2      = 1.0 - sse_oot / max(sst_tot, 1e-30)
    rmse    = float(np.sqrt(sse_oot / n))

    return {
        "Q2":          Q2,
        "RMSE_oot":    rmse,
        "kfold":       kfold,
        "model":       model,
        "predictions": yhat_oot,
        "true_values": y_all,
    }


def _build_M1_design(df, m1_fit_ref):
    """Builds M1 design matrix for a subset, reusing dummy names from a prior fit."""
    X_base, _ = build_design_matrix(df, _RA_PREDICTORS,
                                     add_intercept=True, log_transform=True)
    regimes_arr = _get_regime_col(df)
    ref = m1_fit_ref.get("reference_regime", "advective")
    dummy_names = m1_fit_ref.get("dummy_cols", [])
    dummy_cols  = []
    for dname in dummy_names:
        reg = dname.replace("D_", "")
        dummy_cols.append((regimes_arr == reg).astype(float))
    if dummy_cols:
        return np.column_stack([X_base] + dummy_cols)
    return X_base


def crossval_leave_one_regime_out(df, model="M0"):
    """Leave-one-regime-out CV: trains on other regimes, tests generalization.

    Default model is M0: with M1 the held-out regime's dummy has no training
    rows, so its coefficient is structurally 0 and the test collapses to the
    reference-intercept prediction (documented limitation; use M0 for LORO)."""
    regimes_arr = _get_regime_col(df)
    regimes     = [r for r in _RA_REGIMES if (regimes_arr == r).sum() >= 5]
    results     = {}

    for reg in regimes:
        tr_mask = regimes_arr != reg
        te_mask = regimes_arr == reg

        if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
            df_tr = df[tr_mask].copy()
            df_te = df[te_mask].copy()
        else:
            df_tr = [r for r, m in zip(df, tr_mask) if m]
            df_te = [r for r, m in zip(df, te_mask) if m]

        y_te = safe_log10(_df_to_arrays(df_te, "Cstar"))

        try:
            if model == "M1":
                fit_tr = fit_scaling_M1_regime_dummies(df_tr)
                X_te   = _build_M1_design(df_te, fit_tr)
            else:
                fit_tr = fit_scaling_M0_global(df_tr)
                X_te, _ = build_design_matrix(df_te, _RA_PREDICTORS,
                                               add_intercept=True, log_transform=True)
            yhat_te = X_te @ fit_tr["coef"]
        except Exception:
            results[reg] = {"Q2": np.nan, "RMSE": np.nan, "bias": np.nan,
                            "n_test": int(te_mask.sum()), "predictions": None}
            continue

        res    = y_te - yhat_te
        sse_te = float(np.dot(res, res))
        sst_te = float(np.var(y_te) * len(y_te))
        Q2     = 1.0 - sse_te / max(sst_te, 1e-30)
        rmse   = float(np.sqrt(sse_te / max(len(y_te), 1)))
        bias   = float(np.mean(res))

        results[reg] = {
            "Q2":          Q2,
            "RMSE":        rmse,
            "bias":        bias,
            "n_test":      int(te_mask.sum()),
            "predictions": yhat_te,
        }
    return results


def bootstrap_scaling_coefficients(df, fit_func, n_boot=500, seed=42):
    """Bootstrap resampling for 95% CIs on OLS exponents, n_boot replicates."""
    rng = np.random.default_rng(seed)
    n   = len(df) if not (_HAS_PANDAS and isinstance(df, _pd.DataFrame)) \
          else len(df)

    # Fit on the full dataset to obtain the number of coefficients
    ref_fit = fit_func(df)
    k_coef  = len(ref_fit["coef"])
    coefs_boot = np.empty((n_boot, k_coef))
    r2_boot    = np.empty(n_boot)

    if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
        idx_all = df.index.to_numpy()
    else:
        idx_all = np.arange(n)

    for b in range(n_boot):
        idx_b = rng.choice(idx_all, size=n, replace=True)
        if _HAS_PANDAS and isinstance(df, _pd.DataFrame):
            df_b = df.loc[idx_b].copy()
        else:
            df_b = [df[i] for i in idx_b]
        try:
            fit_b = fit_func(df_b)
            coefs_boot[b] = fit_b["coef"]
            r2_boot[b]    = fit_b["metrics"]["R2"] \
                            if "metrics" in fit_b else np.nan
        except Exception:
            coefs_boot[b] = np.nan
            r2_boot[b]    = np.nan

    return {
        "coef_mean":     np.nanmean(coefs_boot, axis=0),
        "coef_ci95_lo":  np.nanpercentile(coefs_boot, 2.5, axis=0),
        "coef_ci95_hi":  np.nanpercentile(coefs_boot, 97.5, axis=0),
        "R2_mean":       float(np.nanmean(r2_boot)),
        "R2_ci95_lo":    float(np.nanpercentile(r2_boot, 2.5)),
        "R2_ci95_hi":    float(np.nanpercentile(r2_boot, 97.5)),
        "feature_names": ref_fit["feature_names"],
        "n_boot":        n_boot,
    }



def verify_exponent_signs(M3_result):
    """Checks physical sign consistency of M3 exponents per regime."""
    EXPECTED = {
        "log10_PeH":     +1,
        "log10_DaRn":    -1,
        "log10_phi_g":   +1,
        "log10_Sw":      -1,
        "log10_Pi_src":  +1,
        "log10_Pi_vent": -1,
    }
    PHYSICAL_INTERPRETATIONS = {
        ("log10_Sw", +1, "mixed_transition"):
            "Sw positivo nel Misto: compensazione termica/diffusiva",
        ("log10_Sw", +1, "fracture_bypass"):
            "Sw positivo in Frattura: fratture mantengono conduttivita' ad alta Sw",
        ("log10_phi_g", -1, "barometric_pumping"):
            "phi_g negativo in Barometrico: pompaggio dominante su diffusione",
    }

    results = {}
    for regime, fit_r in M3_result["per_regime"].items():
        feat_names = fit_r["feature_names"]
        coef       = fit_r["coef"]
        p_vals     = fit_r["p_val"]
        n_correct  = 0
        n_total    = 0
        ok_per_coef = {}
        anomalies   = []

        for i, (fn, c) in enumerate(zip(feat_names, coef)):
            if fn not in EXPECTED:
                continue
            n_total  += 1
            exp_sign  = EXPECTED[fn]
            act_sign  = int(np.sign(c)) if abs(c) > 1e-6 else 0
            p_val     = float(p_vals[i]) if i < len(p_vals) else np.nan
            is_sig    = p_val < 0.05 if np.isfinite(p_val) else False
            # a coefficient counts as physically correct ONLY if its sign matches;
            # non-significant wrong signs are tracked separately, not absolved
            sign_ok = (act_sign == exp_sign)
            ok_per_coef[fn] = sign_ok
            if sign_ok:
                n_correct += 1
            elif not is_sig:
                anomalies.append("{}: segno {:+d} atteso {:+d}, NON significativo "
                                 "(p={:.3g})".format(fn, act_sign, exp_sign, p_val))
            else:
                key = (fn, act_sign, regime)
                interp = PHYSICAL_INTERPRETATIONS.get(
                    key, "Segno anomalo: p={:.3g}, beta={:.3f}".format(p_val, c)
                )
                anomalies.append("{}: atteso {:+d}, trovato {:+d} — {}".format(
                    fn, exp_sign, act_sign, interp))

        results[regime] = {
            "expected_signs": {fn: EXPECTED[fn] for fn in EXPECTED},
            "coef_values":    {fn: float(c) for fn, c in zip(feat_names, coef)
                               if fn in EXPECTED},
            "physically_ok":  ok_per_coef,
            "n_correct":      n_correct,
            "n_total":        n_total,
            "anomalies":      anomalies,
        }
    return results


def _compose_recommendation(model_results, validation_results, vif_results):
    """Builds the recommendation string from the run's OWN numbers."""
    table   = model_results.get("table", [])
    best    = min(table, key=lambda r: r["AIC"]) if table else {"model": "?", "AIC": float("nan")}
    ft      = validation_results.get("f_tests", {}).get("M0_vs_M1") or {}
    vif     = vif_results.get("vif", {})
    kdp     = vif_results.get("kDP_test", {})
    parts = [
        "Modello con AIC minimo: {} (AIC={:.1f}).".format(best.get("model"), best.get("AIC", float("nan"))),
        "F-test M0 vs M1: p={:.3g}.".format(ft.get("p_val", float("nan"))),
    ]
    if "kDP" in vif:
        parts.append("VIF(kDP)={:.1f}, r(PeH,kDP)={:.2f}, delta_R2={:.4f} -> "
                     "kDP ridondante per costruzione.".format(
                         vif.get("kDP", float("nan")),
                         vif.get("_corr_PeH_kDP", float("nan")),
                         kdp.get("delta_R2_V1_V2", float("nan"))))
    parts.append("M3 fornisce gli esponenti fisici per regime.")
    return " ".join(parts)


def make_regimeaware_scaling_report(df, model_results, validation_results,
                                     vif_results):
    """Builds a structured summary report of the regime-aware pipeline."""
    import datetime as _dt
    summary = dataset_regime_summary(df)
    M1 = model_results["M1"]
    M3 = model_results["M3"]

    M3_laws = {}
    for reg, fit_r in M3["per_regime"].items():
        M3_laws[reg] = scaling_law_string(
            fit_r["coef"], fit_r["feature_names"], model_name="M3_" + reg
        )

    loro = validation_results.get("loro", {})
    LORO_table = [
        {"regime": reg, "Q2": v.get("Q2", np.nan),
         "RMSE": v.get("RMSE", np.nan), "n_test": v.get("n_test", 0)}
        for reg, v in loro.items()
    ]

    vif_table = [
        {"predictor": k, "VIF": v}
        for k, v in vif_results.get("vif", {}).items()
        if not k.startswith("_")
    ]

    report = {
        "timestamp":                _dt.datetime.now().isoformat(),
        "dataset_summary":          summary,
        "model_comparison_table":   model_results.get("table", []),
        "best_predictive_model":    "M1",
        "best_predictive_metrics":  M1["metrics"],
        "best_interpretable_model": "M3",
        "best_interpretable_metrics": M3["metrics_global"],
        "M1_law":                   M1["law_string"],
        "M3_laws_by_regime":        M3_laws,
        "VIF_table":                vif_table,
        "LORO_table":               LORO_table,
        "Q2_kfold":                 validation_results.get("kfold", {}).get("Q2", np.nan),
        "exponent_signs":           validation_results.get("exponent_signs", {}),
        "kDP_redundancy":           vif_results.get("kDP_test", {}),
        "recommendation": _compose_recommendation(model_results,
                                                   validation_results,
                                                   vif_results),
    }
    return report


def export_regimeaware_results_csv(df, results, outdir="./regimeaware_output"):
    """Writes pipeline results (dataset, model coefficients, VIF, LORO) to CSV files."""
    _os.makedirs(outdir, exist_ok=True)
    paths = {}

    if _HAS_PANDAS:
        p = _os.path.join(outdir, "regimeaware_dataset.csv")
        if isinstance(df, _pd.DataFrame):
            df.to_csv(p, index=False)
        else:
            _pd.DataFrame(df).to_csv(p, index=False)
        paths["dataset"] = p

        if "model_results" in results and "table" in results["model_results"]:
            p = _os.path.join(outdir, "model_comparison.csv")
            _pd.DataFrame(results["model_results"]["table"]).to_csv(p, index=False)
            paths["model_comparison"] = p

        if "model_results" in results and "M1" in results["model_results"]:
            M1 = results["model_results"]["M1"]
            rows_m1 = [
                {"feature": fn, "coef": float(c), "se": float(s), "p_val": float(p2)}
                for fn, c, s, p2 in zip(
                    M1["feature_names"], M1["coef"], M1["se"], M1["p_val"]
                )
            ]
            p = _os.path.join(outdir, "M1_coefficients.csv")
            _pd.DataFrame(rows_m1).to_csv(p, index=False)
            paths["M1_coef"] = p

        if "model_results" in results and "M3" in results["model_results"]:
            p = _os.path.join(outdir, "M3_coefficients_by_regime.csv")
            _pd.DataFrame(results["model_results"]["M3"]["coef_table"]).to_csv(
                p, index=False
            )
            paths["M3_coef"] = p

        vif_data = results.get("vif_results", {}).get("vif", {})
        if vif_data:
            p = _os.path.join(outdir, "VIF_table.csv")
            rows_vif = [{"predictor": k, "VIF": v}
                        for k, v in vif_data.items() if not k.startswith("_")]
            _pd.DataFrame(rows_vif).to_csv(p, index=False)
            paths["VIF"] = p

        loro = results.get("validation_results", {}).get("loro", {})
        if loro:
            p = _os.path.join(outdir, "LORO_results.csv")
            _pd.DataFrame([
                {"regime": r, "Q2": v["Q2"], "RMSE": v["RMSE"], "n_test": v["n_test"]}
                for r, v in loro.items()
            ]).to_csv(p, index=False)
            paths["LORO"] = p

    return paths


def plot_regimeaware_parity(df, model_result, savepath=None, dpi=180):
    """Parity plot of predicted vs simulated log10(C*), colored by regime."""
    REGIME_C = {
        "barometric_pumping": "#e67e22",
        "fracture_bypass":    "#27ae60",
        "decay_limited":      "#8e44ad",
        "advective":          "#e74c3c",
        "mixed_transition":   "#2980b9",
        "unknown":            "#aaaaaa",
    }
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PLOT_DARK)
    ax.set_facecolor(PLOT_PANEL)

    y_obs  = model_result.get("y", model_result.get("y_global"))
    if y_obs is None:
        y_obs = safe_log10(_df_to_arrays(df, "Cstar"))
    y_pred = model_result.get("yhat", model_result.get("yhat_global"))
    if y_pred is None:
        raise ValueError("model_result senza 'yhat'/'yhat_global': parity plot impossibile")
    regimes_arr = _get_regime_col(df)

    for reg in list(REGIME_C.keys()):
        mask = regimes_arr == reg
        if not mask.any():
            continue
        ax.scatter(y_obs[mask], y_pred[mask],
                   c=REGIME_C[reg], s=16, alpha=0.65,
                   label=reg.replace("_", " "), linewidths=0)

    lims = [min(y_obs.min(), y_pred.min()) - 0.3,
            max(y_obs.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "w-",  lw=1.5, alpha=0.85)
    ax.plot(lims, [v + np.log10(2) for v in lims], "w--", lw=0.9, alpha=0.4)
    ax.plot(lims, [v - np.log10(2) for v in lims], "w--", lw=0.9, alpha=0.4)
    ax.set_xlim(lims); ax.set_ylim(lims)

    model_name = model_result.get("model", "M?")
    m = model_result.get("metrics", {})
    ax.set_title(
        "Parity Plot — {model}   "
        "R2={r2:.3f}  RMSE={rmse:.3f}  F2={f2:.1%}".format(
            model=model_name,
            r2=m.get("R2", np.nan),
            rmse=m.get("RMSE", np.nan),
            f2=m.get("F2", np.nan),
        ),
        color=PLOT_TEXT, pad=8
    )
    ax.set_xlabel("log10(C*) simulato",  color=PLOT_TEXT)
    ax.set_ylabel("log10(C*) predetto",  color=PLOT_TEXT)
    ax.tick_params(colors=PLOT_TEXT)
    ax.grid(True, color=PLOT_GRID, alpha=0.4)
    ax.legend(framealpha=0.3, fontsize=8, labelcolor=PLOT_TEXT)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", facecolor=PLOT_DARK)
    return fig


def plot_regimeaware_coefficients(M3_result, savepath=None, dpi=180):
    """Bar chart of M3 exponents per regime with SE error bars."""
    EXPECTED_SIGNS = {
        "log10_PeH":     +1, "log10_DaRn":    -1,
        "log10_phi_g":   +1, "log10_Sw":      -1,
        "log10_Pi_src":  +1, "log10_Pi_vent": -1,
    }
    regimes  = list(M3_result["per_regime"].keys())
    feats    = [fn for fn in ["log10_PeH","log10_DaRn","log10_phi_g",
                               "log10_Sw","log10_Pi_src","log10_Pi_vent"]]
    feat_lbl = ["Pe_H","Da_Rn","phi_g","S_w","Pi_src","Pi_vent"]

    n_reg  = len(regimes)
    n_feat = len(feats)
    fig, axes = plt.subplots(1, n_reg, figsize=(3.5 * n_reg, 5),
                              facecolor=PLOT_DARK, sharey=False)
    if n_reg == 1:
        axes = [axes]

    for ax, reg in zip(axes, regimes):
        ax.set_facecolor(PLOT_PANEL)
        fit_r = M3_result["per_regime"][reg]
        coef  = fit_r["coef"]
        se    = fit_r["se"]
        fnames = fit_r["feature_names"]

        betas = []
        errs  = []
        clrs  = []
        for fn in feats:
            if fn in fnames:
                idx = list(fnames).index(fn)
                b   = float(coef[idx])
                s   = float(se[idx])
                exp = EXPECTED_SIGNS.get(fn, 0)
                c   = "#27ae60" if (np.sign(b) == exp or abs(b) < 0.05) \
                               else "#e74c3c"
            else:
                b, s, c = 0.0, 0.0, "#aaaaaa"
            betas.append(b)
            errs.append(s)
            clrs.append(c)

        x = np.arange(n_feat)
        ax.bar(x, betas, color=clrs, alpha=0.85, width=0.65)
        ax.errorbar(x, betas, yerr=errs, fmt="none", color=PLOT_TEXT,
                    capsize=4, lw=1.5)
        ax.axhline(0, color=PLOT_TEXT, lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(feat_lbl, rotation=30, ha="right",
                           color=PLOT_TEXT, fontsize=8)
        ax.tick_params(colors=PLOT_TEXT)
        ax.set_title(reg.replace("_", "\n"), color=PLOT_TEXT, fontsize=9)
        ax.set_ylabel("Esponente beta", color=PLOT_TEXT) if ax == axes[0] else None
        ax.grid(True, axis="y", color=PLOT_GRID, alpha=0.35)

    fig.suptitle("Esponenti M3 per Regime   "
                 "(verde=fisico, rosso=anomalo)",
                 color=PLOT_TEXT, fontsize=11, y=1.01)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", facecolor=PLOT_DARK)
    return fig


def run_regimeaware_scaling_pipeline(
    n_samples=1700,
    seed=42,
    nz=15,
    days=3.0,
    outdir="./regimeaware_output",
    run_M2=True,
    run_M4=True,
    verbose=True,
    skip_dataset_generation=False,
    prebuilt_df=None,
):
    """Runs the full regime-aware scaling law pipeline end to end."""
    _os.makedirs(outdir, exist_ok=True)
    t0_pipeline = _time.time()

    if skip_dataset_generation and prebuilt_df is None:
        raise ValueError("skip_dataset_generation=True richiede prebuilt_df")
    if skip_dataset_generation and prebuilt_df is not None:
        df = prebuilt_df
        if verbose:
            n_df = len(df) if not (_HAS_PANDAS and isinstance(df, _pd.DataFrame)) \
                   else len(df)
            print("[1] Dataset precostruito: n={}".format(n_df))
    else:
        if verbose:
            print("[1] Generazione dataset regime-aware "
                  "(n_samples={}, nz={}, days={})...".format(n_samples, nz, days))
        df = generate_regimeaware_dataset(
            n_samples=n_samples, seed=seed, nz=nz, days=days, verbose=verbose
        )

    if verbose:
        print("[2] Riepilogo dataset per regime...")
    summary = dataset_regime_summary(df)
    if verbose:
        print("  {:25s} {:5s} {:6s} {:12s} {:10s}".format(
            "Regime", "n", "%", "Cin_median", "PeH_median"))
        for row in summary:
            print("  {:25s} {:5d} {:5.1f}% {:12.1f} {:10.4g}".format(
                row["regime"], row["n"], row["percent"],
                row["Cin_median"] if np.isfinite(row["Cin_median"]) else -1,
                row["PeH_median"] if np.isfinite(row["PeH_median"]) else -1))

    if verbose:
        print("[3-7] Fit modelli M0–M4...")
    model_results = compare_scaling_models_M0_M4(df)
    if verbose:
        print("  {:5s} {:5s} {:5s} {:7s} {:7s} {:8s} {:8s} {:6s}".format(
            "Model", "n", "k", "R2", "RMSE", "AIC", "BIC", "F2"))
        for row in model_results["table"]:
            print("  {:5s} {:5d} {:5d} {:7.4f} {:7.4f} {:8.1f} {:8.1f} {:5.1%}".format(
                row["model"], row["n"], row["k"],
                row["R2"], row["RMSE"], row["AIC"], row["BIC"], row["F2"]))

    if verbose:
        print("[8b] F-test nested M0 vs M1/M2...")
    ft_M0_M1 = nested_f_test(model_results["M0"], model_results["M1"])
    ft_M0_M2 = nested_f_test(model_results["M0"], model_results["M2"]) \
               if run_M2 else None
    ft_M1_M2 = nested_f_test(model_results["M1"], model_results["M2"]) \
               if run_M2 else None
    if verbose:
        print("  M0 vs M1: F={:.2f}  p={:.3g}  dk={}".format(
            ft_M0_M1["F_stat"], ft_M0_M1["p_val"], ft_M0_M1["delta_k"]))
        if ft_M0_M2:
            print("  M0 vs M2: F={:.2f}  p={:.3g}".format(
                ft_M0_M2["F_stat"], ft_M0_M2["p_val"]))

    if verbose:
        print("[9] VIF e ridondanza kDP...")
    vif_dict = compute_predictor_vif(df)
    kDP_test  = test_kDP_redundancy(df)
    vif_results = {"vif": vif_dict, "kDP_test": kDP_test}
    if verbose:
        for pred, v in vif_dict.items():
            if not pred.startswith("_"):
                flag = " *** ALTO" if v > 5 else ""
                print("  VIF({}) = {:.2f}{}".format(pred, v, flag))
        print("  Conclusione kDP: {}".format(kDP_test["conclusion"]))

    if verbose:
        print("[11] Validazione LORO (M0: con M1 il dummy del regime escluso "
              "non sarebbe identificabile)...")
    loro_res = crossval_leave_one_regime_out(df, model="M0")
    if verbose:
        for reg, v in loro_res.items():
            print("  LORO {}: Q2={:.3f}  RMSE={:.3f}  n_test={}".format(
                reg, v["Q2"] if np.isfinite(v["Q2"]) else -1,
                v["RMSE"], v["n_test"]))

    if verbose:
        print("[12] Validazione 10-fold...")
    kfold_res = crossval_kfold_scaling(df, model="M0", kfold=10, seed=seed)
    if verbose:
        print("  Q2_10fold (M0) = {:.4f}".format(kfold_res["Q2"]))

    if verbose:
        print("[13] Bootstrap CI 95% (M1, n_boot=500)...")
    boot_res = bootstrap_scaling_coefficients(
        df, fit_scaling_M1_regime_dummies, n_boot=500, seed=seed
    )
    if verbose:
        print("  R2 (bootstrap): {:.4f} [{:.4f}, {:.4f}]".format(
            boot_res["R2_mean"], boot_res["R2_ci95_lo"], boot_res["R2_ci95_hi"]))

    if verbose:
        print("[14] Verifica segni fisici esponenti M3...")
    signs_res = verify_exponent_signs(model_results["M3"])
    if verbose:
        for reg, v in signs_res.items():
            print("  {}: {}/{} segni corretti{}".format(
                reg, v["n_correct"], v["n_total"],
                "" if not v["anomalies"] else " | " + v["anomalies"][0]))

    validation_results = {
        "kfold":          kfold_res,
        "loro":           loro_res,
        "bootstrap":      boot_res,
        "exponent_signs": signs_res,
        "f_tests":        {
            "M0_vs_M1": ft_M0_M1,
            "M0_vs_M2": ft_M0_M2,
            "M1_vs_M2": ft_M1_M2,
        },
    }

    if verbose:
        print("[15] Generazione report strutturato...")
    report = make_regimeaware_scaling_report(df, model_results,
                                              validation_results, vif_results)

    if verbose:
        print("[16] Export CSV in {}...".format(outdir))
    full_results = {
        "model_results":      model_results,
        "validation_results": validation_results,
        "vif_results":        vif_results,
    }
    export_paths = export_regimeaware_results_csv(df, full_results, outdir=outdir)
    if verbose:
        for k, p in export_paths.items():
            print("  {} -> {}".format(k, p))

    if verbose:
        print("[17] Plot parity M1...")
    parity_path = _os.path.join(outdir, "regimeaware_parity_M1.png")
    plot_regimeaware_parity(
        df, model_results["M1"], savepath=parity_path, dpi=180
    )
    if verbose:
        print("[18] Plot esponenti M3...")
    coef_path = _os.path.join(outdir, "regimeaware_coef_M3.png")
    plot_regimeaware_coefficients(
        model_results["M3"], savepath=coef_path, dpi=180
    )

    elapsed = _time.time() - t0_pipeline
    if verbose:
        print("=" * 60)
        print("Pipeline regime-aware completata in {:.1f}s".format(elapsed))
        print("Modello predittivo: M1 (R2={:.4f}, AIC={:.1f})".format(
            model_results["M1"]["metrics"]["R2"],
            model_results["M1"]["metrics"]["AIC"]))
        print("Modello interpretativo: M3 (R2_global={:.4f})".format(
            model_results["M3"]["metrics_global"]["R2"]))
        print("Output in: {}".format(outdir))
        print("=" * 60)

    return {
        "df":                 df,
        "dataset_summary":    summary,
        "model_results":      model_results,
        "validation_results": validation_results,
        "vif_results":        vif_results,
        "report":             report,
        "export_paths":       export_paths,
    }


# theta0 = Sw * phi_m converts saturation to volumetric water content
# regime label from transport_regime(Pe_H, Da_H)


def run_site_validation(site_name, Ra226, phi_m, k0, Sw, ACH,
                        H=2.0, dP=5.0, eps_rn=0.25, phi_f=0.0,
                        lit_type='tuff_unwelded'):
    """Runs the full ODE for one field site using measured site parameters."""
    K0_PHYS_FLOOR = 1e-16          # m^2, diffusion-limited permeability floor
    if (k0 is None) or (not np.isfinite(k0)) or (k0 < K0_PHYS_FLOOR):
        k0 = K0_PHYS_FLOOR
    override = {
        'Ra226_Bqkg': Ra226,
        'phi_m':      phi_m,
        'k0':         k0,
        'theta0':     Sw * phi_m,     # Sw initial volumetric water content
        'ACH_closed': ACH,
        'H':          H,
        'dP_init':    dP,
        'dP_bottom':  dP,
        'eps_max':    eps_rn,
        'phi_f':      phi_f,
    }
    m   = build_model(lit_type, override=override)
    sol = m.simulate_fast(days=2.0, dt_out=3600.0)
    out = m.extract_results(sol)

    t   = np.asarray(out['t'])
    C_bs = np.asarray(out['C_bs'])               # indoor (basement) Rn-222 [Bq/m3]
    mask = t > (t[-1] - 4 * 3600.0)              # average over last 4 hours
    C_ode = float(np.nanmean(C_bs[mask]))

    theta_f = np.asarray(out['theta'])[:, -1]
    Pg_f    = np.asarray(out['Pg'])[:, -1]
    T_f     = m.T_field(sol.t[-1])
    pi      = m.buckingham_pi(theta_f, Pg_f, T_f)

    reg = m.transport_regime(pi.get('Pe_H', np.nan), pi.get('Da_H', np.nan))

    return {
        'site':    site_name,
        'C_ode':   C_ode,
        'Pi_src':  float(pi.get('Pi_src', np.nan)),
        'regime':  reg.get('regime_name', 'unknown'),
        'Pe_H':    float(pi.get('Pe_H', np.nan)),
    }


def run_gold_validation(site_table, src_correction=1.0):
    """Batch field validation over gold-standard sites, returns per-site DataFrame."""
    import pandas as pd
    rows = []
    for s in site_table:
        res = run_site_validation(
            s['site'],
            s['Ra226'] * src_correction,
            s['phi_m'],
            s['k0'],
            s['Sw'],
            s['ACH'],
            s.get('H', 2.0),
            s.get('dP', 5.0),
            s.get('eps_rn', 0.25),
            s.get('phi_f', 0.0),
            s.get('lit_type', 'tuff_unwelded'),
        )
        C_obs  = s['C_meas']
        C_pred = res['C_ode']
        log_ratio = np.log10(C_pred / max(C_obs, 0.1))
        rows.append({
            'site':        s['site'],
            'domain':      s.get('domain', 'unknown'),
            'C_meas':      C_obs,
            'C_ode':       C_pred,
            'src_correction': src_correction,
            'log10_ratio': log_ratio,
            'F2_pass':     abs(log_ratio) <= np.log10(2),
            'F5_pass':     abs(log_ratio) <= np.log10(5),
            'regime':      res['regime'],
        })
    df = pd.DataFrame(rows)

    # metrics grouped by declared geological domain
    for domain in ['all', 'volcanic', 'sedimentary', 'granite_outlimit']:
        sub = df if domain == 'all' else df[df['domain'] == domain]
        if len(sub) == 0:
            continue
        F2  = sub['F2_pass'].mean() * 100
        F5  = sub['F5_pass'].mean() * 100
        MBE = sub['log10_ratio'].mean()
        print(f"  {domain:20s} N={len(sub):2d} "
              f"F2={F2:.1f}% F5={F5:.1f}% MBE={MBE:+.2f}")
    return df



def run_holdout_validation(csv_path='data/site_validation_master.csv',
                           src_calib_Pi=3.9e2, freeze_split=True,
                           verbose=True):
    """Field validation with a frozen derivation/holdout site split.

    src_calib_Pi is expressed in the DIMENSIONLESS Pi_src convention
    (rho_b*Ra*eps/(beta_p*C_ref)); the former default 5.6e7 belonged to the
    old seconds-units proxy and equals ~3.9e2 in this convention. Re-derive it
    from the derivation set after any model change."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    if 'holdout' not in df.columns:
        raise KeyError("CSV must contain a 'holdout' column (0=derivation, 1=holdout)")

    df_deriv = df[df['holdout'] == 0].copy()
    df_hold  = df[df['holdout'] == 1].copy()

    # src_correction is derived only from the derivation set to avoid leakage.
    # dimensionless field source proxy Pi_src = rho_b*Ra226*eps_rn/(beta_p*C_ref),
    # matching the model's Pi_src definition (a previous version divided by
    # lambda, leaving units of seconds in a nominally dimensionless group)
    rho_b  = 2400.0
    beta_p = 0.3   # representative gas/water partition porosity for field sites
    pi_src_field = (rho_b * df_deriv['Ra226'] * df_deriv['eps_rn']
                    / (beta_p * _C_REF))
    pi_src_field_median = float(np.median(pi_src_field))
    src_correction = src_calib_Pi / pi_src_field_median

    if verbose:
        print(f"[holdout] derivation N={len(df_deriv)}  holdout N={len(df_hold)}")
        print(f"[holdout] median Pi_src_field (derivation only) = "
              f"{pi_src_field_median:.3e}")
        print(f"[holdout] derived src_correction = x{src_correction:.2f}")

    def _subset_metrics(df_subset, label):
        if len(df_subset) == 0:
            return None
        site_table = df_subset.to_dict('records')
        res = run_gold_validation(site_table, src_correction=src_correction)
        rows = []
        for domain in ['all', 'volcanic', 'sedimentary', 'granite_outlimit']:
            sub = res if domain == 'all' else res[res['domain'] == domain]
            if len(sub) == 0:
                continue
            rows.append({
                'split':   label,
                'subset':  domain,
                'N':       int(len(sub)),
                'F2_pct':  round(sub['F2_pass'].mean() * 100, 1),
                'F5_pct':  round(sub['F5_pass'].mean() * 100, 1),
                'MBE_log': round(sub['log10_ratio'].mean(), 3),
            })
        return pd.DataFrame(rows)

    if verbose:
        print("\n[holdout] === HOLDOUT-SET metrics (independent) ===")
    metrics_hold = _subset_metrics(df_hold, 'holdout')
    if verbose:
        print("\n[holdout] === DERIVATION-SET metrics (in-sample) ===")
    metrics_deriv = _subset_metrics(df_deriv, 'derivation')

    return {
        'src_correction': src_correction,
        'df_deriv':       df_deriv,
        'df_hold':        df_hold,
        'metrics_hold':   metrics_hold,
        'metrics_deriv':  metrics_deriv,
    }


def sobol_sensitivity_scaling(beta=None, intercept=-2.399, n_base=1024,
                              calc_second_order=False, seed=42, verbose=True):
    """Sobol sensitivity of the M0 log-linear surrogate to its six Pi groups."""
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    if beta is None:
        # single source of truth: the frozen M0 dict (order matches 'names' below).
        # NOTE: regenerate _M0_BETA_FROZEN by rerunning the pipeline after any
        # model change; a previous hard-coded list here had Da_Rn and Sw swapped.
        beta = [_M0_BETA_FROZEN["Pe_H"],   _M0_BETA_FROZEN["Da_Rn"],
                _M0_BETA_FROZEN["phi_g"],  _M0_BETA_FROZEN["Sw"],
                _M0_BETA_FROZEN["Pi_src"], _M0_BETA_FROZEN["Pi_vent"]]

    problem = {
        'num_vars': 6,
        'names': ['log_Pe_H', 'log_Da_Rn', 'log_phi_g',
                  'log_Sw', 'log_Pi_src', 'log_Pi_vent'],
        'bounds': [
            [-6.0,  2.0],   # log10 Pe_H
            [ 1.0,  7.0],   # log10 Da_Rn
            [-2.0, -0.3],   # log10 phi_g
            [-0.62, -0.07], # log10 Sw
            [ 2.0,  5.0],   # log10 Pi_src
            [-1.4,  1.0],   # log10 Pi_vent
        ],
    }

    try:
        param_values = saltelli.sample(problem, n_base,
                                       calc_second_order=calc_second_order,
                                       seed=seed)
    except TypeError:
        # older SALib versions do not accept a seed argument
        np.random.seed(seed)
        param_values = saltelli.sample(problem, n_base,
                                       calc_second_order=calc_second_order)
    Y = intercept + param_values.dot(np.asarray(beta))

    try:
        Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order,
                           print_to_console=False, seed=seed)
    except TypeError:
        Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order,
                           print_to_console=False)
    Si['names'] = problem['names']

    if verbose:
        print("[sobol] First-order indices (S1):")
        for name, s1, s1c in zip(problem['names'], Si['S1'], Si['S1_conf']):
            print(f"  {name:12s} S1 = {s1:+.3f} +/- {s1c:.3f}")
        print("[sobol] Total-order indices (ST):")
        for name, st, stc in zip(problem['names'], Si['ST'], Si['ST_conf']):
            print(f"  {name:12s} ST = {st:+.3f} +/- {stc:.3f}")
    return Si


def predict_with_interval(Pi_groups, beta=None, intercept=-2.399,
                          rmse_log=0.420, c_ref=_C_REF, z=1.96):
    """Predicts indoor radon C_in [Bq/m^3] with a 95% log-space prediction interval.

    Pi_groups order: [Pe_H, Da_Rn, phi_g, Sw, Pi_src, Pi_vent], matching
    _M0_BETA_FROZEN (single source of truth for the frozen exponents)."""
    if beta is None:
        beta = [_M0_BETA_FROZEN["Pe_H"],   _M0_BETA_FROZEN["Da_Rn"],
                _M0_BETA_FROZEN["phi_g"],  _M0_BETA_FROZEN["Sw"],
                _M0_BETA_FROZEN["Pi_src"], _M0_BETA_FROZEN["Pi_vent"]]
    log_C_star = intercept + sum(b * np.log10(max(p, 1e-30))
                                 for b, p in zip(beta, Pi_groups))
    C_in    = 10 ** log_C_star * c_ref
    C_in_lo = 10 ** (log_C_star - z * rmse_log) * c_ref
    C_in_hi = 10 ** (log_C_star + z * rmse_log) * c_ref
    return C_in, C_in_lo, C_in_hi


def flux_uncertainty_analytic(Ra226, eps_rn, rho_b, phi, Sw,
                              sigma_Ra=0.0, sigma_eps=0.0, sigma_rho=0.0,
                              sigma_Sw=0.0, D0=1.1e-5, H=3.0, L_ost=0.25):
    """Canonical diffusive flux with first-order (GUM) uncertainty propagation.

    J = rho_b*A_Ra*eps*sqrt(lambda*D_a)*tanh(H/L_d), D_a = phi*D0*
    exp(-6*Sw*phi - 6*Sw^(14*phi)) (apparent, Rogers & Nielson 1991).
    Log-space variance (independent inputs):

        var(ln J) = (s_A/A)^2 + (s_eps/eps)^2 + (s_rho/rho)^2
                    + (0.5 * dlnD_a/dSw)^2 * s_Sw^2,
        dlnD_a/dSw = -6*phi - 84*phi*Sw^(14*phi - 1)

    Returns dict with J [Bq/m^2/s], the 68%/95% multiplicative factors
    exp(sigma_lnJ), exp(1.96*sigma_lnJ) and the per-component variance budget.
    No fitted parameters: pure first-order propagation of declared input
    uncertainties through the canonical formula."""
    lam  = DECAY_CONSTANTS["Rn222"]
    Sw   = float(np.clip(Sw, 1e-6, 0.9999))
    D_a  = phi * D0 * np.exp(-6.0 * Sw * phi - 6.0 * Sw ** (14.0 * phi))
    L_d  = np.sqrt(D_a / lam)
    J    = rho_b * Ra226 * eps_rn * np.sqrt(lam * D_a) * np.tanh(H / max(L_d, 1e-9))
    dlnD_dSw = -6.0 * phi - 84.0 * phi * Sw ** (14.0 * phi - 1.0)
    budget = {
        "Ra":  (sigma_Ra  / max(Ra226,  1e-30)) ** 2,
        "eps": (sigma_eps / max(eps_rn, 1e-30)) ** 2,
        "rho": (sigma_rho / max(rho_b,  1e-30)) ** 2,
        "Sw":  (0.5 * dlnD_dSw) ** 2 * sigma_Sw ** 2,
    }
    sig_lnJ = float(np.sqrt(sum(budget.values())))
    return {
        "J_Bq_m2_s":  float(J),
        "sigma_lnJ":  sig_lnJ,
        "factor_68":  float(np.exp(sig_lnJ)),
        "factor_95":  float(np.exp(1.96 * sig_lnJ)),
        "J_lo95":     float(J * np.exp(-1.96 * sig_lnJ)),
        "J_hi95":     float(J * np.exp(+1.96 * sig_lnJ)),
        "var_budget": {k: float(v) for k, v in budget.items()},
        "dlnDa_dSw":  float(dlnD_dSw),
    }


def nazaroff_1992(Ra226, eps_rn, phi_m, Sw, H, ACH, D0=1.1e-5, H_room=2.5):
    """Nazaroff (1992)-style diffusion-dominated soil-to-indoor transfer model.

    Millington-Quirk BULK D_eff = D0*phi_g^(10/3)/phi_m^2 (Rev. Geophys. 30, 137).
    Deep pore-air concentration includes the gas/water partition
    beta_p = phi_g + L*theta (L ~ 0.25). Diffusive exhalation from a column of
    depth H: J = C_soil*sqrt(lambda*D_eff)*tanh(H/L_d) [Bq/(m^2 s)], and the
    indoor balance dilutes J over the room height H_room against ventilation
    plus decay: C_in = (J/H_room)/(ACH+lambda). All terms dimensionally [Bq/m^3]."""
    theta     = phi_m * Sw
    phi_g     = phi_m * (1.0 - Sw)
    lambda_Rn = _LAM_RN
    rho_b     = 1600.0                                   # bulk density [kg/m^3]
    beta_p    = max(phi_g + 0.25 * theta, 1e-6)
    # single-convention formulation: Millington-Quirk BULK D_e converted to the
    # APPARENT coefficient D_a = D_e/beta, so that L_d = sqrt(D_a/lambda) and
    # the canonical flux J = rho*A*eps*sqrt(lambda*D_a) carries no residual
    # porosity factor (UNSCEAR 2000 Annex B paras. 97-98)
    D_bulk    = D0 * phi_g ** (10.0 / 3.0) / phi_m ** 2
    D_a       = D_bulk / beta_p
    L_d       = np.sqrt(D_a / lambda_Rn)                 # diffusion length [m]
    J         = rho_b * Ra226 * eps_rn * np.sqrt(lambda_Rn * D_a) * np.tanh(H / max(L_d, 1e-9))
    C_in      = (J / H_room) / (ACH / 3600.0 + lambda_Rn)
    return C_in


def jobbagy_2017(Ra226, eps_rn, phi_m, Sw, H, ACH, D0=1.1e-5, H_room=2.5):
    """Jobbagy et al. (2017)-style transfer model with explicit exhalation flux.

    E = rho_b*A_Ra*eps*lambda*L_d is the classic diffusive exhalation flux;
    indoor dilution acts over the ROOM height H_room, not the soil depth H."""
    theta     = phi_m * Sw
    phi_g     = phi_m * (1.0 - Sw)
    lambda_Rn = _LAM_RN
    rho_b     = 1600.0
    beta_p    = max(phi_g + 0.25 * theta, 1e-6)
    # apparent coefficient D_a = D_bulk/beta: single convention across all
    # benchmark models (UNSCEAR 2000 Annex B paras. 97-98)
    D_a       = (D0 * phi_g ** (10.0 / 3.0) / phi_m ** 2) / beta_p
    L_d       = np.sqrt(D_a / lambda_Rn)                 # radon diffusion length [m]
    E_surface = rho_b * Ra226 * eps_rn * lambda_Rn * L_d # exhalation flux [Bq/(m^2 s)]
    # indoor steady state: surface flux balanced by ventilation + decay removal
    C_in      = (E_surface / H_room) / (ACH / 3600.0 + lambda_Rn)
    return C_in


def sainz_2018(Ra226, eps_rn, phi_m, Sw, H, ACH, D0=1.1e-5, k_adv=0.15, H_room=2.5):
    """Sainz et al. (2018)-style model, adds weak advective enhancement k_adv."""
    theta     = phi_m * Sw
    phi_g     = phi_m * (1.0 - Sw)
    lambda_Rn = _LAM_RN
    rho_b     = 1600.0
    beta_p    = max(phi_g + 0.25 * theta, 1e-6)
    # apparent coefficient, same convention as nazaroff_1992/jobbagy_2017
    D_a       = (D0 * phi_g ** (10.0 / 3.0) / phi_m ** 2) / beta_p
    L_d       = np.sqrt(D_a / lambda_Rn)
    E_diff    = rho_b * Ra226 * eps_rn * lambda_Rn * L_d
    E_total   = E_diff * (1.0 + k_adv)                   # diffusive + weak advective
    C_in      = (E_total / H_room) / (ACH / 3600.0 + lambda_Rn)
    return C_in


def _metrics_logspace(C_pred, C_meas):
    """F2, F5, MBE_log and R^2(log) for paired prediction/measurement arrays."""
    C_pred = np.asarray(C_pred, dtype=float)
    C_meas = np.asarray(C_meas, dtype=float)
    ok = np.isfinite(C_pred) & np.isfinite(C_meas) & (C_pred > 0) & (C_meas > 0)
    C_pred, C_meas = C_pred[ok], C_meas[ok]
    if len(C_pred) == 0:
        return dict(N=0, F2_pct=np.nan, F5_pct=np.nan,
                    MBE_log=np.nan, R2_log=np.nan)
    log_ratio = np.log10(C_pred / C_meas)
    F2  = float(np.mean(np.abs(log_ratio) <= np.log10(2)) * 100)
    F5  = float(np.mean(np.abs(log_ratio) <= np.log10(5)) * 100)
    MBE = float(np.mean(log_ratio))
    # R^2 in log space between log10(C_pred) and log10(C_meas)
    y      = np.log10(C_meas)
    yhat   = np.log10(C_pred)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    R2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(N=int(len(C_pred)), F2_pct=round(F2, 1), F5_pct=round(F5, 1),
                MBE_log=round(MBE, 3), R2_log=round(R2, 3))


def run_literature_benchmark(csv_path='data/site_validation_master.csv',
                             m0_src_correction=7.0, verbose=True):
    """Benchmarks Nazaroff 1992, Jobbagy 2017, Sainz 2018 and M0 on field sites."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    C_meas = df['C_meas'].to_numpy(dtype=float)

    naz = np.array([nazaroff_1992(r.Ra226, r.eps_rn, r.phi_m, r.Sw, r.H, r.ACH)
                    for r in df.itertuples()])
    job = np.array([jobbagy_2017(r.Ra226, r.eps_rn, r.phi_m, r.Sw, r.H, r.ACH)
                    for r in df.itertuples()])
    sai = np.array([sainz_2018(r.Ra226, r.eps_rn, r.phi_m, r.Sw, r.H, r.ACH)
                    for r in df.itertuples()])

    # M0 field prediction: the stationary M0 surrogate (_apply_M0_simple) with
    # the declared source correction, NOT a duplicate of sainz_2018
    C_m0 = np.array([
        _apply_M0_simple({
            'Ra226': r.Ra226 * m0_src_correction, 'eps_rn': r.eps_rn,
            'phi_m': r.phi_m, 'Sw': r.Sw, 'ACH': r.ACH,
            'k0': getattr(r, 'k0', 1e-15), 'dP': getattr(r, 'dP', 5.0),
            'H': r.H,
        })
        for r in df.itertuples()
    ])

    per_site = pd.DataFrame({
        'site':           df['site'],
        'C_meas':         C_meas,
        'C_nazaroff1992': naz,
        'C_jobbagy2017':  job,
        'C_sainz2018':    sai,
        'C_M0_corr':      C_m0,
    })

    rows = []
    for label, pred in [('Nazaroff (1992)', naz),
                        ('Jobbagy et al. (2017)', job),
                        ('Sainz et al. (2018)', sai),
                        (f'M0 simple (x{m0_src_correction:g} src corr.)', C_m0)]:
        m = _metrics_logspace(pred, C_meas)
        m['model'] = label
        rows.append(m)
    summary = pd.DataFrame(rows)[['model', 'N', 'F2_pct', 'F5_pct',
                                  'MBE_log', 'R2_log']]

    if verbose:
        print("[literature] Comparative benchmark on field sites:")
        print(summary.to_string(index=False))

    return {'per_site': per_site, 'summary': summary}


def _soil_column_steady(D_eff, lambda_Rn, S, H, N_z, C_top=0.0):
    """Steady 1D radon diffusion-decay-production profile via finite differences.

    D_eff*d2C/dz2 - lambda_Rn*C + S = 0, Dirichlet at surface, zero-flux at base.
    NOTE: in this pore-space form D_eff is the APPARENT diffusivity D_a
    (L_d = sqrt(D_a/lambda)); pass D_bulk/beta, not the bulk coefficient.
    """
    dz = H / (N_z - 1)
    a  = D_eff / dz ** 2
    main = np.full(N_z, -(2.0 * a + lambda_Rn))
    low  = np.full(N_z, a)
    up   = np.full(N_z, a)
    rhs  = np.full(N_z, -S)

    # Dirichlet boundary at surface node 0
    main[0] = 1.0
    up[0]   = 0.0
    rhs[0]  = C_top
    # Neumann (zero-flux) at base node: mirrored node coupling
    main[-1] = -(2.0 * a + lambda_Rn)
    low[-1]  = 2.0 * a
    rhs[-1]  = -S

    # Thomas algorithm for tridiagonal systems
    n = N_z
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = up[0] / main[0]
    dp[0] = rhs[0] / main[0]
    for i in range(1, n):
        m = main[i] - low[i] * cp[i - 1]
        cp[i] = up[i] / m if i < n - 1 else 0.0
        dp[i] = (rhs[i] - low[i] * dp[i - 1]) / m
    C = np.zeros(n)
    C[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        C[i] = dp[i] - cp[i] * C[i + 1]
    return C


def solve_1D_column(N_z, dz, D_eff, lambda_Rn, S, params):
    """1D reference indoor radon concentration [Bq/m^3] from a single soil column."""
    H   = params['H']
    N_z = int(round(H / dz)) + 1 if dz > 0 else N_z
    C   = _soil_column_steady(D_eff, lambda_Rn, S, H, N_z, C_top=0.0)
    # diffusive entry flux from the topmost interior gradient
    dz_eff = H / (N_z - 1)
    J_entry = D_eff * (C[1] - C[0]) / dz_eff           # [Bq/(m^2 s)]
    ACH = params['ACH']
    # dilution over the ROOM height, not the soil column depth
    H_room = params.get('H_room', 2.5)
    C_indoor = (J_entry / H_room) / (ACH / 3600.0 + lambda_Rn)
    return float(C_indoor)


def _apply_M0_simple(params):
    """Stationary indoor-radon evaluation [Bq/m^3] for the monthly seasonal sweep."""
    rho_b     = 1600.0
    lambda_Rn = _LAM_RN
    phi_m = params['phi_m']; Sw = params['Sw']
    theta = phi_m * Sw
    phi_g = phi_m * (1.0 - Sw)
    D0    = 1.1e-5
    beta_p = max(phi_g + 0.25 * theta, 1e-6)
    # apparent coefficient (single convention with the benchmark models):
    # L_d = sqrt(D_a/lambda), J = rho*A*eps*lambda*L_d
    D_eff = (D0 * phi_g ** (10.0 / 3.0) / phi_m ** 2) / beta_p
    ACH   = params['ACH']
    # indoor dilution uses the room ceiling height, not the soil source depth H
    H_room = params.get('H_room', 2.5)
    L_d   = np.sqrt(D_eff / lambda_Rn)                 # radon diffusion length [m]
    E_diff = rho_b * params['Ra226'] * params['eps_rn'] * lambda_Rn * L_d
    # advective enhancement k_adv turns on when Pe_H approaches/exceeds 1;
    # Pe_H = u*H/D = k0*dP/(mu*D_eff), dimensionless (mu was previously omitted)
    k0  = params.get('k0', 1e-15)
    dP  = params.get('dP', 5.0)
    H   = params.get('H', 2.0)
    mu_air = 1.81e-5
    PeH = k0 * dP / (mu_air * max(D_eff, 1e-30))
    k_adv = 0.15 * float(np.tanh(PeH / 0.1))           # weak advective enhancement
    E_total = E_diff * (1.0 + k_adv)
    C_in = (E_total / H_room) / (ACH / 3600.0 + lambda_Rn)
    return float(C_in)


def _synthetic_ingv_monthly(annual_mean=None, params_base=None, seed=42):
    """Synthetic monthly Sw/ACH series mimicking the Campi Flegrei seasonal cycle."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    if annual_mean is None:
        # self-consistent illustrative series: annual mean taken from the
        # CURRENT stationary model at the mean seasonal state (this is a
        # coherence demo, NOT an independent validation)
        pb = dict(params_base or {'Ra226': 140.0, 'eps_rn': 0.30, 'phi_m': 0.38,
                                  'k0': 1e-15, 'H': 2.0, 'dP': 5.0})
        pb.setdefault('Sw', 0.38); pb.setdefault('ACH', 0.5)
        annual_mean = _apply_M0_simple(pb)
    dates = pd.date_range('2011-01-01', periods=84, freq='MS')
    month = dates.month.to_numpy()
    # winter (high Sw, low ACH) drives higher indoor radon
    season = np.cos(2 * np.pi * (month - 1) / 12.0)    # +1 in Jan, -1 in Jul
    Sw  = 0.38 + 0.10 * season                          # 0.28 - 0.48
    ACH = 0.50 - 0.20 * season                          # 0.30 (winter) - 0.70 (summer)
    # annual mean centred series with seasonal swing and added noise
    C_meas = annual_mean * (1.0 + 0.30 * season) \
        * (1.0 + 0.05 * rng.standard_normal(84))
    return pd.DataFrame({'date': dates, 'Sw_monthly': Sw,
                         'ACH_monthly': ACH, 'C_in': C_meas})


def temporal_validation(monthly_csv=None, params_base=None, verbose=True):
    """Seasonal/temporal validation of the stationary model against monthly data."""
    import os
    import pandas as pd

    if params_base is None:
        params_base = {'Ra226': 140.0, 'eps_rn': 0.30, 'phi_m': 0.38,
                       'k0': 1e-15, 'H': 2.0, 'dP': 5.0}

    source = 'INGV file'
    if monthly_csv and os.path.exists(monthly_csv):
        df = pd.read_csv(monthly_csv, parse_dates=['date'])
        if 'Sw_monthly' not in df.columns:
            df['Sw_monthly'] = params_base.get('Sw', 0.38)
        if 'ACH_monthly' not in df.columns:
            df['ACH_monthly'] = 0.5
    else:
        source = 'synthetic (Sabbarese et al. 2020 seasonal cycle)'
        df = _synthetic_ingv_monthly(params_base=params_base)

    C_pred = []
    for _, row in df.iterrows():
        p = params_base.copy()
        p['Sw']  = float(row['Sw_monthly'])
        p['ACH'] = float(row['ACH_monthly'])
        C_pred.append(_apply_M0_simple(p))
    df['C_pred'] = C_pred

    C_obs  = df['C_in'].to_numpy(dtype=float)
    C_hat  = np.asarray(C_pred, dtype=float)

    # month-by-month temporal metrics; standard R2 = 1 - SS_res/SS_tot (a
    # variance-ratio definition would forgive any constant bias)
    ss_res       = float(np.sum((C_obs - C_hat) ** 2))
    ss_tot       = float(np.sum((C_obs - np.mean(C_obs)) ** 2))
    R2_temporal  = 1.0 - ss_res / max(ss_tot, 1e-30)
    MBE_temporal = float(np.mean(np.log10(C_hat / C_obs)))
    F2_temporal  = float(np.mean(np.abs(np.log10(C_hat / C_obs)) < np.log10(2)))

    # annual-mean consistency is the stationary model's declared target
    C_mean_measured  = float(C_obs.mean())
    C_mean_predicted = float(C_hat.mean())
    ratio_annual     = C_mean_predicted / C_mean_measured

    # seasonal amplitude (peak/trough) of the measured series
    peak_trough = float(C_obs.max() / C_obs.min())

    is_synthetic = source.startswith('synthetic')
    out = {
        'source':           source,
        'is_synthetic':     is_synthetic,
        'n_months':         int(len(df)),
        'R2_temporal':      round(float(R2_temporal), 3),
        'MBE_temporal':     round(MBE_temporal, 3),
        'F2_temporal':      round(F2_temporal, 3),
        'C_mean_measured':  round(C_mean_measured, 1),
        'C_mean_predicted': round(C_mean_predicted, 1),
        'ratio_annual':     round(ratio_annual, 3),
        'peak_trough':      round(peak_trough, 3),
        'monthly':          df,
    }
    if verbose:
        if is_synthetic:
            print("[temporal] ATTENZIONE: serie sintetica generata dagli stessi "
                  "driver stagionali del modello — NON è una validazione "
                  "indipendente, solo un test illustrativo di coerenza.")
        print(f"[temporal] source = {source}  (N = {len(df)} months)")
        print(f"[temporal] R2(temporal)  = {out['R2_temporal']}")
        print(f"[temporal] MBE(temporal) = {out['MBE_temporal']} log10")
        print(f"[temporal] F2(temporal)  = {out['F2_temporal']:.1%}")
        print(f"[temporal] annual mean measured  = {out['C_mean_measured']} Bq/m^3")
        print(f"[temporal] annual mean predicted = {out['C_mean_predicted']} Bq/m^3")
        print(f"[temporal] ratio (pred/meas)     = {out['ratio_annual']}")
        print(f"[temporal] seasonal peak/trough  = {out['peak_trough']}")
    return out




import os
import json
import math
import hashlib
import datetime as _dt

import numpy as np


# frozen M0 exponents; order [Pe_H, Da_Rn, phi_g, Sw, Pi_src, Pi_vent].
# SINGLE SOURCE OF TRUTH for every consumer (sobol, predict_with_interval).
# REGENERATED 2026-08-01 (final): 420 LHS samples over 6 dimensions
# (dP, Ra226, ACH, theta_frac, H, P_atm_amp — the last makes the barometric
# regime reachable: 139/420 samples), predictors built on the SAME
# Rogers-Nielson apparent D_a the solver uses. All 5 regimes populated.
# R2 = 0.738, RMSE_log10 = 0.420, F2 = 58%, Q2_10fold = 0.728.
# 95% bootstrap CIs (500 resamples):
#   Pe_H  [+0.027,+0.119]  Da_Rn [-0.266,-0.067]  phi_g [-0.204,+0.100]
#   Sw    [-3.824,-2.912]  Pi_src[+0.277,+0.444]  Pi_vent[-0.328,-0.080]
# With consistent D_a features Da_Rn recovers its physically expected
# NEGATIVE sign (decay-limited suppression).
_M0_BETA_FROZEN = {
    "Pe_H":    0.071,   # beta_PeH (global)
    "Da_Rn":  -0.163,   # beta_DaRn — physical sign recovered with D_a features
    "phi_g":  -0.052,   # beta_phi_g (CI includes 0: absorbed by DaRn/Sw via D_a)
    "Sw":     -3.348,   # beta_Sw
    "Pi_src":  0.353,   # beta_Pi_src
    "Pi_vent": -0.201,  # beta_Pi_vent
}
# frozen M0 intercept and residual scatter (same fit as above)
_M0_INTERCEPT_FROZEN = -2.399
_M0_RMSE_LOG_FROZEN  = 0.420

# WITHIN-SITE Damkohler exponent for the seismic amplitude module: k changes at
# fixed H, so the relevant exponent comes from the regime-resolved (M3) fit,
# not from the H-dominated global M0 value. This value predates the model
# corrections: REGENERATE via fit_scaling_M3_by_regime before any blind
# Campi Flegrei (C1) prediction.
_SEIS_BETA_DARN = -0.642

# regimes physically expected to respond to bradyseismic/barometric forcing
_RESPONSIVE_REGIMES = ("advective", "fracture_bypass", "barometric_pumping",
                       "mixed_transition")
# decay-limited regime should show no response (R ~ 1)
_INERT_REGIMES = ("decay_limited",)

# column-name fragments forbidden in blind-prediction inputs (anti-leakage)
_FORBIDDEN_INPUT_KEYS = (
    "c_meas", "c_radon", "rn_meas", "radon_meas", "c_in_meas", "measured",
    "c_unrest", "c_quiet", "c_obs", "rn222_meas", "activity_meas",
)


def _assert_no_measurement_leak(records):
    """Raises if any input record contains a radon-measurement-like column."""
    for rec in records:
        for key in rec.keys():
            kl = str(key).strip().lower()
            for bad in _FORBIDDEN_INPUT_KEYS:
                if bad in kl:
                    raise ValueError(
                        "BLIND-VALIDATION VIOLATION: input record contains a "
                        f"measurement-like column '{key}'. Tranche A must hold "
                        "only input parameters (site + deformation + barometric "
                        "forcing). Remove all measured-radon columns and retry."
                    )
    return True


def _seismic_permeability_factor(M_w, k_seis_max=10.0,
                                 A_seis=0.30, gamma_seis=0.90):
    """Peak co-seismic permeability enhancement: k_seis = 1 + A*exp(gamma*M_w)."""
    k_seis = 1.0 + A_seis * math.exp(gamma_seis * float(M_w))
    return float(np.clip(k_seis, 1.0, k_seis_max))


# post-unrest permeability recovery time constants tau [days]
# Elkhoury et al. 2006, Nature 441:1135; Manga et al. 2012, RG 50:RG2004
_TAU_SEIS_RECOVERY_DAYS = {
    "Campi Flegrei":   18.0,   # caldera unrest: slow re-sealing of soft tuff/fractures
    "Etna":             9.0,   # basaltic, more brittle, faster healing
    "Catalonia":        7.0,   # granitic microfracture closure
    "Iceland":          6.0,   # basaltic rift, rapid healing
    "default":         12.0,
}


def _tau_seis_recovery_seconds(site=None, tau_days=None):
    """Resolves the post-unrest permeability recovery time constant tau [s]."""
    if tau_days is not None:
        return float(tau_days) * 86400.0
    d = _TAU_SEIS_RECOVERY_DAYS.get(site, _TAU_SEIS_RECOVERY_DAYS["default"])
    return float(d) * 86400.0



def seismic_recovery_factor(t, t0, k_seis_peak, tau, k_floor=1.0):
    """Post-seismic permeability enhancement k(t), decaying to k_floor.

    k(t) = 1 + (k_seis_peak - 1) * exp(-(t-t0)/tau) for t >= t0 (Elkhoury 2006; Manga 2012)."""
    t   = np.asarray(t, dtype=float)
    dt  = t - float(t0)
    amp = max(float(k_seis_peak) - 1.0, 0.0)
    k   = k_floor + amp * np.exp(-np.clip(dt, 0.0, None) / max(float(tau), 1e-9))
    k   = np.where(dt < 0.0, k_floor, k)   # no enhancement before the unrest peak
    return float(k) if np.ndim(k) == 0 else k


def _time_to_recovery_peak_days(tau, u_eff_days):
    """Extra peak delay from the two-time-constant transport/recovery convolution.

    t_peak = (tau*t_tr)/(tau-t_tr) * ln(tau/t_tr); bounded by 0 < t_peak < max(tau, t_tr)."""
    tau_d = max(float(tau) / 86400.0, 1e-6)
    t_tr  = max(float(u_eff_days), 1e-6)
    if abs(tau_d - t_tr) < 1e-6:
        return float(t_tr)
    t_peak = (tau_d * t_tr) / (tau_d - t_tr) * math.log(tau_d / t_tr)
    return float(max(t_peak, 0.0))




# exponent standard errors used for the band on R; site systematics
# (Ra-226, porosity, geometry) cancel in the C_unrest/C_quiet ratio
_SE_EXP_EFF = {
    "advective":          0.060,   # sqrt(0.045^2 + (0.50*0.05)^2), rounded conservatively
    "fracture_bypass":    0.065,   # fractured channel: slightly higher SE
    "barometric_pumping": 0.055,
    "mixed_transition":   0.045,
    "decay_limited":      0.0,     # exponent physically pinned to 0
}
# relative uncertainty on k_seis at the unrest peak, in log10
_SIGMA_LOG10_KSEIS_DEFAULT = 0.12     # corresponds to delta_k/k ~ 0.28 (1 sigma)


def _amplitude_band_R(R_pred, k_seis, regime, z=1.96,
                      sigma_log10_kseis=_SIGMA_LOG10_KSEIS_DEFAULT):
    """95% confidence band on R via first-order log10 error propagation.

    var(log10 R) = (log10 k_seis)^2 * SE_E^2 + E^2 * sigma_log10(k_seis)^2."""
    if regime in _INERT_REGIMES:
        # inert sites: R expected = 1, band reflects measurement noise only (~+/-10%)
        sig = 0.045   # log10(1.11) ~ 0.045, i.e. ~11% at 1 sigma
        R_lo = 10 ** (0.0 - z * sig)
        R_hi = 10 ** (0.0 + z * sig)
        return float(R_lo), float(R_hi), float(sig)
    E = _effective_exponent(regime)
    SE_E = _SE_EXP_EFF.get(regime, 0.060)
    log_k = math.log10(max(float(k_seis), 1e-30))
    var_logR = (log_k ** 2) * (SE_E ** 2) + (E ** 2) * (sigma_log10_kseis ** 2)
    sig = math.sqrt(max(var_logR, 0.0))
    log_R = math.log10(max(float(R_pred), 1e-30))
    R_lo = 10 ** (log_R - z * sig)
    R_hi = 10 ** (log_R + z * sig)
    return float(R_lo), float(R_hi), float(sig)


def _effective_exponent(regime, beta=_M0_BETA_FROZEN):
    """Effective exponent E = beta_PeH_eff - f_adv*beta_DaRn_seis for the regime.

    Uses the WITHIN-SITE Damkohler exponent _SEIS_BETA_DARN (regime-resolved M3
    fit), not the H-dominated global M0 value: in the C_unrest/C_quiet ratio only
    k varies while the site geometry is fixed."""
    beta_PeH_eff = {
        "advective":          0.140,
        "fracture_bypass":    0.140,
        "barometric_pumping": 0.100,
        "mixed_transition":   0.040,
        "decay_limited":      0.0,
    }.get(regime, beta["Pe_H"])
    f_adv = {
        "advective":          0.50,
        "fracture_bypass":    0.50,
        "barometric_pumping": 0.30,
        "mixed_transition":   0.20,
        "decay_limited":      0.0,
    }.get(regime, 0.0)
    return beta_PeH_eff - f_adv * _SEIS_BETA_DARN


def _amplitude_factor_R(k_seis, regime, beta=_M0_BETA_FROZEN):
    """Predicted radon amplification R = C_unrest/C_quiet from the frozen scaling law.

    R = k_seis^E with E = beta_PeH_eff - f_adv*beta_DaRn (Pe_H scales with k,
    Da_Rn with 1/k). The exponent tables live in _effective_exponent — single
    source of truth shared with the confidence-band computation."""
    if regime in _INERT_REGIMES:
        return 1.0   # physically pinned: no advective lift possible
    exp_eff = _effective_exponent(regime, beta=beta)
    R = float(k_seis) ** exp_eff
    return max(R, 1.0)


def _time_lag_days(k0, phi_g, H, P_atm=1.0132e5, mu=1.82e-5,
                   T_atm=86400.0, rho_g=1.2):
    """Deformation-to-radon peak lag [days] from barometric pumping skin depth.

    z_p = sqrt(2 k0 P_atm / (omega mu phi_g)) (Elberling et al. 1998); Delta-t = H/u_eff."""
    omega = 2.0 * math.pi / T_atm
    phi_g = max(float(phi_g), 1e-3)
    z_p = math.sqrt(2.0 * float(k0) * P_atm / max(omega * mu * phi_g, 1e-30))
    z_p = max(z_p, 1e-3)
    # characteristic barometric Darcy velocity through the skin layer
    dP_char = rho_g * 9.81 * z_p                      # hydrostatic-scale forcing
    u_darcy = (float(k0) / mu) * (dP_char / z_p)      # superficial velocity [m/s]
    # gas travel time uses the PORE (interstitial) velocity u_darcy/phi_g
    u_eff = max(u_darcy / phi_g, 1e-12)
    dt_sec = float(H) / u_eff
    return dt_sec / 86400.0



def decompose_seismic_vs_meteo(t, radon, stack_driver,
                               extra_drivers=None,
                               unrest_t0=None, tau_seis=None,
                               log_radon=True):
    """Split a radon time series into meteo/seasonal (OLS-fit) and seismic parts."""
    t      = np.asarray(t, dtype=float)
    y_raw  = np.asarray(radon, dtype=float)
    N      = y_raw.size
    y      = np.log(np.clip(y_raw, 1e-12, None)) if log_radon else y_raw.copy()

    cols   = {"const": np.ones(N)}
    sd     = np.asarray(stack_driver, dtype=float)
    cols["stack"] = (sd - sd.mean()) / (sd.std() + 1e-12)
    if extra_drivers:
        for name, v in extra_drivers.items():
            v = np.asarray(v, dtype=float)
            cols[name] = (v - v.mean()) / (v.std() + 1e-12)
    has_seis = (unrest_t0 is not None and tau_seis is not None)
    if has_seis:
        env = np.where(t >= float(unrest_t0),
                       np.exp(-np.clip(t - float(unrest_t0), 0.0, None)
                              / max(float(tau_seis), 1e-9)),
                       0.0)
        cols["seismic"] = (env - env.mean()) / (env.std() + 1e-12)

    names = list(cols.keys())
    X     = np.column_stack([cols[n] for n in names])

    def _ols(Xm, yv):
        beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
        yhat = Xm @ beta
        ss_res = float(np.sum((yv - yhat) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2)) + 1e-30
        return beta, yhat, 1.0 - ss_res / ss_tot

    beta_full, yhat_full, R2_full = _ols(X, y)
    coef = {n: float(b) for n, b in zip(names, beta_full)}

    # meteo-only model (drop the seismic column)
    meteo_names = [n for n in names if n != "seismic"]
    Xm = np.column_stack([cols[n] for n in meteo_names])
    _, yhat_meteo, R2_meteo = _ols(Xm, y)

    # partial R2 of the seismic term: extra variance explained beyond meteo
    if has_seis:
        seismic_partial = max(R2_full - R2_meteo, 0.0)
        denom = max(1.0 - R2_meteo, 1e-9)
        seismic_fraction = seismic_partial / denom   # fraction of meteo-residual
    else:
        seismic_partial = 0.0
        seismic_fraction = 0.0

    residual = y - yhat_meteo   # what the stack/seasonal model cannot explain
    return {
        "regressors":          names,
        "coef":                coef,
        "R2_full":             float(R2_full),
        "R2_meteo_only":       float(R2_meteo),
        "R2_seismic_partial":  float(seismic_partial),
        "seismic_fraction":    float(seismic_fraction),
        "log_space":           bool(log_radon),
        "fitted_full":         yhat_full,
        "fitted_meteo":        yhat_meteo,
        "residual_after_meteo": residual,
        "note": (
            "seismic_fraction = partial R2 of the recovery envelope after "
            "removing the stack-effect/seasonal regressors. High value => "
            "the radon transient cannot be explained by weather/season alone."
        ),
    }



def predict_bradyseismic_blind(tranche_A,
                               beta=_M0_BETA_FROZEN,
                               R_inert_tol=1.10,
                               outfile="predictions_frozen.json"):
    """Locked a-priori prediction per site, written to outfile; no measured radon input allowed."""
    # accepts either a list of dicts (CLI) or a CSV path (direct use)
    if isinstance(tranche_A, str):
        import pandas as pd
        records = pd.read_csv(tranche_A).to_dict("records")
    else:
        records = list(tranche_A)
    _assert_no_measurement_leak(records)

    rmse_log = _M0_RMSE_LOG_FROZEN
    z = 1.96
    sites_out = []
    for rec in records:
        site = rec.get("site", "UNKNOWN")
        # regime: use precomputed if present, else infer from dimensionless groups
        regime = rec.get("regime")
        if regime is None:
            regime = infer_physical_regime(rec)   # Noqa: F821 (defined in main module)
        expected = regime in _RESPONSIVE_REGIMES

        # amplitude factor R from seismic permeability enhancement
        M_w = float(rec.get("M_w_unrest", 0.0))
        k_seis = _seismic_permeability_factor(M_w)
        R_pred = _amplitude_factor_R(k_seis, regime, beta=beta)

        # 95% band on R from physical error propagation, not the global M0 RMSE
        R_lo, R_hi, sigma_logR = _amplitude_band_R(R_pred, k_seis, regime, z=z)

        # time lag: barometric-skin transport delay convolved with the seismic
        # permeability recovery transient k(t)=1+(k_seis-1)exp(-(t-t0)/tau)
        phi_g = max(float(rec.get("phi_m", 0.3)) * (1.0 - float(rec.get("Sw", 0.4))), 1e-3)
        # tau resolution: explicit per-site value > site table > default
        tau_seis = _tau_seis_recovery_seconds(
            site=rec.get("seismic_site", site),
            tau_days=rec.get("tau_seis_days"),
        )
        if expected:
            lag_transport = _time_lag_days(rec.get("k0", 1e-13), phi_g,
                                           rec.get("H", 3.0))
            lag_recovery  = _time_to_recovery_peak_days(tau_seis, lag_transport)
            # observable lag is governed by whichever of transport or recovery is slower
            lag = round(lag_recovery, 3)
            lag_transport = round(lag_transport, 3)
        else:
            lag = None            # inert site: no propagating perturbation signal
            lag_transport = None

        sites_out.append({
            "site":            site,
            "lit_type":        rec.get("lit_type"),
            "regime":          regime,
            "expected_effect": bool(expected),
            "k_seis_peak":     round(k_seis, 4),
            "R_pred":          round(R_pred, 4),
            "R_lo95":          round(R_lo, 4),
            "R_hi95":          round(R_hi, 4),
            "sigma_log10_R":   round(sigma_logR, 4),
            "lag_days_pred":   lag,
            # lag decomposition + recovery time constant
            "lag_transport_days": lag_transport,
            "tau_recovery_days":  round(tau_seis / 86400.0, 3),
            # if a site is inert, success means measured R stays below this tolerance
            "R_inert_tolerance": R_inert_tol,
            "R_meas":          None,
            "lag_days_meas":   None,
            "amplitude_pass":  None,
            "lag_pass":        None,
            "regime_pass":     None,
        })

    payload = {
        "candidate":        "C1_bradyseismic_radon_forcing_Campi_Flegrei",
        "model_version":    "volcanic_radon_model_v5_regimeaware",
        "frozen_beta_M0":   beta,
        "rmse_log":         rmse_log,  # global M0 RMSE (corrected-model refit), reference only
        "band_method":      "physical_ratio_propagation",
        "band_method_note": (
            "La banda 95% su R = C_unrest/C_quiet e' derivata per propagazione "
            "del primo ordine in log10 da SE(esponente efficace) e "
            "sigma_log10(k_seis); NON dall'RMSE globale di M0, perche' gli errori "
            "sistematici di sito (Ra-226, porosita', geometria) si cancellano nel "
            "rapporto sullo stesso sito."
        ),
        "sigma_log10_kseis": _SIGMA_LOG10_KSEIS_DEFAULT,
        "SE_exp_eff":        _SE_EXP_EFF,
        "responsive_regimes": list(_RESPONSIVE_REGIMES),
        "inert_regimes":     list(_INERT_REGIMES),
        "n_sites":          len(sites_out),
        "created_utc":      _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "sites":            sites_out,
        # success criteria fixed a-priori, not to be changed after seeing data
        "success_criteria": {
            "amplitude":  "measured R within [R_lo95, R_hi95] for >= 70% of "
                          "responsive sites",
            "lag":        "measured lag within +/- 1 day of lag_days_pred for "
                          ">= 60% of responsive sites",
            "specificity": "regime classification correct (responsive shows "
                           "R>tol, inert shows R<=tol) for >= 80% of all sites",
            "STRONG_discovery": "amplitude AND lag AND specificity all pass",
            "PARTIAL":          "specificity passes AND (amplitude OR lag) passes",
            "NOT_confirmed":    "specificity fails",
        },
    }

    with open(outfile, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Blind prediction written -> {outfile} "
          f"({len(sites_out)} sites, {sum(s['expected_effect'] for s in sites_out)} "
          f"responsive).")
    return payload


def freeze_prediction(prediction_file="predictions_frozen.json",
                      code_file=__file__,
                      manifest_out="C1_FREEZE_MANIFEST.json"):
    """Write a pre-registration manifest with SHA-256 of prediction and code, plus UTC timestamp."""
    def _sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "candidate":            "C1_bradyseismic_radon_forcing_Campi_Flegrei",
        "frozen_utc":           _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "prediction_file":      os.path.basename(prediction_file),
        "prediction_sha256":    _sha256(prediction_file),
        "code_file":            os.path.basename(code_file),
        "code_sha256":          _sha256(code_file),
        "git_tag_expected":     "v5.1-frozen-blind",
        "integrity_note": "This manifest must be deposited with a trusted "
                          "timestamp (OSF/preprint) before Tranche B (measured "
                          "radon) is received. The code must not be modified "
                          "after this point. Any post-hoc change invalidates "
                          "the blind validation.",
    }
    with open(manifest_out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"FREEZE manifest written -> {manifest_out}")
    print(f"      prediction SHA-256: {manifest['prediction_sha256']}")
    print(f"      code       SHA-256: {manifest['code_sha256']}")
    return manifest


def verify_against_measured(prediction_file="predictions_frozen.json",
                            measured_csv="measured_tranche_B.csv",
                            report_out="C1_VERIFICATION_REPORT.json"):
    """Score frozen predictions against measured Tranche B using pre-registered criteria."""
    import pandas as pd

    with open(prediction_file) as fh:
        pred = json.load(fh)
    meas = pd.read_csv(measured_csv).set_index("site").to_dict("index")

    n_resp = n_resp_amp = n_resp_lag = 0
    n_all = n_regime_ok = 0
    detail = []
    for s in pred["sites"]:
        site = s["site"]
        m = meas.get(site)
        n_all += 1
        if m is None:
            detail.append({"site": site, "status": "NO_MEASUREMENT"})
            continue

        R_meas = float(m.get("R_meas", np.nan))
        lag_meas = float(m.get("lag_days_meas", np.nan))
        s["R_meas"] = R_meas
        s["lag_days_meas"] = lag_meas

        # specificity (regime) check
        if s["expected_effect"]:
            regime_ok = R_meas > s["R_inert_tolerance"]
        else:
            regime_ok = R_meas <= s["R_inert_tolerance"]
        s["regime_pass"] = bool(regime_ok)
        n_regime_ok += int(regime_ok)

        # amplitude & lag checks only meaningful for responsive sites
        if s["expected_effect"]:
            n_resp += 1
            amp_ok = (s["R_lo95"] <= R_meas <= s["R_hi95"])
            lag_ok = (abs(lag_meas - s["lag_days_pred"]) <= 1.0)
            s["amplitude_pass"] = bool(amp_ok)
            s["lag_pass"] = bool(lag_ok)
            n_resp_amp += int(amp_ok)
            n_resp_lag += int(lag_ok)

        detail.append(s)

    amp_frac = (n_resp_amp / n_resp) if n_resp else 0.0
    lag_frac = (n_resp_lag / n_resp) if n_resp else 0.0
    reg_frac = (n_regime_ok / n_all) if n_all else 0.0

    amplitude_pass   = amp_frac >= 0.70
    lag_pass         = lag_frac >= 0.60
    specificity_pass = reg_frac >= 0.80

    if specificity_pass and amplitude_pass and lag_pass:
        verdict = "STRONG_DISCOVERY_CONFIRMED"
    elif specificity_pass and (amplitude_pass or lag_pass):
        verdict = "PARTIAL_CONFIRMATION"
    else:
        verdict = "NOT_CONFIRMED"

    report = {
        "candidate":          pred["candidate"],
        "verified_utc":       _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_sites_total":      n_all,
        "n_responsive":       n_resp,
        "amplitude_fraction": round(amp_frac, 3),
        "lag_fraction":       round(lag_frac, 3),
        "specificity_fraction": round(reg_frac, 3),
        "amplitude_pass":     amplitude_pass,
        "lag_pass":           lag_pass,
        "specificity_pass":   specificity_pass,
        "VERDICT":            verdict,
        "per_site":           detail,
    }
    with open(report_out, "w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 70)
    print(f"  CANDIDATE 1 — BLIND VERIFICATION RESULT")
    print("=" * 70)
    print(f"  Specificity (regime) : {reg_frac*100:5.1f}%  -> "
          f"{'PASS' if specificity_pass else 'FAIL'}")
    print(f"  Amplitude (R in band): {amp_frac*100:5.1f}%  -> "
          f"{'PASS' if amplitude_pass else 'FAIL'}")
    print(f"  Lag (+/- 1 day)      : {lag_frac*100:5.1f}%  -> "
          f"{'PASS' if lag_pass else 'FAIL'}")
    print("-" * 70)
    print(f"  VERDICT: {verdict}")
    print("=" * 70)
    return report


def _cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description='volcanicradonmodel v5 - regime-aware ODE radon transport')
    parser.add_argument('--mode',
                        choices=['benchmark', 'validate', 'pipeline',
                                 'holdout', 'sobol',
                                 'literature', 'temporal',
                                 'c1_predict', 'c1_freeze', 'c1_verify'],
                        default='benchmark',
                        help='Run mode: benchmark (lithology sweep), '
                             'validate (gold-standard field sites), '
                             'pipeline (full scaling law), '
                             'holdout (frozen derivation/holdout split), '
                             'sobol (global Sobol sensitivity), '
                             'literature (Nazaroff/Jobbagy/Sainz vs M0 benchmark), '
                             'temporal (seasonal/annual-mean validation), '
                             'c1_predict (Candidate-1 blind prediction from '
                             'Tranche-A inputs), c1_freeze (hash+timestamp '
                             'pre-registration), c1_verify (compare frozen '
                             'prediction vs measured Tranche-B))')
    parser.add_argument('--tranche_a', default='data/c1_tranche_A_inputs.csv',
                        help='Input-only CSV for Candidate-1 blind prediction')
    parser.add_argument('--measured', default='data/c1_tranche_B_measured.csv',
                        help='Measured CSV (Tranche B) for Candidate-1 verify')
    parser.add_argument('--frozen', default='predictions_frozen.json',
                        help='Frozen prediction JSON file (Candidate 1)')
    parser.add_argument('--outdir', default='./output',
                        help='Output directory for CSV and figures')
    parser.add_argument('--nsamples', type=int, default=40,
                        help='Number of ODE samples (pipeline mode)')
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == 'benchmark':
        results = run_lithology_benchmark(nz=15, days=1.5)
        plot_lithology_benchmark(results, save_path=f'{args.outdir}/benchmark.png')
    elif args.mode == 'validate':
        # load gold-standard site table from CSV
        import pandas as pd
        df = pd.read_csv('data/site_validation_master.csv')
        site_table = df.to_dict('records')
        for corr in [1.0, 7.0]:
            print(f"\n[validate] src_correction = {corr}")
            out = run_gold_validation(site_table, src_correction=corr)
            out.to_csv(f'{args.outdir}/validation_corr{int(corr)}.csv', index=False)
    elif args.mode == 'pipeline':
        run_regimeaware_scaling_pipeline(
            n_samples=args.nsamples, outdir=args.outdir)
    elif args.mode == 'holdout':
        # frozen derivation/holdout split: derive x7 on derivation, evaluate on holdout
        res = run_holdout_validation(csv_path='data/site_validation_master.csv')
        if res['metrics_hold'] is not None:
            res['metrics_hold'].to_csv(
                f'{args.outdir}/holdout_metrics.csv', index=False)
        if res['metrics_deriv'] is not None:
            res['metrics_deriv'].to_csv(
                f'{args.outdir}/derivation_metrics.csv', index=False)
    elif args.mode == 'sobol':
        # global Sobol first-order sensitivity of the M0 scaling law
        Si = sobol_sensitivity_scaling(n_base=1024)
        import pandas as pd
        pd.DataFrame({
            'name': Si['names'],
            'S1':   Si['S1'], 'S1_conf': Si['S1_conf'],
            'ST':   Si['ST'], 'ST_conf': Si['ST_conf'],
        }).to_csv(f'{args.outdir}/sobol_indices.csv', index=False)
    elif args.mode == 'literature':
        # benchmark against Jobbagy 2017 and Sainz 2018
        res = run_literature_benchmark(
            csv_path='data/site_validation_master.csv', m0_src_correction=7.0)
        res['summary'].to_csv(
            f'{args.outdir}/literature_benchmark_summary.csv', index=False)
        res['per_site'].to_csv(
            f'{args.outdir}/literature_benchmark_per_site.csv', index=False)
    elif args.mode == 'temporal':
        # seasonal/annual-mean validation vs INGV monthly series (2011-2017)
        out = temporal_validation(
            monthly_csv='data/campi_flegrei_monthly_2011_2017.csv')
        out['monthly'].to_csv(
            f'{args.outdir}/temporal_monthly.csv', index=False)
    elif args.mode == 'c1_predict':
        # candidate 1: blind prediction from input-only Tranche A
        import pandas as pd
        tranche_A = pd.read_csv(args.tranche_a).to_dict('records')
        predict_bradyseismic_blind(
            tranche_A, outfile=f'{args.outdir}/{os.path.basename(args.frozen)}')
    elif args.mode == 'c1_freeze':
        # candidate 1: produce the immutable pre-registration manifest
        freeze_prediction(
            prediction_file=f'{args.outdir}/{os.path.basename(args.frozen)}',
            code_file=__file__,
            manifest_out=f'{args.outdir}/C1_FREEZE_MANIFEST.json')
    elif args.mode == 'c1_verify':
        # candidate 1: run-once verification against measured Tranche B
        verify_against_measured(
            prediction_file=f'{args.outdir}/{os.path.basename(args.frozen)}',
            measured_csv=args.measured,
            report_out=f'{args.outdir}/C1_VERIFICATION_REPORT.json')


if __name__ == '__main__':
    _cli_main()


def soil_surface_flux(model, sol, verbose=True):
    """Net radon exhalation flux at the soil surface z=0 [Bq/m^2/s].

    PRIMARY estimator: the DIRECT discrete surface fluxes exactly as the solver
    applies them at the top boundary face (Dirichlet C=0 at z=0):
        J_diff = D_e[0]*(C[0]-0)/(dz/2)     (bulk effective diffusivity)
        J_adv  = -q_face(0)*C_upwind        (upwind, positive upward)
    summed over the matrix and fracture continua. This is mass-consistent with
    the numerical scheme and does not rely on a residual budget.

    The column budget terms (prod, decay, storage, bottom influx; all in
    partition-porosity form beta = phi_g + L*theta) are returned as DIAGNOSTICS:
    their residual differs from J_surface by the matrix-fracture exchange,
    adsorption exchange and building-entry sink, and by output-sampling error
    in the storage slope."""
    nz  = model.nz
    dz  = model.dz
    lam = model.lam_rn
    y   = sol.y[:, -1]

    theta   = np.clip(y[0:nz],      1e-8, 0.95 * model.p["phi_m"])
    Pg      = np.clip(y[nz:2*nz],   5e4,  2e5)
    Cm      = np.maximum(y[2*nz:3*nz], 0.0)
    Cf      = np.maximum(y[3*nz:4*nz], 0.0)
    Ar      = np.clip(y[6*nz:7*nz], 0.0, 1.0)
    eps_mem = np.clip(y[9*nz:10*nz], 0.01, 0.80)

    t = sol.t[-1]
    T = model.T_field(t)
    S_222 = model.source_rn222(theta, T, Ar, eps_mem)   # [Bq/(m^3_gas s)]

    phi_f   = model.p["phi_f"]
    gam_f   = model.p.get("gamma_sw_f", 0.2)

    def _betas(theta_arr, T_arr):
        L_T     = model._henry_rn(T_arr)
        beta_m_ = np.maximum(model.p["phi_m"] - theta_arr, 1e-12) + L_T * theta_arr
        th_f    = np.clip(theta_arr * gam_f, 1e-12, 0.9999 * phi_f)
        beta_f_ = np.maximum(phi_f - th_f, 1e-12) + L_T * th_f
        return beta_m_, beta_f_

    beta_m, beta_f = _betas(theta, T)

    prod  = float(np.sum(S_222 * beta_m) * dz)   # [Bq m^-2 s^-1], bulk source
    # decay of the COMBINED matrix + fracture inventory
    decay = float(np.sum(lam * (beta_m * Cm + beta_f * Cf)) * dz)

    # storage d/dt int(beta_m*Cm + beta_f*Cf) dz, least-squares slope over the
    # last few output frames (a two-frame difference is noise-dominated for
    # slowly diffusing columns)
    storage = 0.0
    n_fr = min(5, sol.y.shape[1])
    if n_fr >= 2:
        invs, ts = [], []
        for j in range(-n_fr, 0):
            th_j = np.clip(sol.y[0:nz, j], 1e-8, 0.95 * model.p["phi_m"])
            Cm_j = np.maximum(sol.y[2*nz:3*nz, j], 0.0)
            Cf_j = np.maximum(sol.y[3*nz:4*nz, j], 0.0)
            bm_j, bf_j = _betas(th_j, model.T_field(float(sol.t[j])))
            invs.append(float(np.sum(bm_j * Cm_j + bf_j * Cf_j) * dz))
            ts.append(float(sol.t[j]))
        ts = np.asarray(ts); invs = np.asarray(invs)
        if ts[-1] > ts[0]:
            storage = float(np.polyfit(ts - ts[0], invs, 1)[0])

    # bottom boundary Darcy velocities (diagnostics)
    mu    = model.p.get("mu", 1.82e-5)
    k_m   = model._k_gas(theta, model.z)
    k_f   = model.p.get("k_frac", 1e-11)
    dPdz  = np.gradient(Pg, dz)
    u_bot_m = -(k_m[-1] / mu) * dPdz[-1]        # matrix Darcy velocity [m/s]
    u_bot_f = -(k_f / mu) * dPdz[-1]            # fracture Darcy velocity [m/s]
    Cm_bot_eq = float(S_222[-1] / max(lam, 1e-30))
    C_up_m = Cm_bot_eq if u_bot_m < 0 else float(Cm[-1])
    C_up_f = float(Cf[-1])   # fracture inlet is zero-gradient (no deep reservoir)
    J_bottom = -(u_bot_m * C_up_m + u_bot_f * C_up_f)   # influx (>0) when u < 0

    # PRIMARY: direct discrete surface fluxes (positive upward), both continua
    theta_f_arr = np.clip(theta * gam_f, 1e-12, 0.9999 * phi_f)
    Dm0 = float(model._D_eff(theta, "m")[0])
    Df0 = float(model._D_eff(theta_f_arr, "f")[0])
    J_diff_m = Dm0 * float(Cm[0]) / (0.5 * dz)
    J_diff_f = Df0 * float(Cf[0]) / (0.5 * dz)
    u_top_m  = -(k_m[0] / mu) * dPdz[0]
    u_top_f  = -(k_f   / mu) * dPdz[0]
    # upwind at the surface face: outflow (u<0, upward) carries C[0];
    # inflow (u>0, downward) carries atmospheric C ~ 0
    J_adv_m  = -u_top_m * float(Cm[0]) if u_top_m < 0 else 0.0
    J_adv_f  = -u_top_f * float(Cf[0]) if u_top_f < 0 else 0.0

    J_surface = J_diff_m + J_diff_f + J_adv_m + J_adv_f

    residual = prod - decay - storage + J_bottom

    if verbose:
        print(f"[soil_surface_flux]  J_surface = {J_surface:.4e} Bq/m^2/s "
              f"(diff m/f = {J_diff_m:.3e}/{J_diff_f:.3e}, "
              f"adv m/f = {J_adv_m:.3e}/{J_adv_f:.3e})")
        print(f"    bilancio diagnostico: prod = {prod:.4e}, decad = {decay:.4e}, "
              f"storage = {storage:.4e}, influsso fondo = {J_bottom:.4e}, "
              f"residuo = {residual:.4e}")

    return {
        "J_surface": float(J_surface),
        "J_diff_m":  float(J_diff_m),
        "J_diff_f":  float(J_diff_f),
        "J_adv_m":   float(J_adv_m),
        "J_adv_f":   float(J_adv_f),
        "prod":      prod,
        "decay":     decay,
        "storage":   float(storage),
        "J_bottom":  float(J_bottom),
        "budget_residual": float(residual),
    }


def soil_concentration_means(model, sol, z_ref=0.8):
    """Column-mean and reference-depth radon gas concentration [Bq/m^3]."""
    # z_ref=0.8 m is a typical soil-gas sampling depth
    nz  = model.nz
    Cm  = np.maximum(sol.y[2*nz:3*nz, -1], 0.0)
    z   = model.z

    C_mean_col = float(np.mean(Cm))
    idx        = int(np.argmin(np.abs(z - z_ref)))
    C_at_zref  = float(Cm[idx])

    return {
        "C_mean_col":     C_mean_col,
        "C_at_zref":      C_at_zref,
        "z_ref":          float(z_ref),
        "C_mean_kBq":     C_mean_col / 1000.0,
        "C_at_zref_kBq":  C_at_zref / 1000.0,
    }


def soil_outputs(model, sol, z_ref=0.8, verbose=True):
    """Surface radon flux [Bq/m^2/s] and mean soil-gas concentration [Bq/m^3]."""
    flux = soil_surface_flux(model, sol, verbose=verbose)
    conc = soil_concentration_means(model, sol, z_ref=z_ref)
    # analytic (subgrid-exact) thoron flux from the final state
    nz  = model.nz
    th  = np.clip(sol.y[0:nz, -1], 1e-8, 0.95 * model.p["phi_m"])
    T_f = model.T_field(float(sol.t[-1]))
    J_tn = model.thoron_flux_analytic(th, T_f)
    out  = {
        "J_surface_Bq_m2_s": flux["J_surface"],
        "J_thoron_Bq_m2_s":  float(J_tn),
        "ratio_Tn_Rn":       float(J_tn / max(flux["J_surface"], 1e-30)),
        "C_mean_kBq_m3":     conc["C_mean_kBq"],
        "C_at_zref_kBq_m3":  conc["C_at_zref_kBq"],
        "z_ref_m":           conc["z_ref"],
        "_flux_terms":       flux,
        "_conc_terms":       conc,
    }
    if verbose:
        print(f"[soil_outputs]  J_surface = {out['J_surface_Bq_m2_s']:.4e} Bq/m^2/s | "
              f"C_mean = {out['C_mean_kBq_m3']:.2f} kBq/m^3 | "
              f"C({conc['z_ref']:.1f} m) = {out['C_at_zref_kBq_m3']:.2f} kBq/m^3")
    return out
