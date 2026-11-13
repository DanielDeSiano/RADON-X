from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union


class Role(str, Enum):

    BLOCKED_TEST = "test-bloccato"
    VALIDATION = "validazione"
    CALIBRATION = "calibrazione"
    EXPLORATORY = "solo-esplorativo"
    INPUT_PARAMETERS = "parametri-di-ingresso"


@dataclass(frozen=True)
class NotApplicable:

    reason: str


@dataclass(frozen=True)
class NeedsSource:

    what_to_look_for: str


Field = Union[str, NotApplicable, NeedsSource]

REQUIRED_FIELDS = (
    "site",
    "instrument",
    "timestamp",
    "sampling_frequency",
    "units",
    "soil_properties",
    "environmental_conditions",
    "missing_values",
    "licence",
    "split",
)


@dataclass(frozen=True)
class Dataset:

    files: tuple[str, ...]
    role: Role
    site: Field
    instrument: Field
    timestamp: Field
    sampling_frequency: Field
    units: Field
    soil_properties: Field
    environmental_conditions: Field
    missing_values: Field
    licence: Field
    split: str
    notes: str = ""
    provenance_in: str = ""

    def declared(self) -> dict[str, Field]:
        return {name: getattr(self, name) for name in REQUIRED_FIELDS}

    def gaps(self) -> dict[str, NeedsSource]:
        return {
            name: value
            for name, value in self.declared().items()
            if isinstance(value, NeedsSource)
        }


_NA_LAB = NotApplicable(
    "laboratory table of material properties: it has no site, no sampling "
    "instant and no sampling frequency"
)
_NA_NO_GAPS = NotApplicable("tabulated values, complete as published: no gaps to declare")

MANIFEST: tuple[Dataset, ...] = (
    Dataset(
        files=("voltaggio2006_lazio_6siti.csv",),
        role=Role.VALIDATION,
        site="Northern Lazio, Italy: six sites on three lithotypes (volcanic, "
        "epiclastic, Chiani-Tevere fluvio-lacustrine unit)",
        instrument=NeedsSource("accumulation chamber model, in Voltaggio 2006 §2"),
        timestamp=NeedsSource("campaign dates, in Voltaggio 2006 §2"),
        sampling_frequency=NeedsSource("counting time per point, in Voltaggio 2006 §2"),
        units="Ra-226 Bq/kg; g dimensionless; porosity, saturation dimensionless; "
        "density kg/m3; flux Bq/m2/day",
        soil_properties="Ra-226, laboratory emanation g, porosity, saturation at "
        "sampling, dry bulk density: all published per site",
        environmental_conditions=NeedsSource("weather at the flux campaign, not published"),
        missing_values=_NA_NO_GAPS,
        licence="Springer subscription article (closed). The values are tabular data.",
        split="Field validation. No row enters calibration.",
        notes="g is a laboratory value and the source does not state whether the "
        "sample was oven-dried. The dry/plateau reading used in the validation "
        "documents is a hypothesis, not a datum.",
    ),
    Dataset(
        files=("bratislava_eckertova.csv",),
        role=Role.VALIDATION,
        site="Bratislava, FMPI Comenius University campus, sandy-silt ranker soil",
        instrument="Automatic accumulation chamber to ISO 11665-7, two 6-hour "
        "cycles per day; HPGe gamma spectrometry in an underground laboratory for Ra-226",
        timestamp="October 2020 to June 2024, almost four years continuous",
        sampling_frequency="Two 6-hour accumulation cycles per day",
        units="Ra-226 Bq/kg; flux mBq/m2/s; porosity and saturation dimensionless; "
        "density kg/m3; temperature degC",
        soil_properties="Ra-226 28.0+-0.3, Ra-228 25.9+-0.5 Bq/kg; total porosity "
        "0.459; dry bulk density 1430 kg/m3; texture published",
        environmental_conditions="Soil temperature 1-27 degC at 30 cm. Volumetric "
        "moisture measured hourly but published only as a figure.",
        missing_values="Field moisture is the one missing input: recovered as a "
        "window Sw 0.24-0.53 by inverting the published diffusion length.",
        licence="SSRN open preprint; journal version DOI to be reconfirmed on the "
        "publisher page (Crossref gives 10.1016/j.jenvrad.2026.107958)",
        split="Field validation. No row enters calibration.",
        notes="Independent flux at the same campus from Holy et al. 2016 "
        "(15.0+-1.8 mBq/m2/s, summer, chamber) is a cross-check but is a summer "
        "value and the authors warn summer rates run about twice the annual mean.",
    ),
    Dataset(
        files=("kobe_hosoda2009.csv", "kobe_hosoda2009_per_data.csv"),
        role=Role.VALIDATION,
        site="Kobe, Japan, Mount Rokko weathered granite, seven measurement points",
        instrument="Simultaneous radon and thoron exhalation measurement; soil "
        "parameters in the paper's tables",
        timestamp="May, August, November 2007 and January 2008 (four seasons)",
        sampling_frequency="One campaign per season per point",
        units="Ra-226 Bq/kg; density kg/m3 and g/cm3; porosity and saturation "
        "dimensionless; flux mBq/m2/s",
        soil_properties="Dry and grain density, porosity, Ra-226 and water "
        "saturation at each date; emanation measured against moisture",
        environmental_conditions="Water saturation per date; season declared",
        missing_values="Th-232 not published, so no thoron validation is possible. "
        "The row-to-date pairing is not explicit in the source: the comparison "
        "stays per point.",
        licence="J-STAGE, free access",
        split="Validation restricted to points 1 and 2 (undisturbed, off campus). "
        "Points 3, 5, 6, 7 are exploratory only: campus ground, anthropogenic "
        "disturbance. The per-date file is exploratory only.",
    ),
    Dataset(
        files=("zhuo2006_10sites.csv",),
        role=Role.VALIDATION,
        site="Ten sites in Japan and Korea",
        instrument="Naturally ventilated accumulator, 0.28 per hour",
        timestamp="April 2003 to April 2004, seasonally averaged",
        sampling_frequency="Four seasons per site",
        units="Ra-226 Bq/kg; density g/cm3; porosity and saturation dimensionless; "
        "temperature K; flux mBq/m2/s",
        soil_properties="Ra-226, bulk density, total porosity, emanation power of "
        "oven-dried soil at 25 degC, saturation min/mean/max",
        environmental_conditions="Soil temperature min/mean/max per site",
        missing_values="Sites A, G and I carry the authors' own footnote "
        "'confirmed as measuring problems' and are exploratory only.",
        licence="Taylor & Francis, free access",
        split="Validation on the seven sites without the authors' footnote. "
        "Sites A, G, I exploratory only, on the authors' own warning.",
    ),
    Dataset(
        files=("ranger_fattore_conversione.csv",),
        role=Role.VALIDATION,
        site="Ranger uranium province, Northern Territory, Australia",
        instrument=NeedsSource("chamber type, in the IR477/IR616 government reports"),
        timestamp="Government report series, dates in the reports",
        sampling_frequency=NeedsSource("counting time, in the reports"),
        units="Flux mBq/m2/s; conversion factor dimensionless",
        soil_properties="Emanation, porosity and density NOT measured by this source",
        environmental_conditions="Moisture range read off a figure only",
        missing_values="Emanation, porosity, density all absent: the flux cannot be "
        "predicted from first principles here.",
        licence="Australian government reports, reuse with attribution "
        "(Commonwealth Creative Commons licence to be verified)",
        split="Validation restricted to the RE-R quantity. Everything else "
        "exploratory only, for want of inputs.",
    ),
    Dataset(
        files=("santander_larucbed.csv",),
        role=Role.BLOCKED_TEST,
        site="LaRUC reference exhalation bed, Universidad de Cantabria, Santander: "
        "uraniferous soil in a sealed steel tray, thickness 0.165 m",
        instrument="Three independent measurement methods; metrological uncertainties",
        timestamp="Published 2023 (Grossi et al., AMT)",
        sampling_frequency="Per experiment, in the paper",
        units="Ra-226 Bq/kg; flux mBq/m2/s; thickness m",
        soil_properties="Ra-226 19130+-350 Bq/kg; geometry known with metrological "
        "uncertainty. Porosity and density not tabulated here.",
        environmental_conditions="Controlled laboratory",
        missing_values=_NA_NO_GAPS,
        licence="Open access, data on Zenodo",
        split="BLOCKED TEST. Sealed geometry with declared uncertainty: a "
        "discrepancy is attributable to the model. Must stay untouched by tuning.",
        notes="Affected by the Crank-Nicolson defect of audit section 25 until it "
        "was fixed: the corrected model/measurement ratio is 1.11 against the "
        "primary measurement, not the 0.97 the bug flattered.",
    ),
    Dataset(
        files=("huelva_banchi_riferimento.csv",),
        role=Role.BLOCKED_TEST,
        site="Two phosphogypsum metrological reference beds, Universidad de Huelva "
        "(RB1 6.0 cm, 35 kg; RB2 13.0 cm, 73 kg)",
        instrument="Certified exhalation over 48 and 44 experiments respectively",
        timestamp="Gutierrez-Alvarez et al. 2020",
        sampling_frequency=NotApplicable(
            "certified mean over tens of experiments, not a time series"
        ),
        units="Ra-226 Bq/kg; density kg/m3; thickness cm; flux mBq/m2/s",
        soil_properties="Ra-226 451+-7 (RB1) and 420+-8 (RB2) Bq/kg; densities 1080 "
        "and 1011 kg/m3; porosity derived from density with the grain density of "
        "the 2026 set",
        environmental_conditions="Beds seasoned 1-3 months, near dry: residual "
        "saturation 0.05 assumed and declared",
        missing_values="Emanation has a declared band of a factor 2.8 across "
        "sources (0.15 / 0.20 / 0.42): the comparison is presented on the band.",
        licence="Green open access (Radiation Measurements 133:106284)",
        split="BLOCKED TEST, for the same reason as Santander.",
        notes="At the measured emanation 0.20 both certified fluxes sit ABOVE their "
        "own production ceiling (ratios 1.080 and 1.013), which inverts to "
        "emanation 0.2161 and 0.2026, inside the measured band 0.20+-0.04.",
    ),
    Dataset(
        files=("vanderspoel1998_diffusivita.csv",),
        role=Role.BLOCKED_TEST,
        site="Laboratory sand column, TU Eindhoven doctoral thesis",
        instrument="Laboratory diffusivity measurement, thesis chapter",
        timestamp="Thesis 1998",
        sampling_frequency=NeedsSource("per-experiment counting time, in the thesis"),
        units="Diffusivity m2/s; porosity and saturation dimensionless",
        soil_properties="Porosity, saturation and density of the sand published",
        environmental_conditions="Controlled laboratory",
        missing_values=_NA_NO_GAPS,
        licence="Open institutional repository, attribution",
        split="BLOCKED TEST. It is the only source that constrains the diffusivity "
        "law directly rather than through a flux.",
    ),
    Dataset(
        files=("huelva_prismi_e_campo.csv",),
        role=Role.EXPLORATORY,
        site="Huelva phosphogypsum stack: closed-chamber prisms (10 and 20 cm) and "
        "a field core over 88 cm",
        instrument="Closed accumulation chamber 2.4 L for the prisms; core profiles",
        timestamp="Castano-Casco et al. 2026",
        sampling_frequency="Per experiment",
        units="Ra-226 Bq/kg; flux mBq/m2/s; depth cm",
        soil_properties="Ra-226 669 Bq/kg, porosity 0.5, saturation 0.4, density "
        "1210 kg/m3, emanation 0.15 (state not declared)",
        environmental_conditions="Field core with moisture over 13 sections",
        missing_values="The prism rows carry an internal inconsistency of the "
        "source: the measured fluxes sit 29-36% BELOW the production ceiling "
        "computed with the paper's own emanation, which free exhalation cannot do. "
        "Both the measurement and the paper's own FEM refer to the closed-chamber "
        "experiment, where back-diffusion suppresses the net flux.",
        licence="CC-BY with supplementary (Journal of Hazardous Materials 507:141754)",
        split="Field row: validation. Prism rows: exploratory only, on the "
        "internal inconsistency above.",
    ),
    Dataset(
        files=("bessines_ferry2000.csv",),
        role=Role.EXPLORATORY,
        site="Uranium mill tailings cover, Bessines-sur-Gartempe, France, two plots",
        instrument="Continuous flux over one year, annual means tabulated",
        timestamp="Thesis / CEA report 2000, one year",
        sampling_frequency="Continuous, annual means published",
        units="Flux Bq/m2/s; Ra-226 Bq/kg",
        soil_properties="Ra-226 and emanation published; porosity and density not "
        "tabulated here",
        environmental_conditions=NeedsSource("field moisture, not published"),
        missing_values="Only two rows (annual means per plot): no series.",
        licence="Public report, open archive, attribution",
        split="Exploratory only: two annual means with missing inputs.",
        notes="An IRSN expertise report of 2014 documents INDOOR concentrations in "
        "a house on the same commune and the same material: see INDOOR_CANDIDATI.md.",
    ),
    Dataset(
        files=("nureg1980_colonne_vitro.csv",),
        role=Role.EXPLORATORY,
        site="Uranium tailings columns, Vitro site, NUREG/CR-1081",
        instrument="Laboratory columns, US government report 1980",
        timestamp="Report 1980",
        sampling_frequency=NeedsSource("per-column counting schedule, in the report"),
        units="Ra-226 Bq/kg; flux; moisture percent by weight",
        soil_properties="Ra-226, density, porosity published per column",
        environmental_conditions="Moisture given only as the overall range 9-27% by weight",
        missing_values="Per-column moisture missing: only the overall range.",
        licence="US federal government report: free reuse with attribution",
        split="Exploratory only: moisture, the dominant control, is not resolved "
        "per column.",
    ),
    Dataset(
        files=("tomsk_ryzhakova.csv",),
        role=Role.EXPLORATORY,
        site="Tomsk, Russia, Ryzhakova group test site",
        instrument=NeedsSource("chamber and spectrometry, in the group's papers"),
        timestamp=NeedsSource(
            "campaign dates, in the Ryzhakova group publications for the Lagerny site"
        ),
        sampling_frequency=NeedsSource(
            "sampling schedule per point, in the same publications"
        ),
        units="Ra-226 Bq/kg; flux mBq/m2/s; porosity dimensionless",
        soil_properties="Ra-226, porosity, density, moisture published",
        environmental_conditions=NeedsSource("weather during the campaigns"),
        missing_values=_NA_NO_GAPS,
        licence=NeedsSource("access status of the source publication"),
        split="Exploratory only, for want of provenance on instrument and dates.",
    ),
    Dataset(
        files=(
            "hosoda2007_moisture_curve.csv",
            "hosoda2007_calculated_curve.csv",
            "hosoda2007_fig2_raw_scatter.csv",
        ),
        role=Role.EXPLORATORY,
        site="Weathered granite soil, central Ibaraki Prefecture, Japan, laboratory tray",
        instrument="Laboratory tray 275x210 mm, layer 50 mm; HPGe 80000 s for Ra-226",
        timestamp="Hosoda et al. 2007",
        sampling_frequency=NotApplicable(
            "moisture curve: one point per moisture level, not a time series"
        ),
        units="Moisture percent by mass; exhalation mBq/m2/s",
        soil_properties="Ra-226 57.0 and Ra-224 55.8 Bq/kg; density 1730 kg/m3; "
        "porosity 0.30",
        environmental_conditions="Moisture is the controlled variable",
        missing_values="Emanation NOT measured here: read off a Strong & Levins "
        "1982 graph, f = 0.079-0.28. Values extracted from figure vector geometry.",
        licence="Taylor & Francis / J-STAGE, free access "
        "(DOI 10.1080/18811248.2007.9711855)",
        split="Exploratory only: the emanation is not measured, so a flux "
        "comparison would test an assumption, not the model.",
        notes="The Richards step is not robust on this configuration at "
        "millimetric cells (audit section 26): the correct protocol freezes the "
        "water field, and with it the model matches the slab solution to 1-2%.",
    ),
    Dataset(
        files=("gallese2020_materiali.csv", "gallese2022_parametri_modello.csv"),
        role=Role.INPUT_PARAMETERS,
        site="Volcanic building materials of Lazio, characterised in the laboratory "
        "(Tufo di Gallese, Tufo di Riano, Peperino Grigio, Peperino Rosato)",
        instrument="Gamma spectrometry for the radionuclides; exhalation measured "
        "per unit surface",
        timestamp=_NA_LAB,
        sampling_frequency=_NA_LAB,
        units="Radionuclides Bq/kg; exhalation Bq/m2/h; porosity dimensionless; "
        "diffusivity m2/s; density kg/m3",
        soil_properties="Not soil: building material. Ra-226 190+-3 Bq/kg and "
        "porosity 0.43 for Tufo di Gallese, the material of the chamber.",
        environmental_conditions=NotApplicable("material properties, not a measurement campaign"),
        missing_values="The 2022 model-parameter table publishes no uncertainties: "
        "those for density and Ra-226 exist only in the 2020 materials table.",
        licence="2020 CC BY-NC-ND 3.0; 2022 CC BY 4.0 (both verified verbatim on "
        "the PDFs)",
        split="Input parameters. Not a comparison target.",
    ),
    Dataset(
        files=("gallese2020_equilibri.csv",),
        role=Role.VALIDATION,
        site="Scale model room built of Tufo di Gallese blocks: 62x50x35 cm "
        "internal, volume about 0.110 m3, exhaling wall surface 0.78 m2",
        instrument="AER PLUS (Algade) calibrated at the INGV radon chamber for the "
        "absolute-humidity effect; RAD7 (Durridge) in Sniff mode",
        timestamp="Lucchetti et al. 2020, six experimental conditions",
        sampling_frequency="Continuous to equilibrium; equilibrium values tabulated",
        units="Concentration Bq/m3; temperature degC",
        soil_properties=NotApplicable("no soil: the source is the wall material"),
        environmental_conditions="Experimental temperature range per condition (18-31 degC)",
        missing_values="Rn-220 is 'bdl' in three of the six conditions, as published.",
        licence="CC BY-NC-ND 3.0 (verified on the PDF, p.71)",
        split="Validation of the indoor source term, with a declared limit: the "
        "sealed configurations sit a factor about 2 above the prediction from the "
        "published E, because sealing the outer face redirects the outgoing flux "
        "inwards. Under the 2E scenario the discrepancy is -4.3% on test 2.",
        notes="Test 1 (bare pervious walls) and test 5b are not comparable to a "
        "sealed-wall prediction and are used only for the leakage estimate.",
    ),
    Dataset(
        files=("gallese2022_ach.csv",),
        role=Role.CALIBRATION,
        site="Same scale model room, twelve ventilation experiments",
        instrument="AER PLUS and RAD7; air flow imposed and measured, ACH computed "
        "as flow over volume",
        timestamp="Lucchetti, Galli, Tuccimei 2022",
        sampling_frequency="Continuous to a new equilibrium per experiment",
        units="Air flow L/min; ACH 1/h (positive injection, negative extraction); "
        "RIR percent",
        soil_properties=NotApplicable(
            "no soil is involved: the radon source is the tuff wall material of the "
            "chamber, characterised in gallese2020_materiali.csv"
        ),
        environmental_conditions="Inner cover present or absent, declared per experiment",
        missing_values="The absolute baseline concentrations of the 2022 campaign "
        "are not tabulated: only the relative reductions. An absolute comparison "
        "with these data alone is not possible.",
        licence="CC BY 4.0 (verified on the PDF, p.81)",
        split="MIXED, and this is the project's first calibration assignment. "
        "Experiment 5 alone was used to estimate the background ACH of the chamber "
        "(leakage), n0 = 0.0389 per hour: it is CALIBRATION and must never be "
        "quoted as validation. Experiments 1-4 are then pure predictions and are "
        "validation. Experiments 6-12 are validation too, and the model FAILS them "
        "(z from 11.1 to 24.9): recorded as a falsification, not removed.",
        notes="Assigned 2026-08-06, when the estimate was made. Before that date "
        "the project had no dataset in calibration. The reason is in "
        "docs/validazione-indoor-gallese.md.",
    ),
    Dataset(
        files=("neznal2005_transfer_factor.csv",),
        role=Role.VALIDATION,
        site="Five real houses in the Czech Republic (Lounovice A and B, "
        "Potucky, Horni Slavkov A and B), four of them with soil-gas radon and "
        "permeability of the building site",
        instrument="Soil: third quartiles of soil-gas radon and permeability "
        "per the uniform Czech method (Neznal et al. 2004). Indoor: track "
        "detectors, electrets and continuous monitors, declared per row",
        timestamp="Measured 2004-2005; source reopened and re-read in full on "
        "2026-08-31 before transcription",
        sampling_frequency="One value per house and zone, with the averaging "
        "period declared per row",
        units="Soil-gas kBq/m3; permeability m2; indoor Bq/m3",
        soil_properties="cA75 and k75 of the building site; no laboratory soil "
        "sample parameters",
        environmental_conditions=NotApplicable(
            "long-term averages; the paper publishes no weather covariates"
        ),
        missing_values="Every indoor value is CENSORED: a bound or an "
        "approximation, carried by the qualifier column, never a point. Horni "
        "Slavkov A has no tabulated soil parameters. The project bibliography "
        "claimed fifteen houses; the paper tabulates five, and the project "
        "report that would hold the rest is not public.",
        licence="Not declared in the PDF; public document on the author's site",
        split="Validation of the WIDTH of ring B, not of its central value: "
        "four houses at comparable radon potential span more than 2.6 decades "
        "of transfer factor, with censoring making that a lower bound. Never "
        "a calibration target: N=4 and every value is a bound.",
    ),
    Dataset(
        files=("saelices2019_fasi.csv",),
        role=Role.EXPLORATORY,
        site="Pilot house at Saelices el Chico (Salamanca, Spain), on the "
        "former ENUSA uranium site: soil Ra-226 about 1600 Bq/kg, outdoor "
        "radon about 300 Bq/m3",
        instrument="Radon Scout / Radon Scout Home (SARAD) and AlphaE (Bertin), "
        "chamber-checked; passive detectors mostly saturated",
        timestamp="June 2018 to April 2019, nine testing phases",
        sampling_frequency="Continuous monitoring; the transcription carries "
        "per-phase values only - the hourly series exist only as figures",
        units="Concentration Bq/m3; extraction velocity m/s; reduction percent",
        soil_properties="Not sampled in this paper: Ra-226 cited from Frutos "
        "et al. 2011",
        environmental_conditions="Pearson correlations declared by the authors "
        "on closed periods: pressure -0.44, temperature +0.44, wind -0.59; "
        "sump pressure -55 Pa in phase 6",
        missing_values="Ground floor not tabulated for phase 2; closed phases "
        "carry no per-phase concentration, only the pooled averages (54625 "
        "basement, 26421 ground floor).",
        licence="arXiv preprint 2403.15392, freely distributed; the journal "
        "version is Elsevier",
        split="Exploratory until the project has a two-zone indoor model with "
        "depressurisation: nothing in the code can predict these phases "
        "today. First candidate for validating the building-entry ring on a "
        "real house, as data/INDOOR_CANDIDATI.md records.",
    ),
    Dataset(
        files=(
            "iron/Radon_Time_Serie_of_ISAR.csv",
            "iron/Radon_Time_Serie_of_ISCA.csv",
            "iron/Radon_Time_Serie_of_ISMAR.csv",
            "iron/Meteo_Time_Serie_of_ISAR.csv",
        ),
        role=Role.EXPLORATORY,
        site="Island of Ischia, Italy: ISAR indoor (technical room below ground "
        "level), ISCA and ISMAR boreholes. The mapping of station codes to place "
        "names is DEDUCED from the coordinates of the paper's Table 1, not published.",
        instrument="AERC series continuous radon monitors, model per station in the "
        "'Rn Instrument' column",
        timestamp="ISAR and meteo 2018-12-11 to 2021-12-31; ISCA same window but "
        "632 days with data; ISMAR 2019-05-15 to 2019-11-23",
        sampling_frequency="Configurable 15-240 min; 4 h in these files. Median "
        "step verified in QC.md.",
        units="Radon Bq/m3 (raw and humidity-corrected columns); temperature degC; "
        "relative humidity percent",
        soil_properties=NotApplicable(
            "the network publishes no Ra-226, porosity or density for the stations"
        ),
        environmental_conditions="Temperature and relative humidity of the station, "
        "in the paired meteo file",
        missing_values="Four gaps over 24 h in ISAR, max 5.1 days; ISCA has stops "
        "of 265 and 153 days at instrument changes; duplicate rows with "
        "Source=tool/website quantified in QC.md, a naive de-duplication is not "
        "enough; ISCA has a peak of 391572 Bq/m3 and five rows at exactly 99900, "
        "suspected instrument full scale, undocumented.",
        licence="CC BY 4.0, verified on figshare and DataCite. Attribution required.",
        split="Exploratory ONLY, for two reasons that both matter: the boreholes "
        "are at well head below 2 m and are not the soil gas at fixed depth the "
        "geogenic potential uses, so an indoor/soil ratio computed on these files "
        "is NOT comparable with the Nazaroff ratio; and no building metadata "
        "(volume, air exchange rate, foundation) exists for the indoor station.",
        provenance_in="data/iron/README.md and data/iron/QC.md",
        notes="First indoor measurements to enter the project. Raw downloads: must "
        "not be edited. MD5 recorded in QC.md.",
    ),
)


def dataset_for(filename: str) -> Dataset:
    for entry in MANIFEST:
        if filename in entry.files:
            return entry
    raise KeyError(
        f"{filename!r} is not declared in the manifest. A dataset without declared "
        "provenance and split may not be used: add it to radon_x.data_manifest."
    )


def files_with_role(*roles: Role) -> tuple[str, ...]:
    return tuple(
        name for entry in MANIFEST if entry.role in roles for name in entry.files
    )


def assert_usable_for_fitting(filename: str) -> None:
    entry = dataset_for(filename)
    if entry.role is Role.BLOCKED_TEST:
        raise PermissionError(
            f"{filename!r} is a BLOCKED TEST ({entry.split}). It may not inform a "
            "fitted model. Use it once, at the end, for evaluation only."
        )
    if entry.role is Role.EXPLORATORY:
        raise PermissionError(
            f"{filename!r} is exploratory only. Reason: {entry.split} "
            "Fitting on it would learn a declared defect as signal."
        )


def data_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "data"
