# QC dei file IRON — Ischia (ISAR, ISCA, ISMAR)

Data del controllo: 2026-08-06. File scaricati lo stesso giorno da figshare,
articolo 27292554 versione 1 (vedi `README.md` per provenienza e licenza).
**I quattro CSV scaricati non sono stati modificati in alcun modo**: sono la
fonte primaria. Ogni numero di questo documento è stato calcolato
sui file scaricati, con lo script `qc_iron.py` in questa cartella.

## Comandi usati

```bash
# elenco file dell'articolo (API figshare)
curl -s "https://api.figshare.com/v2/articles/27292554" | python3 -m json.tool

# download dei soli 4 file di Ischia (URL presi dalla risposta API)
cd data/iron
curl -sL -o Radon_Time_Serie_of_ISAR.csv  "https://ndownloader.figshare.com/files/49964862"
curl -sL -o Radon_Time_Serie_of_ISCA.csv  "https://ndownloader.figshare.com/files/49964868"
curl -sL -o Radon_Time_Serie_of_ISMAR.csv "https://ndownloader.figshare.com/files/49964865"
curl -sL -o Meteo_Time_Serie_of_ISAR.csv  "https://ndownloader.figshare.com/files/49964679"

# verifica dimensioni e righe
ls -l && wc -l *.csv && md5 *.csv

# QC (stdlib, niente pandas nel venv)
python data/iron/qc_iron.py
```

## File scaricati

| File | Byte (= dichiarati dall'API) | Righe file | Righe dati | MD5 |
|---|---|---|---|---|
| `Radon_Time_Serie_of_ISAR.csv` | 650 787 | 12 466 | 12 465 | `3dc1c02e624ba954d6bd9e2a611b706d` |
| `Radon_Time_Serie_of_ISCA.csv` | 292 997 | 5 514 | 5 513 | `32949f986d21fb69c52e56c9c65c6cd3` |
| `Radon_Time_Serie_of_ISMAR.csv` | 49 137 | 974 | 973 | `0187304c1a91ec1726472f7e07200953` |
| `Meteo_Time_Serie_of_ISAR.csv` | 682 421 | 12 466 | 12 465 | `4ee9f6df1d88cb652edf117fcde83657` |

Le dimensioni in byte coincidono esattamente con quelle dichiarate dall'API
figshare. Le righe file di ISAR (12 466) e ISCA (5 514) coincidono con quanto
annotato dal ricercatore precedente in `../INDOOR_CANDIDATI.md`.

Formato comune: separatore `;`, timestamp `YYYY-MM-DD HH:MM:SS.f`, colonne
`Date;Rn (Bq/m^3);Rn corrected (Bq/m^3);Rn Instrument;Source` (radon) e
`Date;Temperature (°C);Relative Humidity (%);Meteo Instrument;Source` (meteo).
Passo di campionamento mediano: 4 h per tutti i file (il paper dichiara
intervalli impostabili 15–240 min; qui è usato il massimo, 240 min).

## 1. Righe duplicate `Source=tool` / `Source=website`

Il paper spiega che gli operatori possono scaricare i dati sia dallo strumento
(`tool`) sia dal sito del costruttore (`website`), e che questo produce
duplicazioni nel database. Nei nostri file **nessuna riga è duplicata con
tutte e 5 le colonne identiche** (la colonna `Source` differisce sempre).
I duplicati veri sono coppie stessa-data con `Source` diverso:

| File | Righe dati | tool / website | Righe in coppie identiche tranne `Source` | Timestamp esatti condivisi (righe / gruppi) | Coppie quasi-duplicate (0<dt≤2 s, `Source` diverso) | Timestamp unici |
|---|---|---|---|---|---|---|
| Radon ISAR | 12 465 | 6 493 / 5 972 | 8 836 (=4 418 coppie) | 10 812 / 5 406 | 141 | 7 059 |
| Radon ISCA | 5 513 | 2 993 / 2 520 | 942 (=471 coppie) | 3 476 / 1 738 | 16 | 3 775 |
| Radon ISMAR | 973 | 973 / 0 | 0 | 0 | 0 | 973 |
| Meteo ISAR | 12 465 | 6 493 / 5 972 | 4 054 (=2 027 coppie) | 10 812 / 5 406 | 141 | 7 059 |

**Le due copie non sono sempre identiche nei valori.** Tra i gruppi che
condividono il timestamp esatto (sempre coppie, mai gruppi da 3+):

- Radon ISAR: 988 coppie su 5 406 differiscono nei valori — scarto mediano
  0 Bq/m³ sulla colonna grezza (max 1) e 1 Bq/m³ sulla corretta (max 7);
- Radon ISCA: 1 267 coppie su 1 738 differiscono — grezza sempre identica,
  corretta con scarto mediano 2 Bq/m³ (max 117);
- Meteo ISAR: 3 379 coppie su 5 406 differiscono — solo la temperatura,
  scarto mediano 0.1 °C (max 0.3); l'umidità è sempre identica.

In più esistono coppie `tool`/`website` con timestamp sfalsati di 1–2 secondi
e valori quasi uguali (es. ultima coppia ISAR: `22:59:13 tool` 1138/1454 vs
`22:59:14 website` 1139/1455): 141 in ISAR (radon e meteo), 16 in ISCA.
**Conseguenza operativa: la dedup non può essere un semplice
`drop_duplicates` sulle 5 colonne.** Va fatta scegliendo una sorgente
(consiglio: tenere `tool`, che copre tutto, e integrare con `website` solo
dove `tool` manca) oppure per finestra temporale. In ISMAR il problema non
esiste (solo `tool`).

## 2. Colonna grezza vs `Rn corrected`

Il paper descrive la colonna corretta come correzione di calibrazione per
l'effetto dell'umidità sullo strumento, con modelli e coefficienti propri di
ciascuno strumento. I dati lo confermano: il rapporto corretta/grezza è
stretto per strumento ma cambia da strumento a strumento.

| File / strumento | n | Rapporto mediano corr/grezza | [p5, p95] | Periodo |
|---|---|---|---|---|
| ISAR / AERC-102151-C | 12 465 | **1.2825** | [1.2696, 1.2979] | 2018-12-11 → 2021-12-31 |
| ISCA / AERC-102154-C | 16 | 1.1688 | [1.1681, 1.1701] | 2018-12-11 → 2018-12-13 |
| ISCA / AERC-102190-B | 1 912 | **0.9164** | [0.9138, 1.0474] | 2019-05-15 → 2020-05-26 |
| ISCA / AERC-102220-B | 3 585 | 1.0743 | [1.0009, 1.1400] | 2021-02-16 → 2021-12-31 |
| ISMAR / AERC-102197-B | 973 | 1.0306 | [0.9388, 1.0897] | 2019-05-15 → 2019-11-23 |

Differenza tipica (mediana corretta−grezza): ISAR +212 Bq/m³ (~+28%),
ISCA +110 Bq/m³ (~+5% sul file intero, ma **negativa** per lo strumento
102190-B), ISMAR +6 Bq/m³ (~+3%). Chi usa questi dati deve **dichiarare
quale colonna usa**; un confronto tra periodi ISCA con strumenti diversi
sulla colonna grezza mescola calibrazioni diverse.

## 3. Copertura temporale, buchi, valori anomali

| File | Range effettivo | Giorni con dati | Buchi > 24 h |
|---|---|---|---|
| Radon ISAR | 2018-12-11 00:55 → 2021-12-31 22:59 | 1 109 | 4 (max 5.1 g, 2020-03-15→03-20) |
| Radon ISCA | 2018-12-11 00:08 → 2021-12-31 20:13 | 632 | 8 (max **265.2 g**, 2020-05-26→2021-02-16; poi **152.6 g**, 2018-12-13→2019-05-15; 39.5 g, 2020-03-21→04-30) |
| Radon ISMAR | 2019-05-15 00:00 → 2019-11-23 17:17 | 166 | 4 (max 24.6 g, 2019-10-21→11-14) |
| Meteo ISAR | identico a Radon ISAR (stesso logger) | 1 109 | 4 (identici) |

I due buchi enormi di ISCA coincidono con i **cambi di strumento**
(102154-C termina 2018-12-13; 102190-B copre 2019-05-15→2020-05-26;
102220-B riparte 2021-02-16): non sono buchi casuali ma periodi di stazione
ferma. Giorni di sovrapposizione: ISAR∩ISCA = 626, ISAR∩ISMAR = 164,
tutte e tre = 164.

Valori:

- **Nessun valore negativo o non finito in nessun file.** Radon ISAR ha
  4 zeri esatti (grezza e corretta); ISCA e ISMAR nessuno zero.
- Radon ISAR (indoor): grezza 0–7 994, mediana 747 Bq/m³.
- Radon ISCA (borehole): grezza 48–391 572, mediana 2 139 Bq/m³. Presenta
  picchi estremi: 354 righe > 10 kBq/m³, 37 > 50 kBq/m³, 4 > 100 kBq/m³
  (tutte e 4 il 2019-11-12, strumento 102190-B, massimo 391 572 Bq/m³).
  Inoltre 5 righe hanno grezza esattamente **99 900** (2019-12-13→2020-02-04,
  solo 102190-B, con corretta variabile 91 305–103 201): sospetto valore di
  fondo scala dello strumento, da trattare con cautela. Il paper non lo
  documenta: discrepanza dichiarata, non risolta.
- Radon ISMAR (borehole): grezza 6–8 766, mediana 464 Bq/m³.
- Meteo ISAR: T 0.2–29.0 °C (mediana 18.2), RH 50–87 % (mediana 78).

## 4. Discrepanze con la Tabella 1 del paper (dichiarate)

Tabella 1 (letta il 2026-08-06 su nature.com): ISAR e ISCA `Start
2018-12-10`, ISMAR `2019-05-14 → 2019-11-26`.

- ISAR/ISCA: primi dati 2018-12-11 (≈1 giorno dopo lo start dichiarato) — coerente.
- ISMAR: ultimi dati **2019-11-23 17:17**, tre giorni prima dello stop
  dichiarato 2019-11-26 — discrepanza minore, dichiarata.
- La mediana ISCA a 2 139 Bq/m³ è coerente con l'ordine di grandezza
  «~1–2 kBq/m³» annotato in `../INDOOR_CANDIDATI.md`, ma i picchi a
  10²–10⁵ Bq/m³ mostrano che la serie non è stazionaria.
