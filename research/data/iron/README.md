# Dati IRON (INGV) — stazioni di Ischia: ISAR, ISCA, ISMAR

## Provenienza

Quattro file scaricati il 2026-08-06 dal dataset:

> Piersanti, Antonio; Cannelli, Valentina; Galli, Gianfranco; Soldati, Gaia;
> Piochi, Monica; Iovine, Raffaella Silvia; et al. (2025). *Soil radon and meteo
> timeseries from the Italian Radon mOnitoring Network (IRON) 2009-2021*.
> figshare, DOI
> [10.6084/m9.figshare.27292554](https://doi.org/10.6084/m9.figshare.27292554)
> (versione 1: `10.6084/m9.figshare.27292554.v1`), 24 creator, 127 file totali.

Articolo di riferimento (data descriptor). **Attenzione: il titolo dell'articolo non coincide con quello del dataset** — sono due opere distinte con DOI distinti:

> Piersanti A., Cannelli V., Galli G., Soldati G., Piochi M., Iovine R.S., et al.
> (2025), «Soil Radon Time Series from the Italian Radon Monitoring Network
> (IRON)», *Scientific Data* 12:415, 24 autori, DOI
> [10.1038/s41597-025-04735-0](https://doi.org/10.1038/s41597-025-04735-0).


Dei 127 file dell'archivio sono stati scaricati **solo** i quattro relativi a Ischia (`Radon_Time_Serie_of_{ISAR,ISCA,ISMAR}.csv`,
`Meteo_Time_Serie_of_ISAR.csv`). **Non bisogna mai modificarli: sono la fonte primaria.**
Dimensioni, conteggi righe e checksum MD5 sono registrati in `QC.md`.

## Licenza e attribuzione

Il dataset è distribuito con licenza **CC BY 4.0**
(<https://creativecommons.org/licenses/by/4.0/>), come dichiarato dai
metadati figshare letti via API (`license: CC BY`). La licenza richiede
**attribuzione**: ogni uso di questi dati (grafici, CSV derivati, articoli) deve citare il dataset figshare (DOI 10.6084/m9.figshare.27292554) e l'articolo Piersanti et al. 2025 (DOI 10.1038/s41597-025-04735-0), indicando se i dati sono stati modificati (es. dedup, ricampionamento).

## Le tre stazioni

Dalla Tabella 1 del paper (letta il 2026-08-06 su nature.com; colonne:
Station Code, Start, Stop, Latitude, Longitude, Installation Type):

| Codice | Start | Stop | Lat | Lon | Tipo installazione |
|---|---|---|---|---|---|
| ISAR | 2018-12-10 | working | 40.731 | 13.964 | **indoor** |
| ISCA | 2018-12-10 | working | 40.739 | 13.887 | **borehole** |
| ISMAR | 2019-05-14 | 2019-11-26 | 40.701 | 13.912 | **borehole** |

Le tre coppie di coordinate cadono tutte sull'**isola d'Ischia** (settore
orientale, nord-occidentale e meridionale dell'isola, rispettivamente).
**Attenzione:** il paper pubblica solo codici e coordinate, non i toponimi.
Qualunque attribuzione dei codici a località dell'isola (e la stessa lettura del prefisso «IS» come «Ischia») è una **deduzione dalle coordinate della Tabella 1**, non un dato pubblicato. Va dichiarata come tale in ogni uso. Anche la descrizione di ISAR come «locale tecnico interrato» proviene da `../INDOOR_CANDIDATI.md`, non dal paper, che dice soltanto `indoor`.

Perché contano per il progetto: ISAR è la prima serie **C_indoor(t)** del
progetto. Sta accanto a serie di sottosuolo (ISCA, ISMAR) sulla stessa isola già modellata, con meteo co-locata (stessa stazione ISAR, stesso logger,mstessi timestamp).

## Struttura dei file

Separatore `;`, timestamp `YYYY-MM-DD HH:MM:SS.f`, passo mediano 4 h
(240 min, il massimo tra gli intervalli 15–240 min citati dal paper).

- File radon: `Date;Rn (Bq/m^3);Rn corrected (Bq/m^3);Rn Instrument;Source`
- File meteo: `Date;Temperature (°C);Relative Humidity (%);Meteo Instrument;Source`

`Rn corrected` è la serie con correzione di calibrazione per l'effetto
dell'umidità sullo strumento (modelli e coefficienti per singolo strumento,
sezione Methods del paper). `Source` vale `tool` (scarico dallo strumento) o `website` (scarico dal sito del costruttore): il paper avverte che i due canali producono righe duplicate, e che non esiste una serie unica «ufficiale».

## Avvertenze d'uso (obbligatorie)

Registrate in `../INDOOR_CANDIDATI.md` e confermate/quantificate dal QC
(`QC.md`, tutti i numeri):

1. **ISCA e ISMAR sono borehole a testa pozzo (<2 m), non soil gas a
   profondità fissa.** Un rapporto indoor/suolo costruito su questi file
   **non è confrontabile** con il rapporto alla Nazaroff usato nel modello.
2. **Dedup obbligatoria** sulle coppie `Source=tool/website`: in ISAR ~43 %
   delle righe sta in coppie stessa-data, e le copie possono differire di
   1–7 Bq/m³ (radon) o 0.1–0.3 °C (temperatura), oltre a 141 coppie con
   timestamp sfalsati di 1–2 s. Un `drop_duplicates` ingenuo non basta:
   scegliere una sorgente (o una regola per finestra) e dichiararla.
3. **Dichiarare quale colonna Rn si usa.** Il rapporto corretta/grezza è
   ~1.28 per ISAR ma varia per strumento in ISCA (0.92–1.17): sulla colonna
   grezza i tre periodi ISCA non sono omogenei tra loro.
4. **Nessun metadato edificio** per ISAR: né volume, né ACH, né descrizione del locale. Ogni bilancio indoor su ISAR ha questi parametri liberi.
5. ISCA ha due fermi lunghi (265 e 153 giorni) coincidenti con i cambi di
   strumento, picchi fino a 391 572 Bq/m³ e 5 righe a 99 900 esatti
   (sospetto fondo scala, non documentato dal paper). ISMAR copre solo
   2019-05-15 → 2019-11-23 (166 giorni, solo `Source=tool`) e termina tre
   giorni prima dello stop dichiarato in Tabella 1.

## File in questa cartella

| File | Contenuto |
|---|---|
| `Radon_Time_Serie_of_ISAR.csv` | Radon indoor ISAR, 12 465 righe dati, 2018-12-11 → 2021-12-31 |
| `Radon_Time_Serie_of_ISCA.csv` | Radon borehole ISCA, 5 513 righe dati, 2018-12-11 → 2021-12-31 (con fermi) |
| `Radon_Time_Serie_of_ISMAR.csv` | Radon borehole ISMAR, 973 righe dati, 2019-05-15 → 2019-11-23 |
| `Meteo_Time_Serie_of_ISAR.csv` | T e RH alla stazione ISAR, 12 465 righe dati, stessi timestamp del radon ISAR |
| `qc_iron.py` | Script QC (stdlib, read-only sulle fonti) che riproduce i numeri di `QC.md` |
| `QC.md` | Rapporto QC: duplicati, grezza vs corretta, buchi, anomalie, discrepanze |
