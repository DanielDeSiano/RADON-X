# Candidati per la validazione indoor

Esito della ricerca bibliografica

## Stato di verifica

Condizione d'uso, valida per ogni voce di questo documento: prima di usare uno
qualunque di questi dataset per calibrare o validare, riaprire la fonte,
ricontrollare i numeri, e solo allora trascrivere in CSV con header di
provenienza.

Quattro degli otto candidati sono in `data/`: la camera di Gallese (candidato 3),
il dataset IRON (candidato 2), le tabelle di Saelices (candidato 1) e le case
di Neznal (candidato 6), trascritti e dichiarati nel manifest. Celleno
(candidato 4) e Bessines (candidato 5) restano bibliografia. Il candidato 7
(Zhao 2025) è sul disco ma fuori da `data/` e dal suo contratto stretto; del
candidato 8 (NRC/CCHT) i dati grezzi non sono scaricabili.

---

## Gli otto dataset più forti

Ordinati per valore per il §14 del mandato. Tutti «pagina-letta», tutti open access.

### 1. Casa pilota di Saelices el Chico (Spagna) — il migliore in assoluto — TRASCRITTE LE TABELLE

Fuente M., Rabago D., Goggins J., Fuente I., Sainz C., Foley M. (2019), *Science of
the Total Environment* 695:133746, DOI 10.1016/j.scitotenv.2019.133746. PDF libero
su elradon.com; anche arXiv:2403.15392.

Casa di prova, geometria nota, sul sito dell'ex impianto uranifero
gestito da ENUSA: Ra-226 del suolo ~1600 Bq/kg (citato nel testo da Frutos 2011).
Monitoraggio continuo giu 2018–apr 2019 in 9 fasi (casa chiusa alternata a depressurizzazione attiva/passiva). A casa chiusa: **seminterrato 55 kBq/m³, piano terra 26 kBq/m³**; 8 sensori di pressione sub-slab; stazione meteo sul tetto; correlazioni radon-meteo pubblicate (r = −0.59 col vento).

**Perché è il migliore:** due zone, sorgente fortissima e caratterizzata, pressione
sub-slab misurata, e un rapporto C_indoor/C_suolo **misurato su sito reale** — cioè
esattamente ciò che oggi il modello prende da Nazaroff 5e-3. Il flusso e il Ra del suolo dello stesso sito arrivano dall'intercomparison traceRadon (Rábago et al., area Sageras adiacente). Limite: le serie orarie sono pubblicate solo in figura.

**Stato:** le tabelle del preprint (arXiv:2403.15392, riletto per intero il
31/08/2026) sono trascritte in `saelices2019_fasi.csv`: nove fasi con date,
concentrazioni per fase e zona (casa chiusa: 54 625 Bq/m³ seminterrato,
26 421 piano terra; fase 4 attiva: 662 / 328, riduzione 99 %), radon esterno
~300 Bq/m³, pressione al pozzetto −55 Pa in fase 6, correlazioni meteo
(pressione −0.44, temperatura +0.44, vento −0.59). Ruolo `solo-esplorativo`
finché il progetto non ha un modello a due zone con depressurizzazione: è il
primo bersaglio pronto per quel modello. Le serie orarie restano in figura,
non trascritte. La richiesta dei dati orari agli autori resta il passo aperto.

### 2. IRON — rete INGV, stazione indoor a Ischia con borehole vicino — SCARICATO

Piersanti A. et al. (2025), «Soil Radon Time Series from the Italian Radon
Monitoring Network (IRON)», *Scientific Data* 12:415, DOI 10.1038/s41597-025-04735-0.
**Dataset scaricabile**, figshare DOI 10.6084/m9.figshare.27292554, **CC BY 4.0** —
62 stazioni, 621 720 misure 2009-2021.

Per questo dataset ci sono due CSV reali: `ISAR` (indoor, locale
tecnico interrato, 12 465 righe di dati, 2018-2021), `ISCA` (borehole sulla stessa isola, 5 513 righe di dati, periodo sovrapposto, stesso tipo di strumento), più la meteo. Prima possibilità nel progetto di una serie C_indoor(t) accanto a una serie di sottosuolo
sulla **stessa isola** già modellata.

**Avvertenze:** 
(a) il borehole ISCA è a testa
pozzo <2 m, ~1-2 kBq/m³ — NON è il soil gas a profondità fissa, e un rapporto indoor/suolo su questi file **non è confrontabile** col rapporto di Nazaroff;
(b) righe duplicate con `Source=tool/website`: dedup obbligatoria; (c) due colonne
Rn (grezza e corretta per umidità): dichiarare quale si usa; (d) l'attribuzione dei codici stazione ai toponimi è dedotta dalle coordinate, non pubblicata — verificare sul materiale ausiliario. Nessun metadato edificio (né volume né ACH).

**Stato:** i quattro file sono in `data/iron/`, con MD5 e controllo qualità in
`iron/README.md` e `iron/QC.md`, ruolo `solo esplorativo` nel manifest. Il controllo
qualità conta 5 406 doppioni sulle 12 465 righe di ISAR. Le avvertenze sono tutte
riportate nella voce 18 di `DATASETS.md`: nessuna è stata risolta, sono state
dichiarate.

### 3. Camera in scala di Tufo di Gallese (Roma Tre + INGV) — TRASCRITTO, e ha falsificato l'equazione

Lucchetti C. et al. (2020), *Nukleonika* 65(2):71-76, DOI 10.2478/nuka-2020-0010;
Lucchetti C., Galli G., Tuccimei P. (2022), *Advances in Geosciences* 57:81-88,
DOI 10.5194/adgeo-57-81-2022 (CC BY). PDF liberi, tabelle lette per intero.

Camera fisica (interno 62×50×35 cm, V ≈ 0.110 m³) costruita con blocchi di ignimbrite del Lazio (Ra-226 = 190±3 Bq/kg, porosità 43%, esalazione misurata), **12 esperimenti con ACH imposto e noto** (0.09–0.49 h⁻¹, immissione ed estrazione) e riduzioni di concentrazione misurate con incertezza. Equilibri: da 770±50 a 13 154±211 Bq/m³ nelle sei condizioni della Tabella 3.

**Perché conta:** è il banco metrologico dell'equazione di bilancio indoor —
l'analogo, per il ramo indoor, di ciò che i banchi di Huelva/Santander sono per il
flusso dal suolo. Litologia coerente coi siti di Voltaggio.

**Stato:** trascritto in `gallese2020_materiali.csv`, `gallese2020_equilibri.csv`,
`gallese2022_ach.csv` e `gallese2022_parametri_modello.csv`. Il
bilancio a sorgente costante (eq. 1 della proposta) riproduce entro 1 sigma quattro dei cinque
esperimenti a parete rivestita con aria immessa (il quinto, a ricambio piu' basso, lo
manca di 2,4) e fallisce gli altri sette di 10,2-25,4
deviazioni, in modo sistematico: la
falsificazione è registrata nella voce 17 di `DATASETS.md`. Da lì è nato
`risultati/indoor_sorgente_variabile.py`, che aggiunge i due termini mancanti —
retrodiffusione dalla parete e aria di reintegro sotto depressione — e sta entro
2,9 deviazioni sulle nove predizioni, al prezzo di tre parametri stimati su un
esperimento ciascuno.

**Quale colonna, per i numeri qui sopra.** Il «2,4» e il «10,2-25,4» vengono
entrambi da `rir_constant_source_pct` di `risultati/indoor_sorgente_variabile.py`,
cioè dalla forma con C_out trascurato e C_0 preso dai 641 Bq/m³ del paper 2022.
L'altra scrittura dello stesso modello a sorgente costante —
`risultati/gallese_box_model.py`, con E·S/V e C_out espliciti e n0 tarata
sull'esperimento 5 — dà **11,1-24,9** sugli stessi sette e z = 0 sul quinto, per
costruzione, perché è il punto su cui n0 è stata stimata. Le due colonne non
coincidono cifra per cifra: chi cita uno di questi numeri scriva accanto quale ha
letto.

Restano fuori dalla trascrizione le serie temporali: le tabelle danno equilibri e
riduzioni relative, non il transitorio.

### 4. Casa di Celleno (VT) — caso reale con esperimenti quasi-controllati

Portaro M. et al. (2024), *Atmosphere* 15(4):425, DOI 10.3390/atmos15040425 (CC BY).

Casa in tufo del 1960, semi-interrata, due piani, **sette segmenti temporali**. Serie continue su due camere, medie±SD per segmento in tabella
(es. camera A: da 286±202 a 2776±1768 Bq/m³). Permette test **differenziali** fra segmenti, aggirando l'assenza di ACH misurato. Gradiente piano terra/primo piano su tutti i segmenti. Limiti: dati orari solo in figura; nessun soil gas co-locato.

### 5. IRSN Bessines (Francia) —

IRSN, rapporto di expertise PRP-HOM/2014-00005 (2 giu 2014), 24 pp., PDF pubblico
su recherche-expertise.asnr.fr (nessun DOI).

Casa costruita sui *sables cyclonés* di Bessines-sur-Gartempe — **stesso comune e stesso materiale del dataset Ferry 2000** . Tabelle lette: 5 stanze × giorno/notte × inverno/estate (inverno: 12 500–27 500 Bq/m³; estate: 800–1 700; medie annue 8 500–18 700), cantina >50 000, fattore di equilibrio misurato ≈0.4.

**Avvertenza :** la sorgente dominante è una galleria tecnica
interrata che drena i sables e sbocca nel garage — un percorso preferenziale, non l'ingresso diffusivo/avvettivo dalla soletta. Va usato come caso di prova della rete d'ingresso, non come misura di entry rate diffusivo. I flussi di esalazione sono dichiarati misurati ma i valori non compaiono nel rapporto: richiesta dati identificata (Algade/campagna AREVA).

### 6. Neznal & Neznal — case ceche col suolo misurato in situ — TRASCRITTO

RADON v.o.s., atto di convegno (misure 2004-2005, progetto SUJB 2/04/V), PDF
integrale su radon-vos.cz/pdf/transfer_factor.pdf, **riaperto e riletto per
intero il 31/08/2026** prima della trascrizione.

Il paper tabula **cinque case, non quindici**: quattro con suolo caratterizzato
(Louňovice A e B, Potůčky, Horní Slavkov B) più una risanata senza suolo
tabulato (Horní Slavkov A). Il rapporto di progetto da centinaia di pagine che
conterrebbe il resto non è pubblico. Ogni valore indoor è **censurato** (un
limite o un circa), mai un punto.

Coppie verificate: Potůčky soil gas 70.7 kBq/m³ → indoor ~20 000 Bq/m³ media
annua (cantina ~77 000, alla pari del soil gas); Louňovice A e B: suolo quasi
uguale (143.6 vs 122.3 kBq/m³), indoor <100 contro >400 Bq/m³; Horní Slavkov B
≤700 Bq/m³ a potenziale pari a Potůčky (90.9 contro 87.6).

Conclusione degli autori: **il transfer factor medio non predice la singola
casa** — a potenziale quasi uguale le case distano ≥1.66 decadi, e fra il caso
più basso e il più alto ≥2.61 decadi (limite inferiore, per censura).

**Stato:** trascritto in `neznal2005_transfer_factor.csv`, ruolo
`validazione` dell'ampiezza dell'anello B. L'analisi dice che la banda
dichiarata dell'anello B (4.370 decadi) contiene la dispersione osservata, e
che l'ancora di Nazaroff (5e-3) cade dentro l'intervallo misurato.

---

### 7. Zhao et al. 2025 — 52 case con radon ORARIO e ricambio d'aria calcolato — SCARICATO E USATO

Dryad, DOI [10.5061/dryad.kh18932kr](https://doi.org/10.5061/dryad.kh18932kr),
pubblicato il 24 settembre 2025. Zhao, Martin, Khan, Chasar, Sonne, Withers,
Russell, Chan, Walker, Singer. Dati di corredo di uno studio su ventilazione
meccanica e qualita' dell'aria in case di nuova costruzione (Florida, South
Carolina, Georgia).

**Verifica: pagina Dryad letta e API Dryad interrogata il 2 settembre 2026.
SCARICATO a mano dal browser lo stesso giorno** (Dryad ha una sfida anti-bot
con proof-of-work: l'API risponde 401 e `file_stream` rende la pagina
«Validating…»). Sul disco, fuori da `data/`: 221 file, 325 MB. **Ispezionato
e usato**: è il braccio sud-est della misura della risposta alla ventilazione
(domanda V4). Le **52** case sono contate nel foglio
`FSEC HomeCharacteristics_Output_v3_clean_masked.xlsx`, non prese dalla pagina
di Dryad.

Confermato dall'API (`/api/v2/datasets/doi:10.5061/dryad.kh18932kr`):

- **licenza CC0-1.0** — dominio pubblico, nessun vincolo di riuso né obbligo di
  attribuzione. È la licenza migliore possibile per questo progetto.
- i dieci file della versione 397780, con dimensioni esatte: `README.md` (20 702 B),
  `Long-term_Radon.zip` (1 882 B), `IAQ_Activity_Monitoring.zip` (47 126 202 B —
  è qui che stanno le serie continue), `Airflow.zip` (8 404 130 B),
  `Occupant_survey.zip` (439 856 B), `Occupant_activity.zip` (274 631 B),
  `Home_Characteristics.zip` (121 210 B), `WHMV_Config.zip` (55 609 B),
  `IAQ_Sample.zip` (18 239 B), `DeltaQ.zip` (4 044 B).

**Come scaricarlo:** dal browser, sulla pagina Dryad. Il download da riga di
comando NON funziona — `/downloads/file_stream/<id>` risponde 403 e
`/api/v2/files/<id>/download` risponde 401 «must have current bearer token».
Il dato è libero, l'accesso automatico no.

Perche' e' il candidato piu' forte per la domanda V4:

- **52 case unifamiliari** costruite dal 2013, non una camera in scala;
- **radon continuo a intervalli di 60 minuti**, piu' campionatori passivi a
  lungo termine (>= 6 mesi) in `Long-term_Radon.zip`;
- **il tasso di ricambio d'aria complessivo, risolto nel tempo** — e' la
  variabile che manca a ogni altro candidato di questo elenco, ed e' l'ingresso
  che il bilancio indoor chiede (`ACH`);
- CO2, temperatura e umidita' relativa **al minuto** in piu' locali;
- portate misurate dei sistemi di ventilazione meccanica (`Airflow.zip`,
  `WHMV_Config.zip`), caratteristiche costruttive (`Home_Characteristics.zip`)
  e attivita' degli occupanti — cioe' un profilo di occupazione dichiarato,
  che l'esposizione cumulativa richiede;
- scaricabile liberamente, con README di 20 kB.

**L'unita' del radon: VERIFICATA il 2 settembre 2026, non assunta.** Il README
dichiara pCi/L per i soli campionatori passivi di lungo periodo, e tace
sull'unita' della serie CONTINUA. La differenza conta: fra pCi/L e Bq/m3 c'e' un
fattore 37. Invece di dedurla, e' stata misurata — le case che hanno ENTRAMBE
le misure permettono il confronto diretto fra la media della serie continua e
il valore integrato del campionatore passivo:

    rapporto mediano continua / lungo-periodo = 0.956   (n = 27 case)
    ventaglio: da 0.35 a 1.79

Se la serie continua fosse in Bq/m3 il rapporto sarebbe ~37. **E' pCi/L**, la
stessa unita' dei passivi — e il margine e' cosi' largo che nessuna delle
scelte di dettaglio lo puo' ribaltare.

La ricetta del confronto: `Radon_Level_pCi_L` contro la media di `Rn_IN1` sui
file settimanali della stessa casa.

**Altre cautele, non risolte:** il ricambio d'aria e' CALCOLATO col modello a
zona singola dell'ASHRAE dai coefficienti Delta-Q, non misurato con gas
tracciante — e' a sua volta un modello, con l'ipotesi di nessuna apertura
sostanziale di porte o finestre. Il suolo di Florida e Georgia non e'
vulcanico, quindi il dataset serve il ramo indoor del modello, non l'emanazione
dei tufi. Condizione d'uso del documento: riaprire la fonte, ricontrollare i
numeri, e solo allora trascrivere.

### 8. NRC/CCHT Canada 2025 — depressurizzazione controllata con radon orario

Frontiers in Public Health, «Ventilation approaches and radon control in
Canadian houses», 2025,
[DOI 10.3389/fpubh.2025.1569494](https://doi.org/10.3389/fpubh.2025.1569494)
(copia libera su [PMC12057567](https://pmc.ncbi.nlm.nih.gov/articles/PMC12057567/)).

**Verifica: pagina PMC letta il 2 settembre 2026.**

13 case occupate piu' 3 case sperimentali NRC, e prove controllate sui duplex
gemelli del Canadian Centre for Housing Technology. **Radon orario** su piano
basso e alto; differenze di pressione fra interrato e sotto-soletta e fra piani
ed esterno; **depressurizzazione imposta a -5, -10 e -20 Pa con un duct blaster,
su finestre di 48 ore, con la baseline appaiata**. Concentrazioni iniziali a
HRV spento 91-312 Bq/m3, mediana 175.

E' il disegno *before/after* che il paragrafo 6.2 della proposta chiede, ed e'
l'unico candidato che varia la PRESSIONE MOTRICE in modo controllato: e' la
prova diretta del termine `dP_bottom` e della rete d'ingresso.

**Limite dichiarato:** i dati grezzi non sono scaricabili — «included in the
article/Supplementary material, further inquiries can be directed to the
corresponding author». Il gas tracciante non e' stato usato nelle case
occupate causa pandemia, quindi il ricambio d'aria e' una CONFIGURAZIONE
dell'HRV, non una misura. Va trattato come le tabelle di Saelices: leggere il
supplementare, valutare la richiesta agli autori.

## Dataset esterni SCARICATI — fuori da `data/`

Il 2 settembre 2026 sono stati scaricati davvero, non solo censiti. Stanno in
una cartella di lavoro separata, fuori da `data/`, che ha il contratto stretto
degli header e dei sidecar.

**Il più importante: Building America / PNNL**, DOI
[10.25984/1971379](https://doi.org/10.25984/1971379) — **55 case**, 71 file
settimanali, **56 con la colonna del radon a 1 ora**. Colonne `Rn_IN1`,
`Rn_IN2`, `Rn_BR2/3` e **`Rn_BS1` in 27 file: il seminterrato**, cioè la
seconda zona del modello. **16 case misurate due volte**, una settimana con la
ventilazione di tutta la casa accesa e una spenta.

Misura fatta il 2 settembre 2026. Il numero dipende da quale sensore si legge:

| lettura | n | mediana | scende in | in controtendenza |
|---|---|---|---|---|
| `Rn_IN1`, piano abitato | 13 | **−30,2 %** | 12/13 | 309: +3,6 % |
| `Rn_IN1`, `Rn_BS1` solo di ripiego | 14 | **−27,9 %** | 13/14 | 309: +3,6 % |
| `Rn_BS1` se c'è, altrimenti `Rn_IN1` | 14 | **−32,8 %** | 11/14 | 212: +14,8 %, 309: +11,7 %, 310: +7,1 % |

Il «−32,8 % con tre case in controtendenza» è la TERZA riga, che mescola il
seminterrato con il piano abitato — due zone
diverse, il cui rapporto nella settimana spenta va da 1,01 a 4,59. Due delle
tre anomalie cambiano segno col punto di misura: la 212 fa −25,6 % al piano
abitato, la 310 fa −0,8 %. Solo la 309 resta in controtendenza in entrambe le
letture, e scende da +11,7 % a +3,6 %.

Chi cita uno di questi numeri scriva accanto quale colonna ha letto.

Il link diretto ai file (`data.openei.org/files/5845/...`) è stato ricavato
aprendo la pagina in un browser, perché OEDI la renderizza in JavaScript e
l'HTML non lo contiene.

Insieme a quello: tre abitazioni norvegesi a **5 minuti stanza per stanza**
(Mendeley), le scuole Aalto (scaricato, **non ancora aperto**), Autoflux con il
letto di esalazione di Cantabria, le sei camere di diffusione di València,
l'Etna e i quarant'anni di Banglazhang.

**I due Dryad sono stati presi a mano** (84 case in tutto, CC0): il sud-est umido
(Dryad kh18932kr: 52 case, 221 file) e l'Illinois freddo-umido (Dryad 4mw6m90qj:
32 case, 121 file). Le due cifre di case sono contate sui
fogli delle caratteristiche, `FSEC HomeCharacteristics_Output_v3_clean_masked.xlsx`
per il sud-est e `HomeCharacteristics_Output_ICRT_V4-2_clean_masked.xlsx` per
l'Illinois, non prese dalle pagine di Dryad. Dryad non è automatizzabile — non è
un paywall ma una sfida anti-bot con proof-of-work, che restituisce una pagina
«Validating...» al posto del file: si scarica dal browser, come descritto nella
scheda del candidato 7.

## Catalogo dei dataset esterni

Raccolti il 2 settembre 2026 interrogando le API di Zenodo e DataCite (che
rispondono senza autenticazione). **Ognuno e' stato aperto e il contenuto
controllato**; i byte NON stanno in `data/`, perche' quella cartella ha il
contratto stretto degli header di provenienza e dei sidecar.

Che cosa colmano, per lacuna del progetto:

| Lacuna | Dataset | Che cosa porta |
|---|---|---|
| **termine sorgente + metrologia** | Autoflux, CC-BY | radon con la sua incertezza, **contenuto d'acqua volumetrico**, temperatura del suolo, pressione atmosferica e pioggia, ~10 min su tre campagne; flusso con errore standard a 3 h; il letto di esalazione di riferimento dell'Universita' di Cantabria e le tarature ANSTO e INTE-UPC |
| **coefficiente di diffusione** | sei dataset di barriere (UPV Valencia, 2026) | esperimenti a **doppia camera**: concentrazione nella camera sorgente e in quella ricevente, spessore del materiale e specifiche complete della camera. Vernici, carte da parati, laminati, fosfogesso. Misurano direttamente cio' che il modello chiama `D_eff` |
| **radon esterno `C_out_Rn`** | torre ATTO | sei anni a 30 minuti, via Po-214 |
| **modulo sismico (EQ-SEI-11)** | BangLazhang / Tengchong | radon giornaliero su quasi 40 anni con pressione barometrica e precipitazione — il caso di prova della separazione sismico/meteo |
| **metrologia** | interconfronto europeo (heiDATA) | laboratori con principi di misura diversi a confronto, progenie compresa |
| **contesto vulcanico** | Etna, bordo del cratere | Rn-222 in aria, mag-ott 2019 |
| **domanda V4** | Zhao 2025 (Dryad, CC0) | 52 case, radon orario e ricambio d'aria risolto nel tempo |

**Scartato dopo il controllo:** `10.5281/zenodo.17288578` (scuole di Milano)
compariva nelle ricerche sul radon ma il CSV **non ha nessuna colonna radon** —
solo PM, CO2, ozono, temperatura e umidita'. E' il motivo per cui ogni voce di
questa tabella e' stata aperta prima di essere scritta.

**Ancora da prendere, troppo grandi per un download automatico:** le mappe
di flusso per l'Europa centrale 2021 (`10.5281/zenodo.15919530`, NetCDF da
~800 MB l'uno) e i tassi di esalazione di Aghdam (`10.5281/zenodo.13857771`,
879 MB).

## Candidati di contesto (medie di popolazione, per zona geologica)

Utili come vincolo distribuzionale, non come validazione puntuale:

| Fonte | Contenuto | Verifica |
|---|---|---|
| Ambrosino 2025 (Frontiers) | 67 scuole Campania: tufo 276±260, calcestruzzo 144±177 Bq/m³ | pagina letta |
| Quarto 2016 | 471 abitazioni Provincia di Napoli, statistiche per comune | pagina letta |
| Pugliese 2013 | 58 abitazioni a **Ischia**, dipendenza dal piano | pagina letta |
| ARPA Lazio + ISPRA 2013 | survey regionale Lazio con statistiche per comune (copre i comuni di Voltaggio) | pagina letta |
| Giustini 2022 | soil gas + indoor nei comuni vulcanici del viterbese (60–158 kBq/m³ soil gas) | pagina letta, parziale |
| Loffredo 2021 | 607 locali ASL Napoli 3, distribuzione | pagina letta |
| Vičanová (IPCM) | survey nazionale slovacco stratificato | pagina letta |
| Fujimoto 1997 / Sanada 2021 | survey nazionale Giappone (Hyogo AM 29.9 Bq/m³ per Kobe) | parziale |
| Brandýsová 2024 | verifica sperimentale delle radon priority areas slovacche | pagina letta |
| Holý 2016 (Nukleonika) | **non indoor**, ma flusso misurato al campus FMPI (15.0±1.8 mBq/m²/s) + Ra-226 + soil gas + profilo di umidità: colma le assenze 2 e parte della 8 del dataset Bratislava | pagina letta |

Nota bibliografica dal candidato Holý: il DOI della versione di rivista di
Eckertová è risultato da Crossref `10.1016/j.jenvrad.2026.107958` (JER 295:107958,
2026) — è registrato nel manifest e nel sidecar di `bratislava_eckertova.csv` con
la riserva, che resta, di riconfermarlo sulla pagina dell'editore. L'header del CSV
porta ancora `DOI: needs-source`, e va letto come «versione di rivista non ancora
consultata», non come «DOI ignoto».

## Prossimo passo proposto

Fatti: la camera di Gallese, il dataset IRON, le tabelle di Neznal (candidato 6) e di Saelices (candidato 1).

0. **Zhao 2025 su Dryad** (candidato 7, aggiunto il 2/9/2026) — **fatto, e ha
   scavalcato il punto 1 qui sotto**: portava le due cose che il progetto non aveva
   mai avuto insieme, radon orario su **52 edifici reali** e il **ricambio d'aria
   risolto nel tempo**. Scaricato, README letto, unità del radon verificata (pCi/L,
   dal confronto continua-contro-passivi, 0.956 su n = 27) e ricambio d'aria
   riconosciuto come CALCOLATO col modello a zona singola dell'ASHRAE dai Delta-Q,
   non misurato con gas tracciante. La domanda V4 è passata da un sito a decine:
   la risposta alla ventilazione è misurata sulle **tre** campagne (Dryad
   sud-est, Dryad Illinois, Building America/PNNL).
   Quello che resta aperto non è il dato ma una misura che nessuna delle tre
   campagne ha: il **radon esterno** contemporaneo. Senza, il segno stesso della
   conclusione V4 dipende da `C_out`, e l'analisi lo dichiara percorrendo tutta
   la banda invece di scegliere un valore.

1. **Saelices** (candidato 1) per il bilancio a due zone su edificio reale: partire
   dal PDF libero, valutare la richiesta dei dati orari agli autori. Era il
   passo più urgente finché il candidato 7 non è comparso: i tre parametri del
   bilancio a sorgente variabile descrivono una
   camera da 0,110 m3, e finché non c'è un edificio strumentato non esiste modo di
   sapere se quella forma funzionale regge fuori dal laboratorio.
2. **Neznal & Neznal** (candidato 6) per l'incertezza del transfer factor: sono
   **cinque** case, non quindici — il paper ne tabula cinque, quattro con suolo
   caratterizzato (candidato 6). Servono a quantificare quanto vale
   l'ancora di Nazaroff su cui è calibrata la rete d'ingresso, ma con n = 5 e
   ogni valore indoor **censurato** (un limite o un circa, mai un punto) quel
   che se ne ricava è un ordine di grandezza, non un'incertezza.
3. Solo dopo:  una verifica indipendente di tutto ciò che entra in `data/`.
