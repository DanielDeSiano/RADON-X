# Inventario dei dataset

Questo file descrive i dati reali presenti in `data/`. Ogni CSV porta in testa
le righe di provenienza — **con quattro eccezioni**: i file di `data/iron/`
sono pubblicati dagli autori come CSV puri, con l'intestazione delle colonne
sulla prima riga, e aggiungerci righe `#` significherebbe modificare il file
di terzi. Per quei quattro la provenienza sta **solo** nel sidecar
`.metadati.json`, che accompagna ogni CSV di `data/`. Qui c'e' l'indice, la
licenza, lo split e i limiti d'uso.

Regole applicate a tutti i file:

1. I valori sono trascritti alla lettera dalle fonti attraverso i documenti di validazione. Nessun valore e' stato ricalcolato, arrotondato o stimato; la sola trasformazione applicata e' la virgola decimale convertita in punto.
2. Dove la fonte non pubblica un numero, la cella e' vuota e l'assenza e' spiegata nell'header del CSV. Non e' mai stato inventato un valore.
3. Dove il DOI non compare nei documenti di validazione, l'header riporta
   `DOI: needs-source`.
4. Le uscite del modello RADON-X non sono in `data/`. I CSV contengono solo valori
   misurati, valori calcolati dagli autori delle fonti, e quantita' dedotte esplicitamente marcate. Mescolare i risultati del modello con i dati renderebbe impossibile qualsiasi confronto onesto.

Ogni colonna e' classificata nell'header con uno di questi tipi:

- `misurato` — misura pubblicata dalla fonte;
- `calcolato_dagli_autori` — valore prodotto dal modello o dalla formula degli autori della fonte;
- `dedotto` — quantita' ottenuta da valori pubblicati con un passaggio dichiarato
  (per esempio un'inversione o un'identita' algebrica); non e' un dato indipendente;
- `assunto` — scelta dichiarata di chi ha fatto la validazione;
- `non_pubblicato` — la fonte non riporta il numero.

---

## 1. `voltaggio2006_lazio_6siti.csv`

- Sito: Lazio settentrionale (Italia), sei siti su tre litotipi (vulcanico,
  epiclastico, unita' fluvio-lacustre Chiani-Tevere).
- Contenuto: Ra-226, coefficiente di emanazione di laboratorio `g`, porosita',
  saturazione al campionamento, densita' apparente, flusso di radon misurato in
  campo, stima diffusiva degli autori.
- Righe di dati: 6.
- Fonte: Voltaggio M., Tuccimei P. et al. (2006), *Environmental Geochemistry and
  Health* 28:541-551, Tabella 1, pp. 546-547. **DOI: needs-source.**
- Licenza: articolo su abbonamento Springer (accesso chiuso). I valori sono dati
  numerici di tabella.
- Strumento: needs-source. Periodo: needs-source.

- Limite: `g` e' misurato in laboratorio e non e' dichiarato se su campione
  essiccato. L'interpretazione secco/plateau usata nei documenti di validazione e'
  un'ipotesi, non un dato, e non e' registrata in questo file.
- Uso: questi sei siti, insieme ai sette di `zhuo2006`,
  sono i tredici di `risultati/section14_catena_anelli.py`, che prosegue la catena
  oltre il flusso fino a una concentrazione indoor. Lo split del file non ne risente e
  nessun parametro viene tarato qui: volume, ricambio d'aria e area di ingresso sono i
  valori predefiniti del modello originale, non misure di questi siti. La parte
  indoor di quel calcolo **non e' validabile**: Voltaggio pubblica il flusso dal
  suolo e nient'altro.

## 2. `bratislava_eckertova.csv`

- Sito: campus FMPI, Universita' Comenius, Bratislava (Slovacchia). Suolo ranker
  limoso-sabbioso.
- Contenuto: caratterizzazione del sito (Ra-226, Ra-228, emanazione di plateau,
  porosita', densita', tessitura, temperatura), statistiche quadriennali del flusso,
  lunghezza di diffusione osservata, medie dei modelli degli autori.
- Righe di dati: 31 (formato chiave-valore).
- Fonte: Eckertova T. et al. (2026), *Journal of Environmental Radioactivity*;
  testo usato: preprint SSRN id 5974883 (27/12/2025). **DOI di rivista: Crossref
  da' 10.1016/j.jenvrad.2026.107958 (JER 295:107958, 2026), da riconfermare sulla
  pagina dell'editore**; e' quanto registrano il manifest e il sidecar. Fonti di
  controllo: tesi Eckertova 2022; *Radiation Protection Dosimetry*
  198(9-11):771-777 (2022); *Nukleonika* 61(3):281-287 (2016).
- Licenza: preprint SSRN aperto; versione di rivista chiusa. I valori vanno
  riconfrontati con la versione pubblicata prima di essere citati.
- Strumento: camera d'accumulo automatica ISO 11665-7, due cicli da 6 ore al giorno;
  HPGe per il radio; laboratorio accreditato INGEO-ENVILAB per porosita' e densita'.
- Periodo: ottobre 2020 - giugno 2024.

- Celle vuote dichiarate: umidita' volumetrica, misurata ogni ora ma pubblicata solo
  in figura.
- Ambiguita' registrata: il coefficiente di emanazione 0.168 potrebbe appartenere
  alla localita' Horna Streda e non a Bratislava, per cui la tesi 2022 adotta 0.174.

## 3. `kobe_hosoda2009.csv`

- Sito: Kobe (Giappone), granito alterato del monte Rokko. Sei punti (1, 2, 3, 5, 6,
  7); il punto 4 e' escluso perche' privo di campioni di suolo.
- Contenuto: densita' secca e dei grani, porosita', Ra-226, numero di date, flusso di
  radon e di toron misurati, rapporto toron/radon.
- Righe di dati: 6.
- Fonte: Hosoda M. et al. (2009), *Journal of Radiation Research* 50(4):333-343,
  DOI 10.1269/jrr.08121, Tabelle 1 e 3.
- Licenza: open access, PDF libero su J-STAGE.
- Strumento: misura simultanea di radon e toron; dettaglio strumentale needs-source.
- Periodo: maggio, agosto, novembre 2007 e gennaio 2008.
- La validazione si puo fare per i punti 1 e 2;
  Gli autori stessi dichiarano che i punti del campus sono disturbati da
  tubazioni interrate e riporti compattati: un modello a colonna non puo'
  rappresentare una perdita laterale.

- Il confronto sul toron non e' utilizzabile: il Th-232 dei suoli non e' pubblicato. Per il nostro progetto non ha neanche senso tenerlo in considerazione.

## 4. `kobe_hosoda2009_per_data.csv`

- Contenuto: parametri per data di misura (saturazione, emanazione rivalutata a ogni data) e flusso calcolato dagli autori con l'equazione UNSCEAR.
- Righe di dati: 24.
- Fonte: come sopra, Tabella 3.

## 5. `santander_larucbed.csv`

- Sito: banco di esalazione metrologico del laboratorio LaRUC, Universita' di
  Cantabria (Spagna). Suolo di ex miniera d'uranio in vasca d'acciaio sigillata,
  spessore 0.165 m.
- Contenuto: Ra-226, emanazione misurata a secco, porosita', densita', spessore,
  saturazione, temperatura, flusso determinato in tre modi indipendenti.
- Righe di dati: 11 (formato chiave-valore).
- Fonte: Grossi C. et al. (2023), *Atmospheric Measurement Techniques* 16:2655-2672,
  DOI 10.5194/amt-16-2655-2023.
- Licenza: open access (Copernicus). Dati su Zenodo 7331337; licenza esatta del
  deposito needs-source.
- Strumento: enclosure (misura primaria); sistema Autoflux (n = 58); piu' la formula
  teorica del paper.
- Periodo: needs-source.
- Limite: il letto e' quasi secco (saturazione 0.06-0.07), quindi non testa la
  dipendenza dall'umidita'.

Geometria nota, incertezze metrologiche dichiarate, fondo sigillato. E' il tipo di misura da tenere fuori da qualsiasi taratura.

## 6. `bessines_ferry2000.csv`

- Sito: Bessines-sur-Gartempe (Francia), copertura di residui di lavorazione
  dell'uranio, due parcelle.
- Contenuto: banda di Ra-226, emanazione misurata a secco e con accumulo di 21
  giorni, flusso misurato come media annua, calcolo dell'autrice.
- Righe di dati: 2.
- Fonte: Ferry C. (2000), tesi Universite' Paris XI / rapporto CEA-R-5946, Tabella
  4-11; piu' rapporto IAEA XA0201260 per l'emanazione con accumulo. Nessun DOI.
- Licenza: PDF libero sull'archivio ASNR.
- Strumento: flusso in continuo per un anno; dettaglio needs-source.
- Periodo: febbraio 1998 - febbraio 1999.
- Cella vuota dichiarata: saturazione d'acqua, non pubblicata.

## 7. `huelva_banchi_riferimento.csv`

- Sito: laboratorio dell'Universita' di Huelva (Spagna). Due banchi metrologici a
  letto sottile in fosfogesso.
- Contenuto: spessore, massa, Ra-226, densita', emanazione misurata, esalazione di
  riferimento su 48 e 44 esperimenti.
- Righe di dati: 2.
- Fonte: Gutierrez-Alvarez I. et al. (2020), *Radiation Measurements* 133:106284
  (**DOI: needs-source**); emanazione da Lopez-Coto I. et al. (2014), *Journal of
  Hazardous Materials* 280:464-471 (**DOI: needs-source**).
- Licenza: green open access (UPCommons); il 2014 e' open access (idUS).
- Strumento: esalazione certificata, media di 48 e 44 esperimenti; i banchi RB sono
  corretti per l'effetto di camera chiusa.
- Periodo: needs-source; letti stagionati 1-3 mesi.

- Celle vuote dichiarate: saturazione d'acqua e porosita' dei banchi.

## 8. `huelva_prismi_e_campo.csv`

- Sito: deposito di fosfogessi di Huelva, zona 1, piu' prismi di laboratorio.
- Contenuto: parametri del materiale, flusso misurato, flusso calcolato dagli autori
  con il loro FEM RADEXH, tetto di produzione dedotto.
- Righe di dati: 3 (prisma 10 cm, prisma 20 cm, campo).
- Fonte: Castano-Casco E. et al. (2026), *Journal of Hazardous Materials* 507:141754,
  DOI 10.1016/j.jhazmat.2026.141754.
- Licenza: CC-BY, con materiale supplementare.
- Strumento: prismi in camera chiusa da 2.4 L; campo con due camere d'accumulo;
  carota con 13 sezioni fino a 88 cm.
- Periodo: needs-source.
- **Uso consentito: validazione per la riga di campo; solo esplorativo per le due
  righe di prisma.** I flussi dei prismi stanno il 29-36% sotto il tetto di
  produzione calcolato con l'emanazione del paper stesso: misura e FEM degli autori
  si riferiscono all'esperimento in camera chiusa, dove accumulo e retro-diffusione
  sopprimono il flusso netto. Non sono confrontabili con un calcolo di esalazione
  libera.

- Refuso registrato: l'intestazione della Tabella 9 della fonte riporta
  `Bq m-2 h-1` ma i valori sono coerenti solo come `Bq m-2 s-1`. Segnalato, non
  corretto.
- Nodo sull'emanazione, comune a tutto il gruppo Huelva: 0.15 (paper 2026, stato non
  dichiarato), 0.20 +- 0.04 (misura diretta 2014), circa 0.42 (se 0.15 fosse a secco
  e si applicasse la legge di umidita' del paper). Spread di un fattore 2.8. Nessuno
  dei tre valori e' stato scelto.

## 9. `nureg1980_colonne_vitro.csv`

- Sito: colonne di prova sigillate in laboratorio, 0.61 m di diametro, 0.61 m di
  sterili d'uranio Vitro. Due serie.
- Contenuto: Ra-226, coefficiente di diffusione calcolato dagli autori, flusso nudo
  misurato con incertezza, saturazione dedotta.
- Righe di dati: 14 (6 del Set 2, 8 del Set 1).
- Fonte: Rogers V.C. et al. (1980), NUREG/CR-1081 (FBDU-218-2), Tabelle 5-1 e 5-4,
  verificate su immagine di pagina. Nessun DOI; OSTI 5629662. Porosita' e densita'
  provengono dal rapporto gemello RAE-8810-6: giunzione fra rapporti, dichiarata.
- Licenza: rapporto governativo statunitense, libero su OSTI; stato formale di
  licenza needs-source.
- Strumento: canestri a carbone attivo tarati su standard di Ra-226.
- Periodo: 1980; date di misura needs-source.
- Non si può usare:
  Due ragioni distinte: 
  (a) il Set 2 e' circolare per dichiarazione della fonte stessa, che normalizza i Ra-226 ai flussi misurati; 
  (b) il Set 1 e' quasi saturo e la saturazione non e' pubblicata, quindi
  lo scarto non e' separabile dall'incertezza degli ingressi.
- La colonna `Sw_inverted_from_D_derived` NON e' un dato: e' l'inversione del D pubblicato attraverso la correlazione di Rogers & Nielson, cioe' la stessa chiusura del modello. Non va usata come ingresso indipendente.

## 10. `ranger_fattore_conversione.csv`

- Sito: provincia uranifera di Ranger, Territorio del Nord (Australia); clima
  tropicale a stagionalita' secco/umido.
- Contenuto: fattore di conversione RE-R (flusso diviso Ra-226) per classi di
  superficie, con le tre ancore pubblicate indipendenti e due siti nominati.
- Righe di dati: 8.
- Fonte: Internal Report 477 (Akber et al. 2004); tesi C.E. Lawrence, QUT (2005);
  Internal Report 616 (Bollhofer et al. 2013); Mason et al. 1982 e Lawrence et al.
  2009 come citati nei rapporti. Nessun DOI per i rapporti; DOI della tesi
  needs-source.
- Licenza: rapporti governativi australiani ad accesso libero.
- Strumento: camere d'accumulo di campo; dettaglio needs-source.
- Periodo: campagne 2002-2013.
- Porosita', densita' ed emanazione non sono mai misurate in questi rapporti: tre ingressi su sei mancano.
- Copre quasi quattro ordini di grandezza in flusso (18-81000 mBq m-2 s-1, cioe' 3,65
  decadi) e tre in Ra-226 (43-44000 Bq kg-1, 3,01 decadi). Nessuno dei due intervalli
  sta in questo CSV, che contiene solo i RE-R: vengono dai rapporti di origine.
- Celle vuote dichiarate: i RE-R individuali degli otto siti stagionali, e il valore
  assoluto per i letti sabbiosi di torrente (la fonte lo esprime come fattore fino a
  x10).

## 11. `vanderspoel1998_diffusivita.csv`

- Materiale: sabbia Ankersmit, 305 micrometri, porosita' 0.34, T = 293 K.
  Laboratorio, non sito di campo.
- Contenuto: tre ancore di diffusivita' misurata (secca, saturazione 0.79, satura) e
  tre punti della curva di emanazione contro umidita'.
- Righe di dati: 6.
- Fonte: van der Spoel W.H. (1998), tesi TU Eindhoven, DOI 10.6100/IR508320,
  sezione 8.4; il capitolo 8 corrisponde a *Health Physics* 77(2):163-177 (1999).
- Licenza: PDF libero su pure.tue.nl.
- Periodo: 1998.
- Uso consentito: test bloccato per la sola chiusura D(theta). La tesi **non**
  pubblica flussi tabulati, quindi non serve a validare la catena di flusso.
- Il valore di plateau dell'emanazione e' marcato `misurato_approssimato`: i documenti
  di validazione lo riportano come "attorno a 0.25" e va riletto sulla tesi.

## 12. `tomsk_ryzhakova.csv`

- Sito: Lagerny Sad, Tomsk (Russia), limo lessico.
- Contenuto: Ra-226, porosita', densita', contenuto d'acqua, emanazione, radon nel
  suolo a 35 e 70 cm, flusso in camera.
- Righe di dati: 9 (formato chiave-valore).
- Fonte: Ryzhakova N.K. et al., *Radiatsionnaya Gigiena* 13(2):57-64 (2020) e
  11(2):37-44 (2018); Stavitskaya et al., *AIP Conf. Proc.* 2101:020013 (2019);
  brevetto RU2239207C1 (2003) per emanazione e concentrazioni nei fori.
  **DOI: needs-source per tutti.**
- Licenza: articoli open access; brevetto pubblico.
- Strumento: Alfarad Plus, 64 misure (flusso 2018); anello tagliente (densita' e
  porosita'); fori di misura (2003).
- Periodo: 2003 (brevetto) e 2018 (flusso). I due gruppi di dati distano quindici anni e i punti non coincidono.


## 13. `zhuo2006_10sites.csv`

- Sito: dieci siti in Giappone e Corea, quattro stagioni.
- Contenuto: Ra-226, densita', porosita', emanazione a secco, saturazione,
  temperatura, flusso misurato (medie stagionali) e tre stime degli autori.
- Righe di dati: 10, 18 colonne.
- Fonte: Zhuo W., Iida T., Furukawa M. (2006), *Journal of Nuclear Science and Technology* 43(4):479-482, DOI 10.1080/18811248.2006.9711127, Tabelle 3 e 4, p. 481.
- Licenza: free access su Taylor & Francis.
- Strumento: accumulatore ventilato naturalmente (0.28 h-1).
- Periodo: aprile 2003 - aprile 2004.
- Uso: i sette siti in validazione sono, con i sei di
  `voltaggio2006`, i tredici di `risultati/section14_catena_anelli.py`. Vale la stessa
  nota della voce 1: la catena arriva fino a una concentrazione indoor, ma per questi
  siti nessuna misura indoor esiste, quindi quell'ultimo anello e' una predizione e non
  entra in nessuna metrica. Lo split del file non ne risente.


## 14. `hosoda2007_moisture_curve.csv`, `hosoda2007_calculated_curve.csv`, `hosoda2007_fig2_raw_scatter.csv`

- Materiale: suolo granitico alterato, Ibaraki (Giappone). Esperimento di laboratorio su lastra di 5 cm in vassoio d'acciaio.
- Contenuto: curva flusso-umidita' misurata (10 punti con barre d'errore), curva
  calcolata dagli autori (37 vertici), misure individuali non raggruppate (34 punti).
- Righe di dati: 10, 37, 34.
- Fonte: Hosoda M. et al. (2007), Journal of Nuclear Science and Technology 44(4):664-672, DOI 10.1080/18811248.2007.9711855, Figg. 2, 3 e 6(a).
- Licenza: accesso libero su Taylor & Francis.
- Metodo di estrazione: geometria vettoriale esatta dalla figura del PDF, con
  calibrazione lineare degli assi. Non e' una lettura a occhio, ma resta
  un'estrazione da figura e va dichiarata come tale.


---

## 15. `gallese2020_materiali.csv`, `gallese2022_parametri_modello.csv`

- Sito: materiali da costruzione vulcanici del Lazio caratterizzati in laboratorio
  (Tufo di Gallese, Tufo di Riano, Peperino Grigio, Peperino Rosato).
- Contenuto: porosita', Th-232, Ra-226, K-40, esalazione di Rn-222 e Rn-220 per
  unita' di superficie, con incertezze; piu' i parametri di modello del paper 2022
  (D, eta, rho, a(Ra-226), lambda, p) e la geometria della camera.
- Righe di dati: 4 e 13.
- Fonte: Lucchetti C. et al. (2020), *Nukleonika* 65(2):71-76, DOI
  10.2478/nuka-2020-0010, Tabella 2 p.74; Lucchetti, Galli, Tuccimei (2022),
  *Advances in Geosciences* 57:81-88, DOI 10.5194/adgeo-57-81-2022, Tabella 3 p.86
  e §2.1 p.82.
- Licenza: 2020 CC BY-NC-ND 3.0; 2022 CC BY 4.0. Entrambe lette verbatim sui PDF.

- Limite: la tabella dei parametri di modello del 2022 non pubblica incertezze;
  quelle di densita' e Ra-226 esistono solo nella tabella materiali del 2020.

## 16. `gallese2020_equilibri.csv`

- Sito: camera in scala costruita con blocchi di Tufo di Gallese, 62x50x35 cm
  interni, volume circa 0,110 m3, superficie di pareti esalanti 0,78 m2.
- Contenuto: sei condizioni sperimentali con Rn-222 e Rn-220 di equilibrio e
  intervallo di temperatura.
- Righe di dati: 6.
- Fonte: Lucchetti C. et al. (2020), Tabella 3 p.75.
- Strumento: AER PLUS (Algade) tarato alla camera radon dell'INGV per l'effetto
  dell'umidita' assoluta; RAD7 (Durridge) in modo Sniff.
- Licenza: CC BY-NC-ND 3.0.
- Le configurazioni sigillate stanno un fattore circa 2 sopra la predizione ottenuta con la E pubblicata, perche' sigillare la faccia esterna ridirige il flusso uscente verso l'interno; nello scenario 2E lo scarto sul test 2 e' −4,3 %.
- Limite: i test 1 (pareti nude) e 5b non sono confrontabili con una predizione a parete sigillata e servono solo alla stima delle perdite.
- Uso: l'equilibrio del test 1, 770 Bq/m3, e' preso da
  `risultati/indoor_sorgente_variabile.py` come concentrazione a riposo a parete nuda,
  che il paper 2022 non pubblica. E' un prestito fra due campagne e va contato come
  scelta dichiarata; la sua influenza sul risultato sta sotto lo 0,2 %.

## 17. `gallese2022_ach.csv`

- Sito: la stessa camera in scala, dodici esperimenti di ventilazione.
- Contenuto: portata d'aria, ACH (positivo immissione, negativo estrazione),
  riduzione relativa della concentrazione con incertezza, presenza del rivestimento.
- Righe di dati: 12.
- Fonte: Lucchetti, Galli, Tuccimei (2022), Tabella 2 p.85.
- Licenza: CC BY 4.0.
- L'esperimento 5 da solo e' stato usato per stimare l'ACH di fondo della camera (perdite), n0 = 0,0389 h⁻¹: e' **calibrazione** e non puo' mai essere citato come validazione. Gli esperimenti 1-4 sono quindi **predizioni pure** (validazione) e il modello li riproduce entro 1 sigma. Gli esperimenti 6-12 sono validazione e il modello li **falsifica** (z da 11,1 a 24,9): l'ho registrato come una falsificazione.
- Secondo uso. Gli stessi dodici esperimenti alimentano
  `risultati/indoor_sorgente_variabile.py`, che riscrive il bilancio con la sorgente
  dipendente dalla concentrazione interna e dalla differenza di pressione. La
  falsificazione della riga precedente resta: e' il punto di partenza di quel modulo.
  Per questo modello la contabilita' delle righe consumate e' diversa.
  Per il modello a sorgente variabile sono **calibrazione** gli esperimenti 6, 9 e 11,
  uno per parametro (`MAKEUP_COVERED_RATIO`, `REENTRY_BARE_H`, `MAKEUP_BARE_RATIO`), e
  n0 = 0,0389 h⁻¹ e' ereditato invariato da `gallese_box_model.py`, cioe' dalla riga 5.
  Righe che hanno fissato un parametro: quattro su dodici, la 5, la 6, la 9 e la 11. Il
  modulo dichiara nove predizioni, escludendo 6, 9 e 11 ma tenendo dentro la 5, il cui
  dato ha fissato n0. Il quarto coefficiente, `REENTRY_COVERED_H = 0`, e' stato
  stimato guardando i cinque punti del gruppo 1 (esperimenti 1-5) ed e'
  risultato nullo: non viene da una riga in particolare, ma quei cinque punti non erano
  ciechi. Ciechi in senso stretto restano il 7, l'8, il 10 e il 12.
- Risultati misurati sui dodici punti (`risultati/indoor_sorgente_variabile.py`): la
  sorgente costante sbaglia oltre dieci deviazioni su sette punti, il 6, 7, 8, 9, 10, 11
  e 12, con un massimo di 25,4; la sorgente variabile sta entro 2,9 deviazioni sulle
  nove predizioni, mediana 1,1. Sui
  quattro punti del gruppo 1 (parete rivestita, aria immessa) il modello nuovo va
  leggermente peggio del vecchio, e la ragione e' misurata: conta il fondo esterno di
  30 Bq/m3 che il vecchio trascura. Azzerato il fondo i due coincidono. Restano tutti
  entro 1,2 sigma.
- Il massimo 25,4 non contraddice l'11,1-24,9 della riga sopra. Sono due scritture dello
  stesso modello a sorgente costante: `gallese_box_model.py` lo valuta in concentrazione
  assoluta e include C_out = 30 Bq/m3, `indoor_sorgente_variabile.py` lo valuta in
  rapporto C(n)/C_0 e il fondo lo trascura.
- Il ruolo del file nel manifest (`radon_x/data_manifest.py`, che e' l'autorita' per la
  suddivisione) e' `calibrazione`. Il manifest assegna un ruolo per file, quindi non
  puo' esprimere che le righe consumate dipendono dal modello: la ripartizione riga per
  riga sta qui e nei due moduli di `risultati/`.
- I tre parametri stimati valgono per questa camera, con questo materiale e queste due
  configurazioni di parete. Non sono trasferibili a un edificio.


## 18. Serie IRON di Ischia

File: `Radon_Time_Serie_of_ISAR.csv`, `Radon_Time_Serie_of_ISCA.csv`,
`Radon_Time_Serie_of_ISMAR.csv`, `Meteo_Time_Serie_of_ISAR.csv`, tutti in `iron/`.

- Sito: isola d'Ischia. ISAR indoor (locale tecnico interrato), ISCA e ISMAR
  boreholes. **L'attribuzione dei codici stazione ai toponimi e' una deduzione
  dalle coordinate della Tabella 1 del paper, non un dato pubblicato.**
- Contenuto: serie continue di radon (colonna grezza e colonna corretta per
  umidita') e, per ISAR, temperatura e umidita' relativa dell'aria.
- Righe di dati: 12 465 (ISAR), 5 513 (ISCA), 973 (ISMAR), 12 465 (meteo).
- Fonte: dataset figshare DOI 10.6084/m9.figshare.27292554 (24 creator, 2025);
  articolo Piersanti A. et al. (2025), *Scientific Data* 12:415, DOI
  10.1038/s41597-025-04735-0. Attenzione: i due titoli sono diversi.
- Strumento: monitor continui serie AERC, modello per stazione nella colonna
  `Rn Instrument`.
- Licenza: **CC BY 4.0**, verificata su figshare e DataCite. Attribuzione richiesta.
- Provenienza e controllo qualita': `iron/README.md` e `iron/QC.md`. I file scaricati **non vanno modificati**: sono la fonte, con MD5 registrati.
- I boreholes sono a testa pozzo sotto i 2 m e non sono il gas del suolo a profondita' fissa usato per il potenziale geogenico, quindi un rapporto indoor/suolo calcolato su questi file **non e' confrontabile** con il rapporto di Nazaroff; e per la stazione indoor non esiste alcun metadato dell'edificio (volume, ricambio d'aria, fondazione).
- Limiti: quattro buchi oltre 24 h in ISAR (max 5,1 giorni); ISCA ha fermi di 265 e 153 giorni ai cambi strumento; righe duplicate con `Source=tool/website` quantificate in `QC.md`, una deduplicazione ingenua non basta; ISCA ha un picco di 391 572 Bq/m3 e cinque righe a 99 900 esatti.
- Sono le **prime misure indoor entrate nel progetto**.

## 19. `neznal2005_transfer_factor.csv`

- Sito: cinque case reali in Repubblica Ceca (Lounovice A e B, Potucky, Horni
  Slavkov A e B), quattro con suolo caratterizzato dal metodo ceco uniforme.
- Contenuto: coppie (radon nel gas del suolo al terzo quartile, permeabilita'
  k75) → radon indoor, con **ogni valore indoor censurato** (un limite o un
  circa, colonna `qualifier`), mai un punto.
- Fonte: Neznal & Neznal, atto di convegno RADON v.o.s. (misure 2004-2005,
  progetto SUJB 2/04/V), PDF pubblico su radon-vos.cz, **letto per intero il
  31/08/2026**. Nessun DOI; licenza non dichiarata nel PDF.
- Split: **validazione dell'ampiezza dell'anello B**, mai calibrazione (N=4 e
  tutto censurato). L'analisi da' una dispersione del fattore di trasferimento
  ≥2,61 decadi fra le quattro case (≥1,66 a potenziale quasi uguale), contenuta
  nella banda dichiarata di 4,370; l'ancora di Nazaroff (5e-3) cade dentro
  l'intervallo osservato.

## 20. `saelices2019_fasi.csv`

- Sito: casa pilota di Saelices el Chico (Salamanca), ex impianto uranifero
  ENUSA; Ra-226 del suolo ~1600 Bq/kg (citato da Frutos 2011), radon esterno
  ~300 Bq/m3.
- Contenuto: le nove fasi di prova con date (Tabella 1) e le concentrazioni per
  fase e zona con le riduzioni (Tabella 2): casa chiusa 54 625 Bq/m3 nel
  seminterrato e 26 421 al piano terra; depressurizzazione attiva fino a
  662/328 (riduzione 99 %). In testa: correlazioni meteo dichiarate dagli
  autori e pressione al pozzetto (−55 Pa, fase 6).
- Fonte: Fuente et al. 2019, *Sci. Total Environ.* 695:133746, DOI
  10.1016/j.scitotenv.2019.133746; trascritto dal preprint arXiv:2403.15392,
  **letto per intero il 31/08/2026**. Le serie orarie esistono solo in figura
  e non sono trascritte.
- Split: **solo esplorativo**: predire queste fasi richiede un modello indoor a
  due zone con depressurizzazione del suolo, e nessun codice del progetto lo fa.
  E' il bersaglio naturale per un modello di quel tipo.

---
## Il manifest verificabile

- un CSV aggiunto a `data/` senza una voce nel manifest **non puo' essere usato**: `radon_x/data_manifest.py` lo rifiuta con un errore, non passa inosservato.
- un'assenza e' dichiarata (`NotApplicable` con la ragione, oppure `NeedsSource` con cosa cercare e dove), mai dedotta da un silenzio: oggi **14 campi** sono `NeedsSource`.

## Split complessivo

I dati devono essere divisi in calibrazione e validazione. Lo split e' motivato dal tipo di misura, non dal risultato, ed e' registrato nel manifest.

| Insieme | File | 
|---|---|
| Test bloccato | 3 |
| Validazione | 8 |
| **Calibrazione** | **1** (`gallese2022_ach.csv`, righe degli esperimenti 5, 6, 9, 11) |
| Solo esplorativo | 12 |
| Parametri di ingresso | 2 |

| Insieme | File | Motivazione |
|---|---|---|
| **Test bloccato** (3) | `santander_larucbed.csv`, `huelva_banchi_riferimento.csv`, `vanderspoel1998_diffusivita.csv` | Banchi metrologici sigillati e misure di laboratorio con geometria nota e incertezza dichiarata. Sono l'unico gruppo in cui uno scarto e' attribuibile al modello e non agli ingressi. Devono restare non toccati da qualunque taratura futura. |
| **Validazione** (8) | `voltaggio2006_lazio_6siti.csv`, `bratislava_eckertova.csv`, `kobe_hosoda2009.csv` + `kobe_hosoda2009_per_data.csv` (validazione ai soli punti 1 e 2; il file per data resta solo esplorativo) (punti 1 e 2), `zhuo2006_10sites.csv` (sette siti), `ranger_fattore_conversione.csv` (solo RE-R), `gallese2020_equilibri.csv` (termine sorgente indoor), `neznal2005_transfer_factor.csv` (ampiezza dell'anello B) | Siti di campo con parametri pubblicati, piu' i due bersagli indoor. Lo scarto e' interpretabile ma resta convoluto con l'incertezza degli ingressi. |
| **Calibrazione** (1) | `gallese2022_ach.csv` (righe degli esperimenti 5, 6, 9, 11) | Unico file in calibrazione del progetto. Dalla riga 5 viene il ricambio d'aria di fondo della camera, n0 = 0,0389 per ora, che e' un parametro stimato e non puo' mai essere citato come validazione. Gli esperimenti 1-4 restano previsioni pure, e 6-12 sono validazione che il modello a sorgente costante non supera (z da 11,1 a 24,9): registrata come falsificazione, non rimossa. Le righe 6, 9 e 11 fissano i tre parametri del modello a sorgente variabile (`risultati/indoor_sorgente_variabile.py`), che eredita n0 dalla riga 5. Le ragioni sono nella voce 17 qui sopra. |
| **Solo esplorativo** (12) | `kobe_hosoda2009.csv` (punti 3, 5, 6, 7), `bessines_ferry2000.csv`, `nureg1980_colonne_vitro.csv`, `tomsk_ryzhakova.csv`, `huelva_prismi_e_campo.csv` (riga di campo: validazione; righe prisma: solo esplorativo), i tre `hosoda2007_*.csv`, `zhuo2006_10sites.csv` (siti A, G, I), `saelices2019_fasi.csv`, i quattro file di `iron/` | Circolarita' dichiarata, incoerenza interna alla fonte, disturbo antropico noto, ingressi mancanti, oppure — per Saelices e IRON — nessun codice del progetto che possa predirli oggi. Utilizzabili per diagnostica, mai per una metrica di accuratezza. |
| **Parametri di ingresso** (2) | `gallese2020_materiali.csv`, `gallese2022_parametri_modello.csv` | Caratterizzazione del materiale e parametri di modello del paper. Non sono un bersaglio di confronto. |

Il conteggio fra parentesi e' quello dei **file** assegnati dal manifest, ed e' il
motivo per cui alcuni file compaiono in due righe con un qualificatore di riga: il
manifest assegna un ruolo per file, la ripartizione riga per riga vive qui e nei
moduli di `risultati/`.

---

## Cosa manca ancora

Elenco dei dati che il progetto NON possiede. Serve a impedire che un'assenza venga riempita con una stima.

1. **Nessun edificio con i metadati che il bilancio chiede.** Misure indoor in `data/`
   ce ne sono, e da fine agosto ce ne sono anche **in edifici veri**: la camera in
   scala di Gallese (voci 15-17: concentrazioni di equilibrio, ACH imposto e noto,
   riduzioni misurate), la serie della stazione ISAR di Ischia (voce 18), le cinque
   case ceche di Neznal (voce 19) e la casa pilota di Saelices el Chico (voce 20).
   Cio' che manca non e' piu' «un edificio»: sono i metadati che rendono confrontabile
   un edificio.
   - **Volume e ricambio d'aria**: il ricambio d'aria misurato esiste solo per la
     camera da 0,110 m3. Per ISAR, per le case ceche e per Saelices non c'e' ne'
     volume ne' ricambio d'aria ne' fondazione documentati.
   - **Censura**: in Neznal ogni valore indoor e' un limite o un circa, mai un punto
     (N = 4 con suolo caratterizzato), quindi vale per l'ampiezza dell'anello B e non
     per il suo centro.
   - **Predicibilita'**: Saelices porta le due zone e la depressurizzazione del suolo
     accesa e spenta (quattro fasi su nove: una passiva e tre attive), con la
     pressione media al pozzetto, −55 Pa in fase 6. Il **campo** di pressione sotto la
     soletta gli autori lo misurano con otto sensori, ma non e' trascritto qui: in
     `data/` c'e' l'effetto della depressurizzazione, non la sua mappa. E nessun
     codice del progetto sa oggi predire quelle fasi: il file e' `solo esplorativo`
     per questo, non per un difetto del dato.
   Restano bibliografia la casa IRSN sullo stesso comune e materiale del dataset 6
   (Bessines) e la casa di Celleno: fonti, livelli di verifica e ordine proposto in
   `INDOOR_CANDIDATI.md`.
   Il compartimento indoor sta nel motore (`radon_x/core/indoor.py`) e una
   concentrazione indoor esce da ogni simulazione del motore e dai tredici siti di
   `risultati/section14_catena_anelli.py`. Sono numeri che nessun file di `data/`
   puo' contraddire o confermare. Manca anche la **conduttanza della soletta**, che
   quel calcolo usa e che nessun dataset riporta: chi non la fornisce ottiene il
   limite superiore, dichiarato come tale.
2. **Umidita' di campo contestuale ai flussi.** E' l'assenza piu' pesante e si ripete
   su quasi tutte le fonti: Bratislava (misurata ogni ora, pubblicata solo in
   figura), NUREG 1980 (solo l'intervallo complessivo 9-27% in peso), Bessines (non
   pubblicata), Voltaggio (solo la saturazione al campionamento, non una serie),
   Ranger (solo un intervallo letto da figura). Senza umidita' contestuale ogni
   confronto di flusso richiede o un'assunzione o un'inversione model-dipendente.
3. **Serie temporali orarie.** Il gruppo di Bratislava dichiara di fornire i dati
   orari di flusso, umidita' e pressione su richiesta. Senza di essi non si puo'
   calcolare un NSE giornaliero confrontabile con la Tabella 1 del loro paper, ne'
   valutare i transitori che generano gli estremi 2.2 e 29.7 mBq m-2 s-1.
4. **Th-232 dei siti.** Manca sia a Kobe sia in Voltaggio.
5. **Emanazione misurata su parte dei siti.** Ranger non misura l'emanazione (ne'
   porosita' ne' densita'). A Huelva l'emanazione ha tre valori possibili in un
   fattore 2.8. A Voltaggio non e' dichiarato se `g` sia misurato su campione
   essiccato.
6. **Permeabilita' misurata.** Nessun dataset la riporta. Nei documenti di
   validazione e' sempre un valore di riferimento del litotipo.
7. **Abbinamento riga-data a Kobe.** Il paper non lo esplicita; il confronto resta
   per punto.
8. **DOI mancanti.** Voltaggio 2006, Gutierrez-Alvarez 2020, Lopez-Coto 2014, i tre
   lavori di Tomsk, la tesi Lawrence. Vanno letti sulle fonti, non ricostruiti.
9. **Metadati strumentali e periodi.** Marcati `needs-source` in Voltaggio, Kobe,
   Santander, Bessines, Huelva, Ranger. Vanno letti sulle fonti.
10. **Dati di calibrazione dedicati.** Non ce n'e' nessuno. Le quattro righe che oggi
    fissano un parametro stanno tutte in `gallese2022_ach.csv`: la 5 per il ricambio
    d'aria di fondo della camera, la 6, la 9 e la 11 per i tre coefficienti del
    bilancio a sorgente variabile. Vengono dalla stessa campagna su cui poi si valida,
    quindi la separazione fra calibrazione e validazione vive nell'analisi e non nei
    dati.
    Serve una misura indipendente del ricambio di fondo e, per i tre coefficienti, una
    seconda camera o un secondo materiale. Ogni parametro che il modello acquisira' in
    futuro va tarato su dati nuovi, non prelevati dagli insiemi di validazione o di
    test.
