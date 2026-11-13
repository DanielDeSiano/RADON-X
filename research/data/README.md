# Politica sui dati

Nessun set di dati scientifici è stato fornito insieme ai due codici Python originali. I set di dati presenti in questa directory sono stati raccolti separatamente dalla letteratura pubblicata e trascritti dai documenti di validazione. Si veda il file `DATASETS.md` per l’inventario: una voce per file, con fonte, DOI, licenza, strumento, periodo, uso consentito e la suddivisione dichiarata
tra calibrazione e validazione.

## Requisiti per ogni set di dati in questa directory

Ogni set di dati deve indicare provenienza, licenza, unità di misura, informazioni sui sensori, timestamp, metadati di calibrazione e una suddivisione dichiarata tra derivazione e validazione. I dati sintetici devono essere contrassegnati come tali sia nei nomi dei file che nei metadati e non devono mai essere presentati
come validazione indipendente.

## Regole applicate ai file attualmente presenti

1. Ogni file CSV contiene un’intestazione di provenienza costituita da righe `#`: citazione completa con DOI, tabella o pagina di origine, licenza o stato di accesso, strumento, periodo, unità di misura di ogni colonna e indicazione se ciascuna colonna sia misurata, calcolata dagli autori originali o derivata.
2. I valori sono trascritti alla lettera. Nulla viene ricalcolato, arrotondato o stimato. Le virgole decimali e i separatori delle migliaia presenti nei documenti di origine vengono convertiti nella semplice notazione con punto decimale; si tratta di una modifica di formato, non di un cambiamento di valore.
3. Quando una fonte non pubblica un numero, la cella viene lasciata vuota e l’omissione viene spiegata nell’intestazione. Non viene mai inventato alcun valore.
4. Quando un DOI non compare nei documenti di convalida, l’intestazione riporta `DOI: needs-source`. I DOI non vengono mai ricostruiti a caso.



## Vocabolario dei tipi di colonna

Le intestazioni classificano ciascuna colonna come: `misurato` (misurato), `calcolato_dagli_autori`
(calcolato dagli autori della fonte), `dedotto` (dedotto da valori pubblicati attraverso un passaggio dichiarato; non indipendente), `assunto` (un’ipotesi dichiarata formulata durante la validazione), `non_pubblicato` (la fonte non lo riporta).

## Suddivisione

La suddivisione dichiarata è contenuta nel file `DATASETS.md`. In breve: i banchi metrologici sigillati costituiscono un set di test bloccato, i siti sul campo rappresentano la validazione.  ( cioè con i siti sul campo farò la validazione )

L’autorità sulla copertura e sulla suddivisione è però il manifest leggibile da macchina,
`radon_x/data_manifest.py`: assegna un ruolo a ogni file, dichiara i campi mancanti
e rifiuta con un errore un CSV che entri in questa directory senza provenienza. `DATASETS.md`
resta il riferimento per la descrizione e per i limiti. Dove i due divergessero, vale il
manifest.

Un ruolo vale per file. Quando due modelli diversi consumano righe diverse dello stesso
file — è oggi il caso di `gallese2022_ach.csv` — la contabilità riga per riga sta in
`DATASETS.md`, voce 17, perché il manifest non ha modo di esprimerla.
