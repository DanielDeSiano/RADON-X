# Il motore

10.507 righe in 48 file: il motore, i suoi controlli e gli script dei risultati.
Riscrittura del modello di partenza sul ramo diffusivo, con il compartimento
indoor accoppiato.
Servono `numpy` e `scipy`. Dalla cartella:

    python -c "from radon_x.config import ProjectConfig; \
               from radon_x.unione_delle_parti import run_core_simulation; \
               r = run_core_simulation(ProjectConfig.from_json('examples/ischia_demo.json')); \
               print(r.surface_flux_bq_m2_s, r.soil_mean_bq_m3, r.indoor_bq_m3)"

Costa meno di un centesimo di secondo. Esce:

| grandezza | valore |
|---|---|
| flusso superficiale [Bq m⁻² s⁻¹] | 0,24563050774983616 |
| concentrazione media nel suolo [Bq m⁻³] | 137884,30226324478 |
| concentrazione interna [Bq m⁻³] | 7919,830462963914 |

Questi tre numeri non si muovono: sono il riferimento contro cui ogni modifica
viene confrontata.

## Com'è fatto

    radon_x/core/          il motore, 11 moduli
      solver.py            volumi finiti, θ-metodo adattivo
      analytic.py          quattro soluzioni in forma chiusa
      sources.py           produzione da Ra-226 ed emanazione
      diffusion.py         diffusività effettiva
      partition.py         partizione aria-acqua
      boundaries.py        condizioni al contorno
      outputs.py           flusso di superficie, bilancio della colonna
      indoor.py            ingresso e bilancio dell'aria interna
      dual_porosity.py     matrice e frattura
      decay.py             decadimento e progenie
      units.py             conversioni, in un posto solo
    radon_x/numerics/      solutore tridiagonale, controllo del passo
    radon_x/               costanti, errori tipizzati, configurazione, semi
    unione_delle_parti.py  monta i pezzi ed esegue la simulazione
    radon_x/dynamics/      l'aria interna a più zone, la progenie, le forzanti
    tests/                 205 controlli sul motore
    risultati/             gli script che producono i risultati
    data/                  i dataset, ognuno con la sua provenienza
    examples/              la configurazione del sito di prova
    radon_suolo_parte1_finale.py   il modello di partenza, com'era

## Due cose da guardare per prime

`radon_x/core/analytic.py` — quattro soluzioni in **forma chiusa** contro cui il numerico
deve convergere. È l'unico controllo che non dipende da nessun dato: se il
solutore sbaglia, lì si vede subito.

Il **bilancio della colonna chiude all'arrotondamento**: produzione più flusso al
fondo, meno decadimento e accumulo, lascia un residuo di 4,4·10⁻¹⁵ sul flusso di
faccia. Non è imposto, esce dalla discretizzazione conservativa.

## Che cosa non fa

Ramo diffusivo puro. Non contiene advezione barometrica, continuo fratturato in
transitorio, adsorbimento, accessibilità dei pori, isteresi dell'emanazione né un
campo di umidità che evolve: quelli restano nel modello di partenza,
`radon_suolo_parte1_finale.py`, che sta nella radice. Il Rn-220 è
fuori perimetro e vale `NaN`, che è un'assenza dichiarata, non zero.
