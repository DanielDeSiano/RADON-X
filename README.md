# RADON-X

Sito statico per il modello di trasporto del radon di Daniel De Siano.

## Avvio locale

```sh
npm ci
npm run build
npx http-server dist -a 127.0.0.1 -p 8766
```

## GitHub e Vercel

Importa il repository in Vercel. Se pubblichi la cartella come parte del repository
esistente, imposta **Root Directory** a `radon-vercel`. Se la cartella è la radice
del nuovo repository, lascia la Root Directory predefinita. Il file `vercel.json`
imposta installazione, build e directory di output. Non servono variabili
d'ambiente. Vercel pubblica soltanto `dist/`.

Il build Node scarica le wheel WebAssembly versionate di NumPy, SciPy e OpenBLAS
e ne verifica gli hash. Il browser le riceve da Vercel e le carica al primo run
in un Web Worker. Il calcolo utilizza il motore Python originale tramite Pyodide.
La distribuzione del runtime occupa circa 38 MB prima della compressione HTTP.

## Provenienza

- `research/radon_x/`, `research/data/`, `research/examples/`,
  `research/risultati/` e i test provengono da *Radon codice finale.zip*.
- `research/volcanic_radon_model_v11_finale.py` è la versione pubblica di Daniel
  al commit `ace22d4cafc77bd78f392aa8d01ae3bccd2a7ad1`. Il simulatore web usa il
  ramo modulare diffusive dello ZIP, come il laboratorio locale.
- `site/` contiene interfaccia, animazione di 15 secondi e profilo autore.
  L'animazione segue il flusso a scena continua di Bang Motion, con GSAP
  distribuito insieme al sito.

Prima di rendere pubblici il repository e il sito, rivedere con Daniel la scheda
autore e le condizioni di redistribuzione indicate per ciascun dataset in
`research/data/DATASETS.md` e nei file di metadati. Il CV originale e i recapiti
privati sono esclusi dal progetto.
# RADON-X
