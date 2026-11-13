from __future__ import annotations

import math
import random
from dataclasses import dataclass

from section14_dataset import PREDICTORS, Site, groups, load_sites

ANELLO = "A1"

UNITA_BERSAGLIO = "Bq m-2 s-1"


@dataclass(frozen=True)
class Metrics:

    n: int
    mbe: float
    sd: float
    rmse: float
    f2: float
    max_abs: float

    def line(self, label: str, parameters: str) -> str:
        return (
            f"{label:<34}{parameters:>12}{self.n:>5}{self.mbe:>+9.3f}"
            f"{self.sd:>8.3f}{self.rmse:>8.3f}{100 * self.f2:>7.0f}%{self.max_abs:>8.3f}"
        )


def metrics(predicted: list[float], measured: list[float]) -> Metrics:
    residuals = [
        math.log10(p) - math.log10(m) for p, m in zip(predicted, measured) if p > 0 and m > 0
    ]
    n = len(residuals)
    if n == 0:
        raise ValueError("no usable pair: a non-positive flux cannot be compared in log space")
    mbe = sum(residuals) / n
    sd = (sum((r - mbe) ** 2 for r in residuals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    rmse = (sum(r * r for r in residuals) / n) ** 0.5
    f2 = sum(1 for r in residuals if abs(r) <= math.log10(2.0)) / n
    return Metrics(n, mbe, sd, rmse, f2, max(abs(r) for r in residuals))


HEADER = (
    f"{'modello':<34}{'parametri':>12}{'n':>5}{'MBE':>9}{'SD':>8}{'RMSE':>8}{'F2':>8}{'|max|':>8}"
)


def physical(sites: list[Site], use_emanation_state: bool = True) -> list[float]:
    return [s.physical_flux_bq_m2_s(use_emanation_state=use_emanation_state) for s in sites]


def _design(sites: list[Site], intercept: bool = True) -> list[list[float]]:
    rows = []
    for s in sites:
        row = [math.log10(v) for v in s.features()]
        rows.append(([1.0] + row) if intercept else row)
    return rows


def _solve_least_squares(x: list[list[float]], y: list[float]) -> list[float]:
    k = len(x[0])
    a = [[sum(r[i] * r[j] for r in x) for j in range(k)] + [sum(r[i] * t for r, t in zip(x, y))] for i in range(k)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular design matrix: the predictors are collinear")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(k):
            if r == col:
                continue
            factor = a[r][col] / a[col][col]
            for c in range(col, k + 1):
                a[r][c] -= factor * a[col][c]
    return [a[i][k] / a[i][i] for i in range(k)]


def fit_loglinear(sites: list[Site]) -> list[float]:
    y = [math.log10(s.measured_flux_bq_m2_s) for s in sites]
    return _solve_least_squares(_design(sites), y)


def predict_loglinear(coefficients: list[float], sites: list[Site]) -> list[float]:
    return [10.0 ** sum(c * v for c, v in zip(coefficients, row)) for row in _design(sites)]


def fit_mean_ratio(sites: list[Site]) -> float:
    ratios = [
        math.log10(s.measured_flux_bq_m2_s / p)
        for s, p in zip(sites, physical(sites))
        if p > 0
    ]
    return sum(ratios) / len(ratios)


def fit_residual_linear(sites: list[Site]) -> list[float]:
    y = [
        math.log10(s.measured_flux_bq_m2_s) - math.log10(p)
        for s, p in zip(sites, physical(sites))
    ]
    return _solve_least_squares(_design(sites), y)


def predict_residual_linear(coefficients: list[float], sites: list[Site]) -> list[float]:
    corrections = [sum(c * v for c, v in zip(coefficients, row)) for row in _design(sites)]
    return [p * 10.0**d for p, d in zip(physical(sites), corrections)]


def fit_pinn(
    sites: list[Site],
    hidden: int = 8,
    epochs: int = 4000,
    physics_weight: float = 1.0,
    seed: int = 0,
) -> tuple[object, int]:
    import torch

    torch.manual_seed(seed)
    features = torch.tensor(_design(sites, intercept=False), dtype=torch.float64)
    observed = torch.tensor(
        [[math.log10(s.measured_flux_bq_m2_s)] for s in sites], dtype=torch.float64
    )
    physics = torch.tensor([[math.log10(p)] for p in physical(sites)], dtype=torch.float64)

    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1e-12)

    net = torch.nn.Sequential(
        torch.nn.Linear(len(PREDICTORS), hidden, dtype=torch.float64),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1, dtype=torch.float64),
    )
    parameters = sum(p.numel() for p in net.parameters())
    optimiser = torch.optim.Adam(net.parameters(), lr=0.01)

    for _ in range(epochs):
        optimiser.zero_grad()
        predicted = net((features - mean) / std)
        loss = ((predicted - observed) ** 2).mean()
        if physics_weight > 0:
            loss = loss + physics_weight * ((predicted - physics) ** 2).mean()
        loss.backward()
        optimiser.step()

    return (net, mean, std), parameters


def predict_pinn(trained: object, sites: list[Site]) -> list[float]:
    import torch

    net, mean, std = trained
    features = torch.tensor(_design(sites, intercept=False), dtype=torch.float64)
    with torch.no_grad():
        predicted = net((features - mean) / std)
    return [10.0 ** float(v) for v in predicted.flatten()]


def leave_one_out(sites: list[Site], fit, predict) -> list[float]:
    out = []
    for index in range(len(sites)):
        training = sites[:index] + sites[index + 1 :]
        model = fit(training)
        out.append(predict(model, [sites[index]])[0])
    return out


def leave_one_source_out(sites: list[Site], fit, predict) -> list[float]:
    campagne = groups(sites)
    if len(campagne) < 2:
        raise ValueError(
            f"il leave-one-source-out richiede almeno due campagne, ne ha {len(campagne)}: "
            f"togliendo l'unica non resta nulla su cui addestrare"
        )
    out: list[float] = [0.0] * len(sites)
    for indices in campagne.values():
        held = set(indices)
        training = [s for i, s in enumerate(sites) if i not in held]
        model = fit(training)
        for index in indices:
            out[index] = predict(model, [sites[index]])[0]
    return out


def fit_scaled_physics(sites: list[Site]) -> list[float]:
    y = [math.log10(s.measured_flux_bq_m2_s) for s in sites]
    x = [[1.0, math.log10(p)] for p in physical(sites)]
    return _solve_least_squares(x, y)


def predict_scaled_physics(coefficients: list[float], sites: list[Site]) -> list[float]:
    c0, c1 = coefficients
    return [10.0 ** (c0 + c1 * math.log10(p)) for p in physical(sites)]


def fit_physics_plus_saturation(sites: list[Site]) -> list[float]:
    y = [math.log10(s.measured_flux_bq_m2_s) for s in sites]
    x = [
        [1.0, math.log10(p), math.log10(s.saturation)]
        for s, p in zip(sites, physical(sites))
    ]
    return _solve_least_squares(x, y)


def predict_physics_plus_saturation(
    coefficients: list[float], sites: list[Site]
) -> list[float]:
    c0, c1, c2 = coefficients
    return [
        10.0 ** (c0 + c1 * math.log10(p) + c2 * math.log10(s.saturation))
        for s, p in zip(sites, physical(sites))
    ]


@dataclass(frozen=True)
class Interval:

    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_rmse(
    predicted: list[float],
    measured: list[float],
    draws: int = 2000,
    seed: int = 0,
) -> Interval:
    residuals = [
        math.log10(p) - math.log10(m)
        for p, m in zip(predicted, measured)
        if p > 0 and m > 0
    ]
    n = len(residuals)
    if n == 0:
        raise ValueError("no usable pair: a non-positive flux cannot be compared")
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [residuals[rng.randrange(n)] for _ in range(n)]
        values.append((sum(r * r for r in sample) / n) ** 0.5)
    values.sort()
    point = (sum(r * r for r in residuals) / n) ** 0.5
    return Interval(point, values[int(0.025 * draws)], values[int(0.975 * draws) - 1])


def paired_rmse_difference(
    first: list[float],
    second: list[float],
    measured: list[float],
    draws: int = 4000,
    seed: int = 1,
) -> tuple[Interval, float]:
    residuals_first = [
        math.log10(p) - math.log10(m) for p, m in zip(first, measured) if p > 0 and m > 0
    ]
    residuals_second = [
        math.log10(p) - math.log10(m) for p, m in zip(second, measured) if p > 0 and m > 0
    ]
    if len(residuals_first) != len(residuals_second):
        raise ValueError("the two models did not produce a usable pair on the same sites")
    n = len(residuals_first)
    rng = random.Random(seed)
    differences = []
    for _ in range(draws):
        index = [rng.randrange(n) for _ in range(n)]
        a = (sum(residuals_first[i] ** 2 for i in index) / n) ** 0.5
        b = (sum(residuals_second[i] ** 2 for i in index) / n) ** 0.5
        differences.append(a - b)
    differences.sort()
    wins = sum(1 for d in differences if d > 0) / draws
    point = (sum(r * r for r in residuals_first) / n) ** 0.5 - (
        sum(r * r for r in residuals_second) / n
    ) ** 0.5
    return (
        Interval(point, differences[int(0.025 * draws)], differences[int(0.975 * draws) - 1]),
        wins,
    )


def leave_one_out_nested_pinn(
    sites: list[Site],
    weights: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0, 100.0),
    seed: int = 0,
    epochs: int = 4000,
) -> tuple[list[float], list[float]]:
    predicted: list[float] = []
    chosen: list[float] = []
    for index in range(len(sites)):
        training = [s for i, s in enumerate(sites) if i != index]
        best_weight, best_error = weights[0], float("inf")
        for weight in weights:
            inner = leave_one_out(
                training, _pinn_fit(weight, seed, epochs), predict_pinn
            )
            error = metrics(inner, [s.measured_flux_bq_m2_s for s in training]).rmse
            if error < best_error:
                best_weight, best_error = weight, error
        trained, _ = fit_pinn(
            training, physics_weight=best_weight, seed=seed, epochs=epochs
        )
        predicted.append(predict_pinn(trained, [sites[index]])[0])
        chosen.append(best_weight)
    return predicted, chosen


def loo_rmse_across_seeds(
    sites: list[Site], weight: float = 1.0, seeds: tuple[int, ...] = tuple(range(5))
) -> Interval:
    measured = [s.measured_flux_bq_m2_s for s in sites]
    values = sorted(
        metrics(
            leave_one_out(sites, _pinn_fit(weight, seed), predict_pinn), measured
        ).rmse
        for seed in seeds
    )
    return Interval(values[len(values) // 2], values[0], values[-1])


def learning_curve(
    sites: list[Site],
    fit,
    predict,
    sizes: tuple[int, ...] = (5, 7, 9, 11, 12),
    draws: int = 200,
    seed: int = 0,
) -> dict[int, Interval]:
    rng = random.Random(seed)
    out: dict[int, Interval] = {}
    for size in sizes:
        if size >= len(sites):
            continue
        scores = []
        for _ in range(draws):
            order = list(range(len(sites)))
            rng.shuffle(order)
            training = [sites[i] for i in order[:size]]
            held_out = [sites[i] for i in order[size:]]
            try:
                model = fit(training)
                prediction = predict(model, held_out)
            except (ValueError, ZeroDivisionError):
                continue
            scores.append(
                metrics(prediction, [s.measured_flux_bq_m2_s for s in held_out]).rmse
            )
        if not scores:
            continue
        scores.sort()
        out[size] = Interval(
            scores[len(scores) // 2], scores[int(0.05 * len(scores))],
            scores[int(0.95 * len(scores)) - 1],
        )
    return out


def _pinn_fit(weight: float, seed: int = 0, epochs: int = 4000):
    def fit(training: list[Site]):
        trained, _ = fit_pinn(
            training, physics_weight=weight, seed=seed, epochs=epochs
        )
        return trained

    return fit


def main() -> None:
    sites = load_sites()
    measured = [s.measured_flux_bq_m2_s for s in sites]
    n = len(sites)

    print(f"Sezione 14 — catena obbligatoria.  N = {n} siti, "
          f"{len(groups(sites))} campagne indipendenti\n")

    print("PASSO 1-3: in campione (in-sample). Non e' una misura di capacita' predittiva.")
    print(HEADER)
    print("-" * len(HEADER))
    print(metrics(physical(sites, False), measured).line(
        "fisico, stato eman. ignorato", "0"))
    print(metrics(physical(sites), measured).line(
        "fisico, stato eman. applicato", "0"))
    shift = fit_mean_ratio(sites)
    print(metrics([p * 10.0**shift for p in physical(sites)], measured).line(
        "fisico + 1 fattore globale", "1"))
    print(metrics(predict_loglinear(fit_loglinear(sites), sites), measured).line(
        "log-lineare sui predittori", f"{len(PREDICTORS) + 1}"))
    print(metrics(predict_residual_linear(fit_residual_linear(sites), sites), measured).line(
        "residuale lineare", f"{len(PREDICTORS) + 1}"))
    trained, pinn_parameters = fit_pinn(sites)
    print(metrics(predict_pinn(trained, sites), measured).line(
        "rete physics-informed", f"{pinn_parameters}"))
    data_only, _ = fit_pinn(sites, physics_weight=0.0)
    print(metrics(predict_pinn(data_only, sites), measured).line(
        "rete solo-dati", f"{pinn_parameters}"))

    print("\nPASSO 4: fuori campione. LOO = lascia fuori un sito; "
          "LOSO = lascia fuori una campagna intera.")
    for label, protocol in (("LOO", leave_one_out), ("LOSO", leave_one_source_out)):
        print(f"\n  --- {label} ---")
        print(HEADER)
        print("-" * len(HEADER))
        print(metrics(physical(sites), measured).line("fisico (non si addestra)", "0"))
        print(metrics(
            protocol(sites, lambda t: fit_mean_ratio(t),
                     lambda m, s: [p * 10.0**m for p in physical(s)]),
            measured).line("fisico + 1 fattore globale", "1"))
        print(metrics(
            protocol(sites, fit_scaled_physics, predict_scaled_physics), measured).line(
            "fisico scalato (offset+pend.)", "2"))
        print(metrics(
            protocol(sites, fit_physics_plus_saturation,
                     predict_physics_plus_saturation), measured).line(
            "fisico + esponente saturazione", "3"))
        print(metrics(
            protocol(sites, fit_loglinear, predict_loglinear), measured).line(
            "log-lineare sui predittori", f"{len(PREDICTORS) + 1}"))
        print(metrics(
            protocol(sites, fit_residual_linear, predict_residual_linear), measured).line(
            "residuale lineare", f"{len(PREDICTORS) + 1}"))
        print(metrics(
            protocol(sites, _pinn_fit(1.0), predict_pinn), measured).line(
            "rete physics-informed", f"{pinn_parameters}"))
        print(metrics(
            protocol(sites, _pinn_fit(0.0), predict_pinn), measured).line(
            "rete solo-dati", f"{pinn_parameters}"))

    print("\nPASSO 4b: quanto sono grandi queste differenze, su tredici siti.")
    print("  Intervallo al 95% per ricampionamento dei siti. Se due intervalli si")
    print("  sovrappongono quasi del tutto, l'ordine fra quei due modelli non e' un")
    print("  risultato: e' rumore che sembra una classifica.")
    print(f"  {'modello':<32}{'par':>5}   RMSE in LOO [intervallo]")
    intervalli = [
        ("fisico (non si addestra)", "0", physical(sites)),
        ("fisico + 1 fattore globale", "1",
         leave_one_out(sites, fit_mean_ratio,
                       lambda m, s: [p * 10.0**m for p in physical(s)])),
        ("fisico scalato", "2",
         leave_one_out(sites, fit_scaled_physics, predict_scaled_physics)),
        ("fisico + esp. saturazione", "3",
         leave_one_out(sites, fit_physics_plus_saturation,
                       predict_physics_plus_saturation)),
        ("log-lineare", f"{len(PREDICTORS) + 1}",
         leave_one_out(sites, fit_loglinear, predict_loglinear)),
        ("residuale lineare", f"{len(PREDICTORS) + 1}",
         leave_one_out(sites, fit_residual_linear, predict_residual_linear)),
    ]
    for label, parameters, predicted in intervalli:
        print(f"  {label:<32}{parameters:>5}   {bootstrap_rmse(predicted, measured)}")
    print("  Il confronto appaiato fra la fisica e il modello a 2 parametri, sullo")
    print("  stesso ricampionamento, e' il modo corretto di chiedere chi vince:")
    due_parametri = leave_one_out(sites, fit_scaled_physics, predict_scaled_physics)
    differenza, vittorie = paired_rmse_difference(
        physical(sites), due_parametri, measured)
    print(f"    RMSE(fisico) - RMSE(2 par) = {differenza}")
    print(f"    il modello a 2 parametri vince nel {100 * vittorie:.0f}% dei "
          f"ricampionamenti")
    print(f"    lo zero {'e dentro' if differenza.low <= 0 <= differenza.high else 'e fuori'}"
          f" l'intervallo, quindi la differenza "
          f"{'non e distinguibile da zero' if differenza.low <= 0 <= differenza.high else 'e reale'}")

    print("\n  I coefficienti dei due modelli parsimoniosi, stimati su tutti i 13 siti.")
    print("  I predittori sono scelti per ragioni fisiche PRIMA del fit, non dopo aver")
    print("  visto i punteggi: scegliere il sottoinsieme migliore a posteriori sarebbe")
    print("  selezione sul protocollo di valutazione.")
    c0, c1 = fit_scaled_physics(sites)
    print(f"    fisico scalato:  offset {c0:+.3f}   pendenza {c1:+.3f}")
    print("    Una pendenza sotto 1 dice che i flussi misurati variano fra i siti MENO")
    print("    di quanto il modello fisico preveda: ordinamento giusto, escursione")
    print("    troppo ampia. E' un'affermazione sulla fisica, non sul fit.")
    d0, d1, d2 = fit_physics_plus_saturation(sites)
    print(f"    + saturazione:   offset {d0:+.3f}   pendenza {d1:+.3f}   "
          f"esponente saturazione {d2:+.3f}")

    print("\nPASSO 6a: ablation del peso della fisica nella loss (protocollo LOO)")
    print(f"  {'peso fisica':>12}{'MBE':>10}{'SD':>8}{'RMSE':>8}{'F2':>8}")
    for weight in (0.0, 0.1, 1.0, 10.0, 100.0):
        result = metrics(
            leave_one_out(sites, _pinn_fit(weight), predict_pinn), measured)
        print(f"  {weight:>12.1f}{result.mbe:>+10.3f}{result.sd:>8.3f}"
              f"{result.rmse:>8.3f}{100 * result.f2:>7.0f}%")

    print("\nPASSO 6b: ablation dell'ingresso fisico (stato dell'emanazione)")
    print(f"  {'':<30}{'MBE':>10}{'SD':>8}{'RMSE':>8}{'F2':>8}")
    for label, flag in (("stato applicato", True), ("stato ignorato", False)):
        result = metrics(physical(sites, flag), measured)
        print(f"  {label:<30}{result.mbe:>+10.3f}{result.sd:>8.3f}"
              f"{result.rmse:>8.3f}{100 * result.f2:>7.0f}%")

    print("\nPASSO 6c: sensibilita' al parametro pubblicato a (Zhuo), non fittato")
    print(f"  {'a':>6}{'fattore':>10}{'MBE':>10}{'SD':>8}{'F2':>8}")
    for a in (1.5, 1.7, 1.9):
        predicted = [s.physical_flux_bq_m2_s(plateau_a=a) for s in sites]
        result = metrics(predicted, measured)
        print(f"  {a:>6.1f}{1 + a:>10.1f}{result.mbe:>+10.3f}"
              f"{result.sd:>8.3f}{100 * result.f2:>7.0f}%")

    print("\nPASSO 6d: incertezza della rete rispetto al seme casuale (LOO, 5 semi)")
    rmses = []
    for seed in range(5):
        rmses.append(metrics(
            leave_one_out(sites, _pinn_fit(1.0, seed), predict_pinn), measured).rmse)
    spread = max(rmses) - min(rmses)
    print(f"  RMSE per seme: {', '.join(f'{v:.3f}' for v in rmses)}")
    print(f"  escursione fra semi: {spread:.3f} decadi")
    print(f"  mediana con minimo e massimo: {loo_rmse_across_seeds(sites)}")
    print("  Le tabelle del passo 4 riportano il seme 0. La mediana e' il numero da")
    print("  citare: un seme solo e' un'estrazione, non il punteggio del modello.")

    print("\nPASSO 5b: il peso della fisica scelto SENZA guardare il sito tenuto fuori.")
    print("  Il passo 6a sceglie il peso migliore confrontandolo con l'errore LOO di")
    print("  tutto il dataset, quindi quel numero non puo' essere citato come stima")
    print("  fuori campione: i siti tenuti fuori hanno partecipato alla scelta. Qui la")
    print("  scelta si rifa' dentro ogni fold, su un LOO interno ai soli 12 siti di")
    print("  addestramento. Costa 13 x 5 x 12 addestramenti piu' 13 finali.")
    nested, chosen = leave_one_out_nested_pinn(sites)
    print(HEADER)
    print("-" * len(HEADER))
    print(metrics(nested, measured).line("rete, peso scelto annidato", f"{pinn_parameters}"))
    print(f"  intervallo: {bootstrap_rmse(nested, measured)}")
    conteggio = {w: chosen.count(w) for w in sorted(set(chosen))}
    print(f"  peso scelto in ciascuno dei 13 fold: {conteggio}")
    print("  Se la scelta salta da un fold all'altro, e' un risultato: vuol dire che i")
    print("  dati non determinano il parametro.")

    print("\nPASSO 6f: come cala l'errore al crescere del numero di siti.")
    print("  Il rapporto fra siti e parametri e' quello che decide, non N da solo.")
    print(f"  {'N':>4}{'1 parametro':>16}{'2 parametri':>16}{'6 parametri':>16}")
    curve = {
        1: learning_curve(sites, fit_mean_ratio,
                          lambda m, s: [p * 10.0**m for p in physical(s)], draws=400),
        2: learning_curve(sites, fit_scaled_physics, predict_scaled_physics, draws=400),
        6: learning_curve(sites, fit_loglinear, predict_loglinear, draws=400),
    }
    for size in (5, 7, 9, 11, 12):
        row = f"  {size:>4}"
        for count in (1, 2, 6):
            value = curve[count].get(size)
            row += f"{value.point:>16.3f}" if value else f"{'—':>16}"
        print(row)
    print(f"  La fisica a 0 parametri non si addestra: curva piatta a "
          f"{metrics(physical(sites), measured).rmse:.3f} per qualunque N.")
    print("  Un trattino vuol dire che a quel N il modello non si puo' nemmeno")
    print("  stimare: sei parametri richiedono almeno sei righe. A N = 12 il test e' un")
    print("  solo sito, quindi quella colonna e' un residuo singolo e non una media.")

    print("\nPASSO 6e: controllo di capacita'. Etichette permutate, protocollo LOO.")
    print("  Un modello che 'impara' anche dal rumore e' un modello che non ha imparato.")
    shuffled = list(sites)
    random.Random(0).shuffle(shuffled)
    permuted = [
        Site(**{**s.__dict__, "measured_flux_bq_m2_s": t.measured_flux_bq_m2_s})
        for s, t in zip(sites, shuffled)
    ]
    permuted_measured = [s.measured_flux_bq_m2_s for s in permuted]
    print(HEADER)
    print("-" * len(HEADER))
    print(metrics(
        [p for p in physical(permuted)], permuted_measured).line(
        "fisico su etichette permutate", "0"))
    print(metrics(
        leave_one_out(permuted, fit_residual_linear, predict_residual_linear),
        permuted_measured).line("residuale, etich. permutate", f"{len(PREDICTORS) + 1}"))
    print(metrics(
        [predict_residual_linear(fit_residual_linear(permuted), permuted)[i]
         for i in range(len(permuted))],
        permuted_measured).line("residuale, in campione", f"{len(PREDICTORS) + 1}"))


if __name__ == "__main__":
    main()
