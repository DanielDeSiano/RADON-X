from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
FILES = [
    ("Radon_Time_Serie_of_ISAR.csv", "radon"),
    ("Radon_Time_Serie_of_ISCA.csv", "radon"),
    ("Radon_Time_Serie_of_ISMAR.csv", "radon"),
    ("Meteo_Time_Serie_of_ISAR.csv", "meteo"),
]


def load(fname):
    with open(BASE + fname, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    return rows[0], rows[1:]


def quantile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def main():
    for fname, kind in FILES:
        header, data = load(fname)
        print(f"===== {fname} =====")
        print(f"rows(data): {len(data)}  columns: {header}")
        print("Source counts:", dict(Counter(r[4] for r in data)))
        print("Instruments:", dict(Counter(r[3] for r in data)))

        full = Counter(tuple(r) for r in data)
        dup_full = sum(c for c in full.values() if c > 1)
        nosrc = Counter(tuple(r[:4]) for r in data)
        dup_nosrc = sum(c for c in nosrc.values() if c > 1)
        ts_counter = Counter(r[0] for r in data)
        dup_ts = sum(c for c in ts_counter.values() if c > 1)
        print(f"duplicate rows (all 5 cols identical): {dup_full}")
        print(f"rows in groups identical except Source: {dup_nosrc} "
              f"(unique after dedup ignoring Source: {len(nosrc)})")
        print(f"rows sharing an exact timestamp: {dup_ts} "
              f"(unique timestamps: {len(ts_counter)})")

        groups = defaultdict(list)
        for r in data:
            groups[r[0]].append(r)
        multi = {k: v for k, v in groups.items() if len(v) > 1}
        same_vals = sum(1 for v in multi.values()
                        if len({(x[1], x[2]) for x in v}) == 1)
        d1, d2 = [], []
        for v in multi.values():
            if len({(x[1], x[2]) for x in v}) > 1:
                c1 = [float(x[1]) for x in v]
                c2 = [float(x[2]) for x in v]
                d1.append(max(c1) - min(c1))
                d2.append(max(c2) - min(c2))
        print(f"timestamp groups >1 row: {len(multi)} "
              f"(same values: {same_vals}, differing: {len(multi) - same_vals})")
        if d1:
            print(f"   spread within differing groups: "
                  f"col1 median={median(d1)} max={max(d1)}; "
                  f"col2 median={median(d2)} max={max(d2)}")

        parsed = sorted(
            (datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S.%f"), r)
            for r in data
        )
        near = 0
        for (t1, r1), (t2, r2) in zip(parsed, parsed[1:]):
            if 0 < (t2 - t1).total_seconds() <= 2 and r1[4] != r2[4]:
                near += 1
        print(f"near-duplicate pairs (0<dt<=2s, different Source): {near}")

        uts = sorted({t for t, _ in parsed})
        print(f"time range: {uts[0]} -> {uts[-1]} ({uts[-1] - uts[0]})")
        consecutive = zip(uts, uts[1:])
        steps = [(b - a).total_seconds() / 3600 for a, b in consecutive]
        print(f"median sampling step (h): {median(sorted(steps)):.3f}")
        big = sorted(((s, i) for i, s in enumerate(steps) if s > 24),
                     reverse=True)
        print(f"gaps > 24 h: {len(big)}")
        for s, i in big[:6]:
            print(f"   gap {s / 24:7.1f} d : {uts[i]} -> {uts[i + 1]}")

        if kind == "radon":
            raw = [float(r[1]) for r in data]
            cor = [float(r[2]) for r in data]
            for name, vals in [("raw", raw), ("corrected", cor)]:
                nonfin = sum(1 for v in vals if not math.isfinite(v))
                neg = sum(1 for v in vals if v < 0)
                zer = sum(1 for v in vals if v == 0)
                print(f"{name}: min={min(vals)} max={max(vals)} "
                      f"median={median(vals)} nonfinite={nonfin} "
                      f"negatives={neg} zeros={zer}")
            diffs = sorted(c - r for r, c in zip(raw, cor))
            ratios = sorted(c / r for r, c in zip(raw, cor) if r != 0)
            print(f"corrected vs raw: median diff={median(diffs):.1f} Bq/m^3, "
                  f"median ratio={median(ratios):.4f} "
                  f"[p5={quantile(ratios, 0.05):.4f}, "
                  f"p95={quantile(ratios, 0.95):.4f}]")
            by_inst = defaultdict(list)
            for r in data:
                by_inst[r[3]].append(r)
            for inst in sorted(by_inst):
                g = by_inst[inst]
                rr = sorted(float(x[2]) / float(x[1])
                            for x in g if float(x[1]) != 0)
                print(f"   {inst}: n={len(g)} "
                      f"median ratio={median(rr):.4f} "
                      f"[p5={quantile(rr, 0.05):.4f}, "
                      f"p95={quantile(rr, 0.95):.4f}] "
                      f"period {min(x[0] for x in g)[:10]} -> "
                      f"{max(x[0] for x in g)[:10]}")
        else:
            for idx in (1, 2):
                vals = [float(r[idx]) for r in data]
                nonfin = sum(1 for v in vals if not math.isfinite(v))
                print(f"{header[idx]}: min={min(vals)} max={max(vals)} "
                      f"median={median(vals)} nonfinite={nonfin}")
        print()

    _, data = load("Radon_Time_Serie_of_ISCA.csv")
    hi = sorted(((float(r[1]), r[0], r[3]) for r in data), reverse=True)[:8]
    print("ISCA raw Rn spikes:")
    print(f"   rows >10k: {sum(1 for r in data if float(r[1]) > 10000)}, "
          f">50k: {sum(1 for r in data if float(r[1]) > 50000)}, "
          f">100k: {sum(1 for r in data if float(r[1]) > 100000)}")
    for v, d, inst in hi:
        print(f"   {d}  raw={v:.0f}  ({inst})")
    n999 = [r for r in data if float(r[1]) == 99900]
    print(f"   rows with raw == 99900 (suspected full-scale value): {len(n999)} "
          f"({min(r[0] for r in n999)[:10]} -> {max(r[0] for r in n999)[:10]}, "
          f"instruments {sorted({r[3] for r in n999})})")

    def days(fname):
        return {r[0][:10] for r in load(fname)[1]}

    d_isar = days("Radon_Time_Serie_of_ISAR.csv")
    d_isca = days("Radon_Time_Serie_of_ISCA.csv")
    d_ismar = days("Radon_Time_Serie_of_ISMAR.csv")
    print(f"\ncalendar days with data: ISAR={len(d_isar)}, "
          f"ISCA={len(d_isca)}, ISMAR={len(d_ismar)}")
    print(f"ISAR&ISCA common days: {len(d_isar & d_isca)}")
    print(f"ISAR&ISMAR common days: {len(d_isar & d_ismar)}")
    print(f"ISAR&ISCA&ISMAR common days: {len(d_isar & d_isca & d_ismar)}")


if __name__ == "__main__":
    main()
