#!/usr/bin/env python3
"""Where does a boat end up after switching its tracker off? (Experiment 26)

Data: Global Fishing Watch's suspected intentional AIS-disabling events, Welch et al.
2022 (github.com/GlobalFishingWatch/AIS-disabling-high-seas, data/disabling_events.zip).
Unzip disabling_events.csv into data/ (git-ignored).

Each event gives only two points: where the boat went dark and where it reappeared
(12+ hours later). That tests the MOTION GUESS ("where could it be by now?"), not the
flight plan: we don't know where the boat was during any one-hour search window.

Every method gets the same search area (default 972 km^2 = three F10-like drones,
2 km wide sweep, 45 m/s, 1 hour), placed where it thinks the boat most likely is,
centred on where the boat went dark. With no heading known, every guess is the same
in all directions, so the best area is a set of rings around the start point.
Score: share of test boats whose reappearance point lies inside the area.

    python dark_gaps_sim.py                      # 2019 gaps of 12-24 h, learn from 2017-18
"""

import argparse
import csv
import math

import numpy as np

from cuesearch.evaluate import wilson

KNOT_KMH = 1.852


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


def load(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) for r in rows])
    return {
        "year": np.array([int(r["gap_start_timestamp"][:4]) for r in rows]),
        "cls": np.array([r["vessel_class"] for r in rows]),
        "hours": f("gap_hours"),
        "km": haversine_km(f("gap_start_lat"), f("gap_start_lon"), f("gap_end_lat"), f("gap_end_lon")),
    }


def best_rings(radii_km, weights, area_km2, bin_km=0.25):
    """Rings around the start point with the most guess-weight per km^2, up to area_km2.

    radii_km/weights: a cloud of guessed distances (same in every direction).
    Returns a boolean function in(r) for the chosen region.
    """
    edges = np.arange(0.0, max(radii_km.max(), bin_km) + 2 * bin_km, bin_km)
    mass, _ = np.histogram(radii_km, bins=edges, weights=weights)
    ring_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    order = np.argsort(-(mass / ring_area), kind="stable")
    order = order[mass[order] > 0]
    keep = order[np.cumsum(ring_area[order]) <= area_km2]   # densest rings until the area runs out
    chosen = np.zeros(len(mass), dtype=bool)
    chosen[keep] = True
    return lambda r: chosen[np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(chosen) - 1)] & (r < edges[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/disabling_events.csv")
    ap.add_argument("--gap-hours", type=float, nargs=2, default=[12.0, 24.0])
    ap.add_argument("--area-km2", type=float, default=3 * 2.0 * 45.0 * 3600 / 1000)
    ap.add_argument("--straight-kn", type=float, nargs=2, default=[10.0, 30.0])
    args = ap.parse_args()

    d = load(args.data)
    lo, hi = args.gap_hours
    window = (d["hours"] >= lo) & (d["hours"] <= hi)
    train = window & (d["year"] <= 2018)
    test = window & (d["year"] == 2019)
    r_test, h_test, c_test = d["km"][test], d["hours"][test], d["cls"][test]
    n = int(test.sum())
    print(f"test: {n} gaps of {lo:.0f}-{hi:.0f} h in 2019; learn from {int(train.sum())} gaps in 2017-18; "
          f"search area {args.area_km2:.0f} km^2")

    hits = {}
    # 1. Baseline: a circle around the last known position.
    hits["circle around last position"] = r_test <= math.sqrt(args.area_km2 / math.pi)

    # 2. Straight lines at 10-30 kn (the simulation's assumption), per event's own gap length.
    # 3. Learned: net speeds from 2017-18 gaps of the same vessel class, times this gap's length.
    straight = np.zeros(n, dtype=bool)
    learned = np.zeros(n, dtype=bool)
    kn = np.random.default_rng(0).uniform(*args.straight_kn, 4000)
    v_train = {c: d["km"][train & (d["cls"] == c)] / d["hours"][train & (d["cls"] == c)]
               for c in np.unique(d["cls"])}
    for i in range(n):
        h = h_test[i]
        straight[i] = best_rings(kn * KNOT_KMH * h, np.ones(4000), args.area_km2)(r_test[i])
        v = v_train[c_test[i]]
        learned[i] = best_rings(v * h, np.ones(len(v)), args.area_km2)(r_test[i])
    hits[f"straight lines {args.straight_kn[0]:.0f}-{args.straight_kn[1]:.0f} kn"] = straight
    hits["learned from 2017-18, by vessel class"] = learned

    print(f"\n{'method':>40} {'caught':>8} {'95% CI':>16}")
    for name, h in hits.items():
        k = int(h.sum())
        a, b = wilson(k, n)
        print(f"{name:>40} {k / n:>8.1%}  [{a:>5.1%}, {b:>6.1%}]")
    print(f"\nfor scale: circle radius {math.sqrt(args.area_km2 / math.pi):.1f} km; "
          f"test boats' median distance from where they went dark {np.median(r_test):.1f} km")


if __name__ == "__main__":
    main()
