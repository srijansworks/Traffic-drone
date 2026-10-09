#!/usr/bin/env python3
"""Export the real-boat-path results (experiments 14-17) for the demo page.

Runs real_tracks_sim's setup and writes demo/real_results.js, which the "Real boats"
section of demo/index.html reads. Needs the three NOAA files in data/ (see
real_tracks_sim.py).

    python export_real_results.py
"""

import json
import os

import numpy as np

from cuesearch.ais import TrackPrior, sample_boats
from cuesearch.evaluate import team_time_to_find, wilson
from cuesearch.model import BoatPrior, Drone
from real_tracks_sim import load_days
from search_sim import DRONES, boat_rng, plan

TEST_DAY = ["data/AIS_2023_01_01.zip"]
LEARN_DAYS = ["data/AIS_2023_01_02.zip", "data/AIS_2023_01_03.zip"]
TRIALS = 2000
SEED = 1

# (id, label, AIS types, min speed at tip, drone preset, tip ages, straight-line guess kn, learn?)
GROUPS = [
    ("fishing", "Real fishing boats · 3 X10-like drones", [30], 2.0, "x10", [10, 30, 60], (0.0, 12.0), True),
    ("fast", "Real fast boats (pleasure craft at 15+ kn) · 3 F10-like drones", [37], 15.0, "f10", [10],
     (10.0, 40.0), False),
]


def rate(team, boats, drone):
    k = int(np.isfinite(team_time_to_find(team, boats, drone.sweep_radius_m)).sum())
    lo, hi = wilson(k, len(boats))
    return {"p": k / len(boats), "lo": lo, "hi": hi}


def main():
    groups = []
    for gid, label, types, min_sog, preset, ages, guess_kn, learn in GROUPS:
        d = DRONES[preset]
        drone = Drone(d["speed"], 1000.0, d["endurance_min"] * 60)
        tracks = load_days(TEST_DAY, types)
        learned = load_days(LEARN_DAYS, types) if learn else None
        straight = BoatPrior(guess_kn, 100.0)
        rows = []
        for age in ages:
            horizon = age * 60 + drone.endurance_s
            boats, info = sample_boats(tracks, TRIALS, horizon, boat_rng(SEED, age), min_sog_kn=min_sog)
            row = {"age_min": age, "vessels": info["vessels"],
                   "spiral": rate(plan("expanding_square", age * 60, drone, straight, SEED + 1000, 3),
                                  boats, drone),
                   "smart": rate(plan("greedy_shared", age * 60, drone, straight, SEED + 1000, 3),
                                 boats, drone)}
            if learn:
                prior = TrackPrior(learned, horizon, 100.0, min_sog)
                row["learned"] = rate(plan("greedy_shared", age * 60, drone, prior, SEED + 1000, 3),
                                      boats, drone)
            rows.append(row)
            print(gid, age, {k: round(v["p"] * 100, 1) for k, v in row.items() if isinstance(v, dict)})
        groups.append({"id": gid, "label": label, "guess_kn": list(guess_kn), "rows": rows})
    with open(os.path.join("demo", "real_results.js"), "w") as f:
        f.write("window.REAL = ")
        json.dump({"trials": TRIALS, "test_day": "2023-01-01", "learn_days": ["2023-01-02", "2023-01-03"],
                   "groups": groups}, f, separators=(",", ":"))
        f.write(";\n")
    print("Wrote demo/real_results.js")


if __name__ == "__main__":
    main()
