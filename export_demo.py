#!/usr/bin/env python3
"""Export simulation runs for the browser demo (demo/index.html).

The web page does no simulating of its own -- it only plays back what this
script computed with the real cuesearch code, so the demo can never disagree
with search_sim.py.

    python export_demo.py            # writes demo/demo_data.js
    open demo/index.html             # watch it locally
"""

import json
import os

import numpy as np

from cuesearch.evaluate import team_time_to_find, wilson
from cuesearch.model import BoatPrior, Drone
from search_sim import STRATEGIES, TEAM_STRATEGIES, plan

MODES = [(1, STRATEGIES), (3, TEAM_STRATEGIES)]   # (drones per strategy, strategies shown)
PLAYBACK_AGES_MIN = [2, 5, 10, 20]   # tip ages you can pick in the demo
CURVE_AGES_MIN = [0, 2, 4, 6, 8, 10, 15, 20, 30, 45, 60]
BOATS_SHOWN = 300                    # dots on screen per tip age
CURVE_TRIALS = 1000                  # boats per point on the results chart
PATH_EVERY_S = 10                    # keep one drone position per 10 s
SEED = 1


def main():
    drone = Drone()
    prior = BoatPrior()
    rng = np.random.default_rng(SEED)
    keep = int(PATH_EVERY_S / 2.0)   # paths are sampled every 2 s

    # One set of boats per tip age, shared by every mode, so 1 vs 3 drones is
    # a head-to-head on exactly the same boats.
    shown = [prior.sample(BOATS_SHOWN, rng) for _ in PLAYBACK_AGES_MIN]
    curve_boats = [prior.sample(CURVE_TRIALS, rng) for _ in CURVE_AGES_MIN]

    modes = []
    for n, names in MODES:
        playback = []
        for age, boats in zip(PLAYBACK_AGES_MIN, shown):
            runs = {}
            for name in names:
                team = plan(name, age * 60, drone, prior, seed=SEED + 1000, n_drones=n)
                found = team_time_to_find(team, boats, drone.sweep_radius_m)
                runs[name] = {
                    "paths": [np.round(p.xy[::keep] / 1000, 3).tolist() for p in team],   # km
                    "found_s": [None if np.isinf(t) else float(t) for t in found],
                }
            playback.append({"age_min": age, "runs": runs})
            print(f"{n} drone(s): playback tip {age} min done")

        curve = {name: [] for name in names}
        for age, boats in zip(CURVE_AGES_MIN, curve_boats):
            for name in names:
                team = plan(name, age * 60, drone, prior, seed=SEED + 1000, n_drones=n)
                k = int(np.isfinite(team_time_to_find(team, boats, drone.sweep_radius_m)).sum())
                lo, hi = wilson(k, CURVE_TRIALS)
                curve[name].append({"age_min": age, "found": k, "trials": CURVE_TRIALS,
                                    "p": k / CURVE_TRIALS, "lo": lo, "hi": hi})
        print(f"{n} drone(s): curve done")
        modes.append({"drones": n, "strategies": list(names), "playback": playback, "curve": curve})

    data = {
        "assumptions": {
            "drone_speed_ms": drone.speed_ms,
            "sweep_radius_km": drone.sweep_radius_m / 1000,
            "endurance_min": drone.endurance_s / 60,
            "boat_speed_kn": list(prior.speed_kn),
            "cue_error_m": prior.cue_error_m,
        },
        "path_every_s": PATH_EVERY_S,
        "ages_min": PLAYBACK_AGES_MIN,
        "boats": [{"offset_km": np.round(b.offset / 1000, 4).tolist(),
                   "velocity_kms": np.round(b.velocity / 1000, 6).tolist()} for b in shown],
        "modes": modes,
    }
    os.makedirs("demo", exist_ok=True)
    with open(os.path.join("demo", "demo_data.js"), "w") as f:
        f.write("window.DEMO = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print("Wrote demo/demo_data.js")


if __name__ == "__main__":
    main()
