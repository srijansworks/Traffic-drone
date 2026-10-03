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
from search_sim import STRATEGIES, plan

# (id, label, true boat speeds, what the searchers believe)
SCENARIOS = [
    ("right", "Searchers guess right", (10.0, 30.0), (10.0, 30.0)),
    ("wrong", "Searchers guess wrong", (0.0, 40.0), (10.0, 30.0)),
]
# (id, label, drones per strategy, strategies shown -- in legend order)
MODES = [
    ("1", "1 drone", 1, STRATEGIES),
    ("3", "3 drones", 3, ("expanding_square", "ring_search", "greedy_shared", "greedy_independent")),
    ("3c", "3 drones: combos", 3, ("greedy_shared", "ring_then_smart", "guard_plus_smart")),
]
PLAYBACK_AGES_MIN = [2, 5, 10, 20]   # tip ages you can pick in the demo
CURVE_AGES_MIN = [0, 2, 4, 6, 8, 10, 15, 20, 30, 45, 60]
BOATS_SHOWN = 300                    # dots on screen per tip age
CURVE_TRIALS = 1000                  # boats per point on the results chart
PATH_EVERY_S = 10                    # keep one drone position per 10 s
SEED = 1


def main():
    drone = Drone()
    keep = int(PATH_EVERY_S / 2.0)   # paths are sampled every 2 s
    scenarios = []
    for sid, slabel, truth_kn, assumed_kn in SCENARIOS:
        truth, assumed = BoatPrior(truth_kn), BoatPrior(assumed_kn)
        rng = np.random.default_rng(SEED)
        # One set of boats per tip age, shared by every mode in this scenario,
        # so 1 vs 3 drones vs combos is head-to-head on exactly the same boats.
        shown = [truth.sample(BOATS_SHOWN, rng) for _ in PLAYBACK_AGES_MIN]
        curve_boats = [truth.sample(CURVE_TRIALS, rng) for _ in CURVE_AGES_MIN]

        cache = {}   # the same team shows up in more than one mode; plan it once

        def team_for(name, age, n):
            key = (name, age, n)
            if key not in cache:
                cache[key] = plan(name, age * 60, drone, assumed, seed=SEED + 1000, n_drones=n)
            return cache[key]

        modes = []
        for mid, mlabel, n, names in MODES:
            playback = []
            for age, boats in zip(PLAYBACK_AGES_MIN, shown):
                runs = {}
                for name in names:
                    team = team_for(name, age, n)
                    found = team_time_to_find(team, boats, drone.sweep_radius_m)
                    runs[name] = {
                        "paths": [np.round(p.xy[::keep] / 1000, 3).tolist() for p in team],   # km
                        "found_s": [None if np.isinf(t) else float(t) for t in found],
                    }
                playback.append({"age_min": age, "runs": runs})
            curve = {name: [] for name in names}
            for age, boats in zip(CURVE_AGES_MIN, curve_boats):
                for name in names:
                    k = int(np.isfinite(team_time_to_find(team_for(name, age, n), boats,
                                                          drone.sweep_radius_m)).sum())
                    lo, hi = wilson(k, CURVE_TRIALS)
                    curve[name].append({"age_min": age, "found": k, "trials": CURVE_TRIALS,
                                        "p": k / CURVE_TRIALS, "lo": lo, "hi": hi})
            modes.append({"id": mid, "label": mlabel, "drones": n, "strategies": list(names),
                          "playback": playback, "curve": curve})
            print(f"{slabel}: {mlabel} done")

        scenarios.append({
            "id": sid, "label": slabel,
            "truth_kn": list(truth_kn), "assumed_kn": list(assumed_kn),
            "boats": [{"offset_km": np.round(b.offset / 1000, 4).tolist(),
                       "velocity_kms": np.round(b.velocity / 1000, 6).tolist()} for b in shown],
            "modes": modes,
        })

    data = {
        "assumptions": {
            "drone_speed_ms": drone.speed_ms,
            "sweep_radius_km": drone.sweep_radius_m / 1000,
            "endurance_min": drone.endurance_s / 60,
            "cue_error_m": BoatPrior().cue_error_m,
        },
        "path_every_s": PATH_EVERY_S,
        "ages_min": PLAYBACK_AGES_MIN,
        "scenarios": scenarios,
    }
    os.makedirs("demo", exist_ok=True)
    with open(os.path.join("demo", "demo_data.js"), "w") as f:
        f.write("window.DEMO = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print("Wrote demo/demo_data.js")


if __name__ == "__main__":
    main()
