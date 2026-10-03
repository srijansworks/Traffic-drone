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

    playback = []
    for age in PLAYBACK_AGES_MIN:
        boats = prior.sample(BOATS_SHOWN, rng)
        runs = {}
        for name in STRATEGIES:
            team = plan(name, age * 60, drone, prior, seed=SEED + 1000)
            path = team[0]   # the demo shows one drone per strategy
            found = team_time_to_find(team, boats, drone.sweep_radius_m)
            runs[name] = {
                "path": np.round(path.xy[::keep] / 1000, 3).tolist(),      # km
                "found_s": [None if np.isinf(t) else float(t) for t in found],
            }
        playback.append({
            "age_min": age,
            "boats": {"offset_km": np.round(boats.offset / 1000, 4).tolist(),
                      "velocity_kms": np.round(boats.velocity / 1000, 6).tolist()},
            "runs": runs,
        })
        print(f"playback: tip {age} min done")

    curve = {name: [] for name in STRATEGIES}
    for age in CURVE_AGES_MIN:
        boats = prior.sample(CURVE_TRIALS, rng)
        for name in STRATEGIES:
            team = plan(name, age * 60, drone, prior, seed=SEED + 1000)
            k = int(np.isfinite(team_time_to_find(team, boats, drone.sweep_radius_m)).sum())
            lo, hi = wilson(k, CURVE_TRIALS)
            curve[name].append({"age_min": age, "found": k, "trials": CURVE_TRIALS,
                                "p": k / CURVE_TRIALS, "lo": lo, "hi": hi})
        print(f"curve: tip {age} min done")

    data = {
        "strategies": list(STRATEGIES),
        "assumptions": {
            "drone_speed_ms": drone.speed_ms,
            "sweep_radius_km": drone.sweep_radius_m / 1000,
            "endurance_min": drone.endurance_s / 60,
            "boat_speed_kn": list(prior.speed_kn),
            "cue_error_m": prior.cue_error_m,
        },
        "path_every_s": PATH_EVERY_S,
        "playback": playback,
        "curve": curve,
    }
    os.makedirs("demo", exist_ok=True)
    with open(os.path.join("demo", "demo_data.js"), "w") as f:
        f.write("window.DEMO = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print("Wrote demo/demo_data.js")


if __name__ == "__main__":
    main()
