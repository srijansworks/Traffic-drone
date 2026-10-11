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
from cuesearch.ais import TrackPrior, sample_boats
from real_tracks_sim import load_days
from search_sim import NIGHT, STRATEGIES, boat_rng, detect_rng, plan

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
# Real-boat scenarios (needs the NOAA files in data/, see real_tracks_sim.py)
REAL_TEST_DAY = ["data/AIS_2023_01_01.zip"]
REAL_LEARN_DAYS = ["data/AIS_2023_01_02.zip", "data/AIS_2023_01_03.zip"]
REAL_STRAIGHT_KN = (0.0, 12.0)       # the straight-line guess used for real fishing boats
REAL_TRACK_EVERY_S = 30              # keep one boat position per 30 s


def run_modes(drone, shown, curve_boats, guess_for, label, p_detect=1.0, extra=None):
    """Plan every team and score it. guess_for(age_min) -> what the searchers believe.

    p_detect < 1: night camera that misses some passes (every strategy gets the same luck).
    extra: {mode id: strategies to add to that mode}, e.g. the night-aware search.
    """
    keep = int(PATH_EVERY_S / 2.0)   # paths are sampled every 2 s
    cache = {}   # the same team shows up in more than one mode; plan it once

    def team_for(name, age, n):
        key = (name, age, n)
        if key not in cache:
            cache[key] = plan(name, age * 60, drone, guess_for(age), seed=SEED + 1000, n_drones=n,
                              p_detect=p_detect)
        return cache[key]

    def find(team, boats, age):
        return team_time_to_find(team, boats, drone.sweep_radius_m, p_detect=p_detect,
                                 rng=detect_rng(SEED, age))

    modes = []
    for mid, mlabel, n, names in MODES:
        names = tuple(names) + tuple((extra or {}).get(mid, ()))
        playback = []
        for age, boats in zip(PLAYBACK_AGES_MIN, shown):
            runs = {}
            for name in names:
                team = team_for(name, age, n)
                found = find(team, boats, age)
                runs[name] = {
                    "paths": [np.round(p.xy[::keep] / 1000, 3).tolist() for p in team],   # km
                    "found_s": [None if np.isinf(t) else float(t) for t in found],
                }
            playback.append({"age_min": age, "runs": runs})
        curve = {name: [] for name in names}
        for age, boats in zip(CURVE_AGES_MIN, curve_boats):
            for name in names:
                k = int(np.isfinite(find(team_for(name, age, n), boats, age)).sum())
                lo, hi = wilson(k, len(boats))
                curve[name].append({"age_min": age, "found": k, "trials": len(boats),
                                    "p": k / len(boats), "lo": lo, "hi": hi})
        modes.append({"id": mid, "label": mlabel, "drones": n, "strategies": list(names),
                      "playback": playback, "curve": curve})
        print(f"{label}: {mlabel} done")
    return modes


def main():
    drone = Drone()
    scenarios = []
    for sid, slabel, truth_kn, assumed_kn in SCENARIOS:
        truth, assumed = BoatPrior(truth_kn), BoatPrior(assumed_kn)
        rng = np.random.default_rng(SEED)
        # One set of boats per tip age, shared by every mode in this scenario,
        # so 1 vs 3 drones vs combos is head-to-head on exactly the same boats.
        shown = [truth.sample(BOATS_SHOWN, rng) for _ in PLAYBACK_AGES_MIN]
        curve_boats = [truth.sample(CURVE_TRIALS, rng) for _ in CURVE_AGES_MIN]
        scenarios.append({
            "id": sid, "label": slabel,
            "truth_kn": list(truth_kn), "assumed_kn": list(assumed_kn),
            "boats": [{"offset_km": np.round(b.offset / 1000, 4).tolist(),
                       "velocity_kms": np.round(b.velocity / 1000, 6).tolist()} for b in shown],
            "modes": run_modes(drone, shown, curve_boats, lambda age, a=assumed: a, slabel),
        })

    # Night: same boats as "Searchers guess right", but a thermal camera that sees less
    # far and misses some passes. Adds the night-aware smart search to the 3-drone team.
    truth = BoatPrior(SCENARIOS[0][2])
    rng = np.random.default_rng(SEED)
    shown_n = [truth.sample(BOATS_SHOWN, rng) for _ in PLAYBACK_AGES_MIN]
    curve_n = [truth.sample(CURVE_TRIALS, rng) for _ in CURVE_AGES_MIN]
    night_drone = Drone(drone.speed_ms, NIGHT["sweep_radius"], drone.endurance_s)
    scenarios.append({
        "id": "night", "label": "Night, thermal camera", "night": True,
        "truth_kn": list(SCENARIOS[0][2]), "assumed_kn": list(SCENARIOS[0][3]),
        "sweep_radius_km": NIGHT["sweep_radius"] / 1000, "p_detect": NIGHT["p_detect"],
        "boats": [{"offset_km": np.round(b.offset / 1000, 4).tolist(),
                   "velocity_kms": np.round(b.velocity / 1000, 6).tolist()} for b in shown_n],
        "modes": run_modes(night_drone, shown_n, curve_n, lambda age: truth, "Night",
                           p_detect=NIGHT["p_detect"], extra={"3": ("greedy_shared_night",)}),
    })

    # Real boats: NOAA AIS fishing-vessel paths (see cuesearch/ais.py). Both real
    # scenarios hunt the SAME real boats; only the searchers' guess differs.
    tracks = load_days(REAL_TEST_DAY, [30])
    learned = load_days(REAL_LEARN_DAYS, [30])
    horizon = lambda age: age * 60 + drone.endurance_s
    shown = [sample_boats(tracks, BOATS_SHOWN, horizon(a), boat_rng(SEED, a))[0] for a in PLAYBACK_AGES_MIN]
    curve_boats = [sample_boats(tracks, CURVE_TRIALS, horizon(a), boat_rng(SEED + 1, a))[0]
                   for a in CURVE_AGES_MIN]
    step = int(REAL_TRACK_EVERY_S / shown[0].dt_s)
    real_boats = [{"dt_s": b.dt_s * step, "track_km": np.round(b.xy[:, ::step] / 1000, 3).tolist()}
                  for b in shown]
    straight = BoatPrior(REAL_STRAIGHT_KN)
    priors = {}

    def learned_for(age):
        if age not in priors:
            priors[age] = TrackPrior(learned, horizon(age))
        return priors[age]

    for sid, slabel, guess_for, guess_txt in [
        ("real_straight", "Real fishing boats, straight-line guess", lambda age: straight,
         f"{REAL_STRAIGHT_KN[0]:.0f}–{REAL_STRAIGHT_KN[1]:.0f} knots, straight lines"),
        ("real_learned", "Real fishing boats, learned guess", learned_for,
         "real fishing tracks from Jan 2–3"),
    ]:
        scenarios.append({
            "id": sid, "label": slabel, "real": True, "guess_txt": guess_txt,
            "boats_ref": "real_boats",
            "modes": run_modes(drone, shown, curve_boats, guess_for, slabel),
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
        "real_boats": real_boats,
    }
    os.makedirs("demo", exist_ok=True)
    with open(os.path.join("demo", "demo_data.js"), "w") as f:
        f.write("window.DEMO = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print("Wrote demo/demo_data.js")


if __name__ == "__main__":
    main()
