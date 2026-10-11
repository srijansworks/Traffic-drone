#!/usr/bin/env python3
"""Found isn't caught. How many boats does a patrol boat actually reach in time?

Same searches as search_sim.py. When a drone finds the boat it keeps watching, and
a patrol boat drives out to meet it (cuesearch/intercept.py). A boat counts as
CAUGHT only if the patrol reaches it while a drone is still watching.

    python intercept_sim.py --drones 3 --cue-ages 10
        # F10-like drones, 10-30 kn boats, patrol 20 km away at 25 kn
    python intercept_sim.py --drones 3 --drone x10 --real --cue-ages 10
        # real fishing paths (Jan 1), searchers learn from Jan 2-3
    python intercept_sim.py --relay-min 120
        # relief drones keep watching 2 more hours after the first batteries run out

Write your prediction in PREDICTIONS.md BEFORE you run anything.
"""

import argparse

import numpy as np

from cuesearch.ais import TrackPrior, sample_boats
from cuesearch.evaluate import team_time_to_find
from cuesearch.intercept import intercept_times
from cuesearch.model import KNOT, BoatPrior, Drone
from real_tracks_sim import load_days
from search_sim import DRONES, boat_rng, plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cue-ages", type=float, nargs="+", default=[10.0], help="minutes")
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--drone", choices=sorted(DRONES), default=None)
    ap.add_argument("--drone-speed", type=float, default=45.0, help="m/s")
    ap.add_argument("--sweep-radius", type=float, default=1000.0, help="m")
    ap.add_argument("--endurance-min", type=float, default=60.0)
    ap.add_argument("--boat-speed-kn", type=float, nargs=2, default=[10.0, 30.0])
    ap.add_argument("--real", action="store_true",
                    help="real fishing paths (Jan 1) as boats; searchers learn from Jan 2-3")
    ap.add_argument("--patrol-km", type=float, default=20.0, help="patrol boat's distance from the tip spot at t=0")
    ap.add_argument("--patrol-kn", type=float, default=25.0, help="patrol boat speed")
    ap.add_argument("--relay-min", type=float, default=0.0,
                    help="extra minutes of watching after the search drones' batteries run out")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.drone:
        args.drone_speed = DRONES[args.drone]["speed"]
        args.endurance_min = DRONES[args.drone]["endurance_min"]

    drone = Drone(args.drone_speed, args.sweep_radius, args.endurance_min * 60)
    start = (args.patrol_km * 1000.0, 0.0)
    v_patrol = args.patrol_kn * KNOT
    if args.real:
        tracks = load_days(["data/AIS_2023_01_01.zip"], [30])
        learned = load_days(["data/AIS_2023_01_02.zip", "data/AIS_2023_01_03.zip"], [30])
    else:
        truth = BoatPrior(tuple(args.boat_speed_kn))

    print(f"patrol {args.patrol_km:.0f} km away at {args.patrol_kn:.0f} kn; "
          f"drones watch {args.relay_min:.0f} min past their batteries")
    print(f"{'cue age':>8} {'search':>18} {'found':>7} {'caught (head to tip)':>21} {'caught (wait)':>14}")
    for age in args.cue_ages:
        search_end = age * 60 + drone.endurance_s
        custody_end = search_end + args.relay_min * 60
        if args.real:
            boats = sample_boats(tracks, args.trials, custody_end, boat_rng(args.seed, age))[0]
            guess = TrackPrior(learned, search_end)
        else:
            boats = truth.sample(args.trials, boat_rng(args.seed, age))
            guess = truth
        for name in ("expanding_square", "greedy_shared"):
            team = plan(name, age * 60, drone, guess, seed=args.seed + 1000, n_drones=args.drones)
            found_at = age * 60 + team_time_to_find(team, boats, drone.sweep_radius_m)
            caught = [np.isfinite(intercept_times(boats, found_at, custody_end, start, v_patrol, p)).mean()
                      for p in ("tip", "wait")]
            label = "spiral" if name == "expanding_square" else "smart, shared map"
            print(f"{age:>6.0f} m {label:>18} {np.isfinite(found_at).mean():>7.1%} "
                  f"{caught[0]:>21.1%} {caught[1]:>14.1%}")


if __name__ == "__main__":
    main()
