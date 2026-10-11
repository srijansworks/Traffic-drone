#!/usr/bin/env python3
"""How close are the drones to the theoretical best? Compare against Brown's FAB.

FAB (cuesearch/fab.py) is the classic optimal plan for a moving target, but it
spreads search effort anywhere at once, which real drones can't do. So read its
number as a ceiling: "with this much sweeping, the best possible plan finds X%".
The smart search's gap to it is the room left for a better flyable plan.

    python fab_sim.py --drones 3 --cue-ages 10 30                  # straight-line boats
    python fab_sim.py --drones 3 --drone x10 --real --cue-ages 10 30 60
        # real Jan 1 fishing paths; smart search and FAB both guess from Jan 2-3 tracks

FAB's number is an average chance of finding each boat (it uses "random search"
detection), the drones' is a count of boats seen (perfect within 1 km). Close
enough to compare, not identical.

Write your prediction in PREDICTIONS.md BEFORE you run anything.
"""

import argparse

import numpy as np

from cuesearch.ais import TrackPrior, sample_boats
from cuesearch.evaluate import team_time_to_find
from cuesearch.fab import fab_plan
from cuesearch.model import BoatPrior, Drone
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
    ap.add_argument("--cell-m", type=float, default=500.0, help="FAB grid size")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.drone:
        args.drone_speed = DRONES[args.drone]["speed"]
        args.endurance_min = DRONES[args.drone]["endurance_min"]

    drone = Drone(args.drone_speed, args.sweep_radius, args.endurance_min * 60)
    if args.real:
        tracks = load_days(["data/AIS_2023_01_01.zip"], [30])
        learned = load_days(["data/AIS_2023_01_02.zip", "data/AIS_2023_01_03.zip"], [30])
    else:
        truth = BoatPrior(tuple(args.boat_speed_kn))

    print(f"{'cue age':>8} {'spiral':>8} {'smart':>8} {'FAB ceiling':>12}")
    for age in args.cue_ages:
        horizon = age * 60 + drone.endurance_s
        if args.real:
            boats = sample_boats(tracks, args.trials, horizon, boat_rng(args.seed, age))[0]
            guess = TrackPrior(learned, horizon)
        else:
            boats = truth.sample(args.trials, boat_rng(args.seed, age))
            guess = truth
        out = []
        for name in ("expanding_square", "greedy_shared"):
            team = plan(name, age * 60, drone, guess, seed=args.seed + 1000, n_drones=args.drones)
            out.append(np.isfinite(team_time_to_find(team, boats, drone.sweep_radius_m)).mean())
        fab = fab_plan(age * 60, drone, guess, args.drones, cell_m=args.cell_m, seed=args.seed + 1000)
        out.append(fab.p_found(boats).mean())
        print(f"{age:>6.0f} m " + " ".join(f"{p:>8.1%}" for p in out[:2]) + f" {out[2]:>12.1%}")


if __name__ == "__main__":
    main()
