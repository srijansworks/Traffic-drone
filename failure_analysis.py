#!/usr/bin/env python3
"""Which boats get missed? Break the result down by how fast the boat really goes.

Same setup as search_sim.py (same boats, same plans for the same flags), but for
one tip age, and it reports the FIND RATE INSIDE EACH SPEED GROUP. That's the
fair way to compare groups: "30% of the misses were fast boats" can just mean
there were a lot of fast boats.

    python failure_analysis.py --drones 3 --strategy greedy_shared --cue-age 2 \\
        --assumed-speed-kn 10 30 --boat-speed-kn 0 40
        # the wrong-guess case: are the misses the stopped boats or the fast ones?

Write your prediction in PREDICTIONS.md BEFORE you run anything.
"""

import argparse

import numpy as np

from cuesearch.evaluate import team_time_to_find, wilson
from cuesearch.model import KNOT, BoatPrior, Drone
from search_sim import ALL_STRATEGIES, DRONES, boat_rng, plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cue-age", type=float, default=10.0, help="minutes")
    ap.add_argument("--strategy", choices=ALL_STRATEGIES, default="greedy_shared")
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--trials", type=int, default=4000, help="random boats")
    ap.add_argument("--drone", choices=sorted(DRONES), default=None)
    ap.add_argument("--drone-speed", type=float, default=45.0, help="m/s")
    ap.add_argument("--sweep-radius", type=float, default=1000.0, help="m")
    ap.add_argument("--endurance-min", type=float, default=60.0)
    ap.add_argument("--boat-speed-kn", type=float, nargs=2, default=[10.0, 30.0], help="TRUE boat speeds")
    ap.add_argument("--assumed-speed-kn", type=float, nargs=2, default=None, help="what the searchers believe")
    ap.add_argument("--cue-error", type=float, default=100.0, help="m")
    ap.add_argument("--bin-kn", type=float, default=5.0, help="width of each speed group")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.drone:
        args.drone_speed = DRONES[args.drone]["speed"]
        args.endurance_min = DRONES[args.drone]["endurance_min"]

    drone = Drone(args.drone_speed, args.sweep_radius, args.endurance_min * 60)
    truth = BoatPrior(tuple(args.boat_speed_kn), args.cue_error)
    assumed = BoatPrior(tuple(args.assumed_speed_kn or args.boat_speed_kn), args.cue_error)

    boats = truth.sample(args.trials, boat_rng(args.seed, args.cue_age))
    team = plan(args.strategy, args.cue_age * 60, drone, assumed, seed=args.seed + 1000, n_drones=args.drones)
    found = np.isfinite(team_time_to_find(team, boats, drone.sweep_radius_m))
    speed_kn = np.linalg.norm(boats.velocity, axis=1) / KNOT

    lo_kn, hi_kn = args.boat_speed_kn
    print(f"{args.strategy}, {args.drones} drone(s), tip {args.cue_age:.0f} min old; "
          f"boats really {lo_kn:.0f}-{hi_kn:.0f} kn, searchers assume "
          f"{assumed.speed_kn[0]:.0f}-{assumed.speed_kn[1]:.0f} kn")
    print(f"overall found {found.mean():.1%} of {args.trials}\n")
    print(f"{'boat speed':>12} {'boats':>6} {'found':>7} {'95% CI':>16}")
    edges = np.arange(lo_kn, hi_kn + args.bin_kn, args.bin_kn)
    edges[-1] = hi_kn
    for a, b in zip(edges[:-1], edges[1:]):
        last = b >= hi_kn
        group = (speed_kn >= a) & ((speed_kn <= b) if last else (speed_kn < b))
        n = int(group.sum())
        if n == 0:
            continue
        k = int(found[group].sum())
        ci_lo, ci_hi = wilson(k, n)
        print(f"{a:>5.0f}-{b:<3.0f} kn {n:>6} {k / n:>7.1%}  [{ci_lo:>5.1%}, {ci_hi:>6.1%}]")


if __name__ == "__main__":
    main()
