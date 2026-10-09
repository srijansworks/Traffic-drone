#!/usr/bin/env python3
"""Run the searches against REAL boat paths (AIS data) instead of straight-line boats.

Same strategies and plans as search_sim.py. Only the boats change: each test boat
is a real vessel's next hour or so, starting at a random moment it was moving
(see cuesearch/ais.py). The searchers still plan with their usual guess
(--assumed-speed-kn), because that's all a real searcher would have.

    # get one day of data first (~320 MB), into data/ (git-ignored):
    curl -o data/AIS_2023_01_01.zip \\
        https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_01.zip

    python real_tracks_sim.py --drone x10 --drones 3 --assumed-speed-kn 0 12 --cue-ages 10
        # fishing boats (AIS type 30) vs three X10s

    python real_tracks_sim.py --drone x10 --drones 3 --cue-ages 10 \\
        --searcher-tracks data/AIS_2023_01_02.zip data/AIS_2023_01_03.zip
        # the smart search guesses from real tracks on OTHER days instead of straight lines

Write your prediction in PREDICTIONS.md BEFORE you run anything.
"""

import argparse

import numpy as np

from cuesearch.ais import TrackPrior, load_tracks, sample_boats
from cuesearch.evaluate import team_time_to_find, wilson
from cuesearch.model import BoatPrior, Drone
from search_sim import ALL_STRATEGIES, DRONES, boat_rng, plan


def load_days(paths, vessel_types):
    tracks = {}
    for path in paths:
        for mmsi, tr in load_tracks(path, vessel_types).items():
            tracks[f"{path}:{mmsi}"] = tr  # same boat on two days = two separate tracks
    return tracks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["data/AIS_2023_01_01.zip"], help="NOAA daily AIS zip(s)")
    ap.add_argument("--vessel-types", type=int, nargs="+", default=[30], help="AIS type codes; 30 = fishing")
    ap.add_argument("--min-sog-kn", type=float, default=2.0, help="boat must be moving at least this fast at the tip")
    ap.add_argument("--cue-ages", type=float, nargs="+", default=[10.0], help="minutes")
    ap.add_argument("--strategies", nargs="+", choices=ALL_STRATEGIES,
                    default=["expanding_square", "greedy_shared"])
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--trials", type=int, default=2000, help="random boats per cue age")
    ap.add_argument("--drone", choices=sorted(DRONES), default=None)
    ap.add_argument("--drone-speed", type=float, default=45.0, help="m/s")
    ap.add_argument("--sweep-radius", type=float, default=1000.0, help="m")
    ap.add_argument("--endurance-min", type=float, default=60.0)
    ap.add_argument("--assumed-speed-kn", type=float, nargs=2, default=[0.0, 12.0],
                    help="what the searchers believe about boat speed")
    ap.add_argument("--searcher-tracks", nargs="+", default=None,
                    help="AIS zip(s) the searchers learn from instead of straight lines; "
                         "must be different days from --data")
    ap.add_argument("--cue-error", type=float, default=100.0, help="m")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.drone:
        args.drone_speed = DRONES[args.drone]["speed"]
        args.endurance_min = DRONES[args.drone]["endurance_min"]

    drone = Drone(args.drone_speed, args.sweep_radius, args.endurance_min * 60)
    assumed = BoatPrior(tuple(args.assumed_speed_kn), args.cue_error)
    if args.searcher_tracks and set(args.searcher_tracks) & set(args.data):
        ap.error("--searcher-tracks must not include the test day(s) in --data")
    tracks = load_days(args.data, args.vessel_types)
    learned = load_days(args.searcher_tracks, args.vessel_types) if args.searcher_tracks else None

    print(f"{'cue age':>8} {'strategy':>20} {'found':>7} {'95% CI':>15}")
    for age_min in args.cue_ages:
        horizon = age_min * 60 + drone.endurance_s
        boats, info = sample_boats(tracks, args.trials, horizon, boat_rng(args.seed, age_min),
                                   cue_error_m=args.cue_error, min_sog_kn=args.min_sog_kn)
        guess = assumed
        if learned:
            guess = TrackPrior(learned, horizon, args.cue_error, args.min_sog_kn)
        for name in args.strategies:
            team = plan(name, age_min * 60, drone, guess, seed=args.seed + 1000, n_drones=args.drones)
            k = int(np.isfinite(team_time_to_find(team, boats, drone.sweep_radius_m)).sum())
            lo, hi = wilson(k, args.trials)
            print(f"{age_min:>6.0f} m {name:>20} {k / args.trials:>7.1%} [{lo:>5.1%},{hi:>6.1%}]")
        print(f"         ({info['vessels']} different vessels)")


if __name__ == "__main__":
    main()
