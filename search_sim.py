#!/usr/bin/env python3
"""Stale-cue search simulator. A satellite spots a boat; a drone arrives later.

The question: how old can the tip be before the drone can't find the boat?
And does a smart search (probability_greedy) beat the standard pattern
(expanding_square) -- and by how much, as the tip gets older?

For each cue age, both strategies plan one path, then we throw thousands of
random boats at those paths and count how many get found.

    python search_sim.py                              # the main experiment
    python search_sim.py --trials 300 --cue-ages 0 20 # quick look
    python search_sim.py --drone-speed 15             # a quadcopter instead
    python search_sim.py --assumed-speed-kn 10 30 --boat-speed-kn 0 40
        # the searcher THINKS boats do 10-30 kn, but real ones do 0-40.
        # This is the honesty test: the smart search only knows what you told it.
    python search_sim.py --drones 3
        # a team of 3 drones per strategy. Adds "smart, separate" vs
        # "smart, shared map" -- does sharing what you've searched matter?
    python search_sim.py --drones 3 --strategies greedy_shared ring_then_smart guard_plus_smart
        # pick exactly which strategies to run (see ALL_STRATEGIES below)

Write your prediction in PREDICTIONS.md BEFORE you run anything.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cuesearch.evaluate import team_time_to_find, wilson  # noqa: E402
from cuesearch.model import BoatPrior, Drone  # noqa: E402
from cuesearch.planners import probability_greedy  # noqa: E402
from cuesearch.team import (greedy_independent, greedy_shared, guard_plus_smart,  # noqa: E402
                            ring_team, ring_then_smart, square_team)

STRATEGIES = ("expanding_square", "ring_search", "probability_greedy")
TEAM_STRATEGIES = ("expanding_square", "ring_search", "greedy_independent", "greedy_shared")
ALL_STRATEGIES = STRATEGIES + ("greedy_independent", "greedy_shared", "ring_then_smart", "guard_plus_smart")
COLORS = {"expanding_square": "#888888", "ring_search": "#e67e22", "probability_greedy": "#1f6fd1",
          "greedy_independent": "#8e7cc3", "greedy_shared": "#1f6fd1",
          "ring_then_smart": "#16a085", "guard_plus_smart": "#c0392b"}


def strategies(n_drones):
    # With one drone, "separate" and "shared map" are the same thing as v1.
    return STRATEGIES if n_drones == 1 else TEAM_STRATEGIES


def plan(name, cue_age_s, drone, assumed, seed, n_drones=1):
    """Every strategy returns a list of paths, one per drone."""
    if name == "expanding_square":
        return square_team(cue_age_s, drone, n_drones)
    if name == "ring_search":
        return ring_team(cue_age_s, drone, assumed, n_drones)
    if name == "greedy_independent":
        return greedy_independent(cue_age_s, drone, assumed, n_drones, seed=seed)
    if name == "greedy_shared":
        return greedy_shared(cue_age_s, drone, assumed, n_drones, seed=seed)
    if name == "ring_then_smart":
        return ring_then_smart(cue_age_s, drone, assumed, n_drones, seed=seed)
    if name == "guard_plus_smart":
        return guard_plus_smart(cue_age_s, drone, assumed, n_drones, seed=seed)
    return [probability_greedy(cue_age_s, drone, assumed, seed=seed)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cue-ages", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 15, 20, 30, 45, 60],
                    help="minutes between the satellite seeing the boat and the drone arriving")
    ap.add_argument("--trials", type=int, default=2000, help="random boats per cue age")
    ap.add_argument("--drone-speed", type=float, default=45.0, help="m/s")
    ap.add_argument("--sweep-radius", type=float, default=1000.0, help="m")
    ap.add_argument("--endurance-min", type=float, default=60.0)
    ap.add_argument("--boat-speed-kn", type=float, nargs=2, default=[10.0, 30.0], help="TRUE boat speeds")
    ap.add_argument("--assumed-speed-kn", type=float, nargs=2, default=None,
                    help="what the searchers believe (default: same as truth, i.e. NO mismatch -- "
                         "set this to test what happens when the searcher is wrong)")
    ap.add_argument("--cue-error", type=float, default=100.0, help="m, satellite position error")
    ap.add_argument("--drones", type=int, default=1, help="drones per strategy (same for all)")
    ap.add_argument("--strategies", nargs="+", choices=ALL_STRATEGIES, default=None,
                    help="which strategies to run (default depends on --drones)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    drone = Drone(args.drone_speed, args.sweep_radius, args.endurance_min * 60)
    truth = BoatPrior(tuple(args.boat_speed_kn), args.cue_error)
    assumed = BoatPrior(tuple(args.assumed_speed_kn or args.boat_speed_kn), args.cue_error)
    os.makedirs(args.out, exist_ok=True)

    # Boats come from a different random stream than the searcher's guesses.
    rng = np.random.default_rng(args.seed)
    names = tuple(args.strategies) if args.strategies else strategies(args.drones)
    if "guard_plus_smart" in names and args.drones < 2:
        ap.error("guard_plus_smart needs --drones 2 or more")
    rows, curves, paths = [], {}, {}
    print(f"{'cue age':>8} {'strategy':>20} {'found':>7} {'95% CI':>15} {'median find':>12}")
    for age_min in args.cue_ages:
        boats = truth.sample(args.trials, rng)
        for name in names:
            team = plan(name, age_min * 60, drone, assumed, seed=args.seed + 1000, n_drones=args.drones)
            t = team_time_to_find(team, boats, drone.sweep_radius_m)
            k = int(np.isfinite(t).sum())
            lo, hi = wilson(k, args.trials)
            median = np.median(t)
            med_txt = f"{median / 60:.1f} min" if np.isfinite(median) else "never"
            print(f"{age_min:>6.0f} m {name:>20} {k / args.trials:>7.1%} "
                  f"[{lo:>5.1%},{hi:>6.1%}] {med_txt:>12}")
            rows.append(dict(cue_age_min=age_min, strategy=name, drones=args.drones,
                             trials=args.trials, found=k,
                             p_found=k / args.trials, ci_lo=lo, ci_hi=hi,
                             median_find_min=median / 60))
            curves[(age_min, name)] = t
            paths[(age_min, name)] = team

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plot_p_found(rows, names, args, os.path.join(args.out, "p_found_vs_cue_age.png"))
    plot_cumulative(curves, names, drone, os.path.join(args.out, "find_time_curves.png"))
    plot_snapshot(paths, names, truth, args, os.path.join(args.out, "snapshot.png"))
    print(f"\nSaved results.csv and 3 plots to {args.out}/")


def plot_p_found(rows, names, args, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in names:
        r = [x for x in rows if x["strategy"] == name]
        x = [v["cue_age_min"] for v in r]
        p = np.array([v["p_found"] for v in r])
        lo = np.array([v["ci_lo"] for v in r])
        hi = np.array([v["ci_hi"] for v in r])
        ax.plot(x, p * 100, "o-", color=COLORS[name], label=name.replace("_", " "))
        ax.fill_between(x, lo * 100, hi * 100, color=COLORS[name], alpha=0.2)
    ax.set_xlabel("Cue age when drone arrives (minutes)")
    ax.set_ylabel("Boats found within endurance (%)")
    ax.set_ylim(0, 102)
    team = f"{args.drones} drones, each " if args.drones > 1 else "Drone "
    ax.set_title(f"{team}{args.drone_speed:.0f} m/s, sees {args.sweep_radius:.0f} m, "
                 f"{args.endurance_min:.0f} min search; boats {args.boat_speed_kn[0]:.0f}-"
                 f"{args.boat_speed_kn[1]:.0f} kn", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cumulative(curves, names, drone, out):
    # Skip a 0-minute tip: every boat is found instantly, so it's a flat line.
    ages = sorted({a for a, _ in curves if a > 0}) or sorted({a for a, _ in curves})
    show = [ages[0], ages[len(ages) // 2], ages[-1]] if len(ages) >= 3 else ages
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grid = np.linspace(0, drone.endurance_s, 300)
    styles = ["-", "--", ":"]
    for style, age in zip(styles, show):
        for name in names:
            t = curves[(age, name)]
            frac = (t[None, :] <= grid[:, None]).mean(axis=1)
            ax.plot(grid / 60, frac * 100, style, color=COLORS[name],
                    label=f"{name.replace('_', ' ')}, tip {age:.0f} min old")
    ax.set_xlabel("Minutes spent searching")
    ax.set_ylabel("Boats found so far (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_snapshot(paths, names, truth, args, out):
    """Bird's-eye view of every strategy's search paths for one mid-range cue age."""
    ages = sorted({a for a, _ in paths})
    age = ages[len(ages) // 2]
    rng = np.random.default_rng(args.seed + 7)
    boats = truth.sample(400, rng)
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 5.2), squeeze=False,
                             sharex=True, sharey=True)
    axes = axes[0]
    for ax, name in zip(axes, names):
        team = paths[(age, name)]
        path = team[0]
        start = boats.at(path.times[0]) / 1000
        end = boats.at(path.times[-1]) / 1000
        ax.scatter(start[:, 0], start[:, 1], s=5, color="#e08a1e", alpha=0.5,
                   label="possible boats when drone arrives")
        ax.scatter(end[:, 0], end[:, 1], s=5, color="#c0392b", alpha=0.25,
                   label="...and when the drone runs out")
        for j, p in enumerate(team):
            ax.plot(p.xy[:, 0] / 1000, p.xy[:, 1] / 1000, color=COLORS[name], lw=1.2,
                    linestyle=["-", "--", ":", "-."][j % 4],
                    label="drone path" if len(team) == 1 else f"drone {j + 1}")
        ax.plot(0, 0, "k*", ms=12, label="satellite cue")
        ax.set_title(f"{name.replace('_', ' ')} (tip {age:.0f} min old)", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("km")
    axes[0].set_ylabel("km")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
