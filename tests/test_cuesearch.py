"""Stale-cue search tests. Run: python tests/test_cuesearch.py

These check the mechanics -- that boats move the way we said, the drone flies
at the speed we said, and "found" means what we think it means. They do NOT
check which strategy wins. That's the experiment, not a test.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cuesearch.evaluate import time_to_find, wilson  # noqa: E402
from cuesearch.model import KNOT, BoatPrior, Boats, Drone  # noqa: E402
from cuesearch.planners import expanding_square, probability_greedy, ring_search  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def one_boat(x, y, vx, vy):
    return Boats(offset=np.array([[x, y]], float), velocity=np.array([[vx, vy]], float))


def test_boat_motion():
    print("boat motion")
    b = one_boat(100, 0, 10, 0)
    check("moves 10 m/s for 60 s", np.allclose(b.at(60.0), [[700, 0]]))
    check("array of times has shape (n, k, 2)", b.at(np.array([0.0, 1.0, 2.0])).shape == (1, 3, 2))

    rng = np.random.default_rng(0)
    boats = BoatPrior((10, 30), cue_error_m=0).sample(5000, rng)
    speed_kn = np.linalg.norm(boats.velocity, axis=1) / KNOT
    check("sampled speeds stay in range", speed_kn.min() >= 10 - 1e-9 and speed_kn.max() <= 30 + 1e-9)
    heading = np.arctan2(boats.velocity[:, 1], boats.velocity[:, 0])
    check("headings cover all directions", np.histogram(heading, bins=8)[0].min() > 400)


def test_paths_fly_at_drone_speed():
    print("paths")
    drone = Drone(speed_ms=45, sweep_radius_m=1000, endurance_s=600)
    for name, path in [("expanding_square", expanding_square(0, drone)),
                       ("ring_search", ring_search(0, drone, BoatPrior())),
                       ("probability_greedy", probability_greedy(0, drone, BoatPrior(), n_particles=300))]:
        steps = np.linalg.norm(np.diff(path.xy, axis=0), axis=1)
        check(f"{name}: starts at cue", np.allclose(path.xy[0], 0))
        # Samples are 2 s apart along the path; straight-line gaps only come up
        # short where the path turns a corner between two samples.
        check(f"{name}: never faster than the drone", steps.max() <= 45 * 2.0 + 1e-6,
              f"max {steps.max():.1f}")
        check(f"{name}: full speed except at corners", np.mean(np.isclose(steps, 90.0)) > 0.9)
        check(f"{name}: one sample per 2 s for the whole endurance",
              len(path.times) == 301 and np.isclose(path.times[-1] - path.times[0], 600))

    sq = expanding_square(0, drone)
    corners = np.array([[0, 0], [2000, 0], [2000, 2000], [-2000, 2000]])
    check("expanding square legs are 1,1,2 spacings",
          all(np.min(np.linalg.norm(sq.xy - c, axis=1)) < 90 for c in corners))


def test_ring_search_rides_the_ring():
    print("ring search")
    drone = Drone(speed_ms=45, sweep_radius_m=1000, endurance_s=1800)
    prior = BoatPrior((10, 30))
    path = ring_search(600, drone, prior)
    v_ring = 20 * KNOT
    r = np.linalg.norm(path.xy, axis=1)
    ring_r = v_ring * path.times
    # Ring starts at 10 min * 10.3 m/s = 6.2 km; drone closes at 45 - 10.3 m/s.
    reached = np.argmax(r >= ring_r)
    check("reaches the ring", reached > 0, "never reached it")
    check("reaches it when the math says (~3 min)",
          abs(path.times[reached] - 600 - 600 * v_ring / (45 - v_ring)) < 4,
          f"at {path.times[reached] - 600:.0f} s")
    after = slice(reached + 1, None)
    check("then stays on the ring (within 100 m)",
          np.max(np.abs(r[after] - ring_r[after])) < 100,
          f"max off by {np.max(np.abs(r[after] - ring_r[after])):.0f} m")
    angle = np.unwrap(np.arctan2(path.xy[after, 1], path.xy[after, 0]))
    check("and goes around it, not just outward", abs(angle[-1] - angle[0]) > np.pi)


def test_greedy_is_deterministic():
    print("determinism")
    drone = Drone(endurance_s=300)
    a = probability_greedy(600, drone, BoatPrior(), n_particles=300, seed=4)
    b = probability_greedy(600, drone, BoatPrior(), n_particles=300, seed=4)
    check("same seed, same path", np.array_equal(a.xy, b.xy))


def test_greedy_sweeps_guesses():
    print("greedy bookkeeping")
    drone = Drone(endurance_s=1200)
    _, particles, alive = probability_greedy(300, drone, BoatPrior(), n_particles=500,
                                             seed=2, return_particles=True)
    check("some guesses get ruled out", (~alive).sum() > 0)
    check("not every guess is ruled out by a 20 min search", alive.sum() > 0)


def test_time_to_find():
    print("time to find")
    drone = Drone(speed_ms=45, sweep_radius_m=1000, endurance_s=600)
    path = expanding_square(0, drone)

    t = time_to_find(path, one_boat(0, 0, 0, 0), 1000)
    check("boat sitting on the cue is found instantly", t[0] == 0)

    t = time_to_find(path, one_boat(500_000, 0, 0, 0), 1000)
    check("boat 500 km away is never found", np.isinf(t[0]))

    # Worked by hand: boat parked at (2500, 1000). The square flies east to
    # (2000, 0), then north. On that north leg the boat is 500 m to the side,
    # so it comes within 1000 m once y >= 1000 - sqrt(1000^2 - 500^2) = 134 m.
    # Distance flown: 2000 + 134 = 2134 m at 45 m/s = 47.4 s -> next sample, 48 s.
    t = time_to_find(path, one_boat(2500, 1000, 0, 0), 1000)
    check("hand-worked stationary boat found at 48 s", t[0] == 48.0, f"got {t[0]}")

    # Chunking must not change the answer.
    rng = np.random.default_rng(3)
    boats = BoatPrior().sample(1234, rng)
    a = time_to_find(path, boats, 1000, chunk=100)
    b = time_to_find(path, boats, 1000, chunk=5000)
    check("chunk size doesn't change results", np.array_equal(a, b))


def test_wilson():
    print("confidence interval")
    lo, hi = wilson(0, 10)
    check("0/10 lower bound is 0", lo == 0 and 0.2 < hi < 0.35, f"{lo:.3f},{hi:.3f}")
    lo, hi = wilson(50, 100)
    check("50/100 is centred on 50%", abs((lo + hi) / 2 - 0.5) < 1e-9 and 0.39 < lo < 0.41)
    lo, hi = wilson(10, 10)
    check("10/10 upper bound is 100%", hi == 1.0 and lo < 1.0)


if __name__ == "__main__":
    test_boat_motion()
    test_paths_fly_at_drone_speed()
    test_ring_search_rides_the_ring()
    test_greedy_is_deterministic()
    test_greedy_sweeps_guesses()
    test_time_to_find()
    test_wilson()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("All tests passed.")
