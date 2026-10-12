"""Stale-cue search tests. Run: python tests/test_cuesearch.py

These check the mechanics -- that boats move the way we said, the drone flies
at the speed we said, and "found" means what we think it means. They do NOT
check which strategy wins. That's the experiment, not a test.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cuesearch.evaluate import team_time_to_find, time_to_find, wilson  # noqa: E402
from cuesearch.model import KNOT, BoatPrior, Boats, Drone, TrackBoats  # noqa: E402
from cuesearch.planners import expanding_square, probability_greedy, ring_search  # noqa: E402
from cuesearch.team import (greedy_shared, guard_plus_smart, ring_team, ring_then_smart,  # noqa: E402
                            rotate, square_team)

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


def test_team():
    print("drone teams")
    drone = Drone(speed_ms=45, sweep_radius_m=1000, endurance_s=900)
    prior = BoatPrior()

    v1 = probability_greedy(300, drone, prior, n_particles=400, seed=5)
    solo = greedy_shared(300, drone, prior, 1, n_particles=400, seed=5)
    check("shared-map team of 1 flies exactly the v1 smart search path",
          len(solo) == 1 and np.allclose(solo[0].xy, v1.xy))

    team = greedy_shared(300, drone, prior, 3, n_particles=400, seed=5)
    steps = np.concatenate([np.linalg.norm(np.diff(p.xy, axis=0), axis=1) for p in team])
    check("shared-map drones fly at drone speed", steps.max() <= 90 + 1e-6)
    gaps = [np.linalg.norm(team[a].xy[-1] - team[b].xy[-1]) for a, b in [(0, 1), (0, 2), (1, 2)]]
    check("shared-map drones split up instead of flying together", min(gaps) > 2000,
          f"closest pair {min(gaps):.0f} m apart at the end")

    sq = expanding_square(0, drone)
    turned = rotate(sq, np.pi / 2)
    check("rotating 90 degrees turns east into north", np.allclose(turned.xy[10], [0, sq.xy[10][0]]))

    rings = ring_team(600, drone, prior, 3)
    ang = [np.degrees(np.arctan2(p.xy[-1][1], p.xy[-1][0])) for p in rings]
    diffs = sorted((a - ang[0]) % 360 for a in ang[1:])
    check("ring team is spread evenly (120 degrees apart)", np.allclose(diffs, [120, 240], atol=1e-6),
          f"{diffs}")
    check("square team has one path per drone", len(square_team(0, drone, 4)) == 4)

    rng = np.random.default_rng(9)
    boats = prior.sample(300, rng)
    each = [time_to_find(p, boats, 1000) for p in rings]
    check("team finds a boat when its FIRST drone does",
          np.array_equal(team_time_to_find(rings, boats, 1000), np.min(each, axis=0)))


def test_combos():
    print("combo strategies")
    drone = Drone(speed_ms=45, sweep_radius_m=1000, endurance_s=900)
    prior = BoatPrior((10, 30))
    v_ring = 20 * KNOT

    team = ring_then_smart(600, drone, prior, 3, n_particles=400, seed=3)
    check("ring-then-smart: one path per drone", len(team) == 3)
    reach_s = 600 * v_ring / (45 - v_ring)          # same math as the ring test, ~178 s
    early = int(reach_s / 2) - 2                     # a few steps before reaching the ring
    for j, p in enumerate(team):
        a = 2 * np.pi * j / 3
        along = p.xy[early] @ np.array([np.cos(a), np.sin(a)])
        off = np.linalg.norm(p.xy[early] - along * np.array([np.cos(a), np.sin(a)]))
        check(f"ring-then-smart drone {j + 1}: dashes straight out on its own heading",
              off < 1e-6 and along > 0, f"off-line {off:.1f} m")
    r = np.linalg.norm(team[0].xy, axis=1)
    reached = np.argmax(r >= v_ring * team[0].times)
    check("ring-then-smart: stops dashing when it reaches the ring",
          abs(team[0].times[reached] - 600 - reach_s) < 4, f"at {team[0].times[reached] - 600:.0f} s")
    steps = np.concatenate([np.linalg.norm(np.diff(p.xy, axis=0), axis=1) for p in team])
    check("ring-then-smart: never faster than the drone", steps.max() <= 90 + 1e-6)

    g = guard_plus_smart(600, drone, prior, 3, n_particles=400, seed=3)
    check("guard-plus-smart: one path per drone", len(g) == 3)
    check("guard-plus-smart: drone 1 flies the standard square", np.allclose(g[0].xy, expanding_square(600, drone).xy))
    # Needs boats that can still be near the tip (slow ones) for the guard to
    # cross anything off; with 10-30 kn boats and a 10 min tip there are none.
    slow = BoatPrior((0, 30))
    g2 = guard_plus_smart(120, drone, slow, 3, n_particles=400, seed=3)
    alone = greedy_shared(120, drone, slow, 2, n_particles=400, seed=3)
    check("guard-plus-smart: the guard's sweeps change where the searchers go",
          not np.allclose(g2[1].xy, alone[0].xy))
    try:
        guard_plus_smart(600, drone, prior, 1)
        check("guard-plus-smart refuses a 1-drone team", False)
    except ValueError:
        check("guard-plus-smart refuses a 1-drone team", True)


def test_boats_never_share_the_searchers_random_stream():
    print("boat randomness is separate from the searcher's")
    from search_sim import boat_rng
    prior = BoatPrior()
    clashes = []
    for seed in range(1, 6):
        guesses = prior.sample(50, np.random.default_rng(seed + 1000)).offset
        for age in [0, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 45, 60]:
            boats = prior.sample(50, boat_rng(seed, age)).offset
            if np.allclose(boats, guesses):
                clashes.append((seed, age))
    check("no seed/age gives boats identical to the planner's guesses", not clashes, str(clashes))
    a = prior.sample(5, boat_rng(1, 10)).offset
    b = prior.sample(5, boat_rng(1, 10)).offset
    check("same seed and age -> same boats", np.allclose(a, b))
    c = prior.sample(5, boat_rng(1, 20)).offset
    check("different ages -> different boats", not np.allclose(a, c))


def test_track_boats_match_straight_lines():
    print("recorded tracks")
    boats = BoatPrior().sample(40, np.random.default_rng(3))
    dt, horizon = 10.0, 4200.0
    grid = np.arange(0, horizon + dt, dt)
    tracks = TrackBoats(xy=boats.at(grid), dt_s=dt)
    t = np.array([0.0, 5.0, 123.4, 4000.0])
    check("a straight track replays like a straight-line boat", np.allclose(tracks.at(t), boats.at(t)))
    check("scalar time works too", np.allclose(tracks.at(77.7), boats.at(77.7)))
    path = probability_greedy(600, Drone(), BoatPrior(), seed=1)
    a = time_to_find(path, boats, 1000.0)
    b = time_to_find(path, tracks, 1000.0)
    check("same find times for tracks and straight lines", np.allclose(a, b, equal_nan=True) and
          np.array_equal(np.isfinite(a), np.isfinite(b)))
    turn = TrackBoats(xy=np.array([[[0, 0], [100, 0], [100, 100]]], float), dt_s=10.0)
    check("a turning track goes round the corner", np.allclose(turn.at(15.0), [[100, 50]]))
    check("after the track ends the boat stays put", np.allclose(turn.at(99.0), [[100, 100]]))


def test_track_prior_from_fake_ais():
    print("guessing from recorded tracks")
    from cuesearch.ais import TrackPrior, sample_boats
    t = np.arange(0.0, 7200.0, 60.0)
    lat = 30.0 + t * (5 * KNOT) / 110_540.0       # heading due north at 5 kn
    lon = np.full_like(t, -80.0)
    tracks = {"a": (t, lat, lon, np.full_like(t, 5.0))}
    boats, info = sample_boats(tracks, 30, 1800.0, np.random.default_rng(0), cue_error_m=0.0)
    check("a 5 kn northbound track samples as ~5 kn", np.allclose(info["straight_line_kn"], 5.0, atol=0.05))
    check("and goes north", np.allclose(boats.at(1800.0)[:, 1], 1800.0 * 5 * KNOT, rtol=0.01) and
          np.allclose(boats.at(1800.0)[:, 0], 0.0, atol=1.0))
    stopped = {"b": (t, np.full_like(t, 30.0), lon, np.zeros_like(t))}
    try:
        sample_boats(stopped, 5, 1800.0, np.random.default_rng(0))
        check("a boat that never moves is never used", False)
    except ValueError:
        check("a boat that never moves is never used", True)
    prior = TrackPrior(tracks, 1800.0, cue_error_m=0.0)
    check("TrackPrior samples like a BoatPrior", len(prior.sample(7, np.random.default_rng(1))) == 7)
    path = probability_greedy(300, Drone(), prior, seed=1)
    check("the smart search runs on a TrackPrior", len(path.times) > 10)


def test_fab():
    print("Brown's FAB (classic optimal search)")
    from cuesearch.fab import fab_plan, water_fill
    g = np.array([4.0, 2.0, 1.0, 0.0, 0.5])
    e = water_fill(g, 3.0, 1.0)
    check("water-filling spends exactly the budget", abs(e.sum() - 3.0) < 1e-9)
    check("no effort where nothing can be", e[3] == 0)
    obj = lambda x: (g * (1 - np.exp(-x))).sum()
    rng = np.random.default_rng(0)
    others = [rng.dirichlet(np.ones(5) * (g > 0) + 1e-9) * 3.0 for _ in range(300)]
    check("water-filling beats 300 random splits", all(obj(e) >= obj(o) - 1e-9 for o in others))
    check("equal cells get equal effort", np.allclose(water_fill(np.ones(4), 2.0, 1.0), 0.5))

    still = BoatPrior(speed_kn=(0.0, 0.0), cue_error_m=0.0)
    d = Drone(speed_ms=10.0, sweep_radius_m=50.0, endurance_s=600.0)
    plan = fab_plan(0.0, d, still, n_particles=50, cell_m=100.0, step_s=60.0)
    expect = 1 - np.exp(-(2 * 50 * 10 * 600) / 100.0 ** 2)
    p = plan.p_found(still.sample(5, np.random.default_rng(1))).mean()
    check("a boat that never moves gets all the effort", abs(p - expect) < 1e-9, f"{p:.4f} vs {expect:.4f}")

    plan = fab_plan(300.0, Drone(), BoatPrior(), n_particles=800, iters=15, seed=2)
    k = len(plan.times)
    check("each step spends the drones' sweep budget",
          np.allclose(plan.effort.sum(axis=1), 2 * 1000.0 * 45.0 * 60.0))
    parts = BoatPrior().sample(800, np.random.default_rng(2))
    fab_p = plan.p_found(parts).mean()
    flat = plan.effort.copy()
    for j in range(k):
        used = flat[j] > 0
        flat[j] = np.where(used, flat[j].sum() / used.sum(), 0.0)
    flat_plan = type(plan)(plan.times, plan.origin, plan.cell_m, plan.shape, flat)
    check("FAB beats spreading the same cells evenly (on its own guesses)",
          fab_p >= flat_plan.p_found(parts).mean() - 1e-9,
          f"{fab_p:.3f} vs {flat_plan.p_found(parts).mean():.3f}")


def test_intercept():
    print("patrol boat intercept")
    from cuesearch.intercept import intercept_times, patrol_position
    still = one_boat(5000, 0, 0, 0)
    start = (20000.0, 0.0)
    t = intercept_times(still, np.array([600.0]), 10000.0, start, 10.0, "wait", dt=1.0)
    check("waiting patrol: 15 km at 10 m/s after a find at 600 s -> ~2100 s", abs(t[0] - 2100) <= 1, str(t))
    t = intercept_times(still, np.array([600.0]), 10000.0, start, 10.0, "tip", dt=1.0)
    check("patrol heading to the tip is 6 km closer by then -> ~1500 s", abs(t[0] - 1500) <= 1, str(t))
    check("patrol stops at the tip spot", np.allclose(patrol_position(np.array([5000.0]), start, 10.0, "tip"), 0))
    fleeing = one_boat(0, 0, -20, 0)
    t = intercept_times(fleeing, np.array([600.0]), 20000.0, start, 10.0, "tip")
    check("a boat faster than the patrol, running away, is never caught", np.isinf(t[0]))
    t = intercept_times(still, np.array([600.0]), 1000.0, start, 10.0, "tip")
    check("no catch if the drones stop watching first", np.isinf(t[0]))
    t = intercept_times(still, np.array([np.inf]), 10000.0, start, 10.0, "tip")
    check("a boat never found is never caught", np.isinf(t[0]))
    tracks = TrackBoats(xy=still.at(np.arange(0, 10001, 10.0)), dt_s=10.0)
    t2 = intercept_times(tracks, np.array([600.0]), 10000.0, start, 10.0, "tip", dt=1.0)
    check("works the same for recorded tracks", abs(t2[0] - 1500) <= 1, str(t2))


def test_night_camera():
    print("night camera that sometimes misses")
    boats = BoatPrior().sample(400, np.random.default_rng(4))
    path = probability_greedy(300, Drone(), BoatPrior(), seed=1)
    day = time_to_find(path, boats, 1000.0)
    same = time_to_find(path, boats, 1000.0, p_detect=1.0, rng=np.random.default_rng(0))
    check("p_detect = 1 is exactly the daytime camera", np.array_equal(day, same))
    never = time_to_find(path, boats, 1000.0, p_detect=0.0, rng=np.random.default_rng(0))
    check("p_detect = 0 finds nothing", np.isinf(never).all())
    half = time_to_find(path, boats, 1000.0, p_detect=0.5, rng=np.random.default_rng(0))
    check("a missing camera never finds a boat the perfect one didn't, or sooner",
          np.all(half[np.isfinite(half)] >= day[np.isfinite(half)]) and np.isfinite(day[np.isfinite(half)]).all())
    still = one_boat(0, 0, 0, 0)
    hover = type(path)(times=np.arange(0, 100.0), xy=np.zeros((100, 2)))
    got = np.mean([np.isfinite(time_to_find(hover, still, 100.0, p_detect=0.5, rng=np.random.default_rng(k)))[0]
                   for k in range(400)])
    check("hovering over one boat is ONE pass: found ~50% of the time", 0.43 < got < 0.57, f"{got:.2f}")
    day_plan = greedy_shared(300, Drone(), BoatPrior(), 3, seed=1)
    night_plan = greedy_shared(300, Drone(), BoatPrior(), 3, seed=1, p_detect=0.7)
    check("a planner told its camera misses flies a different path",
          not all(np.allclose(a.xy, b.xy) for a, b in zip(day_plan, night_plan)))


def test_tello_lookpoints():
    print("Tello look-point plans")
    from cuesearch.lookpoints import (Room, TelloSpec, find_times, plan_time, sample_targets,
                                      smart_plan, spiral_plan)
    room, spec = Room(), TelloSpec()
    tg = sample_targets(2000, np.random.default_rng(1), room)
    r = np.linalg.norm(tg, axis=1)
    check("box spots are 0.8-2.0 m from the tip and inside the room",
          r.min() >= 0.8 - 1e-9 and r.max() <= 2.0 + 1e-9 and room.inside(tg).all())
    for name, pts in (("spiral", spiral_plan(room, spec, 70)), ("smart", smart_plan(room, spec, 70))):
        p = np.array(pts)
        check(f"{name}: every look point is inside the room", len(p) > 0 and room.inside(p).all())
        check(f"{name}: fits the time budget", plan_time(pts, spec) <= 70 + 1e-9, f"{plan_time(pts, spec):.1f}")
        legs = np.linalg.norm(np.diff(np.vstack([[0, 0], p]), axis=0), axis=1)
        check(f"{name}: no move shorter than the Tello allows", legs.min() >= spec.min_leg_m - 1e-9)
    one = np.array([[1.0, 0.0], [2.0, 0.0]])
    ft = find_times([(1.0, 0.5)], one, spec)
    expect = np.hypot(1.0, 0.5) / spec.speed_ms + spec.move_overhead_s + spec.spin_s
    check("a box within range of a scan is found when that scan ends", abs(ft[0] - expect) < 1e-9)
    check("a box out of range is missed", np.isinf(ft[1]))


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
    test_team()
    test_combos()
    test_track_boats_match_straight_lines()
    test_tello_lookpoints()
    test_night_camera()
    test_intercept()
    test_fab()
    test_track_prior_from_fake_ais()
    test_boats_never_share_the_searchers_random_stream()
    test_wilson()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("All tests passed.")
