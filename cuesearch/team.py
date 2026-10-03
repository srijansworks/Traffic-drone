"""Several drones searching for the same boat. The boat counts as found the
first moment ANY drone sees it.

Every strategy gets the same number of drones, so the comparison stays fair:

square_team        -- each drone flies the standard expanding square, rotated
                      so their first legs head in different directions.
ring_team          -- drones fly out on evenly spaced headings and circle the
                      ring together, each one owning its own slice.
greedy_independent -- each drone runs the v1 smart search on its own, with its
                      own guesses. Nobody tells anyone where they've looked.
greedy_shared      -- one shared map of guesses. Every 30 s the drones pick
                      headings one at a time; guesses one drone is about to
                      cover are "claimed", so the next drone goes after
                      different ones. Water any drone checks is crossed off
                      for everyone.

Combos (built on greedy_shared):

ring_then_smart    -- drones first dash straight out to the ring on evenly
                      spaced headings (fast, like ring search), then switch to
                      shared-map smart search.
guard_plus_smart   -- one drone stays home and flies the standard square around
                      the tip location, catching boats that stopped or went
                      slow. The rest run shared-map smart search, and the
                      guard's sweeps cross off water on the shared map too.

The difference between the last two is the experiment: does sharing a map
matter, or do three drones do just as well searching separately?
"""

import numpy as np

from .model import KNOT, SearchPath
from .planners import _n_steps, expanding_square, probability_greedy, ring_search


def rotate(path, angle):
    """The same flight, turned by `angle` radians around the cue point."""
    c, s = np.cos(angle), np.sin(angle)
    xy = path.xy @ np.array([[c, s], [-s, c]])
    return SearchPath(times=path.times, xy=xy)


def _spread(n):
    return [2 * np.pi * k / n for k in range(n)]


def square_team(cue_age_s, drone, n_drones, dt=2.0):
    base = expanding_square(cue_age_s, drone, dt=dt)
    return [rotate(base, a) for a in _spread(n_drones)]


def ring_team(cue_age_s, drone, prior, n_drones, dt=2.0):
    # The ring is the same in every direction, so a rotated copy is an exact
    # ring search that started on a different heading.
    base = ring_search(cue_age_s, drone, prior, dt=dt)
    return [rotate(base, a) for a in _spread(n_drones)]


def greedy_independent(cue_age_s, drone, prior, n_drones, dt=2.0, seed=0, **kw):
    return [probability_greedy(cue_age_s, drone, prior, dt=dt, seed=seed + k, **kw)
            for k in range(n_drones)]


def greedy_shared(cue_age_s, drone, prior, n_drones, dt=2.0, n_particles=2000,
                  n_headings=16, replan_s=30.0, lookahead_s=120.0,
                  lookahead_dt=10.0, seed=0, dash_to_ring=False, fixed_paths=()):
    """Smart search with one shared map. With n_drones=1 (and no options) this
    flies exactly the same path as v1 probability_greedy (a test checks that).

    dash_to_ring: each drone first flies straight out on its own evenly spaced
                  heading until it reaches the ring the boat is probably on
                  (average assumed speed x time), then joins the smart search.
    fixed_paths:  pre-planned paths of other drones (e.g. a guard). They don't
                  make decisions, but whatever they sweep is crossed off the map.
    """
    rng = np.random.default_rng(seed)
    particles = prior.sample(n_particles, rng)
    alive = np.ones(n_particles, dtype=bool)
    r2 = drone.sweep_radius_m ** 2
    step = drone.speed_ms * dt

    angles = np.linspace(0.0, 2 * np.pi, n_headings, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)          # (H, 2)
    look_t = lookahead_dt * np.arange(1, int(lookahead_s / lookahead_dt) + 1)  # (L,)

    n_steps = _n_steps(drone, dt)
    times = cue_age_s + dt * np.arange(n_steps)
    xy = np.zeros((n_drones, n_steps, 2))
    steps_per_replan = max(1, int(round(replan_s / dt)))

    def sweep(pos, t):
        d2 = ((particles.at(t) - pos) ** 2).sum(axis=1)
        alive[d2 <= r2] = False

    def choose_heading(pos, t, claimed):
        """Best heading for one drone, ignoring guesses teammates already claimed.
        Returns (heading, indices of guesses this drone now claims)."""
        idx = np.flatnonzero(alive & ~claimed)
        if idx.size == 0:
            idx = np.flatnonzero(alive)
        if idx.size == 0:
            return None, idx
        future = particles.at(t + look_t)[idx]                          # (A, L, 2)
        drone_future = pos + dirs[:, None, :] * (drone.speed_ms * look_t)[None, :, None]  # (H, L, 2)
        d2 = ((drone_future[:, None, :, :] - future[None, :, :, :]) ** 2).sum(axis=-1)    # (H, A, L)
        covered = (d2 <= r2).any(axis=2)                                # (H, A)
        score = covered.sum(axis=1)
        if score.max() > 0:
            best = int(np.argmax(score))
            return dirs[best], idx[covered[best]]
        # Nothing reachable within the lookahead: head for the nearest guess.
        target = particles.at(t + lookahead_s)[idx]
        i = int(np.argmin(((target - pos) ** 2).sum(axis=1)))
        v = target[i] - pos
        return v / (np.linalg.norm(v) + 1e-9), idx[[i]]

    v_ring = float(np.mean(prior.speed_kn)) * KNOT

    def still_dashing(p, t):
        return dash_to_ring and drone.speed_ms > v_ring and np.linalg.norm(p) < v_ring * t

    pos = np.zeros((n_drones, 2))
    for j in range(n_drones):
        sweep(pos[j], times[0])
    for fp in fixed_paths:
        sweep(fp.xy[0], times[0])
    if dash_to_ring:
        spread = 2 * np.pi * np.arange(n_drones) / n_drones
        headings = np.stack([np.cos(spread), np.sin(spread)], axis=1)
    else:
        headings = np.tile(dirs[0], (n_drones, 1))
    dashing = np.array([still_dashing(pos[j], times[0]) for j in range(n_drones)])
    k = 0
    while k < n_steps - 1:
        claimed = np.zeros(n_particles, dtype=bool)
        for j in range(n_drones):
            if dashing[j]:
                continue
            chosen, mine = choose_heading(pos[j], times[k], claimed)
            if chosen is not None:
                headings[j] = chosen
                claimed[mine] = True
        for _ in range(steps_per_replan):
            if k >= n_steps - 1:
                break
            pos = pos + headings * step
            k += 1
            xy[:, k] = pos
            for j in range(n_drones):
                sweep(pos[j], times[k])
            for fp in fixed_paths:
                sweep(fp.xy[k], times[k])
            was = dashing.copy()
            dashing = np.array([was[j] and still_dashing(pos[j], times[k]) for j in range(n_drones)])
            if (was & ~dashing).any():
                break   # someone just reached the ring: replan now, don't overshoot

    return [SearchPath(times=times, xy=xy[j]) for j in range(n_drones)]


def ring_then_smart(cue_age_s, drone, prior, n_drones, dt=2.0, seed=0, **kw):
    return greedy_shared(cue_age_s, drone, prior, n_drones, dt=dt, seed=seed,
                         dash_to_ring=True, **kw)


def guard_plus_smart(cue_age_s, drone, prior, n_drones, dt=2.0, seed=0, **kw):
    if n_drones < 2:
        raise ValueError("guard_plus_smart needs at least 2 drones (1 guard + 1 searcher)")
    guard = expanding_square(cue_age_s, drone, dt=dt)
    return [guard] + greedy_shared(cue_age_s, drone, prior, n_drones - 1, dt=dt, seed=seed,
                                   fixed_paths=[guard], **kw)
