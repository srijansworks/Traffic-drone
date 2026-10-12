"""The search, shrunk to a room, for a real Tello.

The full-size search assumes a downward camera that sees a 1 km circle as it flies.
A Tello only has a front camera, so here the drone stops at LOOK POINTS and spins
360 degrees at each one, which sees a full circle around that point. A find only
counts if the marker is within `sense_m` (measured from its size in the image),
the room-sized version of "the camera sees boats within 1 km".

Units are metres. Frame: x = forward from the takeoff spot, y = left, origin = the
takeoff spot = the "tip spot" where the satellite last saw the boat.

The "boat" is a box with a marker on every side. A helper places it at a random
spot r_min..r_max from the tip, in a random direction, inside the room: like a boat
that moved after the satellite pass and then stopped.

Two plans, same rules as the full-size project:
  spiral_plan -- the standard pattern: look points on a square spiral from the tip.
  smart_plan  -- the Bayesian search: a cloud of guesses; each next look point is the
                 one that checks the most still-possible guesses per second of flying
                 and spinning; guesses inside a finished scan are crossed off.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Room:
    """Where the drone may fly: a box around the takeoff spot, minus a safety margin."""

    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0

    def inside(self, p, pad=0.0):
        p = np.asarray(p, dtype=float)
        return ((p[..., 0] >= self.x_min + pad) & (p[..., 0] <= self.x_max - pad) &
                (p[..., 1] >= self.y_min + pad) & (p[..., 1] <= self.y_max - pad))


@dataclass
class TelloSpec:
    """How the real Tello behaves on these moves (rough; time a few runs and update)."""

    speed_ms: float = 0.5        # 'go' command speed
    move_overhead_s: float = 1.5  # start, stop and settle around each move
    spin_s: float = 9.0          # four 90-degree turns, with a pause after each
    sense_m: float = 0.75        # a find counts if the marker is this close
    min_leg_m: float = 0.2       # the Tello refuses 'go' moves shorter than 20 cm


def sample_targets(n, rng, room, r_min=0.8, r_max=2.0):
    """Where the helper might put the box: r_min..r_max from the tip, random direction,
    inside the room. Rejection-sampled so every spot inside that ring is equally likely."""
    out = np.empty((0, 2))
    while len(out) < n:
        k = 4 * (n - len(out)) + 16
        r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2, k))   # uniform over the ring's area
        a = rng.uniform(0, 2 * np.pi, k)
        p = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
        out = np.vstack([out, p[room.inside(p)]])
    return out[:n]


def leg_time(a, b, spec):
    d = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
    return d / spec.speed_ms + spec.move_overhead_s


def plan_time(points, spec, start=(0.0, 0.0)):
    """Seconds to fly to every look point in order and spin at each."""
    t, here = 0.0, np.asarray(start, dtype=float)
    for p in points:
        t += leg_time(here, p, spec) + spec.spin_s
        here = np.asarray(p, dtype=float)
    return t


def spiral_plan(room, spec, budget_s):
    """Standard square spiral of look points around the tip, spaced so scans just overlap."""
    step = 2 * spec.sense_m * 0.8
    pts, x, y, leg = [], 0.0, 0.0, 0
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    while leg < 40:
        n = leg // 2 + 1
        dx, dy = dirs[leg % 4]
        for _ in range(n):
            x, y = x + dx * step, y + dy * step
            if room.inside((x, y)):
                pts.append((x, y))
        leg += 1
    return _trim_to_budget(pts, spec, budget_s)


def smart_plan(room, spec, budget_s, r_min=0.8, r_max=2.0, n_guesses=3000, grid_m=0.25, seed=0):
    """Greedy Bayesian look points: most still-possible guesses checked per second."""
    rng = np.random.default_rng(seed)
    guesses = sample_targets(n_guesses, rng, room, r_min, r_max)
    alive = np.ones(len(guesses), dtype=bool)
    xs = np.arange(room.x_min, room.x_max + 1e-9, grid_m)
    ys = np.arange(room.y_min, room.y_max + 1e-9, grid_m)
    cand = np.array([(x, y) for x in xs for y in ys])
    cand = cand[room.inside(cand)]
    # which guesses each candidate's scan would check
    covers = (((cand[:, None, :] - guesses[None, :, :]) ** 2).sum(-1) <= spec.sense_m ** 2)
    here, t, pts = np.zeros(2), 0.0, []
    while True:
        gain = covers[:, alive].sum(axis=1).astype(float)
        dist = np.linalg.norm(cand - here, axis=1)
        cost = dist / spec.speed_ms + spec.move_overhead_s + spec.spin_s
        ok = (dist >= spec.min_leg_m) & (t + cost <= budget_s) & (gain > 0)
        if not ok.any():
            break
        score = np.where(ok, gain / cost, -1.0)
        i = int(np.argmax(score))
        pts.append(tuple(cand[i]))
        alive &= ~covers[i]
        t += cost[i]
        here = cand[i]
    return pts


def _trim_to_budget(points, spec, budget_s):
    out, here, t = [], np.zeros(2), 0.0
    for p in points:
        if np.linalg.norm(np.asarray(p) - here) < spec.min_leg_m:
            continue
        c = leg_time(here, p, spec) + spec.spin_s
        if t + c > budget_s:
            break
        out.append(p)
        t += c
        here = np.asarray(p, dtype=float)
    return out


def find_times(points, targets, spec):
    """Seconds after takeoff until each target is seen (end of the scan that sees it); inf = missed."""
    out = np.full(len(targets), np.inf)
    here, t = np.zeros(2), 0.0
    for p in points:
        t += leg_time(here, p, spec) + spec.spin_s
        seen = np.isinf(out) & (np.linalg.norm(targets - np.asarray(p), axis=1) <= spec.sense_m)
        out[seen] = t
        here = np.asarray(p, dtype=float)
    return out
