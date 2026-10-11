"""After the drones find the boat: can a patrol boat actually get there?

Finding the boat isn't the end. A drone keeps watching it ("custody") and a patrol
boat drives out to stop and board it. The boat only counts as CAUGHT if the patrol
reaches it while a drone is still watching.

Model (all simple on purpose, like the rest of the project):
  - The patrol boat starts at patrol_start (metres from the tip spot) at t=0, the
    satellite pass, and moves at a constant speed in a straight line.
  - Policy "tip":  it heads for the tip spot straight away and waits there.
    Policy "wait": it stays put until a drone finds the boat.
  - Once the boat is found, the patrol knows exactly where it is (the drone is
    watching) and takes the fastest straight-line course to meet it.
  - Custody ends at custody_end_s (seconds since the satellite pass): when the
    drones' batteries run out, or later if relief drones take over.
"""

import numpy as np


def patrol_position(t, start, speed_ms, policy):
    """Where the patrol is at times t (array), before it knows where the boat is."""
    t = np.asarray(t, dtype=float)
    start = np.asarray(start, dtype=float)
    if policy == "wait":
        return np.broadcast_to(start, t.shape + (2,)).copy()
    if policy != "tip":
        raise ValueError(f"unknown patrol policy {policy!r}")
    dist = np.linalg.norm(start)
    gone = np.minimum(speed_ms * t, dist)[..., None]
    return start - start / max(dist, 1e-9) * gone


def intercept_times(boats, found_at_s, custody_end_s, start, speed_ms, policy="tip", dt=10.0):
    """Seconds since the satellite pass when the patrol reaches each boat; inf if never.

    found_at_s: (n,) seconds since the satellite pass when a drone first saw each
                boat (inf = never found).
    """
    n = len(boats)
    out = np.full(n, np.inf)
    found = np.isfinite(found_at_s) & (found_at_s <= custody_end_s)
    if not found.any():
        return out
    idx = np.flatnonzero(found)
    tf = found_at_s[idx]
    p0 = patrol_position(tf, start, speed_ms, policy)                  # (m, 2)
    grid = np.arange(0.0, custody_end_s - tf.min() + dt, dt)          # time after the find
    when = tf[:, None] + grid[None, :]                                 # (m, G)
    sub = boats[idx]
    pos = np.stack([sub[i:i + 1].at(when[i])[0] for i in range(len(idx))])  # (m, G, 2)
    reach = speed_ms * grid[None, :]
    ok = (np.linalg.norm(pos - p0[:, None, :], axis=-1) <= reach) & (when <= custody_end_s)
    hit = ok.any(axis=1)
    out[idx[hit]] = when[hit, ok[hit].argmax(axis=1)]
    return out
