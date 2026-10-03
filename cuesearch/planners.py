"""Three ways to search. All start at the cue point at the moment the drone arrives.

expanding_square   -- the BASELINE. A standard search pattern from the
                      IAMSAR manual: fly outward in a growing square spiral
                      centred on the last known position. It ignores the fact
                      that the boat has been moving away the whole time.

ring_search        -- the FAIR BASELINE. Knows the boat has moved, so it
                      spirals around the ring where the boat probably is.

probability_greedy -- the CHALLENGER. Keep a cloud of "where could the boat
                      be?" guesses (particles). Every 30 s, look at 16 possible
                      headings and pick the one that would sweep the most
                      guesses in the next couple of minutes. Guesses the drone
                      has already swept are deleted, because if the boat had
                      been there we'd have seen it. That deletion is the core
                      idea of Bayesian search: looking and finding nothing is
                      also information.

Why a plan doesn't depend on the true boat: until the drone sees the boat, it
learns nothing about where the boat is except "not where I've already looked."
So each strategy flies one fixed path for a given cue age, and we can test that
one path against thousands of random boats at once.
"""

import numpy as np

from .model import KNOT, SearchPath


def _fly_waypoints(waypoints, speed, dt, n_steps, t0):
    """Sample a constant-speed flight along a polyline every dt seconds."""
    pts = np.asarray(waypoints, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.minimum(speed * dt * np.arange(n_steps), arc[-1])
    xy = np.stack([np.interp(s, arc, pts[:, 0]), np.interp(s, arc, pts[:, 1])], axis=1)
    return SearchPath(times=t0 + dt * np.arange(n_steps), xy=xy)


def _n_steps(drone, dt):
    return int(round(drone.endurance_s / dt)) + 1


def expanding_square(cue_age_s, drone, dt=2.0):
    """Legs of 1, 1, 2, 2, 3, 3... track spacings, turning left each time."""
    spacing = 2.0 * drone.sweep_radius_m  # adjacent passes just touch
    total = drone.speed_ms * drone.endurance_s
    waypoints = [(0.0, 0.0)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    flown, leg = 0.0, 0
    while flown < total:
        length = spacing * (leg // 2 + 1)
        dx, dy = directions[leg % 4]
        x, y = waypoints[-1]
        waypoints.append((x + dx * length, y + dy * length))
        flown += length
        leg += 1
    return _fly_waypoints(waypoints, drone.speed_ms, dt, _n_steps(drone, dt), cue_age_s)


def ring_search(cue_age_s, drone, prior, dt=2.0):
    """The FAIR baseline: a simple pattern that knows the boat has moved.

    The boat could have gone any direction, so "where it probably is" isn't a
    point -- it's a ring around the cue, growing at the boat's average speed.
    (The average of all the guesses is the cue point itself, so flying to
    "the average position" is just the expanding square again.)

    So: fly straight out until we reach the ring, then spiral around it,
    moving outward exactly as fast as the ring grows. Uses only the average
    boat speed -- less information than probability_greedy gets.
    """
    v_ring = float(np.mean(prior.speed_kn)) * KNOT
    v_drone = drone.speed_ms
    n_steps = _n_steps(drone, dt)
    times = cue_age_s + dt * np.arange(n_steps)
    xy = np.zeros((n_steps, 2))

    pos = np.zeros(2)
    on_ring = False
    for k in range(1, n_steps):
        t = times[k - 1]
        r = np.linalg.norm(pos)
        radial = pos / r if r > 1.0 else np.array([1.0, 0.0])
        ring_r = v_ring * t
        if not on_ring and r < ring_r and v_drone > v_ring:
            vel = radial * v_drone                   # still flying out to the ring
        else:
            on_ring = True
            # Keep pace with the ring, and correct any drift onto it.
            v_r = np.clip(v_ring + (ring_r - r) / dt, -v_drone, v_drone)
            v_t = np.sqrt(max(v_drone ** 2 - v_r ** 2, 0.0))
            tangent = np.array([-radial[1], radial[0]])
            vel = v_r * radial + v_t * tangent
        pos = pos + vel * dt
        xy[k] = pos
    return SearchPath(times=times, xy=xy)


def probability_greedy(cue_age_s, drone, prior, dt=2.0, n_particles=2000,
                       n_headings=16, replan_s=30.0, lookahead_s=120.0,
                       lookahead_dt=10.0, seed=0, return_particles=False):
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
    xy = np.zeros((n_steps, 2))
    steps_per_replan = max(1, int(round(replan_s / dt)))

    def sweep(pos, t):
        d2 = ((particles.at(t) - pos) ** 2).sum(axis=1)
        alive[d2 <= r2] = False

    def choose_heading(pos, t):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            return None
        future = particles.at(t + look_t)[idx]                          # (A, L, 2)
        drone_future = pos + dirs[:, None, :] * (drone.speed_ms * look_t)[None, :, None]  # (H, L, 2)
        d2 = ((drone_future[:, None, :, :] - future[None, :, :, :]) ** 2).sum(axis=-1)    # (H, A, L)
        score = (d2 <= r2).any(axis=2).sum(axis=1)                      # (H,)
        if score.max() > 0:
            return dirs[int(np.argmax(score))]
        # Nothing reachable within the lookahead: head for the nearest guess.
        target = particles.at(t + lookahead_s)[idx]
        nearest = target[np.argmin(((target - pos) ** 2).sum(axis=1))]
        v = nearest - pos
        return v / (np.linalg.norm(v) + 1e-9)

    pos = np.zeros(2)
    sweep(pos, times[0])
    heading = dirs[0]
    k = 0
    while k < n_steps - 1:
        chosen = choose_heading(pos, times[k])
        if chosen is not None:
            heading = chosen
        for _ in range(steps_per_replan):
            if k >= n_steps - 1:
                break
            pos = pos + heading * step
            k += 1
            xy[k] = pos
            sweep(pos, times[k])

    path = SearchPath(times=times, xy=xy)
    if return_particles:
        return path, particles, alive
    return path
