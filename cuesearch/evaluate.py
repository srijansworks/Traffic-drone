"""Scoring: did the drone find the boat, and how long did it take?"""

import numpy as np


def time_to_find(path, boats, sweep_radius_m, chunk=500, p_detect=1.0, rng=None):
    """Search time (seconds after arriving at the cue) until each boat is seen.

    Returns an array of length len(boats); np.inf means never found.

    p_detect: chance the camera actually spots a boat on each PASS over it (a pass
              = the boat coming inside the sweep radius, until it leaves again).
              1.0 = perfect daytime camera. Less than 1 (night, thermal) needs rng.
    """
    r2 = sweep_radius_m ** 2
    out = np.full(len(boats), np.inf)
    search_t = path.times - path.times[0]
    for start in range(0, len(boats), chunk):
        sl = slice(start, start + chunk)
        pos = boats[sl].at(path.times)                                 # (n, N, 2)
        hit = ((pos - path.xy[None, :, :]) ** 2).sum(axis=-1) <= r2    # (n, N)
        if p_detect < 1.0:
            # each pass starts where hit turns on; keep only the passes that succeed
            starts = hit & ~np.concatenate([np.zeros((hit.shape[0], 1), bool), hit[:, :-1]], axis=1)
            hit = starts & (rng.random(hit.shape) < p_detect)
        found = hit.any(axis=1)
        first = hit.argmax(axis=1)
        out[sl][found] = search_t[first[found]]
    return out


def team_time_to_find(paths, boats, sweep_radius_m, p_detect=1.0, rng=None):
    """Like time_to_find, but for several drones: found when the FIRST one sees it.
    With p_detect < 1, each drone's passes succeed or fail independently."""
    return np.min([time_to_find(p, boats, sweep_radius_m, p_detect=p_detect, rng=rng) for p in paths], axis=0)


def wilson(k, n, z=1.96):
    """95% confidence interval for a success rate k/n.

    Better than p +/- 1.96*sqrt(p(1-p)/n), which breaks near 0% and 100%
    -- exactly where search results tend to live.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
