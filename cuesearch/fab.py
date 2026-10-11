"""Brown's FAB algorithm: the classic "best possible" search plan for a moving target.

Brown (1980), "Optimal search for a moving target in discrete time and space",
Operations Research 28:1275-1289. The Navy and Coast Guard search theory this
project keeps getting compared to is built on it.

The idea: split the sea into cells and time into steps. Each step, the searchers
have a fixed amount of search effort (area they can sweep). FAB decides how much
effort to put in every cell at every step so the chance of finding the boat is as
high as possible. Detection is "random search" (Koopman's exponential law): putting
effort e into a cell of area A finds a boat that's in it with chance 1 - exp(-e/A).

Not a flyable plan: effort can be split across any cells, anywhere, every step.
Real drones can't teleport or be in ten places at once; they have to fly a path.

Not a true ceiling either, for THIS project's sensor. FAB is optimal under random-
search detection (one pass over a boat finds it only ~63% of the time). The drones
here have a perfect 1 km camera, which beats that law, so a flyable path can score
higher than FAB. Read FAB as "what the classic optimal-search theory gives".

Needs MANY particles (default 300,000). With a few thousand spread over a big area
it piles effort exactly where its own guesses sit and misses the gaps between them:
experiment 19 first scored 91% on its own guesses but 27% on fresh boats.

Here the boat's motion is the same particle cloud the smart search uses: each
particle is one possible path. Brown's forward-and-backward pass then becomes:
for one time step at a time, hold every other step's effort fixed, work out how
likely each particle is to still be unfound, and re-spread that step's effort
optimally (water-filling). Repeat until nothing changes.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FabPlan:
    times: np.ndarray    # (K,) seconds since the cue, middle of each step
    origin: np.ndarray   # (2,) metres, corner of cell (0, 0)
    cell_m: float
    shape: tuple         # (nx, ny)
    effort: np.ndarray   # (K, nx*ny) swept area placed in each cell each step, m^2

    def cells(self, xy):
        """Flat cell index for positions (..., 2); -1 if outside the grid."""
        ij = np.floor((xy - self.origin) / self.cell_m).astype(int)
        nx, ny = self.shape
        ok = (ij[..., 0] >= 0) & (ij[..., 0] < nx) & (ij[..., 1] >= 0) & (ij[..., 1] < ny)
        return np.where(ok, ij[..., 0] * ny + ij[..., 1], -1)

    def p_found(self, boats):
        """Chance each boat is found by the end of the plan."""
        c = self.cells(boats.at(self.times))                    # (n, K)
        k = np.broadcast_to(np.arange(len(self.times)), c.shape)
        e = np.where(c >= 0, self.effort[k, np.maximum(c, 0)], 0.0)
        return 1.0 - np.exp(-e.sum(axis=1) / self.cell_m ** 2)


def water_fill(g, budget, area):
    """Best way to spread `budget` effort over cells whose (unnormalised) chance of
    holding a still-unfound boat is g, under exponential detection.

    Maximises sum g_c (1 - exp(-e_c/area)) with sum e_c = budget. The answer is
    e_c = area * max(0, ln(g_c / lam)): fill the likeliest cells first, like
    pouring water into a container whose floor is shaped by -ln(g).
    """
    e = np.zeros_like(g, dtype=float)
    pos = np.flatnonzero(g > 0)
    if pos.size == 0 or budget <= 0:
        return e
    order = pos[np.argsort(-g[pos])]
    lg = np.log(g[order])
    m = np.arange(1, len(order) + 1)
    ln_lam = (np.cumsum(lg) - budget / area) / m        # if the top m cells are used
    # use the largest m whose m-th cell still sits above the water line
    m_use = int(np.flatnonzero(lg > ln_lam)[-1]) + 1
    e[order[:m_use]] = area * (lg[:m_use] - ln_lam[m_use - 1])
    return e


def fab_plan(cue_age_s, drone, prior, n_drones=1, cell_m=500.0, step_s=60.0,
             n_particles=300_000, iters=25, seed=0):
    """FAB plan for the search window [cue_age_s, cue_age_s + endurance]."""
    rng = np.random.default_rng(seed)
    particles = prior.sample(n_particles, rng)
    k_steps = int(round(drone.endurance_s / step_s))
    times = cue_age_s + step_s * (np.arange(k_steps) + 0.5)
    budget = n_drones * 2 * drone.sweep_radius_m * drone.speed_ms * step_s   # m^2 per step
    area = cell_m ** 2

    pos = particles.at(times)                                   # (P, K, 2)
    lo = pos.reshape(-1, 2).min(axis=0) - cell_m
    hi = pos.reshape(-1, 2).max(axis=0) + cell_m
    shape = tuple(np.ceil((hi - lo) / cell_m).astype(int))
    plan = FabPlan(times=times, origin=lo, cell_m=cell_m, shape=shape,
                   effort=np.zeros((k_steps, shape[0] * shape[1])))
    cell = plan.cells(pos)                                      # (P, K), all inside
    n_cells = shape[0] * shape[1]

    load = np.zeros((n_particles, k_steps))                     # effort/area each particle sees
    for _ in range(iters):
        before = load.sum(axis=1).copy()
        for k in range(k_steps):
            other = load.sum(axis=1) - load[:, k]
            g = np.bincount(cell[:, k], weights=np.exp(-other), minlength=n_cells)
            e = water_fill(g, budget, area)
            plan.effort[k] = e
            load[:, k] = e[cell[:, k]] / area
        if np.abs(load.sum(axis=1) - before).max() < 1e-4:
            break
    return plan
