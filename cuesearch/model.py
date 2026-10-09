"""The world: a boat that was spotted, and a drone sent to find it.

Everything is in metres and seconds. Time t=0 is the moment the satellite
saw the boat. The cue says "boat at (0, 0)" -- that is all the drone knows.

The boat model is deliberately simple for V1: it picks a random heading and
a random speed at t=0 and holds them. Real boats turn, slow down, and loiter.
That simplification is the first thing to attack in failure analysis.
"""

from dataclasses import dataclass

import numpy as np

KNOT = 0.514444  # metres per second


@dataclass
class Boats:
    """A batch of boats (or boat hypotheses). Row i is one boat."""

    offset: np.ndarray    # (n, 2) where the boat really was at t=0 (cue error)
    velocity: np.ndarray  # (n, 2) metres per second, constant

    def at(self, t):
        """Positions at time t. Scalar t -> (n, 2). Array t of shape (k,) -> (n, k, 2)."""
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            return self.offset + self.velocity * t
        return self.offset[:, None, :] + self.velocity[:, None, :] * t[None, :, None]

    def __len__(self):
        return len(self.offset)

    def __getitem__(self, sl):
        return Boats(offset=self.offset[sl], velocity=self.velocity[sl])


@dataclass
class TrackBoats:
    """Boats that follow recorded paths (real ship tracks) instead of straight lines.

    xy[i, j] is boat i's position at time j * dt_s (metres, relative to the cue
    point, t=0 = the satellite pass). Between samples the boat moves in a straight
    line; after the last sample it stays put, so tracks must cover the whole search.
    """

    xy: np.ndarray  # (n, K, 2)
    dt_s: float

    def at(self, t):
        t = np.asarray(t, dtype=float)
        k = self.xy.shape[1]
        u = np.clip(t / self.dt_s, 0, k - 1)
        j = np.minimum(np.floor(u).astype(int), k - 2)
        w = (u - j)[..., None]
        if t.ndim == 0:
            return self.xy[:, j] * (1 - w) + self.xy[:, j + 1] * w
        return self.xy[:, j] * (1 - w[None]) + self.xy[:, j + 1] * w[None]

    def __len__(self):
        return len(self.xy)

    def __getitem__(self, sl):
        return TrackBoats(xy=self.xy[sl], dt_s=self.dt_s)


@dataclass
class BoatPrior:
    """What we believe about a boat we haven't found yet.

    speed_kn:    boat speed is uniform in this range. A panga or go-fast boat
                 cruises roughly 10-30 knots; change this, it matters a lot.
    cue_error_m: how wrong the satellite's position fix is (1 standard deviation).
    """

    speed_kn: tuple = (10.0, 30.0)
    cue_error_m: float = 100.0

    def sample(self, n, rng):
        heading = rng.uniform(0.0, 2 * np.pi, n)
        speed = rng.uniform(self.speed_kn[0], self.speed_kn[1], n) * KNOT
        direction = np.stack([np.cos(heading), np.sin(heading)], axis=1)
        offset = rng.normal(0.0, self.cue_error_m, (n, 2))
        return Boats(offset=offset, velocity=speed[:, None] * direction)


@dataclass
class Drone:
    """The searcher.

    speed_ms:       45 m/s is about 100 mph, roughly a fixed-wing like the F10.
                    A quadcopter is more like 15 m/s.
    sweep_radius_m: the drone sees the boat if it's within this distance.
                    This is a "cookie-cutter" sensor -- perfect inside, blind
                    outside. Real cameras fade with range. The 1000 m default
                    is a guess; it's a question to ask a real operator.
    endurance_s:    how long the drone can search after it reaches the cue point.
    """

    speed_ms: float = 45.0
    sweep_radius_m: float = 1000.0
    endurance_s: float = 3600.0


@dataclass
class SearchPath:
    """Where the drone was at each time step. times are seconds since the cue."""

    times: np.ndarray  # (N,)
    xy: np.ndarray     # (N, 2)
