"""Real boat paths from AIS (ship tracking) data, turned into boats for the simulator.

Data: NOAA Marine Cadastre daily files (US waters), e.g.
https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_01.zip
Columns: MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO, CallSign,
VesselType, Status, Length, Width, Draft, Cargo, TransceiverClass.

How a real track becomes a test boat:
  1. Pick a vessel at random (each vessel equally likely, so one chatty boat with
     thousands of pings can't dominate).
  2. Pick a random moment t0 in one of its unbroken stretches of track (no gap over
     MAX_GAP_S) that is moving at >= min_sog_kn -- that's the "satellite pass".
     A boat tied up in port isn't what a dark-vessel tip is about.
  3. Its position at t0, plus the satellite's position error, is the cue point.
     The next horizon_s seconds of track become the boat's path.

Caveat: these boats broadcast AIS, so they're legal boats going about their day,
not boats trying to escape. They're real about speed, turning, and loitering.
"""

import csv
import io
import zipfile

import numpy as np

from .model import KNOT, TrackBoats

MAX_GAP_S = 300.0
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQUATOR = 111_320.0


def load_tracks(zip_path, vessel_types=(30,)):
    """Read one daily file. Returns {mmsi: (t_seconds, lat, lon, sog_kn)} sorted by time.

    vessel_types: AIS type codes to keep. 30 = fishing.
    """
    keep = {str(v) for v in vessel_types}
    rows = {}
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
            header = next(reader)
            col = {h: i for i, h in enumerate(header)}
            iv, im, it = col["VesselType"], col["MMSI"], col["BaseDateTime"]
            ila, ilo, isog = col["LAT"], col["LON"], col["SOG"]
            for r in reader:
                if r[iv] not in keep:
                    continue
                ts = r[it]  # 2023-01-01T00:00:01
                secs = int(ts[11:13]) * 3600 + int(ts[14:16]) * 60 + int(ts[17:19])
                rows.setdefault(r[im], []).append((secs, float(r[ila]), float(r[ilo]), float(r[isog])))
    tracks = {}
    for mmsi, pts in rows.items():
        a = np.array(sorted(pts))
        _, first = np.unique(a[:, 0], return_index=True)  # drop duplicate timestamps
        tracks[mmsi] = a[first].T
    return tracks


def _windows(t, sog, horizon_s, min_sog_kn):
    """Ping indices i where a path of horizon_s can start: moving, and no gap before t[i] + horizon_s."""
    gap_after = np.flatnonzero(np.diff(t) > MAX_GAP_S)       # break between i and i+1
    seg_end = np.empty(len(t))
    ends = np.append(gap_after, len(t) - 1)
    starts = np.insert(gap_after + 1, 0, 0)
    for s, e in zip(starts, ends):
        seg_end[s:e + 1] = t[e]
    return np.flatnonzero((seg_end - t >= horizon_s) & (sog >= min_sog_kn))


def sample_boats(tracks, n, horizon_s, rng, cue_error_m=100.0, min_sog_kn=2.0, dt_s=10.0):
    """n test boats from real tracks. Returns (TrackBoats, info dict)."""
    usable = {}
    for mmsi, (t, lat, lon, sog) in tracks.items():
        ok = _windows(t, sog, horizon_s, min_sog_kn)
        if len(ok):
            usable[mmsi] = ok
    if not usable:
        raise ValueError("no track long enough; try a shorter horizon or lower min_sog_kn")
    ids = sorted(usable)
    grid = np.arange(0.0, horizon_s + dt_s, dt_s)
    xy = np.empty((n, len(grid), 2))
    mean_kn = np.empty(n)
    for i in range(n):
        mmsi = ids[rng.integers(len(ids))]
        t, lat, lon, sog = tracks[mmsi]
        tt = t[rng.choice(usable[mmsi])] + grid
        la = np.interp(tt, t, lat)
        lo = np.interp(tt, t, lon)
        x = (lo - lo[0]) * M_PER_DEG_LON_EQUATOR * np.cos(np.radians(la[0]))
        y = (la - la[0]) * M_PER_DEG_LAT
        err = rng.normal(0.0, cue_error_m, 2)  # the cue is off by err, so the boat is at -err
        xy[i, :, 0] = x - err[0]
        xy[i, :, 1] = y - err[1]
        mean_kn[i] = np.hypot(x[-1], y[-1]) / horizon_s / KNOT
    info = dict(vessels=len(ids), straight_line_kn=mean_kn)
    return TrackBoats(xy=xy, dt_s=dt_s), info


class TrackPrior:
    """The searcher's guess cloud made of REAL past tracks instead of straight lines.

    Drop-in for BoatPrior: the smart searches only call .sample(n, rng), and the ring
    searches read .speed_kn. Build it from DIFFERENT days than the test boats, or the
    searcher is peeking at the answers.
    """

    def __init__(self, tracks, horizon_s, cue_error_m=100.0, min_sog_kn=2.0):
        self.tracks, self.horizon_s = tracks, horizon_s
        self.cue_error_m, self.min_sog_kn = cue_error_m, min_sog_kn
        _, info = sample_boats(tracks, 500, horizon_s, np.random.default_rng(0), cue_error_m, min_sog_kn)
        # Typical straight-line speed of these tracks, for strategies that fly to a ring.
        self.speed_kn = tuple(np.percentile(info["straight_line_kn"], [10, 90]))

    def sample(self, n, rng):
        return sample_boats(self.tracks, n, self.horizon_s, rng, self.cue_error_m, self.min_sog_kn)[0]
