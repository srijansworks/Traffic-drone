#!/usr/bin/env python3
"""Fly the stale-tip search on a real Tello, indoors.

The story at room scale: the takeoff spot is where the "satellite" last saw the boat.
While nobody is looking, a helper puts the "boat" (a box with the same marker on all
four sides) somewhere 0.8-2.0 m away. The Tello then flies a search plan: it moves to
each look point and spins 360 degrees. It counts a find only if the marker is within
0.75 m (judged from the marker's size in the picture), then hovers and lands.

ALWAYS go in this order, and don't skip steps on a new day or in a new room:

    python tello_search.py --dry-run                  # no drone: draws both plans
    python tello_search.py --ground                   # drone on the floor, motors off:
                                                      #   hold the box at 0.75 m and check
                                                      #   the distance readout with a tape
    python tello_search.py --axis-test                # tiny flight: forward, back, LEFT,
                                                      #   back. Check it really goes left!
    python tello_search.py --fly --strategy spiral    # the real thing
    python tello_search.py --fly --strategy smart

Keys in the video window:  L = land   DELETE/BACKSPACE = cut motors (emergency)
Landing with L waits for the current move to finish (a few seconds). If the drone is
about to hit something, use DELETE.

Before flying: prop guards on, battery 40%+, a clear room at least 5 m x 5 m
(--room sets the flyable box; it keeps 0.5 m from the walls you give it), nobody
inside the flyable box, textured floor (Tello's position sensor needs texture).

Every flight is logged to flights/<time>_<strategy>/ (plan, events, video).
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

from cuesearch.lookpoints import Room, TelloSpec, plan_time, smart_plan, spiral_plan

DRONEFOLLOW = os.environ.get("DRONEFOLLOW", os.path.expanduser("~/Documents/dronefollow"))
MIN_BATTERY = 40
MAX_FLIGHT_S = 150


def make_plan(args):
    half_x, half_y = args.room[0] / 2 - 0.5, args.room[1] / 2 - 0.5
    room = Room(-half_x, half_x, -half_y, half_y)
    spec = TelloSpec(speed_ms=args.speed_cms / 100, sense_m=args.sense_m)
    if args.strategy == "smart":
        pts = smart_plan(room, spec, args.budget_s, args.r_min, args.r_max)
    else:
        pts = spiral_plan(room, spec, args.budget_s)
    return room, spec, pts


def draw_plan(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, strat in zip(axes, ("spiral", "smart")):
        args.strategy = strat
        room, spec, pts = make_plan(args)
        p = np.vstack([[0, 0], np.array(pts)])
        # drawn as seen from above, standing behind the drone: forward = up the page,
        # right = right (so the plot's x is -y, since y means LEFT)
        ax.add_patch(plt.Rectangle((-room.y_max, room.x_min), room.y_max - room.y_min,
                                   room.x_max - room.x_min, fill=False, ls="--", color="grey"))
        for r in (args.r_min, args.r_max):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#e08a1e", ls=":"))
        for q in pts:
            ax.add_patch(plt.Circle((-q[1], q[0]), spec.sense_m, color="#1f6fd1", alpha=0.12))
        ax.plot(-p[:, 1], p[:, 0], "o-", color="#1f6fd1")
        ax.plot(0, 0, "k+", ms=14)
        for i, q in enumerate(pts, 1):
            ax.annotate(str(i), (-q[1], q[0]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(f"{strat}: {len(pts)} look points, {plan_time(pts, spec):.0f} s")
        ax.set_aspect("equal")
        ax.set_xlabel("metres to the right of the takeoff spot")
        ax.set_ylabel("metres forward")
    fig.suptitle("Tello search plans. + = takeoff / tip spot, dotted = where the box may be, "
                 "blue = what each 360 scan checks", fontsize=9)
    fig.tight_layout()
    out = "tello_plans.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")
    for strat in ("spiral", "smart"):
        args.strategy = strat
        room, spec, pts = make_plan(args)
        print(f"\n{strat} (forward m, left m): " + "  ".join(f"({x:+.2f},{y:+.2f})" for x, y in pts))


class Spotter:
    """Watches the video for the box's marker and says when it's close enough."""

    def __init__(self, marker_id, marker_m, sense_m):
        sys.path.insert(0, DRONEFOLLOW)
        from dronefollow.detection import distance_for_size_ratio
        from dronefollow.markers import get_dictionary, make_detector
        self._detect = make_detector(get_dictionary("DICT_4X4_50"))
        self._dist = distance_for_size_ratio
        self.marker_id, self.marker_m, self.sense_m = marker_id, marker_m, sense_m
        self._hits = []

    def look(self, frame):
        """Returns (distance_m or None, corners or None, found_now)."""
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect(gray)
        if ids is None:
            return None, None, False
        for c, i in zip(corners, ids.flatten()):
            if int(i) != self.marker_id:
                continue
            pts = c.reshape(4, 2)
            size_px = (np.linalg.norm(pts[0] - pts[2]) + np.linalg.norm(pts[1] - pts[3])) / 2 / np.sqrt(2)
            d = self._dist(size_px / frame.shape[1], self.marker_m)
            now = time.monotonic()
            if d <= self.sense_m:
                self._hits = [h for h in self._hits if now - h < 1.0] + [now]
            # two close sightings within a second = a find (one frame can be a fluke)
            return d, pts, len(self._hits) >= 2
        return None, None, False


def connect(args):
    sys.path.insert(0, DRONEFOLLOW)
    from djitellopy import Tello
    tello = Tello()
    tello.connect()
    bat = tello.get_battery()
    print(f"Connected. Battery {bat}%")
    tello.streamon()
    return tello, bat


def video_loop(tello, args, spotter, flight, log_dir):
    """Main thread: show video, spot the marker, handle keys. Runs until flight.done."""
    import cv2
    sys.path.insert(0, DRONEFOLLOW)
    from dronefollow.safety import is_emergency_key
    reader = tello.get_frame_read()
    writer = None
    while not flight["done"]:
        frame = reader.frame
        if frame is None:
            time.sleep(0.01)
            continue
        frame = frame.copy()
        if args.rgb_swap:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        d, pts, found = spotter.look(frame)
        if found and not flight["found_at"] and flight["flying"]:
            flight["found_at"] = time.monotonic() - flight["t0"]
            flight["events"].append(("found", round(flight["found_at"], 1), round(d, 2)))
        if pts is not None:
            cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0) if d <= args.sense_m else (0, 200, 255), 3)
        status = flight["status"]
        if d is not None:
            status += f"   marker {d:.2f} m" + ("  IN RANGE" if d <= args.sense_m else "")
        if flight["found_at"]:
            status = f"FOUND at {flight['found_at']:.1f} s   " + status
        cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "L land   DELETE cut motors", (12, frame.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if log_dir:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(os.path.join(log_dir, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            writer.write(frame)
        cv2.imshow("tello search", frame)
        key = cv2.waitKey(1) & 0xFF
        if is_emergency_key(key):
            tello.emergency()
            flight["events"].append(("EMERGENCY motor cut", _elapsed(flight)))
            flight["abort"] = flight["done"] = True
        elif key in (ord("l"), ord("L"), ord("q"), ord("Q")):
            flight["abort"] = True
            flight["status"] = "landing (after this move)..."
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def fly_worker(tello, moves, args, flight):
    """Worker thread: takeoff, run the moves, land. Checks abort/found between moves."""
    try:
        flight["status"] = "taking off"
        tello.takeoff()
        flight["t0"] = time.monotonic()
        flight["flying"] = True
        for label, fn in moves:
            if flight["abort"] or flight["found_at"]:
                break
            if time.monotonic() - flight["t0"] > MAX_FLIGHT_S:
                flight["events"].append(("time limit", MAX_FLIGHT_S))
                break
            flight["status"] = label
            fn()
            flight["events"].append((label, round(time.monotonic() - flight["t0"], 1)))
        if flight["found_at"]:
            flight["status"] = "FOUND - hovering"
            time.sleep(3)
        flight["status"] = "landing"
        tello.land()
    except Exception as e:  # any error: try to land, report it
        flight["events"].append(("error", repr(e)))
        try:
            tello.land()
        except Exception:
            tello.emergency()
    finally:
        flight["flying"] = False
        flight["end"] = round(time.monotonic() - flight["t0"], 1) if flight["t0"] else None
        flight["done"] = True


def search_moves(tello, pts, args):
    """Relative 'go' moves between look points, plus a 360 scan (4 x 90 degrees) at each.
    The drone never turns except during scans, which end facing the way they started,
    so 'forward/left' stay the same as at takeoff."""
    moves, here = [], np.zeros(2)
    for i, p in enumerate(pts, 1):
        d = (np.asarray(p) - here) * 100
        parts = int(np.ceil(np.linalg.norm(d) / 450))   # the Tello's 'go' takes at most 500 cm
        for k in range(parts):
            dx, dy = d / parts
            moves.append((f"to look point {i}/{len(pts)}",
                          lambda dx=int(round(dx)), dy=int(round(dy)): tello.go_xyz_speed(dx, dy, 0, args.speed_cms)))
        for q in range(4):
            moves.append((f"scanning at point {i} ({(q + 1) * 90} deg)",
                          lambda: (tello.rotate_clockwise(90), time.sleep(0.6))))
        here = np.asarray(p)
    return moves


def axis_moves(tello):
    go = lambda x, y: (lambda: tello.go_xyz_speed(x, y, 0, 30))
    return [("FORWARD 50 cm", go(50, 0)), ("back", go(-50, 0)),
            ("LEFT 50 cm (check it goes LEFT!)", go(0, 50)), ("back", go(0, -50))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="no drone: draw both plans")
    mode.add_argument("--ground", action="store_true", help="connect and show the marker distance; motors stay off")
    mode.add_argument("--axis-test", action="store_true", help="tiny flight to check forward/left directions")
    mode.add_argument("--fly", action="store_true", help="fly the search")
    ap.add_argument("--strategy", choices=("smart", "spiral"), default="smart")
    ap.add_argument("--room", type=float, nargs=2, default=[5.0, 5.0], metavar=("FORWARD_M", "SIDE_M"),
                    help="clear floor around the takeoff spot (takeoff in the middle); 0.5 m margin kept")
    ap.add_argument("--r-min", type=float, default=0.8, help="box at least this far from the tip, m")
    ap.add_argument("--r-max", type=float, default=2.0, help="box at most this far from the tip, m")
    ap.add_argument("--sense-m", type=float, default=0.75, help="a find counts within this distance")
    ap.add_argument("--marker-m", type=float, default=0.15, help="printed marker width on the box, m")
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--budget-s", type=float, default=70.0, help="search time limit for the plan")
    ap.add_argument("--speed-cms", type=int, default=50, help="move speed, cm/s (10-100)")
    ap.add_argument("--rgb-swap", action="store_true", help="fix blue-looking video (see dronefollow README)")
    args = ap.parse_args()

    if args.dry_run:
        draw_plan(args)
        return

    tello, bat = connect(args)
    spotter = Spotter(args.marker_id, args.marker_m, args.sense_m)
    flight = {"done": False, "abort": False, "found_at": None, "flying": False, "t0": None,
              "status": "on the ground (motors off)", "events": []}

    if args.ground:
        print("Ground test: motors stay off. Close the window or press L to stop.")
        t = threading.Thread(target=lambda: (_wait_abort(flight), flight.update(done=True)), daemon=True)
        t.start()
        video_loop(tello, args, spotter, flight, None)
        tello.streamoff()
        return

    if bat < MIN_BATTERY:
        sys.exit(f"Battery {bat}% is below {MIN_BATTERY}%. Charge it first.")
    if args.axis_test:
        moves, label, pts = axis_moves(tello), "axis-test", []
    else:
        room, spec, pts = make_plan(args)
        moves, label = search_moves(tello, pts, args), args.strategy
        print(f"{label}: {len(pts)} look points, planned {plan_time(pts, spec):.0f} s")
    log_dir = os.path.join("flights", time.strftime("%Y%m%d_%H%M%S") + f"_{label}")
    os.makedirs(log_dir, exist_ok=True)
    input(f"Prop guards on, room clear, box placed? Press Enter to take off ({label})... ")
    worker = threading.Thread(target=fly_worker, args=(tello, moves, args, flight), daemon=True)
    worker.start()
    video_loop(tello, args, spotter, flight, log_dir)
    worker.join(timeout=15)
    with open(os.path.join(log_dir, "flight.json"), "w") as f:
        json.dump({"strategy": label, "look_points_m": pts, "battery_start": bat,
                   "found_at_s": flight["found_at"], "flight_s": flight.get("end"),
                   "sense_m": args.sense_m, "marker_m": args.marker_m, "events": flight["events"],
                   "box_position_m": None, "notes": ""}, f, indent=1)
    print(f"Logged to {log_dir}/flight.json  (fill in box_position_m with a tape measure)")
    tello.streamoff()
    tello.end()


def _elapsed(flight):
    return round(time.monotonic() - flight["t0"], 1) if flight["t0"] else 0.0


def _wait_abort(flight):
    while not flight["abort"]:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
