"""Doorway gate: a cat resting in a doorway gives way to the robot (development gate, no rendering).

Every doorway, both approach directions, every speed level, and both resting states (pause, sit):
100 deterministic trials. The robot starts 1.2 m from the doorway, facing it, and is asked to drive
straight through (a fresh forward request every 0.1 s; its safety layer stops it short of the cat).
Each trial records:
  - when the cat starts to give way, measured from when the robot starts waiting (a forward
    request held back by safety), if it ever waits (gate: within 0.5 s);
  - when the robot's straight path is clear of the cat (gate: within 2.0 s of the robot starting
    to wait; a robot that never has to wait found its path clear in time);
  - when the robot is through the doorway (gate: the drive itself at that speed plus 3 s of
    waiting; at 0.20 m/s the 1.6 m drive alone takes 8 s);
  - the smallest gap, at every physics step, from the cat as drawn to the robot's footprint (gate:
    49 mm: the cat keeps 0.30 m by its own motion, but a robot driving up to it may come closer,
    down to its own 50 mm safety buffer), cat contacts and other collisions (gate: none).

    .venv\\Scripts\\python tools\\doorway_check.py [--workers N]
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

YIELD_AFTER_WAIT = 0.5  # s
MIN_FOOTPRINT_GAP = 0.049  # m from a cat (as drawn) to the robot's footprint, at every physics step
CLEAR_AFTER = 2.0  # s
THROUGH_BY = 13.0  # s simulated at most per trial
WAIT_ALLOWANCE = 3.0  # s the robot may spend waiting for the cat, on top of the drive itself
START = 1.2  # m from the doorway centre
STATES = ("pause", "sit")


def trials():
    from robot_env import config as C
    out = []
    for d, (dx, dy) in enumerate(C.DOORS):
        along_x = abs(dy) < 1.0  # corridor doors are in walls along x (passage along y)
        for side in (-1, 1):
            for level in range(len(C.SPEED_LEVELS)):
                for state in STATES:
                    out.append((d, dx, dy, along_x, side, level, state))
    return out


def run_trial(t) -> dict:
    d, dx, dy, along_x, side, level, state = t
    from robot_env import config as C
    from robot_env.cats import CAT_GAP, ROBOT_GAP, WALL_GAP  # noqa: F401
    from robot_env.system import RobotSystem

    # passage direction: +y/-y through a corridor door, +x/-x through the divider door
    ux, uy = (0.0, 1.0) if along_x else (1.0, 0.0)
    ux, uy = ux * side, uy * side
    heading = math.atan2(uy, ux)
    s = RobotSystem(cats=1, cat_seed=2)
    from robot_env.layout import RoomMap
    room = RoomMap(s.sim.model)
    # the robot starts where it fits (the robot's own planning map), about START from the doorway
    start = next(d_ for d_ in (START, 1.1, 1.0, 1.3, 0.9) if room.free[room.cell(dx - d_ * ux, dy - d_ * uy)])
    s.reset(dx - start * ux, dy - start * uy, heading, (dx + 2.5 * ux, dy + 2.5 * uy))
    herd = s.cats
    placed = False
    for off in (0.1, 0.0, 0.2, -0.1):  # the cat in the doorway, facing on through it
        herd.place(0, dx + off * ux, dy + off * uy, heading, state=state)
        c = herd.cats[0]
        g = herd.gaps(c, c.x, c.y, c.yaw)
        if g[0] >= WALL_GAP and g[1] >= ROBOT_GAP:
            placed = True
            break
    if not placed:
        s.close()
        return {"trial": t, "placed": False}
    cat = herd.cats[0]
    v = C.SPEED_LEVELS[level][0]
    wait_at = yield_at = clear_at = through_at = near_at = None
    low = math.inf
    k = 0
    per_tick = round(C.CONTROL_PERIOD / C.PHYSICS_DT)
    while s.time < THROUGH_BY + 1e-9:
        if k % 5 == 0:
            s.drive(v, 0.0)  # a fresh forward request every 0.1 s, from the first step
        for _ in range(per_tick):  # every physics step: the cat as drawn against the robot now
            s.advance(C.PHYSICS_DT)
            low = min(low, herd.current_gaps()[0]["footprint"])
        k += 1
        rx, ry, _ = s.sim.true_pose()
        if wait_at is None and s._waiting_forward():
            wait_at = s.time
        if yield_at is None and cat.state in ("yield", "flee"):
            yield_at = s.time
        _, ahead = herd._in_robot_path(cat)
        if near_at is None and ahead < 1.5:
            near_at = s.time
        in_path, _ = herd._in_robot_path(cat)
        if clear_at is None and yield_at is not None and not in_path:
            clear_at = s.time
        if (rx - dx) * ux + (ry - dy) * uy > 0.4:
            through_at = s.time
            break
    contacts, collisions = s.cat_contacts, s.collisions - s.cat_contacts  # a cat contact is also a collision
    s.close()
    return {"trial": t, "placed": True, "wait": wait_at, "yield": yield_at, "clear": clear_at, "near": near_at,
            "through": through_at, "low": low, "contacts": contacts, "collisions": collisions, "start": start}


def main() -> int:
    from robot_env import config as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(run_trial, trials()))
    fails = []
    lows = []
    for r in results:
        d, dx, dy, along_x, side, level, state = r["trial"]
        name = f"door {d} ({dx:+.1f},{dy:+.2f}) side {side:+d} level {level + 1} {state}"
        if not r["placed"]:
            fails.append(f"{name}: no legal resting pose in the doorway")
            continue
        lows.append(r["low"])
        why = []
        if r["yield"] is None:
            why.append("never gave way")
        elif r["wait"] is not None and r["yield"] > r["wait"] + YIELD_AFTER_WAIT + 0.02:
            why.append(f"gave way {r['yield'] - r['wait']:.2f} s after the robot started waiting")
        # the path must clear within 2 s of the robot being held up (if it never was, it cleared in time)
        if r["wait"] is not None and (r["clear"] is None or r["clear"] > r["wait"] + CLEAR_AFTER + 0.02):
            why.append("path not clear in time" if r["clear"] is None else
                       f"path clear {r['clear'] - r['wait']:.2f} s after the robot started waiting")
        v = C.SPEED_LEVELS[level][0]
        allowed = (r["start"] + 0.4) / v + WAIT_ALLOWANCE  # the drive itself at this speed, plus waiting
        if r["through"] is None or r["through"] > allowed:
            why.append(f"robot not through in {allowed:.1f} s")
        if r["contacts"]:
            why.append(f"{r['contacts']} cat contacts")
        if r["collisions"]:
            why.append(f"{r['collisions']} other collisions")
        if r["low"] < MIN_FOOTPRINT_GAP:
            why.append(f"cat {1000 * r['low']:.0f} mm from the robot's footprint")
        if why:
            fails.append(f"{name}: " + "; ".join(why))
    for f in fails:
        print("FAIL", f)
    n = len(results)
    print(f"{n - len(fails)}/{n} trials pass; smallest gap from a cat (as drawn) to the robot's footprint "
          f"{1000 * min(lows):.1f} mm (gate {1000 * MIN_FOOTPRINT_GAP:.0f} mm: the robot's safety buffer is 50 mm)")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
