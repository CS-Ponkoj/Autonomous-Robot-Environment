"""Build the saved building map from the robot's own mapping drives, and report its quality.

    python -m AD.models.mapplan.build --seeds 6000-6039 --workers 4 --out map_v1
    python -m AD.models.mapplan.build --seeds 6000-6039 --snapshot 40 --out map_v1_s40
    python -m AD.models.mapplan.build --seeds 6040-6079 --prior map_v1 --out map_v2

Each mapping drive is a task (start and goal) driven by the built-in driver without cats, with
its own lidar, depth and odometry. Its static map (cells seen occupied over CONFIRM_SPAN, nothing
refused) and its seen-free cells are taken at the end of the drive, or at --snapshot seconds
(odometry drift grows with time, so a shorter drive maps with less of it), placed in the
building frame with the drive's start pose, and added as one vote per drive and cell; the map is
the vote result (SavedMap.finish). Every map cell samples the drive's grid at its own centre, so
each drive gives each cell at most one vote and leaves no holes between cells. Drives that ended
in a collision do not vote.

With --prior <map>, each mapping drive starts from that saved map with drift correction on
(MapDriver, match=True: the robot keeps its pose aligned with the walls it already knows, as a
robot does when it maps a building it has partly mapped), and votes only what it saw itself.

Seeds 6000-6039 are used by nothing else (benchmarks use 1000-1009 and 2000-2019, recordings
10000-10599). The report (runs/<out>/report.json) gives, per drive, the outcome and the
odometry drift when the map was taken (the driver's own pose against the simulator's true pose:
evaluation only, never used to make the map), and a check of the map against the simulator's
floor plan (also evaluation only): how many cells the map calls occupied where the floor plan is
open."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .savedmap import CELL, HALF, N, SavedMap

RUNS = Path(__file__).resolve().parent / "runs"


def _sample_drive_grid(grid: np.ndarray, pose0) -> np.ndarray:
    """A driver-frame grid (SIZE x SIZE) sampled at every map cell centre: (N, N) bool."""
    from robot_env.baseline import SIZE
    x0, y0, yaw = pose0
    c = (np.arange(N) + 0.5) * CELL - HALF
    wx, wy = np.meshgrid(c, c, indexing="ij")
    dx, dy = wx - x0, wy - y0
    cs, sn = math.cos(yaw), math.sin(yaw)
    sx, sy = cs * dx + sn * dy, -sn * dx + cs * dy  # building frame to the drive's start frame
    i = np.floor(sx / CELL).astype(int) + SIZE // 2
    j = np.floor(sy / CELL).astype(int) + SIZE // 2
    ok = (i >= 0) & (i < SIZE) & (j >= 0) & (j < SIZE)
    out = np.zeros((N, N), bool)
    out[ok] = grid[i[ok], j[ok]]
    return out


def _drive(args) -> dict:
    seed, snapshot, prior = args
    from robot_env import config as C
    from robot_env.baseline import CONFIRM_SPAN, BaselineDriver
    from robot_env.layout import RoomMap
    from robot_env.system import RobotSystem

    system = RobotSystem(cats=0, cat_seed=seed)
    task = RoomMap(system.sim.model).sample_task(seed)
    system.reset(*task.start, task.goal)
    if prior:
        from .driver import MapDriver
        driver = MapDriver(prior, match=True)
        driver.reset(initial_pose=task.start)
    else:
        driver = BaselineDriver()
    outcome = "timeout"
    end = min(snapshot, C.EPISODE_TIME_LIMIT) if snapshot else C.EPISODE_TIME_LIMIT
    while system.time < end - 1e-9:
        obs = system.observe()
        if obs.goal_distance <= C.GOAL_RADIUS:
            outcome = "success"
            break
        system.apply(driver.decide(obs))
        system.advance(C.DECISION_PERIOD)
        if system.collisions:
            outcome = "collision"
            break
    if outcome == "timeout" and snapshot and system.time < C.EPISODE_TIME_LIMIT - 1e-9:
        outcome = "snapshot"
    taken = system.time
    if prior:
        static = driver.own_static(taken)
        free = driver.own_free & ~static
    else:
        static = (driver._last - driver._first >= CONFIRM_SPAN) & ~(driver._refused > taken)
        free = driver._seen_free & ~static
    occ_v = _sample_drive_grid(static, task.start)
    free_v = _sample_drive_grid(free, task.start) & ~occ_v
    # drift: the driver's pose placed with the start pose, against the true pose (evaluation only)
    x0, y0, yaw0 = task.start
    px, py, pth = driver.pose
    est = (x0 + math.cos(yaw0) * px - math.sin(yaw0) * py, y0 + math.sin(yaw0) * px + math.cos(yaw0) * py)
    tx, ty, tth = system.ground_truth()["pose"]
    system.close()
    drift_m = math.hypot(est[0] - tx, est[1] - ty)
    drift_deg = abs(math.degrees(math.atan2(math.sin(yaw0 + pth - tth), math.cos(yaw0 + pth - tth))))
    return {"seed": seed, "outcome": outcome, "time_s": round(taken, 1), "drift_m": round(drift_m, 3),
            "drift_deg": round(drift_deg, 2), "occ": occ_v, "free": free_v}


def floor_plan_check(m: SavedMap) -> dict:
    """Map cells called occupied where the simulator's floor plan is open, and the open floor the
    map covers as free (evaluation only)."""
    from robot_env import config as C
    from robot_env.layout import RoomMap, clearance
    from robot_env.system import RobotSystem

    system = RobotSystem(cats=0)
    shapes = RoomMap(system.sim.model).shapes
    system.close()
    c = (np.arange(N) + 0.5) * CELL - HALF
    x, y = np.meshgrid(c, c, indexing="ij")
    inside = (np.abs(x) < C.FLOOR_HALF_SIZE) & (np.abs(y) < C.FLOOR_HALF_SIZE)
    open_ = inside & (clearance(shapes, x, y) > CELL)  # at least one cell from any obstacle
    return {"occupied_cells": int(m.occupied.sum()), "occupied_on_open_floor": int((m.occupied & open_).sum()),
            "open_floor_cells": int(open_.sum()), "open_floor_mapped_free": int((m.free & open_).sum()),
            "open_floor_unseen": int((open_ & ~m.free & ~m.occupied).sum()),
            "note": "open floor: floor-plan clearance over one cell; occupied there means a map error"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="6000-6039", help="seed range a-b, or one seed")
    p.add_argument("--snapshot", type=float, default=None, help="take each drive's map at this many seconds")
    p.add_argument("--prior", default=None, help="a map folder: map with it and drift correction on")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="map_v1")
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--min-share", type=float, default=0.3)
    a = p.parse_args(argv)
    lo, _, hi = a.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    t0 = time.time()
    with ProcessPoolExecutor(max(a.workers, 1), max_tasks_per_child=1) as pool:
        drives = list(pool.map(_drive, [(s, a.snapshot, str(RUNS / a.prior / "map.npz") if a.prior else None) for s in seeds]))
    m = SavedMap.empty()
    for d in drives:
        occ, free = d.pop("occ"), d.pop("free")
        if d["outcome"] != "collision":
            m.occ_votes += occ
            m.free_votes += free
    m.finish(a.min_votes, a.min_share)
    out = RUNS / a.out
    m.save(out / "map.npz")
    drift = np.array([d["drift_m"] for d in drives])
    report = {"seeds": a.seeds, "snapshot_s": a.snapshot, "prior": a.prior, "drives": len(drives), "build_s": round(time.time() - t0, 1),
              "outcomes": {k: sum(d["outcome"] == k for d in drives) for k in ("success", "timeout", "snapshot", "collision")},
              "drift_m": {"median": round(float(np.median(drift)), 3), "max": round(float(drift.max()), 3)},
              "drift_deg": {"median": float(np.median([d["drift_deg"] for d in drives])),
                            "max": max(d["drift_deg"] for d in drives)},
              "min_votes": a.min_votes, "min_share": a.min_share,
              "floor_plan_check": floor_plan_check(m), "per_drive": drives}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_drive"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
