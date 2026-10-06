"""Bridging dropped lidar readings, from the observation alone (no simulator truth): each ray's
last valid reading is kept in an odometry frame (the encoder velocities integrated from scan to
scan) for MEMORY_AGE. A ray that drops out takes the nearest remembered return that now falls in
its sector (moved by the robot's own motion since), or, if its own last reading was no return,
no return; with neither it stays unknown. A scan time going backwards (a reset) clears the
memory. The state is a plain tuple, so a driver can hold it."""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import config as C
from .types import Observation

MEMORY_AGE = 0.10  # s a ray's last valid reading may stand in for one that dropped out


def bridge(obs: Observation, state: tuple | None) -> tuple[Observation, tuple]:
    """(the observation with dropped readings bridged where the memory allows, the new state)."""
    n, t = len(obs.lidar), obs.scan_time
    if state is None or t < state[0] or len(state[4]) != n:
        pose = np.zeros(3)
        pts = np.full((n, 2), np.nan)  # each ray's last return, odometry frame
        free = np.zeros(n, bool)  # its last valid reading was no return
        seen = np.full(n, -np.inf)  # when that reading was taken
    else:
        last, pose, pts, free, seen = state
        pose = pose.copy()
        if t > last:
            dt = t - last
            v, w = obs.velocity_estimate
            mid = pose[2] + 0.5 * w * dt
            pose += (v * dt * math.cos(mid), v * dt * math.sin(mid), w * dt)
        pts, free, seen = pts.copy(), free.copy(), seen.copy()
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    a = np.asarray(obs.lidar_angles)
    valid = np.asarray(obs.lidar_valid, bool)
    hit = valid & (obs.lidar < C.LIDAR_RANGE)
    lx, ly = obs.lidar[hit] * np.cos(a[hit]), obs.lidar[hit] * np.sin(a[hit])
    pts[hit] = np.column_stack((x + c * lx - s * ly, y + s * lx + c * ly))
    free[valid] = ~hit[valid]
    seen[valid] = t
    state = (t, pose, pts, free, seen)
    missing = ~valid
    if not missing.any():
        return obs, state
    recent = seen >= t - MEMORY_AGE - 1e-9
    lidar, ok = np.array(obs.lidar, float), valid.copy()
    use = recent & ~free
    nearest = np.full(n, np.inf)
    if use.any():  # remembered returns, back in the robot's present frame, by the ray they now fall on
        dx, dy = pts[use, 0] - x, pts[use, 1] - y
        px, py = c * dx + s * dy, -s * dx + c * dy
        step = 2 * math.pi / n
        ray = np.round((np.arctan2(py, px) - a[0]) / step).astype(int) % n
        np.minimum.at(nearest, ray, np.hypot(px, py))
    filled = missing & np.isfinite(nearest)
    lidar[filled], ok[filled] = nearest[filled], True
    clear = missing & ~filled & recent & free
    lidar[clear], ok[clear] = C.LIDAR_RANGE, True
    return dataclasses.replace(obs, lidar=lidar, lidar_valid=ok), state
