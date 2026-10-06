"""The forward depth sensor's geometry and its obstacle points, from the observation alone (no
simulator truth). A frame is DEPTH_ROWS x DEPTH_COLS ranges along fixed rays from the camera's
mount; returns between the floor (DEPTH_FLOOR) and the robot's own top plus 1 cm (anything the
robot would sweep into) become obstacle points in the robot frame, one per DEPTH_VOXEL cell, moved
by the robot's own motion since the frame was taken (constant encoder velocity)."""

from __future__ import annotations

import math

import numpy as np

from . import config as C
from .types import Observation


def depth_directions() -> np.ndarray:
    """(DEPTH_ROWS, DEPTH_COLS, 3) unit ray directions in the robot body frame (x forward, y
    left, z up): rows top to bottom, columns left to right, the whole frustum pitched down."""
    el = C.DEPTH_VFOV / 2 - (np.arange(C.DEPTH_ROWS) + 0.5) * C.DEPTH_VFOV / C.DEPTH_ROWS - C.DEPTH_PITCH
    az = C.DEPTH_HFOV / 2 - (np.arange(C.DEPTH_COLS) + 0.5) * C.DEPTH_HFOV / C.DEPTH_COLS
    e, a = np.meshgrid(el, az, indexing="ij")
    return np.stack((np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)), axis=-1)


_DIRS = depth_directions()


def depth_points(obs: Observation) -> np.ndarray:
    """(n, 2) obstacle points in the robot frame at obs.time, from the observation's depth frame
    (none without one)."""
    if obs.depth is None:
        return np.empty((0, 2))
    d = np.asarray(obs.depth, float)
    ok = np.asarray(obs.depth_valid, bool) & (d < C.DEPTH_MAX)
    p = _DIRS[ok] * d[ok][:, None] + np.array(C.DEPTH_ORIGIN)  # body frame
    # height above the floor, levelled by the IMU's gravity reading (a pitched body would
    # otherwise lift far floor returns into obstacles)
    up = np.array((0.0, 0.0, 1.0)) if obs.depth_up is None else np.asarray(obs.depth_up, float)
    z = C.DEPTH_BODY_Z + p @ up
    keep = (z > C.DEPTH_FLOOR) & (z <= C.ROBOT_TOP + 0.01)
    x, y = p[keep, 0], p[keep, 1]
    if not len(x):
        return np.empty((0, 2))
    cells = np.unique(np.round(np.column_stack((x, y)) / C.DEPTH_VOXEL).astype(np.int64), axis=0)
    pts = cells * C.DEPTH_VOXEL
    dt = obs.time - obs.depth_time if obs.depth_time is not None else 0.0
    if dt > 0.0:  # the robot moved since the frame: its points, back in the present robot frame
        v, w = obs.velocity_estimate
        th = w * dt
        mid = 0.5 * th
        tx, ty = v * dt * math.cos(mid), v * dt * math.sin(mid)
        c, s = math.cos(th), math.sin(th)
        dx, dy = pts[:, 0] - tx, pts[:, 1] - ty
        pts = np.column_stack((c * dx + s * dy, -s * dx + c * dy))
    return pts


# A depth point is remembered while the sensor cannot see where it is (nothing there could show
# it gone), until it is back in view (where this frame decides), out of the safety radius, or
# DEPTH_MEMORY old (a bound on the memory, not a guess that it has gone)
DEPTH_MEMORY = 30.0  # s
NEAR_EDGE = C.DEPTH_ORIGIN[0] + (C.DEPTH_BODY_Z + C.DEPTH_ORIGIN[2]) / math.tan(C.DEPTH_VFOV / 2 + C.DEPTH_PITCH) + 0.03


def _visible(pts: np.ndarray) -> np.ndarray:
    """Points (robot frame) inside what the sensor sees now: ahead of its near edge (where its
    lowest rays meet the floor) and within its side angles (less one column)."""
    bearing = np.arctan2(pts[:, 1], pts[:, 0] - C.DEPTH_ORIGIN[0])
    return (pts[:, 0] >= NEAR_EDGE) & (np.abs(bearing) <= C.DEPTH_HFOV / 2 - C.DEPTH_HFOV / C.DEPTH_COLS)


def remember(obs: Observation, state: tuple | None) -> tuple[np.ndarray, tuple]:
    """(the depth points to use now: this frame's, plus those remembered where the sensor cannot
    see them now, such as a low box the robot has come too close to see; the
    new state, a plain tuple). Odometry from the encoder velocity carries remembered points; a
    time going backwards (a reset) clears the memory."""
    now = obs.time
    if state is None or now < state[0]:
        pose, pts, born = np.zeros(3), np.empty((0, 2)), np.empty(0)
    else:
        last, pose, pts, born = state
        pose = pose.copy()
        dt = now - last
        if dt > 0.0:
            v, w = obs.velocity_estimate
            mid = pose[2] + 0.5 * w * dt
            pose += (v * dt * math.cos(mid), v * dt * math.sin(mid), w * dt)
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    if len(pts):  # the remembered points, in the present robot frame
        dx, dy = pts[:, 0] - x, pts[:, 1] - y
        local = np.column_stack((c * dx + s * dy, -s * dx + c * dy))
        keep = ((now - born <= DEPTH_MEMORY) & ~_visible(local)
                & (np.hypot(local[:, 0], local[:, 1]) <= C.SAFETY_CONSIDER_RADIUS + 0.5))
        local, pts, born = local[keep], pts[keep], born[keep]
    else:
        local = np.empty((0, 2))
    current = depth_points(obs)
    if len(current):  # this frame's points join the memory (odometry frame)
        wx, wy = x + c * current[:, 0] - s * current[:, 1], y + s * current[:, 0] + c * current[:, 1]
        pts = np.vstack((pts, np.column_stack((wx, wy))))
        born = np.concatenate((born, np.full(len(current), now)))
    return np.vstack((current, local)), (now, pose, pts, born)
