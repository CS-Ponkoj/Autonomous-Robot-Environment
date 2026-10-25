"""A coarse local occupancy map built only from what the robot itself measures: the binned lidar
and the wheel-encoder speed and turn rate of AD.common.features.vector(obs), integrated over the
observation times. No simulation state is read, so the same code rebuilds the maps of the
recordings (replaying their feature rows) and runs live in the driver.

The memory is a world-frame grid anchored where the episode started (dead reckoning; slow drift
does not matter for a few metres around the robot):
- occupied: set where a lidar sector returns a hit, fading slowly, and cleared quickly when a
  later ray passes through the cell (a cat that moved on leaves no ghost);
- free: set along each sector's ray up to its hit, fading a little faster.
The model sees an egocentric crop, CELLS x CELLS at RES metres, the robot in the middle facing
up (row 0 is ahead, column 0 is left), with a third channel marking the goal (on the border when
it is farther than the crop)."""

from __future__ import annotations

import math

import numpy as np

from robot_env import config as C

from ...common.features import LIDAR_BINS

CELLS = 32
RES = 0.25  # m per cell: the crop is 8 m across, a little more than the lidar's 5 m range
CHANNELS = 3  # occupied, free, goal
WORLD = 400  # world grid cells (100 m): wide enough that an episode never leaves it
OCC_DECAY = 0.995  # per 0.1 s step: half-life about 14 s
FREE_DECAY = 0.99  # about 7 s
CLEAR = 0.3  # an occupied cell a ray passes through keeps this share
GOAL_SCALE = 10.0  # features.vector stores min(goal distance, 10 m) / 10
_BIN_ANGLES = -math.pi + (np.arange(LIDAR_BINS) * (C.LIDAR_RAYS // LIDAR_BINS) + (C.LIDAR_RAYS // LIDAR_BINS - 1) / 2) \
    * 2 * math.pi / C.LIDAR_RAYS  # each 5 degree sector's centre (ray 0 at -180 degrees)
_STEPS = (np.arange(int(C.LIDAR_RANGE / (RES / 2))) + 0.5) * (RES / 2)  # free-space samples along a ray
_half = CELLS / 2
_EGO_X = ((_half - np.arange(CELLS) - 0.5) * RES)[:, None].repeat(CELLS, 1)  # ahead, per row
_EGO_Y = ((_half - np.arange(CELLS) - 0.5) * RES)[None, :].repeat(CELLS, 0)  # left, per column


class LocalMap:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.occ = np.zeros((WORLD, WORLD), np.float32)
        self.free = np.zeros((WORLD, WORLD), np.float32)
        self.pose = np.zeros(3)  # x, y, heading in the start frame
        self.last = None  # (time, v, omega) of the previous step

    def _cells(self, x: np.ndarray, y: np.ndarray):
        i = np.floor(x / RES).astype(np.int64) + WORLD // 2
        j = np.floor(y / RES).astype(np.int64) + WORLD // 2
        ok = (i >= 0) & (i < WORLD) & (j >= 0) & (j < WORLD)
        return i[ok], j[ok]

    def step(self, f: np.ndarray, t: float) -> np.ndarray:
        """Add one observation (a features.vector row and its time); return the egocentric crop,
        (CHANNELS, CELLS, CELLS) uint8 (0-255)."""
        from ...common.features import ODOMETRY
        v, w = float(f[ODOMETRY][0]) * C.MAX_LINEAR_SPEED, float(f[ODOMETRY][1]) * C.MAX_ANGULAR_SPEED
        if self.last is not None:  # dead reckoning with the previous speed and turn rate (midpoint rule)
            t0, v0, w0 = self.last
            dt = min(max(t - t0, 0.0), 1.0)
            mid = self.pose[2] + w0 * dt / 2
            self.pose += (v0 * dt * math.cos(mid), v0 * dt * math.sin(mid), w0 * dt)
            steps = max(dt / C.DECISION_PERIOD, 0.0)
            self.occ *= OCC_DECAY ** steps
            self.free *= FREE_DECAY ** steps
        self.last = (t, v, w)
        px, py, th = self.pose
        r = f[:LIDAR_BINS] * C.LIDAR_RANGE
        ang = th + _BIN_ANGLES
        ca, sa = np.cos(ang), np.sin(ang)
        along = _STEPS[None, :] < (r[:, None] - RES / 2)  # free up to half a cell short of the hit
        d = np.broadcast_to(_STEPS[None, :], along.shape)[along]
        rows = np.nonzero(along)[0]
        fi, fj = self._cells(px + d * ca[rows], py + d * sa[rows])
        self.free[fi, fj] = 1.0
        self.occ[fi, fj] *= CLEAR
        hit = r < C.LIDAR_RANGE - 1e-3
        hi, hj = self._cells(px + r[hit] * ca[hit], py + r[hit] * sa[hit])
        self.occ[hi, hj] = 1.0
        # the crop: each egocentric cell centre in the start frame, nearest world cell
        c, s = math.cos(th), math.sin(th)
        wx, wy = px + c * _EGO_X - s * _EGO_Y, py + s * _EGO_X + c * _EGO_Y
        i = np.clip(np.floor(wx / RES).astype(np.int64) + WORLD // 2, 0, WORLD - 1)
        j = np.clip(np.floor(wy / RES).astype(np.int64) + WORLD // 2, 0, WORLD - 1)
        out = np.zeros((CHANNELS, CELLS, CELLS), np.float32)
        out[0], out[1] = self.occ[i, j], self.free[i, j]
        gd = float(f[LIDAR_BINS]) * GOAL_SCALE
        gb = math.atan2(float(f[LIDAR_BINS + 1]), float(f[LIDAR_BINS + 2]))
        lim = (_half - 0.5) * RES
        gx, gy = gd * math.cos(gb), gd * math.sin(gb)
        k = max(abs(gx), abs(gy)) / lim
        if k > 1:  # farther than the crop: on its border, in the goal's direction
            gx, gy = gx / k, gy / k
        gr = min(int(_half - gx / RES), CELLS - 1)
        gc = min(int(_half - gy / RES), CELLS - 1)
        out[2, gr, gc] = 1.0
        return np.round(out * 255).astype(np.uint8)


def replay(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The maps of one recorded episode: its feature rows in time order and their times."""
    m = LocalMap()
    return np.stack([m.step(x[k], float(t[k])) for k in range(len(x))]) if len(x) else \
        np.zeros((0, CHANNELS, CELLS, CELLS), np.uint8)
