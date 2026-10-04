"""Rule-based baseline driver: the reference that later drivers and realism changes
are compared against.

It sees only the Observation (lidar, encoder velocity, goal distance and bearing), through the
same decide(observation) plug as every driver. It never touches the simulator, the world map,
the task generator, or ground truth. Like a real robot, it builds its own memory:

1. Odometry: integrates the encoder velocity estimate into a pose in its own start frame
   (trapezoidal: the average of the previous and current estimates over each interval).
2. Mapping: a 0.1 m occupancy grid from its lidar hits (a hit marks a cell occupied for the
   rest of the episode; the world is static). Unknown cells are treated as free (optimistic
   exploration). Clearing cells along rays was tried and rejected: grazing rays erased real
   walls (6/20 development successes instead of 16/20).
3. Planning: a wavefront (breadth-first distance field) from the goal over the grid inflated
   by the robot radius, recomputed every 0.5 s or when new obstacles appear on the path.
4. Following: walk down the distance field for about 0.5 m from the robot's cell and head for
   that point, turn in place when the heading error is large, and slow down when the corridor
   ahead is short.
5. Recovery: when it commands motion but the encoders show none for 1 s (for example the
   safety layer refuses every move in a tight spot), it backs up briefly and replans.
With ideal sensors this works well; with realistic odometry drift and lidar noise (later
realism rounds) it degrades, which is what a baseline should reveal.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C
from .types import Command, Decision, Observation

CELL = 0.1  # m
SIZE = 220  # cells per side (22 m): the start is at the center, so any goal on a 10 x 10 m floor fits
INFLATE = int(math.ceil((C.CIRCUMSCRIBED_RADIUS + 0.06) / CELL))  # cells of clearance around obstacles
REPLAN_PERIOD = 0.5  # s
STUCK_AFTER = 1.0  # s of commanded but absent motion
BACKUP_TIME = 0.8  # s of reversing to get out
LOOKAHEAD = 5  # cells (0.5 m)
HALF_WIDTH = C.FOOTPRINT_HALF_WIDTH + 0.04
_DISK = [(dx, dy) for dx in range(-INFLATE, INFLATE + 1) for dy in range(-INFLATE, INFLATE + 1)
         if dx * dx + dy * dy <= INFLATE * INFLATE]
_STEPS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]


def _shift(a: np.ndarray, dx: int, dy: int, fill) -> np.ndarray:
    """a shifted by (dx, dy) cells, padding with `fill` (no wrap-around)."""
    out = np.full_like(a, fill)
    xs = slice(max(dx, 0), a.shape[0] + min(dx, 0))
    xd = slice(max(-dx, 0), a.shape[0] + min(-dx, 0))
    ys = slice(max(dy, 0), a.shape[1] + min(dy, 0))
    yd = slice(max(-dy, 0), a.shape[1] + min(-dy, 0))
    out[xs, ys] = a[xd, yd]
    return out


class BaselineDriver:
    name = "rule_baseline"

    def __init__(self, speed: tuple[float, float] = C.SPEED_LEVELS[C.DEFAULT_SPEED_LEVEL]):
        self.v_max, self.w_max = speed
        self.reset()

    def reset(self) -> None:
        self.pose = np.zeros(3)  # x, y, yaw in the driver's own start frame
        self._last_time: float | None = None
        self._last_vel = (0.0, 0.0)
        self.occupied = np.zeros((SIZE, SIZE), bool)
        self.dist = None
        self._next_plan = 0.0
        self._goal = None
        self._still_since: float | None = None
        self._backup_until = -1.0

    # ----- memory -----
    def _cell(self, x, y):
        return (np.floor(np.asarray(x) / CELL).astype(int) + SIZE // 2,
                np.floor(np.asarray(y) / CELL).astype(int) + SIZE // 2)

    def _integrate(self, obs: Observation) -> None:
        if self._last_time is not None:
            dt = obs.time - self._last_time
            v = 0.5 * (self._last_vel[0] + obs.velocity_estimate[0])
            w = 0.5 * (self._last_vel[1] + obs.velocity_estimate[1])
            th = self.pose[2] + 0.5 * w * dt  # midpoint heading
            self.pose += (v * math.cos(th) * dt, v * math.sin(th) * dt, w * dt)
        self._last_time = obs.time
        self._last_vel = tuple(obs.velocity_estimate)

    def _map(self, obs: Observation) -> bool:
        """Add the scan's hits. Returns True when a new occupied cell appeared."""
        x, y, th = self.pose
        hit = obs.lidar_valid & (obs.lidar < C.LIDAR_RANGE - 1e-6)
        a = th + obs.lidar_angles[hit]
        cx, cy = self._cell(x + obs.lidar[hit] * np.cos(a), y + obs.lidar[hit] * np.sin(a))
        inside = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
        cx, cy = cx[inside], cy[inside]
        new = bool((~self.occupied[cx, cy]).any())
        self.occupied[cx, cy] = True
        return new

    # ----- planning -----
    def _plan(self) -> None:
        blocked = self.occupied.copy()
        for dx, dy in _DISK:
            blocked |= _shift(self.occupied, dx, dy, False)
        rx, ry = self._cell(self.pose[0], self.pose[1])
        gx, gy = (int(np.clip(c, 0, SIZE - 1)) for c in self._cell(*self._goal))  # far goals: plan to the edge
        ii, jj = np.ogrid[:SIZE, :SIZE]
        for cx, cy in ((rx, ry), (gx, gy)):
            # Near the robot and the goal only the real obstacle cells block (not the safety
            # margin), so a robot standing inside the margin can still plan its way out.
            near = (ii - cx) ** 2 + (jj - cy) ** 2 <= (INFLATE + 1) ** 2
            blocked[near & ~self.occupied] = False
        dist = np.full((SIZE, SIZE), np.inf)
        dist[gx, gy] = 0.0
        reached = None
        for k in range(4 * SIZE):
            best = dist
            for dx, dy, cost in _STEPS:
                best = np.minimum(best, _shift(dist, dx, dy, np.inf) + cost)
            best[blocked] = np.inf
            best[gx, gy] = 0.0
            if np.array_equal(best, dist):
                break
            dist = best
            if 0 <= rx < SIZE and 0 <= ry < SIZE and np.isfinite(dist[rx, ry]):
                reached = k if reached is None else reached
                if k - reached > LOOKAHEAD + 2:  # the robot's neighbourhood is settled: stop early
                    break
        self.dist = dist

    def _target_heading(self) -> float | None:
        if self.dist is None:
            return None
        cx, cy = self._cell(self.pose[0], self.pose[1])
        if not (0 <= cx < SIZE and 0 <= cy < SIZE) or not np.isfinite(self.dist[cx, cy]):
            return None
        for _ in range(LOOKAHEAD):  # walk down the distance field: stays in planned free space
            best = (self.dist[cx, cy], cx, cy)
            for dx, dy, _cost in _STEPS:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < SIZE and 0 <= ny < SIZE and self.dist[nx, ny] < best[0]:
                    best = (self.dist[nx, ny], nx, ny)
            if best[1:] == (cx, cy):
                break
            cx, cy = best[1], best[2]
        tx, ty = (cx - SIZE // 2 + 0.5) * CELL, (cy - SIZE // 2 + 0.5) * CELL
        if self.dist[cx, cy] == 0.0 or math.hypot(tx - self.pose[0], ty - self.pose[1]) < 0.05:
            tx, ty = self._goal
        return math.atan2(ty - self.pose[1], tx - self.pose[0]) - self.pose[2]

    # ----- driving -----
    def decide(self, obs: Observation) -> Decision:
        self._integrate(obs)
        new_obstacle = self._map(obs)
        x, y, th = self.pose
        self._goal = (x + obs.goal_distance * math.cos(th + obs.goal_bearing),
                      y + obs.goal_distance * math.sin(th + obs.goal_bearing))
        if obs.time >= self._next_plan or new_obstacle or self.dist is None:
            self._plan()
            self._next_plan = obs.time + REPLAN_PERIOD
        if obs.time < self._backup_until:
            return Decision(Command(-0.5 * self.v_max * C.MANUAL_REVERSE_FACTOR, 0.0), obs.seq)
        moving = abs(obs.velocity_estimate[0]) > 0.02 or abs(obs.velocity_estimate[1]) > 0.05
        if moving or self._still_since is None:
            self._still_since = None if moving else obs.time
        elif obs.time - self._still_since > STUCK_AFTER:
            self._backup_until, self._still_since = obs.time + BACKUP_TIME, None
            self._next_plan = obs.time  # replan after backing up
        heading = self._target_heading()
        if heading is None:
            heading = obs.goal_bearing  # no known route: head for the goal and let safety stop us
        heading = math.atan2(math.sin(heading), math.cos(heading))
        w = float(np.clip(2.0 * heading, -self.w_max, self.w_max))
        if abs(heading) > 0.5:
            return Decision(Command(0.0, w), obs.seq)  # turn in place first
        r = np.where(obs.lidar_valid, obs.lidar, 0.3)
        along, lateral = r * np.cos(obs.lidar_angles), np.abs(r * np.sin(obs.lidar_angles))
        ahead = float(np.min(np.where((along > 0) & (lateral < HALF_WIDTH), along, C.LIDAR_RANGE)))
        room = float(np.clip((ahead - 0.25) / 0.6, 0.0, 1.0))
        v = min(self.v_max * room * math.cos(heading) ** 2, max(0.08, obs.goal_distance))
        return Decision(Command(v, w), obs.seq)
