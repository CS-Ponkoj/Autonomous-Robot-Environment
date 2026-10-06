"""Rule-based baseline driver: the reference that later drivers and realism changes
are compared against.

It sees only the Observation (lidar, encoder velocity, goal distance and bearing), through the
same decide(observation) plug as every driver. It never touches the simulator, the world map,
the task generator, or ground truth. Like a real robot, it builds its own memory:

1. Odometry: integrates the encoder velocity estimate into a pose in its own start frame
   (trapezoidal: the average of the previous and current estimates over each interval).
2. Mapping: a 0.1 m occupancy grid from its lidar hits. A cell seen over CONFIRM_SPAN or longer
   is static for the rest of the episode; one seen only briefly (a cat walking by) lapses
   DYNAMIC_TTL after its last sighting, so moving things leave no phantom walls. Unknown cells
   are treated as free (optimistic exploration). Clearing cells along rays was tried and
   rejected: grazing rays erased real walls (6/20 development successes instead of 16/20).
3. Planning: a wavefront (breadth-first distance field) from the goal over the grid inflated
   by the robot radius, recomputed every 0.5 s or when new obstacles appear on the path. With no
   route (space it could not move through closing the way), it explores instead: the wavefront
   runs from the frontier, the cells its rays have seen free next to cells never seen.
4. Following: walk down the distance field for about 0.5 m from the robot's cell and head for
   that point, turn in place when the heading error is large, and slow down when the corridor
   ahead is short.
5. Recovery: when it commands motion but the encoders show none for STUCK_AFTER (the safety
   layer refusing every move, say beside a corner the plan passed too close to), it marks the
   space just ahead as refused in its own map for REFUSED_TTL and replans around it; then it
   backs up only if the lidar shows the space behind clear (never blind), and otherwise turns
   in place toward the side with more room. It never crawls below V_CREEP while the lane ahead
   has room (a crawl reads as standing still), since the safety layer guards every move.
With ideal sensors this works well; with realistic odometry drift and lidar noise (later
realism rounds) it degrades, which is what a baseline should reveal.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C
from .lidar import bridge
from .sensing import depth_points
from .types import Command, Decision, Observation

STOP_COMMAND = Command(0.0, 0.0)

CELL = 0.1  # m
SIZE = 220  # cells per side (22 m): the start is at the center, so any goal on a 10 x 10 m floor fits
INFLATE = int(math.ceil((C.CIRCUMSCRIBED_RADIUS + 0.06) / CELL))  # cells of clearance around obstacles
REPLAN_PERIOD = 0.5  # s
STUCK_AFTER = 1.0  # s of commanded but absent motion
BACKUP_TIME = 0.6  # s of reversing to get out (only when the space behind is seen clear)
TURN_TIME = 0.8  # s of turning in place to get out, when backing up is not seen clear
REAR_CLEAR = 0.45  # m the lidar must see free behind the robot before it backs up
REFUSED_TTL = 8.0  # s the space ahead where the robot could not move stays blocked in its map
CONFIRM_SPAN = 0.5  # s a cell must be seen over to count as static
DYNAMIC_TTL = 2.0  # s a briefly seen cell stays occupied after its last sighting
V_CREEP = 0.06  # m/s: the slowest forward command while the lane ahead has room
DEPTH_APART = 2  # cells (0.2 m) a depth point must be from anything the lidar has hit, to enter the map
LOOKAHEAD = 5  # cells (0.5 m)
HALF_WIDTH = C.FOOTPRINT_HALF_WIDTH + 0.04
_DISK = [(dx, dy) for dx in range(-INFLATE, INFLATE + 1) for dy in range(-INFLATE, INFLATE + 1)
         if dx * dx + dy * dy <= INFLATE * INFLATE]
_APART = [(dx, dy) for dx in range(-DEPTH_APART, DEPTH_APART + 1) for dy in range(-DEPTH_APART, DEPTH_APART + 1)]
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
        self.occupied = np.zeros((SIZE, SIZE), bool)  # derived: static, recent, or refused cells
        self._seen_free = np.zeros((SIZE, SIZE), bool)  # cells a valid ray has passed through
        self._near_lidar = np.zeros((SIZE, SIZE), bool)  # within DEPTH_APART of a lidar hit, ever
        self._first = np.full((SIZE, SIZE), np.inf)  # first and last sighting of each cell
        self._last = np.full((SIZE, SIZE), -np.inf)
        self._refused = np.full((SIZE, SIZE), -np.inf)  # blocked until this time (no progress there)
        self._turn_until = -1.0
        self._last_v = 0.0  # the forward speed last commanded
        self._turn_w = 0.0
        self.dist = None
        self._next_plan = 0.0
        self._now = 0.0
        self._goal = None
        self._still_since: float | None = None
        self._backup_until = -1.0
        self._lidar = None  # robot_env.lidar's state: dropped readings bridged, as in the safety layer

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
        px, py = x + obs.lidar[hit] * np.cos(a), y + obs.lidar[hit] * np.sin(a)
        lx, ly = self._cell(px, py)  # the lidar's cells, and the band around them (DEPTH_APART)
        ok = (lx >= 0) & (lx < SIZE) & (ly >= 0) & (ly < SIZE)
        for dx, dy in _APART:
            nx, ny = lx[ok] + dx, ly[ok] + dy
            fit = (nx >= 0) & (nx < SIZE) & (ny >= 0) & (ny < SIZE)
            self._near_lidar[nx[fit], ny[fit]] = True
        # what the lidar plane passes over, from the depth sensor: only cells the lidar has never
        # come near (skirting and trim under a wall the lidar sees, in a frame where it misses
        # that wall, would otherwise close doorways a cell at a time)
        low = depth_points(obs)
        if len(low):
            wx = x + low[:, 0] * math.cos(th) - low[:, 1] * math.sin(th)
            wy = y + low[:, 0] * math.sin(th) + low[:, 1] * math.cos(th)
            dx_, dy_ = self._cell(wx, wy)
            fit = (dx_ >= 0) & (dx_ < SIZE) & (dy_ >= 0) & (dy_ < SIZE)
            fit[fit] = ~self._near_lidar[dx_[fit], dy_[fit]]
            px, py = np.concatenate((px, wx[fit])), np.concatenate((py, wy[fit]))
        cx, cy = self._cell(px, py)
        inside = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
        cx, cy = cx[inside], cy[inside]
        self._first[cx, cy] = np.minimum(self._first[cx, cy], obs.time)
        # the space each valid ray crossed before its return (or its whole range): seen free
        valid = obs.lidar_valid
        steps = np.arange(CELL / 2, C.LIDAR_RANGE, CELL)
        ra = th + obs.lidar_angles[valid]
        reach = np.where(obs.lidar[valid] < C.LIDAR_RANGE - 1e-6, obs.lidar[valid] - CELL, C.LIDAR_RANGE)
        d = steps[None, :]
        keep = d < reach[:, None]
        fx, fy = self._cell(x + (d * np.cos(ra)[:, None])[keep], y + (d * np.sin(ra)[:, None])[keep])
        ok = (fx >= 0) & (fx < SIZE) & (fy >= 0) & (fy < SIZE)
        self._seen_free[fx[ok], fy[ok]] = True
        self._last[cx, cy] = obs.time
        before = self.occupied
        self.occupied = ((self._last - self._first >= CONFIRM_SPAN) | (obs.time - self._last < DYNAMIC_TTL)
                         | (self._refused > obs.time))
        return bool((self.occupied & ~before).any())

    def _refuse_ahead(self, now: float) -> None:
        """Mark the robot's lane just ahead as refused (no progress there) for REFUSED_TTL."""
        x, y, th = self.pose
        along = np.arange(C.FOOTPRINT_HALF_LENGTH + 0.05, C.FOOTPRINT_HALF_LENGTH + 0.36, CELL / 2)
        across = np.arange(-HALF_WIDTH, HALF_WIDTH + 1e-9, CELL / 2)
        a, b = np.meshgrid(along, across)
        cx, cy = self._cell(x + a * math.cos(th) - b * math.sin(th), y + a * math.sin(th) + b * math.cos(th))
        ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
        self._refused[cx[ok], cy[ok]] = now + REFUSED_TTL
        self.occupied[cx[ok], cy[ok]] = True

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
        goal = np.zeros((SIZE, SIZE), bool)
        goal[gx, gy] = True
        dist = self._wavefront(goal, blocked, rx, ry)
        inside = 0 <= rx < SIZE and 0 <= ry < SIZE
        if inside and not np.isfinite(dist[rx, ry]):
            # no route: explore toward the frontier (seen free, next to never seen, not blocked)
            unseen = ~self._seen_free & ~self.occupied
            near_unseen = np.zeros_like(unseen)
            for dx, dy, _cost in _STEPS[:4]:
                near_unseen |= _shift(unseen, dx, dy, False)
            frontier = self._seen_free & near_unseen & ~blocked
            if frontier.any():
                dist = self._wavefront(frontier, blocked, rx, ry)
            if not np.isfinite(dist[rx, ry]) and (self._refused > -np.inf).any():
                # nothing reachable at all: the refused space may be what closed it; forget it
                self._refused[:] = -np.inf
                self.occupied = (self._last - self._first >= CONFIRM_SPAN) | (self._now - self._last < DYNAMIC_TTL)
                return self._plan()
        self.dist = dist

    @staticmethod
    def _wavefront(sources: np.ndarray, blocked: np.ndarray, rx: int, ry: int) -> np.ndarray:
        """Grid distance to the nearest source cell, around blocked cells; stops once the robot's
        neighbourhood (rx, ry) has settled."""
        dist = np.where(sources, 0.0, np.inf)
        reached = None
        for k in range(4 * SIZE):
            best = dist
            for dx, dy, cost in _STEPS:
                best = np.minimum(best, _shift(dist, dx, dy, np.inf) + cost)
            best[blocked & ~sources] = np.inf
            best[sources] = 0.0
            if np.array_equal(best, dist):
                break
            dist = best
            if 0 <= rx < SIZE and 0 <= ry < SIZE and np.isfinite(dist[rx, ry]):
                reached = k if reached is None else reached
                if k - reached > LOOKAHEAD + 2:  # the robot's neighbourhood is settled: stop early
                    break
        return dist

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
        self._now = obs.time
        # a dropped reading takes its ray's recent return (kept up to date every decision)
        seen, self._lidar = bridge(obs, self._lidar)
        self._integrate(obs)
        new_obstacle = self._map(seen)
        x, y, th = self.pose
        self._goal = (x + obs.goal_distance * math.cos(th + obs.goal_bearing),
                      y + obs.goal_distance * math.sin(th + obs.goal_bearing))
        if obs.time >= self._next_plan or new_obstacle or self.dist is None:
            self._plan()
            self._next_plan = obs.time + REPLAN_PERIOD
        if obs.time < self._backup_until:
            self._last_v = 0.0
            return Decision(Command(-0.5 * self.v_max * C.MANUAL_REVERSE_FACTOR, 0.0), obs.seq)
        if obs.time < self._turn_until:
            self._last_v = 0.0
            return Decision(Command(0.0, self._turn_w), obs.seq)
        r = np.where(seen.lidar_valid, seen.lidar, C.LIDAR_RANGE)
        along, lateral = r * np.cos(obs.lidar_angles), np.abs(r * np.sin(obs.lidar_angles))
        moving = abs(obs.velocity_estimate[0]) > 0.02 or abs(obs.velocity_estimate[1]) > 0.05
        if moving or self._still_since is None:
            self._still_since = None if moving else obs.time
        elif obs.time - self._still_since > STUCK_AFTER:
            # no progress: the way ahead is refused here; remember it, plan around it, and get out
            # by backing up only into space seen clear, else by turning toward the roomier side
            self._still_since = None
            if self._last_v > 0.04:  # it tried to go forward and could not: that space is refused
                self._refuse_ahead(obs.time)
            self._plan()
            self._next_plan = obs.time + REPLAN_PERIOD
            behind = (along < 0) & (lateral < HALF_WIDTH)
            known = seen.lidar_valid[behind]
            if known.all() and np.min(-along[behind], initial=C.LIDAR_RANGE) - C.FOOTPRINT_HALF_LENGTH >= REAR_CLEAR:
                self._backup_until = obs.time + BACKUP_TIME
            else:
                left = float(np.mean(np.where(seen.lidar_valid, r, 0.0)[obs.lidar_angles > 0]))
                right = float(np.mean(np.where(seen.lidar_valid, r, 0.0)[obs.lidar_angles < 0]))
                self._turn_w = self.w_max * (1.0 if left >= right else -1.0)
                self._turn_until = obs.time + TURN_TIME
            return Decision(STOP_COMMAND, obs.seq)
        heading = self._target_heading()
        lost = heading is None
        if heading is None:
            heading = obs.goal_bearing  # no known route: head for the goal and let safety stop us
        heading = math.atan2(math.sin(heading), math.cos(heading))
        w = float(np.clip(2.0 * heading, -self.w_max, self.w_max))
        if abs(heading) > 0.5:
            self._last_v = 0.0
            return Decision(Command(0.0, w), obs.seq)  # turn in place first
        # a reading still unknown counts as no return here (the speed choice), since the safety
        # layer blocks any motion into it
        ahead = float(np.min(np.where((along > 0) & (lateral < HALF_WIDTH), along, C.LIDAR_RANGE)))
        room = float(np.clip((ahead - 0.25) / 0.6, 0.0, 1.0))
        v = min(self.v_max * room * math.cos(heading) ** 2, max(0.08, obs.goal_distance))
        if room > 0.0:  # never a crawl the encoders read as standing still
            v = max(v, min(V_CREEP, self.v_max))
        elif lost:  # no route and the way ahead blocked: turn toward the roomier side, never stand
            left = float(np.mean(np.where(seen.lidar_valid, r, 0.0)[obs.lidar_angles > 0]))
            right = float(np.mean(np.where(seen.lidar_valid, r, 0.0)[obs.lidar_angles < 0]))
            w = self.w_max * (1.0 if left >= right else -1.0)
        self._last_v = v
        return Decision(Command(v, w), obs.seq)
