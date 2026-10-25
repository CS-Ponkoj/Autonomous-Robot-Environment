"""The built-in driver with the saved building map, and the same planner as a route source for the
learned models.

MapDriver is robot_env's BaselineDriver (subclassed here; robot_env is not changed) whose memory
starts from the saved map instead of empty: at reset it is given the drive's start pose on the
map (initial_pose, as an operator sets a real robot's pose when it is switched on), resamples the
map onto its own grid and marks the mapped walls as static. From then on everything is as before:
odometry, its own lidar and depth mapping on top (new obstacles and cats still enter, briefly
seen cells still lapse), the wavefront route and the frontier fallback.

With match=True it also corrects odometry drift every MATCH_PERIOD by matching the lidar hits to
the saved walls: each hit scores its distance to the nearest saved wall (a distance field made
once per drive, read with bilinear interpolation, capped at MATCH_CAP), and a coarse-to-fine
search over x, y and heading offsets keeps the pose with the lowest mean distance. A correction
larger than one cell also forgets the cells the drive confirmed since the last correction (off
the saved walls), since they were mapped with the drifted pose and would stand as ghost walls.
It is off by default until the mapping drives show drift that needs it (build.py reports it).
It matches the scan's valid returns (not the dropout-bridged scan the mapping uses).
Drift along a long featureless corridor cannot be seen in one scan and stays until a doorway or
corner comes into view; the first large correction also forgets every cell confirmed during the
drive (re-seen within CONFIRM_SPAN).

MapRouteSensor is the hybrid route sensor (AD.models.hybrid.route.RouteSensor) with MapDriver as
its planner: the same 4 numbers, from a planner that knows the building."""

from __future__ import annotations

import math

import numpy as np

from robot_env import config as C
from robot_env.baseline import CELL, CONFIRM_SPAN, DYNAMIC_TTL, SIZE, BaselineDriver, _APART, _shift
from robot_env.types import Observation

from ...common.features import LABEL_LEVEL
from ..hybrid.route import ROUTE_FEATURES, RouteSensor
from .savedmap import SavedMap, to_driver_frame

MATCH_PERIOD = 0.5  # s between drift corrections (match=True)
MATCH_LEVELS = ((0.2, 0.04), (0.1, 0.02), (0.05, 0.01))  # (m, rad) search steps, coarse to fine
MATCH_CAP = 0.3  # m: a hit farther than this from any saved wall (a cat, new furniture) counts as this
MATCH_GAIN_M = 0.01  # m: the mean hit distance must improve by this much before a correction applies


class MapDriver(BaselineDriver):
    name = "builtin_map"

    def __init__(self, map_path=None, speed: tuple[float, float] = C.SPEED_LEVELS[C.DEFAULT_SPEED_LEVEL],
                 match: bool = False, odo_bias: tuple[float, float] = (0.0, 0.0),
                 nonmap_ttl: float | None = None, nonmap_clear: bool = False):
        """odo_bias (testing the drift correction only): (relative speed error, yaw-rate bias in
        rad/s) added to this driver's own odometry, as from worn wheels; the robot is unchanged.

        nonmap_ttl: with a saved map every real wall is already known, so an obstacle that is not on
        the map (a cat, a moved chair) is kept only this many seconds after it was last seen,
        instead of becoming static after CONFIRM_SPAN. Without it a cat resting half a second in a
        doorway closes that doorway for the rest of the drive. None: BaselineDriver's rule.

        nonmap_clear (with nonmap_ttl): an off-map obstacle lapses only once a later lidar ray has
        passed through its cell (the cat has gone) and nonmap_ttl has passed since its last hit;
        one that has not been seen gone (a chair behind the robot) stays. A pure time limit also
        forgets real unmapped furniture out of view, and the robot loops back into it."""
        self.saved = SavedMap.load(map_path)
        self.match = match
        self.nonmap_ttl = nonmap_ttl
        self.nonmap_clear = nonmap_clear
        self.odo_bias = tuple(float(b) for b in odo_bias)
        self._placed = False
        super().__init__(speed)

    def reset(self, initial_pose=None, **kw) -> None:
        super().reset()
        self._placed = initial_pose is not None
        self._next_match = 0.0
        self.own_free = np.zeros((SIZE, SIZE), bool)  # cells this drive's own rays saw free (map building)
        self._freed = np.full((SIZE, SIZE), -np.inf)  # last time a ray passed through each cell
        if not self._placed:
            return  # BaselineDriver.__init__ resets without a pose; decide() refuses to drive unplaced
        occ, free = to_driver_frame(self.saved, tuple(float(v) for v in initial_pose))
        self._first[occ] = -1e9  # seen long before the drive: static from the first decision
        self._last[occ] = 0.0
        near = occ.copy()
        for dx, dy in _APART:
            near |= _shift(occ, dx, dy, False)
        self._near_lidar |= near
        self._seen_free |= free
        self.occupied = self._last - self._first >= CONFIRM_SPAN
        from scipy.ndimage import distance_transform_edt
        self._wall_dist = distance_transform_edt(~occ) * CELL  # m to the nearest saved wall cell
        self._prior = occ
        self._last_fix = 0.0

    def decide(self, obs: Observation):
        if not self._placed:
            raise RuntimeError("MapDriver needs reset(initial_pose=(x, y, yaw)) before driving")
        if self.match and obs.time >= self._next_match:
            self._integrate_then_match(obs)
        return super().decide(obs)

    def _integrate(self, obs: Observation) -> None:
        if self.odo_bias != (0.0, 0.0):
            from dataclasses import replace
            v, w = obs.velocity_estimate
            obs = replace(obs, velocity_estimate=(v * (1.0 + self.odo_bias[0]), w + self.odo_bias[1]))
        super()._integrate(obs)

    def _map(self, obs: Observation) -> bool:
        """BaselineDriver's mapping, with the cells this drive's rays see free also kept apart from
        the saved map's (own_free), so a map built from drives with a saved map counts only what
        each drive saw itself."""
        full = self._seen_free
        before = self.occupied
        self._seen_free = np.zeros_like(full)  # this scan's rays only
        try:
            new = super()._map(obs)
        finally:
            fresh = self._seen_free
            self.own_free |= fresh
            self._freed[fresh] = obs.time
            self._seen_free = full | self.own_free
        if self.nonmap_ttl is None or not self._placed:
            return new
        # saved walls stay; anything else lapses nonmap_ttl after its last sighting. Its first
        # sighting is also moved up to its last one, so BaselineDriver's own formula (used by the
        # "nothing reachable" replan and by ghost clearing) can never make it static either.
        off = (self._first > -1e8) & np.isfinite(self._last)
        self._first[off] = self._last[off]
        keep = obs.time - self._last < self.nonmap_ttl
        if self.nonmap_clear:  # not yet seen gone: keep it
            keep |= np.isfinite(self._last) & ~(self._freed > self._last)
        self.occupied = (self._first < -1e8) | keep | (self._refused > obs.time)
        return bool((self.occupied & ~before).any())

    def own_static(self, now: float) -> np.ndarray:
        """Cells this drive itself confirmed occupied (static, not refused), in its own grid."""
        return ((self._last - self._first >= CONFIRM_SPAN) & (self._last > 0.0) & ~(self._refused > now))

    # ----- drift correction (match=True) -----
    def _integrate_then_match(self, obs: Observation) -> None:
        """Correct the pose by matching this scan to the saved walls. Runs before decide()
        integrates the same observation: integrated here once, decide() then adds dt = 0."""
        self._integrate(obs)
        self._next_match = obs.time + MATCH_PERIOD
        hit = obs.lidar_valid & (obs.lidar < C.LIDAR_RANGE - 1e-6)
        if hit.sum() < 20:
            return
        r, a = obs.lidar[hit], obs.lidar_angles[hit]
        x, y, th = self.pose
        field = self._wall_dist

        def cost(dx, dy, dth):
            ang = th + dth + a
            u = (x + dx + r * np.cos(ang)) / CELL + SIZE // 2 - 0.5  # fractional cell coordinates
            v = (y + dy + r * np.sin(ang)) / CELL + SIZE // 2 - 0.5
            i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
            ok = (i0 >= 0) & (i0 < SIZE - 1) & (j0 >= 0) & (j0 < SIZE - 1)
            fu, fv, i0, j0 = u[ok] - i0[ok], v[ok] - j0[ok], i0[ok], j0[ok]
            d = (field[i0, j0] * (1 - fu) * (1 - fv) + field[i0 + 1, j0] * fu * (1 - fv)
                 + field[i0, j0 + 1] * (1 - fu) * fv + field[i0 + 1, j0 + 1] * fu * fv)
            d = np.minimum(d, MATCH_CAP)
            return float((d.sum() + MATCH_CAP * (~ok).sum()) / len(u))

        base = cost(0.0, 0.0, 0.0)
        best = (base, 0.0, 0.0, 0.0)
        for step, astep in MATCH_LEVELS:  # re-centred on the best offset at every level
            _, bx, by, bt = best
            for dx in (bx - step, bx, bx + step):
                for dy in (by - step, by, by + step):
                    for dth in (bt - astep, bt, bt + astep):
                        c = cost(dx, dy, dth)
                        if c < best[0]:
                            best = (c, dx, dy, dth)
        if best[0] > base - MATCH_GAIN_M:
            return
        self.pose += np.array(best[1:])
        # forget what was mapped with the drifted pose when any hit moved by more than a cell
        if math.hypot(best[1], best[2]) + abs(best[3]) * float(np.median(r)) > CELL:
            self._seen_free &= ~self._prior  # drifted rays may have marked saved walls as seen free
            ghost = (self._first >= self._last_fix) & ~self._prior
            self._first[ghost], self._last[ghost] = np.inf, -np.inf
            self.occupied = ((self._last - self._first >= CONFIRM_SPAN) | (obs.time - self._last < DYNAMIC_TTL)
                             | (self._refused > obs.time))
            self.dist = None  # replan on the corrected pose and map
            near = self._prior | self.occupied  # depth points stay out only near walls still mapped
            for dx, dy in _APART:
                near = near | _shift(self._prior | self.occupied, dx, dy, False)
            self._near_lidar = near
        self._last_fix = obs.time


class MapRouteSensor(RouteSensor):
    """RouteSensor with MapDriver as the planner. Route length is scaled by `scale` metres
    (routes across the building are often longer than the online sensor's 10 m)."""

    NAME = "map"
    LEN_SCALE = 20.0

    def __init__(self, map_path=None, scale: float = LEN_SCALE, match: bool = True):
        # RouteSensor.__init__ builds an online planner; this sensor needs the map planner instead
        self.planner = MapDriver(map_path, C.SPEED_LEVELS[LABEL_LEVEL], match=match)
        self.throttle = False
        self.scale = scale
        self._last_plan, self._depth = -math.inf, 0

    def reset(self, initial_pose=None, **kw) -> None:
        self.planner.reset(initial_pose=initial_pose)
        self._last_plan, self._depth = -math.inf, 0

    def read(self, obs: Observation) -> np.ndarray:
        out = super().read(obs)
        p = self.planner
        out[3] = 0.0
        if p.dist is not None:
            cx, cy = p._cell(p.pose[0], p.pose[1])
            if 0 <= cx < p.dist.shape[0] and 0 <= cy < p.dist.shape[1] and np.isfinite(p.dist[cx, cy]):
                out[3] = min(float(p.dist[cx, cy]) * CELL / self.scale, 1.0)
        assert out.shape == (ROUTE_FEATURES,)
        return out


class MapRouteSensorTTL(MapRouteSensor):
    """The map route source where obstacles off the saved map lapse once they have been seen gone
    (a lidar ray through their cell) and NONMAP_TTL seconds have passed since their last hit
    (MapDriver nonmap_ttl and nonmap_clear): a cat that rested in a doorway no longer closes it,
    and unmapped furniture out of view is still remembered."""

    NAME = "map2"
    NONMAP_TTL = 5.0

    def __init__(self, map_path=None, scale: float = MapRouteSensor.LEN_SCALE, match: bool = True,
                 nonmap_ttl: float = NONMAP_TTL, nonmap_clear: bool = True):
        super().__init__(map_path, scale, match)
        self.planner.nonmap_ttl = nonmap_ttl
        self.planner.nonmap_clear = nonmap_clear
