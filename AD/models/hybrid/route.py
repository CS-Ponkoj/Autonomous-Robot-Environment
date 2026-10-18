"""Design sketch (nothing trained): a hybrid input that tells a learned policy which way the
built-in planner's route goes, so the policy decides how to drive (speed, turning, yielding to
cats) while the planner supplies what a memoryless policy lacks: a map and a route.

The route sensor is the built-in driver itself, run alongside and fed the same observations every
decision (its map and route come from lidar, depth and odometry only, never ground truth); its
commands are thrown away. From its plan it reads, every decision:

  route_sin, route_cos  the direction to the route point LOOKAHEAD cells (0.5 m) ahead along the
                        planned route, relative to the heading (the same point the built-in
                        driver steers for)
  route_known           1 if a route to the goal (or, failing that, to a frontier) exists, else 0
  route_len             the planned route length to the goal in metres, / 10, capped at 1 (equal
                        to the straight goal distance in open space, longer around walls)

These 4 numbers are appended to AD.common.features.vector(obs) (78), so a hybrid policy reads 82.
The CNN's scalar branch takes them like the goal values; a policy started from a cloned CNN
would start with zero weights on them (no change in behaviour until training uses them).

Reads the driver's planning state through its private members (_target_heading, dist, pose,
_cell): robot_env is not changed, only read. If those change, this sketch needs updating."""

from __future__ import annotations

import math

import numpy as np

from robot_env import config as C
from robot_env.baseline import BaselineDriver
from robot_env.types import Observation

from ...common.features import LABEL_LEVEL, vector

ROUTE_FEATURES = 4
CELL = 0.1  # m: the built-in driver's map cell (robot_env.baseline.CELL)
REPLAN_EVERY = 0.5  # s: at most one new route search per this much time (the planner's own period)


class RouteSensor:
    """The built-in planner run as a sensor: its route direction and length, every decision.

    throttle=True (off by default): the planner replans at most once per REPLAN_EVERY, with one
    retry per search; left alone it also replans on every newly seen obstacle cell (nearly every
    decision with moving cats). With the compiled route search (robot_env.kernels.wavefront) that
    costs about 10 ms at the 95th percentile, so the default matches what the hybrid was trained
    with; the throttle is for slower machines."""

    def __init__(self, throttle: bool = False):
        self.planner = BaselineDriver(C.SPEED_LEVELS[LABEL_LEVEL])
        self.throttle = throttle
        self._last_plan, self._depth = -math.inf, 0
        if throttle:
            search = self.planner._plan

            def throttled(tight: bool = False):
                p = self.planner
                if self._depth == 0 and p.dist is not None and p._now - self._last_plan < REPLAN_EVERY:
                    return None  # keep the last route
                if self._depth >= 2:
                    return None  # one retry per search at most (each search is a full-map wavefront)
                if self._depth == 0:
                    self._last_plan = p._now
                self._depth += 1
                try:
                    return search(tight)  # its own retry (a tight plan) runs inside this search
                finally:
                    self._depth -= 1
            self.planner._plan = throttled

    def reset(self) -> None:
        self.planner.reset()
        self._last_plan, self._depth = -math.inf, 0

    def read(self, obs: Observation) -> np.ndarray:
        """Advance the planner on this observation (its map and route update) and read its route."""
        self.planner.decide(obs)  # the command is not used; this keeps the map and plan current
        p = self.planner
        heading = p._target_heading()
        out = np.zeros(ROUTE_FEATURES, np.float32)
        if heading is None:  # no route: the goal direction, flagged as unknown
            heading = obs.goal_bearing
        else:
            out[2] = 1.0
        out[0], out[1] = math.sin(heading), math.cos(heading)
        if p.dist is not None:
            cx, cy = p._cell(p.pose[0], p.pose[1])
            if 0 <= cx < p.dist.shape[0] and 0 <= cy < p.dist.shape[1] and np.isfinite(p.dist[cx, cy]):
                out[3] = min(float(p.dist[cx, cy]) * CELL / 10.0, 1.0)
        return out


def hybrid_vector(obs: Observation, sensor: RouteSensor) -> np.ndarray:
    """features.vector(obs) with the route sensor's 4 values appended (82 numbers)."""
    return np.concatenate([vector(obs), sensor.read(obs)])
