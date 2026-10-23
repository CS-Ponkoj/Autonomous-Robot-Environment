"""Route input for learned policies: which way a planner's route goes, so the policy decides how
to drive (speed, turning, yielding to cats) while the planner supplies what a memoryless policy
lacks: a map and a route.

Route sources are pluggable. Each has reset(initial_pose=None, **kw) and read(obs) returning the
4 numbers below, plus a NAME and a LEN_SCALE (metres per unit of route_len). Sources register
by name (register_route_source) and are made with make_route_source(name); a route checkpoint
stores the spec of the source it was trained with (route_spec), and drivers may run another
source by name at evaluation. "online" is the built-in planner below (its map built live).

The route sensor is the built-in driver itself, run alongside and fed the same observations every
decision (its map and route come from lidar, depth and odometry only, never ground truth); its
commands are thrown away. From its plan it reads, every decision:

  route_sin, route_cos  the direction to the route point LOOKAHEAD cells (0.5 m) ahead along the
                        planned route, relative to the heading (the same point the built-in
                        driver steers for)
  route_known           1 if a route to the goal (or, failing that, to a frontier) exists, else 0
  route_len             the planned route length to the goal in metres, / LEN_SCALE (10 for
                        "online"), capped at 1 (equal to the straight goal distance in open
                        space, longer around walls)

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
    """The built-in planner run as a sensor ("online"): its route direction and length, every
    decision. The initial pose is not needed (its map is built from the start of the episode).

    throttle=True (off by default): the planner replans at most once per REPLAN_EVERY, with one
    retry per search; left alone it also replans on every newly seen obstacle cell (nearly every
    decision with moving cats). With the compiled route search (robot_env.kernels.wavefront) that
    costs about 10 ms at the 95th percentile, so the default matches what the hybrid was trained
    with; the throttle is for slower machines."""

    NAME = "online"
    LEN_SCALE = 10.0  # m per unit of route_len (kept from the first hybrid checkpoints)

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

    def reset(self, initial_pose=None, **kw) -> None:
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
                out[3] = min(float(p.dist[cx, cy]) * CELL / self.LEN_SCALE, 1.0)
        return out


_SOURCES: dict = {RouteSensor.NAME: RouteSensor}
# sources defined elsewhere, imported on first use (their modules import this one)
_LAZY = {"map": ("AD.models.mapplan.driver", "MapRouteSensor")}  # the planner on the saved building map


def register_route_source(name: str, cls) -> None:
    """Make a route source available by name (a class with reset(initial_pose=None, **kw) and
    read(obs) -> 4 values; its route-length scale as LEN_SCALE or an instance `scale`)."""
    _SOURCES[name] = cls


def route_sources() -> list[str]:
    return sorted(set(_SOURCES) | set(_LAZY))


def make_route_source(name: str = "online", **kw):
    if name not in _SOURCES and name in _LAZY:
        import importlib
        module, cls = _LAZY[name]
        register_route_source(name, getattr(importlib.import_module(module), cls))
    if name not in _SOURCES:
        raise ValueError(f"unknown route source {name!r}; known: {route_sources()}")
    source = _SOURCES[name](**kw)
    source.source_name = name  # the registered name (a subclass may still carry its parent's NAME)
    source.source_kwargs = dict(kw)  # what it was made with (route_spec records the map it used)
    return source


def len_scale(source) -> float:
    """Metres per unit of route_len for this source (an instance `scale` wins over LEN_SCALE)."""
    return float(getattr(source, "scale", None) or source.LEN_SCALE)


def _map_details(source) -> dict:
    """For the "map" source: which saved map (its folder name, e.g. map_v2) and whether drift
    correction was on, so a checkpoint stays tied to its map if the default map changes."""
    from pathlib import Path
    path = getattr(source, "source_kwargs", {}).get("map_path")
    if path is None:
        from AD.models.mapplan.savedmap import DEFAULT
        path = DEFAULT
    planner = getattr(source, "planner", None)
    return {"map": Path(path).parent.name, "match": bool(getattr(planner, "match", False))}


def route_spec(source) -> dict:
    """What a checkpoint stores about the route source it was trained with: its name and route
    length scale, and for "map" the saved map's id and drift correction (or whatever the source's
    own describe() returns)."""
    name = getattr(source, "source_name", source.NAME)
    spec = {"name": name, "len_scale": len_scale(source)}
    if callable(getattr(source, "describe", None)):
        spec.update(source.describe())
    elif name == "map":
        spec.update(_map_details(source))
    return spec


def source_from_spec(spec: dict):
    """The route source a checkpoint was trained with, as it was: for "map", the same saved map
    (AD/models/mapplan/runs/<map>/map.npz) and drift correction; older specs without these get the
    current defaults."""
    name = spec.get("name", "online")
    if name == "map" and "map" in spec:
        from AD.models.mapplan.savedmap import DEFAULT
        return make_route_source(name, map_path=DEFAULT.parent.parent / spec["map"] / "map.npz",
                                 match=bool(spec.get("match", True)))
    return make_route_source(name)


def reset_driver(driver, initial_pose=None) -> None:
    """Reset any driver, passing the episode's start pose to those that take it (route drivers)."""
    if not hasattr(driver, "reset"):
        return
    import inspect
    params = inspect.signature(driver.reset).parameters
    if initial_pose is not None and ("initial_pose" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())):
        driver.reset(initial_pose=initial_pose)
    else:
        driver.reset()


def hybrid_vector(obs: Observation, sensor) -> np.ndarray:
    """features.vector(obs) with the route sensor's 4 values appended (82 numbers)."""
    return np.concatenate([vector(obs), sensor.read(obs)])
