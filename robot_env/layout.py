"""Room map for task generation: obstacle shapes, an inflated occupancy grid, path
search, and seeded start/goal sampling.

This is planning and evaluation code. It reads the obstacle layout from the model,
so drivers must not use it (the test-only waypoint follower is the stated exception).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import config as C


@dataclass(frozen=True)
class Shape:
    kind: str  # "box" or "cylinder"
    x: float
    y: float
    yaw: float
    hx: float  # box half sizes, or cylinder radius in hx
    hy: float


@dataclass(frozen=True)
class Task:
    seed: int
    start: tuple[float, float, float]  # x, y, yaw
    goal: tuple[float, float]
    path: tuple[tuple[float, float], ...]  # inflated-grid path, start to goal
    path_length: float

    def estimated_travel_time(self, speed: float = 0.30, turn_rate: float = 1.0) -> float:
        """Path length at `speed` plus the heading changes along it at `turn_rate`."""
        pts = np.array(self.path)
        heading = self.start[2]
        turning = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            seg = math.atan2(b[1] - a[1], b[0] - a[0])
            turning += abs(math.atan2(math.sin(seg - heading), math.cos(seg - heading)))
            heading = seg
        return self.path_length / speed + turning / turn_rate


def shapes_from_model(model: mujoco.MjModel) -> list[Shape]:
    """Every solid geom on a static body (the world or bodies welded to it), in world
    coordinates, that reaches below OVERHEAD_CLEARANCE. Excludes the floor, the robot,
    the goal marker, visual-only geoms, and overhead parts (door headers, desk tops)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor = model.geom("floor").id
    shapes = []
    for g in range(model.ngeom):
        body = model.geom_bodyid[g]
        static = model.body_weldid[body] == 0 and model.body_mocapid[body] < 0
        if g == floor or not static or model.geom_contype[g] == 0:
            continue
        x, y, z = data.geom_xpos[g]
        xmat = data.geom_xmat[g].reshape(3, 3)
        half_height = float(np.abs(xmat[2]) @ model.geom_size[g]) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX             else float(model.geom_size[g][1])
        if z - half_height >= C.OVERHEAD_CLEARANCE:
            continue
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])
        size = model.geom_size[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            shapes.append(Shape("box", float(x), float(y), yaw, float(size[0]), float(size[1])))
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shapes.append(Shape("cylinder", float(x), float(y), 0.0, float(size[0]), float(size[0])))
        else:
            raise ValueError(f"unsupported obstacle geom type {model.geom_type[g]}")
    return shapes


def region_of(x: float, y: float) -> str | None:
    for name, (x0, x1, y0, y1) in C.ROOMS.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return None


def clearance(shapes: list[Shape], px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Distance from points to the nearest obstacle surface (0 inside)."""
    best = np.full(np.shape(px), np.inf)
    for s in shapes:
        dx, dy = px - s.x, py - s.y
        if s.kind == "cylinder":
            d = np.maximum(np.hypot(dx, dy) - s.hx, 0.0)
        else:
            c, sn = math.cos(s.yaw), math.sin(s.yaw)
            lx, ly = c * dx + sn * dy, -sn * dx + c * dy
            d = np.hypot(np.maximum(np.abs(lx) - s.hx, 0.0), np.maximum(np.abs(ly) - s.hy, 0.0))
        best = np.minimum(best, d)
    return best


class RoomMap:
    def __init__(self, model: mujoco.MjModel):
        self.shapes = shapes_from_model(model)
        n = int(round(2 * C.FLOOR_HALF_SIZE / C.GRID_RESOLUTION))
        self.n = n
        centers = -C.FLOOR_HALF_SIZE + (np.arange(n) + 0.5) * C.GRID_RESOLUTION
        self.cx, self.cy = np.meshgrid(centers, centers, indexing="ij")
        self.clear = clearance(self.shapes, self.cx, self.cy)
        self.free = self.clear >= C.CIRCUMSCRIBED_RADIUS + C.PLANNING_CLEARANCE

    def cell(self, x: float, y: float) -> tuple[int, int]:
        i = int((x + C.FLOOR_HALF_SIZE) / C.GRID_RESOLUTION)
        j = int((y + C.FLOOR_HALF_SIZE) / C.GRID_RESOLUTION)
        return min(max(i, 0), self.n - 1), min(max(j, 0), self.n - 1)

    def point_clearance(self, x: float, y: float) -> float:
        return float(clearance(self.shapes, np.array(x), np.array(y)))

    def find_path(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]] | None:
        """A* on the inflated grid (8-connected). Returns world points or None."""
        s, g = self.cell(*start), self.cell(*goal)
        if not (self.free[s] and self.free[g]):
            return None
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
        open_heap = [(0.0, s)]
        cost = {s: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == g:
                break
            for di, dj, step in moves:
                nb = (cur[0] + di, cur[1] + dj)
                if not (0 <= nb[0] < self.n and 0 <= nb[1] < self.n) or not self.free[nb]:
                    continue
                if di and dj and not (self.free[cur[0] + di, cur[1]] and self.free[cur[0], cur[1] + dj]):
                    continue  # no corner cutting
                nc = cost[cur] + step
                if nc < cost.get(nb, math.inf):
                    cost[nb] = nc
                    parent[nb] = cur
                    heapq.heappush(open_heap, (nc + math.hypot(g[0] - nb[0], g[1] - nb[1]), nb))
        if g not in cost:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        cells.reverse()
        pts = [(float(self.cx[c]), float(self.cy[c])) for c in cells]
        pts[0], pts[-1] = (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))
        if len(pts) < 2:
            pts = pts + pts
        return self.smooth_path(pts)

    def line_free(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        n = max(2, int(math.dist(a, b) / (C.GRID_RESOLUTION / 2)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            if not self.free[self.cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))]:
                return False
        return True

    def smooth_path(self, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Shortcut the grid path wherever a straight line stays in inflated free space."""
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def frozen_task(self, seed: int) -> Task:
        """A held-out task exactly as frozen (config.HELDOUT_TASKS), with its path in this world.
        Raises ValueError if it is no longer a valid task here (start or goal not free or too
        close to an obstacle, the same region, or no path)."""
        (sx, sy, yaw), (gx, gy) = C.HELDOUT_TASKS[seed]
        for x, y in ((sx, sy), (gx, gy)):
            if not (self.free[self.cell(x, y)] and self.point_clearance(x, y) >= C.START_GOAL_MIN_CLEARANCE):
                raise ValueError(f"held-out task {seed}: ({x:.3f}, {y:.3f}) is no longer a free start/goal")
        if region_of(sx, sy) == region_of(gx, gy) or math.dist((sx, sy), (gx, gy)) < C.START_GOAL_MIN_SEPARATION:
            raise ValueError(f"held-out task {seed}: start and goal no longer valid together")
        path = self.find_path((sx, sy), (gx, gy))
        if path is None:
            raise ValueError(f"held-out task {seed}: no path any more")
        length = float(sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:])))
        return Task(seed, (float(sx), float(sy), float(yaw)), (float(gx), float(gy)), tuple(path), length)

    def sample_task(self, seed: int, max_tries: int = 2000) -> Task:
        """Seeded start and goal in free space, far enough apart, with a feasible path. The
        held-out seeds give their frozen tasks (config.HELDOUT_TASKS), checked to be valid here."""
        if seed in C.HELDOUT_TASKS:
            return self.frozen_task(seed)
        rng = np.random.default_rng(seed)
        lim = C.FLOOR_HALF_SIZE - C.START_GOAL_MIN_CLEARANCE

        def free_point() -> tuple[float, float] | None:
            for _ in range(max_tries):
                x, y = rng.uniform(-lim, lim, size=2)
                if self.point_clearance(x, y) >= C.START_GOAL_MIN_CLEARANCE and self.free[self.cell(x, y)]:
                    return float(x), float(y)
            return None

        for _ in range(max_tries):
            start = free_point()
            goal = free_point()
            if start is None or goal is None:
                break
            if math.dist(start, goal) < C.START_GOAL_MIN_SEPARATION:
                continue
            if region_of(*start) == region_of(*goal):
                continue  # every task crosses at least one doorway
            path = self.find_path(start, goal)
            if path is None:
                continue
            length = float(sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:])))
            yaw = float(rng.uniform(-math.pi, math.pi))
            return Task(seed, (start[0], start[1], yaw), goal, tuple(path), length)
        raise RuntimeError(f"could not sample a feasible task for seed {seed}")
