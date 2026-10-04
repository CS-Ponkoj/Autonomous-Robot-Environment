"""Wandering cats (a demo and experiment feature; off by default).

A cat is a mocap (kinematic) body: its motion is prescribed by this module, not by contact
forces, so its own policy must keep it out of walls, furniture, the robot, and other cats. Every
physics step the cat moves a tiny, speed-limited amount, and a new pose is only committed when
the whole body envelope (two circles covering the collision capsule and head) is clear. So a cat
never overlaps static geometry and stops before touching the robot or another cat.

Behavior (a seeded random stream, independent of the task and noise seeds): walk with gentle
wandering, pause, sit (body tilted up, rear on the floor), and occasionally dart. A cat closer
than FLEE_DISTANCE to the robot turns away and walks off.

Cats are solid (their collision geoms are in group 5, which the lidar sees) and contact with the
robot counts as a collision. The safety layer treats each lidar scan as a static snapshot: there
is no velocity-aware prediction of moving obstacles yet, so cats are a demo and experiment
feature, not safety evidence. Legs, ears, and tail are visual only (not sensed, not solid).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .layout import clearance, shapes_from_model

MAX_CATS = 4
COATS = (("0.86 0.52 0.22 1", "0.95 0.75 0.5 1"),   # orange tabby
         ("0.42 0.43 0.46 1", "0.62 0.63 0.66 1"),   # grey
         ("0.08 0.08 0.09 1", "0.95 0.95 0.95 1"),   # black and white
         ("0.55 0.42 0.30 1", "0.80 0.70 0.58 1"))   # brown

# Body envelope in the cat frame (x forward): circles (x offset, radius) covering the collision
# capsule (x -0.235 to 0.195, half-width 0.075) and the head sphere (x up to 0.26).
ENVELOPE = ((-0.10, 0.14), (0.12, 0.14))
WALL_GAP = 0.03  # m kept between the envelope and walls or furniture
ROBOT_GAP = 0.30  # m kept between the envelope and the robot's circumscribed circle
CAT_GAP = 0.05  # m kept between two cats' envelopes
FLEE_DISTANCE = 0.70  # m (envelope to robot circle): closer than this, the cat walks away

V_MAX = 1.0  # m/s
ACCEL = 3.0  # m/s^2
YAW_RATE = 3.0  # rad/s
YAW_ACCEL = 20.0  # rad/s^2
SIT_PITCH = math.radians(35)
PITCH_RATE = 2.0  # rad/s

RASTER = 0.02  # m: clearance raster cell
RASTER_SLACK = RASTER * math.sqrt(2) / 2  # max distance from a point to its nearest raster node
_raster_cache: dict = {}


def cat_xml(n: int) -> str:
    """MJCF for n cats (mocap bodies), injected into the world only when cats are enabled."""
    if not 0 <= n <= MAX_CATS:
        raise ValueError(f"cats must be 0 to {MAX_CATS}")
    parts = []
    for i in range(n):
        fur, light = COATS[i % len(COATS)]
        x = -20.0 - 2 * i  # spawned properly at reset
        parts.append(f'''
    <body name="cat{i}" mocap="true" pos="{x} 0 0">
      <!-- collision (sensed by the lidar, solid for the robot): body capsule and head -->
      <geom name="cat{i}_body" type="capsule" fromto="-0.16 0 0.13 0.12 0 0.13" size="0.075"
            group="5" rgba="{fur}" solref="0.02 1" solimp="0.9 0.95 0.001"/>
      <geom name="cat{i}_head" type="sphere" pos="0.2 0 0.2" size="0.06"
            group="5" rgba="{fur}" solref="0.02 1" solimp="0.9 0.95 0.001"/>
      <!-- visual only (group 5 is not drawn: the body and head are drawn by these copies) -->
      <geom type="capsule" fromto="-0.16 0 0.13 0.12 0 0.13" size="0.074" rgba="{fur}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="sphere" pos="0.2 0 0.2" size="0.059" rgba="{fur}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="ellipsoid" pos="0.04 0 0.11" size="0.07 0.06 0.05" rgba="{light}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="capsule" fromto="0.08 0.04 0.0 0.09 0.04 0.1" size="0.018" rgba="{fur}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="capsule" fromto="0.08 -0.04 0.0 0.09 -0.04 0.1" size="0.018" rgba="{fur}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="capsule" fromto="-0.14 0.045 0.0 -0.13 0.045 0.1" size="0.02" rgba="{fur}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="capsule" fromto="-0.14 -0.045 0.0 -0.13 -0.045 0.1" size="0.02" rgba="{fur}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="capsule" fromto="-0.22 0 0.15 -0.32 0 0.24" size="0.016" rgba="{fur}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="capsule" fromto="-0.32 0 0.24 -0.36 0 0.32" size="0.014" rgba="{fur}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="ellipsoid" pos="0.255 0 0.185" size="0.025 0.035 0.022" rgba="{light}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="box" pos="0.19 0.035 0.26" size="0.012 0.018 0.022" euler="0 0 0.3" rgba="{fur}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="box" pos="0.19 -0.035 0.26" size="0.012 0.018 0.022" euler="0 0 -0.3" rgba="{fur}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="sphere" pos="0.245 0.025 0.215" size="0.009" rgba="0.55 0.75 0.25 1" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="sphere" pos="0.245 -0.025 0.215" size="0.009" rgba="0.55 0.75 0.25 1" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="sphere" pos="0.278 0 0.19" size="0.007" rgba="0.85 0.45 0.5 1" contype="0" conaffinity="0" group="2" mass="0"/>
    </body>''')
    return "".join(parts)


def _clearance_raster(model) -> tuple[np.ndarray, float]:
    """Distance to the nearest static obstacle at every RASTER node of the floor (cached)."""
    shapes = shapes_from_model(model)
    key = tuple((s.kind, round(s.x, 4), round(s.y, 4), round(s.yaw, 4), round(s.hx, 4), round(s.hy, 4)) for s in shapes)
    if key not in _raster_cache:
        half = C.FLOOR_HALF_SIZE
        axis = np.arange(-half, half + RASTER / 2, RASTER)
        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        grid = clearance(shapes, gx, gy)
        # outside the floor counts as blocked
        _raster_cache[key] = (grid, -half)
    return _raster_cache[key]


@dataclass
class Cat:
    index: int
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    w: float = 0.0
    pitch: float = 0.0
    state: str = "walk"
    state_until: float = 0.0
    target_v: float = 0.0
    target_yaw: float = 0.0
    blocked: bool = False
    frozen: bool = False  # touching the robot: holds still (never pushes)
    transitions: list = field(default_factory=list)
    logged: int = 0  # transitions already reported to the drive log


class CatHerd:
    def __init__(self, sim, n: int, seed: int):
        if not 0 <= n <= MAX_CATS:
            raise ValueError(f"cats must be 0 to {MAX_CATS}")
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("cat_seed must be a non-negative integer")
        self.sim, self.n, self.seed = sim, n, int(seed)
        m = sim.model
        self._mocap = [m.body_mocapid[m.body(f"cat{i}").id] for i in range(n)]
        self._grid, self._origin = _clearance_raster(m)
        self.cats: list[Cat] = []
        self.time = 0.0

    # ----- geometry -----
    def _static_clear(self, px: float, py: float) -> float:
        i = int(round((px - self._origin) / RASTER))
        j = int(round((py - self._origin) / RASTER))
        if not (0 <= i < self._grid.shape[0] and 0 <= j < self._grid.shape[1]):
            return -1.0
        return float(self._grid[i, j]) - RASTER_SLACK

    @staticmethod
    def _circles(x: float, y: float, yaw: float):
        c, s = math.cos(yaw), math.sin(yaw)
        return [(x + c * dx, y + s * dx, r) for dx, r in ENVELOPE]

    def _robot(self):
        rx, ry, _ = self.sim.true_pose()
        return rx, ry

    def gaps(self, cat: Cat, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        """(wall gap, robot gap, other-cat gap) of the envelope at a candidate pose."""
        rx, ry = self._robot()
        circles = self._circles(x, y, yaw)
        wall = min(self._static_clear(cx, cy) - r for cx, cy, r in circles)
        robot = min(math.hypot(cx - rx, cy - ry) - r - C.CIRCUMSCRIBED_RADIUS for cx, cy, r in circles)
        other = math.inf
        for o in self.cats:
            if o is cat:
                continue
            for ox, oy, orad in self._circles(o.x, o.y, o.yaw):
                for cx, cy, r in circles:
                    other = min(other, math.hypot(cx - ox, cy - oy) - r - orad)
        return wall, robot, other

    def pose_ok(self, cat: Cat, x: float, y: float, yaw: float) -> bool:
        wall, robot, other = self.gaps(cat, x, y, yaw)
        return wall >= WALL_GAP and robot >= ROBOT_GAP and other >= CAT_GAP

    # ----- episode -----
    def reset(self, start: tuple[float, float], goal: tuple[float, float]) -> None:
        """Respawn deterministically from cat_seed at free points away from the robot and goal."""
        self.rng = np.random.default_rng(self.seed)
        self.time = 0.0
        self.cats = []
        for i in range(self.n):
            cat = Cat(i)
            for _ in range(5000):
                x, y = self.rng.uniform(-C.FLOOR_HALF_SIZE + 0.3, C.FLOOR_HALF_SIZE - 0.3, 2)
                yaw = float(self.rng.uniform(-math.pi, math.pi))
                if math.hypot(x - start[0], y - start[1]) < 1.5 or math.hypot(x - goal[0], y - goal[1]) < 1.0:
                    continue
                cat.x, cat.y, cat.yaw = float(x), float(y), yaw
                if self.pose_ok(cat, cat.x, cat.y, cat.yaw) and \
                        min(self._static_clear(cx, cy) - r for cx, cy, r in self._circles(x, y, yaw)) >= 0.1:
                    break
            else:
                raise RuntimeError("no free spawn point for a cat")
            cat.target_yaw = cat.yaw
            self.cats.append(cat)
            self._enter(cat, "walk")
        self._write_poses()

    # ----- behavior (50 Hz) -----
    def _enter(self, cat: Cat, state: str) -> None:
        r = self.rng
        cat.state = state
        duration = {"walk": r.uniform(3.0, 8.0), "pause": r.uniform(1.0, 4.0), "sit": r.uniform(3.0, 8.0),
                     "dart": r.uniform(0.4, 0.8), "flee": 1.0}[state]
        cat.state_until = self.time + duration
        cat.target_v = {"walk": r.uniform(0.15, 0.35), "pause": 0.0, "sit": 0.0, "dart": 0.9, "flee": 0.35}[state]
        if state in ("walk", "dart"):
            cat.target_yaw = self._open_heading(cat)
        cat.transitions.append((round(self.time, 3), state))

    def _open_heading(self, cat: Cat, avoid: tuple[float, float] | None = None) -> float:
        """A heading with room ahead: random candidates scored by clearance 0.5 m ahead (and,
        when fleeing, by distance from `avoid`)."""
        best, best_score = cat.yaw, -math.inf
        for _ in range(12):
            h = cat.yaw + float(self.rng.uniform(-math.pi, math.pi))
            ax, ay = cat.x + 0.5 * math.cos(h), cat.y + 0.5 * math.sin(h)
            score = min(self._static_clear(ax, ay), 0.6)
            if avoid is not None:
                score += 2.0 * math.hypot(ax - avoid[0], ay - avoid[1])
            if score > best_score:
                best, best_score = h, score
        return best

    def tick(self) -> None:
        """Behavior decisions, called on every 50 Hz control tick."""
        rx, ry = self._robot()
        r = self.rng
        for cat in self.cats:
            robot_gap = min(math.hypot(cx - rx, cy - ry) - rad - C.CIRCUMSCRIBED_RADIUS
                            for cx, cy, rad in self._circles(cat.x, cat.y, cat.yaw))
            if robot_gap < FLEE_DISTANCE and cat.state not in ("flee",):
                self._enter(cat, "flee")
                cat.target_yaw = self._open_heading(cat, avoid=(rx, ry))
            elif cat.blocked and cat.state in ("walk", "dart", "flee"):
                cat.target_yaw = self._open_heading(cat, avoid=(rx, ry) if cat.state == "flee" else None)
            elif self.time >= cat.state_until:
                if cat.state in ("walk", "flee", "dart"):
                    self._enter(cat, "dart" if r.random() < 0.08 else "pause" if r.random() < 0.6 else "walk")
                elif cat.state == "pause":
                    self._enter(cat, "sit" if r.random() < 0.4 else "walk")
                else:
                    self._enter(cat, "walk")
            elif cat.state == "walk" and r.random() < 0.02:
                cat.target_yaw = cat.yaw + float(r.normal(0.0, 0.6))  # gentle wandering
            cat.blocked = False

    # ----- motion (every physics step) -----
    def freeze(self, index: int, frozen: bool = True) -> None:
        """A cat touching the robot stops at once and stays still until the contact ends (a
        kinematic body must never push the robot)."""
        cat = self.cats[index]
        cat.frozen = frozen
        if frozen:
            cat.v = cat.w = 0.0

    def new_transitions(self) -> list[tuple[int, float, str]]:
        """State changes not yet reported: (cat index, time, new state)."""
        out = []
        for cat in self.cats:
            out += [(cat.index, t, st) for t, st in cat.transitions[cat.logged:]]
            cat.logged = len(cat.transitions)
        return out

    def step(self, dt: float) -> None:
        self.time += dt
        for cat in self.cats:
            if cat.frozen:
                continue
            sitting = cat.state == "sit"
            want_pitch = SIT_PITCH if sitting else 0.0
            cat.pitch += float(np.clip(want_pitch - cat.pitch, -PITCH_RATE * dt, PITCH_RATE * dt))
            v0, w0 = cat.v, cat.w
            err = math.atan2(math.sin(cat.target_yaw - cat.yaw), math.cos(cat.target_yaw - cat.yaw))
            want_w = 0.0 if sitting or abs(cat.pitch) > 0.05 else float(np.clip(3.0 * err, -YAW_RATE, YAW_RATE))
            cat.w += float(np.clip(want_w - cat.w, -YAW_ACCEL * dt, YAW_ACCEL * dt))
            want_v = 0.0 if sitting or abs(cat.pitch) > 0.05 else cat.target_v * max(0.0, math.cos(err)) ** 2
            cat.v += float(np.clip(want_v - cat.v, -ACCEL * dt, ACCEL * dt))
            yaw = cat.yaw + cat.w * dt
            x, y = cat.x + cat.v * math.cos(yaw) * dt, cat.y + cat.v * math.sin(yaw) * dt
            if self.pose_ok(cat, x, y, yaw):
                cat.x, cat.y, cat.yaw = x, y, math.atan2(math.sin(yaw), math.cos(yaw))
            else:
                # Not clear: stay put this step and brake within the acceleration limits
                # (a kinematic body cannot pass through anything; the behavior picks a new heading).
                cat.v = max(0.0, v0 - ACCEL * dt) if v0 > 0 else min(0.0, v0 + ACCEL * dt)
                cat.w = 0.0 if abs(w0) <= YAW_ACCEL * dt else w0 - math.copysign(YAW_ACCEL * dt, w0)
                cat.blocked = True
        self._write_poses()

    def _write_poses(self) -> None:
        d = self.sim.data
        for cat, mid in zip(self.cats, self._mocap):
            # sitting: pitch up about the hips (x -0.12, z 0.06), rear on the floor
            px, pz = -0.12, 0.06
            sp, cp = math.sin(cat.pitch), math.cos(cat.pitch)
            back = px * (1 - cp) + pz * sp  # keeps the hips fixed while the body tilts up
            lift = pz * (1 - cp) - px * sp
            d.mocap_pos[mid] = (cat.x + math.cos(cat.yaw) * back, cat.y + math.sin(cat.yaw) * back, lift)
            cy, sy = math.cos(cat.yaw / 2), math.sin(cat.yaw / 2)
            cp, sp = math.cos(-cat.pitch / 2), math.sin(-cat.pitch / 2)
            d.mocap_quat[mid] = (cy * cp, -sy * sp, cy * sp, sy * cp)  # yaw then pitch (nose up)

    def truth(self) -> list[dict]:
        """Evaluation-only state of every cat (for logs)."""
        return [{"cat": c.index, "pose": [c.x, c.y, c.yaw], "v": c.v, "w": c.w, "pitch": c.pitch,
                 "state": c.state} for c in self.cats]
