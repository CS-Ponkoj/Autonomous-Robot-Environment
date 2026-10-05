"""Wandering cats: realistic, animated, and solid (on by default in the window: 4 cats).

Each cat is the rigged cat model (robot_env/cat_rig.py: "Cat" by Vr-cvantorium, CC BY 4.0) with
its own coat, animated procedurally (robot_env/cat_motion.py). Its motion is prescribed, not
pushed by contact forces, so this module keeps it out of walls, furniture, the robot, and other
cats.

Timing. Behaviour decisions run on the 50 Hz control tick. Motion and animation are computed
every ANIM_PERIOD (100 Hz), one sample ahead: at the start of each interval the cat's pose at
its end is computed (root motion within the speed and acceleration limits, a clearance check of
the whole visible cat with a margin for the interval, then the skeleton). On every physics step
the root, the colliders, and the skin are interpolated between the two samples, so the
colliders move continuously at the physics rate and the skin drawn on screen is exactly the
pose the colliders have.

Sensing and contact. Fourteen colliders per cat (two torso capsules, the head, two capsules per
leg, and three along the tail), fitted to the mesh and attached to its bones, are solid for the
robot and seen by the lidar (group 5). Only the ears are visual. Contact with the robot counts as a
collision; the safety layer treats each lidar scan as a still snapshot (no motion prediction
yet), so cats are a realism and experiment feature, not safety evidence.
"""

from __future__ import annotations

import copy
import functools
import math
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import kernels
from .cat_motion import SIT_UP_TIME, CatAnimator
from .cat_rig import load_rig, model_parts
from .layout import clearance, region_of, shapes_from_model
from .sim import WorldExtra

MAX_CATS = 4
DEFAULT_CATS = 4
ANIM_PERIOD = 0.01  # s: motion and animation sample period (100 Hz); colliders interpolate between
ROOT_SUBSTEPS = 4  # root integration steps per sample (speed and acceleration limits per substep)
COATS = ("cat_anisotropic1.png", "coat_ginger_mackerel.png", "coat_black_tuxedo.png", "coat_grey_tabby.png")
COAT_NAMES = ("brown tabby", "ginger tabby", "black and white", "grey tabby")

TAIL_ROOM = 0.06  # m of spare gap at which the tail sways fully (less room: a calmer tail)
SIGHT_RANGE = 6.0  # m: farther than this a cat pays the robot no attention
ALERT_DISTANCE = 1.2  # m: a robot this close (in sight) makes a cat raise its tail
TAIL_LIFT = 1.0  # rad the tail base rises when the cat is alert
LOOK_ROOM = 0.15  # m beyond the robot gap: closer than this a cat does not turn its head to the robot
CLOSER_TOLERANCE = 1e-4  # m: "never closer" inside a gap, to numerical precision
MOVE_ROOM = 0.005  # m a cat's root moves keep beyond each gap, leaving room for its limbs to move
SWEEP_TOL = 0.0005  # m: half of the agreed 1 mm tolerance on gaps between two samples
LAND_TOL = 0.0009  # m into a gap that a paw put down short may take (inside the agreed 1 mm)
WALL_GAP = 0.03  # m kept between the visible cat and walls or furniture
ROBOT_GAP = 0.30  # m kept between the visible cat and the robot's circumscribed circle
CAT_GAP = 0.05  # m kept between two cats
FLEE_DISTANCE = 0.70  # m (cat to robot circle): closer than this, the cat walks away
MIN_TURN_SPEED = 0.05  # m/s: slower than this a cat turns on the spot with pivot steps...
PIVOT_RATE = 0.8  # rad/s ...at most this fast (tools/gait_check.py: paws stay planted up to here at 100 Hz)
SCARE_DISTANCE = 0.25  # m (cat to robot circle): this close, the cat flees even from a still robot (inside
# ROBOT_GAP, which cats keep themselves: only a robot that moved in on a cat can startle it)

V_MAX = 1.0  # m/s
ACCEL = 3.0  # m/s^2
YAW_RATE = 3.0  # rad/s
YAW_ACCEL = 20.0  # rad/s^2

NAV_CELL = 0.1  # m: route-planning grid
NAV_CLEAR = 0.20  # m of clearance a route cell needs (the cat's half-width plus its wall gap)
ROUTE_ROBOT_ROOM = 0.25  # m beyond the robot gap that routes keep from the robot (half width + room)
ARRIVE = 0.3  # m from the destination counts as arrived
STALL_TIME = 4.0  # s without 0.15 m of progress: a travelling cat picks a new route
RESTING = ("sit", "pause")
SIT_CHECK_TIME = 3.0  # s: the longest a sit-down and getting up (paws shuffling first) may take
SIT_CHECK_CHUNK = 6  # sit-down poses checked per 10 ms sample (about 2 ms of work)
SIT_ROOM = 0.05  # m beyond the robot and cat gaps a sit-down keeps from them where they are (they move)
SIT_RISE = 0.01  # m: a seated cat gets up once the robot comes within this of the robot gap from its
# room to get up in (still clear of it); a room is taken only this clear of the robot and the cats
ROBOT_RELIEF = 0.005  # m inside ROBOT_GAP before the robot counts as having driven in (not a creep)
ROOM_STEP = 0.01  # m: a seated cat's room to get up in is covered by circles grouping moves this small
PATROL_MEMORY = 120.0  # s: a room never visited counts as last seen this long ago
PATROL_BASE = 10.0  # s added to every room's weight, so a recent room is still possible
# Local planner: every PLAN_PERIOD each moving cat scores candidate motions over PLAN_HORIZON
PLAN_PERIOD = 0.1  # s
PLAN_HORIZON = 0.6  # s
PLAN_STEPS = 12  # 0.05 s apart: fine enough to see a turning head or tail swing in
PLAN_CUSHION = 0.015  # m of room a cat's plans keep beyond each gap, for its legs, head, and tail
PLAN_ROOM_WEIGHT = 0.8  # how much a cat prefers room to spare
PLAN_ESCAPE = 0.6  # score for winning back 1 cm of room when inside the cushion
PLAN_ROOM = 0.15  # m of spare gap beyond which more room scores no better
PLAN_KEEP = 0.08  # score bonus for keeping the current command (no dithering)
PLAN_IDLE = 0.25  # score cost of standing still when the cat wants to go somewhere
PLAN_TROUBLE = 0.3  # score cost of a command that, held on, would run into something
PLAN_TURN_COMMIT = 0.3  # score bonus for keeping the way round while turning around
PLAN_REJECT_TIME = 0.5  # s a command the per-sample guard refused is not planned again
PLAN_TURN_SPACE = 0.35  # m of clearance around its middle that a cat needs to turn round
PLAN_SPACE_GAIN = 1.5  # score per PLAN_TURN_SPACE of extra room gained while turning around
SIDESTEP_SPEED = 0.08  # m/s: a slow side step out of a tight spot (the gait re-plants its paws)
REVERSE_SPEED = 0.12  # m/s: a short backing-up step
# Yielding: a cat in the robot's way moves aside (the robot never pushes a cat)
YIELD_DISTANCE = 1.5  # m ahead of the robot that counts as its path
YIELD_TTC = 2.5  # s: a moving robot this close in time makes the cat move aside
YIELD_SPEED = 0.6  # m/s: a cat getting out of the robot's way...
YIELD_HURRY = 1.3  # ...hurries more the faster the robot closes in (this times its closing speed, up to V_MAX)
CAT_REACH = 0.45  # m: farther than any part of a cat reaches from its root (0.374 m at rest; the tail sways)
ROBOT_HARD = 0.10  # m kept from the robot's circle even when the robot drives inside ROBOT_GAP
YIELD_BACK_COST = 1.0  # m added to a yield route toward the robot's side of the cat
YIELD_SIDE = 0.15  # m beyond the robot's half width on each side that counts as its path
CHOKE_RADIUS = 0.7  # m around a doorway centre where cats pass but never rest
ESCAPE_STEP = 0.25  # s per step of a worked-out way out of a tight spot
ESCAPE_ROOM = 0.025  # m of spare gap that counts as out of the tight spot (the planner's cushion and 1 cm)
ESCAPE_DISTANCE = 0.12  # m from where it was stuck
ESCAPE_DEPTH = 24  # steps at most
ESCAPE_EXPANSIONS = 600  # poses explored at most (the search runs only after a stall)
GIVE_WAY_AFTER = 2.0  # s a cat that wants to move may get nowhere before it gives way
GIVE_WAY_MOVE = 0.05  # m of progress that counts as getting somewhere
ESCAPE_RETRY = 0.5  # s after a search found no way out, the cat looks again
UNSTICK_AFTER = 1.0  # s blocked: the cat first steps toward open space, then carries on
STUCK_AFTER = 0.5  # s with no feasible way forward: the planner may then use its cushion

RASTER = 0.02  # m: clearance raster cell
RASTER_SLACK = RASTER * math.sqrt(2) / 2  # max distance from a point to its nearest raster node
_raster_cache: dict = {}


# ----- colliders and envelope, fitted to the mesh (cat frame, bind pose) -----
@dataclass(frozen=True)
class Collider:
    name: str
    bone_a: int
    off_a: tuple  # endpoint a in bone_a's frame
    bone_b: int
    off_b: tuple  # endpoint b in bone_b's frame (= a for a sphere)
    radius: float
    sphere: bool = False


def _brake(value: float, step: float) -> float:
    """Toward zero by at most step (a held cat's speed decays within the limits)."""
    return 0.0 if abs(value) <= step else value - math.copysign(step, value)


def yield_speed(closing: float) -> float:
    """How fast a cat gets out of the robot's way: YIELD_SPEED, more when the robot closes in on it
    fast (YIELD_HURRY times the closing speed; a robot moving away or past does not hurry it), at
    most V_MAX."""
    return min(max(YIELD_SPEED, YIELD_HURRY * max(0.0, closing)), V_MAX)


def _root_rates(v, w, cmd_v, cmd_w, dt):
    """Forward speed and turn rate one step toward a command, within the acceleration limits
    (scalars or arrays). Slower than MIN_TURN_SPEED a cat turns only by pivot steps (at most
    PIVOT_RATE), so a faster turn first slows to that rate while the cat keeps that much speed."""
    rate = np.where(np.abs(v) >= MIN_TURN_SPEED, YAW_RATE, PIVOT_RATE)
    want = np.minimum(np.maximum(cmd_w, -rate), rate)
    w = w + np.minimum(np.maximum(want - w, -YAW_ACCEL * dt), YAW_ACCEL * dt)
    new_v = v + np.minimum(np.maximum(cmd_v - v, -ACCEL * dt), ACCEL * dt)
    return _hold_speed(v, new_v, w)


_LIMITS = np.array([MIN_TURN_SPEED, YAW_RATE, PIVOT_RATE, YAW_ACCEL, ACCEL])  # for kernels.rollout


def _hold_speed(v, new_v, w):
    """While a cat still turns faster than PIVOT_RATE it keeps up to MIN_TURN_SPEED of its speed
    (it may slow, never speed up, for this), so it never pivots faster than its pivot steps."""
    keep = (np.abs(w) > PIVOT_RATE + 1e-9) & (np.abs(new_v) < np.minimum(np.abs(v), MIN_TURN_SPEED))
    new_v = np.where(keep, np.sign(v) * np.minimum(np.abs(v), MIN_TURN_SPEED), new_v)
    if np.ndim(new_v) == 0:
        return float(new_v), float(w)
    return new_v, w


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _cover(poses: np.ndarray, step: float = ROOM_STEP) -> np.ndarray:
    """A few circles covering the circles of a sequence of poses (P, n, 3): for each envelope
    circle, its positions in turn grouped while within `step` of the group's first, each group
    one circle there, big enough for every member."""
    out = []
    for i in range(poses.shape[1]):
        track = poses[:, i]
        start = 0
        while start < len(track):
            c = track[start, :2]
            d = np.hypot(track[start:, 0] - c[0], track[start:, 1] - c[1])
            end = start + int(np.argmax(d > step)) if (d > step).any() else len(track)
            out.append((c[0], c[1], float((d[:end - start] + track[start:end, 2]).max())))
            start = end
    return np.array(out)


def _core(joint_name: str) -> str:
    """'Character1_LeftArm_015' -> 'LeftArm'."""
    parts = joint_name.split("_")
    return parts[1] if parts[0].startswith("Character") else parts[0]


@functools.cache
def colliders() -> tuple[Collider, ...]:
    """Torso (two capsules), head (sphere), and two capsules per leg, fitted to the mesh: each
    radius covers 85% of the skin vertices that bone group carries (so the collider stays inside
    the visible cat), and the torso axis follows the body's centre, not the spine joints."""
    rig = load_rig()
    v, J, W = rig.vertices, rig.joints, rig.weights
    dom = J[np.arange(len(v)), W.argmax(1)]
    j = rig.joint

    def group(*names):
        return np.isin(dom, [j(n) for n in names])

    def axis_radius(pts, a, b, q=85):
        ab = b - a
        t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        return float(np.percentile(np.linalg.norm(pts - (a + t[:, None] * ab), axis=1), q))

    out = []
    torso = v[group("Hips_01", "Spine_010", "Spine1_011", "Spine2_012", "Spine3_013", "LeftUpLeg_02",
                    "RightUpLeg_06", "LeftShoulder_014", "RightShoulder_018")]

    def centre(x0):
        sl = torso[np.abs(torso[:, 0] - x0) < 0.012]
        return np.array([x0, 0.0, (sl[:, 2].min() + sl[:, 2].max()) / 2])
    rear, mid, front = centre(-0.17), centre(-0.04), centre(0.08)
    for name, (a, ba), (b, bb) in (("torso_rear", (rear, "Hips_01"), (mid, "Spine1_011")),
                                   ("torso_front", (mid, "Spine1_011"), (front, "Spine3_013"))):
        pts = torso[(torso[:, 0] >= min(a[0], b[0]) - 0.02) & (torso[:, 0] <= max(a[0], b[0]) + 0.02)]
        out.append(Collider(name, j(ba), tuple(a - rig.bind_pos[j(ba)]), j(bb), tuple(b - rig.bind_pos[j(bb)]),
                            axis_radius(pts, a, b)))
    head = v[group("Head_024", "Neck1_023")]
    head = head[head[:, 2] < 0.285]  # without the ears (visual only)
    hc = (head.min(0) + head.max(0)) / 2
    out.append(Collider("head", j("Head_024"), tuple(hc - rig.bind_pos[j("Head_024")]), j("Head_024"),
                        tuple(hc - rig.bind_pos[j("Head_024")]), float(np.percentile(np.linalg.norm(head - hc, axis=1), 85)), True))
    for side in ("Left", "Right"):
        for name, a, b, grp in ((f"{side}_upper_arm", f"{side}Arm", f"{side}ForeArm", (f"{side}Arm",)),
                                (f"{side}_forearm", f"{side}ForeArm", f"{side}Hand", (f"{side}ForeArm", f"{side}Hand")),
                                (f"{side}_shank", f"{side}Leg", f"{side}Foot", (f"{side}Leg",)),
                                (f"{side}_foot", f"{side}Foot", f"{side}ToeBase", (f"{side}Foot", f"{side}ToeBase"))):
            ja = [k for k, nm in enumerate(rig.joint_names) if _core(nm) == a][0]
            jb = [k for k, nm in enumerate(rig.joint_names) if _core(nm) == b][0]
            pa, pb = rig.bind_pos[ja].copy(), rig.bind_pos[jb].copy()
            ids = [k for k, nm in enumerate(rig.joint_names) if _core(nm) in grp]
            pts = v[np.isin(dom, ids)]
            r = axis_radius(pts, pa, pb)
            if pb[2] - r < 0.0:  # a paw: rest the capsule's end on the floor, not below it
                pb[2] = r
            out.append(Collider(name, ja, (0.0, 0.0, 0.0), jb, tuple(pb - rig.bind_pos[jb]), r))
    # The tail is solid too (a robot would bump it, and the lidar sees it), so the robot keeps its
    # safety distance from the whole cat as drawn: three capsules along the tail's bones.
    for name, a, b, grp in (("tail_base", "Tail_026", "Tail2_028", ("Tail_026", "Tail1_027")),
                            ("tail_mid", "Tail2_028", "Tail4_030", ("Tail2_028", "Tail3_029")),
                            ("tail_tip", "Tail4_030", "TailEnd_032", ("Tail4_030", "Tail5_031", "TailEnd_032"))):
        ja, jb = j(a), j(b)
        pts = v[group(*grp)]
        r = axis_radius(pts, rig.bind_pos[ja], rig.bind_pos[jb])
        out.append(Collider(name, ja, (0.0, 0.0, 0.0), jb, (0.0, 0.0, 0.0), r))
    return tuple(out)


@functools.cache
def envelope_points() -> tuple[tuple[int, tuple, float], ...]:
    """Circles (bone, offset in the bone frame, radius) whose union contains the whole visible
    cat (projected on the floor) in any pose the bones take: each skin vertex belongs to the
    circle of its dominant bone group, with a radius covering every such vertex plus 1 cm."""
    rig = load_rig()
    v, J, W = rig.vertices, rig.joints, rig.weights
    dom = J[np.arange(len(v)), W.argmax(1)]
    groups = {
        "Hips_01": ("Hips_01", "Spine_010", "LeftUpLeg_02", "RightUpLeg_06", "Tail_026", "Tail1_027"),
        "Spine2_012": ("Spine1_011", "Spine2_012"),
        "Spine3_013": ("Spine3_013", "LeftShoulder_014", "RightShoulder_018", "Neck_022"),
        "Head_024": ("Neck1_023", "Head_024", "EyeL_00", "EyeR_025"),
        "Tail2_028": ("Tail2_028", "Tail3_029"),
        "TailEnd_032": ("Tail4_030", "Tail5_031", "TailEnd_032"),
        "LeftLeg_03": ("LeftLeg_03", "LeftFoot_04", "LeftToeBase_05"),
        "RightLeg_07": ("RightLeg_07", "RightFoot_08", "RightToeBase_09"),
        "LeftForeArm_016": ("LeftArm_015", "LeftForeArm_016", "LeftHand_017"),
        "RightForeArm_020": ("RightArm_019", "RightForeArm_020", "RightHand_021"),
    }
    out = []
    for bone, members in groups.items():
        b = rig.joint(bone)
        pts = v[np.isin(dom, [rig.joint(m) for m in members])][:, :2]
        c = (pts.min(0) + pts.max(0)) / 2
        r = float(np.linalg.norm(pts - c, axis=1).max()) + 0.01
        off = np.array([c[0], c[1], 0.0]) - rig.bind_pos[b] * [1, 1, 0]
        out.append((b, tuple(off), r))
    return tuple(out)


# ----- world model -----
def cat_world(n: int) -> WorldExtra:
    """Assets, bodies (bones and colliders), skins, and files for n cats (none when n == 0)."""
    if not 0 <= n <= MAX_CATS:
        raise ValueError(f"cats must be 0 to {MAX_CATS}")
    assets, bodies, skins, files = [], [], [], {}
    for i in range(n):
        textures = {"anisotropic1": COATS[i % len(COATS)], "anisotropic2": "cat_anisotropic2.png",
                    "anisotropic3": "cat_anisotropic3.png"}
        a, b, s, f = model_parts(f"cat{i}_", textures)
        assets.append(a)
        bodies.append(b)
        skins.append(s)
        files.update(f)
        for k, col in enumerate(colliders()):
            geom = (f'type="sphere" size="{col.radius:.4f}"' if col.sphere else
                    f'type="capsule" size="{col.radius:.4f} {0.5 * _bind_length(col):.4f}"')
            bodies.append(f'<body name="cat{i}_c{k}" mocap="true" pos="{-20 - 2 * i} 0 0">'
                          f'<geom name="cat{i}_{col.name}" {geom} group="5" rgba=".9 .5 .2 1" '
                          f'solref="0.02 1" solimp="0.9 0.95 0.001"/></body>')
    return WorldExtra("".join(assets), "".join(bodies), "".join(skins), files)


def _bind_length(col: Collider) -> float:
    rig = load_rig()
    a = rig.bind_pos[col.bone_a] + col.off_a
    b = rig.bind_pos[col.bone_b] + col.off_b
    return float(np.linalg.norm(b - a))


_raster_of_model: dict = {}  # id(model) -> (model, raster): the last few models (static geoms never move)


def _clearance_raster(model) -> tuple[np.ndarray, float]:
    """Distance to the nearest static obstacle at every RASTER node of the floor (cached per
    layout, and per model object, since reading a model's obstacles costs milliseconds)."""
    hit = _raster_of_model.get(id(model))
    if hit is not None and hit[0] is model:
        return hit[1]
    shapes = shapes_from_model(model)
    key = tuple((s.kind, round(s.x, 4), round(s.y, 4), round(s.yaw, 4), round(s.hx, 4), round(s.hy, 4)) for s in shapes)
    if key not in _raster_cache:
        half = C.FLOOR_HALF_SIZE
        axis = np.arange(-half, half + RASTER / 2, RASTER)
        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        _raster_cache[key] = (clearance(shapes, gx, gy), -half)
    if len(_raster_of_model) >= 4:  # keep a few (each entry holds its model alive)
        _raster_of_model.pop(next(iter(_raster_of_model)))
    _raster_of_model[id(model)] = (model, _raster_cache[key])
    return _raster_cache[key]


_nav_cache: dict = {}


def _nav_grid(model) -> tuple[np.ndarray, float]:
    """Cells of the route grid where a cat fits (clearance at least NAV_CLEAR), cached per world."""
    grid, origin = _clearance_raster(model)
    key = id(grid)
    if key not in _nav_cache:
        step = int(round(NAV_CELL / RASTER))
        _nav_cache[key] = (grid[::step, ::step] >= NAV_CLEAR, origin)
    return _nav_cache[key]


_route_tree: dict = {}  # the last breadth-first tree: (grid, start, avoid) -> (free cells, predecessors)


def plan_route(model, start: tuple[float, float], goal: tuple[float, float],
               avoid: tuple[float, float, float] | None = None) -> list[tuple[float, float]] | None:
    """A route for a cat from start to goal over the free route grid (8-connected breadth-first
    search), straightened where the straight line stays free. `avoid` (x, y, radius) keeps the
    route out of a disc (the robot), except for cells farther from its centre than the start (a
    cat already inside may still leave). None when there is no route. The search tree from a
    start serves every goal (the same route as a search that stops at the goal), so routes from
    one place to many candidate spots (a cat choosing where to give way) search once."""
    free, origin = _nav_grid(model)
    n = free.shape[0]
    if avoid is not None:
        ax, ay, ar = avoid
        centres = origin + NAV_CELL * np.arange(n)
        d = np.hypot(centres[:, None] - ax, centres[None, :] - ay)
        inside = math.hypot(start[0] - ax, start[1] - ay)
        free = free & ~((d < ar) & (d < inside - 1e-6))

    def cell(p):
        return (min(max(int(round((p[0] - origin) / NAV_CELL)), 0), n - 1),
                min(max(int(round((p[1] - origin) / NAV_CELL)), 0), n - 1))

    def nearest_free(c):
        if free[c]:
            return c
        for r in range(1, 6):
            for i in range(c[0] - r, c[0] + r + 1):
                for j in range(c[1] - r, c[1] + r + 1):
                    if 0 <= i < n and 0 <= j < n and free[i, j]:
                        return (i, j)
        return None
    a, b = nearest_free(cell(start)), nearest_free(cell(goal))
    if a is None or b is None:
        return None
    key = (id(_nav_grid(model)[0]), a, (float(start[0]), float(start[1])), avoid)
    if key not in _route_tree:
        came = np.empty(n * n, dtype=np.int64)
        kernels.bfs_tree(free, a[0], a[1], came)
        _route_tree.clear()
        _route_tree[key] = (free, came)
    free, came = _route_tree[key]
    k = b[0] * n + b[1]
    if came[k] == -2:
        return None
    chain = [k]
    while came[chain[-1]] >= 0:
        chain.append(int(came[chain[-1]]))
    chain.reverse()
    ci = np.array([c // n for c in chain], dtype=np.int64)
    cj = np.array([c % n for c in chain], dtype=np.int64)
    keep = np.empty(len(chain), dtype=np.int64)
    m = kernels.straighten(free, ci, cj, keep)
    return [(origin + ci[i] * NAV_CELL, origin + cj[i] * NAV_CELL) for i in keep[1:m]]


def _plan_route_reference(model, start: tuple[float, float], goal: tuple[float, float],
                          avoid: tuple[float, float, float] | None = None) -> list[tuple[float, float]] | None:
    """plan_route in plain Python (the specification the kernels are tested against)."""
    from collections import deque
    free, origin = _nav_grid(model)
    n = free.shape[0]
    if avoid is not None:
        ax, ay, ar = avoid
        centres = origin + NAV_CELL * np.arange(n)
        d = np.hypot(centres[:, None] - ax, centres[None, :] - ay)
        inside = math.hypot(start[0] - ax, start[1] - ay)
        free = free & ~((d < ar) & (d < inside - 1e-6))

    def cell(p):
        return (min(max(int(round((p[0] - origin) / NAV_CELL)), 0), n - 1),
                min(max(int(round((p[1] - origin) / NAV_CELL)), 0), n - 1))

    def nearest_free(c):
        if free[c]:
            return c
        for r in range(1, 6):
            for i in range(c[0] - r, c[0] + r + 1):
                for j in range(c[1] - r, c[1] + r + 1):
                    if 0 <= i < n and 0 <= j < n and free[i, j]:
                        return (i, j)
        return None
    a, b = nearest_free(cell(start)), nearest_free(cell(goal))
    if a is None or b is None:
        return None
    came = {a: None}
    queue = deque([a])
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    while queue:
        c = queue.popleft()
        if c == b:
            break
        for di, dj in steps:
            nb = (c[0] + di, c[1] + dj)
            if 0 <= nb[0] < n and 0 <= nb[1] < n and nb not in came and free[nb] and \
                    (di == 0 or dj == 0 or (free[c[0] + di, c[1]] and free[c[0], c[1] + dj])):
                came[nb] = c
                queue.append(nb)
    if b not in came:
        return None
    cells = [b]
    while came[cells[-1]] is not None:
        cells.append(came[cells[-1]])
    cells.reverse()

    def line_free(c0, c1):
        k = max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]), 1)
        for t in range(k + 1):
            i = int(round(c0[0] + (c1[0] - c0[0]) * t / k))
            j = int(round(c0[1] + (c1[1] - c0[1]) * t / k))
            if not free[i, j]:
                return False
        return True
    kept = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_free(cells[i], cells[j]):
            j -= 1
        kept.append(cells[j])
        i = j
    return [(origin + c[0] * NAV_CELL, origin + c[1] * NAV_CELL) for c in kept[1:]]


def _quat_from_z(d: np.ndarray) -> np.ndarray:
    """Quaternions (w, x, y, z) rotating +z onto each row of d (unit vectors), vectorized."""
    q = np.empty((len(d), 4))
    q[:, 0] = 1.0 + np.minimum(np.maximum(d[:, 2], -1.0), 1.0)
    q[:, 1] = -d[:, 1]  # z cross d
    q[:, 2] = d[:, 0]
    q[:, 3] = 0.0
    flip = q[:, 0] < 1e-9  # pointing straight down
    q[flip] = (0.0, 1.0, 0.0, 0.0)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


@dataclass
class Sample:
    """A cat's pose at one animation sample: root, joint positions/rotations in the cat frame, and
    its colliders in the world (endpoints, and the orientation of each collider's axis)."""
    x: float
    y: float
    yaw: float
    pos: np.ndarray
    rot: np.ndarray
    ends: np.ndarray | None = None  # (colliders, 2, 3) world endpoints
    quat: np.ndarray | None = None  # (colliders, 4) world orientation (+z along the axis)
    local: np.ndarray | None = None  # envelope circle centres in the cat frame (cached; read only)
    circ: np.ndarray | None = None  # envelope circles at this sample's own root (cached; read only)


@dataclass
class Cat:
    index: int
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    w: float = 0.0
    pitch: float = 0.0  # kept for the log format (whole-body pitch is now done by the skeleton)
    state: str = "walk"
    state_until: float = 0.0
    target_v: float = 0.0
    target_yaw: float = 0.0
    blocked: bool = False  # could not move at all in the last sample
    cmd_v: float = 0.0  # the local planner's command: forward speed, turn rate, side step
    cmd_w: float = 0.0
    cmd_lat: float = 0.0
    plan_at: float = 0.0  # next planning time
    idle_since: float | None = None  # the planner found no way forward since then
    no_sit: bool = False  # sitting down here did not fit (stays standing until its next state)
    sit_probe: list | None = None  # a sit-down being checked: [animator copy, last pose, poses done, paws, swept]
    room: np.ndarray | None = None  # while sitting: circles (m, 3) of the room it needs to get up in
    rejected: dict = field(default_factory=dict)  # command -> until when the hard guard refused it
    held: int = 0  # samples rejected because the animated pose came too close (evaluation)
    escape: list = field(default_factory=list)  # worked-out steps out of a tight spot: (v, w, lat, end time)
    escaped: int = 0  # times the cat worked its way out of a tight spot (evaluation)
    sees_robot: bool = False  # the robot is in plain sight (refreshed every behaviour tick)
    still_from: tuple | None = None  # (time, x, y) since the cat, wanting to move, last gained GIVE_WAY_MOVE
    frozen: bool = False  # touching the robot: holds still (never pushes)
    transitions: list = field(default_factory=list)
    logged: int = 0  # transitions already reported to the drive log
    animator: CatAnimator | None = None
    route: list = field(default_factory=list)  # waypoints to the current destination (travel)
    destination: tuple | None = None
    blocked_since: float | None = None  # when the cat started being blocked (unstick after UNSTICK_AFTER)
    progress_at: float = 0.0  # time of the last 0.15 m of progress (stall detection)
    progress_from: tuple = (0.0, 0.0)
    visited: set = field(default_factory=set)  # regions this cat has been in (evaluation)
    last_in: dict = field(default_factory=dict)  # region -> last time the cat was there (patrol choice)
    prev: Sample | None = None  # interpolation runs from prev (time t0) to next (t0 + ANIM_PERIOD)
    next: Sample | None = None
    t0: float = 0.0


class CatHerd:
    def __init__(self, sim, n: int, seed: int):
        if not 0 <= n <= MAX_CATS:
            raise ValueError(f"cats must be 0 to {MAX_CATS}")
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("cat_seed must be a non-negative integer")
        self.sim, self.n, self.seed = sim, n, int(seed)
        if n:
            kernels.warm()  # compiled (or loaded) before the first step, not during a frame
        m = sim.model
        self.rig = load_rig()
        self.cols = colliders()
        self.env = envelope_points()
        self._col_mocap = np.array([[m.body_mocapid[m.body(f"cat{i}_c{k}").id] for k in range(len(self.cols))]
                                    for i in range(n)], dtype=int).reshape(n, len(self.cols))
        self._bone_mocap = np.array([[m.body_mocapid[m.body(f"cat{i}_b{j}").id] for j in range(len(self.rig.joint_names))]
                                     for i in range(n)], dtype=int).reshape(n, len(self.rig.joint_names))
        self._col_a = np.array([c.bone_a for c in self.cols])
        self._col_b = np.array([c.bone_b for c in self.cols])
        self._off_a = np.array([c.off_a for c in self.cols])
        self._off_b = np.array([c.off_b for c in self.cols])
        self._env_bone = np.array([e[0] for e in self.env])
        self._env_off = np.array([e[1] for e in self.env])
        self._env_r = np.array([e[2] for e in self.env])
        names = self.rig.joint_names
        self._eye_bones = np.array([names.index("Character1_EyeL_00"), names.index("Character1_EyeR_025")])
        self._grid, self._origin = _clearance_raster(m)
        self.cats: list[Cat] = []
        self.time = 0.0
        self._radii = np.array([c.radius for c in self.cols])
        k = len(self.cols)
        # per-cat blend: value = start + delta * f, f = (time - t0) / ANIM_PERIOD (0 when frozen)
        self._E0, self._dE = np.zeros((n, k, 2, 3)), np.zeros((n, k, 2, 3))
        self._Q0, self._dQ = np.zeros((n, k, 4)), np.zeros((n, k, 4))
        self._t0 = np.zeros(n)
        self._still = np.ones(n, dtype=bool)  # frozen or holding one sample: f = 0
        self.col_world = np.zeros((n, k, 2, 3))  # collider endpoints now (world)
        self.col_world_prev = np.zeros_like(self.col_world)
        sim.pre_render.append(self.write_bones)

    # ----- geometry -----
    def _static_clear(self, px, py):
        """Clearance from static obstacles at points (vectorized; -1 off the floor). Bilinear
        between raster nodes, so it changes smoothly as the cat moves; the distance field is
        1-Lipschitz, so a weighted mean of the nodes overestimates it by at most RASTER_SLACK,
        which is subtracted."""
        if np.ndim(px) == 0 and np.ndim(py) == 0:
            return kernels.clearance_at(self._grid, self._origin, RASTER, RASTER_SLACK, float(px), float(py))
        px, py = np.broadcast_arrays(np.asarray(px, dtype=float), np.asarray(py, dtype=float))
        out = np.empty(px.shape)
        kernels.clearance(self._grid, self._origin, RASTER, RASTER_SLACK, np.ascontiguousarray(px).ravel(),
                          np.ascontiguousarray(py).ravel(), out.reshape(-1))
        return out

    def circles(self, cat: Cat, x=None, y=None, yaw=None, pose: Sample | None = None) -> np.ndarray:
        """Envelope circles (n, 3: x, y, r) of the whole visible cat at a root pose (a sample's own
        circles are computed once and kept with it; callers only read them)."""
        s = pose or cat.next
        own = x is None and y is None and yaw is None
        if own and s.circ is not None:
            return s.circ
        x = s.x if x is None else x
        y = s.y if y is None else y
        yaw = s.yaw if yaw is None else yaw
        if s.local is None:
            s.local = s.pos[self._env_bone] + np.einsum("nij,nj->ni", s.rot[self._env_bone], self._env_off)
        out = np.empty((len(s.local), 3))
        kernels.place_circles(s.local, self._env_r, float(x), float(y), float(yaw), out)
        if own:
            s.circ = out
        return out

    def _robot(self):
        rx, ry, _ = self.sim.true_pose()
        return rx, ry

    def gaps(self, cat: Cat, x: float, y: float, yaw: float, pose: Sample | None = None,
             inflate=0.0, sweep: bool = False) -> tuple[float, float, float]:
        """(wall gap, robot gap, other-cat gap) of the whole visible cat at a candidate root pose,
        its envelope circles grown by `inflate` (per circle). With `sweep`, the other cats that
        already moved this sample are grown by half their own move too (see pose_ok)."""
        circ = self.circles(cat, x, y, yaw, pose)
        grow = np.broadcast_to(np.asarray(inflate, dtype=float), (len(circ),))
        rx, ry = self._robot()
        others = [o for o in self.cats if o is not cat and o.next is not None]
        oc = np.array([self.circles(o) for o in others]).reshape(len(others), len(circ), 3)
        og = np.zeros(oc.shape[:2])
        for k, o in enumerate(others):
            if sweep and o.t0 == self.time and o.prev is not o.next:
                og[k] = self._half_chord(o, oc[k], self.circles(o, pose=o.prev))
        rooms = self._rooms(cat, len(circ))
        if len(rooms):  # a seated cat's room to get up in counts as that cat
            oc, og = np.concatenate([oc, rooms]), np.concatenate([og, np.zeros(rooms.shape[:2])])
        out = np.empty(3)
        kernels.gaps(circ, np.ascontiguousarray(grow), self._grid, self._origin, RASTER, RASTER_SLACK, rx, ry,
                     C.CIRCUMSCRIBED_RADIUS, oc, og, np.ones(len(oc), dtype=np.bool_), out)
        return float(out[0]), float(out[1]), float(out[2])

    def _rooms(self, cat: Cat, n: int) -> np.ndarray:
        """The other cats' rooms to get up in (see _sit_check) as rows of n circles (rooms, n, 3),
        the last row padded with circles out of reach."""
        rooms = [o.room for o in self.cats if o is not cat and o.room is not None]
        if not rooms:
            return np.zeros((0, n, 3))
        allc = np.concatenate(rooms)
        pad = (-len(allc)) % n
        if pad:
            allc = np.concatenate([allc, np.tile([[1e9, 1e9, 0.0]], (pad, 1))])
        return allc.reshape(-1, n, 3)

    def eyes(self, cat: Cat) -> np.ndarray:
        """Both eyes' world positions now (the interpolated pose), for what the cat can see."""
        f = 0.0 if cat.next is cat.prev else min(max((self.time - cat.t0) / ANIM_PERIOD, 0.0), 1.0)
        a, b = cat.prev, cat.next
        local = a.pos[self._eye_bones] + (b.pos[self._eye_bones] - a.pos[self._eye_bones]) * f
        x, y = a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f
        yaw = a.yaw + math.remainder(b.yaw - a.yaw, 2 * math.pi) * f
        c, sn = math.cos(yaw), math.sin(yaw)
        return np.stack([x + c * local[:, 0] - sn * local[:, 1], y + sn * local[:, 0] + c * local[:, 1], local[:, 2]], 1)

    def current_gaps(self) -> list[dict]:
        """Every cat's gaps at this instant (evaluation): all cats at their interpolated poses at
        the same time, against walls and furniture, the robot where it is now (to its circle and to
        its footprint rectangle), and each other."""
        now = [self.circles(c, s.x, s.y, s.yaw, s) for c, s in ((c, self._current_sample(c)) for c in self.cats)]
        rx, ry, ryaw = self.sim.true_pose()
        cy, sy = math.cos(ryaw), math.sin(ryaw)
        out = []
        for i, ci in enumerate(now):
            r = ci[:, 2]
            wall = float((self._static_clear(ci[:, 0], ci[:, 1]) - r).min())
            robot = float((np.hypot(ci[:, 0] - rx, ci[:, 1] - ry) - r - C.CIRCUMSCRIBED_RADIUS).min())
            lx = (ci[:, 0] - rx) * cy + (ci[:, 1] - ry) * sy  # in the robot's frame
            ly = -(ci[:, 0] - rx) * sy + (ci[:, 1] - ry) * cy
            dx = np.maximum(np.abs(lx) - C.FOOTPRINT_HALF_LENGTH, 0.0)
            dy = np.maximum(np.abs(ly) - C.FOOTPRINT_HALF_WIDTH, 0.0)
            footprint = float((np.hypot(dx, dy) - r).min())
            other = math.inf
            for j, cj in enumerate(now):
                if j != i:
                    d = np.hypot(ci[:, None, 0] - cj[None, :, 0], ci[:, None, 1] - cj[None, :, 1])
                    other = min(other, float((d - r[:, None] - cj[None, :, 2]).min()))
            out.append({"wall": wall, "robot": robot, "footprint": footprint, "cat": other})
        return out

    @staticmethod
    def _half_chord(cat: Cat, end: np.ndarray, start: np.ndarray) -> np.ndarray:
        """Half of how far each envelope circle moves in a straight line from start to end, less
        the numerical tolerance (never below zero)."""
        out = np.empty(len(end))
        kernels.half_chord(end, start, SWEEP_TOL, out)
        return out

    def pose_ok(self, cat: Cat, x: float, y: float, yaw: float, margin: float = 0.0, pose: Sample | None = None,
                sweep_from: Sample | None = None) -> bool:
        """Every gap holds at the candidate pose with `margin` to spare. With `sweep_from` (the
        pose the cat moves from), the proof covers the whole move: the colliders and skin move
        each envelope circle in a straight line between the two samples, and a distance along a
        straight path dips below its smaller end by at most half the path, so each circle is
        grown by half its own move (other cats moving in the same sample likewise). A cat already
        inside a gap (the robot drove up to it, say) may still move, as long as no part comes
        closer than it is now; no gap that holds may be given up."""
        inflate = 0.0
        if sweep_from is not None:
            start = self.circles(cat, pose=sweep_from)
            inflate = self._half_chord(cat, self.circles(cat, x, y, yaw, pose), start)
        new = self.gaps(cat, x, y, yaw, pose, inflate, sweep=sweep_from is not None)
        if all(n >= limit + margin for n, limit in zip(new, (WALL_GAP, ROBOT_GAP, CAT_GAP))):
            return True  # every gap holds with room to spare (the usual case)
        now = None if sweep_from is None else self.gaps(cat, sweep_from.x, sweep_from.y, sweep_from.yaw, sweep_from)
        robot_limit = ROBOT_GAP if now is None else self._robot_limit(cat, now[1])
        for k, limit in enumerate((WALL_GAP, robot_limit, CAT_GAP)):
            if new[k] >= limit + margin:
                continue
            # inside the margin: only a cat that already was (something came close), never closer
            if now is None or now[k] >= limit + margin or new[k] < now[k] - CLOSER_TOLERANCE:
                return False
        return True

    def _landing_ok(self, cat: Cat, pose: Sample) -> bool:
        """A paw put down short (the root held): every gap stays above its limit less LAND_TOL, a
        fixed floor; a gap already below that floor may not shrink at all. The robot's limit is
        its comfort gap ROBOT_GAP here, always (never ROBOT_HARD: a landing is the cat's own move,
        so it must not lower the floor it is measured against)."""
        start = cat.prev
        inflate = self._half_chord(cat, self.circles(cat, pose.x, pose.y, pose.yaw, pose), self.circles(cat, pose=start))
        new = self.gaps(cat, pose.x, pose.y, pose.yaw, pose, inflate, sweep=True)
        now = self.gaps(cat, start.x, start.y, start.yaw, start)
        for k, limit in enumerate((WALL_GAP, ROBOT_GAP, CAT_GAP)):
            floor = limit - LAND_TOL
            if new[k] < floor and not (now[k] < floor and new[k] >= now[k]):
                return False
        return True

    # ----- episode -----
    def reset(self, start: tuple[float, float], goal: tuple[float, float]) -> None:
        """Respawn deterministically from cat_seed at free points away from the robot and goal."""
        self.rng = np.random.default_rng(self.seed)
        self.time = 0.0
        self.cats = []
        bind = Sample(0.0, 0.0, 0.0, self.rig.bind_pos.copy(), np.tile(np.eye(3), (len(self.rig.joint_names), 1, 1)))
        for i in range(self.n):
            cat = Cat(i, animator=CatAnimator(self.rig, seed=self.seed * 101 + i))
            for _ in range(5000):
                x, y = self.rng.uniform(-C.FLOOR_HALF_SIZE + 0.3, C.FLOOR_HALF_SIZE - 0.3, 2)
                yaw = float(self.rng.uniform(-math.pi, math.pi))
                if math.hypot(x - start[0], y - start[1]) < 1.5 or math.hypot(x - goal[0], y - goal[1]) < 1.0:
                    continue
                cat.next = Sample(float(x), float(y), yaw, bind.pos, bind.rot)
                if self.pose_ok(cat, float(x), float(y), yaw, margin=0.07):
                    break
            else:
                raise RuntimeError("no free spawn point for a cat")
            cat.x, cat.y, cat.yaw = cat.next.x, cat.next.y, cat.next.yaw
            cat.target_yaw = cat.yaw
            cat.animator.reset(cat.x, cat.y, cat.yaw)
            local, bob = cat.animator.update(ANIM_PERIOD, cat.x, cat.y, cat.yaw, 0.0, 0.0)
            pos, rot = cat.animator._fk(local)
            pos[:, 2] += bob
            pos[:, 0] += cat.animator.shift  # sitting: the body moves forward
            cat.next = self._finish(Sample(cat.x, cat.y, cat.yaw, pos, rot))
            cat.prev = cat.next
            cat.t0 = 0.0
            cat.plan_at = PLAN_PERIOD * i / self.n  # the cats plan in turn, not all in one frame
            self.cats.append(cat)
            self._load_blend(cat)
            self._enter(cat, "walk")
        self._interpolate(first=True)
        self.write_bones()

    # ----- behaviour (50 Hz) -----
    def _enter(self, cat: Cat, state: str) -> None:
        r = self.rng
        cat.state = state
        cat.no_sit, cat.sit_probe = False, None
        duration = {"walk": r.uniform(3.0, 8.0), "pause": r.uniform(1.0, 4.0), "sit": r.uniform(3.0, 8.0),
                    "dart": r.uniform(0.4, 0.8), "flee": 3.0, "travel": 60.0, "yield": 8.0}[state]
        if state != "sit":  # a sitting cat gets up first (the state's own time starts once it is up)
            duration += cat.animator.sit * SIT_UP_TIME
        cat.state_until = self.time + duration
        cat.target_v = {"walk": r.uniform(0.15, 0.3), "pause": 0.0, "sit": 0.0, "dart": 0.9, "flee": 0.35,
                        "travel": r.uniform(0.30, 0.50), "yield": YIELD_SPEED}[state]
        if state == "walk":
            cat.target_yaw = self._open_heading(cat)
        elif state == "dart":
            # a dart is a burst forward: the roomiest heading within 60 degrees of the present one
            options = cat.yaw + r.uniform(-math.pi / 3, math.pi / 3, 6)
            room = [float(self._static_clear(cat.x + 0.6 * math.cos(h), cat.y + 0.6 * math.sin(h))) for h in options]
            cat.target_yaw = float(options[int(np.argmax(room))])
        if state == "travel" and not self._new_route(cat):
            cat.state, cat.state_until = "walk", self.time + 3.0
            cat.target_yaw = self._open_heading(cat)
        cat.transitions.append((self.time, cat.state))

    def in_choke(self, cat: Cat) -> bool:
        """Any part of the cat within CHOKE_RADIUS of a doorway centre, or in the corridor (cats
        pass through these but never rest there: they would block the robot)."""
        circ = self.circles(cat)
        doors = np.asarray(C.DOORS)
        d = np.hypot(circ[:, None, 0] - doors[None, :, 0], circ[:, None, 1] - doors[None, :, 1])
        return bool((d < CHOKE_RADIUS).any()) or region_of(cat.x, cat.y) == "corridor"

    def _point_in_choke(self, x: float, y: float) -> bool:
        doors = np.asarray(C.DOORS)
        return bool((np.hypot(doors[:, 0] - x, doors[:, 1] - y) < CHOKE_RADIUS + 0.25).any()) or \
            region_of(x, y) == "corridor"

    def _rest(self, cat: Cat) -> None:
        """Sit or pause here, unless this is a doorway or the corridor: then carry on elsewhere."""
        if self.in_choke(cat):
            self._enter(cat, "travel")
        else:
            self._enter(cat, "sit" if self.rng.random() < 0.35 else "pause")

    def _in_robot_path(self, cat: Cat) -> tuple[bool, float]:
        """(the cat is in the robot's path ahead, distance ahead of its nearest part)."""
        rx, ry, ryaw = self.sim.true_pose()
        circ = self.circles(cat)
        c, sn = math.cos(ryaw), math.sin(ryaw)
        px = (circ[:, 0] - rx) * c + (circ[:, 1] - ry) * sn
        py = -(circ[:, 0] - rx) * sn + (circ[:, 1] - ry) * c
        inside = (px > -0.1) & (px < YIELD_DISTANCE) & (np.abs(py) < C.FOOTPRINT_HALF_WIDTH + YIELD_SIDE + circ[:, 2])
        return bool(inside.any()), float(px[inside].min()) if inside.any() else math.inf

    def _yield_spot(self, cat: Cat) -> bool:
        """Plan a route to a free spot out of the robot's path, out of doorways (a corridor will
        do), and away from the other cats' spots; the nearest by route length, a spot on the far
        side of the cat from the robot counting as nearer (a cat steps away from a robot coming at
        it, not past it). False if there is none."""
        rx, ry, ryaw = self.sim.true_pose()
        c, sn = math.cos(ryaw), math.sin(ryaw)
        taken = [o.destination for o in self.cats if o is not cat and o.destination is not None]
        taken += [(o.x, o.y) for o in self.cats if o is not cat]
        doors = np.asarray(C.DOORS)
        cat_ahead = (cat.x - rx) * c + (cat.y - ry) * sn
        best = None
        for radius in (0.7, 1.1, 1.6):
            for k in range(24):
                a = 2 * math.pi * k / 24
                gx, gy = cat.x + radius * math.cos(a), cat.y + radius * math.sin(a)
                if self._static_clear(gx, gy) < NAV_CLEAR + 0.05 or \
                        (np.hypot(doors[:, 0] - gx, doors[:, 1] - gy) < CHOKE_RADIUS + 0.25).any():
                    continue
                px = (gx - rx) * c + (gy - ry) * sn
                py = -(gx - rx) * sn + (gy - ry) * c
                if px > -0.4 and abs(py) < C.FOOTPRINT_HALF_WIDTH + YIELD_SIDE + 0.45:
                    continue  # still in the robot's way
                if any(math.hypot(gx - tx, gy - ty) < 0.5 for tx, ty in taken):
                    continue
                route = plan_route(self.sim.model, (cat.x, cat.y), (gx, gy), avoid=self._robot_keepout())
                if not route:
                    continue
                length = sum(math.hypot(b[0] - a_[0], b[1] - a_[1]) for a_, b in zip([(cat.x, cat.y)] + route, route))
                if px < cat_ahead:
                    length += YIELD_BACK_COST  # toward the robot's side: only if nothing better
                if best is None or length < best[0]:
                    best = (length, route, (gx, gy))
            if best is not None:
                break
        if best is None:
            return False
        cat.route, cat.destination = best[1], best[2]
        cat.progress_at, cat.progress_from = self.time, (cat.x, cat.y)
        return True

    def _plan(self, cat: Cat) -> None:
        """Local planner (dynamic-window style). Candidate commands (forward speed, turn rate, a
        slow side step) are rolled out over PLAN_HORIZON within the acceleration limits; the
        whole body (envelope circles, inflated) is checked at each step against static obstacles,
        the robot, and the other cats (both moving on at their present velocity). A candidate is
        feasible if it keeps every gap, or, for a cat already inside a gap's margin, never brings
        it closer. The best feasible one by progress toward the wanted heading, room to spare,
        and smoothness becomes the command until the next plan; none feasible means stop."""
        if cat.state in RESTING or cat.target_v <= 0.0:
            cat.cmd_v = cat.cmd_w = cat.cmd_lat = 0.0
            return
        tv = cat.target_v
        vs = np.array([tv, 0.66 * tv, 0.33 * tv, 0.0, -REVERSE_SPEED])
        ws = np.linspace(-YAW_RATE, YAW_RATE, 9)
        V, W = (a.ravel() for a in np.meshgrid(vs, ws, indexing="ij"))
        W = np.where(np.abs(V) < MIN_TURN_SPEED, np.clip(W, -PIVOT_RATE, PIVOT_RATE), W)
        V = np.concatenate([V, [0.0, 0.0]])
        W = np.concatenate([W, [0.0, 0.0]])
        L = np.zeros_like(V)
        L[-2:] = (SIDESTEP_SPEED, -SIDESTEP_SPEED)
        dt = PLAN_HORIZON / PLAN_STEPS
        times = dt * np.arange(1, PLAN_STEPS + 1)
        # Two rollouts per command: held for the whole horizon (where it leads: scoring), and held
        # for one plan period, then braking to a stop (what the cat commits to before it plans
        # again: feasibility). A move is allowed when the cat can make it and still stop clear.
        n = len(V)
        hold = np.concatenate([np.full(n, PLAN_STEPS), np.full(n, round(PLAN_PERIOD / dt))])
        Xb, Yb, Thb = self._rollout(cat, np.tile(V, 2), np.tile(W, 2), np.tile(L, 2), dt, hold)
        X, Y, Th = Xb[:n], Yb[:n], Thb[:n]
        both = self._plan_gaps(cat, Xb, Yb, Thb, times)
        worst_full, tight = both[:n].min(1), both[n:]  # (K,), (K, S): no cushion
        now0 = float(self._plan_gaps(cat, np.array([[cat.x]]), np.array([[cat.y]]), np.array([[cat.yaw]]),
                                     np.zeros(1))[0, 0])
        worst0 = tight.min(1)
        # commands the per-sample guard just refused (the plan's steps missed a closer pose)
        cat.rejected = {k: t for k, t in cat.rejected.items() if t > self.time}
        allowed = np.ones(len(V), dtype=bool)
        for rv_, rw_, rl_ in cat.rejected:
            allowed &= ~((np.abs(V - rv_) < 1e-9) & (np.abs(W - rw_) < 1e-9) & (np.abs(L - rl_) < 1e-9))
        idle = (V == 0.0) & (W == 0.0) & (L == 0.0)
        # The cushion keeps room for the legs, head, and tail: a cat never plans into it, and one
        # already inside (something came close) only plans moves that never come closer, scored to
        # win room back. A pocket it cannot leave this way is left by a worked-out escape.
        worst, now = worst0 - PLAN_CUSHION, now0 - PLAN_CUSHION
        feasible = worst >= 0.0
        if now < 0.0:
            feasible |= worst >= now - CLOSER_TOLERANCE
        feasible &= allowed
        if cat.idle_since is not None and self.time - cat.idle_since >= STUCK_AFTER and not (feasible & ~idle).any():
            # Stuck inside the cushion where every move comes a little closer (a corner): the
            # cushion is room to spare, not a limit, so a cat that has stood stuck may use it, down
            # to the MOVE_ROOM a root move keeps anyway (each sample is still proven as always).
            feasible |= allowed & ~idle & (worst0 >= MOVE_ROOM)
        if not feasible.any():
            # Nothing keeps every gap (a robot coming fast): make the most of it, taking the move that
            # keeps the most room, rather than freezing in place. Each sample is still proven safe.
            best_effort = allowed & ~idle
            if not best_effort.any():
                cat.cmd_v = cat.cmd_w = cat.cmd_lat = 0.0
                cat.idle_since = self.time if cat.idle_since is None else cat.idle_since
                return
            k = int(np.argmax(np.where(best_effort, worst0, -np.inf)))
            cat.cmd_v, cat.cmd_w, cat.cmd_lat = float(V[k]), float(W[k]), float(L[k])
            cat.idle_since = None
            return
        hx, hy = math.cos(cat.target_yaw), math.sin(cat.target_yaw)
        progress = ((X[:, -1] - cat.x) * hx + (Y[:, -1] - cat.y) * hy) / max(tv * PLAN_HORIZON, 0.05)
        align = np.cos(Th[:, -1] - cat.target_yaw)
        room = np.minimum(np.maximum(worst, 0.0), PLAN_ROOM) / PLAN_ROOM
        change = np.abs(V - cat.cmd_v) / V_MAX + 0.3 * np.abs(W - cat.cmd_w) / YAW_RATE
        side = 0.03 * (1.0 if cat.index % 2 == 0 else -1.0) * np.sign(W)  # each cat's own escape side
        # Turning around (the wanted heading is behind): keep turning the same way round. Both ways
        # score alike near 180 degrees, and swapping every plan would rock the cat on the spot.
        err = math.remainder(cat.target_yaw - cat.yaw, 2 * math.pi)
        same_way = (np.sign(W) == np.sign(cat.cmd_w)) & (W != 0.0)
        turning_around = abs(err) > math.pi / 2
        commit = PLAN_TURN_COMMIT * same_way * (turning_around and cat.cmd_w != 0.0)
        # A cat cannot turn round in a pocket: it first steps out (often backwards) into space wide
        # enough to turn. While turning around, plans that bring its middle into more open space
        # score better.
        if turning_around:
            here = min(float(self._static_clear(cat.x, cat.y)), PLAN_TURN_SPACE)
            there = np.minimum(self._static_clear(X[:, -1], Y[:, -1]), PLAN_TURN_SPACE)
            space = PLAN_SPACE_GAIN * (there - here) / PLAN_TURN_SPACE
        else:
            space = 0.0
        trouble = PLAN_TROUBLE * (worst_full < PLAN_CUSHION)  # held on, this would run into something
        # inside its cushion (a pocket, or something came close): first win room back
        escape = PLAN_ESCAPE * np.clip((worst - now) / 0.01, 0.0, 1.0) if now < 0.0 else 0.0
        score = (progress + 0.5 * align + PLAN_ROOM_WEIGHT * room - 0.4 * change + side
                 - 0.1 * np.abs(L) / SIDESTEP_SPEED - (0.1 if turning_around else 0.3) * (V < 0)
                 - PLAN_IDLE * idle + commit + space - trouble + escape)
        keep = (np.abs(V - cat.cmd_v) < 1e-9) & (np.abs(W - cat.cmd_w) < 1e-9) & (np.abs(L - cat.cmd_lat) < 1e-9)
        score = score + PLAN_KEEP * keep
        k = int(np.argmax(np.where(feasible, score, -np.inf)))
        cat.cmd_v, cat.cmd_w, cat.cmd_lat = float(V[k]), float(W[k]), float(L[k])
        if idle[k]:
            cat.idle_since = self.time if cat.idle_since is None else cat.idle_since  # nowhere to go
        else:
            cat.idle_since = None

    def _escape_search(self, cat: Cat) -> list:
        """A way out of a tight spot: best-first search over slow steps (2 cm forward, back, or to a
        side; 0.15 rad of pivot), each checked at four poses along it (the body pose as it is now,
        against walls, furniture, the robot, and the other cats where they are now), for a pose
        ESCAPE_DISTANCE from where it is stuck with ESCAPE_ROOM to spare (room for the planner to take
        over again). Returns the steps as (v, w, lat, end time) commands, or []."""
        import heapq
        steps = ((0.08, 0.0, 0.0), (-0.08, 0.0, 0.0), (0.0, 0.6, 0.0), (0.0, -0.6, 0.0), (0.0, 0.0, 0.08), (0.0, 0.0, -0.08))
        fractions = np.array([0.25, 0.5, 0.75, 1.0])

        def room(x, y, yaw):  # (n,) arrays of poses -> tightest margin beyond the gaps
            return self._plan_gaps(cat, x[:, None], y[:, None], yaw[:, None], np.zeros(1))[:, 0]

        start = (cat.x, cat.y, cat.yaw)
        here = float(room(*(np.array([v]) for v in start))[0])
        key = lambda p: (round(p[0] / 0.02), round(p[1] / 0.02), round(p[2] / 0.15))  # noqa: E731
        seen = {key(start)}
        heap = [(-here, 0, start, [])]
        count = 0
        while heap and count < ESCAPE_EXPANSIONS:
            neg, _, (x, y, yaw), path = heapq.heappop(heap)
            if -neg >= ESCAPE_ROOM and math.hypot(x - start[0], y - start[1]) >= ESCAPE_DISTANCE:
                return [st + (self.time + ESCAPE_STEP if i == 0 else 0.0,) for i, st in enumerate(path)]
            if len(path) >= ESCAPE_DEPTH:
                continue
            xs, ys, ts, ends = [], [], [], []
            for v, w, lat in steps:
                t = ESCAPE_STEP * fractions
                th = yaw + w * t
                mid = yaw + 0.5 * w * t  # heading half-way (arc), for the translation
                px = x + (v * np.cos(mid) - lat * np.sin(mid)) * t
                py = y + (v * np.sin(mid) + lat * np.cos(mid)) * t
                xs.append(px), ys.append(py), ts.append(th)
                ends.append((float(px[-1]), float(py[-1]), float(th[-1])))
            margins = room(np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)).reshape(len(steps), len(fractions))
            # as for any root move: MOVE_ROOM kept beyond every gap (inside it, never closer)
            floor = min(here, MOVE_ROOM) - CLOSER_TOLERANCE
            for (v, w, lat), end, m in zip(steps, ends, margins):
                if m.min() < floor or key(end) in seen:
                    continue
                seen.add(key(end))
                count += 1
                heapq.heappush(heap, (-float(m[-1]), count, end, path + [(v, w, lat)]))
        return []

    def _rollout(self, cat: Cat, V, W, L, dt: float, hold):
        """Root poses (K, PLAN_STEPS) for commands V, W, L held for `hold` steps of dt (per command),
        then braking to a stop (no side step), within the acceleration limits."""
        n = len(V)
        hold = np.ascontiguousarray(np.broadcast_to(hold, (n,)), dtype=np.int64)
        X, Y, Th = np.empty((n, PLAN_STEPS)), np.empty((n, PLAN_STEPS)), np.empty((n, PLAN_STEPS))
        kernels.rollout(float(cat.x), float(cat.y), float(cat.yaw), float(cat.v), float(cat.w),
                        *(np.ascontiguousarray(a, dtype=float) for a in (V, W, L)), hold, float(dt), _LIMITS, X, Y, Th)
        return X, Y, Th

    def _robot_limit(self, cat: Cat, gap: float | None = None) -> float:
        """The robot gap the cat keeps: ROBOT_GAP, its own comfort distance; but once the robot has
        driven inside that (the robot's doing, not the cat's), only ROBOT_HARD, so the cat is free
        to turn and get away rather than pinned by "never closer"; and the same for a seated cat
        getting up after the robot came near it. (A robot standing still may creep a little: it
        has driven in only once the gap is ROBOT_RELIEF inside, since a cat never moves itself
        inside ROBOT_GAP.)"""
        if cat.animator.sit > 0.0 and cat.animator.sit_target == 0.0 and \
                self._room_robot_gap(cat) < ROBOT_GAP + SIT_RISE:
            return ROBOT_HARD
        if gap is None:
            circ = self.circles(cat)
            rx, ry = self._robot()
            gap = float((np.hypot(circ[:, 0] - rx, circ[:, 1] - ry) - circ[:, 2] - C.CIRCUMSCRIBED_RADIUS).min())
        return ROBOT_GAP if gap >= ROBOT_GAP - ROBOT_RELIEF else ROBOT_HARD

    def _plan_gaps(self, cat: Cat, X, Y, Th, times) -> np.ndarray:
        """Tightest gap (the smallest margin over walls, robot, and cats, in metres beyond each
        limit) of the whole cat at root poses X, Y, Th (K, S) at future times (S)."""
        s = cat.next
        loc = s.pos[self._env_bone] + np.einsum("nij,nj->ni", s.rot[self._env_bone], self._env_off)
        rx, ry, ryaw = self.sim.true_pose()
        rv, _ = self.sim.true_velocity()
        prx = rx + rv * math.cos(ryaw) * times  # (S,)
        pry = ry + rv * math.sin(ryaw) * times
        reach = 2 * CAT_REACH + (V_MAX + REVERSE_SPEED) * float(times.max()) + CAT_GAP + 0.1
        near = [o for o in self.cats if o is not cat and o.next is not None and
                math.hypot(o.x - cat.x, o.y - cat.y) <= reach + abs(o.v) * float(times.max())]  # others are too far
        oc = np.concatenate([self.circles(o) for o in near]) if near else np.zeros((0, 3))
        ovx = np.concatenate([np.full(len(self._env_r), o.v * math.cos(o.yaw)) for o in near]) if near else np.zeros(0)
        ovy = np.concatenate([np.full(len(self._env_r), o.v * math.sin(o.yaw)) for o in near]) if near else np.zeros(0)
        rooms = [o.room for o in near if o.room is not None]  # a seated cat's room to get up in (still)
        if rooms:
            oc = np.concatenate([oc, *rooms])
            ovx, ovy = (np.concatenate([a, np.zeros(len(oc) - len(a))]) for a in (ovx, ovy))
        X, Y, Th = (np.ascontiguousarray(a, dtype=float) for a in (X, Y, Th))
        tight = np.empty(X.shape)
        kernels.plan_gaps(X, Y, Th, np.ascontiguousarray(loc[:, 0]), np.ascontiguousarray(loc[:, 1]), self._env_r,
                          self._grid, self._origin, RASTER, RASTER_SLACK, WALL_GAP, prx, pry,
                          C.CIRCUMSCRIBED_RADIUS + self._robot_limit(cat), np.ascontiguousarray(oc[:, 0]),
                          np.ascontiguousarray(oc[:, 1]), np.ascontiguousarray(oc[:, 2]), ovx, ovy,
                          np.ascontiguousarray(times, dtype=float), CAT_GAP, tight)
        return tight

    def _open_direction(self, cat: Cat) -> float:
        """The heading with the most room around the cat (16 directions, 0.3 m away): room from
        walls and furniture, and from the other cats and the robot."""
        rx, ry = self._robot()
        movers = [(o.x, o.y, 0.45) for o in self.cats if o is not cat]
        movers.append((rx, ry, C.CIRCUMSCRIBED_RADIUS + ROBOT_GAP + 0.2))
        best, best_clear = cat.yaw, -math.inf
        for k in range(16):
            h = k * math.pi / 8
            px, py = cat.x + 0.3 * math.cos(h), cat.y + 0.3 * math.sin(h)
            clear = float(self._static_clear(px, py))
            for mx, my, reach in movers:
                clear = min(clear, math.hypot(px - mx, py - my) - reach)
            if clear > best_clear:
                best, best_clear = h, clear
        return best

    def _robot_keepout(self) -> tuple[float, float, float]:
        """The disc around the robot that cat routes stay out of: its circle, the cat gap, a cat's
        half width, and a little room."""
        rx, ry = self._robot()
        return rx, ry, C.CIRCUMSCRIBED_RADIUS + ROBOT_GAP + ROUTE_ROBOT_ROOM

    def _robot_closing_speed(self, cat: Cat) -> float:
        """How fast the robot is coming toward the cat (m/s; negative when moving away)."""
        rx, ry, ryaw = self.sim.true_pose()
        v, _ = self.sim.true_velocity()
        dx, dy = cat.x - rx, cat.y - ry
        n = math.hypot(dx, dy)
        return 0.0 if n < 1e-6 else v * (math.cos(ryaw) * dx + math.sin(ryaw) * dy) / n

    def _new_route(self, cat: Cat, away_from: tuple[float, float] | None = None) -> bool:
        """Pick a destination in another region (or, fleeing, at least 1.5 m farther from the
        robot) and plan a route to it."""
        here = region_of(cat.x, cat.y)
        rooms = [name for name in C.ROOMS if name != here] if away_from is None else list(C.ROOMS)
        # Like a cat patrolling its territory, it prefers the rooms it has not been in for a while.
        weights = np.array([self.time - cat.last_in.get(name, -PATROL_MEMORY) + PATROL_BASE for name in rooms])
        weights = weights / weights.sum()
        for _ in range(20):
            name = rooms[int(self.rng.choice(len(rooms), p=weights))]
            x0, x1, y0, y1 = C.ROOMS[name]
            gx, gy = float(self.rng.uniform(x0 + 0.3, x1 - 0.3)), float(self.rng.uniform(y0 + 0.3, y1 - 0.3))
            if self._static_clear(gx, gy) < NAV_CLEAR + 0.1 or self._point_in_choke(gx, gy):
                continue
            if away_from is not None and math.hypot(gx - away_from[0], gy - away_from[1]) < \
                    math.hypot(cat.x - away_from[0], cat.y - away_from[1]) + 1.5:
                continue
            route = plan_route(self.sim.model, (cat.x, cat.y), (gx, gy), avoid=self._robot_keepout())
            if route:
                cat.route, cat.destination = route, (gx, gy)
                cat.progress_at, cat.progress_from = self.time, (cat.x, cat.y)
                return True
        return False

    def _follow_route(self, cat: Cat) -> None:
        """Head for the next waypoint; drop waypoints as they are reached."""
        while cat.route and math.hypot(cat.route[0][0] - cat.x, cat.route[0][1] - cat.y) < (
                ARRIVE if len(cat.route) == 1 else 0.25):
            cat.route.pop(0)
        if not cat.route:
            if cat.state in ("flee", "yield"):
                cat.state_until = self.time  # safe: the next tick picks what to do
            else:
                self._rest(cat)
            return
        wx, wy = cat.route[0]
        cat.target_yaw = math.atan2(wy - cat.y, wx - cat.x)
        if math.hypot(cat.x - cat.progress_from[0], cat.y - cat.progress_from[1]) > 0.15:
            cat.progress_at, cat.progress_from = self.time, (cat.x, cat.y)
        elif self.time - cat.progress_at > STALL_TIME:  # stuck (another cat, the robot): new route
            if not self._new_route(cat):
                self._enter(cat, "walk")

    def _should_yield(self, cat: Cat, robot_v: float, robot_blocked: bool) -> bool:
        """The cat is in the robot's path and the robot is coming (contact within YIELD_TTC) or
        waiting for it (a fresh forward request held back by safety). A parked robot does not
        herd cats."""
        in_path, ahead = self._in_robot_path(cat)
        if not in_path:
            return False
        if robot_blocked:
            return True
        return robot_v > 0.05 and ahead / robot_v < YIELD_TTC

    def _open_heading(self, cat: Cat, avoid: tuple[float, float] | None = None) -> float:
        """A heading with room ahead: random candidates scored by clearance 0.5 m ahead (and,
        when fleeing, by distance from `avoid`)."""
        best, best_score = cat.yaw, -math.inf
        for _ in range(12):
            h = cat.yaw + float(self.rng.uniform(-math.pi, math.pi))
            ax, ay = cat.x + 0.5 * math.cos(h), cat.y + 0.5 * math.sin(h)
            score = min(float(self._static_clear(ax, ay)), 0.6)
            if avoid is not None:
                score += 2.0 * math.hypot(ax - avoid[0], ay - avoid[1])
            if score > best_score:
                best, best_score = h, score
        return best

    def tick(self, robot_blocked: bool = False) -> None:
        """Behaviour decisions, called on every 50 Hz control tick. `robot_blocked`: the robot has
        a fresh forward request that its safety layer is holding back (it is waiting)."""
        rx, ry = self._robot()
        rv, _ = self.sim.true_velocity()
        r = self.rng
        for cat in self.cats:
            # Social reactions (watching, fleeing, giving way) need the robot in sight; walls and
            # furniture hide it. Keeping clear of it (planner, routes) does not.
            cat.sees_robot = math.hypot(rx - cat.x, ry - cat.y) < SIGHT_RANGE and \
                any(self.sim.sees_robot(eye) for eye in self.eyes(cat))  # either eye
            circ = self.circles(cat)
            robot_gap = float((np.hypot(circ[:, 0] - rx, circ[:, 1] - ry) - circ[:, 2] - C.CIRCUMSCRIBED_RADIUS).min())
            closing = self._robot_closing_speed(cat)
            scared = cat.sees_robot and (robot_gap < SCARE_DISTANCE or (robot_gap < FLEE_DISTANCE and closing > 0.05))
            in_path = cat.state not in ("flee", "yield") and not cat.frozen and cat.sees_robot and \
                self._should_yield(cat, rv, robot_blocked)
            if in_path or (scared and cat.state not in ("flee", "yield")):
                # In the robot's way (it is coming, or waiting for the cat): step out of its path to a
                # free spot; the robot waits and never pushes. Otherwise, scared: walk away from it.
                if (in_path or self._in_robot_path(cat)[0]) and self._yield_spot(cat):
                    self._enter(cat, "yield")  # keeps the route to the spot
                else:
                    self._enter(cat, "flee")
                    if not self._new_route(cat, away_from=(rx, ry)):
                        cat.target_yaw = self._open_heading(cat, avoid=(rx, ry))
            elif ((cat.blocked_since is not None and self.time - cat.blocked_since > UNSTICK_AFTER) or
                  (cat.idle_since is not None and self.time - cat.idle_since > UNSTICK_AFTER) or
                  (cat.still_from is not None and self.time - cat.still_from[0] > GIVE_WAY_AFTER)) and \
                    cat.state in ("travel", "flee", "walk", "yield"):
                # wedged, or getting nowhere (a pocket, two cats meeting in a narrow place): work
                # out a way into open space step by step and take it (or, failing that, give way
                # toward open space), then carry on
                cat.blocked_since = cat.idle_since = None
                cat.still_from = (self.time, cat.x, cat.y)
                cat.route = []
                cat.state, cat.state_until = "walk", self.time + 1.5
                cat.target_v, cat.target_yaw = 0.15, self._open_direction(cat)
                cat.escape = self._escape_search(cat)
                if cat.escape:
                    cat.state_until = self.time + ESCAPE_STEP * len(cat.escape) + 0.5
                    cat.escaped += 1
                else:  # no way out yet (others crowding it): look again soon, as they move
                    cat.still_from = (self.time - GIVE_WAY_AFTER + ESCAPE_RETRY, cat.x, cat.y)
                cat.transitions.append((self.time, "walk"))
            elif cat.state in ("flee", "yield") and cat.route and self.time < cat.state_until:
                if cat.state == "yield":  # the robot closing in fast: the cat hurries out of its way
                    cat.target_v = yield_speed(closing)
                self._follow_route(cat)
            elif cat.state == "travel" and self.time < cat.state_until:
                self._follow_route(cat)
            elif self.time >= cat.state_until:
                if cat.state in ("walk", "flee", "dart", "travel", "yield"):
                    roll = r.random()
                    if roll < 0.06:
                        self._enter(cat, "dart")
                    elif roll < 0.45:
                        self._rest(cat)
                    else:
                        self._enter(cat, "travel" if roll < 0.8 else "walk")
                elif cat.state == "pause":
                    roll = r.random()
                    if roll < 0.3 and not self.in_choke(cat):
                        self._enter(cat, "sit")
                    else:
                        self._enter(cat, "travel" if r.random() < 0.6 else "walk")
                else:
                    self._enter(cat, "travel" if r.random() < 0.6 else "walk")
            elif cat.state == "walk" and r.random() < 0.02:
                # gentle wandering: a small change of heading, toward where there is room to walk
                options = cat.yaw + r.normal(0.0, 0.6, 4)
                room = [float(self._static_clear(cat.x + 0.5 * math.cos(h), cat.y + 0.5 * math.sin(h))) for h in options]
                cat.target_yaw = float(options[int(np.argmax(room))])
            region = region_of(cat.x, cat.y)
            if region:
                cat.visited.add(region)
                cat.last_in[region] = self.time
            if cat.blocked:
                cat.blocked_since = self.time if cat.blocked_since is None else cat.blocked_since
            else:
                cat.blocked_since = None
            ref = cat.still_from
            if cat.state not in ("walk", "travel", "flee", "yield", "dart") or cat.frozen:
                cat.still_from = None
            elif ref is None or math.hypot(cat.x - ref[1], cat.y - ref[2]) > GIVE_WAY_MOVE:
                cat.still_from = (self.time, cat.x, cat.y)

    # ----- motion and animation (100 Hz samples, interpolated every physics step) -----
    def freeze(self, index: int, frozen: bool = True) -> None:
        """A cat touching the robot stops at once, holding the pose it has now (no snap), and stays
        still until the contact ends (a kinematic body must never push the robot)."""
        cat = self.cats[index]
        if frozen and not cat.frozen:
            held = self._finish(self._current_sample(cat))
            # the colliders held exactly as the physics has them now (rebuilding them from the
            # blended joints can differ by a few millimetres: a snap into whatever it touches)
            held.ends = self.col_world[index].copy()
            f = 0.0 if self._still[index] else min(max((self.time - self._t0[index]) / ANIM_PERIOD, 0.0), 1.0)
            q = self._Q0[index] + self._dQ[index] * f
            held.quat = q / np.linalg.norm(q, axis=1, keepdims=True)
            cat.prev = cat.next = held
            cat.x, cat.y, cat.yaw = held.x, held.y, held.yaw
            cat.v = cat.w = 0.0
        cat.frozen = frozen
        if not frozen:
            cat.t0 = self.time  # resume from the held pose
        self._load_blend(cat)

    def place(self, index: int, x: float, y: float, yaw: float, v: float = 0.0, state: str | None = None) -> None:
        """Put a cat at a pose (tests, tools, and fault injection), moving at v along its heading;
        its gait restarts there and the colliders and skin follow at once."""
        cat = self.cats[index]
        cat.x, cat.y, cat.yaw, cat.v, cat.w = float(x), float(y), float(yaw), float(v), 0.0
        cat.target_v, cat.target_yaw = float(v), float(yaw)
        cat.plan_at = self.time  # plans at once from here
        cat.sit_probe, cat.room = None, None
        if state is not None:
            cat.state, cat.state_until, cat.no_sit = state, math.inf, False
        cat.animator.reset(cat.x, cat.y, cat.yaw)
        local, bob = cat.animator.update(ANIM_PERIOD, cat.x, cat.y, cat.yaw, cat.v, 0.0)
        pos, rot = cat.animator._fk(local)
        pos[:, 2] += bob
        pos[:, 0] += cat.animator.shift  # sitting: the body moves forward
        cat.prev = cat.next = self._finish(Sample(cat.x, cat.y, cat.yaw, pos, rot))
        cat.t0 = self.time
        self._load_blend(cat)
        self._interpolate(first=True)
        self.write_bones()
        import mujoco
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def new_transitions(self) -> list[tuple[int, float, str]]:
        """State changes not yet reported: (cat index, time, new state)."""
        out = []
        for cat in self.cats:
            out += [(cat.index, t, st) for t, st in cat.transitions[cat.logged:]]
            cat.logged = len(cat.transitions)
        return out

    def _advance_sample(self, cat: Cat) -> None:
        """Compute the cat's pose one ANIM_PERIOD ahead: root motion within the limits (sub-stepped
        at the physics rate), a clearance check of the whole cat with a margin covering the
        interval, and the skeleton."""
        cat.prev, cat.t0 = cat.next, self.time
        cat.blocked = False
        dt = ANIM_PERIOD / ROOT_SUBSTEPS
        x, y, yaw, v, w = cat.x, cat.y, cat.yaw, cat.v, cat.w
        # Sitting: the cat sits down only once it is still with every paw planted, and only if the
        # whole sit-down fits here; it stays put (no plan, no command) until it is fully up again
        # (any new state, a flight or giving way included, first gets it up: at most SIT_UP_TIME)
        an = cat.animator
        if cat.state != "sit" or cat.no_sit:
            an.sit_target = 0.0
            cat.sit_probe = None
        elif an.sit_target > 0.0 and self._room_robot_gap(cat) < ROBOT_GAP + SIT_RISE:
            an.sit_target, cat.no_sit = 0.0, True  # the robot has come near its room: up while it can
        elif an.sit_target == 0.0 and abs(cat.v) < 0.01 and abs(cat.w) < 0.05 and \
                all(leg.planted for leg in an.legs.values()):
            verdict = self._sit_check(cat)
            if verdict is not None:
                cat.sit_probe = None
                an.sit_target = 1.0 if verdict else 0.0
                cat.no_sit = not verdict  # if it does not fit, it stays standing until its next state
        if an.sit == 0.0 and an.sit_target == 0.0:
            cat.room = None  # up: the room to get up in is free again
        an.calm = cat.sit_probe is not None
        resting = cat.state in RESTING or an.sit > 0.0
        if not resting and self.time >= cat.plan_at - 1e-9:
            self._plan(cat)
            cat.plan_at = self.time + PLAN_PERIOD
        if cat.escape and not resting:
            # a worked-out way out of a tight spot: each slow step for ESCAPE_STEP seconds
            cv, cw, cl = cat.escape[0][:3]
            if self.time + 1e-9 >= cat.escape[0][3]:
                cat.escape.pop(0)
                if cat.escape:
                    cat.escape[0] = cat.escape[0][:3] + (self.time + ESCAPE_STEP,)
        else:
            cv, cw, cl = (0.0, 0.0, 0.0) if resting else (cat.cmd_v, cat.cmd_w, cat.cmd_lat)
        for _ in range(ROOT_SUBSTEPS):
            v, w = _root_rates(v, w, cv, cw, dt)
            yaw += w * dt
            x += (v * math.cos(yaw) - cl * math.sin(yaw)) * dt
            y += (v * math.sin(yaw) + cl * math.cos(yaw)) * dt
        blocked_now = False
        if not self._clear_move(cat, x, y, yaw):
            # The hard guard: the plan assumed the present body pose; if this sample would come
            # too close after all, stay put, brake, and plan again at once (a kinematic body
            # cannot pass through anything).
            x, y, yaw = cat.x, cat.y, cat.yaw
            w = _brake(cat.w, YAW_ACCEL * ANIM_PERIOD)
            v, w = _hold_speed(cat.v, _brake(cat.v, ACCEL * ANIM_PERIOD), w)
            cat.blocked = blocked_now = True
            cat.plan_at = self.time
            cat.escape = []
            if cv or cw or cl:
                cat.rejected[(cv, cw, cl)] = self.time + PLAN_REJECT_TIME
        cat.x, cat.y, cat.yaw, cat.v, cat.w = x, y, math.atan2(math.sin(yaw), math.cos(yaw)), v, w
        rx, ry = self._robot()
        look = None
        g = self.gaps(cat, x, y, yaw)
        if cat.sees_robot and math.hypot(rx - x, ry - y) < 2.5 and g[1] > ROBOT_GAP + LOOK_ROOM and \
                not (cat.animator.calm or cat.animator.sit > 0.0 or cat.animator.sit_target > 0.0):
            look = (rx, ry)  # watches the robot (not when so close that turning its head would crowd it)
        snap = cat.animator.snapshot()
        lat = 0.0 if blocked_now else cl
        # the tail sways freely in the open and calms near walls, cats, and the robot
        room = min(g[0] - WALL_GAP, g[1] - ROBOT_GAP, g[2] - CAT_GAP)
        cat.animator.tail_target = min(max(room / TAIL_ROOM, 0.0), 1.0)
        # alert: the robot close and in sight, or the cat getting out of its way: tail up
        alert = cat.state in ("yield", "flee") or (cat.sees_robot and math.hypot(rx - x, ry - y) < ALERT_DISTANCE)
        cat.animator.tail_lift_target = TAIL_LIFT if alert else 0.0
        new = self._animate(cat, v, w, look, lat)
        if not self.pose_ok(cat, new.x, new.y, new.yaw, pose=new, sweep_from=cat.prev):
            # The new body pose (head, a leg, or the tail swinging) would come too close. Undo the
            # animation step and try a calm one (head straight, tail still, no new settling step),
            # then standing as it is (posture held); if even that is too
            # close, reject the whole sample: the cat keeps its previous root and pose (the
            # animator is restored, so its paws stay planted), brakes, and plans again without
            # that command.
            cat.animator.restore(snap)
            cat.animator.tail_target = 0.0
            if cat.animator.sit_target > 0.0:  # no room to sit here: stand up again, stay standing
                cat.no_sit, cat.animator.sit_target = True, 0.0
            new = self._animate(cat, v, w, None, lat, settle=False)
            if not self.pose_ok(cat, new.x, new.y, new.yaw, pose=new, sweep_from=cat.prev):
                # still too close: stand as it is (posture held; a paw in the air still lands),
                # and plan again without this command
                cat.animator.restore(snap)
                old = cat.prev
                cat.x, cat.y, cat.yaw = old.x, old.y, old.yaw
                new = self._animate(cat, 0.0, 0.0, None, 0.0, settle=False, freeze=True)
                cat.plan_at = self.time
                cat.escape = []
                if cv or cw or cl:
                    cat.rejected[(cv, cw, cl)] = self.time + PLAN_REJECT_TIME
            ok = self.pose_ok(cat, new.x, new.y, new.yaw, pose=new, sweep_from=cat.prev)
            # a paw in the air whose step would carry it too close (finishing a step is all that
            # standing still moves): withdraw the step (the paw back down where it lifted from),
            # or else put it down short, straight below where it is; the leg straightening as it
            # lands may take that one up to LAND_TOL into a gap (within the gaps' 1 mm
            # tolerance). Without these a cat could stand forever on three legs.
            for back in (True, False):
                if ok:
                    break
                cat.animator.restore(snap)
                if not cat.animator.put_paws_down(back):
                    break
                old = cat.prev
                cat.x, cat.y, cat.yaw = old.x, old.y, old.yaw
                new = self._animate(cat, 0.0, 0.0, None, 0.0, settle=False, freeze=True)
                cat.plan_at = self.time
                cat.escape = []
                if cv or cw or cl:
                    cat.rejected[(cv, cw, cl)] = self.time + PLAN_REJECT_TIME
                ok = (self.pose_ok(cat, new.x, new.y, new.yaw, pose=new, sweep_from=cat.prev) if back
                      else self._landing_ok(cat, new))
            if not ok:
                cat.animator.restore(snap)
                old = cat.prev
                cat.x, cat.y, cat.yaw = old.x, old.y, old.yaw
                cat.w = _brake(cat.w, YAW_ACCEL * ANIM_PERIOD)
                cat.v, cat.w = _hold_speed(cat.v, _brake(cat.v, ACCEL * ANIM_PERIOD), cat.w)
                cat.blocked = True
                cat.plan_at = self.time
                cat.held += 1
                cat.escape = []
                if cv or cw or cl:
                    cat.rejected[(cv, cw, cl)] = self.time + PLAN_REJECT_TIME
                new = Sample(old.x, old.y, old.yaw, old.pos, old.rot)
        cat.next = self._finish(new)
        self._load_blend(cat)

    def _sit_check(self, cat: Cat) -> bool | None:
        """Whether the whole sit-down from here fits, and getting up again: every 10 ms pose (paws
        shuffling under the body, sitting, then standing up; the root fixed) keeps the wall gap,
        and SIT_ROOM beyond the robot and cat gaps from where they are now, over each move, as
        pose_ok's sweep measures it (the tail is still throughout, so the cat gets up through the
        poses checked here). Worked out on a copy of the animator, SIT_CHECK_CHUNK poses per call so no
        frame stalls: None while still checking (begun again if a paw moves meanwhile). On a pass,
        the area those poses sweep (every pose from the first sitting one, each circle grown by
        its move to the next) becomes the cat's room (`room`) until it is up again: the other cats
        keep their gap from it as from the cat itself, so none can stand where it needs to get up;
        the robot does not, so a seated cat gets up once the robot comes within SIT_RISE of the
        robot gap from its room (and if the robot keeps coming, that get-up keeps only ROBOT_HARD
        from it: the robot's doing). Each sample is still checked as it comes."""
        an = cat.animator
        paws = np.array([leg.world for leg in an.legs.values()])
        if cat.sit_probe is None or not np.array_equal(cat.sit_probe[3], paws):
            probe = copy.copy(an)
            probe.legs = {name: copy.copy(leg) for name, leg in an.legs.items()}
            probe.restore(an.snapshot())  # its own copies of every changing value
            probe.sit_target = 1.0
            cat.sit_probe = [probe, cat.next, 0, paws, []]
        probe, prev, done, _, swept = cat.sit_probe
        for _ in range(SIT_CHECK_CHUNK):
            s = self._animate(cat, 0.0, 0.0, None, an=probe)
            circ = self.circles(cat, s.x, s.y, s.yaw, s)
            inflate = self._half_chord(cat, circ, self.circles(cat, pose=prev))
            g = self.gaps(cat, s.x, s.y, s.yaw, s, inflate, sweep=True)
            if g[0] < WALL_GAP or g[1] < ROBOT_GAP + SIT_ROOM or g[2] < CAT_GAP + SIT_ROOM:
                return False
            if probe.sit > 0.0 or probe.sit_target == 0.0:
                swept.append(circ + np.c_[np.zeros((len(circ), 2)), inflate])
            if probe.sit >= 1.0:
                probe.sit_target = 0.0  # sat: now getting up
            elif probe.sit == 0.0 and probe.sit_target == 0.0:
                return self._take_room(cat, _cover(np.array(swept)))
            prev, done = s, done + 1
            if done >= round(SIT_CHECK_TIME / ANIM_PERIOD):
                return False  # would not get down and up in time
        cat.sit_probe[1], cat.sit_probe[2] = prev, done
        return None

    def _take_room(self, cat: Cat, room: np.ndarray) -> bool:
        """Claim this room to sit and get up in, if the robot and every other cat (and its room)
        are SIT_RISE clear of it now (they moved while it was checked)."""
        n = len(self._env_r)
        others = [self.circles(o) for o in self.cats if o is not cat and o.next is not None]
        oc = np.concatenate([np.concatenate(others) if others else np.zeros((0, 3)),
                             self._rooms(cat, n).reshape(-1, 3)])
        if len(oc):
            d = np.hypot(room[:, None, 0] - oc[None, :, 0], room[:, None, 1] - oc[None, :, 1])
            if float((d - room[:, None, 2] - oc[None, :, 2]).min()) < CAT_GAP + SIT_RISE:
                return False
        cat.room = room
        if self._room_robot_gap(cat) < ROBOT_GAP + SIT_RISE:
            cat.room = None
            return False
        return True

    def _room_robot_gap(self, cat: Cat) -> float:
        """The robot gap of the cat's room to sit and get up in (inf if it has none)."""
        if cat.room is None:
            return math.inf
        rx, ry = self._robot()
        return float((np.hypot(cat.room[:, 0] - rx, cat.room[:, 1] - ry) - cat.room[:, 2]).min()
                     - C.CIRCUMSCRIBED_RADIUS)

    def _sit_fits(self, cat: Cat) -> bool:
        """_sit_check to the end (tests and tools)."""
        cat.sit_probe = None
        while (verdict := self._sit_check(cat)) is None:
            pass
        cat.sit_probe = None
        return verdict

    def _finish(self, s: Sample) -> Sample:
        """Fill a sample's world colliders (once per sample, not per physics step)."""
        k = len(self._col_a)
        s.ends, s.quat = np.empty((k, 2, 3)), np.empty((k, 4))
        kernels.finish_colliders(s.pos, s.rot, self._col_a, self._off_a, self._col_b, self._off_b,
                                 float(s.x), float(s.y), float(s.yaw), s.ends, s.quat)
        return s

    def _load_blend(self, cat: Cat) -> None:
        """Store the cat's blend between its two samples for the per-step interpolation."""
        i, a, b = cat.index, cat.prev, cat.next
        self._E0[i], self._dE[i] = a.ends, b.ends - a.ends
        qb = np.where((np.einsum("ij,ij->i", a.quat, b.quat) < 0)[:, None], -b.quat, b.quat)
        self._Q0[i], self._dQ[i] = a.quat, qb - a.quat
        self._t0[i] = cat.t0
        self._still[i] = cat.frozen or a is b

    def _animate(self, cat: Cat, v: float, w: float, look, lat: float = 0.0, settle: bool = True,
                 freeze: bool = False, an: CatAnimator | None = None) -> Sample:
        """One animation step at the cat's (new) root (with `an`, of that animator instead)."""
        an = cat.animator if an is None else an
        _, bob = an.update(ANIM_PERIOD, cat.x, cat.y, cat.yaw, v, w, look, lat, settle, freeze)
        pos, rot = an.pose  # the full pose of that update
        pos[:, 2] += bob
        pos[:, 0] += an.shift  # sitting: the body moves forward
        return Sample(cat.x, cat.y, cat.yaw, pos, rot)

    def _clear_move(self, cat: Cat, x: float, y: float, yaw: float) -> bool:
        """The root move alone (the present body pose carried to the new root) keeps the whole
        cat clear over the sample. A quick first check: the animated pose is then proven over
        the whole sample (pose_ok with sweep_from)."""
        return self.pose_ok(cat, x, y, yaw, margin=MOVE_ROOM, sweep_from=cat.next)

    def _current_sample(self, cat: Cat) -> Sample:
        """The interpolated pose now (root lerp, joint position lerp, rotation nlerp)."""
        f = 0.0 if cat.next is cat.prev else min(max((self.time - cat.t0) / ANIM_PERIOD, 0.0), 1.0)
        a, b = cat.prev, cat.next
        dyaw = math.remainder(b.yaw - a.yaw, 2 * math.pi)
        rot = a.rot + (b.rot - a.rot) * f
        u, _, vt = np.linalg.svd(rot)  # back to proper rotations (nlerp for matrices)
        rot = u @ vt
        return Sample(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.yaw + dyaw * f, a.pos + (b.pos - a.pos) * f, rot)

    def step(self, dt: float) -> None:
        """Every physics step: new 100 Hz samples when due, then the colliders at the
        interpolated pose."""
        self.time += dt
        for cat in self.cats:
            if cat.frozen:
                continue
            if self.time - cat.t0 >= ANIM_PERIOD - 1e-9:
                self._advance_sample(cat)
        self._interpolate()

    def _interpolate(self, first: bool = False) -> None:
        """Every physics step: blend every cat's colliders between its two samples (endpoints
        lerp, orientation normalized lerp along the shorter way) and write the mocap bodies."""
        if not self.cats:
            return
        self.col_world_prev[:] = self.col_world
        f = np.where(self._still, 0.0, np.clip((self.time - self._t0) / ANIM_PERIOD, 0.0, 1.0))
        d = self.sim.data
        kernels.blend_colliders(self._E0, self._dE, self._Q0, self._dQ, f, self.col_world, d.mocap_pos,
                                d.mocap_quat, self._col_mocap)
        if first:
            self.col_world_prev[:] = self.col_world

    def write_bones(self) -> None:
        """Pose the skinned meshes (bone bodies) at the same interpolated pose as the colliders;
        run before every render."""
        if not self.cats:
            return
        d = self.sim.data
        for i, cat in enumerate(self.cats):
            s = self._current_sample(cat)
            # A bone body's orientation is its posed rotation (relative to the bind pose) times its
            # bind rotation: the skin was bound to the bind orientation (the right eye's is turned
            # 180 degrees; without it that eyeball faces into the head).
            kernels.pose_bones(s.pos, s.rot, self.rig.bind_rot, float(s.x), float(s.y), float(s.yaw),
                               self._bone_mocap[i], d.mocap_pos, d.mocap_quat)

    def contact_velocity(self, index: int, point: np.ndarray) -> np.ndarray:
        """World velocity of the cat surface at a contact point: the motion of the nearest collider
        over the last physics step (final world transforms, so the root motion is included once)."""
        now, before = self.col_world[index], self.col_world_prev[index]
        a, b = now[:, 0], now[:, 1]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
        dist = np.linalg.norm(point - (a + t[:, None] * ab), axis=1) - self._radii
        k = int(np.argmin(dist))
        p_now = now[k, 0] + t[k] * (now[k, 1] - now[k, 0])
        p_before = before[k, 0] + t[k] * (before[k, 1] - before[k, 0])
        return (p_now - p_before) / C.PHYSICS_DT

    def truth(self) -> list[dict]:
        """Evaluation-only state of every cat (for logs)."""
        return [{"cat": c.index, "pose": [c.x, c.y, c.yaw], "v": c.v, "w": c.w, "pitch": c.pitch,
                 "state": c.state} for c in self.cats]
