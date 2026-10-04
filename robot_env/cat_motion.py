"""Procedural animation of the realistic cat: a four-beat walk (and a trot for fast moves) with
planted paws held still on the floor by inverse kinematics, body bob, spine bend in turns, a
rate-limited head look, tail sway, and breathing.

Everything is computed in the cat's own frame (x forward, y left, z up, paws on z = 0 at the
bind pose); all the rig's joints are aligned with that frame at the bind pose, so a joint's
local rotation is a rotation in its parent's frame. Deterministic: only simulated time, the
cat's motion, and its seed drive it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import kernels

from .cat_rig import Rig

# name: (shoulder blade or None, upper joint, lower joint, IK end joint, floor-contact joint,
#        walk phase offset). Hind legs plant the toe and place the ankle (hock) above it.
LEGS = {
    "LH": (None, "LeftUpLeg_02", "LeftLeg_03", "LeftFoot_04", "LeftToeBase_05", 0.00),
    "LF": ("LeftShoulder_014", "LeftArm_015", "LeftForeArm_016", "LeftHand_017", "LeftHand_017", 0.25),
    "RH": (None, "RightUpLeg_06", "RightLeg_07", "RightFoot_08", "RightToeBase_09", 0.50),
    "RF": ("RightShoulder_018", "RightArm_019", "RightForeArm_020", "RightHand_021", "RightHand_021", 0.75),
}
TROT_OFFSET = {"LH": 0.0, "RF": 0.0, "LF": 0.5, "RH": 0.5}  # diagonal pairs
WALK_DUTY, TROT_DUTY = 0.62, 0.45  # share of the cycle a paw is planted
TROT_SPEED = 0.55  # m/s: faster than this the cat trots
STEP_HEIGHT = 0.03  # m a swinging paw is lifted
MAX_REACH = 0.95  # share of a leg's length it may stretch before a planted paw must step
HEAD_RATE = 2.5  # rad/s the head may turn
SETTLE_DIST = 0.02  # m: standing, a paw further than this from its rest spot steps back to it
TAIL_LIFT_RATE = 2.0  # rad/s: how fast the tail rises or settles
TAIL_AMP_RATE = 0.1  # rad/s: how fast the tail's sway may grow or calm
BEND_RATE = 0.5  # rad/s: how fast the spine's bend into a turn may change
PIVOT_STEP = 0.012  # m: turning on the spot, a paw this far off its spot under the body steps


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


@dataclass
class Leg:
    name: str
    scapula: int | None
    upper: int
    lower: int
    paw: int  # end of the IK chain (wrist, or the hind ankle)
    contact: int  # the joint planted on the floor (front wrist, hind toe)
    offset: float
    neutral: np.ndarray  # contact joint at the bind pose (cat frame)
    planted: bool = True
    world: np.ndarray = field(default_factory=lambda: np.zeros(3))  # paw joint position (world)
    swing_from: np.ndarray = field(default_factory=lambda: np.zeros(3))
    swing_to: np.ndarray = field(default_factory=lambda: np.zeros(3))
    progress: float = 0.0  # swing progress 0..1
    duration: float = 0.25  # s for this swing
    lifted_cycle: int = -1  # gait cycle in which it last lifted (one step per cycle)


class CatAnimator:
    """Joint rotations (local, per joint) and the root height offset for one cat."""

    def __init__(self, rig: Rig, seed: int = 0):
        self.rig = rig
        self.rng = np.random.default_rng(seed)
        j = rig.joint
        self.legs = {name: Leg(name, j(s) if s else None, j(u), j(lo), j(p), j(c), off, rig.bind_pos[j(c)].copy())
                     for name, (s, u, lo, p, c, off) in LEGS.items()}
        self.spine = [j(n) for n in ("Spine_010", "Spine1_011", "Spine2_012", "Spine3_013")]
        self.neck = [j("Neck_022"), j("Neck1_023")]
        self.head = j("Head_024")
        self.tail = [j(n) for n in ("Tail_026", "Tail1_027", "Tail2_028", "Tail3_029", "Tail4_030")]
        self._order = self._parent_order()
        self._parents = np.ascontiguousarray(rig.parents, dtype=np.int64)
        self._identity = np.tile(np.eye(3), (len(rig.joint_names), 1, 1))
        self._spine_ids = np.array(self.spine, dtype=np.int64)
        self._tail_ids = np.array(self.tail, dtype=np.int64)
        self._posture_ids = [*self.spine, self.neck[0], self.head, *self.tail]
        self.half_stride_max = 0.9 * min(self._half_reach(leg) for leg in self.legs.values())
        self.cycles = 0.0
        self.time = 0.0
        self.head_yaw = 0.0
        self.bend = 0.0  # spine bend into turns (rad), eased at BEND_RATE
        self._started = False
        self._tail_phase = float(self.rng.uniform(0, 2 * math.pi))
        self._tail_rate = float(self.rng.uniform(0.25, 0.45))
        self.tail_target = 1.0  # 0..1: how freely the tail may sway (set by the herd: less near walls)
        self.tail_amp = 0.0  # rad: the sway now, eased toward its target (never snapping)
        self.tail_lift_target = 0.0  # rad: how high to raise the tail (set by the herd when alert)
        self.tail_lift = 0.0  # rad, eased toward the target

    # ----- geometry -----
    def _parent_order(self) -> list:
        rig, order, seen = self.rig, [], set()
        while len(order) < len(rig.joint_names):
            for k in range(len(rig.joint_names)):
                if k not in seen and (rig.parents[k] < 0 or rig.parents[k] in seen):
                    order.append(k)
                    seen.add(k)
        return order

    def _chain(self, leg: Leg):
        """(top joint of the swing, total length, vertical drop) of a leg at the bind pose (cached)."""
        cache = self.__dict__.setdefault("_chain_cache", {})
        if leg.name not in cache:
            cache[leg.name] = self._chain_uncached(leg)
        return cache[leg.name]

    def _chain_uncached(self, leg: Leg):
        rig = self.rig
        top = leg.scapula if leg.scapula is not None else leg.upper
        pts = [rig.bind_pos[k] for k in ([leg.scapula] if leg.scapula is not None else []) + [leg.upper, leg.lower, leg.paw]]
        length = sum(np.linalg.norm((b - a)[[0, 2]]) for a, b in zip(pts, pts[1:]))
        return top, length, rig.bind_pos[top][2] - rig.bind_pos[leg.paw][2]  # to the IK end joint

    def _half_reach(self, leg: Leg) -> float:
        _, length, drop = self._chain(leg)
        return math.sqrt(max((MAX_REACH * 0.97 * length) ** 2 - drop ** 2, 0.0))

    @staticmethod
    def _to_world(p, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2]])

    @staticmethod
    def _to_cat(p, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = p[0] - x, p[1] - y
        return np.array([c * dx + s * dy, -s * dx + c * dy, p[2]])

    def snapshot(self) -> tuple:
        """Everything update() changes, to undo a rejected step."""
        legs = tuple((leg.planted, leg.world.copy(), leg.swing_from.copy(), leg.swing_to.copy(), leg.progress,
                      leg.duration, leg.lifted_cycle) for leg in self.legs.values())
        return legs, self.cycles, self.time, self.head_yaw, self.tail_amp, self.bend, self.tail_lift

    def restore(self, snap: tuple) -> None:
        legs, self.cycles, self.time, self.head_yaw, self.tail_amp, self.bend, self.tail_lift = snap
        for leg, (planted, world, frm, to, progress, duration, lifted) in zip(self.legs.values(), legs):
            leg.planted, leg.progress, leg.duration, leg.lifted_cycle = planted, progress, duration, lifted
            leg.world, leg.swing_from, leg.swing_to = world.copy(), frm.copy(), to.copy()

    def reset(self, x: float, y: float, yaw: float) -> None:
        for leg in self.legs.values():
            leg.planted, leg.progress, leg.lifted_cycle = True, 0.0, -1
            leg.world = self._to_world(leg.neutral, x, y, yaw)
        self.cycles, self.head_yaw, self.bend, self.tail_amp, self.tail_lift = 0.0, 0.0, 0.0, 0.0, 0.0
        self._started = True

    # ----- per update -----
    def update(self, dt: float, x: float, y: float, yaw: float, v: float, w: float,
               look: tuple[float, float] | None = None, lat: float = 0.0,
               settle: bool = True, freeze: bool = False) -> tuple[dict, float]:
        """Advance by dt for a cat at (x, y, yaw) moving at v (forward, m/s), w (turn, rad/s), and
        lat (a side step to its left, m/s). settle=False: no new settling or pivot step starts (a
        paw already in the air still lands). freeze=True also holds the posture (head, spine,
        tail, breathing, gait phase) where it is. Returns ({joint: local rotation}, root height offset)."""
        if not self._started:
            self.reset(x, y, yaw)
        if not freeze:
            self.time += dt
        speed = abs(v)
        trot = speed > TROT_SPEED
        duty = TROT_DUTY if trot else WALK_DUTY
        moving = speed > 0.02 or abs(w) > 0.15 or abs(lat) > 0.02
        half = min(0.5 * (0.10 + 0.22 * speed), self.half_stride_max)
        freq = (speed / (2 * half) + 0.25 * abs(w)) if moving else 0.0
        if not freeze:
            self.cycles += freq * dt
        # body: a small bob per step, the spine bends into turns, breathing at the chest
        bob = 0.003 * math.sin(4 * math.pi * self.cycles) * min(speed / 0.3, 1.0)
        # the spine bends into turns, easing (never jumping when the turn rate changes)
        want_bend = float(clip(w * 0.10, -0.25, 0.25))
        if not freeze:
            self.bend += float(clip(want_bend - self.bend, -BEND_RATE * dt, BEND_RATE * dt))
        bend = self.bend
        breathe = 0.008 * math.sin(2 * math.pi * self.time / 2.6)  # about 23 breaths a minute
        # head: toward the look target, turning at most HEAD_RATE
        want = 0.0
        if look is not None:
            to = self._to_cat(np.array([look[0], look[1], 0.0]), x, y, yaw)
            want = float(clip(math.atan2(to[1], to[0] - 0.18), -1.0, 1.0))
        if not freeze:
            self.head_yaw += float(clip(want - self.head_yaw, -HEAD_RATE * dt, HEAD_RATE * dt))
        # tail: a slow wave running to the tip, larger when idle
        want_amp = (0.10 if moving else 0.18) * self.tail_target
        if not freeze:
            self.tail_amp += float(clip(want_amp - self.tail_amp, -TAIL_AMP_RATE * dt, TAIL_AMP_RATE * dt))
        amp = self.tail_amp
        # an alert cat (the robot close) raises its tail, eased; most of the lift at the base
        if not freeze:
            self.tail_lift += float(clip(self.tail_lift_target - self.tail_lift, -TAIL_LIFT_RATE * dt, TAIL_LIFT_RATE * dt))
        # spine, neck and head, tail (a wave running to the tip, most of the lift at the base):
        # kernels.posture, written into the local rotations (the dict holds views of them)
        L = self._identity.copy()
        kernels.posture(L, self._spine_ids, bend, breathe, self.neck[0], self.head, self.head_yaw, self._tail_ids,
                        self._tail_rate, self.time, self._tail_phase, amp, self.tail_lift)
        local: dict[int, np.ndarray] = {jj: L[jj] for jj in self._posture_ids}
        # legs (turning on the spot: paws step in diagonal pairs as soon as they drift a little)
        # turning or side-stepping on the spot: paws step in diagonal pairs
        pivot = speed <= 0.02 and (abs(w) > 0.15 or abs(lat) > 0.02)
        swinging = sum(not leg.planted for leg in self.legs.values())
        for leg in self.legs.values():
            if leg.planted:
                offset = TROT_OFFSET[leg.name] if trot else leg.offset
                pos_cycle = self.cycles + offset
                cycle, within = math.floor(pos_cycle), pos_cycle - math.floor(pos_cycle)
                lift = False
                if moving and not pivot and within >= duty and leg.lifted_cycle != cycle:
                    lift, leg.lifted_cycle = True, cycle
                    leg.duration = (1.0 - duty) / max(freq, 1e-6)
                elif self._reach(leg, x, y, yaw) > MAX_REACH:
                    lift, leg.duration = True, 0.18  # stretched too far: step now
                elif settle and ((pivot and self._pair_free(leg)) or (not moving and swinging == 0)):
                    drift = np.linalg.norm(leg.world[:2] - self._to_world(leg.neutral, x, y, yaw)[:2])
                    if drift > (PIVOT_STEP if pivot else SETTLE_DIST):
                        lift, leg.duration = True, 0.2 if pivot else 0.22  # step back under the body
                if lift:
                    leg.planted, leg.progress = False, 0.0
                    leg.swing_from = leg.world.copy()
                    leg.duration = float(clip(leg.duration, 0.12, 0.45))
                    swinging += 1
            if not leg.planted:
                leg.progress = min(1.0, leg.progress + dt / leg.duration)
                rest = (1.0 - leg.progress) * leg.duration  # time until it lands
                ahead = half * math.copysign(1.0, v) if moving and speed > 0.02 else 0.0
                gx = x + (v * math.cos(yaw) - lat * math.sin(yaw)) * rest  # where the body will be on landing
                gy = y + (v * math.sin(yaw) + lat * math.cos(yaw)) * rest
                side = 0.5 * PIVOT_STEP * math.copysign(1.0, lat) if pivot and abs(lat) > 0.02 else 0.0
                goal = self._to_world(leg.neutral + np.array([ahead, side, 0.0]), gx, gy, yaw + w * rest)
                if leg.progress <= dt / leg.duration + 1e-9:
                    leg.swing_to = goal  # a new step aims straight at its footfall
                else:  # a changing footfall (the cat speeds up, stops, or turns) is followed smoothly
                    leg.swing_to = leg.swing_to + (goal - leg.swing_to) * min(1.0, dt / 0.06)
                e = smoothstep(leg.progress)
                p = leg.swing_from + (leg.swing_to - leg.swing_from) * e
                p[2] = leg.neutral[2] + STEP_HEIGHT * math.sin(math.pi * leg.progress)
                leg.world = p
                if leg.progress >= 1.0:
                    leg.planted = True
                    leg.world = leg.swing_to.copy()
                    leg.world[2] = leg.neutral[2]
        pos, rot = self._fk_array(L)
        for leg in self.legs.values():
            contact = self._to_cat(leg.world, x, y, yaw) - [0.0, 0.0, bob]
            end = contact - (self.rig.bind_pos[leg.contact] - self.rig.bind_pos[leg.paw])  # foot kept level
            for k, m in self._leg_ik(leg, end, pos, rot).items():
                local[k] = L[k] = m
        # the legs' own joints and everything below them, now that their rotations are known (the
        # rest of the skeleton does not depend on them): the full pose, as _fk(local) would give
        self.pose = self._fk_array(L, pos, rot)
        return local, bob

    def _pair_free(self, leg: Leg) -> bool:
        """While pivoting, a paw may lift if every lifted paw is its diagonal partner."""
        partner = {"LH": "RF", "RF": "LH", "LF": "RH", "RH": "LF"}[leg.name]
        return all(o.planted or o.name == partner for o in self.legs.values() if o is not leg)

    def _reach(self, leg: Leg, x, y, yaw) -> float:
        top, length, _ = self._chain(leg)
        rig = self.rig
        end = self._to_cat(leg.world, x, y, yaw) - (rig.bind_pos[leg.contact] - rig.bind_pos[leg.paw])
        t = end - rig.bind_pos[top]
        sideways = t[1] - (rig.bind_pos[leg.paw][1] - rig.bind_pos[top][1])  # off the leg's own plane
        return float(math.sqrt(t[0] ** 2 + t[2] ** 2 + sideways ** 2) / length)

    def _fk(self, local: dict, pos: np.ndarray | None = None,
            rot: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Joint positions and rotations in the cat frame for the given local rotations; all
        joints are computed parents first (a compiled kernel, robot_env/kernels.py).
        Given the pose (pos, rot) from the same local rotations without the legs', only the legs'
        joints and their descendants are recomputed (the same result as the full pass)."""
        rig = self.rig
        if not hasattr(self, "_levels"):
            depth = {}
            for k in self._order:
                depth[k] = 0 if rig.parents[k] < 0 else depth[rig.parents[k]] + 1
            self._levels = [np.array([k for k in self._order if depth[k] == lv]) for lv in range(max(depth.values()) + 1)]
            self._offsets = rig.bind_pos - np.where(rig.parents[:, None] >= 0, rig.bind_pos[rig.parents], 0.0)
            below = set()
            for leg in self.legs.values():
                below |= {leg.upper} | ({leg.scapula} if leg.scapula is not None else set())
            for k in self._order:  # parents first: everything under a leg joint
                if rig.parents[k] in below:
                    below.add(k)
            self._fk_order = np.array(self._order, dtype=np.int64)
            self._leg_order = np.array([k for k in self._order if k in below], dtype=np.int64)
        return self._fk_array(self._local_array(local), pos, rot)

    def _local_array(self, local: dict) -> np.ndarray:
        """The local rotations as one (joints, 3, 3) array (identity where none is given)."""
        L = self._identity.copy()
        for k, m in local.items():
            L[k] = m
        return L

    def _fk_array(self, L: np.ndarray, pos: np.ndarray | None = None,
                  rot: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """_fk with the local rotations as an array."""
        if not hasattr(self, "_levels"):
            self._fk({})
        if pos is None:
            pos, rot, order = self.rig.bind_pos.copy(), L.copy(), self._fk_order
        else:
            pos, rot, order = pos.copy(), rot.copy(), self._leg_order
        kernels.fk(order, self._parents, self._offsets, L, pos, rot)
        return pos, rot

    def _leg_ik(self, leg: Leg, target: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> dict:
        """The leg IK (kernels.leg_ik; the method is described at _leg_ik_reference)."""
        out = np.empty((4, 3, 3))
        kernels.leg_ik(np.ascontiguousarray(target, dtype=float), pos, rot, self.rig.bind_pos, self._parents,
                       -1 if leg.scapula is None else leg.scapula, leg.upper, leg.lower, leg.paw, out)
        joints = {leg.upper: out[1], leg.lower: out[2], leg.paw: out[3]}
        if leg.scapula is not None:
            joints[leg.scapula] = out[0]
        return joints

    def _leg_ik_reference(self, leg: Leg, target: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> dict:
        """Place the paw joint at `target` (cat frame). Front legs first turn the shoulder blade so
        the elbow can keep its natural bend; then a sideways swing (about x) at the hip or
        shoulder and a two-bone solution in the leg's side plane (about y), folding the same way
        as at the bind pose. Solved in the parents' actual frames (spine bend, breathing)."""
        rig = self.rig
        out = {}
        pitch_total = 0.0
        if leg.scapula is not None:
            Rs_parent = rot[rig.parents[leg.scapula]]
            ts = Rs_parent.T @ (target - pos[leg.scapula])
            # turn the blade so the shoulder joint sits at the bind distance from the paw target
            # (the elbow then keeps its natural bend): intersect the circle the shoulder joint
            # moves on with the circle of that distance around the target (side plane)
            arm0 = (rig.bind_pos[leg.upper] - rig.bind_pos[leg.scapula])[[0, 2]]
            r = math.hypot(arm0[0], arm0[1])
            dist = float(np.linalg.norm((rig.bind_pos[leg.paw] - rig.bind_pos[leg.upper])[[0, 2]]))
            # side-plane target: the sideways offset (beyond the bind one) folds into the depth,
            # because the shoulder's sideways swing takes it up
            lat_s = rig.bind_pos[leg.paw][1] - rig.bind_pos[leg.scapula][1]
            t2s = np.array([ts[0], -math.hypot(ts[2], ts[1] - lat_s)])
            D = math.hypot(t2s[0], t2s[1])
            phi0 = math.atan2(arm0[1], arm0[0])
            if abs(r - dist) < D < r + dist:
                a_ = (r * r - dist * dist + D * D) / (2 * D)
                h = math.sqrt(max(r * r - a_ * a_, 0.0))
                u = t2s / D
                perp = np.array([-u[1], u[0]])
                cands = [u * a_ + perp * h, u * a_ - perp * h]
                pt = min(cands, key=lambda c: abs(math.remainder(math.atan2(c[1], c[0]) - phi0, 2 * math.pi)))
                phi = math.atan2(pt[1], pt[0])
            else:
                phi = math.atan2(t2s[1], t2s[0])  # out of reach: point the blade at the target
            scap = -math.remainder(phi - phi0, 2 * math.pi)
            out[leg.scapula] = rot_y(scap)
            pitch_total += scap
            Rp = Rs_parent @ out[leg.scapula]
            upper_pos = pos[leg.scapula] + Rp @ (rig.bind_pos[leg.upper] - rig.bind_pos[leg.scapula])
        else:
            Rp = rot[rig.parents[leg.upper]]
            upper_pos = pos[leg.upper]
        hip_b, knee_b, paw_b = rig.bind_pos[leg.upper], rig.bind_pos[leg.lower], rig.bind_pos[leg.paw]
        t = Rp.T @ (target - upper_pos)  # target from the hip/shoulder, parent frame (= bind axes)
        # sideways swing: the roll about x that brings the target to the bind lateral offset
        # (solve t_y cos r + t_z sin r = lateral0; take the smaller solution)
        lat0 = paw_b[1] - hip_b[1]
        rho = math.hypot(t[1], t[2])
        delta = math.atan2(t[2], t[1])
        spread = math.acos(float(clip(lat0 / max(rho, 1e-9), -1.0, 1.0)))
        roll = min((math.remainder(delta + spread, 2 * math.pi), math.remainder(delta - spread, 2 * math.pi)), key=abs)
        t = rot_x(roll).T @ t
        a0, b0 = (knee_b - hip_b)[[0, 2]], (paw_b - knee_b)[[0, 2]]
        la, lb = math.hypot(a0[0], a0[1]), math.hypot(b0[0], b0[1])
        t2 = t[[0, 2]]
        dn = math.hypot(t2[0], t2[1])
        d = clip(dn, abs(la - lb) + 1e-4, (la + lb) * 0.999)
        c0 = a0 + b0  # the joint folds to the side of the hip-to-paw line it is on at bind
        bend = 1.0 if math.remainder(math.atan2(a0[1], a0[0]) - math.atan2(c0[1], c0[0]), 2 * math.pi) > 0 else -1.0
        base = math.atan2(t2[1], t2[0])
        alpha = math.acos(clip((la * la + d * d - lb * lb) / (2 * la * d), -1.0, 1.0))
        ua = base + bend * alpha
        knee = np.array([math.cos(ua), math.sin(ua)]) * la
        ld = (t2 / max(dn, 1e-9)) * d - knee
        # rotation about y by angle a maps an x-z direction angle phi to phi - a
        up = -(ua - math.atan2(a0[1], a0[0]))
        low = -(math.atan2(ld[1], ld[0]) - math.atan2(b0[1], b0[0])) - up
        out[leg.upper] = rot_x(roll) @ rot_y(up)
        out[leg.lower] = rot_y(low)
        out[leg.paw] = rot_y(-(up + low + pitch_total)) @ rot_x(-roll)  # paw (and hind foot) kept level
        return out
