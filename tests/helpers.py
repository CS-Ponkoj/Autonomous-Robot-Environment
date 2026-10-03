"""Shared test helpers. The waypoint follower is TEST-ONLY: it reads the true pose
and the planning map, so it shows the goal is reachable through the shared drive
and safety path. It is not evidence of autonomous perception or manual usability."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from robot_env import config as C
from robot_env.layout import Task
from robot_env.sim import RobotSim
from robot_env.system import RobotSystem
from robot_env.types import Command, Decision, Observation


def empty_system() -> RobotSystem:
    return RobotSystem(RobotSim(include_obstacles=False))


SWING_EPS = 0.002  # rad/s: samples smaller than this belong to no swing


def swing_lobes(samples, sign: float) -> tuple[float, list[float]]:
    """Back-swing analysis of yaw-rate samples after a turn is released.

    Samples are sign-normalised so the commanded direction is positive. Returns
    (worst_opposite, lobes): worst_opposite = max(0, -min(w)) over ALL samples; lobes = the
    peak of every sign run after the initial commanded-direction run, in order (opposite,
    rebound, opposite, ...). Samples with |w| < SWING_EPS belong to no run."""
    w = [x * sign for x in samples]
    worst = max([0.0] + [-x for x in w])
    runs: list[tuple[int, float]] = []
    run_sign, peak = 0, 0.0
    for x in w:
        if abs(x) < SWING_EPS:
            continue
        sx = 1 if x > 0 else -1
        if sx != run_sign:
            if run_sign:
                runs.append((run_sign, peak))
            run_sign, peak = sx, 0.0
        peak = max(peak, abs(x))
    if run_sign:
        runs.append((run_sign, peak))
    if runs and runs[0][0] == 1:
        runs = runs[1:]  # the initial decay in the commanded direction is not a swing
    return worst, [p for _, p in runs]


def hold(system: RobotSystem, v: float, w: float, seconds: float, period: float = C.DECISION_PERIOD) -> None:
    """Send the same command at the decision rate for `seconds`."""
    for _ in range(int(round(seconds / period))):
        system.drive(v, w)
        system.advance(period)


class WaypointFollower:
    """Test-only driver that follows the planned path using the true pose."""

    name = "test_waypoint_follower"

    def __init__(self, system: RobotSystem, task: Task):
        self.system = system
        self.points = list(task.path[1:])
        self.blocked_since: float | None = None
        self.backing_until = 0.0

    def command(self) -> tuple[float, float]:
        s = self.system
        if s.time < self.backing_until:
            return -0.15, 0.0
        res = s.last_result
        if "clearance" in res.reasons and abs(res.command.v) < 0.02 and abs(res.command.omega) < 0.05:
            self.blocked_since = s.time if self.blocked_since is None else self.blocked_since
            if s.time - self.blocked_since > 0.5:  # blocked: back up briefly, like a driver would
                self.blocked_since = None
                self.backing_until = s.time + 0.6
                return -0.15, 0.0
        else:
            self.blocked_since = None
        x, y, yaw = s.sim.true_pose()
        while len(self.points) > 1 and math.dist((x, y), self.points[0]) < 0.2:
            self.points.pop(0)
        tx, ty = self.points[0]
        err = math.atan2(ty - y, tx - x) - yaw
        err = math.atan2(math.sin(err), math.cos(err))
        if abs(err) > 0.5:
            return 0.0, math.copysign(1.0, err)
        return 0.3, max(-1.0, min(1.0, 2.5 * err))

    def decide(self, obs: Observation) -> Decision:
        """Same decide(observation) plug as every driver (this one also reads the true pose)."""
        v, w = self.command()
        return Decision(Command(v, w), obs.seq)


def tire_and_caster_forces(s):
    import numpy as np
    m, d = s.sim.model, s.sim.data
    tires = {m.geom("left_tire").id: 0, m.geom("right_tire").id: 1}
    casters = {m.geom("front_caster").id, m.geom("rear_caster").id}
    f6 = np.zeros(6)
    tire = [0.0, 0.0]
    caster = 0.0
    for c in range(d.ncon):
        for g in (d.contact[c].geom1, d.contact[c].geom2):
            if g in tires or g in casters:
                mujoco.mj_contactForce(m, d, c, f6)
                if g in tires:
                    tire[tires[g]] += f6[0]
                else:
                    caster += f6[0]
    return tire, caster


def loading_run(s, phases):
    """phases: (seconds, v, w, mode) with mode drive / release / ebrake. Returns per-tire static
    load, peak load, steps without contact, longest gap (s), the speed when braking starts, and
    the true path length while braking (world translational speed, so lateral slip counts)."""
    s.flags.manual_mode = s.flags.manual_input_held = True
    for _ in range(400):  # land and settle
        s.advance(C.PHYSICS_DT)
    static, _ = tire_and_caster_forces(s)
    peak, lost, gap, longest, path, shares, v0 = [0.0, 0.0], 0, [0.0, 0.0], 0.0, 0.0, [], None
    for seconds, v, w, mode in phases:
        if mode == "ebrake" and v0 is None:
            v0 = math.hypot(*s.sim.data.qvel[:2])
        s.flags.emergency_brake = mode == "ebrake"
        s.flags.manual_input_held = mode != "release"
        for i in range(int(round(seconds / C.PHYSICS_DT))):
            if i % 10 == 0:
                s.drive(v, w)
            s.advance(C.PHYSICS_DT)
            tire, caster = tire_and_caster_forces(s)
            for k in range(2):
                peak[k] = max(peak[k], tire[k])
                if tire[k] <= 1e-9:
                    lost += 1
                    gap[k] += C.PHYSICS_DT
                    longest = max(longest, gap[k])
                else:
                    gap[k] = 0.0
            if mode == "drive" and i * C.PHYSICS_DT > 0.8:
                shares.append(caster / max(sum(tire) + caster, 1e-9))
            if mode == "ebrake":
                path += math.hypot(*s.sim.data.qvel[:2]) * C.PHYSICS_DT
    ratio = max(p / st for p, st in zip(peak, static))
    return dict(ratio=ratio, lost=lost, longest=longest, path=path, shares=shares, v0=v0,
                warnings=sum(int(w.number) for w in s.sim.data.warning))


def min_clearance_run(s, target, v, w, seconds):
    """Drive and return the smallest exact distance between any robot collider and `target`."""
    m, d = s.sim.model, s.sim.data
    robot = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == s.sim.robot_body and m.geom_contype[g]]
    fromto = np.zeros(6)
    low = math.inf
    for i in range(int(round(seconds / C.PHYSICS_DT))):
        if i % 10 == 0:
            s.drive(v, w)
        s.advance(C.PHYSICS_DT)
        if i % 10 == 0:
            low = min(low, min(mujoco.mj_geomDistance(m, d, g, target, 1.0, fromto) for g in robot))
    return low


def free_door_edge(model, data, door: str):
    """The free (swinging) edge of an open door leaf and the outward unit direction along the
    leaf. The hinge edge is the one next to a door jamb."""
    g = model.geom(door).id
    size, rot = model.geom_size[g], data.geom_xmat[g].reshape(3, 3)
    axis = (rot[:, 0] if size[0] > size[1] else rot[:, 1])[:2]
    half = max(size[0], size[1])
    center = data.geom_xpos[g][:2].copy()
    jambs = [data.geom_xpos[j][:2] for j in range(model.ngeom) if "jamb" in (model.geom(j).name or "")]
    edge = max((center + k * axis * half for k in (1, -1)),
               key=lambda p: min(np.linalg.norm(p - j) for j in jambs))
    return edge, (edge - center) / half


def door_edge_starts(room, edge, out, offsets=(-0.04, 0.0, 0.04)):
    """Head-on starts toward a door edge: the farthest point (1.0 m down to 0.45 m) on the
    approach line where the robot fits, for each sideways offset; offsets without room are skipped."""
    side = np.array([-out[1], out[0]])
    for off in offsets:
        for dist in np.arange(1.0, 0.44, -0.05):
            start = edge + out * dist + side * off
            if room.point_clearance(*start) > C.CIRCUMSCRIBED_RADIUS + 0.03:
                yield off, start, math.atan2(-out[1], -out[0])
                break
