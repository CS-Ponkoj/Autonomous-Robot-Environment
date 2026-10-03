"""Per-level validation of the manual speed levels. Every test runs for every exposed level,
so exposing a new level in SPEED_LEVELS automatically puts it through the same checks."""

import math

import mujoco
import numpy as np
import pytest

from robot_env import config as C
from robot_env.layout import RoomMap
from robot_env.safety import SafetyResult
from robot_env.sim import RobotSim
from robot_env.system import RobotSystem
from robot_env.types import STOP, Command
from tests.helpers import door_edge_starts, empty_system, free_door_edge, loading_run, min_clearance_run

LEVELS = list(enumerate(C.SPEED_LEVELS))
IDS = [f"level{i + 1}" for i, _ in LEVELS]
per_level = pytest.mark.parametrize("level", LEVELS, ids=IDS)
POLE = '<geom name="thin_pole" type="cylinder" pos="1.2 {y} 0.3" size="0.012 0.3" rgba="1 0 1 1"/>'


def run(s, v, w, seconds, each=None):
    """Hold (v, w) for `seconds`, calling each(s) after every physics step."""
    for i in range(int(round(seconds / C.PHYSICS_DT))):
        if i % 10 == 0:
            s.drive(v, w)
        s.advance(C.PHYSICS_DT)
        if each:
            each(s)


def predictor_stop(v0):
    return abs(v0) * C.SAFETY_DELAY + v0 * v0 / (2 * C.SAFETY_DECEL)


def world_clearance(s, solids):
    """Smallest exact distance between any robot collider and the given world geoms."""
    m, d = s.sim.model, s.sim.data
    robot = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == s.sim.robot_body and m.geom_contype[g]]
    fromto = np.zeros(6)
    return min(mujoco.mj_geomDistance(m, d, r, g, 1.0, fromto) for r in robot for g in solids)


def track_walls(s):
    """World solids of the track split by the lidar plane: (crossing it, entirely below it).
    Below-plane parts (the 12 mm skirting boards) are invisible to the lidar by design (K2)."""
    m, d = s.sim.model, s.sim.data
    mujoco.mj_forward(m, d)
    floor = m.geom("floor").id
    lidar_z = d.site_xpos[s.sim._lidar_site][2]
    solids = [g for g in range(m.ngeom) if m.geom_contype[g] and g != floor
              and m.body_rootid[m.geom_bodyid[g]] != s.sim.robot_body]
    top = {g: d.geom_xpos[g][2] + m.geom_size[g][2] for g in solids}
    return [g for g in solids if top[g] >= lidar_z], [g for g in solids if top[g] < lidar_z]


# ----- 1. tracking -----

@per_level
def test_level_tracking(level):
    """Settled motion at each level: true linear speed within 5% (forward and reverse), pure
    turns within 2%, arc radius within 2%, applied exactly equal to approved, encoders within
    2% of applied, and the channel that should be zero stays at rest."""
    _, (v, w) = level
    for cv, cw, start in [(v, 0.0, (-4.0, 0.0, 0.0)), (-v * C.MANUAL_REVERSE_FACTOR, 0.0, (4.0, 0.0, 0.0)),
                          (0.0, w, (0.0, 0.0, 0.0)), (0.0, -w, (0.0, 0.0, 0.0)),
                          (v, w, (0.0, 0.0, 0.0)), (v, -w, (0.0, 0.0, 0.0))]:
        s = empty_system()
        s.reset(*start, (4.5, 4.5))
        ramp = max(abs(cv) / C.SMOOTH_ACCEL, abs(cw) / C.SMOOTH_ANG_ACCEL)
        run(s, cv, cw, ramp + 1.0)
        true_v, true_w, enc_v, enc_w = [], [], [], []

        def sample(s):
            assert s.last_result.reasons == () and s.applied == s.last_result.command == Command(cv, cw)
            tv, tw = s.sim.true_velocity()
            ev, ew = s.sim.velocity_estimate()
            true_v.append(tv)
            true_w.append(tw)
            enc_v.append(ev)
            enc_w.append(ew)

        run(s, cv, cw, 0.5, sample)
        tv, tw, ev, ew = (np.array(a) for a in (true_v, true_w, enc_v, enc_w))
        case = (cv, cw, "mean", tv.mean(), tw.mean())
        # Mean and worst sample both inside the agreed limits (measured worst error is about 0.25%).
        if cv and not cw:
            assert abs(tv.mean() - cv) <= 0.05 * abs(cv) and np.abs(tv - cv).max() <= 0.05 * abs(cv), case
            assert abs(tw.mean()) <= 0.02 and np.abs(tw).max() <= 0.05, case
        if cw and not cv:
            assert abs(tw.mean() - cw) <= 0.02 * abs(cw) and np.abs(tw - cw).max() <= 0.02 * abs(cw), case
            assert abs(tv.mean()) <= 0.01 and np.abs(tv).max() <= 0.02, case
        if cv and cw:
            radius = np.abs(tv / tw)
            assert np.abs(radius - abs(cv / cw)).max() <= 0.02 * abs(cv / cw), case
        for enc, cmd in ((ev, cv), (ew, cw)):
            if cmd:
                assert np.abs(enc - cmd).max() <= 0.02 * abs(cmd), (case, enc.mean())
        s.close()


# ----- 2. starts, stops, and tire loading -----

@per_level
def test_level_ordinary_starts_and_releases_keep_tires_loaded(level):
    _, (v, w) = level
    for cv, cw in [(v, 0.0), (-v * C.MANUAL_REVERSE_FACTOR, 0.0), (v, w), (v, -w)]:
        s = empty_system()
        s.reset(-1.5 if cv > 0 and not cw else (1.5 if cv < 0 else 0.0), 0.0, 0.0, (4.5, 4.5))
        r = loading_run(s, [(2.0, cv, cw, "drive"), (1.0, 0.0, 0.0, "release")])
        assert r["lost"] == 0 and r["ratio"] <= 1.5 and r["warnings"] == 0, (cv, cw, r)
        if not cw:
            assert max(r["shares"]) <= 0.02
        s.close()
    s = empty_system()
    s.reset(0.0, 0.0, 0.0, (4.5, 4.5))
    r = loading_run(s, [(2.0, 0.0, w, "drive")])
    assert max(r["shares"]) <= 0.15 and r["lost"] == 0
    s.close()


@per_level
def test_level_emergency_stop_loading_and_path(level):
    """Zero lost tire contact, no bounce, peak at most 3 times static, no solver warning, and a
    true braking path inside the predictor, straight and at both turning extremes."""
    _, (v, w) = level
    for cw in (0.0, w, -w):
        s = empty_system()
        s.reset(0.0 if cw else -4.0, 0.0, 0.0, (4.5, 4.5))
        r = loading_run(s, [(2.0, v, cw, "drive"), (1.0, v, cw, "ebrake")])
        assert r["lost"] == 0 and r["longest"] == 0.0 and r["ratio"] <= 3.0 and r["warnings"] == 0, (cw, r)
        assert r["path"] <= predictor_stop(r["v0"]), (cw, r["path"], predictor_stop(r["v0"]))
        s.close()


# ----- 5. stops from every level -----

HARD = ["emergency_brake", "focus_lost", "episode_over", "invalid_input", "no_command",
        "command_expired", "stale_scan", "clearance"]


@per_level
@pytest.mark.parametrize("reason", HARD)
def test_level_hard_stops(level, reason):
    """From cruising at the level: target, motor controls, and applied are zero on the first
    applicable control tick; the true path after that stays inside the predictor."""
    _, (v, _) = level
    s = empty_system()
    s.reset(-4.0, 0.0, 0.0, (4.5, 4.5))
    run(s, v, 0.0, v / C.SMOOTH_ACCEL + 0.6)
    # The trigger is the moment the stop condition becomes true; the path is measured from
    # there (including the wait for the next control tick) until the robot is at rest.
    keep_driving = True
    if reason in ("emergency_brake", "focus_lost", "episode_over"):
        setattr(s.flags, reason, True)
        trigger = s.time
    elif reason == "invalid_input":
        s.drive(float("nan"), 0.0)
        keep_driving, trigger = False, s.time
    elif reason == "no_command":
        s._requested = s._issued_at = None  # white-box: no public way to drop a request mid-run
        keep_driving, trigger = False, s.time
    elif reason == "command_expired":
        keep_driving, trigger = False, s._issued_at + C.COMMAND_LIFETIME
    elif reason == "stale_scan":
        s.scan_enabled = False
        trigger = s._scan_time + C.SCAN_MAX_AGE
    else:
        s.safety.filter = lambda *a: SafetyResult(STOP, ("clearance",))
        trigger = s.time
    v_trigger, path, applied_at = None, 0.0, None
    for i in range(int(2.5 / C.PHYSICS_DT)):
        if keep_driving and i % 10 == 0:
            s.drive(v, 0.0)
        if v_trigger is None and s.time >= trigger - 1e-9:
            v_trigger = math.hypot(*s.sim.data.qvel[:2])
        s.advance(C.PHYSICS_DT)
        if v_trigger is not None:
            path += math.hypot(*s.sim.data.qvel[:2]) * C.PHYSICS_DT
        if applied_at is None and reason in s.last_result.reasons:
            applied_at = s.time  # the first control tick that applies the stop:
            assert s.last_result.command == STOP and s.applied == STOP and not s.sim.data.ctrl.any()
    assert applied_at is not None and applied_at - trigger <= C.CONTROL_PERIOD + C.PHYSICS_DT + 1e-9
    assert path <= predictor_stop(v_trigger), (path, predictor_stop(v_trigger))
    s.close()


@per_level
def test_level_pivot_emergency_stop_angle(level):
    """A hard stop from a pivot at the level's turn rate stops within the predictor's angular model."""
    _, (_, w) = level
    s = empty_system()
    s.reset(0.0, 0.0, 0.0, (4.5, 4.5))
    run(s, 0.0, w, w / C.SMOOTH_ANG_ACCEL + 0.6)
    w0 = s.sim.true_velocity()[1]
    s.flags.emergency_brake = True
    turned = 0.0
    for _ in range(int(1.5 / C.PHYSICS_DT)):
        s.advance(C.PHYSICS_DT)
        turned += abs(s.sim.true_velocity()[1]) * C.PHYSICS_DT
    assert turned <= abs(w0) * C.SAFETY_DELAY + w0 * w0 / (2 * C.SAFETY_ANG_DECEL), turned
    s.close()


@per_level
def test_level_manual_release_is_a_bounded_smoothed_stop(level):
    _, (v, _) = level
    s = empty_system()
    s.reset(-4.0, 0.0, 0.0, (4.5, 4.5))
    s.flags.manual_mode = s.flags.manual_input_held = True
    run(s, v, 0.0, v / C.SMOOTH_ACCEL + 0.6)
    s.flags.manual_input_held = False
    path, t_zero = 0.0, None
    for i in range(int(1.5 / C.PHYSICS_DT)):
        s.advance(C.PHYSICS_DT)
        path += math.hypot(*s.sim.data.qvel[:2]) * C.PHYSICS_DT
        if t_zero is None and s.applied == STOP:
            t_zero = (i + 1) * C.PHYSICS_DT
    assert s.last_result.reasons == ("released",)
    assert t_zero <= v / C.SMOOTH_DECEL + C.CONTROL_PERIOD + 1e-9
    assert path <= v * v / (2 * C.SMOOTH_DECEL) + v * (C.CONTROL_PERIOD + C.SAFETY_DELAY) + 0.01
    s.close()


# ----- 3. thin obstacles -----

def _pole_run(v, w, start, off, phase=0.0):
    s = RobotSystem(RobotSim(include_obstacles=False, extra_world_xml=POLE.format(y=off)))
    x, y, yaw = start
    s.reset(x, y, yaw + math.radians(phase), (-4.5, -4.5))
    seconds = abs(1.2 - x) / max(abs(v), 0.05) + 3.0
    low = min_clearance_run(s, s.sim.model.geom("thin_pole").id, v, w, seconds)
    hit = s.collisions
    s.close()
    return low, hit


@per_level
def test_level_thin_pole_straight(level):
    _, (v, _) = level
    for off in np.round(np.arange(-0.10, 0.1001, 0.02), 3):
        for phase in (0.0, 0.5):
            low, hit = _pole_run(v, 0.0, (-1.0, 0.0, 0.0), off, phase)
            assert hit == 0 and C.SAFETY_BUFFER <= low < 0.10, (off, phase, low)


@per_level
def test_level_thin_pole_reverse_and_curves(level):
    _, (v, _) = level
    rv = -v * C.MANUAL_REVERSE_FACTOR
    for off in (-0.04, 0.0, 0.04):
        low, hit = _pole_run(rv, 0.0, (-1.0, 0.0, math.pi), off)
        assert hit == 0 and C.SAFETY_BUFFER <= low < 0.10, ("reverse", off, low)
    for sign in (1, -1):  # 2 m radius arcs toward the pole, both turn directions
        # The start offset that leads into the pole depends on the level (speed and turn rate ramp
        # at different rates), so try four: none may come closer than the buffer, and at least
        # one must really reach the pole.
        lows = []
        for y0 in (-0.30, -0.20, -0.13, -0.06):
            low, hit = _pole_run(v, sign * v / 2.0, (0.1, y0 * sign, 0.0), 0.0)
            assert hit == 0 and low >= C.SAFETY_BUFFER, ("arc", sign, y0, low)
            lows.append(low)
        assert min(lows) < 0.10, ("arc never reached the pole", sign, lows)


@pytest.fixture(scope="module")
def furnished():
    s = RobotSystem()
    mujoco.mj_forward(s.sim.model, s.sim.data)
    yield s, RoomMap(s.sim.model)
    s.close()


@per_level
def test_level_free_door_edges(level, furnished):
    _, (v, _) = level
    s, room = furnished
    m, d = s.sim.model, s.sim.data
    for door in ("door_office", "door_storage", "door_reception", "door_office_lab"):
        g = m.geom(door).id
        edge, out = free_door_edge(m, d, door)
        starts = list(door_edge_starts(room, edge, out))
        assert len(starts) == 3
        for off, start, yaw in starts:
            s.reset(float(start[0]), float(start[1]), yaw, (0.0, 0.0))
            low = min_clearance_run(s, g, v, 0.0, 1.0 / v + 3.0)
            assert s.collisions == 0 and C.SAFETY_BUFFER <= low < 0.10, (door, off, low)


# ----- 4. walls, corners, doorways, stress -----

@per_level
def test_level_wall_approaches(level):
    """Head-on, oblique, and reversing approaches reach the level's speed, then stop with the
    full buffer from the wall; an arc into a corner never touches."""
    _, (v, w) = level
    rv = -v * C.MANUAL_REVERSE_FACTOR
    for cv, cw, start in [(v, 0.0, (1.5, 0.0, 0.0)), (v, 0.0, (1.5, -1.0, 0.3)),
                          (rv, 0.0, (3.0, 0.0, math.pi)), (v, w / 3, (1.0, 1.5, 0.6))]:
        s = empty_system()
        s.reset(*start, (-4.5, -4.5))
        seconds = 12.0 if cw else 4.0 / abs(cv) + 2.0  # long enough to reach the east wall
        walls, below = track_walls(s)
        assert walls and below
        top, low, low_below = [0.0], [math.inf], [math.inf]

        def each(s):
            top[0] = max(top[0], abs(s.sim.true_velocity()[0]))
            if round(s.sim.time / C.PHYSICS_DT) % 10 == 0:
                low[0] = min(low[0], world_clearance(s, walls))
                low_below[0] = min(low_below[0], world_clearance(s, below))

        run(s, cv, cw, seconds, each)
        # full buffer from what the lidar can see; below-plane skirting: never touched
        assert s.collisions == 0 and low[0] >= C.SAFETY_BUFFER and low_below[0] > 0.0, (cv, cw, start, low[0], low_below[0])
        if not cw:
            assert top[0] >= 0.97 * abs(cv), (cv, top[0])  # the approach really reached the level
            assert low[0] < 0.10  # and really reached the wall
        s.close()


@per_level
def test_level_through_a_doorway(level):
    _, (v, _) = level
    s = RobotSystem()
    s.reset(-2.5, -0.3, math.pi / 2, (0.0, 0.0))  # corridor, facing the office doorway
    run(s, v, 0.0, 1.5 / v + 2.0)
    assert s.collisions == 0 and s.sim.true_pose()[1] > 1.0  # through the 0.9 m doorway
    s.close()


@per_level
def test_level_furnished_stress(level):
    """Seeded random poses near walls, jambs, and furniture with this level's commands."""
    i, (v, w) = level
    s = RobotSystem()
    room = RoomMap(s.sim.model)
    rng = np.random.default_rng(3000 + i)
    rv = -v * C.MANUAL_REVERSE_FACTOR
    commands = [(v, 0.0), (v, w), (v, -w), (v, w / 2), (v, -w / 2), (rv, 0.0), (rv, w), (0.0, w), (0.0, -w), (0.0, 0.0)]
    trials, failed = 40, []
    for t in range(trials):
        while True:
            x, y = rng.uniform(-4.85, 4.85, 2)
            if C.CIRCUMSCRIBED_RADIUS + 0.06 < room.point_clearance(x, y) < 0.8:
                break
        s.reset(x, y, rng.uniform(-math.pi, math.pi), (0.0, 0.0))
        for _ in range(int(rng.integers(4, 12))):
            cv, cw = commands[rng.integers(len(commands))]
            for _ in range(int(rng.integers(1, 8))):
                s.drive(cv, cw)
                s.advance(0.1)
        if s.collisions:
            failed.append((t, round(float(x), 3), round(float(y), 3)))
    assert not failed, f"{len(failed)} of {trials} trials collided: {failed}"
    s.close()


# ----- 8. transitions between levels -----

TRANSITIONS = [(a, a + 1) for a in range(len(C.SPEED_LEVELS) - 1)] + \
              [(a + 1, a) for a in range(len(C.SPEED_LEVELS) - 1)] + \
              [(C.DEFAULT_SPEED_LEVEL, len(C.SPEED_LEVELS) - 1), (len(C.SPEED_LEVELS) - 1, C.DEFAULT_SPEED_LEVEL)]


@pytest.mark.parametrize("a,b", sorted(set(TRANSITIONS)))
def test_level_transitions_are_continuous(a, b):
    """Driving at level a, then switching to level b (forward, then through zero into reverse):
    every physics step the applied command moves toward the target at no more than the
    applicable rate and never overshoots it."""
    s = empty_system()
    s.reset(-4.0, 0.0, 0.0, (4.5, 4.5))
    (va, wa), (vb, wb) = C.SPEED_LEVELS[a], C.SPEED_LEVELS[b]
    plan = [(1.5, va, wa / 4), (1.5, vb, wb / 4), (1.5, -vb * C.MANUAL_REVERSE_FACTOR, 0.0)]
    for seconds, cv, cw in plan:
        for i in range(int(round(seconds / C.PHYSICS_DT))):
            if i % 10 == 0:
                s.drive(cv, cw)
            before = s.applied
            s.advance(C.PHYSICS_DT)
            target, after = s.last_result.command, s.applied
            for p, q, goal, up, down in ((before.v, after.v, target.v, C.SMOOTH_ACCEL, C.SMOOTH_DECEL),
                                         (before.omega, after.omega, target.omega, C.SMOOTH_ANG_ACCEL,
                                          C.SMOOTH_ANG_DECEL)):
                dt = C.PHYSICS_DT
                if p * q >= 0 and abs(q) > abs(p):
                    assert abs(q) - abs(p) <= up * dt + 1e-9  # speeding up: the acceleration limit
                elif p * q >= 0:
                    assert abs(p) - abs(q) <= down * dt + 1e-9  # slowing down: the deceleration limit
                else:
                    assert abs(p) / down + abs(q) / up <= dt + 1e-9  # crossing zero: both parts
                assert min(p, goal) - 1e-12 <= q <= max(p, goal) + 1e-12  # no overshoot
    assert s.collisions == 0
    s.close()


def _predictor_ramp(target, accel, seconds):
    """Predictor travel from rest: delay at zero, then ramp at `accel` up to `target`."""
    t, c, x = 0.0, 0.0, 0.0
    while t < seconds - 1e-9:
        if t >= C.SAFETY_DELAY:
            c = min(abs(target), c + accel * 0.0005)
        x += c * 0.0005
        t += 0.0005
    return x


@per_level
def test_level_acceleration_within_the_predictor(level):
    """From rest, linear and angular travel never exceeds the predictor's ramp (delay, then
    SAFETY_ACCEL / SAFETY_ANG_ACCEL), straight, pivoting, and with the combined level command;
    the measured peak accelerations stay below the predictor's values."""
    _, (v, w) = level
    for cv, cw in [(v, 0.0), (0.0, w), (v, w)]:
        s = empty_system()
        s.reset(0.0 if cw else -4.0, 0.0, 0.0, (4.5, 4.5))
        seconds = max(cv / C.SMOOTH_ACCEL, cw / C.SMOOTH_ANG_ACCEL) + 0.3
        lin, ang = 0.0, 0.0
        peak_a, peak_alpha, window = 0.0, 0.0, []
        for i in range(int(round(seconds / C.PHYSICS_DT))):
            if i % 10 == 0:
                s.drive(cv, cw)
            s.advance(C.PHYSICS_DT)
            tv, tw = s.sim.true_velocity()
            lin += abs(tv) * C.PHYSICS_DT
            ang += abs(tw) * C.PHYSICS_DT
            window.append((tv, tw))
            if len(window) > 20:  # 10 ms averaging window for the acceleration peaks
                (v0, w0), (v1, w1) = window[-21], window[-1]
                peak_a = max(peak_a, abs(v1 - v0) / 0.01)
                peak_alpha = max(peak_alpha, abs(w1 - w0) / 0.01)
        if cv:
            assert lin <= _predictor_ramp(cv, C.SAFETY_ACCEL, seconds) + 0.005, (cv, cw, lin)
            assert peak_a <= C.SAFETY_ACCEL, (cv, cw, peak_a)
        if cw:
            assert ang <= _predictor_ramp(cw, C.SAFETY_ANG_ACCEL, seconds) + 0.005, (cv, cw, ang)
            assert peak_alpha <= C.SAFETY_ANG_ACCEL, (cv, cw, peak_alpha)
        s.close()
