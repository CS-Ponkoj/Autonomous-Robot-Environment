"""RobotSystem: reset state, telemetry (requested / approved / applied), endpoint snapshot."""

import math

import pytest

from robot_env import config as C
from robot_env.types import Command
from tests.helpers import empty_system, hold


def applied_from_ctrl(s):
    """Motor target reconstructed from the actual MuJoCo actuator controls."""
    m = s.sim.model
    left = s.sim.data.ctrl[m.actuator("left_motor").id] * C.WHEEL_RADIUS
    right = s.sim.data.ctrl[m.actuator("right_motor").id] * C.WHEEL_RADIUS
    return (left + right) / 2, (right - left) / C.WHEEL_SEPARATION


def test_reset_clears_system_flags_but_keeps_operator_flags():
    s = empty_system()
    s.flags.episode_over = True
    s.flags.manual_mode = True
    s.flags.manual_input_held = False
    s.flags.emergency_brake = True
    s.flags.focus_lost = True
    s.reset(-2.0, 0.0, 0.0, (2.5, 2.5))
    assert not s.flags.episode_over and not s.flags.manual_mode
    assert s.flags.emergency_brake and s.flags.focus_lost
    s.flags.emergency_brake = s.flags.focus_lost = False
    hold(s, 0.3, 0.0, 1.0)
    assert s.sim.true_velocity()[0] > 0.2  # an env-style driver works right after reset


@pytest.mark.parametrize("sequence", [
    [(0.3, 0.0, 0.4)],                          # start-up
    [(0.0, 1.0, 0.4)],                          # turning
    [(0.3, 0.0, 1.0), (0.0, 0.0, 0.3)],         # braking
    [(0.3, 0.5, 1.0), (-0.15, -0.5, 0.6)],      # reversing while turning
])
def test_applied_telemetry_matches_motor_controls(sequence):
    s = empty_system()
    s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
    prev = Command(0.0, 0.0)
    for v, w, secs in sequence:
        for _ in range(int(round(secs / 0.02))):
            s.drive(v, w)
            s.advance(0.02)
            av, aw = applied_from_ctrl(s)
            assert abs(s.applied.v - av) < 1e-9 and abs(s.applied.omega - aw) < 1e-9
            # applied moves from its previous value toward what safety approved, never past it,
            # at no more than the smoother's rates (one control tick per advance here)
            ap = s.last_result.command
            for a, p0, target, up, down in ((s.applied.v, prev.v, ap.v, C.SMOOTH_ACCEL, C.SMOOTH_DECEL),
                                            (s.applied.omega, prev.omega, ap.omega, C.SMOOTH_ANG_ACCEL,
                                             C.SMOOTH_ANG_DECEL)):
                assert min(p0, target) - 1e-9 <= a <= max(p0, target) + 1e-9
                assert abs(a - p0) <= max(up, down) * C.CONTROL_PERIOD + 1e-9
            prev = s.applied


def test_startup_applied_differs_from_approved():
    s = empty_system()
    s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
    s.drive(0.3, 0.0)
    s.advance(C.PHYSICS_DT)
    assert s.last_result.command == Command(0.3, 0.0)  # approved
    assert s.applied.v == pytest.approx(C.SMOOTH_ACCEL * C.PHYSICS_DT)  # smoothed every physics step


def test_endpoint_observation_is_current():
    s = empty_system()
    s.reset(-1.5, 0.0, 0.0, (2.5, 0.0))
    hold(s, 0.3, 0.0, 1.0)
    obs = s.observe()
    assert obs.time == pytest.approx(s.time)
    assert obs.goal_distance == pytest.approx(s.sim.goal_sensor()[0], abs=1e-12)
    assert obs.scan_time <= obs.time and obs.time - obs.scan_time < C.CONTROL_PERIOD + 1e-9
    v_est = s.sim.velocity_estimate()
    assert obs.velocity_estimate == pytest.approx(v_est)


def test_endpoint_snapshot_keeps_stale_scan_honest():
    s = empty_system()
    s.reset(-1.5, 0.0, 0.0, (2.5, 0.0))
    s.advance(0.1)
    s.scan_enabled = False
    t_scan = s.observe().scan_time
    s.advance(0.5)
    assert s.observe().scan_time == t_scan  # not relabeled as fresh
    assert s.observe().time == pytest.approx(s.time)


def test_goal_reached_in_final_interval_is_detected():
    from robot_env.env import RobotGoalEnv
    import numpy as np
    env = RobotGoalEnv()
    env.reset(options={"task_seed": 1006})
    s = env.system
    # Place the robot 5 mm outside the goal radius, facing it: from rest (velocity
    # smoother) it covers about 1 cm in one 0.1 s step, so arrival happens inside the step.
    gx, gy = env.task.goal
    s.reset(gx - C.GOAL_RADIUS - 0.005, gy, 0.0, (gx, gy))
    env._prev_distance = s.observe().goal_distance
    _, _, terminated, _, info = env.step(np.array([0.3, 0.0], dtype=np.float32))
    assert terminated and info["is_success"]
    assert s.observe().goal_distance <= C.GOAL_RADIUS
    assert math.isclose(info["sim_time"], s.time)
    env.close()


@pytest.fixture
def cruise():
    """Factory: a system cruising at (v, w); every system made is closed after the test."""
    made = []

    def make(v=0.3, w=1.0):
        s = empty_system()
        made.append(s)
        s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
        for _ in range(50):
            s.drive(v, w)
            s.advance(C.CONTROL_PERIOD)
        assert s.applied == Command(v, w)
        return s

    yield make
    for s in made:
        s.close()


@pytest.mark.parametrize("trigger", ["emergency_brake", "focus_lost", "episode_over", "invalid_input",
                                     "command_expired", "stale_scan", "no_command"])
def test_safety_stops_bypass_the_smoother(cruise, trigger):
    """Comfort smoothing never delays a safety stop: zero is applied on the next control tick."""
    s = cruise()
    if trigger in ("emergency_brake", "focus_lost", "episode_over"):
        setattr(s.flags, trigger, True)
        s.drive(0.3, 1.0)
        s.advance(C.CONTROL_PERIOD)
    elif trigger == "invalid_input":
        s.drive(float("nan"), 1.0)
        s.advance(C.CONTROL_PERIOD)
    elif trigger == "no_command":
        s._requested = s._issued_at = None  # the request is gone (white-box: no public way mid-run)
        s.advance(C.CONTROL_PERIOD)
    elif trigger == "command_expired":
        while "command_expired" not in s.last_result.reasons:
            s.advance(C.CONTROL_PERIOD)
    else:
        s.scan_enabled = False
        while "stale_scan" not in s.last_result.reasons:
            s.drive(0.3, 1.0)
            s.advance(C.CONTROL_PERIOD)
    assert trigger in s.last_result.reasons
    assert s.applied == Command(0.0, 0.0)


def test_clearance_stop_bypasses_the_smoother(cruise):
    """A clearance STOP (for example, an obstacle appearing at speed) is applied at once;
    a clearance slow-down that is not a stop uses the ordinary ramp."""
    from robot_env.safety import SafetyResult
    s = cruise()
    s.safety.filter = lambda *args: SafetyResult(Command(0.1, 0.0), ("clearance",))
    s.drive(0.3, 1.0)
    s.advance(C.CONTROL_PERIOD)
    assert s.applied.v == pytest.approx(0.3 - C.SMOOTH_DECEL * C.CONTROL_PERIOD)
    s.safety.filter = lambda *args: SafetyResult(Command(0.0, 0.0), ("clearance",))
    s.drive(0.3, 1.0)
    s.advance(C.CONTROL_PERIOD)
    assert s.applied == Command(0.0, 0.0)


@pytest.mark.parametrize("v,w", [(0.5, 0.0), (0.3, 1.0), (0.0, -1.5), (-0.15, 0.5)])
def test_releasing_the_keys_ramps_down_within_bounds(cruise, v, w):
    """The manual driver letting go is the one stop that ramps (no stop rocking). It must
    still stop within the smoother's rates: bounded time and travel."""
    s = cruise(v, w)
    s.flags.manual_mode = True
    s.flags.manual_input_held = False
    travel = 0.0  # path length, not the endpoint chord (arcs)

    def tick():
        nonlocal travel
        for _ in range(round(C.CONTROL_PERIOD / C.PHYSICS_DT)):
            s.advance(C.PHYSICS_DT)
            travel += math.hypot(*s.sim.data.qvel[:2]) * C.PHYSICS_DT  # world speed, includes slip

    prev, ticks = s.applied, 0
    while s.applied != Command(0.0, 0.0):
        tick()
        ticks += 1
        assert s.last_result.reasons == ("released",)
        assert abs(prev.v) - abs(s.applied.v) <= C.SMOOTH_DECEL * C.CONTROL_PERIOD + 1e-9
        assert abs(prev.omega) - abs(s.applied.omega) <= C.SMOOTH_ANG_DECEL * C.CONTROL_PERIOD + 1e-9
        assert abs(s.applied.v) <= abs(prev.v) + 1e-12 and abs(s.applied.omega) <= abs(prev.omega) + 1e-12
        prev = s.applied
    ramp = max(abs(v) / C.SMOOTH_DECEL, abs(w) / C.SMOOTH_ANG_DECEL)
    assert ticks <= math.ceil(ramp / C.CONTROL_PERIOD - 1e-9) + 1
    for _ in range(25):
        tick()
    bound = v * v / (2 * C.SMOOTH_DECEL) + abs(v) * (C.CONTROL_PERIOD + C.SAFETY_DELAY) + 0.01
    assert travel <= bound


def test_reversing_ramps_through_zero(cruise):
    """Sampled at every physics step: each channel reaches exactly zero before changing sign."""
    s = cruise(0.3, 0.5)
    seen_zero_v = seen_zero_w = False
    s.drive(-0.15, -0.5)
    for i in range(int(0.8 / C.PHYSICS_DT)):
        if i % 10 == 0:
            s.drive(-0.15, -0.5)
        before = s.applied
        s.advance(C.PHYSICS_DT)
        a = s.applied
        assert before.v * a.v >= 0 and before.omega * a.omega >= 0  # never jumps across zero
        seen_zero_v |= a.v == 0.0
        seen_zero_w |= a.omega == 0.0
    assert seen_zero_v and seen_zero_w and s.applied == Command(-0.15, -0.5)
    s.close()


def test_applied_equals_motor_controls_at_every_physics_step():
    """Acceleration, a clearance reduction, manual release, reversal, and a hard stop: the
    reported applied command is what the actuators received, at every physics step."""
    from robot_env.safety import SafetyResult
    s = empty_system()
    s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
    real_filter = s.safety.filter

    def phase(seconds, command, held=True):
        s.flags.manual_input_held = held
        for i in range(int(round(seconds / C.PHYSICS_DT))):
            if i % 10 == 0 and command is not None:
                s.drive(*command)
            s.advance(C.PHYSICS_DT)
            av, aw = applied_from_ctrl(s)
            assert abs(s.applied.v - av) < 1e-9 and abs(s.applied.omega - aw) < 1e-9

    s.flags.manual_mode = True
    phase(0.4, (0.3, 1.0))                       # acceleration
    s.safety.filter = lambda *a: SafetyResult(Command(0.1, 0.2), ("clearance",))
    phase(0.2, (0.3, 1.0))                       # clearance reduction (ramps)
    s.safety.filter = real_filter
    phase(0.3, (0.3, 1.0))
    phase(0.3, None, held=False)                 # manual release (ramps)
    phase(0.4, (0.3, 1.0))
    phase(0.4, (-0.15, -1.0))                    # reversal
    s.flags.emergency_brake = True
    phase(0.1, (-0.15, -1.0))                    # hard stop
    braked = [e for e in s.safety_log if e.reasons == ("emergency_brake",)]
    assert braked and braked[-1].applied == Command(0.0, 0.0)  # zero at the event itself
    s.close()


def test_ordinary_safety_event_records_the_applied_command_before_smoothing(cruise):
    """An ordinary change (manual release) is logged at its control tick with the motor command
    that was running then; smoothing toward the new target only starts after the event."""
    s = cruise(0.3, 1.0)
    s.flags.manual_mode = True
    s.flags.manual_input_held = True
    for _ in range(5):
        s.drive(0.3, 1.0)
        s.advance(C.CONTROL_PERIOD)
    running = s.applied
    s.flags.manual_input_held = False
    s.advance(C.CONTROL_PERIOD)
    released = [e for e in s.safety_log if e.reasons == ("released",)]
    assert released and released[-1].applied == running and released[-1].approved == Command(0.0, 0.0)
    assert abs(s.applied.v) < abs(running.v) and abs(s.applied.omega) < abs(running.omega)


def _ramp(p, goal, up, down, t):
    """Independent closed form of the smoother over time t toward a fixed goal."""
    if p * goal < 0:  # slow to zero first, then speed up the other way
        t0 = abs(p) / down
        if t <= t0:
            return math.copysign(abs(p) - down * t, p)
        return math.copysign(min(abs(goal), up * (t - t0)), goal)
    if abs(goal) >= abs(p):
        return math.copysign(min(abs(goal), abs(p) + up * t), goal if goal else p)
    return math.copysign(max(abs(goal), abs(p) - down * t), p)


def test_smoother_rates_hold_for_irregular_advance_durations():
    """Irregular advance() calls with carried fractional steps: between observations the
    applied command changes no faster than the applicable rate times the elapsed sim time."""
    import random
    rng = random.Random(7)
    s = empty_system()
    s.reset(-1.5, 0.0, 0.0, (2.5, 2.5))
    commands = [(0.3, 1.0), (0.5, -0.5), (-0.15, 1.5), (0.0, 0.0), (0.4, 0.0), (0.0, -1.5)]
    total = 0.0
    command = commands[0]
    for k in range(400):
        if k % 40 == 0:
            command = commands[(k // 40) % len(commands)]
            block_start = s.time
        s.drive(*command)
        before, t0 = s.applied, s.time
        duration = rng.uniform(0.0001, 0.037)
        total += duration
        s.advance(duration)
        dt = s.time - t0
        target = s.last_result.command
        settled = t0 - block_start >= C.CONTROL_PERIOD and s.last_result.reasons == ()  # target fixed all interval
        for p, a, goal, up, down in ((before.v, s.applied.v, target.v, C.SMOOTH_ACCEL, C.SMOOTH_DECEL),
                                     (before.omega, s.applied.omega, target.omega, C.SMOOTH_ANG_ACCEL,
                                      C.SMOOTH_ANG_DECEL)):
            if p * a >= 0 and abs(a) <= abs(p):
                assert abs(p) - abs(a) <= down * dt + 1e-9
            elif p * a >= 0:
                assert abs(a) - abs(p) <= up * dt + 1e-9
            else:  # crossed zero: slowed to zero, then sped up the other way
                assert abs(p) / down + abs(a) / up <= dt + 1e-9
            if settled:  # exact progress: a frozen or slow smoother fails here
                assert a == pytest.approx(_ramp(p, goal, up, down, dt), abs=up * C.PHYSICS_DT + 1e-9)
    # carried fractions: exactly the whole physics steps that fit in the total time ran
    assert s.time == pytest.approx(int(total / C.PHYSICS_DT + 1e-9) * C.PHYSICS_DT, abs=1e-9)
    s.close()
