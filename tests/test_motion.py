"""Step 5: motion check on an empty track (true velocity is used for acceptance)."""

import math

import pytest

from robot_env import config as C
from robot_env.safety import predict_poses
from tests.helpers import empty_system, hold, loading_run


def settled_velocity(system, v, w, seconds=2.0):
    """Average true velocity over the last 0.5 s of a held command."""
    hold(system, v, w, seconds - 0.5)
    samples = []
    for _ in range(25):
        system.drive(v, w)
        system.advance(0.02)
        samples.append(system.sim.true_velocity())
    return (sum(s[0] for s in samples) / 25, sum(s[1] for s in samples) / 25)


def test_wheel_directions():
    s = empty_system()
    s.reset(-2.0, 0.0, 0.0, (2.5, 2.5))
    hold(s, 0.3, 0.0, 1.0)
    x, y, yaw = s.sim.true_pose()
    assert x > -1.8 and abs(y) < 0.01 and abs(yaw) < 0.02  # forward is +x
    s.reset(0.0, 0.0, 0.0, (2.5, 2.5))
    hold(s, 0.0, 1.0, 1.0)
    assert s.sim.true_pose()[2] > 0.8  # positive omega turns left (counterclockwise)


@pytest.mark.parametrize("v,w", [(0.3, 0.0), (0.5, 0.0), (-0.15, 0.0), (0.0, 1.0), (0.0, -1.0),
                                 (0.0, 1.5), (0.3, 0.5)])
def test_settled_speed_within_10_percent(v, w):
    s = empty_system()
    s.reset(2.0 if v < 0 else -2.0, 0.0, 0.0, (2.5, 2.5))
    tv, tw = settled_velocity(s, v, w)
    if v:
        assert abs(tv - v) <= 0.1 * abs(v), (tv, v)
    else:
        assert abs(tv) < 0.01
    if w:
        assert abs(tw - w) <= 0.1 * abs(w), (tw, w)
    else:
        assert abs(tw) < 0.02
    ev, ew = s.observe().velocity_estimate  # encoder estimate agrees when not slipping
    assert abs(ev - tv) < 0.03 and abs(ew - tw) < 0.1


@pytest.mark.parametrize("v,w", [(0.5, 0.0), (-0.15, 0.0), (0.0, 1.5)])
def test_stop_tolerance_within_1s(v, w):
    s = empty_system()
    s.reset(2.0 if v < 0 else -2.0, 0.0, 0.0, (2.5, 2.5))
    hold(s, v, w, 1.5)
    hold(s, 0.0, 0.0, 1.0)
    tv, tw = s.sim.true_velocity()
    assert abs(tv) < 0.02 and abs(tw) < 0.05


@pytest.mark.parametrize("v", [0.3, 0.5, -0.15])
def test_braking_distance_within_safety_model(v):
    """Measured stopping travel must not exceed what the safety predictor assumes."""
    s = empty_system()
    s.reset(2.0 if v < 0 else -2.0, 0.0, 0.0, (2.5, 2.5))
    hold(s, v, 0.0, 1.5)
    v0 = s.sim.true_velocity()[0]
    x0 = s.sim.true_pose()[0]
    hold(s, 0.0, 0.0, 1.0)
    measured = abs(s.sim.true_pose()[0] - x0)
    predicted = abs(predict_poses(v0, 0.0, 0.0, 0.0)[-1, 0])
    assert measured <= predicted, (measured, predicted)


@pytest.mark.parametrize("w", [1.0, 1.5])
def test_turn_braking_within_safety_model(w):
    s = empty_system()
    s.reset(0.0, 0.0, 0.0, (2.5, 2.5))
    hold(s, 0.0, w, 1.5)
    w0 = s.sim.true_velocity()[1]
    yaw0 = s.sim.true_pose()[2]
    hold(s, 0.0, 0.0, 1.0)
    turned = abs(math.atan2(math.sin(s.sim.true_pose()[2] - yaw0), math.cos(s.sim.true_pose()[2] - yaw0)))
    predicted = abs(predict_poses(0.0, w0, 0.0, 0.0)[-1, 2])
    assert turned <= predicted, (turned, predicted)


@pytest.mark.parametrize("v", [0.3, 0.5])
def test_acceleration_not_faster_than_safety_model(v):
    """Travel in the first 0.3 s from rest must not exceed the predictor's ramp."""
    s = empty_system()
    s.reset(-2.0, 0.0, 0.0, (2.5, 2.5))
    x0 = s.sim.true_pose()[0]
    hold(s, v, 0.0, 0.3, period=0.02)
    measured = s.sim.true_pose()[0] - x0
    # predictor travel over the same 0.3 s (delay, then ramp at SAFETY_ACCEL)
    t, cv, x = 0.0, 0.0, 0.0
    while t < 0.3 - 1e-9:
        if t >= C.SAFETY_DELAY:
            cv = min(v, cv + C.SAFETY_ACCEL * 0.001)
        x += cv * 0.001
        t += 0.001
    assert measured <= x + 0.01, (measured, x)


HEADINGS = [math.radians(d) for d in range(-180, 180, 15)]  # every 15 degrees, includes mirrors


@pytest.mark.parametrize("v", [0.3, -0.15, 0.5])
def test_straight_start_stop_has_no_heading_twitch(v):
    """Regression (owner report: holding a key jittered). With the old pyramidal friction
    cone, start/stop grip depended on heading: yaw spikes up to 1.1 rad/s and 5.7 degrees
    of drift. At every heading, forward and reverse, with repeated starts and stops, the
    robot must stay on its line."""
    s = empty_system()
    worst_rate = worst_heading = worst_lateral = 0.0
    for h in HEADINGS:
        s.reset(0.0, 0.0, h, (4.0, 4.0))
        for _ in range(3):  # repeated start/stop cycles
            hold(s, v, 0.0, 1.0, period=0.02)
            hold(s, 0.0, 0.0, 0.4, period=0.02)
            worst_rate = max(worst_rate, abs(s.sim.true_velocity()[1]))
        x, y, yaw = s.sim.true_pose()
        dheading = math.degrees(math.atan2(math.sin(yaw - h), math.cos(yaw - h)))
        lateral = -math.sin(h) * x + math.cos(h) * y  # sideways offset from the start line
        worst_heading = max(worst_heading, abs(dheading))
        worst_lateral = max(worst_lateral, abs(lateral))
    assert worst_heading <= 0.2, worst_heading
    assert worst_lateral <= 0.005, worst_lateral


def test_peak_yaw_rate_during_starts_and_stops_is_tiny():
    s = empty_system()
    peak = 0.0
    for h in HEADINGS:
        s.reset(0.0, 0.0, h, (4.0, 4.0))
        for v, secs in ((0.3, 0.6), (0.0, 0.3), (0.5, 0.6), (0.0, 0.3)):
            for _ in range(int(round(secs / 0.02))):
                s.drive(v, 0.0)
                s.advance(0.02)
                peak = max(peak, abs(s.sim.true_velocity()[1]))
    assert peak <= 0.05, peak


# ----- stop rock (back-swing after releasing a turn) -----

def test_swing_lobes_analyzer():
    from tests.helpers import swing_lobes
    # starts in the commanded direction: that decay is not a swing; every later lobe is kept
    worst, lobes = swing_lobes([1.0, 0.5, 0.1, -0.3, -0.1, 0.05, -0.02], 1)
    assert worst == pytest.approx(0.3) and lobes == pytest.approx([0.3, 0.05, 0.02])
    # starts opposite: the first lobe is kept (regression: an earlier analysis dropped it)
    worst, lobes = swing_lobes([-0.1, 0.2], 1)
    assert worst == pytest.approx(0.1) and lobes == pytest.approx([0.1, 0.2])
    # negative command; a neutral gap (below the threshold) does not split a lobe
    worst, lobes = swing_lobes([-1.0, -0.2, 0.4, 0.0, 0.001, 0.1], -1)
    assert worst == pytest.approx(0.4) and lobes == pytest.approx([0.4])
    # no crossing: nothing opposite
    assert swing_lobes([0.5, 0.2, 0.0], 1) == (0.0, [])
    # threshold-sized samples: below the threshold they form no lobe, but still count as worst
    worst, lobes = swing_lobes([0.5, -0.0019, 0.0], 1)
    assert worst == pytest.approx(0.0019) and lobes == []
    worst, lobes = swing_lobes([0.5, -0.002, 0.0], 1)
    assert lobes == pytest.approx([0.002])
    # growing later lobes are reported in order, so a growth check can see them
    _, lobes = swing_lobes([1.0, -0.1, 0.2, -0.3], 1)
    assert lobes == pytest.approx([0.1, 0.2, 0.3])


def _release_turn(s, sign):
    """Release a settled pivot (the manual driver lets go). Returns yaw-rate samples at every
    physics step, the yaw gained, and the yaw the motor commands asked for."""
    from robot_env.sim import yaw_from_quat
    for i in range(int(1.0 / C.PHYSICS_DT)):
        if i % 10 == 0:
            s.drive(0.0, sign * 1.0)
        s.advance(C.PHYSICS_DT)
    y0 = yaw_from_quat(s.sim.data.qpos[3:7])
    commanded = 0.0
    s.flags.manual_input_held = False
    samples = []
    for _ in range(int(0.6 / C.PHYSICS_DT)):
        s.advance(C.PHYSICS_DT)
        samples.append(s.sim.true_velocity()[1])
        commanded += s.applied.omega * C.PHYSICS_DT
    gained = (yaw_from_quat(s.sim.data.qpos[3:7]) - y0 + math.pi) % (2 * math.pi) - math.pi
    return samples, gained, commanded


@pytest.mark.parametrize("heading", [0.4, 0.8, 2.5, -2.0])
@pytest.mark.parametrize("sign", [1, -1])
def test_releasing_a_turn_does_not_rock(heading, sign):
    """Stop rock: after letting go of a turn key, the body must not swing back (at most
    0.05 rad/s), later swings must not grow, and it stops where the wheels say."""
    from tests.helpers import swing_lobes
    s = empty_system()
    s.reset(-1.5, 0.0, heading, (2.5, 2.5))
    s.flags.manual_mode = s.flags.manual_input_held = True
    samples, gained, commanded = _release_turn(s, sign)
    assert max(0.0, max(-sign * w for w in samples)) <= 0.05  # direct, independent of the analyzer
    _, lobes = swing_lobes(samples, sign)
    assert all(b <= a + 1e-3 for a, b in zip(lobes, lobes[1:])), lobes
    assert abs(math.degrees(gained - commanded)) <= 0.5
    s.close()


@pytest.mark.parametrize("heading", [0.4, 2.5])
def test_repeated_left_right_switching_then_release_does_not_rock(heading):
    from tests.helpers import swing_lobes
    s = empty_system()
    s.reset(-1.5, 0.0, heading, (2.5, 2.5))
    s.flags.manual_mode = s.flags.manual_input_held = True
    for k in range(6):
        for i in range(int(0.4 / C.PHYSICS_DT)):
            if i % 10 == 0:
                s.drive(0.0, 1.0 if k % 2 == 0 else -1.0)
            s.advance(C.PHYSICS_DT)
    s.flags.manual_input_held = False
    samples = []
    for _ in range(int(0.6 / C.PHYSICS_DT)):
        s.advance(C.PHYSICS_DT)
        samples.append(s.sim.true_velocity()[1])
    assert max(0.0, max(samples)) <= 0.05  # the last command was -1, so opposite is positive
    _, lobes = swing_lobes(samples, -1.0)
    assert all(b <= a + 1e-3 for a, b in zip(lobes, lobes[1:])), lobes
    s.close()


@pytest.mark.parametrize("w", [1.0, 1.5, -1.5])
def test_pivot_turn_tracks_the_command_within_2_percent(w):
    s = empty_system()
    s.reset(-1.5, 0.0, 0.3, (2.5, 2.5))
    _, rate = settled_velocity(s, 0.0, w)
    assert rate == pytest.approx(w, rel=0.02)
    s.close()


@pytest.mark.parametrize("w", [1.0, -1.0])
def test_arc_radius_matches_the_command_within_2_percent(w):
    s = empty_system()
    s.reset(-1.5, 0.0, 0.3, (2.5, 2.5))
    v, rate = settled_velocity(s, 0.3, w)
    assert abs(v / rate) == pytest.approx(0.3, rel=0.02)
    s.close()


# ----- tire loading: starts, stops, and emergency stops (no caster slam) -----

@pytest.mark.parametrize("v,w", [(0.3, 0.0), (0.5, 0.0), (-0.25, 0.0), (0.3, 1.0), (0.3, -1.0),
                                 (0.5, 1.5), (0.5, -1.5)])
def test_ordinary_starts_and_releases_keep_both_tires_loaded(v, w):
    """Regression: accelerating pitched the body onto a caster and briefly unloaded the tires."""
    s = empty_system()
    s.reset(-1.5 if v > 0 else 1.5, 0.0, 0.0, (2.5, 2.5))
    r = loading_run(s, [(1.5, v, w, "drive"), (1.0, 0.0, 0.0, "release")])
    assert r["lost"] == 0 and r["ratio"] <= 1.5 and r["warnings"] == 0, r
    if w == 0.0:
        assert max(r["shares"]) <= 0.02  # the casters carry almost nothing while cruising
    s.close()


def test_casters_carry_little_weight_in_a_saturated_pivot():
    s = empty_system()
    s.reset(0.0, 0.0, 0.0, (2.5, 2.5))
    r = loading_run(s, [(2.0, 0.0, 1.5, "drive")])
    assert max(r["shares"]) <= 0.15 and r["lost"] == 0
    s.close()


@pytest.mark.parametrize("v", [0.3, 0.5])
@pytest.mark.parametrize("w", [0.0, 1.5, -1.5])
def test_emergency_stop_does_not_slam_or_bounce(v, w):
    """Regression: a hard stop locked the wheels (3 N m motors) and slammed the robot onto a
    caster at up to 10 times the static tire load. The braking path stays inside the
    predictor's stopping distance."""
    s = empty_system()
    s.reset(0.0, 0.0, 0.0, (2.5, 2.5))
    r = loading_run(s, [(2.0, v, w, "drive"), (1.0, v, w, "ebrake")])  # one static measurement, at rest
    v0 = r["v0"]
    predictor = v0 * C.SAFETY_DELAY + v0 * v0 / (2 * C.SAFETY_DECEL)
    assert r["ratio"] <= 3.0 and r["longest"] <= 0.05 and r["lost"] == 0 and r["warnings"] == 0, r
    assert r["path"] <= predictor, (r["path"], predictor)
    s.close()
