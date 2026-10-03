"""Step 5: motion check on an empty track (true velocity is used for acceptance)."""

import math

import pytest

from robot_env import config as C
from robot_env.safety import predict_poses
from tests.helpers import empty_system, hold


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
