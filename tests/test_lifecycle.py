"""Episode lifecycle and data-ownership regressions: the episode-end stop in Gym, driver reset
on restart, read-only observation arrays, and drive-log timestamps (cats and validation)."""

import json
import math

import numpy as np
import pytest

from robot_env import config as C
from robot_env.baseline import BaselineDriver
from robot_env.drive_log import DriveLog, IncompleteLogError, read_log
from robot_env.env import RobotGoalEnv
from robot_env.layout import RoomMap
from robot_env.safety import SafetyLayer
from robot_env.system import STOP, RobotSystem


# ----- 1. Gym: an ended episode latches the shared stop -----
def _assert_stopped_next_step(env):
    s = env.system
    # at once, when env.step() returns: no request, no decision, motors at zero
    assert s.flags.episode_over
    assert s.requested is None and s.command_age is None and s.last_decision is None
    assert s.applied == STOP and s._target == STOP and s._wheel_v == 0.0 and s._wheel_w == 0.0
    assert s.last_result.command == STOP and s.last_result.reasons == ("episode_over",)
    assert np.allclose(s.sim.data.ctrl, 0.0)
    # and it stays latched
    s.advance(C.PHYSICS_DT)
    assert s.flags.episode_over
    assert s.last_result.command == STOP and "episode_over" in s.last_result.reasons
    assert s.applied == STOP
    assert np.allclose(s.sim.data.ctrl, 0.0)


def _drive_until_end(env, action, steps=4000):
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            return terminated, truncated, info
    raise AssertionError("the episode never ended")


def test_gym_goal_latches_the_episode_stop():
    env = RobotGoalEnv(cats=0)
    env.reset(seed=1000)
    s = env.system
    x, y, yaw = s.sim.true_pose()
    s.sim.goal[:] = (x + 0.45 * math.cos(yaw), y + 0.45 * math.sin(yaw))  # straight ahead, just past the radius
    terminated, truncated, info = _drive_until_end(env, (1.0, 0.0))
    assert terminated and info["is_success"]
    _assert_stopped_next_step(env)
    env.reset(seed=1001)
    assert not env.system.flags.episode_over
    assert env.system.requested is None and env.system.last_result.reasons == ("no_command",)
    env.close()


def test_gym_timeout_latches_the_episode_stop(monkeypatch):
    monkeypatch.setattr(C, "EPISODE_TIME_LIMIT", 1.0)
    env = RobotGoalEnv(cats=0)
    env.reset(seed=1000)
    terminated, truncated, _ = _drive_until_end(env, (0.5, 0.0))  # still asking to move
    assert truncated and not terminated
    _assert_stopped_next_step(env)
    env.close()


def _wall_ahead_env(collision_ends_episode):
    """Fault injection: no clearance filter, robot facing a wall 0.4 m away."""
    s = RobotSystem(safety=SafetyLayer(clearance_enabled=False))
    env = RobotGoalEnv(collision_ends_episode=collision_ends_episode, system=s)
    env.reset(seed=1000)
    half = C.FLOOR_HALF_SIZE
    s.reset(half - 0.4 - C.FOOTPRINT_HALF_LENGTH, -3.5, 0.0, (-4.0, -4.0))
    return env


def test_gym_terminating_collision_latches_the_episode_stop():
    env = _wall_ahead_env(True)
    terminated, truncated, info = _drive_until_end(env, (1.0, 0.0))
    assert terminated and info["collision"]
    _assert_stopped_next_step(env)
    env.close()


def test_gym_collision_that_does_not_end_the_episode_does_not_stop_it():
    env = _wall_ahead_env(False)
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        if info["collision"]:
            break
    assert info["collision"] and not terminated and not truncated
    assert not env.system.flags.episode_over
    env.close()


# ----- 2. restart and next goal reset a stateful driver -----
def _drive(s, d, seconds):
    for _ in range(int(round(seconds / C.DECISION_PERIOD))):
        s.apply(d.decide(s.observe()))
        s.advance(C.DECISION_PERIOD)


def _state(d):
    return (d._last_time, d._last_vel, int(np.count_nonzero(d.occupied)))


@pytest.mark.gui
@pytest.mark.parametrize("next_seed", [0, 1])
def test_app_new_episode_resets_the_baseline_driver(next_seed):
    from robot_env.app import App
    d = BaselineDriver()
    app = App(1000, screenshot=None, frames=1, driver=d)
    fresh = _state(BaselineDriver())
    _drive(app.system, d, 1.2)
    assert _state(d) != fresh
    app.new_episode(app.seed + next_seed)  # R restarts the same goal, N the next one
    assert _state(d) == fresh
    app.frames_left = 1
    app.run()


def _headless_app(driver):
    """An App without a window: only what new_episode uses (the production method itself)."""
    from robot_env.app import App
    from robot_env.manual import ManualInput
    app = App.__new__(App)
    app.system = RobotSystem()
    app.room = RoomMap(app.system.sim.model)
    app.driver = driver
    app.input = ManualInput(C.DEFAULT_SPEED_LEVEL)
    return app


@pytest.mark.parametrize("next_seed", [0, 1])
def test_new_episode_resets_the_baseline_driver_headless(next_seed):
    d = BaselineDriver()
    app = _headless_app(d)
    app.new_episode(1000)
    fresh = _state(BaselineDriver())
    _drive(app.system, d, 1.2)
    assert _state(d) != fresh
    app.new_episode(1000 + next_seed)  # R: same goal, N: next goal
    assert _state(d) == fresh and app.system.time == 0.0
    app.system.close()


def test_new_episode_accepts_a_driver_without_reset_headless():
    class Stateless:
        name = "stateless"

        def decide(self, obs):
            return None

    app = _headless_app(Stateless())
    app.new_episode(1000)
    app.new_episode(1001)
    assert app.seed == 1001
    app.system.close()


@pytest.mark.gui
def test_app_new_episode_accepts_a_driver_without_reset():
    from robot_env.app import App

    class Stateless:
        name = "stateless"

        def decide(self, obs):
            return None

    app = App(1000, screenshot=None, frames=1, driver=Stateless())
    app.new_episode(1001)
    app.run()


# ----- 3. observation arrays are read-only snapshots -----
@pytest.mark.parametrize("with_camera", [False, pytest.param(True, marks=pytest.mark.opengl)])
def test_observation_arrays_are_independent_and_read_only(with_camera):
    s = RobotSystem()
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    s.advance(0.1)
    sim_angles = s.sim.lidar_angles.copy()
    scan_before = s.sim.scan()[0]
    for _ in range(2):  # before and after a reset
        o = s.observe(with_camera=with_camera)
        for arr in (o.lidar, o.lidar_valid, o.lidar_angles):
            assert not np.shares_memory(arr, s.sim.lidar_angles)
            with pytest.raises(ValueError):
                arr += arr  # in-place preprocessing must not silently change shared state
        with pytest.raises(ValueError):
            np.rad2deg(o.lidar_angles, out=o.lidar_angles)
        with pytest.raises(ValueError):
            s.sim.lidar_angles[0] = 1.0
        assert np.array_equal(s.sim.lidar_angles, sim_angles)
        s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
        s.advance(0.1)
    assert np.allclose(s.sim.scan()[0], scan_before)
    s.close()


# ----- 4. a cat episode's log round-trips -----
def test_a_cat_episode_log_round_trips(tmp_path):
    """The reported run: goal seed 1000, level 2, 3 cats, cat seed 16, baseline driver."""
    s = RobotSystem(cats=3, cat_seed=16)
    task = RoomMap(s.sim.model).sample_task(1000)
    s.log = DriveLog(tmp_path / "cat.jsonl")
    s.reset(*task.start, task.goal)
    d = BaselineDriver()
    while s.time < 14.0 and s.observe().goal_distance > C.GOAL_RADIUS:
        s.apply(d.decide(s.observe()))
        s.advance(C.DECISION_PERIOD)
    s.close()
    recs = read_log(tmp_path / "cat.jsonl")
    states = [r for r in recs if r["kind"] == "event" and r["name"] == "cat_state"]
    assert len(states) >= 4 and states[0]["t"] == 0.0
    times = [r["t"] for r in recs if "t" in r]
    assert all(b >= a for a, b in zip(times, times[1:]))


# ----- 5. strict timestamp validation -----
def _valid_log(tmp_path):
    s = RobotSystem()
    s.log = DriveLog(tmp_path / "ok.jsonl")
    s.reset(-2.0, 0.0, 0.0, (4.0, 4.0))
    s.drive(0.2, 0.0)
    s.advance(0.2)
    s.close()
    return (tmp_path / "ok.jsonl").read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("field,raw", [
    ("t", "NaN"), ("t", "Infinity"), ("t", "-Infinity"), ("t", "1e999"), ("t", '"0.1"'), ("t", "true"), ("t", None),
    ("scan_time", "NaN"), ("scan_time", "1e999"), ("scan_time", '"x"'), ("scan_time", "false"), ("scan_time", None),
    ("command_issued", "NaN"), ("command_issued", "-1e999"), ("command_issued", '"x"'), ("command_issued", "true"),
    ("command_expires", "Infinity"), ("command_expires", '"x"'),
    ("t", "1" + "0" * 400), ("scan_time", "-1" + "0" * 400), ("command_issued", "1" + "0" * 400),
    ("command_expires", "-1" + "0" * 400),
])
def test_invalid_timestamps_are_rejected(tmp_path, field, raw):
    lines = _valid_log(tmp_path)
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec["kind"] == "tick" and rec["tick"] == 3:
            assert field in rec
            if raw is None:
                del rec[field]
                lines[i] = json.dumps(rec)
            else:
                rec[field] = "__X__"
                lines[i] = json.dumps(rec).replace('"__X__"', raw)
            break
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for allow in (False, True):
        with pytest.raises(IncompleteLogError):
            read_log(bad, allow_incomplete=allow)


def test_event_without_a_finite_time_is_rejected(tmp_path):
    lines = _valid_log(tmp_path)
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec["kind"] == "event":
            lines[i] = json.dumps(rec).replace(f'"t": {json.dumps(rec["t"])}', '"t": NaN', 1)
            break
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IncompleteLogError):
        read_log(bad)
