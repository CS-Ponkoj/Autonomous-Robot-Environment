"""Step 8: Gymnasium environment, goal task, seeds, reset, reward, and termination."""

import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from robot_env import config as C
from robot_env.actions import ACTION_NAMES, ACTIONS, to_command
from robot_env.env import RobotGoalEnv
from robot_env.layout import RoomMap
from robot_env.types import Command
from tests.helpers import WaypointFollower


@pytest.fixture(scope="module")
def env():
    e = RobotGoalEnv()
    yield e
    e.close()


def test_gymnasium_checker_passes(env):
    check_env(env, skip_render_check=True)


def test_same_seed_same_task_and_observation(env):
    o1, i1 = env.reset(seed=7)
    o2, i2 = env.reset(seed=7)
    for k in o1:
        assert np.array_equal(o1[k], o2[k])
    assert i1["task_seed"] == i2["task_seed"]


def test_task_seed_option_selects_task(env):
    env.reset(options={"task_seed": 1003})
    assert env.task.seed == 1003


def test_observation_has_no_ground_truth(env):
    obs, info = env.reset(seed=1)
    assert set(obs) == {"lidar", "lidar_valid", "velocity_estimate", "goal"}
    assert "ground_truth" in info  # ground truth is evaluation-only, in info


def test_reset_restores_complete_state(env):
    env.reset(options={"task_seed": 1001})
    for _ in range(20):
        env.step(np.array([0.3, 0.5], dtype=np.float32))
    obs_a, info_a = env.reset(options={"task_seed": 1001})
    assert info_a["sim_time"] == 0.0 and info_a["collisions"] == 0 and info_a["intervention_events"] == 0
    obs_b, _ = env.reset(options={"task_seed": 1001})
    for k in obs_a:
        assert np.array_equal(obs_a[k], obs_b[k])


@pytest.mark.parametrize("seed", C.HELDOUT_SEEDS)
def test_heldout_tasks_are_valid_and_fit_the_time_limit(seed):
    room = RoomMap(RobotGoalEnv().system.sim.model)
    t = room.sample_task(seed)
    assert room.point_clearance(*t.start[:2]) >= C.START_GOAL_MIN_CLEARANCE
    assert room.point_clearance(*t.goal) >= C.START_GOAL_MIN_CLEARANCE
    assert math.dist(t.start[:2], t.goal) >= C.START_GOAL_MIN_SEPARATION
    assert all(room.line_free(a, b) for a, b in zip(t.path[:-1], t.path[1:]))
    assert t.estimated_travel_time() <= C.EPISODE_TIME_LIMIT / 2


def test_standing_still_cannot_succeed(env):
    env.reset(options={"task_seed": 1000})
    total, terminated, truncated, info = 0.0, False, False, {}
    while not (terminated or truncated):
        _, r, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        total += r
    assert truncated and not info["is_success"]
    assert info["sim_time"] == pytest.approx(C.EPISODE_TIME_LIMIT, abs=0.11)


def test_reward_is_progress_toward_goal(env):
    env.reset(options={"task_seed": 1006})
    # Face the goal first, then drive toward it: reward should be positive.
    for _ in range(40):
        b = env.system.observe().goal_bearing
        if abs(b) < 0.1:
            break
        env.step(np.array([0.0, math.copysign(1.0, b)], dtype=np.float32))
    d0 = env.system.observe().goal_distance
    _, r, *_ = env.step(np.array([0.3, 0.0], dtype=np.float32))
    d1 = env.system.observe().goal_distance
    assert r == pytest.approx(d0 - d1, abs=1e-6) and r > 0


@pytest.mark.parametrize("seed", C.HELDOUT_SEEDS)
def test_heldout_goal_reachable_through_drive_and_safety(seed):
    """Test-only waypoint follower (true pose + planning map) drives every held-out goal
    through the shared drive() and safety path. Not evidence of manual usability."""
    env = RobotGoalEnv()
    env.reset(options={"task_seed": seed})
    follower = WaypointFollower(env.system, env.task)
    info, terminated, truncated, total = {}, False, False, 0.0
    while not (terminated or truncated):
        v, w = follower.command()
        _, r, terminated, truncated, info = env.step(np.array([v, w], dtype=np.float32))
        total += r
    assert info["is_success"], info
    assert info["collisions"] == 0
    assert total > C.REWARD_GOAL  # progress plus the goal bonus
    env.close()


def test_collision_ends_episode_with_penalty():
    from robot_env.safety import SafetyLayer
    from robot_env.sim import RobotSim
    from robot_env.system import RobotSystem
    env = RobotGoalEnv(system=RobotSystem(RobotSim(), SafetyLayer(clearance_enabled=False)))
    env.reset(options={"task_seed": 1000})
    terminated, info, reward = False, {}, 0.0
    for _ in range(600):
        _, reward, terminated, truncated, info = env.step(np.array([0.5, 0.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated and info["collision"] and not info["is_success"]
    assert reward <= C.REWARD_COLLISION + 1.0
    env.close()


def test_collision_can_continue_when_configured():
    from robot_env.safety import SafetyLayer
    from robot_env.sim import RobotSim
    from robot_env.system import RobotSystem
    env = RobotGoalEnv(collision_ends_episode=False,
                       system=RobotSystem(RobotSim(), SafetyLayer(clearance_enabled=False)))
    env.reset(options={"task_seed": 1000})
    hit = False
    for _ in range(300):
        _, _, terminated, truncated, info = env.step(np.array([0.5, 0.0], dtype=np.float32))
        hit = hit or info["collision"]
        assert not terminated or info["is_success"]
        if truncated:
            break
    assert hit and info["collisions"] >= 1  # recorded although the episode continued
    env.close()


def test_discrete_actions_and_adapter():
    assert ACTION_NAMES == ("forward", "turn_left", "turn_right", "stop", "back_up")
    assert to_command("forward") == Command(0.30, 0.0)
    assert to_command(1) == Command(0.0, 1.0)
    for c in ACTIONS.values():
        assert abs(c.v) <= C.MAX_LINEAR_SPEED and abs(c.omega) <= C.MAX_ANGULAR_SPEED


@pytest.mark.parametrize("seed", C.HELDOUT_SEEDS[:3])
def test_driver_plug_drives_through_system_apply(seed):
    """decide(observation) -> Decision -> RobotSystem.apply, at the 10 Hz decision rate."""
    env = RobotGoalEnv()
    env.reset(options={"task_seed": seed})
    s = env.system
    driver = WaypointFollower(s, env.task)
    while s.time < C.EPISODE_TIME_LIMIT and s.observe().goal_distance > C.GOAL_RADIUS:
        decision = driver.decide(s.observe())
        s.apply(decision)
        assert s.last_decision is decision
        s.advance(C.DECISION_PERIOD)
    assert s.observe().goal_distance <= C.GOAL_RADIUS and s.collisions == 0
    env.close()
