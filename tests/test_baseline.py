"""Rule-based baseline driver: sees only the Observation, deterministic, collision-free."""

import ast
import inspect

import numpy as np

from robot_env import baseline
from robot_env import config as C
from robot_env.baseline import BaselineDriver
from robot_env.types import Observation

FORBIDDEN = {"robot_env.sim", "robot_env.system", "robot_env.layout", "robot_env.env", "robot_env.app",
             "robot_env.drive_log", "mujoco", "gymnasium"}


def test_baseline_imports_nothing_privileged():
    """The driver module may not import the simulator, MuJoCo, the map or task generator, the
    environment, or logging (where truth lives)."""
    tree = ast.parse(inspect.getsource(baseline))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: "from . import config" or "from .types import ..."
                base = "robot_env" + ("." + node.module if node.module else "")
                imported |= {base + "." + a.name for a in node.names} if not node.module else {base}
            else:
                imported.add(node.module)
    assert not (imported & FORBIDDEN), imported & FORBIDDEN
    assert imported <= {"__future__", "math", "numpy", "robot_env.config", "robot_env.types"}, imported


def _obs(seq, t, ranges, goal=(3.0, 0.0), vel=(0.0, 0.0)):
    return Observation(seq, t, np.asarray(ranges, float), np.ones(C.LIDAR_RAYS, bool), np.array(C.LIDAR_ANGLES),
                       t, vel, goal[0], goal[1], False)


def test_baseline_runs_on_a_bare_observation_and_holds_no_privileged_object():
    d = BaselineDriver()
    out = d.decide(_obs(1, 0.0, np.full(C.LIDAR_RAYS, 4.0)))
    assert out.obs_seq == 1 and out.command.v > 0  # open space, goal ahead: drives forward
    for value in vars(d).values():  # its whole memory is plain numbers and arrays
        assert isinstance(value, (int, float, bool, str, tuple, np.ndarray, type(None))), type(value)


def test_baseline_turns_toward_a_goal_behind_it():
    d = BaselineDriver()
    out = d.decide(_obs(1, 0.0, np.full(C.LIDAR_RAYS, 4.0), goal=(3.0, 3.0)))
    assert out.command.v == 0.0 and out.command.omega > 0  # turn in place toward the goal on the left


def test_baseline_is_deterministic_and_collision_free_on_development_seeds():
    """Two runs of the same seeds give identical results; no development seed collides."""
    import tools.eval_baseline as ev
    from robot_env.layout import RoomMap
    from robot_env.system import RobotSystem
    s = RobotSystem()
    room = RoomMap(s.sim.model)
    first = [ev.run_episode(s, room, seed, C.DEFAULT_SPEED_LEVEL) for seed in (2001, 2007)]
    second = [ev.run_episode(s, room, seed, C.DEFAULT_SPEED_LEVEL) for seed in (2001, 2007)]
    assert first == second
    assert all(r["outcome"] == "success" and r["collisions"] == 0 for r in first), first
    s.close()
