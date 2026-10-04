"""Gymnasium environment: reach the goal without collisions.

Action: Box(2) = [v (m/s), omega (rad/s)], applied through the same drive() and
safety path as every other driver. One step = one decision (0.1 s).
Ground truth (true pose and velocity) is only in `info`, never in the observation.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config as C
from .layout import RoomMap, Task
from .system import RobotSystem
from .types import Observation


class RobotGoalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / C.DECISION_PERIOD),
                "observation_version": C.OBSERVATION_VERSION, "action_version": C.ACTION_VERSION}

    def __init__(self, render_mode: str | None = None, collision_ends_episode: bool = True,
                 system: RobotSystem | None = None, cats: int = 0, cat_seed: int = 0):
        self.render_mode = render_mode
        self.collision_ends_episode = collision_ends_episode
        if system is not None and (cats or cat_seed):
            herd = system.cats
            if herd is None or herd.n != cats or herd.seed != cat_seed:
                raise ValueError("cats/cat_seed disagree with the given system; set them on the system")
        self.system = system or RobotSystem(cats=cats, cat_seed=cat_seed)
        self.room = RoomMap(self.system.sim.model)
        self.task: Task | None = None
        self.action_space = spaces.Box(
            low=np.array([-C.MAX_LINEAR_SPEED, -C.MAX_ANGULAR_SPEED], dtype=np.float32),
            high=np.array([C.MAX_LINEAR_SPEED, C.MAX_ANGULAR_SPEED], dtype=np.float32),
            dtype=np.float32)
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(0.0, C.LIDAR_RANGE, shape=(C.LIDAR_RAYS,), dtype=np.float32),
            "lidar_valid": spaces.MultiBinary(C.LIDAR_RAYS),
            "velocity_estimate": spaces.Box(np.array([-2.0, -6.0], dtype=np.float32),
                                            np.array([2.0, 6.0], dtype=np.float32), dtype=np.float32),
            "goal": spaces.Box(np.array([0.0, -np.pi], dtype=np.float32),
                               np.array([20.0, np.pi], dtype=np.float32), dtype=np.float32),
        })
        self._renderer = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        task_seed = (options or {}).get("task_seed")
        if task_seed is None:
            task_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.task = self.room.sample_task(int(task_seed))
        x, y, yaw = self.task.start
        self.system.reset(x, y, yaw, self.task.goal)
        self._prev_distance = self.system.observe().goal_distance
        self._prev_interventions = 0
        self._prev_collisions = 0
        return self._obs(), self._info(success=False, collided=False)

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(2)
        self.system.drive(a[0], a[1])
        self.system.advance(C.DECISION_PERIOD)
        obs = self.system.observe()

        new_collision = self.system.collisions > self._prev_collisions
        new_interventions = self.system.intervention_events - self._prev_interventions
        self._prev_collisions = self.system.collisions
        self._prev_interventions = self.system.intervention_events
        reached = obs.goal_distance <= C.GOAL_RADIUS

        reward = self._prev_distance - obs.goal_distance
        self._prev_distance = obs.goal_distance
        reward += C.REWARD_INTERVENTION * new_interventions
        collided = new_collision
        success = reached and not collided  # a collision in the same step is a failure
        if collided:
            reward += C.REWARD_COLLISION
        elif success:
            reward += C.REWARD_GOAL
        terminated = bool(success or (collided and self.collision_ends_episode))
        truncated = bool(not terminated and self.system.time >= C.EPISODE_TIME_LIMIT - 1e-9)
        if terminated or truncated:
            self.system.flags.episode_over = True  # the shared safety stop; reset clears it
            if self.system.cats is not None:
                self.system.finish_cat_contacts("episode_end")
        if self.system.log is not None and (terminated or truncated or collided):
            outcome = "success" if success else "collision" if collided else "timeout"
            self.system.log.event("episode_" + outcome, self.system.time, terminated=terminated, truncated=truncated)
        return self._obs(), float(reward), terminated, truncated, self._info(success, collided)

    def _obs(self) -> dict:
        o: Observation = self.system.observe()
        return {
            "lidar": np.clip(o.lidar, 0.0, C.LIDAR_RANGE).astype(np.float32),
            "lidar_valid": o.lidar_valid.astype(np.int8),
            "velocity_estimate": np.clip(np.array(o.velocity_estimate), [-2, -6], [2, 6]).astype(np.float32),
            "goal": np.array([min(o.goal_distance, 20.0), o.goal_bearing], dtype=np.float32),
        }

    def _info(self, success: bool, collided: bool) -> dict:
        s = self.system
        info = {
            "seq": s.observe().seq,
            "observation_version": C.OBSERVATION_VERSION,
            "action_version": C.ACTION_VERSION,
            "sim_time": s.time,
            "task_seed": self.task.seed if self.task else None,
            "is_success": success,
            "collision": collided,
            "collisions": s.collisions,
            "intervention_events": s.intervention_events,
            "intervention_time": s.intervention_time,
            "safety_reasons": s.last_result.reasons,
            "approved_command": (s.last_result.command.v, s.last_result.command.omega),  # after safety
            "applied_command": (s.applied.v, s.applied.omega),  # motor target after the smoother
            "ground_truth": s.ground_truth(),  # evaluation only
        }
        return info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.system.sim.render_camera()
        return None

    def close(self):
        self.system.close()
