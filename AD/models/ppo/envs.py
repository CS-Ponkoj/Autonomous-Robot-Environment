"""Wrappers that let a PPO policy see and act like the CNN policies, around robot_env's own
RobotGoalEnv (which is used as is): observations are AD.common.features.vector(obs) (78
numbers, with v and omega masked as the cloned policies were trained), actions are either the 8
discrete actions or a continuous (v, omega) within speed level 2's caps. The reward is the
environment's own: progress toward the goal, +10 on arrival, -10 for a collision, -0.1 per
safety intervention.

Each reset picks a training layout (goal seeds 10000-10499, never a validation or test seed).
The cats are fixed per environment (count and seed): building a world with cats takes seconds, so
each parallel environment keeps its own and only the layout changes between episodes.

squash=True (continuous): the policy's raw output a goes through tanh and the caps, exactly as the
CNN's continuous head does, so a PPO actor started from a cloned CNN drives as that CNN did."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from robot_env import config as C
from robot_env.env import RobotGoalEnv

from ...common import dataset
from ...common.features import ACTION_NAMES, FEATURES, LABEL_LEVEL, ODOMETRY, to_command, vector

V_CAP, W_CAP = C.SPEED_LEVELS[LABEL_LEVEL]


class FeatureEnv(gym.Env):
    """One training layout per episode, with this environment's own cats."""

    metadata = {"render_modes": []}

    def __init__(self, discrete: bool = True, seeds=dataset.TRAIN, odometry: bool = False, cats: int = 0,
                 cat_seed: int = 0, squash: bool = False, route: bool = False):
        """route=True: the hybrid's observation, the built-in planner's route (AD/models/hybrid/
        route.py) appended (82 values); the planner reads the same observations every step."""
        self.discrete, self.odometry, self.squash = discrete, odometry, squash
        self.seeds = list(seeds)
        self.inner = RobotGoalEnv(cats=cats, cat_seed=cat_seed)
        self.sensor = None
        n = FEATURES
        if route:
            from ..hybrid.route import ROUTE_FEATURES, RouteSensor
            self.sensor, n = RouteSensor(), FEATURES + ROUTE_FEATURES
        self.observation_space = spaces.Box(-1.5, 1.5, shape=(n,), dtype=np.float32)
        if discrete:
            self.action_space = spaces.Discrete(len(ACTION_NAMES))
        elif squash:  # raw (pre-tanh) values; tanh(3) is 0.995 of the cap
            self.action_space = spaces.Box(-3.0, 3.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(np.array([-V_CAP, -W_CAP], np.float32), np.array([V_CAP, W_CAP], np.float32))

    def _features(self) -> np.ndarray:
        obs = self.inner.system.observe()
        x = vector(obs)
        if not self.odometry:
            x[ODOMETRY] = 0.0  # features.ODOMETRY: the v and omega slice
        if self.sensor is not None:
            x = np.concatenate([x, self.sensor.read(obs)])
        return x

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        task_seed = int(self.np_random.choice(self.seeds))  # seeded by reset(seed=...): reproducible
        self.inner.reset(options={"task_seed": task_seed})
        if self.sensor is not None:
            self.sensor.reset()
        return self._features(), {"task_seed": task_seed}

    def step(self, action):
        if self.discrete:
            cmd = to_command(int(action))
            a = np.array([cmd.v, cmd.omega], np.float32)
        elif self.squash:
            a = np.tanh(np.asarray(action, np.float32)) * np.array([V_CAP, W_CAP], np.float32)
        else:
            a = np.clip(np.asarray(action, np.float32), [-V_CAP, -W_CAP], [V_CAP, W_CAP])
        _, reward, terminated, truncated, info = self.inner.step(a)
        out = {"is_success": info["is_success"], "collision": info["collision"],
               "intervention_events": info["intervention_events"]}
        return self._features(), reward, terminated, truncated, out

    def close(self):
        if self.inner is not None:
            self.inner.close()
            self.inner = None
