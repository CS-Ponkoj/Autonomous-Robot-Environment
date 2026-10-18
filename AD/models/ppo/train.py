"""PPO (Stable-Baselines3) starting from a cloned CNN policy, in robot_env's own environment
(through AD/models/ppo/envs.py; the environment is not changed).

    .venv\\Scripts\\python -m AD.models.ppo.train --base cnn/dagger_cont_r3 --envs 4 --save-every 100000

The actor is the CNN itself: its lidar and goal branches and first trunk layer are the features,
its last layer gives the action mean, and the environment squashes it as the CNN does (tanh times
the caps), so before any update PPO drives exactly as the cloned policy. The critic has its own
copy of the features and a new value head; for the first `--warmup` steps only the critic learns
(an untrained critic would otherwise pull the actor off course).

Every `--save-every` steps the actor is written back in the CNN's checkpoint format as its own run
(AD/models/ppo/runs/ppo_r<k>/model.pt, with meta.json), so CNNDriver, final.py and rebench.py
use it unchanged; AD/models/ppo/bench.py benchmarks and logs each one. Rollouts: CPU processes;
the networks: the GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ...common import tracker
from ..cnn.policy import CNNPolicy, load as load_cnn, save as save_cnn

RUNS = tracker.AD / "models" / "ppo" / "runs"
CATS = (0, 4, 2, 4, 1, 3)  # per parallel environment (the practice benchmark uses 0 and 4)


def make_env(rank: int, odometry: bool, route: bool = False):
    def thunk():
        from .envs import FeatureEnv
        return FeatureEnv(discrete=False, odometry=odometry, cats=CATS[rank % len(CATS)], cat_seed=7000 + rank,
                          squash=True, route=route)
    return thunk


def extractor_class():
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class CNNFeatures(BaseFeaturesExtractor):
        """The cloned CNN up to its last layer (128 features)."""

        def __init__(self, observation_space, ckpt: str):
            super().__init__(observation_space, features_dim=128)
            base = load_cnn(ckpt)
            self.lidar, self.scalars, self.hidden = base.lidar, base.scalars, base.trunk[0]
            self.register_buffer("mask", base.mask.clone())
            self.frames, self.n_in = base.frames, base.n_in

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            from ...common.features import LIDAR_BINS
            x = x.view(x.shape[0], self.frames, self.n_in) * self.mask
            z = torch.cat([self.lidar(x[:, :, :LIDAR_BINS]), self.scalars(x[:, :, LIDAR_BINS:].flatten(1))], 1)
            return torch.relu(self.hidden(z))
    return CNNFeatures


def to_cnn(model, base_ckpt: str) -> CNNPolicy:
    """The PPO actor (mean action) as a continuous-head CNNPolicy."""
    net = load_cnn(base_ckpt)
    ext = model.policy.pi_features_extractor
    net.lidar.load_state_dict(ext.lidar.state_dict())
    net.scalars.load_state_dict(ext.scalars.state_dict())
    net.trunk[0].load_state_dict(ext.hidden.state_dict())
    net.trunk[2].load_state_dict(model.policy.action_net.state_dict())
    return net.cpu().eval()


def callbacks(base_ckpt: str, a, t_start: float):
    from stable_baselines3.common.callbacks import BaseCallback
    runs = tracker.AD / "models" / getattr(a, "out_model", "ppo") / "runs"

    class Warmup(BaseCallback):
        """Only the critic learns for the first `steps` steps."""

        def __init__(self, steps: int):
            super().__init__()
            self.steps, self.frozen = steps, False

        def _actor_params(self):
            p = self.model.policy
            return [*p.pi_features_extractor.parameters(), *p.action_net.parameters(), p.log_std]

        def _set(self, frozen: bool):
            for q in self._actor_params():
                q.requires_grad_(not frozen)
            self.frozen = frozen

        def _on_training_start(self):
            self._set(self.steps > 0)

        def _on_step(self):
            if self.frozen and self.num_timesteps >= self.steps:
                self._set(False)
                print(f"warmup over at {self.num_timesteps} steps: the actor learns now", flush=True)
            return True

    class Checkpoints(BaseCallback):
        """The actor as its own run every `every` steps, with what training looked like."""

        def __init__(self, every: int):
            super().__init__()
            self.every, self.k, self.next = every, 0, every

        def _on_step(self):
            if self.num_timesteps >= self.next:
                self.next += self.every
                self.k += 1
                run = f"{a.run_prefix}{self.k}"
                save_cnn(to_cnn(self.model, base_ckpt), runs / run / "model.pt", {"run": run})
                buf = list(self.model.ep_info_buffer)
                meta = {"run": run, "base": a.base, "timesteps": int(self.num_timesteps),
                        "train_s": round(time.time() - t_start, 1), "envs": a.envs, "cats": list(CATS[:a.envs]),
                        "recent_episodes": len(buf),
                        "recent_return": round(float(np.mean([e["r"] for e in buf])), 3) if buf else None,
                        "recent_success": round(float(np.mean([e.get("is_success", 0) for e in buf])), 3) if buf else None,
                        "params": {"lr": a.lr, "n_steps": a.n_steps, "batch": a.batch, "epochs": a.epochs,
                                   "clip": a.clip, "gamma": a.gamma, "log_std_init": a.log_std, "warmup": a.warmup}}
                (runs / run / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
                self.model.save(runs / run / "sb3_model")  # to continue training from here
                print(f"checkpoint {run}: {meta}", flush=True)
            return True

    return [Warmup(a.warmup), Checkpoints(a.save_every)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default="cnn/dagger_cont_r3", help="<model>/<run> of the continuous CNN to start from")
    p.add_argument("--run-prefix", default="ppo_r")
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=600_000, help="total environment steps")
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("--warmup", type=int, default=20_000, help="steps during which only the critic learns")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--n-steps", type=int, default=512, help="steps per environment per update")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--log-std", type=float, default=-1.0, help="initial log std of the (pre-tanh) action noise")
    p.add_argument("--resume", default=None, help="a saved sb3_model.zip to continue from")
    p.add_argument("--out-model", default="ppo", help="model folder for the checkpoints (hybrid for a hybrid base)")
    a = p.parse_args(argv)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    base_ckpt = str(tracker.AD / "models" / a.base.split("/")[0] / "runs" / a.base.split("/")[1] / "model.pt")
    base = load_cnn(base_ckpt)
    assert base.head == "continuous" and base.frames == 1, "PPO starts from a 1-frame continuous CNN"
    route = base.extra > 0  # a hybrid base: the environments add the route sensor's values
    env = VecMonitor(SubprocVecEnv([make_env(r, base.odometry, route) for r in range(a.envs)]), info_keywords=("is_success",))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.resume:
        model = PPO.load(a.resume, env=env, device=device)
    else:
        model = PPO("MlpPolicy", env, learning_rate=a.lr, n_steps=a.n_steps, batch_size=a.batch, n_epochs=a.epochs,
                    clip_range=a.clip, gamma=a.gamma, gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                    device=device, seed=0, verbose=1,
                    policy_kwargs=dict(features_extractor_class=extractor_class(),
                                       features_extractor_kwargs={"ckpt": base_ckpt},
                                       share_features_extractor=False, net_arch=dict(pi=[], vf=[64]),
                                       activation_fn=nn.ReLU, log_std_init=a.log_std, ortho_init=False))  # keep the loaded CNN weights
        with torch.no_grad():  # the actor's mean starts as the CNN's last layer
            model.policy.action_net.load_state_dict(base.trunk[2].state_dict())
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=callbacks(base_ckpt, a, t0), reset_num_timesteps=not a.resume)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
