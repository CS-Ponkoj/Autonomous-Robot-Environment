"""The MLP brain: features.vector(obs) -> one of the 5 actions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from robot_env.types import Decision

from robot_env import config as C
from robot_env.types import Command

from ...common.features import ACTION_NAMES, FEATURES, LABEL_LEVEL, ODOMETRY, to_command, vector

# continuous head: (v, omega) as tanh shares of speed level 2's caps (the label level)
SCALE = torch.tensor(C.SPEED_LEVELS[LABEL_LEVEL])


class Net(nn.Module):
    """head "discrete": logits over the 8 actions; "cont": (v, omega) in level 2's caps."""

    def __init__(self, hidden: int = 256, layers: int = 3, dropout: float = 0.1, head: str = "discrete"):
        super().__init__()
        self.head = head
        mods, width = [], FEATURES
        for _ in range(layers):
            mods += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        mods.append(nn.Linear(width, len(ACTION_NAMES) if head == "discrete" else 2))
        self.body = nn.Sequential(*mods)

    def forward(self, x):
        out = self.body(x)
        return torch.tanh(out) * SCALE.to(out.device) if self.head == "cont" else out


def load(path: Path) -> Net:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Net(**ck["arch"])
    net.load_state_dict(ck["state"])
    net.eval()
    net.no_odometry = ck.get("no_odometry", False)
    return net


class MLPDriver:
    """Picks the most likely action each tick (GPU when present)."""

    name = "mlp"

    def __init__(self, path, device: str | None = None):
        torch.set_num_threads(1)
        # the GPU by default; "cpu" for benchmark worker processes when RAM is short (each CUDA
        # context costs about 1.5 GB per process, and this network decides in well under 1 ms)
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = load(Path(path)).to(self.dev)

    def decide(self, obs):
        x = vector(obs)
        if self.net.no_odometry:
            x[ODOMETRY] = 0.0
        with torch.no_grad():
            out = self.net(torch.from_numpy(x)[None].to(self.dev))[0]
        if self.net.head == "cont":
            return Decision(Command(float(out[0]), float(out[1])), obs.seq)
        return Decision(to_command(int(out.argmax().item())), obs.seq)
