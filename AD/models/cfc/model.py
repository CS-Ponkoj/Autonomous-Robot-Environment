"""A CfC (closed-form continuous-time) recurrent driving policy, in plain PyTorch.

The cell follows the default CfC of Hasani et al. (2022, "Closed-form continuous-time neural
networks"): a backbone layer over [input, state], two candidate states ff1 and ff2, and a time gate
sigma(t_a * dt + t_b) that blends them, so the state evolves with the elapsed time dt (here the
0.1 s decision period). The input encoding is the CNN policy's: a circular 1D convolution over the
72 lidar bins plus the goal values (measured v and omega masked against the copycat shortcut);
the head gives a continuous (v, omega) within speed level 2's caps. Its memory runs through the
whole episode (CfCDriver keeps the state between decisions)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from robot_env import config as C
from robot_env.types import Command, Decision, Observation

from ...common.features import FEATURES, LABEL_LEVEL, LIDAR_BINS, ODOMETRY, vector

SCALARS = FEATURES - LIDAR_BINS
V_CAP, W_CAP = C.SPEED_LEVELS[LABEL_LEVEL]
DT = C.DECISION_PERIOD


class CfCCell(nn.Module):
    def __init__(self, n_in: int, hidden: int, backbone: int = 128):
        super().__init__()
        self.hidden = hidden
        self.backbone = nn.Sequential(nn.Linear(n_in + hidden, backbone), nn.Tanh())
        self.ff1, self.ff2 = nn.Linear(backbone, hidden), nn.Linear(backbone, hidden)
        self.time_a, self.time_b = nn.Linear(backbone, hidden), nn.Linear(backbone, hidden)

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: float = DT) -> torch.Tensor:
        z = self.backbone(torch.cat([x, h], 1))
        gate = torch.sigmoid(self.time_a(z) * dt + self.time_b(z))
        return torch.tanh(self.ff1(z)) * (1.0 - gate) + gate * torch.tanh(self.ff2(z))


class CfCPolicy(nn.Module):
    def __init__(self, hidden: int = 64, width: int = 32):
        super().__init__()
        self.hidden_size, self.width = hidden, width
        w = width

        def conv(cin, cout, k, stride=1):
            return nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, padding_mode="circular")
        self.lidar = nn.Sequential(conv(1, w, 5), nn.ReLU(), conv(w, 2 * w, 5, 2), nn.ReLU(),
                                   conv(2 * w, 2 * w, 3, 2), nn.ReLU(), nn.Flatten(),
                                   nn.Linear(2 * w * (LIDAR_BINS // 4), 96), nn.ReLU())
        self.scalars = nn.Sequential(nn.Linear(SCALARS, 32), nn.ReLU())
        self.cell = CfCCell(128, hidden)
        self.head = nn.Linear(hidden, 2)
        mask = torch.ones(FEATURES)
        mask[ODOMETRY] = 0.0
        self.register_buffer("mask", mask)
        self.register_buffer("caps", torch.tensor([V_CAP, W_CAP]))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, FEATURES) -> (N, 128)."""
        x = x * self.mask
        return torch.cat([self.lidar(x[:, None, :LIDAR_BINS]), self.scalars(x[:, LIDAR_BINS:])], 1)

    def initial(self, n: int, device=None) -> torch.Tensor:
        return torch.zeros(n, self.hidden_size, device=device)

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One decision: (B, FEATURES), state (B, hidden) -> (v, omega) (B, 2), new state."""
        h = self.cell(self.encode(x), h)
        return torch.tanh(self.head(h)) * self.caps, h

    def forward(self, xs: torch.Tensor, h: torch.Tensor | None = None):
        """xs: (B, T, FEATURES) -> commands (B, T, 2), last state."""
        b, t, _ = xs.shape
        e = self.encode(xs.reshape(b * t, -1)).view(b, t, -1)  # the encoder once for all steps
        h = self.initial(b, xs.device) if h is None else h
        outs = []
        for k in range(t):
            h = self.cell(e[:, k], h)
            outs.append(h)
        return torch.tanh(self.head(torch.stack(outs, 1))) * self.caps, h


def params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def save(net: CfCPolicy, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden": net.hidden_size, "width": net.width, "state": net.state_dict(), **(extra or {})}, path)


def load(path) -> CfCPolicy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = CfCPolicy(ck["hidden"], ck["width"])
    net.load_state_dict(ck["state"])
    return net.eval()


class CfCDriver:
    """A trained CfCPolicy as a robot_env driver; its state carries from decision to decision."""

    def __init__(self, path, device: str | None = "cpu"):
        torch.set_num_threads(1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load(path).to(self.device)
        self.name = f"cfc:{Path(path).parent.name}"
        self.reset()

    def reset(self) -> None:
        self.h = self.net.initial(1, self.device)

    def decide(self, obs: Observation) -> Decision:
        with torch.no_grad():
            out, self.h = self.net.step(torch.from_numpy(vector(obs))[None].to(self.device), self.h)
        v, w = out[0].cpu().tolist()
        return Decision(Command(v, w), obs.seq)
