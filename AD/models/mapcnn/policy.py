"""The map CNN: a small 2D convolution over the local occupancy map (maps.py) beside the 1D-CNN
branches of AD/models/cnn over the 78 features (72 lidar bins circular, goal and contact; the
measured speed and turn rate masked out), and a continuous (v, omega) head within speed level
2's caps. MapDriver runs it as a robot_env driver, building the map as it goes."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from robot_env import config as C
from robot_env.types import Command, Decision, Observation

from ...common.features import FEATURES, LABEL_LEVEL, LIDAR_BINS, ODOMETRY, vector
from . import maps

SCALARS = FEATURES - LIDAR_BINS
V_CAP, W_CAP = C.SPEED_LEVELS[LABEL_LEVEL]  # 0.30 m/s, 1.0 rad/s


class MapPolicy(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.width = w = width
        mask = torch.ones(FEATURES)
        mask[ODOMETRY] = 0.0  # against the copycat shortcut (the map uses odometry only for its geometry)
        self.register_buffer("mask", mask)

        def conv1(cin, cout, k, stride=1):
            return nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, padding_mode="circular")
        self.lidar = nn.Sequential(conv1(1, w, 5), nn.ReLU(), conv1(w, 2 * w, 5, 2), nn.ReLU(),
                                   conv1(2 * w, 2 * w, 3, 2), nn.ReLU(),
                                   nn.Flatten(), nn.Linear(2 * w * (LIDAR_BINS // 4), 128), nn.ReLU())
        self.scalars = nn.Sequential(nn.Linear(SCALARS, 64), nn.ReLU())
        self.map = nn.Sequential(  # (B, 3, 32, 32) -> 128
            nn.Conv2d(maps.CHANNELS, w // 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16
            nn.Conv2d(w // 2, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 4
            nn.Flatten(), nn.Linear(2 * w * (maps.CELLS // 8) ** 2, 128), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(128 + 64 + 128, 128), nn.ReLU(), nn.Linear(128, 2))
        self.register_buffer("caps", torch.tensor([V_CAP, W_CAP]))

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """x: (B, 78) features.vector rows; m: (B, 3, 32, 32) maps, uint8 or already in [0, 1].
        Returns (v, omega) (B, 2) inside the caps."""
        x = x * self.mask
        m = m.float() / 255.0 if m.dtype == torch.uint8 else m
        z = torch.cat([self.lidar(x[:, None, :LIDAR_BINS]), self.scalars(x[:, LIDAR_BINS:]), self.map(m)], 1)
        return torch.tanh(self.trunk(z)) * self.caps


def params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def save(net: MapPolicy, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"width": net.width, "state": net.state_dict(), **(extra or {})}, path)


def load(path) -> MapPolicy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = MapPolicy(ck.get("width", 32))
    net.load_state_dict(ck["state"])
    return net.eval()


class MapDriver:
    """A trained MapPolicy as a robot_env driver (CPU by default: about 1 ms per decision, so many
    benchmark workers run side by side). last_map is the map of the latest decision (DAgger)."""

    def __init__(self, path, device: str = "cpu"):
        torch.set_num_threads(1)
        self.device = device
        self.net = load(path).to(device)
        self.name = f"mapcnn:{Path(path).parent.name}"
        self.map = maps.LocalMap()
        self.last_map = None

    def reset(self) -> None:
        self.map.reset()

    def decide(self, obs: Observation) -> Decision:
        f = vector(obs)
        self.last_map = self.map.step(f, obs.time)
        with torch.no_grad():
            out = self.net(torch.from_numpy(f)[None].to(self.device),
                           torch.from_numpy(self.last_map)[None].to(self.device)).cpu()
        v, w = out[0].tolist()
        return Decision(Command(float(v), float(w)), obs.seq)
