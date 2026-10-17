"""A 1D-CNN driving policy: a circular convolution over the 72 lidar bins (5 degree sectors),
an MLP over the goal and odometry values, and a head that picks one of the 8 discrete actions or
gives a continuous (v, omega) command within speed level 2's caps.

The input is AD.common.features.vector(obs) (78 numbers), optionally the last `frames` of them
stacked (a short memory, the oldest first). CNNDriver wraps a trained network as a robot_env
driver for the benchmark and the window."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robot_env import config as C
from robot_env.types import Command, Decision, Observation

from ...common.features import ACTION_NAMES, FEATURES, LABEL_LEVEL, LIDAR_BINS, to_command, vector

SCALARS = FEATURES - LIDAR_BINS  # goal distance, sin, cos, v, omega, contact
ODOMETRY = (LIDAR_BINS + 3, LIDAR_BINS + 4)  # the measured v and omega in a feature frame
V_CAP, W_CAP = C.SPEED_LEVELS[LABEL_LEVEL]  # 0.30 m/s, 1.0 rad/s


class CNNPolicy(nn.Module):
    def __init__(self, head: str = "discrete", frames: int = 1, width: int = 32, odometry: bool = True):
        """odometry=False hides the measured speed and turn rate from the network: cloned from a
        driver whose motion changes slowly, a policy that sees them learns to keep doing what it
        is doing (the "copycat" shortcut) instead of reacting to the scan and the goal."""
        super().__init__()
        assert head in ("discrete", "continuous")
        self.head, self.frames, self.width, self.odometry = head, frames, width, odometry
        mask = torch.ones(FEATURES)
        if not odometry:
            mask[list(ODOMETRY)] = 0.0
        self.register_buffer("mask", mask)
        w = width

        def conv(cin, cout, k, stride=1):  # the scan is a ring, and so is the padding
            return nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, padding_mode="circular")
        self.lidar = nn.Sequential(  # (B, frames, 72) -> 128
            conv(frames, w, 5), nn.ReLU(),
            conv(w, 2 * w, 5, 2), nn.ReLU(),  # 36 bins
            conv(2 * w, 2 * w, 3, 2), nn.ReLU(),  # 18 bins
            nn.Flatten(), nn.Linear(2 * w * (LIDAR_BINS // 4), 128), nn.ReLU())
        self.scalars = nn.Sequential(nn.Linear(frames * SCALARS, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(128 + 64, 128), nn.ReLU(),
                                   nn.Linear(128, len(ACTION_NAMES) if head == "discrete" else 2))
        self.register_buffer("caps", torch.tensor([V_CAP, W_CAP]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, frames * FEATURES), each frame laid out as features.vector. Discrete: action
        logits (B, 8). Continuous: (v, omega) (B, 2), inside the caps."""
        x = x.view(x.shape[0], self.frames, FEATURES) * self.mask
        z = torch.cat([self.lidar(x[:, :, :LIDAR_BINS]), self.scalars(x[:, :, LIDAR_BINS:].flatten(1))], 1)
        out = self.trunk(z)
        return out if self.head == "discrete" else torch.tanh(out) * self.caps


def stack(x: np.ndarray, seed: np.ndarray, frames: int) -> np.ndarray:
    """Each row with the frames-1 rows before it in the same episode (a seed's rows are in time
    order; at the start the first frame repeats, as CNNDriver does)."""
    if frames == 1:
        return x
    n, f = x.shape
    start = np.r_[0, np.flatnonzero(seed[1:] != seed[:-1]) + 1]
    first = np.repeat(start, np.diff(np.r_[start, n]))  # each row's episode start
    out = np.empty((n, frames * f), np.float32)
    for k in range(frames):  # block k is the frame (frames - 1 - k) steps back
        out[:, k * f:(k + 1) * f] = x[np.maximum(np.arange(n) - (frames - 1 - k), first)]
    return out


def params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def save(net: CNNPolicy, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": net.head, "frames": net.frames, "width": net.width, "odometry": net.odometry,
                "state": net.state_dict(),
                **(extra or {})}, path)


def load(path) -> CNNPolicy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = CNNPolicy(ck["head"], ck["frames"], ck.get("width", 32), ck.get("odometry", True))
    net.load_state_dict(ck["state"])
    return net.eval()


def commands(net: CNNPolicy, out: torch.Tensor) -> list[Command]:
    if net.head == "discrete":
        return [to_command(int(i)) for i in out.argmax(1)]
    return [Command(float(v), float(w)) for v, w in out.tolist()]


class CNNDriver:
    """A trained CNNPolicy as a robot_env driver: on the GPU when there is one (the simulation
    stays on the CPU), else on one CPU thread so many run side by side."""

    def __init__(self, path, device: str | None = None):
        """device None: the GPU if there is one. "cpu" for many side-by-side rollout workers: the
        net decides in under 2 ms there, and each worker saves the GPU context's host memory."""
        torch.set_num_threads(1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load(path).to(self.device)
        self.name = f"cnn:{Path(path).parent.name}"
        self.hist: deque = deque(maxlen=self.net.frames)

    def reset(self) -> None:
        self.hist.clear()

    def decide(self, obs: Observation) -> Decision:
        f = vector(obs)
        if not self.hist:
            self.hist.extend([f] * self.net.frames)
        else:
            self.hist.append(f)
        with torch.no_grad():
            out = self.net(torch.from_numpy(np.concatenate(self.hist))[None].to(self.device)).cpu()
        return Decision(commands(self.net, out)[0], obs.seq)
